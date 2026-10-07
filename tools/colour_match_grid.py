#!/usr/bin/env python3
"""Detect and correct brightness / colour differences between the tiles of a flat scan grid, then stitch.

Why: a long scan can be interrupted and resumed (new camera session, different gain or
white balance), the camera's automatic exposure drifts from tile to tile, and the field
illumination is never uniform (darker on one side, darker in the corners). In the mosaic
each of these shows up as a step at a tile boundary; the first two also shift the colour
that the layer model reads. This tool measures all of them where the evidence is
strongest, in the registered overlap of physically adjacent tiles, corrects them with
recorded, reversible per-tile factors, and leaves the originals untouched.

Method (8-bit RGB; medians everywhere; nothing is fitted per pixel):

1. Inventory ``mosaic_r<row>_c<col>.png`` tiles. Register a few pairs to learn on which
   side the next column lies (the scanner's ``invert_x`` moved left here) and the
   horizontal / vertical neighbour vectors.
2. Illumination field ("flat field"): every tile is divided by its own per-channel mean,
   the per-pixel median over a few hundred tiles removes the specimen (its content is
   uncorrelated between tiles, the illumination is the same in all of them), and a
   mild Gaussian smoothing (sigma = 1/32 of the width) removes what is left of it. The
   field is normalised to mean 1 per channel, so the average brightness of a tile does
   not change; only its spatial profile does.
3. Every neighbouring pair (same row, adjacent columns; same column, adjacent rows) is
   registered by template matching on the flat-fielded tiles and the per-channel ratio
   of medians A/B over the overlap is recorded. Both tiles looked at the same physical
   region, so after the field correction the ratio is an exposure / colour difference
   between the two tiles. The median ratio of each edge kind (all horizontal pairs, all
   vertical pairs) is what every pair of that kind has in common - the part of the
   illumination the field did not capture - and is removed before solving; otherwise a
   0.5 % residual would be chained over 50 rows into a 30 % ramp.
4. Gains are solved from all edges at once: one free level per column plus a penalised
   deviation per tile (weighted least squares on log gains; edges weighted by the flat
   substrate share of their overlap and Huber re-weighted by residual). A run resumed
   with a different camera gain therefore gets its whole block of columns corrected, a
   single tile whose own edges agree that it is darker (auto-exposure reacting to bright
   content) gets its own factor, and a tiny systematic bias of one edge kind cannot be
   chained along 50 rows into a ramp. The longest run of columns without a step keeps
   mean gain 1. Gains outside ``GAIN_LIMITS`` are clipped and flagged; too many flags
   stop the run. ``--gain-mode column`` pins the tile deviations to 0 (the previous,
   conservative behaviour). Bare-substrate plateau medians per column
   (``flakepipeline.color_diagnostics``) are reported as an independent cross-check.
5. Output is a derived dataset: corrected tiles are new PNG copies, tiles that need no
   change are hard links to the originals (or byte copies where links are impossible).
   When the next column lies to the left, column indices are mirrored
   (``new_col = ncols-1-col``) so the stitcher's "column index increases to the right"
   convention holds. ``session.json`` / ``events.jsonl`` are rewritten with the new file
   hashes and the provenance of every tile (source file, source hash, gain, flat-field
   record), mirroring the scanner's own derived-dataset convention.
6. ``--stitch`` runs ``run_stitch.py`` on the derived dataset with a measured stitch
   profile. Kimi is used whenever ``run_stitch.py`` finds credentials (``--no-ai`` is
   passed through only when requested), as the owner requires. The stitcher's own
   render-time flat field then finds an almost uniform field and changes little.

Limits: the gains describe this acquisition, not the camera; they are multiplicative, so
a tile with clipped highlights stays clipped (the clipped fraction is recorded). The
analysis colour check against the reference substrate colour still runs on the result.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TILE_RE = re.compile(r"^mosaic_r(?P<row>\d+)_c(?P<col>\d+)\.(?:png|jpg|jpeg|tif|tiff|bmp)$", re.I)
SCHEMA_VERSION = 2
DEFAULT_MIN_STEP = 0.025          # column-summary threshold (report only in tile mode)
GAIN_LIMITS = (0.5, 2.0)
TILE_PENALTY = 0.3                # cost of a per-tile deviation from its column level (edge weight = 1)
FLAT_SAMPLES = 400                # tiles used for the illumination field
FLAT_SCALE_DIV = 4                # field is estimated at 1/4 tile size
FLAT_SMOOTH_DIV = 32              # Gaussian sigma = tile width / FLAT_SMOOTH_DIV (at the field's scale)
MEASURE_SCALE = 0.5               # tiles are registered / measured at this scale
LUM = np.array([0.299, 0.587, 0.114], np.float32)


class ColourMatchError(ValueError):
    """An actionable problem; nothing has been written when it is raised before `apply`."""


# ----------------------------------------------------------------------------
# Inventory
# ----------------------------------------------------------------------------
def inventory(data_dir: Path) -> dict:
    grid = {}
    for path in sorted(Path(data_dir).iterdir()):
        match = TILE_RE.fullmatch(path.name)
        if match and path.is_file():
            key = (int(match["row"]), int(match["col"]))
            if key in grid:
                raise ColourMatchError(f"Duplicate grid coordinate {key}")
            grid[key] = path
    if not grid:
        raise ColourMatchError(f"No mosaic_r<row>_c<col> tiles in {data_dir}")
    rows = sorted({r for r, _ in grid})
    cols = sorted({c for _, c in grid})
    missing = [(r, c) for c in cols for r in rows if (r, c) not in grid]
    if missing:
        raise ColourMatchError(f"Incomplete grid; missing {missing[:10]}")
    if len(cols) < 2:
        raise ColourMatchError("At least two columns are needed to compare tile colours")
    return {"grid": grid, "rows": rows, "cols": cols}


def _load_rgb(path) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ColourMatchError(f"Cannot decode {path}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def _gray(rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _resize(rgb: np.ndarray, scale: float) -> np.ndarray:
    import cv2
    if scale == 1:
        return rgb
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# Registration by template matching (origin of B in A's pixel coordinates)
# ----------------------------------------------------------------------------
def register(A: np.ndarray, B: np.ndarray, side: str, *, strip: float = 0.04, margin: float = 0.14,
             search: float = 0.45) -> tuple[int, int, float]:
    """Register B against A given where B lies: 'left', 'right', 'below' or 'above' of A.

    A strip along B's edge that faces A is matched inside the facing part of A.
    Returns (dx, dy, score): B's origin in A's coordinates and the normalised
    cross-correlation of the best match.
    """
    import cv2
    ga, gb = _gray(A), _gray(B)
    h, w = ga.shape
    sw, sh = max(8, int(round(w * strip))), max(8, int(round(h * strip)))
    mx, my = int(round(w * margin)), int(round(h * margin))
    if side == "left":          # B's right edge faces A's left part
        tx, ty, tw, th = w - sw, my, sw, h - 2 * my
        sx0, sy0, sx1, sy1 = 0, 0, int(round(w * search)), h
    elif side == "right":
        tx, ty, tw, th = 0, my, sw, h - 2 * my
        sx0, sy0, sx1, sy1 = w - int(round(w * search)), 0, w, h
    elif side == "below":       # B's top edge faces A's bottom part
        tx, ty, tw, th = mx, 0, w - 2 * mx, sh
        sx0, sy0, sx1, sy1 = 0, h - int(round(h * search)), w, h
    elif side == "above":
        tx, ty, tw, th = mx, h - sh, w - 2 * mx, sh
        sx0, sy0, sx1, sy1 = 0, 0, w, int(round(h * search))
    else:
        raise ValueError(side)
    res = cv2.matchTemplate(ga[sy0:sy1, sx0:sx1], gb[ty:ty + th, tx:tx + tw], cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(res)
    return int(x + sx0 - tx), int(y + sy0 - ty), float(score)


def detect_direction(inv: dict, *, samples: int = 6, seed: int = 0) -> dict:
    """Which side the next column and the next row lie on, with median vectors (full-resolution px)."""
    rng = np.random.default_rng(seed)
    rows, cols, grid = inv["rows"], inv["cols"], inv["grid"]
    picks = []
    for _ in range(samples):
        picks.append((rows[int(rng.integers(0, len(rows)))], cols[int(rng.integers(0, len(cols) - 1))]))
    votes = {"left": [], "right": []}
    for r, c in picks:
        A, B = _load_rgb(grid[(r, c)]), _load_rgb(grid[(r, cols[cols.index(c) + 1])])
        for side in ("left", "right"):
            dx, dy, score = register(A, B, side)
            votes[side].append((score, dx, dy))
    best = max(votes, key=lambda s: np.median([v[0] for v in votes[s]]))
    col_vec = [float(np.median([v[1] for v in votes[best]])), float(np.median([v[2] for v in votes[best]]))]
    col_score = float(np.median([v[0] for v in votes[best]]))
    row_vec, row_scores = None, []
    if len(rows) >= 2:
        vs = []
        for _ in range(samples):
            r, c = rows[int(rng.integers(0, len(rows) - 1))], cols[int(rng.integers(0, len(cols)))]
            A, B = _load_rgb(grid[(r, c)]), _load_rgb(grid[(rows[rows.index(r) + 1], c)])
            dx, dy, score = register(A, B, "below")
            vs.append((score, dx, dy))
        row_vec = [float(np.median([v[1] for v in vs])), float(np.median([v[2] for v in vs]))]
        row_scores = [v[0] for v in vs]
    if col_score < 0.5:
        raise ColourMatchError(f"Adjacent columns do not register (best NCC {col_score:.2f}); check that tiles overlap")
    return {"next_column_side": best, "next_column_vector_dxdy": col_vec, "next_column_ncc": round(col_score, 3),
            "next_row_vector_dxdy": row_vec, "next_row_ncc": round(float(np.median(row_scores)), 3) if row_scores else None}


# ----------------------------------------------------------------------------
# Illumination field
# ----------------------------------------------------------------------------
def estimate_flatfield(inv: dict, *, samples: int = FLAT_SAMPLES, scale_div: int = FLAT_SCALE_DIV,
                       smooth_div: int = FLAT_SMOOTH_DIV, seed: int = 0, log=print) -> dict:
    """Per-pixel median of mean-normalised tiles, mildly smoothed; mean 1 per channel."""
    import cv2
    grid = inv["grid"]
    keys = list(grid)
    rng = np.random.default_rng(seed)
    if len(keys) > samples:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), samples, replace=False))]
    stack, shape, rejected = [], None, 0
    for key in keys:
        img = _load_rgb(grid[key])
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // scale_div, h // scale_div), interpolation=cv2.INTER_AREA).astype(np.float32)
        if shape is None:
            shape = small.shape
        elif small.shape != shape:
            raise ColourMatchError(f"Tiles differ in size: {grid[key]}")
        g = small @ LUM
        if g.std() < 3.0 or g.mean() < 8.0 or g.mean() > 247.0:       # blank / blown-out tile: no illumination information
            rejected += 1
            continue
        means = small.reshape(-1, 3).mean(axis=0)
        if np.any(means < 1.0):
            rejected += 1
            continue
        stack.append((small / means).astype(np.float16))
    if len(stack) < 8:
        raise ColourMatchError(f"Only {len(stack)} tiles usable for the illumination field ({rejected} rejected)")
    arr = np.stack(stack)                                   # (n, h, w, 3) float16
    h, w = shape[:2]
    med = np.empty((h, w, 3), np.float32)
    step = max(1, (1 << 26) // (arr.shape[0] * w * 3 * 8))    # ~512 MB float64 per chunk
    for y0 in range(0, h, step):
        med[y0:y0 + step] = np.median(arr[:, y0:y0 + step].astype(np.float32), axis=0)
    del arr
    sigma = max(w, h) / float(smooth_div)
    field = cv2.GaussianBlur(med, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE).astype(np.float32)
    if not np.all(np.isfinite(field)):
        raise ColourMatchError("Illumination field has non-finite values; the sample tiles are unusable")
    resid_rms = float(np.sqrt(np.mean((med - field) ** 2)))
    field = np.maximum(field, 1e-3)
    field /= field.reshape(-1, 3).mean(axis=0)
    xprof = field.mean(axis=0)
    yprof = field.mean(axis=1)
    info = {"model": f"smoothed_median_sigma{sigma:.1f}px", "sigma_px": round(sigma, 2),
            "scale_div": scale_div, "field_size_wh": [w, h], "samples_used": len(stack), "samples_rejected": rejected,
            "smoothing_residual_rms": round(resid_rms, 5), "range": [round(float(field.min()), 4), round(float(field.max()), 4)],
            "edge_to_edge_ratio_left_over_right": (xprof[0] / xprof[-1]).round(4).tolist(),
            "edge_to_edge_ratio_top_over_bottom": (yprof[0] / yprof[-1]).round(4).tolist(),
            "nonuniformity_max_abs": round(float(np.max(np.abs(field - 1))), 4)}
    log(f"[flatfield] {info['model']} from {len(stack)} tiles: left/right {info['edge_to_edge_ratio_left_over_right']}, "
        f"top/bottom {info['edge_to_edge_ratio_top_over_bottom']}, range {info['range']}, smoothing residual RMS {resid_rms:.4f}")
    return {"field": field, "info": info}


def flatfield_correction(ff: dict | None, w: int, h: int) -> np.ndarray | None:
    """Per-pixel multiplicative correction (1 / field) at full tile size, or None."""
    import cv2
    if ff is None:
        return None
    full = cv2.resize(ff["field"], (w, h), interpolation=cv2.INTER_LINEAR)
    return (1.0 / np.maximum(full, 1e-3)).astype(np.float32)


# ----------------------------------------------------------------------------
# Edge measurement
# ----------------------------------------------------------------------------
def overlap_ratio(A: np.ndarray, B: np.ndarray, dx: int, dy: int) -> dict | None:
    """Per-channel ratio of medians A/B over the registered overlap; None when too small."""
    import cv2
    h, w = A.shape[:2]
    xa0, xa1 = max(0, dx), min(w, w + dx)
    ya0, ya1 = max(0, dy), min(h, h + dy)
    if xa1 - xa0 < 24 or ya1 - ya0 < 24:
        return None
    a = A[ya0:ya1, xa0:xa1].astype(np.float32)
    b = B[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx].astype(np.float32)
    a = cv2.blur(a, (5, 5))[3:-3, 3:-3].reshape(-1, 3)
    b = cv2.blur(b, (5, 5))[3:-3, 3:-3].reshape(-1, 3)
    ok = np.all((a > 20) & (a < 245) & (b > 20) & (b < 245), axis=1)
    if ok.sum() < 500:
        return None
    a, b = a[ok], b[ok]
    lum = a @ LUM
    bright = lum >= np.percentile(lum, 50)
    med_lum = float(np.median(lum))
    flat = float(np.mean(np.abs(lum - med_lum) <= 0.1 * med_lum))      # share of pixels on one flat level (substrate)
    return {"ratio_all": (np.median(a, 0) / np.median(b, 0)).tolist(),
            "ratio_bright": (np.median(a[bright], 0) / np.median(b[bright], 0)).tolist(),
            "pixels": int(ok.sum()), "flat_fraction": round(flat, 4), "overlap_wh": [int(xa1 - xa0), int(ya1 - ya0)]}


class _TileCache:
    """LRU of measurement-scale, flat-fielded tiles (uint8), bounded by count."""

    def __init__(self, grid, corr, scale, capacity):
        self.grid, self.corr, self.scale, self.capacity = grid, corr, scale, capacity
        self.store: OrderedDict = OrderedDict()

    def get(self, key):
        img = self.store.get(key)
        if img is not None:
            self.store.move_to_end(key)
            return img
        rgb = _load_rgb(self.grid[key])
        if self.corr is not None:
            rgb = np.clip(np.rint(rgb.astype(np.float32) * self.corr), 0, 255).astype(np.uint8)
        img = _resize(rgb, self.scale)
        self.store[key] = img
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)
        return img


def measure_edges(inv: dict, direction: dict, ff: dict | None, *, min_ncc: float = 0.6,
                  scale: float = MEASURE_SCALE, log=print) -> list[dict]:
    """One record per neighbouring pair (horizontal: same row, next column; vertical: same column, next row)."""
    rows, cols, grid = inv["rows"], inv["cols"], inv["grid"]
    side = direction["next_column_side"]
    sample = _load_rgb(next(iter(grid.values())))
    h, w = sample.shape[:2]
    corr = flatfield_correction(ff, w, h)
    if w * scale < 640:                      # small tiles are measured at full size
        scale = 1.0
    cache = _TileCache(grid, corr, scale, capacity=2 * len(rows) + 4)
    pdx_h, pdy_h = (np.array(direction["next_column_vector_dxdy"]) * scale).tolist()
    pdx_v, pdy_v = (np.array(direction["next_row_vector_dxdy"]) * scale).tolist() if direction.get("next_row_vector_dxdy") else (None, None)
    edges, n_bad = [], 0
    for ci, c in enumerate(cols):
        for ri, r in enumerate(rows):
            A = cache.get((r, c))
            pairs = []
            if ri + 1 < len(rows):
                pairs.append(("v", (rows[ri + 1], c), "below", pdx_v, pdy_v))
            if ci + 1 < len(cols):
                pairs.append(("h", (r, cols[ci + 1]), side, pdx_h, pdy_h))
            for kind, key_b, bside, pdx, pdy in pairs:
                B = cache.get(key_b)
                dx, dy, score = register(A, B, bside)
                rec = {"kind": kind, "a": [r, c], "b": list(key_b), "dx": dx / scale, "dy": dy / scale, "ncc": round(score, 3)}
                stats = None
                if score >= min_ncc and (pdx is None or (abs(dx - pdx) <= 0.1 * A.shape[1] and abs(dy - pdy) <= 0.15 * A.shape[0])):
                    stats = overlap_ratio(A, B, dx, dy)
                if stats:
                    rec.update(stats)
                else:
                    rec["rejected"] = "registration" if score < min_ncc else "overlap"
                    n_bad += 1
                edges.append(rec)
        log(f"[edges] column c{c}: {sum(1 for e in edges if e['a'][1] == c and 'ratio_all' in e)} usable edges so far from this column")
    good = [e for e in edges if "ratio_all" in e]
    if not good:
        raise ColourMatchError("No neighbouring pair could be registered; colours cannot be compared")
    for kind in ("h", "v"):
        rs = np.array([e["ratio_all"] for e in good if e["kind"] == kind])
        if len(rs):
            log(f"[edges] {len(rs)} {kind}-edges: median A/B {np.median(rs, 0).round(4).tolist()}, "
                f"IQR {(np.percentile(rs, 75, 0) - np.percentile(rs, 25, 0)).round(4).tolist()}")
    log(f"[edges] {len(good)} usable of {len(edges)} ({n_bad} rejected)")
    return edges


# ----------------------------------------------------------------------------
# Gains
# ----------------------------------------------------------------------------
def solve_gains(edges: list[dict], inv: dict, *, huber: float = 0.03, tile_penalty: float = TILE_PENALTY,
                iters: int = 4, mode: str = "tile", min_step: float = DEFAULT_MIN_STEP, baseline: bool = True,
                log=print) -> dict:
    """Log-gains from all usable edges: one level per column plus a penalised deviation per tile.

    ``gain(tile) = column_level + tile_deviation``. Column levels are free (a resumed run, a drift between
    columns), tile deviations cost ``tile_penalty`` x deviation^2 against the edge misfit, so a tile is
    moved only where its own edges agree on it (an auto-exposure reaction to bright content) and a tiny
    systematic bias of one edge kind cannot be chained along 50 rows into a ramp. ``mode="column"`` pins
    the deviations to 0. Edges are weighted by the flat (substrate) share of their overlap and Huber
    re-weighted by their residual, so textured or mis-registered overlaps count less.

    With ``baseline`` the median log-ratio of each edge kind (median over boundaries of per-boundary
    medians) is removed first: what every horizontal (or vertical) pair has in common is illumination
    the field did not capture, not an exposure difference.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve
    cols, rows = inv["cols"], inv["rows"]
    keys = [(r, c) for c in cols for r in rows]
    idx = {k: i for i, k in enumerate(keys)}
    cidx = {c: i for i, c in enumerate(cols)}
    n, m = len(keys), len(cols)
    use = [e for e in edges if "ratio_all" in e]
    if not use:
        raise ColourMatchError("No usable edges")
    ia = np.array([idx[tuple(e["a"])] for e in use])
    ib = np.array([idx[tuple(e["b"])] for e in use])
    ca = np.array([cidx[e["a"][1]] for e in use])
    cb = np.array([cidx[e["b"][1]] for e in use])
    d = -np.log(np.clip(np.array([e["ratio_all"] for e in use], np.float64), 1e-3, 1e3))   # g_a - g_b = d
    kinds = np.array([e["kind"] for e in use])
    # baseline per edge kind = median over boundaries of the per-boundary medians (a boundary is one column pair
    # for horizontal edges, one column for vertical edges), so a few stepped boundaries cannot bias it
    groups = np.array([(e["a"][1], e["b"][1]) if e["kind"] == "h" else (e["a"][1], -1) for e in use])
    baselines = {}
    for kind in ("h", "v"):
        sel = np.flatnonzero(kinds == kind)
        if len(sel):
            per_group = [np.median(d[sel[(groups[sel] == g).all(axis=1)]], axis=0) for g in np.unique(groups[sel], axis=0)]
            med = np.median(np.array(per_group), axis=0)
            baselines[kind] = np.exp(-med).round(5).tolist()          # typical A/B ratio of this edge kind
            if baseline:
                d[sel] -= med
    w0 = np.array([min(1.0, e.get("flat_fraction", 1.0) / 0.6) for e in use])
    w = w0.copy()
    penalty = 1e6 if mode == "column" else float(tile_penalty)
    # unknowns: n tile deviations, then m column levels
    m_e = len(use)
    r_idx = np.repeat(np.arange(m_e), 4)
    c_idx = np.stack([ia, n + ca, ib, n + cb], axis=1).ravel()
    v_idx = np.tile(np.array([1.0, 1.0, -1.0, -1.0]), m_e)
    A = coo_matrix((v_idx, (r_idx, c_idx)), shape=(m_e, n + m)).tocsr()
    prior = diags(np.concatenate([np.full(n, penalty), np.full(m, 1e-6)]))
    x = np.zeros((n + m, 3))
    for _ in range(iters):
        AtW = A.T @ diags(w)
        L = (AtW @ A + prior).tocsc()
        for ch in range(3):
            x[:, ch] = spsolve(L, AtW @ d[:, ch])
        resid = (A @ x) - d
        r = np.abs(resid).max(axis=1)
        w = w0 * np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-9))
    g = x[:n] + x[n:][np.array([cidx[c] for _, c in keys])]
    col_level = x[n:]
    # gauge: the longest run of columns without a step keeps mean gain 1 (the reference acquisition)
    segments, current = [], [cols[0]]
    for c0, c1 in zip(cols, cols[1:]):
        if np.max(np.abs(col_level[cidx[c1]] - col_level[cidx[c0]])) > min_step:
            segments.append(current)
            current = [c1]
        else:
            current.append(c1)
    segments.append(current)
    reference = max(segments, key=len)
    shift = np.mean([g[idx[(r, c)]] for c in reference for r in rows], axis=0)
    g -= shift
    col_level -= shift
    resid = (A @ x) - d
    before = np.sqrt(np.mean(d ** 2))
    after = np.sqrt(np.mean(resid ** 2))
    outliers = [{"a": use[i]["a"], "b": use[i]["b"], "kind": use[i]["kind"], "residual": resid[i].round(4).tolist(),
                 "weight": round(float(w[i]), 3)} for i in np.flatnonzero(w < 0.5 * w0)]
    gains = np.exp(g)
    lo, hi = GAIN_LIMITS
    flagged = [keys[i] for i in np.flatnonzero(np.any((gains < lo) | (gains > hi), axis=1))]
    if len(flagged) > 0.05 * n:
        raise ColourMatchError(f"Implausible gain for {len(flagged)} of {n} tiles (outside {GAIN_LIMITS}); refusing to correct")
    gains = np.clip(gains, lo, hi)
    per_tile = {f"r{r}_c{c}": gains[idx[(r, c)]].round(5).tolist() for (r, c) in keys}
    dev = np.exp(x[:n])
    col_gain = {c: np.exp(col_level[cidx[c]]) for c in cols}
    col_median = {c: np.median(np.array([gains[idx[(r, c)]] for r in rows]), axis=0) for c in cols}
    steps = []
    for c0, c1 in zip(cols, cols[1:]):
        rel = col_gain[c1] / col_gain[c0]
        if np.max(np.abs(rel - 1)) > min_step:
            steps.append({"col_a": c0, "col_b": c1, "relative_gain_b_over_a": rel.round(4).tolist()})
    changed = [k for k in keys if np.any(np.abs(gains[idx[k]] - 1) > 1e-6)]
    log(f"[gains] mode={mode}, tile penalty {penalty:g}; edge log-ratio RMS {before:.4f} -> residual {after:.4f}; "
        f"{len(outliers)} down-weighted edges; {len(flagged)} flagged tiles; gain range {gains.min():.3f}..{gains.max():.3f}; "
        f"tile deviations within {np.abs(dev - 1).max():.3f}")
    for c in cols:
        log(f"        c{c}: level {col_gain[c].round(4).tolist()}  median tile gain {col_median[c].round(4).tolist()}")
    return {"mode": mode, "gains_rgb_by_tile": per_tile, "tiles_corrected": len(changed),
            "edge_rms_before": round(float(before), 5), "edge_rms_after": round(float(after), 5),
            "edges_used": len(use), "downweighted_edges": outliers, "flagged_tiles": [f"r{r}_c{c}" for r, c in flagged],
            "column_level_gain_rgb": {str(c): v.round(5).tolist() for c, v in col_gain.items()},
            "column_median_gain_rgb": {str(c): v.round(5).tolist() for c, v in col_median.items()},
            "tile_deviation_max_abs": round(float(np.abs(dev - 1).max()), 5),
            "column_steps": steps, "segments": segments, "reference_segment": reference,
            "median_ratio_by_edge_kind": baselines, "baseline_removed": bool(baseline),
            "min_step": min_step, "huber": huber, "tile_penalty": penalty}


def plateau_columns(inv: dict, *, max_rows: int = 24) -> dict:
    """Bare-substrate plateau median RGB per column of the ORIGINAL tiles (independent cross-check)."""
    from PIL import Image
    from flakepipeline import color_diagnostics as cd
    rows, cols, grid = inv["rows"], inv["cols"], inv["grid"]
    step = max(1, len(rows) // max_rows)
    out = {}
    for c in cols:
        values = []
        for r in rows[::step]:
            with Image.open(grid[(r, c)]) as im:
                small = im.convert("RGB")
                small.thumbnail((480, 480))
                bg = cd.background_colour(np.asarray(small))
            if bg["method"] == "brightest_flat_luminance_plateau":
                values.append(bg["median_rgb"])
        out[c] = {"n": len(values), "median_rgb": np.median(values, 0).round(2).tolist() if values else None}
    return out


# ----------------------------------------------------------------------------
# Apply: derived dataset
# ----------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply(inv: dict, solution: dict, direction: dict, out_dir: Path, *, source_dir: Path, ff: dict | None = None,
          log=print) -> dict:
    from PIL import Image
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ColourMatchError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cols, rows, grid = inv["cols"], inv["rows"], inv["grid"]
    mirror = direction["next_column_side"] == "left"
    ncols = len(cols)
    corr = None
    ff_record = None
    if ff is not None:
        sample = _load_rgb(next(iter(grid.values())))
        corr = flatfield_correction(ff, sample.shape[1], sample.shape[0])
        np.save(out_dir / "flat_field.npy", ff["field"])
        ff_record = dict(ff["info"], file="flat_field.npy", file_sha256=_sha256(out_dir / "flat_field.npy"),
                         applied_as="pixel / field (field normalised to mean 1 per channel)")
    records = []
    for c in cols:
        new_c = (ncols - 1 - cols.index(c)) if mirror else cols.index(c)
        n_corr = 0
        for r in rows:
            src = grid[(r, c)]
            dst = out_dir / f"mosaic_r{r}_c{new_c}{src.suffix.lower()}"
            gain = np.array(solution["gains_rgb_by_tile"][f"r{r}_c{c}"], np.float32)
            gain_changes = bool(np.any(np.abs(gain - 1) > 1e-6))
            if corr is not None or gain_changes:
                with Image.open(src) as im:
                    rgb = np.asarray(im.convert("RGB")).astype(np.float32)
                factor = gain[None, None, :] if corr is None else corr * gain[None, None, :]
                out = np.clip(np.rint(rgb * factor), 0, 255).astype(np.uint8)
                Image.fromarray(out).save(dst, format="PNG" if dst.suffix == ".png" else None, compress_level=1)
                kind, clipped = "corrected_copy", float(np.mean(np.any(out >= 255, axis=2)))
                n_corr += 1
            else:
                try:
                    os.link(src, dst)
                    kind = "hard_link_to_original"
                except OSError:            # other volume, or a filesystem without hard links (exFAT)
                    shutil.copy2(src, dst)
                    kind = "byte_copy_of_original"
                clipped = None
            records.append({"row": r, "source_col": c, "col": new_c, "filename": dst.name, "kind": kind,
                            "gain_rgb": gain.round(5).tolist() if kind == "corrected_copy" else None,
                            "flat_field": bool(corr is not None) if kind == "corrected_copy" else False,
                            "clipped_fraction": clipped, "source_filename": src.name, "source_sha256": _sha256(src),
                            "sha256": _sha256(dst), "bytes": dst.stat().st_size})
        log(f"[apply] column c{c} -> c{new_c}: {n_corr}/{len(rows)} corrected copies, median gain "
            f"{solution['column_median_gain_rgb'][str(c)]}")
    manifest = {"schema_version": SCHEMA_VERSION, "kind": "colour_matched_grid", "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_dir": str(source_dir), "mirrored_columns": mirror,
                "column_mapping": f"new_col = {ncols - 1} - index(old_col)" if mirror else "new_col = index(old_col)",
                "direction": direction, "flat_field": ff_record,
                "gains": {k: v for k, v in solution.items() if k != "gains_rgb_by_tile"}, "tiles": records}
    _write_derived_records(source_dir, out_dir, manifest, log)
    (out_dir / "colour_match_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def _write_derived_records(source_dir: Path, out_dir: Path, manifest: dict, log) -> None:
    """Derived session.json / events.jsonl carrying the new hashes, when the source has them."""
    src_session, src_events = Path(source_dir) / "session.json", Path(source_dir) / "events.jsonl"
    if not src_session.is_file():
        log("[records] source has no session.json; the derived dataset carries only colour_match_manifest.json")
        return
    session = json.loads(src_session.read_text(encoding="utf-8-sig"))
    captures = {}
    if src_events.is_file():
        for line in src_events.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                e = json.loads(line)
                if e.get("event") == "capture_success":
                    captures[e.get("filename") or e.get("image_relative_path")] = e
    now = datetime.now(timezone.utc).isoformat()
    gains = manifest["gains"]
    note = {"schema_version": SCHEMA_VERSION, "purpose": "stitching-only derived dataset: illumination field and per-tile exposure/colour gains corrected",
            "built_at_utc": now, "source_session_id": session.get("session_id"),
            "column_mapping": manifest["column_mapping"], "mirrored_columns": manifest["mirrored_columns"],
            "flat_field": manifest["flat_field"], "gain_mode": gains["mode"], "tiles_corrected": gains["tiles_corrected"],
            "column_median_gain_rgb": gains["column_median_gain_rgb"], "column_steps": gains["column_steps"],
            "warning": "Derived for seam appearance and provenance; not evidence of one continuous acquisition. Originals untouched."}
    session = dict(session)
    session["derived_colour_correction"] = note
    session["position_trusted"] = False
    session["position_trust_basis"] = "derived colour-matched dataset; see derived_colour_correction"
    if manifest["mirrored_columns"] and isinstance(session.get("scan_config"), dict):
        sc = dict(session["scan_config"])
        sc["invert_x"] = False
        sc["derived_note"] = "columns mirrored in this derived copy; invert_x therefore false here"
        session["scan_config"] = sc
    events = [{"event": "session_started", "timestamp_utc": now, "session_id": session.get("session_id"),
               "event_id": hashlib.sha256(b"derived-start").hexdigest()[:32], "derived_colour_correction": True, "sequence": 1}]
    for i, t in enumerate(manifest["tiles"], 2):
        base = dict(captures.get(t["source_filename"], {}))
        base.update(event="capture_success", filename=t["filename"], image_relative_path=t["filename"], row=t["row"], col=t["col"],
                    image_sha256=t["sha256"], image_bytes=t["bytes"], sequence=i, session_id=session.get("session_id"),
                    timestamp_utc=base.get("timestamp_utc", now),
                    event_id=hashlib.sha256(("derived-" + t["filename"]).encode()).hexdigest()[:32],
                    derived_colour_correction={"kind": t["kind"], "gain_rgb": t["gain_rgb"], "flat_field": t["flat_field"],
                                               "source_filename": t["source_filename"], "source_sha256": t["source_sha256"],
                                               "source_col": t["source_col"], "clipped_fraction": t["clipped_fraction"],
                                               "source_event_id": base.get("event_id")})
        events.append(base)
    events.append({"event": "session_finished", "timestamp_utc": now, "session_id": session.get("session_id"), "status": session.get("status"),
                   "positions_completed": len(manifest["tiles"]), "photos_saved": len(manifest["tiles"]),
                   "planned_positions": session.get("planned_positions"), "derived_colour_correction": True,
                   "event_id": hashlib.sha256(b"derived-finish").hexdigest()[:32], "sequence": len(events) + 1})
    session["event_count"] = len(events)
    (out_dir / "session.json").write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    log(f"[records] derived session.json and events.jsonl written ({len(events)} events)")


# ----------------------------------------------------------------------------
# Stitch profile and command
# ----------------------------------------------------------------------------
def stitch_profile(inv: dict, direction: dict, *, scale_div: int = 4, out_scale: float = 0.25) -> dict:
    """Measured nominal vectors for run_stitch.py in the (possibly mirrored) output orientation."""
    sample = next(iter(inv["grid"].values()))
    from PIL import Image
    with Image.open(sample) as im:
        w, h = im.size
    cdx, cdy = direction["next_column_vector_dxdy"]
    if direction["next_column_side"] == "left":
        dx_h, dy_h = -cdx, -cdy
    else:
        dx_h, dy_h = cdx, cdy
    if direction["next_row_vector_dxdy"] is None:
        raise ColourMatchError("A vertical neighbour vector is required for the stitch profile (need at least 2 rows)")
    dx_v, dy_v = direction["next_row_vector_dxdy"]
    profile = {"schema_version": 1, "name": "measured-colour-match-grid", "image_size": [w, h],
               "nominal_vectors": [float(dx_v), float(dy_v), float(dx_h), float(dy_h)],
               "geometry_source": "measured: template registration of adjacent tiles by tools/colour_match_grid.py",
               "scale_div": scale_div, "out_scale": out_scale}
    import stitch_profile as SP
    SP.validate_profile(profile)
    return profile


def stitch_command(out_dir: Path, profile_path: Path, *, no_ai: bool = False, workers: int = 4, extra: list[str] | None = None) -> list[str]:
    cmd = [sys.executable, str(REPO / "run_stitch.py"), "--data", str(out_dir), "--work", str(out_dir / "_stitch_work"),
           "--out", str(out_dir / "mosaic.png"), "--stitch-profile", str(profile_path), "--workers", str(workers)]
    if no_ai:
        cmd.append("--no-ai")
    cmd.extend(extra or [])
    return cmd


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, type=Path, help="flat-grid directory with mosaic_r<row>_c<col>.png (+ session.json/events.jsonl)")
    parser.add_argument("--out", type=Path, default=None, help="derived dataset directory (default: <data>_colour_matched)")
    parser.add_argument("--gain-mode", choices=("tile", "column"), default="tile", help="one gain per tile (default) or per column")
    parser.add_argument("--tile-penalty", type=float, default=TILE_PENALTY, help="penalty on per-tile deviations from the column level")
    parser.add_argument("--no-flat-field", action="store_true", help="do not correct the illumination field (only exposure/colour gains)")
    parser.add_argument("--min-step", type=float, default=DEFAULT_MIN_STEP, help="column-step threshold for the summary (fraction)")
    parser.add_argument("--measure-scale", type=float, default=MEASURE_SCALE, help="tiles are registered and measured at this scale")
    parser.add_argument("--max-rows", type=int, default=24, help="rows sampled per column for the substrate-plateau cross-check")
    parser.add_argument("--report-only", action="store_true", help="measure and solve, write the report, change nothing")
    parser.add_argument("--stitch", action="store_true", help="run run_stitch.py on the derived dataset afterwards")
    parser.add_argument("--no-ai", action="store_true", help="pass --no-ai to run_stitch.py (default: Kimi when configured)")
    parser.add_argument("--scale-div", type=int, default=4, help="run_stitch.py registration cache downsampling (must divide the tile size)")
    parser.add_argument("--out-scale", type=float, default=0.25, help="mosaic output scale; above 1/scale-div needs --full")
    parser.add_argument("--full", action="store_true", help="pass --full to run_stitch.py: render from the original tiles instead of the cache")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path, default=None, help="report JSON path (default: <out>/colour_match_report.json or beside --data)")
    args = parser.parse_args(argv)
    log = lambda m: print(m, flush=True)  # noqa: E731
    data = args.data.resolve()
    out = (args.out or data.with_name(data.name + "_colour_matched")).resolve()
    try:
        if args.out_scale > 1 / args.scale_div + 1e-12 and not args.full:
            raise ColourMatchError(f"--out-scale {args.out_scale} exceeds the cache resolution 1/{args.scale_div}; add --full "
                                   "(render from the original tiles) or lower --out-scale")
        inv = inventory(data)
        log(f"[inventory] {len(inv['grid'])} tiles, {len(inv['rows'])} rows x {len(inv['cols'])} columns")
        direction = detect_direction(inv)
        log(f"[direction] next column lies {direction['next_column_side']} at {np.round(direction['next_column_vector_dxdy']).tolist()} px (NCC {direction['next_column_ncc']}); "
            f"next row at {np.round(direction['next_row_vector_dxdy']).tolist() if direction['next_row_vector_dxdy'] else None}")
        ff = None if args.no_flat_field else estimate_flatfield(inv, log=log)
        edges = measure_edges(inv, direction, ff, scale=args.measure_scale, log=log)
        solution = solve_gains(edges, inv, mode=args.gain_mode, tile_penalty=args.tile_penalty, min_step=args.min_step, log=log)
        plateaus = plateau_columns(inv, max_rows=args.max_rows)
        report = {"schema_version": SCHEMA_VERSION, "kind": "colour_match_report", "data_dir": str(data),
                  "measured_at_utc": datetime.now(timezone.utc).isoformat(), "direction": direction,
                  "flat_field": ff["info"] if ff else None, "edges": edges, "gains": solution,
                  "substrate_plateau_by_column_original_tiles": {str(c): v for c, v in plateaus.items()}}
        log(f"[decision] column steps (> {args.min_step:.1%}): {[(s['col_a'], s['col_b']) for s in solution['column_steps']]}; "
            f"{solution['tiles_corrected']} tiles get a gain != 1")
        report_path = args.report or (out / "colour_match_report.json" if not args.report_only else data.with_name(data.name + "_colour_match_report.json"))
        if args.report_only:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            log(f"[report] {report_path} (nothing corrected)")
            return 0
        manifest = apply(inv, solution, direction, out, source_dir=data, ff=ff, log=log)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        profile = stitch_profile(inv, direction, scale_div=args.scale_div, out_scale=args.out_scale)
        profile_path = out / "stitch_profile.json"
        profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        cmd = stitch_command(out, profile_path, no_ai=args.no_ai, workers=args.workers, extra=["--full"] if args.full else None)
        (out / "stitch_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        n_copies = sum(1 for t in manifest["tiles"] if t["kind"] == "corrected_copy")
        log(f"[dataset] {out}: {len(manifest['tiles'])} tiles, {n_copies} corrected copies; profile {profile['nominal_vectors']}")
        log("[stitch] " + " ".join(cmd))
        if args.stitch:
            return subprocess.call(cmd, cwd=str(REPO))
        return 0
    except ColourMatchError as exc:
        log(f"Colour match stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
