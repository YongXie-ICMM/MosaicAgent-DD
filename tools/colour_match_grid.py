#!/usr/bin/env python3
"""Detect and correct colour steps between scan columns of a flat grid, then stitch.

Why: a long scan can be interrupted and resumed (new camera session, different gain or
white balance), or the camera's automatic exposure can drift between columns. The mosaic
then shows a visible step at a column boundary, and the layer model reads the shifted
columns differently. This tool measures the step where the evidence is strongest, in the
registered overlap of physically adjacent tiles, decides which boundaries are real colour
steps, applies one recorded per-channel gain per column segment, and leaves everything
else untouched.

Method (all numbers are 8-bit RGB medians; nothing is fitted per pixel):

1. Inventory ``mosaic_r<row>_c<col>.png`` tiles. Register a few pairs to learn on which
   side the next column lies (the scanner's ``invert_x`` moved left here) and the
   horizontal / vertical neighbour vectors.
2. For every column boundary (c, c+1) register each row pair by template matching and
   take the per-channel ratio of medians A/B over the overlap. The median over rows is
   the boundary ratio. Within one run this ratio is not 1: the left edge of a field is
   darker than its right edge (illumination falloff), so the median over *all*
   boundaries is the vignetting baseline. The excess of a boundary over that baseline
   is the colour step.
3. A boundary whose excess exceeds ``--min-step`` (default 2.5 %) in any channel is a
   step. Columns between steps form segments; the longest segment is the reference
   (gain 1). Gains are chained across the steps, so several interruptions are handled.
   Bare-substrate plateau medians (``flakepipeline.color_diagnostics``) are reported
   as an independent cross-check.
4. Output is a derived dataset: corrected tiles are new PNG copies, untouched tiles are
   hard links to the originals (no pixel of the originals changes). When the next
   column lies to the left, column indices are mirrored (``new_col = ncols-1-col``) so
   the stitcher's "column index increases to the right" convention holds; the mapping
   is recorded. If the source directory carries ``session.json`` / ``events.jsonl``,
   derived copies with the new file hashes and the provenance of every tile are
   written, mirroring the scanner's own ``derived_merge`` convention.
5. ``--stitch`` runs ``run_stitch.py`` on the derived dataset with a measured stitch
   profile. Kimi is used whenever ``run_stitch.py`` finds credentials (``--no-ai`` is
   passed through only when requested), as the owner requires.

Limits: one gain per column; drift inside a column is reported in the per-row residuals
but not corrected. The gains describe this acquisition, not the camera; the analysis
colour check against the reference substrate colour still runs on the result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
SCHEMA_VERSION = 1
DEFAULT_MIN_STEP = 0.025
GAIN_LIMITS = (0.5, 2.0)


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
        raise ColourMatchError("At least two columns are needed to compare column colours")
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
    """Which side the next column and the next row lie on, with median vectors."""
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
# Boundary colour statistics
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
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    bright = lum >= np.percentile(lum, 50)
    return {"ratio_all": (np.median(a, 0) / np.median(b, 0)).tolist(),
            "ratio_bright": (np.median(a[bright], 0) / np.median(b[bright], 0)).tolist(),
            "pixels": int(ok.sum()), "overlap_wh": [int(xa1 - xa0), int(ya1 - ya0)]}


def measure_boundaries(inv: dict, direction: dict, *, max_rows: int = 24, min_ncc: float = 0.6, log=print) -> list[dict]:
    """One record per column boundary with per-row registered overlap ratios."""
    rows, cols, grid = inv["rows"], inv["cols"], inv["grid"]
    side = direction["next_column_side"]
    pdx, pdy = direction["next_column_vector_dxdy"]
    step = max(1, len(rows) // max_rows)
    sample_rows = rows[::step]
    out = []
    for ci in range(len(cols) - 1):
        ca, cb = cols[ci], cols[ci + 1]
        per_row = []
        for r in sample_rows:
            A, B = _load_rgb(grid[(r, ca)]), _load_rgb(grid[(r, cb)])
            dx, dy, score = register(A, B, side)
            if score < min_ncc or abs(dx - pdx) > 0.1 * A.shape[1] or abs(dy - pdy) > 0.15 * A.shape[0]:
                continue
            stats = overlap_ratio(A, B, dx, dy)
            if stats:
                stats.update(row=r, dx=dx, dy=dy, ncc=round(score, 3))
                per_row.append(stats)
        rec = {"col_a": ca, "col_b": cb, "rows_sampled": len(sample_rows), "rows_registered": len(per_row)}
        if per_row:
            arr = np.array([p["ratio_all"] for p in per_row])
            rec["ratio_all_median"] = np.median(arr, 0).round(5).tolist()
            rec["ratio_all_iqr"] = (np.percentile(arr, 75, 0) - np.percentile(arr, 25, 0)).round(5).tolist()
            rec["ratio_bright_median"] = np.median([p["ratio_bright"] for p in per_row], 0).round(5).tolist()
            rec["per_row"] = per_row
        out.append(rec)
        log(f"[boundary] c{ca}|c{cb}: {len(per_row)}/{len(sample_rows)} rows registered"
            + (f", A/B median {np.round(rec['ratio_all_median'], 4).tolist()}" if per_row else ""))
    return out


def plateau_columns(inv: dict, *, max_rows: int = 24) -> dict:
    """Bare-substrate plateau median RGB per column (independent cross-check)."""
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
# Decision: steps, segments, gains
# ----------------------------------------------------------------------------
def decide_gains(boundaries: list[dict], cols: list[int], *, min_step: float = DEFAULT_MIN_STEP) -> dict:
    usable = [b for b in boundaries if b.get("ratio_all_median")]
    if len(usable) < 1:
        raise ColourMatchError("No column boundary could be registered; colours cannot be compared")
    ratios = np.array([b["ratio_all_median"] for b in usable])
    baseline = np.median(ratios, 0)          # vignetting-only ratio: most boundaries are within one run
    if len(usable) < 3:
        baseline_note = "fewer than 3 boundaries: baseline is unreliable, treat steps with care"
    else:
        baseline_note = "median over all boundaries"
    steps = []
    for b in boundaries:
        if not b.get("ratio_all_median"):
            b["excess"] = None
            b["is_step"] = False
            continue
        excess = (np.array(b["ratio_all_median"]) / baseline)
        b["excess"] = excess.round(5).tolist()          # A relative to B after removing vignetting
        b["max_abs_step"] = round(float(np.max(np.abs(excess - 1))), 5)
        b["is_step"] = bool(b["max_abs_step"] > min_step)
        if b["is_step"]:
            steps.append(b)
    # segments of columns separated by steps
    segments, current = [], [cols[0]]
    for b, c_next in zip(boundaries, cols[1:]):
        if b.get("is_step"):
            segments.append(current)
            current = [c_next]
        else:
            current.append(c_next)
    segments.append(current)
    reference = max(segments, key=len)
    # chain log-gains: gain[c_b] / gain[c_a] = excess (so that corrected B matches A's level)
    log_rel = {cols[0]: np.zeros(3)}
    for b, c_next in zip(boundaries, cols[1:]):
        e = np.array(b["excess"]) if b.get("is_step") else np.ones(3)
        log_rel[c_next] = log_rel[b["col_a"]] + np.log(e)
    ref_level = np.mean([log_rel[c] for c in reference], axis=0)
    gains = {}
    for c in cols:
        g = np.exp(log_rel[c] - ref_level)
        gains[c] = g.round(5).tolist()
        if np.any(g < GAIN_LIMITS[0]) or np.any(g > GAIN_LIMITS[1]):
            raise ColourMatchError(f"Implausible gain {g} for column {c}; refusing to correct")
    corrected = [c for c in cols if c not in reference and any(abs(x - 1) > 1e-6 for x in gains[c])]
    return {"vignetting_baseline_ratio": baseline.round(5).tolist(), "baseline_note": baseline_note,
            "min_step": min_step, "steps": [{"col_a": s["col_a"], "col_b": s["col_b"], "excess": s["excess"],
                                             "max_abs_step": s["max_abs_step"]} for s in steps],
            "segments": segments, "reference_segment": reference, "gains_rgb": {str(c): gains[c] for c in cols},
            "corrected_columns": corrected}


# ----------------------------------------------------------------------------
# Apply: derived dataset
# ----------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply(inv: dict, decision: dict, direction: dict, out_dir: Path, *, source_dir: Path, log=print) -> dict:
    from PIL import Image
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ColourMatchError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cols, rows, grid = inv["cols"], inv["rows"], inv["grid"]
    mirror = direction["next_column_side"] == "left"
    ncols = len(cols)
    records = []
    for c in cols:
        gain = np.array(decision["gains_rgb"][str(c)], np.float32)
        correct = any(abs(float(g) - 1) > 1e-6 for g in gain)
        new_c = (ncols - 1 - cols.index(c)) if mirror else cols.index(c)
        for r in rows:
            src = grid[(r, c)]
            dst = out_dir / f"mosaic_r{r}_c{new_c}{src.suffix.lower()}"
            if correct:
                with Image.open(src) as im:
                    rgb = np.asarray(im.convert("RGB")).astype(np.float32)
                out = np.clip(np.rint(rgb * gain), 0, 255).astype(np.uint8)
                Image.fromarray(out).save(dst, format="PNG" if dst.suffix == ".png" else None, compress_level=1)
                kind, clipped = "gain_corrected_copy", float(np.mean(np.any(out >= 255, axis=2)))
            else:
                try:
                    os.link(src, dst)
                    kind = "hard_link_to_original"
                except OSError:            # other volume, or a filesystem without hard links (exFAT)
                    shutil.copy2(src, dst)
                    kind = "byte_copy_of_original"
                clipped = None
            records.append({"row": r, "source_col": c, "col": new_c, "filename": dst.name, "kind": kind,
                            "gain_rgb": gain.round(5).tolist() if correct else None, "clipped_fraction": clipped,
                            "source_filename": src.name, "source_sha256": _sha256(src), "sha256": _sha256(dst),
                            "bytes": dst.stat().st_size})
        log(f"[apply] column c{c} -> c{new_c}: {'gain ' + str(gain.round(4).tolist()) if correct else 'unchanged (hard links)'}")
    manifest = {"schema_version": SCHEMA_VERSION, "kind": "colour_matched_grid", "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_dir": str(source_dir), "mirrored_columns": mirror,
                "column_mapping": f"new_col = {ncols - 1} - index(old_col)" if mirror else "new_col = index(old_col)",
                "direction": direction, "decision": decision, "tiles": records}
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
    note = {"schema_version": SCHEMA_VERSION, "purpose": "stitching-only derived dataset: column colour steps corrected",
            "built_at_utc": now, "source_session_id": session.get("session_id"),
            "column_mapping": manifest["column_mapping"], "mirrored_columns": manifest["mirrored_columns"],
            "gains_rgb": manifest["decision"]["gains_rgb"], "corrected_columns": manifest["decision"]["corrected_columns"],
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
                    derived_colour_correction={"kind": t["kind"], "gain_rgb": t["gain_rgb"], "source_filename": t["source_filename"],
                                               "source_sha256": t["source_sha256"], "source_col": t["source_col"],
                                               "clipped_fraction": t["clipped_fraction"], "source_event_id": base.get("event_id")})
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
    parser.add_argument("--min-step", type=float, default=DEFAULT_MIN_STEP, help="colour step threshold on the vignetting-free boundary ratio (fraction)")
    parser.add_argument("--max-rows", type=int, default=24, help="row pairs sampled per boundary")
    parser.add_argument("--report-only", action="store_true", help="measure and decide, write the report, change nothing")
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
        boundaries = measure_boundaries(inv, direction, max_rows=args.max_rows, log=log)
        decision = decide_gains(boundaries, inv["cols"], min_step=args.min_step)
        plateaus = plateau_columns(inv, max_rows=args.max_rows)
        report = {"schema_version": SCHEMA_VERSION, "kind": "colour_match_report", "data_dir": str(data),
                  "measured_at_utc": datetime.now(timezone.utc).isoformat(), "direction": direction,
                  "boundaries": [{k: v for k, v in b.items() if k != "per_row"} for b in boundaries],
                  "per_row": {f"c{b['col_a']}|c{b['col_b']}": b.get("per_row", []) for b in boundaries},
                  "decision": decision, "substrate_plateau_by_column": {str(c): v for c, v in plateaus.items()}}
        log(f"[decision] vignetting baseline A/B {np.round(decision['vignetting_baseline_ratio'], 4).tolist()}; "
            f"steps at {[(s['col_a'], s['col_b']) for s in decision['steps']]}; reference columns {decision['reference_segment'][0]}..{decision['reference_segment'][-1]}; "
            f"corrected columns {decision['corrected_columns']}")
        for c in decision["corrected_columns"]:
            log(f"           gain c{c} = {decision['gains_rgb'][str(c)]}")
        report_path = args.report or (out / "colour_match_report.json" if not args.report_only else data.with_name(data.name + "_colour_match_report.json"))
        if args.report_only:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            log(f"[report] {report_path} (nothing corrected)")
            return 0
        manifest = apply(inv, decision, direction, out, source_dir=data, log=log)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        profile = stitch_profile(inv, direction, scale_div=args.scale_div, out_scale=args.out_scale)
        profile_path = out / "stitch_profile.json"
        profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        cmd = stitch_command(out, profile_path, no_ai=args.no_ai, workers=args.workers, extra=["--full"] if args.full else None)
        (out / "stitch_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        log(f"[dataset] {out}: {len(manifest['tiles'])} tiles, {len(decision['corrected_columns'])} columns corrected; profile {profile['nominal_vectors']}")
        log("[stitch] " + " ".join(cmd))
        if args.stitch:
            return subprocess.call(cmd, cwd=str(REPO))
        return 0
    except ColourMatchError as exc:
        log(f"Colour match stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
