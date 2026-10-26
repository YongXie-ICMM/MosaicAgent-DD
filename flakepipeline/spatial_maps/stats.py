"""Exact pixel statistics shared by the separate and combined spatial maps.

Internal class IDs are 0=substrate, 1=1L, 2=2L, 3=TL. The caller must adapt
native MosaicAgent integer masks (1=2L, 2=1L) before calling ``analyze``.
Colour PNGs are decoded by RGB values, never by palette index. Exclusions and
non-scan support are separate, disjoint accounting categories. Validation uses
explicit exceptions so it remains enabled under ``python -O``.
"""
from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path

import numpy as np
from PIL import Image

LAYERS = ("1L", "2L", "TL")
RGB = np.array([(0, 0, 0), (0, 128, 0), (128, 0, 0), (128, 128, 0)], dtype=np.uint8)


def require(condition, message):
    """Reject invalid data without depending on Python assertions."""
    if not condition:
        raise ValueError(message)


def format_pct(value):
    return "N/A" if value is None else f"{value:.1f}%"


def sha(path):
    """Hash a file incrementally, without loading a large source into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path):
    """Read an exact-colour segmentation image into the canonical class IDs.

    Transparent pixels and colours outside ``RGB`` are errors, even if they
    occur in a region that the caller intends to exclude. No interpolation or
    nearest-colour matching is performed.
    """
    with Image.open(path) as image:
        if "transparency" in image.info or "A" in image.getbands():
            alpha = np.asarray(image.convert("RGBA"))[..., 3]
            require(np.all(alpha == 255),
                    "Transparent mask pixels have no defined counting semantics")
        if image.mode == "P":
            raw_palette = image.getpalette("RGB")
            require(raw_palette is not None and len(raw_palette) % 3 == 0,
                    "Palette mask has no valid RGB palette")
            palette = np.asarray(raw_palette, dtype=np.uint8).reshape(-1, 3)
            indices = np.asarray(image)
            lookup = np.full(256, 255, dtype=np.uint8)
            for index in np.unique(indices):
                require(index < len(palette), "Mask index is outside the stored palette")
                match = np.flatnonzero(np.all(RGB == palette[index], axis=1))
                require(len(match) == 1,
                        f"Unknown palette colour {palette[index].tolist()} in {path}")
                lookup[index] = match[0]
            return lookup[indices]
        array = np.asarray(image.convert("RGB"))
        labels = np.full(array.shape[:2], 255, dtype=np.uint8)
        for index, colour in enumerate(RGB):
            labels[np.all(array == colour, axis=2)] = index
        require(not np.any(labels == 255), f"Unknown mask colours in {path}")
        return labels


def _binary_mask(mask, shape, name, default):
    if mask is None:
        return np.full(shape, default, dtype=bool)
    mask = np.asarray(mask)
    require(mask.shape == shape, f"{name} dimensions must match the label mask")
    require(mask.dtype.kind in "biuf" and np.all((mask == 0) | (mask == 1)),
            f"{name} must contain only binary 0/1 or boolean values")
    return mask.astype(bool, copy=True)


def _pixel_integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def analyze(labels, rectangles, columns, rows, valid_mask=None, exclusion_mask=None):
    """Count each retained pixel exactly once in an equal-extent analysis grid.

    Rectangles use half-open integer coordinates ``x1,y1,x2,y2``. Their union
    with ``exclusion_mask`` defines user exclusions. Only supported pixels
    (``valid_mask``) can count as excluded; outside support is ``non_scan_px``.
    All fractions use retained pixels: composition divides by 1L+2L+TL, while
    coverage divides by retained canvas including substrate. Undefined
    percentages are ``None`` (JSON null), not zero or NaN.

    Returns ``(table, xedges, yedges, global_result)``. The input arrays are not
    modified. ``rectangles`` may be an empty sequence or ``None``.
    """
    labels = np.asarray(labels)
    require(labels.ndim == 2 and np.issubdtype(labels.dtype, np.integer)
            and np.all((labels >= 0) & (labels <= 3)),
            "Labels must be a 2D integer array containing only class IDs 0..3")
    require(_pixel_integer(columns) and _pixel_integer(rows),
            "Grid sizes must be integers")
    height, width = labels.shape
    require(1 <= columns <= width and 1 <= rows <= height,
            "Grid dimensions must fit the image")
    columns, rows = int(columns), int(rows)
    valid = _binary_mask(valid_mask, labels.shape, "valid_mask", True)
    region = _binary_mask(exclusion_mask, labels.shape, "exclusion_mask", False)
    for rect in (() if rectangles is None else rectangles):
        require(isinstance(rect, dict) and all(k in rect for k in ("x1", "y1", "x2", "y2")),
                "Exclusion rectangles must contain x1, y1, x2, y2")
        x1, y1, x2, y2 = (rect[k] for k in ("x1", "y1", "x2", "y2"))
        require(all(_pixel_integer(v) for v in (x1, y1, x2, y2)),
                "Exclusion coordinates must be integer pixels")
        require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
                "Exclusion rectangle outside source mask")
        region[y1:y2, x1:x2] = True
    excluded = valid & region
    retained = valid & ~region
    xedges = np.linspace(0, width, columns + 1, dtype=int)
    yedges = np.linspace(0, height, rows + 1, dtype=int)
    table = []
    for r, (y0, y1) in enumerate(zip(yedges[:-1], yedges[1:])):
        for c, (x0, x1) in enumerate(zip(xedges[:-1], xedges[1:])):
            crop = labels[y0:y1, x0:x1]
            keep = retained[y0:y1, x0:x1]
            counts = np.bincount(crop[keep].astype(np.intp, copy=False), minlength=4)
            kept_px = int(counts.sum())
            crystal = int(counts[1:].sum())
            total = int(crop.size)
            scan_px = int(valid[y0:y1, x0:x1].sum())
            excluded_px = int(excluded[y0:y1, x0:x1].sum())
            status = ("no_scan" if not scan_px else "excluded" if not kept_px
                      else "no_crystal" if not crystal else "valid")
            record = {
                "row_top_1based": r + 1, "column_left_1based": c + 1,
                "x0_px": int(x0), "y0_px": int(y0), "x1_px": int(x1), "y1_px": int(y1),
                "tile_canvas_px": total, "valid_scan_px": scan_px,
                "non_scan_px": total - scan_px, "excluded_px": excluded_px,
                "retained_canvas_px": kept_px, "substrate_background_px": int(counts[0]),
                "crystal_px": crystal,
                "crystal_coverage_pct": 100 * crystal / kept_px if kept_px else None,
                "status": status,
            }
            for k, layer in enumerate(LAYERS, 1):
                record[f"{layer}_px"] = int(counts[k])
                record[f"{layer}_of_crystal_pct"] = 100 * int(counts[k]) / crystal if crystal else None
                record[f"{layer}_of_retained_canvas_pct"] = 100 * int(counts[k]) / kept_px if kept_px else None
            require(kept_px + excluded_px + record["non_scan_px"] == total,
                    "Tile pixels not conserved")
            if crystal:
                require(np.isclose(sum(record[f"{k}_of_crystal_pct"] for k in LAYERS), 100),
                        "Composition does not sum to 100%")
            if kept_px:
                require(np.isclose(sum(record[f"{k}_of_retained_canvas_pct"] for k in LAYERS),
                                   record["crystal_coverage_pct"]), "Coverage classes not conserved")
            table.append(record)
    counts = np.bincount(labels[retained].astype(np.intp, copy=False), minlength=4)
    require(sum(t["substrate_background_px"] for t in table) == int(counts[0]),
            "Substrate counts not conserved")
    for k, layer in enumerate(LAYERS, 1):
        require(sum(t[f"{layer}_px"] for t in table) == int(counts[k]),
                "Class counts not conserved")
    kept_px = int(counts.sum())
    crystal = int(counts[1:].sum())
    global_result = {
        "canvas_px": width * height, "valid_scan_px": int(valid.sum()),
        "non_scan_px": int(valid.size - valid.sum()),
        "retained_canvas_px": kept_px, "crystal_px": crystal,
        "excluded_px": int(excluded.sum()),
        "crystal_coverage_pct": 100 * crystal / kept_px if kept_px else None,
        "counts_px": {k: int(counts[i]) for i, k in enumerate(("background",) + LAYERS)},
        "of_crystal_pct": {k: 100 * int(counts[i]) / crystal if crystal else None
                           for i, k in enumerate(LAYERS, 1)},
        "of_retained_canvas_pct": {k: 100 * int(counts[i]) / kept_px if kept_px else None
                                   for i, k in enumerate(LAYERS, 1)},
    }
    require(sum(t["tile_canvas_px"] for t in table) == width * height,
            "Grid does not partition canvas")
    for field in ("valid_scan_px", "non_scan_px", "excluded_px", "retained_canvas_px"):
        require(sum(t[field] for t in table) == global_result[field],
                f"Global {field} not conserved")
    require(kept_px + global_result["excluded_px"] + global_result["non_scan_px"] == width * height,
            "Global pixels not conserved")
    for layer in LAYERS if crystal else ():
        weighted = sum(t[f"{layer}_of_crystal_pct"] * t["crystal_px"]
                       for t in table if t["crystal_px"]) / crystal
        require(np.isclose(weighted, global_result["of_crystal_pct"][layer], atol=1e-12),
                "Weighted global fractions differ")
    return table, xedges, yedges, global_result
