"""Overlap colour matching for the ordinary stitcher, without copying source tiles.

The existing colour_match_grid solver supplies the RGB gains.  This adapter binds
its measurements to the stitcher's selected Tile identities and solved positions,
including images held inside ZIPs.  It does not change positions, raw images, QC,
or the segmentation model.  The caller saves the returned report and applies the
gains only when rendering.  A connected, measured overlap graph is required;
an isolated tile must never acquire an unsupported, apparently successful gain.
"""
from __future__ import annotations

import json
import operator

import numpy as np

import tiles as T
from blend import _resolve_positions
from tools import colour_match_grid as cm


class StitchColourError(ValueError):
    """An actionable failure carrying the evidence measured before the failure."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _fail(report, reason, message):
    report.update(status="failed", failure_reason=reason, error=message)
    raise StitchColourError(message, report)


def _grid_integer(value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("boolean is not a grid coordinate")
    return operator.index(value)


def _connected_graph(inv, edges):
    """Return actual components; rejected edges provide no gain constraint."""
    neighbours = {key: set() for key in inv["grid"]}
    for edge in edges:
        if "ratio_all" not in edge:
            continue
        a, b = tuple(edge["a"]), tuple(edge["b"])
        neighbours[a].add(b)
        neighbours[b].add(a)
    remaining = set(neighbours)
    components = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component, pending = [seed], [seed]
        while pending:
            for key in sorted(neighbours[pending.pop()]):
                if key in remaining:
                    remaining.remove(key)
                    component.append(key)
                    pending.append(key)
        components.append(sorted(component))
    return components


def solve_tile_gains(tiles, positions, flatfield, *, cache_dir=None, scale_div=4,
                     binding=None, log=print):
    """Return ``({tid: [R, G, B]}, report)`` for a complete rectangular grid.

    ``positions`` are full-resolution solved coordinates, in exactly the renderer's
    format. ``flatfield`` uses the renderer's BGR order and may be downsampled; it
    is converted to RGB for overlap measurement. Native pixels are loaded through
    tiles.load_full, not the registration thumbnail cache, so narrow overlaps keep
    enough samples. Measurement width is 960 px, or native width for smaller tiles.
    ``binding`` is a caller-supplied JSON provenance record; it is copied unchanged.

    ``cache_dir`` is recorded only for provenance. Reusing compressed thumbnails
    would couple colour estimation to registration scale and lose narrow overlaps.
    A single-tile input has no relative colour difference and returns identity.
    No output file or corrected source dataset is written here.
    """
    tiles = list(tiles)
    report = {
        "schema_version": 1, "status": "measuring",
        "method": "registered_overlap_rgb_column_plus_penalised_tile",
        "source_binding": json.loads(json.dumps(binding or {}, allow_nan=False)),
        "source_pixels_modified": False, "positions_changed": False,
        "gain_channel_order": "RGB", "flatfield_input_order": "BGR",
        "gain_limits": list(cm.GAIN_LIMITS), "tile_penalty": cm.TILE_PENALTY,
        "measurement_source": "original_tile_pixels",
        "registration_cache_used_for_measurement": False,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "scale_div": scale_div, "tiles": [], "edges": [],
    }
    if not tiles:
        _fail(report, "empty_grid", "Colour matching has no selected tiles.")
    try:
        pos = _resolve_positions(tiles, positions)
    except Exception as error:
        _fail(report, "invalid_positions", f"Colour matching positions are invalid: {error}")
    grid, tid_set, pos_by_key = {}, set(), {}
    for tile, point in zip(tiles, pos):
        try:
            key = (_grid_integer(tile.nominal_row), _grid_integer(tile.col))
        except (TypeError, ValueError) as error:
            _fail(report, "invalid_grid", f"Tile {tile.tid} has invalid grid identity: {error}")
        if key in grid or tile.tid in tid_set:
            _fail(report, "duplicate_identity", f"Colour matching has duplicate grid position or tile ID: {key}, {tile.tid}")
        if point is None:
            _fail(report, "missing_position", f"Colour matching needs a finite solved position for {tile.tid}.")
        grid[key], pos_by_key[key] = tile, np.asarray(point, dtype=np.float64)
        tid_set.add(tile.tid)
        report["tiles"].append({"tid": tile.tid, "row": key[0], "col": key[1],
                                "source_container": str(tile.zip_path), "source_inner": str(tile.inner),
                                "solved_position_xy": list(point)})
    rows, cols = sorted({key[0] for key in grid}), sorted({key[1] for key in grid})
    expected = {(r, c) for r in rows for c in cols}
    missing = sorted(expected - grid.keys())
    report.update(rows=rows, cols=cols, tile_count=len(tiles),
                  missing_grid_positions=[list(key) for key in missing])
    if missing or rows != list(range(rows[0], rows[-1] + 1)) or cols != list(range(cols[0], cols[-1] + 1)):
        _fail(report, "incomplete_grid", "Automatic colour matching requires a contiguous complete rectangular grid; missing or excluded positions must be resolved first.")
    inv = {"rows": rows, "cols": cols, "grid": grid}
    if len(tiles) == 1:
        gains = {tiles[0].tid: [1.0, 1.0, 1.0]}
        report.update(status="identity_single_tile", gains_rgb_by_tid=gains,
                      graph={"connected": True, "components": [[tiles[0].tid]],
                             "anchor_tid": tiles[0].tid},
                      note="One tile has no relative colour difference to measure.")
        return gains, report

    def vector(kind):
        pairs = ([(r, c, r, cn) for c, cn in zip(cols, cols[1:]) for r in rows]
                 if kind == "h" else
                 [(r, c, rn, c) for r, rn in zip(rows, rows[1:]) for c in cols])
        return (np.median([pos_by_key[(rb, cb)] - pos_by_key[(ra, ca)]
                           for ra, ca, rb, cb in pairs], axis=0) if pairs else None)

    col_vec, row_vec = vector("h"), vector("v")
    if col_vec is not None and abs(col_vec[0]) < 1e-6:
        _fail(report, "invalid_geometry", "Adjacent columns have no horizontal separation in solved positions.")
    if row_vec is not None and row_vec[1] <= 0:
        _fail(report, "invalid_geometry", "Recorded rows must increase downwards for colour-overlap measurement; solved geometry disagrees.")
    direction = {
        "source": "median_adjacent_solved_positions_full_resolution",
        "next_column_side": "left" if col_vec is not None and col_vec[0] < 0 else "right",
        "next_column_vector_dxdy": col_vec.tolist() if col_vec is not None else [0.0, 0.0],
        "next_row_vector_dxdy": row_vec.tolist() if row_vec is not None else None,
    }
    report["direction"] = direction
    image_shape = None

    def load_rgb(tile):
        nonlocal image_shape
        try:
            bgr = T.load_full(tile)
        except Exception as error:
            raise ValueError(f"Cannot load colour source {tile.tid}: {error}") from error
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError(f"Colour source {tile.tid} is not a decoded uint8 three-channel image.")
        if image_shape is None:
            image_shape = bgr.shape
        elif bgr.shape != image_shape:
            raise ValueError(f"Colour source {tile.tid} dimensions {bgr.shape} differ from {image_shape}.")
        return np.ascontiguousarray(bgr[:, :, ::-1])

    try:
        first = load_rgb(tiles[0])
        measurement_scale = min(1.0, 960.0 / first.shape[1])
        report["native_image_size_wh"] = [first.shape[1], first.shape[0]]
        report["measurement_scale"] = measurement_scale
        ff = None
        if flatfield is not None:
            field = np.asarray(flatfield)
            if field.ndim != 3 or field.shape[2] != 3 or not np.all(np.isfinite(field)) or np.any(field <= 0):
                raise ValueError("The colour measurement flat field must have finite positive BGR channels.")
            ff = {"field": np.ascontiguousarray(field[:, :, ::-1]), "info": {"source": "renderer_flatfield"}}
        try:
            edges = cm.measure_edges(inv, direction, ff, scale=measurement_scale,
                                     load_rgb=load_rgb, log=log)
        except cm.ColourMatchError as error:
            if not hasattr(error, "edges"):
                raise
            edges = error.edges
    except Exception as error:
        _fail(report, "measurement_failed", f"Colour overlap measurement failed: {error}")
    for edge in edges:
        if "ratio_all" in edge:
            ratio = np.asarray(edge["ratio_all"], dtype=np.float64)
            if ratio.shape != (3,) or not np.all(np.isfinite(ratio)) or np.any(ratio <= 0):
                edge.pop("ratio_all")
                edge["rejected"] = "invalid_rgb_ratio"
        edge["tid_a"] = grid[tuple(edge["a"])].tid
        edge["tid_b"] = grid[tuple(edge["b"])].tid
    report["edges"] = edges
    report["edges_measured"] = len(edges)
    report["edges_used"] = sum("ratio_all" in edge for edge in edges)
    report["edges_rejected"] = len(edges) - report["edges_used"]
    components = _connected_graph(inv, edges)
    report["graph"] = {"connected": len(components) == 1,
                       "anchor_tid": grid[min(grid)].tid,
                       "components": [[grid[key].tid for key in component] for component in components]}
    if len(components) != 1:
        _fail(report, "disconnected_colour_graph",
              f"Colour overlaps form {len(components)} disconnected components; refusing unsupported tile gains. Check the listed rejected overlaps and original images.")
    try:
        solution = cm.solve_gains(edges, inv, mode="tile", log=log)
    except Exception as error:
        _fail(report, "gain_solve_failed", f"Colour gain solve failed: {error}")
    gains = {tile.tid: solution["gains_rgb_by_tile"][f"r{r}_c{c}"] for (r, c), tile in grid.items()}
    gain_values = np.asarray(list(gains.values()), dtype=np.float64)
    if not np.all(np.isfinite(gain_values)) or np.any(gain_values < cm.GAIN_LIMITS[0]) or np.any(gain_values > cm.GAIN_LIMITS[1]):
        _fail(report, "invalid_gain", "Colour solve produced a nonfinite or implausible gain.")
    before, after = [], []
    for edge in edges:
        if "ratio_all" in edge:
            ratio = np.asarray(edge["ratio_all"], dtype=np.float64)
            corrected = ratio * np.asarray(gains[edge["tid_a"]]) / np.asarray(gains[edge["tid_b"]])
            edge["ratio_after_gain_rgb"] = corrected.tolist()
            before.append(np.log(ratio))
            after.append(np.log(corrected))
    report.update(status="matched", gains_rgb_by_tid=gains,
                  solver={key: value for key, value in solution.items() if key != "gains_rgb_by_tile"},
                  measured_overlap_log_rms_before=float(np.sqrt(np.mean(np.asarray(before) ** 2))),
                  measured_overlap_log_rms_after=float(np.sqrt(np.mean(np.asarray(after) ** 2))),
                  clipped_gain_tiles=solution["flagged_tiles"],
                  note="Relative multiplicative RGB correction only; clipped image highlights cannot be recovered. Global substrate-reference colour calibration is separate.")
    return gains, report
