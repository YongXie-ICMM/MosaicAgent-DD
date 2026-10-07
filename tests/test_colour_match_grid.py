"""Illumination field, per-tile exposure gains and column colour steps on a synthetic flat grid (no hardware, no network)."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import colour_match_grid as cm  # noqa: E402

SUBSTRATE = np.array([219.0, 171.0, 170.0])
FLAKE = np.array([196.0, 129.0, 171.0])
TILE_W, TILE_H = 320, 180
COL_VEC = (-290, 8)      # next column lies LEFT (scanner invert_x), slight shear
ROW_VEC = (4, 150)       # next row lies below (30 px overlap)
X_TILT = 0.92            # left edge / right edge of the illumination field
Y_TILT = 0.96            # bottom edge / top edge


def specimen(seed=0, size=(1700, 1000)):
    """Bright substrate with many dark triangles and fine texture, as a float RGB canvas."""
    from PIL import ImageDraw
    rng = np.random.default_rng(seed)
    w, h = size
    img = Image.new("RGB", (w, h), tuple(int(v) for v in SUBSTRATE))
    draw = ImageDraw.Draw(img)
    for _ in range(600):
        cx, cy, s = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(8, 40))
        colour = tuple(int(v) for v in np.clip(FLAKE + rng.normal(0, 3, 3), 0, 255))
        draw.polygon([(cx, cy), (cx - int(0.6 * s), cy + s), (cx + int(0.6 * s), cy + s)], fill=colour)
    canvas = np.asarray(img).astype(np.float32)
    canvas += rng.normal(0, 1.5, canvas.shape)
    return canvas


def vignette():
    vx = np.linspace(X_TILT, 1.0, TILE_W, dtype=np.float32)
    vy = np.linspace(1.0, Y_TILT, TILE_H, dtype=np.float32)
    return (vy[:, None] * vx[None, :])[:, :, None]


def make_grid(tmp_path, *, ncols=5, nrows=4, gain_last_cols=None, n_gain_cols=2, jitter=0.0, with_records=True, seed=0):
    """Cut tiles so that column index increases towards the LEFT; apply a 2-D illumination field, optional
    per-tile exposure jitter and an optional brightness step on the last columns (a resumed run).
    Returns the folder and the per-tile exposure factor that was applied (1 = nominal)."""
    canvas = specimen(seed)
    rng = np.random.default_rng(seed + 1)
    folder = tmp_path / "scan"
    folder.mkdir()
    x_right = canvas.shape[1] - TILE_W - 10 - (nrows - 1) * ROW_VEC[0]
    vig = vignette()
    events, exposure = [], {}
    for c in range(ncols):
        x0 = x_right + c * COL_VEC[0]
        for r in range(nrows):
            y0 = 10 + r * ROW_VEC[1] + c * COL_VEC[1]
            x = x0 + r * ROW_VEC[0]
            factor = np.ones(3, np.float32)
            if gain_last_cols is not None and c >= ncols - n_gain_cols:
                factor = factor / np.array(gain_last_cols, np.float32)      # brighter / bluer second run
            if jitter:
                factor = factor * float(np.clip(rng.normal(1.0, jitter), 0.9, 1.1))
            exposure[(r, c)] = factor
            tile = canvas[y0:y0 + TILE_H, x:x + TILE_W] * vig * factor
            arr = np.clip(np.rint(tile), 0, 255).astype(np.uint8)
            name = f"mosaic_r{r}_c{c}.png"
            Image.fromarray(arr).save(folder / name)
            data = (folder / name).read_bytes()
            events.append({"event": "capture_success", "timestamp_utc": "2026-10-06T06:00:00+00:00", "session_id": "s1",
                           "event_id": f"e{r}_{c}", "row": r, "col": c, "filename": name, "image_relative_path": name,
                           "saved_image_resolution": [TILE_W, TILE_H], "image_sha256": __import__("hashlib").sha256(data).hexdigest(),
                           "image_bytes": len(data), "sequence": len(events) + 2})
    if with_records:
        session = {"schema_version": 2, "session_id": "s1", "status": "completed", "stage_simulated": False,
                   "camera": {"backend": "opencv", "actual_resolution": [TILE_W, TILE_H]},
                   "scan_config": {"nx": ncols, "ny": nrows, "dx_steps": 143, "dy_steps": 76, "order": "Y_first", "invert_x": True,
                                   "acquisition_contract": {"schema_version": 1, "expected_image_size": [TILE_W, TILE_H],
                                                            "actual_image_size": [TILE_W, TILE_H],
                                                            "verification_status": "verified_received_frame", "raw_images_resized": False}},
                   "planned_positions": ncols * nrows, "photos_saved": ncols * nrows}
        (folder / "session.json").write_text(json.dumps(session), encoding="utf-8")
        with (folder / "events.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "session_started", "timestamp_utc": "2026-10-06T06:00:00+00:00", "session_id": "s1", "event_id": "start", "sequence": 1}) + "\n")
            for e in events:
                f.write(json.dumps(e) + "\n")
    return folder, exposure


def quiet(*_):
    pass


def pipeline(folder, *, flat=True, mode="tile"):
    inv = cm.inventory(folder)
    direction = cm.detect_direction(inv)
    ff = cm.estimate_flatfield(inv, log=quiet) if flat else None
    edges = cm.measure_edges(inv, direction, ff, log=quiet)
    solution = cm.solve_gains(edges, inv, mode=mode, log=quiet)
    return inv, direction, ff, edges, solution


def test_direction_and_vectors_are_measured_from_the_tiles(tmp_path):
    folder, _ = make_grid(tmp_path)
    inv = cm.inventory(folder)
    direction = cm.detect_direction(inv)
    assert direction["next_column_side"] == "left"
    assert abs(direction["next_column_vector_dxdy"][0] - COL_VEC[0]) <= 2
    assert abs(direction["next_column_vector_dxdy"][1] - COL_VEC[1]) <= 2
    assert abs(direction["next_row_vector_dxdy"][0] - ROW_VEC[0]) <= 2
    assert abs(direction["next_row_vector_dxdy"][1] - ROW_VEC[1]) <= 2


def test_illumination_field_is_recovered_from_the_tiles(tmp_path):
    folder, _ = make_grid(tmp_path, jitter=0.02)
    inv = cm.inventory(folder)
    ff = cm.estimate_flatfield(inv, log=quiet)
    info = ff["info"]
    assert info["model"].startswith("smoothed_median") and info["samples_used"] == 20
    assert np.allclose(info["edge_to_edge_ratio_left_over_right"], X_TILT, atol=0.015), info
    assert np.allclose(info["edge_to_edge_ratio_top_over_bottom"], 1 / Y_TILT, atol=0.015), info
    corr = cm.flatfield_correction(ff, TILE_W, TILE_H)
    assert corr.shape == (TILE_H, TILE_W, 3)
    # after correction a tile's left and right edges sit at the same level
    flat = vignette()[:, :, 0] * corr[:, :, 0]
    assert abs(flat[:, :16].mean() / flat[:, -16:].mean() - 1) < 0.015


def test_per_tile_gains_undo_exposure_jitter_and_the_resumed_run_step(tmp_path):
    gain = (0.94, 0.955, 0.92)
    folder, exposure = make_grid(tmp_path, gain_last_cols=gain, jitter=0.02)
    inv, direction, ff, edges, solution = pipeline(folder)
    assert solution["mode"] == "tile" and solution["edges_used"] >= 28
    # corrected brightness = applied exposure x solved gain must be the same for every tile (up to one global scale)
    rel = np.array([np.log(np.array(solution["gains_rgb_by_tile"][f"r{r}_c{c}"])) + np.log(exposure[(r, c)]) for (r, c) in exposure])
    assert np.all(rel.std(axis=0) < 0.01), rel.std(axis=0)
    # the reference run keeps gain ~1 (20 tiles with 2 % jitter: the mean of 12 reference tiles sets the gauge)
    ref = np.mean([solution["column_median_gain_rgb"][str(c)] for c in (0, 1, 2)], axis=0)
    assert np.allclose(ref, 1.0, atol=0.02), ref
    # per column, applied exposure x solved gain is the same level everywhere: the resumed columns' 6-8 % step is gone
    level = np.array([np.median([np.array(solution["gains_rgb_by_tile"][f"r{r}_c{c}"]) * exposure[(r, c)] for r in range(4)], axis=0)
                      for c in range(5)])
    assert np.all(level.max(axis=0) / level.min(axis=0) < 1.012), level
    assert np.all(np.array(solution["column_median_gain_rgb"]["3"]) < 0.97)
    assert solution["reference_segment"] == [0, 1, 2]
    assert [(s["col_a"], s["col_b"]) for s in solution["column_steps"]] == [(2, 3)]
    assert solution["edge_rms_after"] < 0.4 * solution["edge_rms_before"]


def test_column_mode_gives_one_gain_per_column(tmp_path):
    gain = (0.95, 0.95, 0.93)
    folder, _ = make_grid(tmp_path, gain_last_cols=gain, jitter=0.01)
    inv, direction, ff, edges, solution = pipeline(folder, mode="column")
    for c in range(5):
        per_tile = {tuple(solution["gains_rgb_by_tile"][f"r{r}_c{c}"]) for r in range(4)}
        assert len(per_tile) == 1
    assert np.allclose(solution["column_median_gain_rgb"]["4"], gain, atol=0.015)


def test_uniform_scan_without_flat_field_changes_nothing_and_hard_links(tmp_path):
    folder, _ = make_grid(tmp_path)
    inv, direction, ff, edges, solution = pipeline(folder, flat=False)
    assert solution["column_steps"] == [] and solution["reference_segment"] == [0, 1, 2, 3, 4]
    for t, g in solution["gains_rgb_by_tile"].items():
        assert np.allclose(g, 1.0, atol=0.02), (t, g)
    # with gains forced to exactly 1 the derived dataset is all hard links (originals never duplicated or altered)
    solution["gains_rgb_by_tile"] = {k: [1.0, 1.0, 1.0] for k in solution["gains_rgb_by_tile"]}
    out = tmp_path / "out"
    manifest = cm.apply(inv, solution, direction, out, source_dir=folder, ff=None, log=quiet)
    assert all(t["kind"] == "hard_link_to_original" for t in manifest["tiles"])
    t = manifest["tiles"][0]
    assert os.stat(out / t["filename"]).st_ino == os.stat(folder / t["source_filename"]).st_ino


def test_apply_writes_mirrored_corrected_dataset_with_valid_derived_records(tmp_path):
    gain = (0.94, 0.955, 0.92)
    folder, _ = make_grid(tmp_path, gain_last_cols=gain, jitter=0.02)
    inv, direction, ff, edges, solution = pipeline(folder)
    out = tmp_path / "out"
    manifest = cm.apply(inv, solution, direction, out, source_dir=folder, ff=ff, log=quiet)
    assert manifest["mirrored_columns"] is True
    assert manifest["flat_field"]["file"] == "flat_field.npy" and (out / "flat_field.npy").is_file()
    by_source = {(t["row"], t["source_col"]): t for t in manifest["tiles"]}
    assert by_source[(0, 4)]["col"] == 0 and by_source[(0, 0)]["col"] == 4     # new_col = 4 - old_col
    assert all(t["kind"] == "corrected_copy" and t["flat_field"] for t in manifest["tiles"])
    # corrected pixels equal source x field correction x gain
    t = by_source[(1, 4)]
    src = np.asarray(Image.open(folder / t["source_filename"]).convert("RGB")).astype(np.float32)
    dst = np.asarray(Image.open(out / t["filename"]).convert("RGB")).astype(np.float32)
    corr = cm.flatfield_correction(ff, TILE_W, TILE_H)
    expected = np.clip(np.rint(src * corr * np.array(t["gain_rgb"], np.float32)), 0, 255)
    assert np.max(np.abs(dst - expected)) <= 1 and np.mean(np.abs(dst - expected)) < 0.01   # float rounding order only
    assert cm._sha256(folder / t["source_filename"]) == t["source_sha256"]       # originals untouched
    # derived records: hashes match the new files and pass the stitcher's acquisition check
    from tools.stitch_setup import acquisition_metadata as am
    import stitch_profile as SP
    tiles = SP.scan_grid_tiles(out)
    inventory = SP.inspect_tiles(tiles)
    evidence, warnings = am.inspect_acquisition(out, inventory, "flat-grid")
    assert evidence["present"] and evidence["checksum_verified_selected_tiles"] == len(manifest["tiles"])
    session = json.loads((out / "session.json").read_text(encoding="utf-8"))
    note = session["derived_colour_correction"]
    assert note["gain_mode"] == "tile" and note["flat_field"]["model"].startswith("smoothed_median")
    assert [(s["col_a"], s["col_b"]) for s in note["column_steps"]] == [(2, 3)]
    assert session["position_trusted"] is False
    # the stitch profile is valid and points right / down
    profile = cm.stitch_profile(inv, direction)
    dx_v, dy_v, dx_h, dy_h = profile["nominal_vectors"]
    assert dx_h > 0 and dy_v > 0 and abs(dx_h - 290) <= 2 and abs(dy_v - 150) <= 2


def test_cli_report_only_changes_nothing(tmp_path, capsys):
    folder, _ = make_grid(tmp_path, gain_last_cols=(0.95, 0.95, 0.93))
    before = {p.name: cm._sha256(p) for p in folder.iterdir()}
    code = cm.main(["--data", str(folder), "--report-only", "--max-rows", "4"])
    assert code == 0
    report = json.loads((tmp_path / "scan_colour_match_report.json").read_text(encoding="utf-8"))
    assert [(s["col_a"], s["col_b"]) for s in report["gains"]["column_steps"]] == [(2, 3)]
    assert report["flat_field"]["model"].startswith("smoothed_median") and len(report["edges"]) == 5 * 3 + 4 * 4
    assert {p.name: cm._sha256(p) for p in folder.iterdir()} == before
    assert not (tmp_path / "scan_colour_matched").exists()
    assert "column steps (> 2.5%): [(2, 3)]" in capsys.readouterr().out


def test_cli_builds_dataset_and_stitch_command_without_running(tmp_path):
    folder, _ = make_grid(tmp_path, gain_last_cols=(0.95, 0.95, 0.93))
    out = tmp_path / "derived"
    code = cm.main(["--data", str(folder), "--out", str(out), "--max-rows", "4"])
    assert code == 0
    assert (out / "colour_match_manifest.json").is_file() and (out / "stitch_profile.json").is_file()
    cmd = (out / "stitch_command.txt").read_text(encoding="utf-8")
    assert "run_stitch.py" in cmd and "--no-ai" not in cmd      # Kimi when configured, as the owner requires
    assert not (out / "mosaic.png").exists()
    cmd_offline = cm.stitch_command(out, out / "stitch_profile.json", no_ai=True)
    assert "--no-ai" in cmd_offline


def test_cli_no_flat_field_keeps_unchanged_tiles_as_links(tmp_path):
    folder, _ = make_grid(tmp_path, gain_last_cols=(0.95, 0.95, 0.93))
    out = tmp_path / "derived"
    assert cm.main(["--data", str(folder), "--out", str(out), "--no-flat-field", "--gain-mode", "column", "--max-rows", "4"]) == 0
    manifest = json.loads((out / "colour_match_manifest.json").read_text(encoding="utf-8"))
    assert manifest["flat_field"] is None
    kinds = {t["source_col"]: t["kind"] for t in manifest["tiles"]}
    assert kinds[3] == kinds[4] == "corrected_copy"
    assert all(not t["flat_field"] for t in manifest["tiles"])


def test_falls_back_to_byte_copies_where_hard_links_are_impossible(tmp_path, monkeypatch):
    folder, _ = make_grid(tmp_path)
    inv, direction, ff, edges, solution = pipeline(folder, flat=False)
    solution["gains_rgb_by_tile"] = {k: [1.0, 1.0, 1.0] for k in solution["gains_rgb_by_tile"]}

    def no_links(*_a, **_k):
        raise OSError("Operation not supported")
    monkeypatch.setattr(cm.os, "link", no_links)
    out = tmp_path / "out"
    manifest = cm.apply(inv, solution, direction, out, source_dir=folder, ff=None, log=quiet)
    assert all(t["kind"] == "byte_copy_of_original" for t in manifest["tiles"])
    t = manifest["tiles"][0]
    assert cm._sha256(out / t["filename"]) == t["source_sha256"] == t["sha256"]


def test_cli_full_resolution_needs_full_flag(tmp_path, capsys):
    folder, _ = make_grid(tmp_path, gain_last_cols=(0.95, 0.95, 0.93))
    out = tmp_path / "derived"
    assert cm.main(["--data", str(folder), "--out", str(out), "--max-rows", "4", "--scale-div", "2", "--out-scale", "1"]) == 1
    assert "add --full" in capsys.readouterr().out and not out.exists()
    assert cm.main(["--data", str(folder), "--out", str(out), "--max-rows", "4", "--scale-div", "2", "--out-scale", "1", "--full"]) == 0
    cmd = (out / "stitch_command.txt").read_text(encoding="utf-8")
    assert "--full" in cmd and "--workers 4" in cmd


def test_refuses_implausible_gain(tmp_path):
    # the resumed columns are 2.5x darker: the gain to match them (2.5) is outside the plausible range for 40 % of the tiles
    folder, _ = make_grid(tmp_path, gain_last_cols=(2.5, 2.5, 2.5))
    inv = cm.inventory(folder)
    direction = cm.detect_direction(inv)
    edges = cm.measure_edges(inv, direction, None, log=quiet)
    with pytest.raises(cm.ColourMatchError, match="Implausible"):
        cm.solve_gains(edges, inv, log=quiet)
