"""Column colour-step detection and correction on a synthetic flat grid (no hardware, no network)."""
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
ROW_VEC = (4, 160)       # next row lies below


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


def make_grid(tmp_path, *, ncols=5, nrows=4, gain_last_cols=None, n_gain_cols=2, with_records=True, seed=0):
    """Cut tiles so that column index increases towards the LEFT; apply vignetting and an optional
    brightness step on the last columns (a resumed run with different gain / white balance)."""
    canvas = specimen(seed)
    folder = tmp_path / "scan"
    folder.mkdir()
    x_right = canvas.shape[1] - TILE_W - 10 - (nrows - 1) * ROW_VEC[0]
    vignette = np.linspace(0.92, 1.0, TILE_W, dtype=np.float32)[None, :, None]
    events = []
    for c in range(ncols):
        x0 = x_right + c * COL_VEC[0]
        for r in range(nrows):
            y0 = 10 + r * ROW_VEC[1] + c * COL_VEC[1]
            x = x0 + r * ROW_VEC[0]
            tile = canvas[y0:y0 + TILE_H, x:x + TILE_W] * vignette
            if gain_last_cols is not None and c >= ncols - n_gain_cols:
                tile = tile / np.array(gain_last_cols, np.float32)      # brighter / bluer second run
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
    return folder


def test_direction_and_vectors_are_measured_from_the_tiles(tmp_path):
    folder = make_grid(tmp_path)
    inv = cm.inventory(folder)
    direction = cm.detect_direction(inv)
    assert direction["next_column_side"] == "left"
    assert abs(direction["next_column_vector_dxdy"][0] - COL_VEC[0]) <= 2
    assert abs(direction["next_column_vector_dxdy"][1] - COL_VEC[1]) <= 2
    assert abs(direction["next_row_vector_dxdy"][0] - ROW_VEC[0]) <= 2
    assert abs(direction["next_row_vector_dxdy"][1] - ROW_VEC[1]) <= 2


def test_step_is_found_and_gain_estimated_within_one_percent(tmp_path):
    gain = (0.94, 0.955, 0.92)
    folder = make_grid(tmp_path, gain_last_cols=gain)
    inv = cm.inventory(folder)
    direction = cm.detect_direction(inv)
    boundaries = cm.measure_boundaries(inv, direction, log=lambda *_: None)
    decision = cm.decide_gains(boundaries, inv["cols"])
    assert [(s["col_a"], s["col_b"]) for s in decision["steps"]] == [(2, 3)]
    assert decision["reference_segment"] == [0, 1, 2]
    assert decision["corrected_columns"] == [3, 4]
    for c in (3, 4):
        assert np.allclose(decision["gains_rgb"][str(c)], gain, atol=0.01), decision["gains_rgb"][str(c)]
    assert np.allclose(decision["gains_rgb"]["1"], 1.0)
    # the vignetting baseline is the left-edge/right-edge brightness ratio, well below 1
    assert all(0.9 < v < 0.98 for v in decision["vignetting_baseline_ratio"])


def test_uniform_scan_has_no_steps_and_only_hard_links(tmp_path):
    folder = make_grid(tmp_path)
    inv = cm.inventory(folder)
    direction = cm.detect_direction(inv)
    decision = cm.decide_gains(cm.measure_boundaries(inv, direction, log=lambda *_: None), inv["cols"])
    assert decision["steps"] == [] and decision["corrected_columns"] == []
    out = tmp_path / "out"
    manifest = cm.apply(inv, decision, direction, out, source_dir=folder, log=lambda *_: None)
    assert all(t["kind"] == "hard_link_to_original" for t in manifest["tiles"])
    # a hard link shares the inode: the original bytes are never duplicated or altered
    t = manifest["tiles"][0]
    assert os.stat(out / t["filename"]).st_ino == os.stat(folder / t["source_filename"]).st_ino


def test_apply_writes_mirrored_corrected_dataset_with_valid_derived_records(tmp_path):
    gain = (0.94, 0.955, 0.92)
    folder = make_grid(tmp_path, gain_last_cols=gain)
    inv = cm.inventory(folder)
    direction = cm.detect_direction(inv)
    decision = cm.decide_gains(cm.measure_boundaries(inv, direction, log=lambda *_: None), inv["cols"])
    out = tmp_path / "out"
    manifest = cm.apply(inv, decision, direction, out, source_dir=folder, log=lambda *_: None)
    assert manifest["mirrored_columns"] is True
    by_source = {(t["row"], t["source_col"]): t for t in manifest["tiles"]}
    assert by_source[(0, 4)]["col"] == 0 and by_source[(0, 0)]["col"] == 4     # new_col = 4 - old_col
    corrected = [t for t in manifest["tiles"] if t["kind"] == "gain_corrected_copy"]
    assert {t["source_col"] for t in corrected} == {3, 4}
    assert all(t["clipped_fraction"] == 0.0 for t in corrected)
    # corrected pixels equal the gain applied to the source
    t = by_source[(1, 4)]
    src = np.asarray(Image.open(folder / t["source_filename"]).convert("RGB")).astype(np.float32)
    dst = np.asarray(Image.open(out / t["filename"]).convert("RGB")).astype(np.float32)
    expected = np.clip(np.rint(src * np.array(t["gain_rgb"], np.float32)), 0, 255)
    assert np.array_equal(dst, expected)
    # originals untouched
    assert cm._sha256(folder / t["source_filename"]) == t["source_sha256"]
    # derived records: hashes match the new files and pass the stitcher's acquisition check
    from tools.stitch_setup import acquisition_metadata as am
    import stitch_profile as SP
    tiles = SP.scan_grid_tiles(out)
    inventory = SP.inspect_tiles(tiles)
    evidence, warnings = am.inspect_acquisition(out, inventory, "flat-grid")
    assert evidence["present"] and evidence["checksum_verified_selected_tiles"] == len(manifest["tiles"])
    session = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert session["derived_colour_correction"]["corrected_columns"] == [3, 4]
    assert session["position_trusted"] is False
    # the stitch profile is valid and points right / down
    profile = cm.stitch_profile(inv, direction)
    dx_v, dy_v, dx_h, dy_h = profile["nominal_vectors"]
    assert dx_h > 0 and dy_v > 0 and abs(dx_h - 290) <= 2 and abs(dy_v - 160) <= 2


def test_cli_report_only_changes_nothing(tmp_path, capsys):
    folder = make_grid(tmp_path, gain_last_cols=(0.95, 0.95, 0.93))
    before = {p.name: cm._sha256(p) for p in folder.iterdir()}
    code = cm.main(["--data", str(folder), "--report-only", "--max-rows", "4"])
    assert code == 0
    report = json.loads((tmp_path / "scan_colour_match_report.json").read_text(encoding="utf-8"))
    assert report["decision"]["corrected_columns"] == [3, 4]
    assert {p.name: cm._sha256(p) for p in folder.iterdir()} == before
    assert not (tmp_path / "scan_colour_matched").exists()
    assert "steps at [(2, 3)]" in capsys.readouterr().out


def test_cli_builds_dataset_and_stitch_command_without_running(tmp_path):
    folder = make_grid(tmp_path, gain_last_cols=(0.95, 0.95, 0.93))
    out = tmp_path / "derived"
    code = cm.main(["--data", str(folder), "--out", str(out), "--max-rows", "4"])
    assert code == 0
    assert (out / "colour_match_manifest.json").is_file() and (out / "stitch_profile.json").is_file()
    cmd = (out / "stitch_command.txt").read_text(encoding="utf-8")
    assert "run_stitch.py" in cmd and "--no-ai" not in cmd      # Kimi when configured, as the owner requires
    assert not (out / "mosaic.png").exists()
    cmd_offline = cm.stitch_command(out, out / "stitch_profile.json", no_ai=True)
    assert "--no-ai" in cmd_offline


def test_falls_back_to_byte_copies_where_hard_links_are_impossible(tmp_path, monkeypatch):
    folder = make_grid(tmp_path)
    inv = cm.inventory(folder)
    direction = cm.detect_direction(inv)
    decision = cm.decide_gains(cm.measure_boundaries(inv, direction, log=lambda *_: None), inv["cols"])

    def no_links(*_a, **_k):
        raise OSError("Operation not supported")
    monkeypatch.setattr(cm.os, "link", no_links)
    out = tmp_path / "out"
    manifest = cm.apply(inv, decision, direction, out, source_dir=folder, log=lambda *_: None)
    assert all(t["kind"] == "byte_copy_of_original" for t in manifest["tiles"])
    t = manifest["tiles"][0]
    assert cm._sha256(out / t["filename"]) == t["source_sha256"] == t["sha256"]


def test_cli_full_resolution_needs_full_flag(tmp_path, capsys):
    folder = make_grid(tmp_path, gain_last_cols=(0.95, 0.95, 0.93))
    out = tmp_path / "derived"
    assert cm.main(["--data", str(folder), "--out", str(out), "--max-rows", "4", "--scale-div", "2", "--out-scale", "1"]) == 1
    assert "add --full" in capsys.readouterr().out and not out.exists()
    assert cm.main(["--data", str(folder), "--out", str(out), "--max-rows", "4", "--scale-div", "2", "--out-scale", "1", "--full"]) == 0
    cmd = (out / "stitch_command.txt").read_text(encoding="utf-8")
    assert "--full" in cmd and "--workers 4" in cmd


def test_refuses_implausible_gain(tmp_path):
    # the resumed columns are 2.5x darker: the gain to match them (2.5) is outside the plausible range
    folder = make_grid(tmp_path, gain_last_cols=(2.5, 2.5, 2.5))
    inv = cm.inventory(folder)
    direction = cm.detect_direction(inv)
    boundaries = cm.measure_boundaries(inv, direction, log=lambda *_: None)
    with pytest.raises(cm.ColourMatchError, match="Implausible"):
        cm.decide_gains(boundaries, inv["cols"])
