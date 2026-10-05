"""Offline safety/geometry regressions. No camera, stage, model, or network calls."""
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tiles as T
import register as R
import blend as B
import run_stitch as RS
import stitch_profile as SP


def profile(size=(256, 192)):
    w, h = size
    return {"schema_version": 1, "name": "measured-test", "image_size": list(size),
            "nominal_vectors": [-w / 64, h * .75, w * .75, h / 48],
            "geometry_source": "measured-adjacent-pairs", "scale_div": 2, "out_scale": .5}


def png(size=(256, 192), level=6, value=123):
    out = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(out, format="PNG", compress_level=level)
    return out.getvalue()


def tile(folder, name="0001.png", size=(256, 192), *, z=False, value=123):
    if z:
        path = folder / "a_L1_down.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(name, png(size, value=value))
    else:
        path = folder
        path.mkdir(exist_ok=True)
        (path / name).write_bytes(png(size, value=value))
    return T.Tile(1, 0, "down", 0, 1., name, str(path), name, 0)


def grid(folder, size=(256, 192), n=2):
    folder.mkdir()
    for c in range(n):
        for r in range(n):
            (folder / f"mosaic_r{r}_c{c}.png").write_bytes(png(size))
    return SP.scan_grid_tiles(folder)


def invoke(monkeypatch, data, work, *extra):
    monkeypatch.setattr(sys, "argv", ["run_stitch.py", "--data", str(data), "--work", str(work), *extra])
    return RS.main()


def forbid_ai(*args, **kwargs):
    raise AssertionError("No Kimi client or inference is permitted here")


def test_historical_default_retains_exact_calibration(tmp_path):
    ts = [tile(tmp_path, size=SP.HISTORICAL_SIZE, z=True)]
    inv = SP.inspect_tiles(ts)
    resolved = SP.resolve_profile(None, inv)
    assert resolved["profile"]["nominal_vectors"] == list(SP.HISTORICAL_VECTORS)
    assert resolved["scale_div"] == 8 and resolved["out_scale"] == 1 / 8
    assert resolved["warnings"]
    with SP.geometry_context(resolved["profile"]):
        assert R.NOMINAL_FULL == pytest.approx((-74.6, 1904.56, 3583.58, 153.4))
        assert (T.TILE_W, R.TILE_W, B.TILE_W) == (3840,) * 3


def test_nonhistorical_default_and_profile_mismatch_rejected(tmp_path):
    inv = SP.inspect_tiles([tile(tmp_path)])
    with pytest.raises(SP.ProfileError, match="Provide --stitch-profile"):
        SP.resolve_profile(None, inv)
    with pytest.raises(SP.ProfileError, match="does not match"):
        SP.resolve_profile(profile((320, 240)), inv)


def test_rescale_requires_three_explicit_facts():
    for missing in SP.CONFIRMATIONS:
        flags = dict.fromkeys(SP.CONFIRMATIONS, True)
        flags[missing] = False
        with pytest.raises(SP.ProfileError, match="explicit confirmation"):
            SP.rescaled_historical_profile((1920, 1080), **flags)
    p = SP.rescaled_historical_profile((1920, 1080), **dict.fromkeys(SP.CONFIRMATIONS, True))
    assert p["nominal_vectors"] == pytest.approx(np.array(SP.HISTORICAL_VECTORS) / 2)
    p.pop("confirmations")
    with pytest.raises(SP.ProfileError, match="confirmations"):
        SP.validate_profile(p)


@pytest.mark.parametrize("vectors", [[0, -20, 30, 0], [0, 192, 192, 0],
                                    [float("nan"), 144, 192, 0], [0, 144, float("inf"), 0],
                                    [255, 1, 1, 191]])
def test_invalid_geometry_rejected(vectors):
    p = profile()
    p["nominal_vectors"] = vectors
    with pytest.raises(SP.ProfileError):
        SP.validate_profile(p)


def test_all_images_checked_and_mixed_or_unreadable_rejected(tmp_path):
    first = tile(tmp_path, "0001.png")
    second = tile(tmp_path, "0002.png", (320, 240))
    with pytest.raises(SP.ProfileError, match="Mixed tile resolutions"):
        SP.inspect_tiles([first, second])
    (tmp_path / "0002.png").write_bytes(b"not an image")
    with pytest.raises(SP.ProfileError, match="Unreadable tile image.*0002"):
        SP.inspect_tiles([first, second])


def test_zip_and_directory_evidence_and_compression_do_not_set_geometry(tmp_path):
    directory = tmp_path / "dir"
    a = tile(directory)
    (directory / a.inner).write_bytes(png(level=0))
    inv_a = SP.inspect_tiles([a])
    (directory / a.inner).write_bytes(png(level=9))
    inv_b = SP.inspect_tiles([a])
    assert inv_a["tiles"][0]["source"]["size"] > 20 * inv_b["tiles"][0]["source"]["size"]
    assert SP.resolve_profile(profile(), inv_a) == SP.resolve_profile(profile(), inv_b)
    assert inv_a["fingerprint"] != inv_b["fingerprint"]
    z = tile(tmp_path, z=True)
    inv_z = SP.inspect_tiles([z])
    assert inv_z["tiles"][0]["crc"]
    assert inv_z["image_size"] == inv_a["image_size"]
    assert "sha256" in inv_a["tiles"][0]


def test_work_binding_rejects_changed_inputs_geometry_and_legacy(tmp_path):
    raw = tmp_path / "raw"
    ts = [tile(raw)]
    inv = SP.inspect_tiles(ts)
    resolved = SP.resolve_profile(profile(), inv)
    binding = SP.make_run_binding(resolved, inv)
    work = tmp_path / "work"
    work.mkdir()
    (work / "state.json").write_text(json.dumps({"run_binding": binding}))
    SP.check_work_binding(work, binding)
    variants = []
    for key, value in (("scale_div", 1), ("out_scale", .25), ("full", True)):
        variants.append({**resolved, key: value})
    p = profile()
    p["nominal_vectors"][0] -= 1
    variants.append({**resolved, "profile": p})
    for changed in variants:
        with pytest.raises(SP.ProfileError, match="new --work"):
            SP.check_work_binding(work, SP.make_run_binding(changed, inv))
    (raw / ts[0].inner).write_bytes(png(value=124))
    changed_inv = SP.inspect_tiles(ts)
    with pytest.raises(SP.ProfileError, match="inventory changed"):
        SP.check_work_binding(work, SP.make_run_binding(resolved, changed_inv))
    (work / "state.json").write_text("{}")
    with pytest.raises(SP.ProfileError, match="legacy/unbound"):
        SP.check_work_binding(work, binding)


def test_adding_default_work_does_not_change_raw_directory_fingerprint(tmp_path):
    ts = [tile(tmp_path)]
    before = SP.inspect_tiles(ts)
    (tmp_path / "_stitch_work").mkdir()
    (tmp_path / "_stitch_work" / "state.json").write_text("{}")
    assert SP.inspect_tiles(ts)["fingerprint"] == before["fingerprint"]


def test_geometry_applies_to_every_consumer_and_restores_on_error():
    prior = [(m.TILE_W, m.TILE_H) for m in (T, R, B)]
    nominal = R.NOMINAL_FULL
    with pytest.raises(RuntimeError, match="synthetic"):
        with SP.geometry_context(profile()):
            assert [(m.TILE_W, m.TILE_H) for m in (T, R, B)] == [(256, 192)] * 3
            assert R.NOMINAL_FULL == tuple(profile()["nominal_vectors"])
            assert RS.T.TILE_W == 256  # rescue and duplicates use the same module
            assert T.NOMINAL_PITCH_Y == 144
            raise RuntimeError("synthetic")
    assert [(m.TILE_W, m.TILE_H) for m in (T, R, B)] == prior
    assert R.NOMINAL_FULL == nominal


def test_flat_grid_has_physical_order_and_missing_coordinates_are_errors(tmp_path):
    raw = tmp_path / "raw"
    ts = grid(raw)
    assert [(t.col_idx, t.nominal_row) for t in ts] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(t.direction == "down" for t in ts)
    (raw / "mosaic_r1_c1.png").unlink()
    with pytest.raises(SP.ProfileError, match="missing coordinates"):
        SP.scan_grid_tiles(raw)


def test_flat_grid_rejects_unknown_images_and_duplicate_coordinate(tmp_path):
    raw = tmp_path / "raw"
    grid(raw)
    (raw / "mosaic_r00_c00.png").write_bytes(png())
    with pytest.raises(SP.ProfileError, match="Duplicate grid coordinate"):
        SP.scan_grid_tiles(raw)
    (raw / "mosaic_r00_c00.png").unlink()
    (raw / "preview.png").write_bytes(png())
    with pytest.raises(SP.ProfileError, match="Unrecognized images"):
        SP.scan_grid_tiles(raw)


def test_preflight_is_read_only_and_never_initializes_ai(tmp_path, monkeypatch, capsys):
    raw, work = tmp_path / "raw", tmp_path / "work"
    grid(raw)
    pf = tmp_path / "profile.json"
    pf.write_text(json.dumps(profile()))
    monkeypatch.setattr(RS.KA, "KimiClient", forbid_ai)
    monkeypatch.setattr(RS.T, "build_cache", forbid_ai)
    report = tmp_path / "report.json"
    invoke(monkeypatch, raw, work, "--stitch-profile", str(pf), "--preflight-only", "--no-ai",
           "--preflight-report", str(report))
    assert not work.exists()
    saved = json.loads(report.read_text())
    assert saved["ok"] and saved["input_evidence"]["tile_count"] == 4
    assert saved["layout_source"] == "flat-physical-grid"
    assert "Preflight:" in capsys.readouterr().out


def test_early_failure_before_ai_cache_or_work_creation(tmp_path, monkeypatch):
    raw, work = tmp_path / "raw", tmp_path / "work"
    grid(raw)
    monkeypatch.setattr(RS.KA, "KimiClient", forbid_ai)
    monkeypatch.setattr(RS.T, "build_cache", forbid_ai)
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, raw, work)
    assert exc.value.code == 2
    assert not work.exists()


def test_cli_requires_no_ai_for_preflight(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, tmp_path, tmp_path / "work", "--preflight-only")
    assert exc.value.code == 2


def test_force_cannot_reuse_unbound_cache(tmp_path, monkeypatch):
    raw, work = tmp_path / "raw", tmp_path / "work"
    grid(raw)
    (work / "cache").mkdir(parents=True)
    artifact = work / "cache" / "sentinel.bin"
    artifact.write_bytes(b"untouched")
    pf = tmp_path / "profile.json"
    pf.write_text(json.dumps(profile()))
    monkeypatch.setattr(RS.KA, "KimiClient", forbid_ai)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, raw, work, "--stitch-profile", str(pf), "--force", "--no-ai")
    assert artifact.read_bytes() == b"untouched"
    assert not (work / "state.json").exists()


def test_cli_records_binding_and_scopes_geometry_for_pipeline(tmp_path, monkeypatch):
    raw, work = tmp_path / "raw", tmp_path / "work"
    grid(raw)
    pf = tmp_path / "profile.json"
    pf.write_text(json.dumps(profile()))
    monkeypatch.setattr(RS.KA, "KimiClient", forbid_ai)
    calls = []
    def capture(args, work, cache_dir, ts, pool, client, state, started):
        calls.append((T.TILE_W, R.TILE_H, B.TILE_W, R.NOMINAL_FULL))
        assert pool is None and client is None
        assert state.d["preflight"]["input_evidence"]["image_size"] == [256, 192]
    monkeypatch.setattr(RS, "_run_pipeline", capture)
    for _ in range(2):
        invoke(monkeypatch, raw, work, "--stitch-profile", str(pf), "--no-ai")
    assert len(calls) == 2 and calls[0][:3] == (256, 192, 256)
    assert T.TILE_W == 3840
    assert (work / "preflight.json").exists()
    assert json.loads((work / "state.json").read_text())["run_binding"]["profile"] == SP.validate_profile(profile())


def test_scale_validation_prevents_coordinate_distortion_and_upsampling():
    p = profile()
    with pytest.raises(SP.ProfileError, match="divide both"):
        SP.resolve_profile(p, {"image_size": [256, 192]}, scale_div=3)
    with pytest.raises(SP.ProfileError, match="exceeds cached"):
        SP.resolve_profile(p, {"image_size": [256, 192]}, out_scale=.75)
    assert SP.resolve_profile(p, {"image_size": [256, 192]}, out_scale=1, full=True)["out_scale"] == 1


def test_historical_name_scan_does_not_silently_omit_other_images(tmp_path):
    tile(tmp_path, size=SP.HISTORICAL_SIZE, z=True)
    other = tmp_path / "another-column"
    tile(other)
    with pytest.raises(SP.ProfileError, match="only part of the image inventory"):
        RS._preflight_scan(str(tmp_path), tmp_path / "_work")


def test_row_major_layout_is_rejected_before_source_scan(tmp_path):
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps({"major_axis": "row"}))
    with pytest.raises(SP.ProfileError, match="column-major"):
        RS._preflight_scan(str(tmp_path), tmp_path / "_work", str(layout))


def test_mixed_pixel_modes_and_16bit_images_fail_explicitly(tmp_path):
    a = tile(tmp_path, "0001.png")
    b = tile(tmp_path, "0002.png")
    Image.new("L", (256, 192), 123).save(tmp_path / b.inner)
    with pytest.raises(SP.ProfileError, match="Mixed tile pixel modes"):
        SP.inspect_tiles([a, b])
    Image.fromarray(np.full((192, 256), 32000, dtype=np.uint16)).save(tmp_path / b.inner)
    with pytest.raises(SP.ProfileError, match="Unsupported tile pixel mode"):
        SP.inspect_tiles([b])


def test_cli_wrong_profile_and_changed_profile_fail_without_mutation(tmp_path, monkeypatch, capsys):
    raw, work = tmp_path / "raw", tmp_path / "work"
    grid(raw, size=(320, 180))
    pf = tmp_path / "profile.json"
    pf.write_text(json.dumps(profile((640, 360))))
    monkeypatch.setattr(RS.KA, "KimiClient", forbid_ai)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, raw, work, "--stitch-profile", str(pf), "--no-ai")
    assert "does not match actual tile headers [320, 180]" in capsys.readouterr().out
    assert not work.exists()
    pf.write_text(json.dumps(profile((320, 180))))
    monkeypatch.setattr(RS, "_run_pipeline", lambda *a: None)
    invoke(monkeypatch, raw, work, "--stitch-profile", str(pf), "--no-ai")
    original = (work / "state.json").read_bytes()
    changed = profile((320, 180))
    changed["nominal_vectors"][0] -= 1
    pf.write_text(json.dumps(changed))
    with pytest.raises(SystemExit):
        invoke(monkeypatch, raw, work, "--stitch-profile", str(pf), "--no-ai", "--force")
    assert (work / "state.json").read_bytes() == original


def test_historical_rescale_rejects_changed_aspect_ratio_in_both_entrypoints():
    confirmations = dict.fromkeys(SP.CONFIRMATIONS, True)
    with pytest.raises(SP.ProfileError, match="same aspect ratio"):
        SP.rescaled_historical_profile((1920, 1200), **confirmations)
    p = SP.rescaled_historical_profile((1920, 1080), **confirmations)
    p["image_size"][1] = 1200
    with pytest.raises(SP.ProfileError, match="same aspect ratio"):
        SP.validate_profile(p)


def test_flat_grid_excludes_documented_review_candidates_but_rejects_other_images(tmp_path):
    raw = tmp_path / "raw"
    grid(raw)
    review = raw / "review_candidates"
    tile(review)
    logs = raw / "history" / "console_logs"
    logs.mkdir(parents=True)
    (logs / "scan.log").write_text("log only")
    assert len(SP.scan_grid_tiles(raw)) == 4
    unknown = raw / "another_scan"
    tile(unknown)
    with pytest.raises(SP.ProfileError, match="Unexpected nested image"):
        SP.scan_grid_tiles(raw)
    assert len(SP.scan_grid_tiles(raw, ignored_paths=(unknown,))) == 4
