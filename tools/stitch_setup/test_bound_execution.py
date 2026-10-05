"""Prepared commands must revalidate inputs before work/cache/model creation."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

import run_stitch as RS
import stitch_profile as SP
from tools.stitch_setup import service
from tools.stitch_setup.acquisition_metadata import inspect_acquisition
from tools.stitch_setup.test_acquisition_metadata import acquisition, save, prepare_body


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Bound execution checks must happen before any model client")
    monkeypatch.setattr(RS.KA, "KimiClient", forbidden)


def prepared(acquisition):
    folder, _, _ = acquisition
    result = service.dispatch("prepare", prepare_body(folder))
    argv = json.loads(Path(result["argv_path"]).read_text())
    return result, argv[1:]


def invoke(monkeypatch, argv, *, preflight=True):
    monkeypatch.setattr(sys, "argv", argv + (["--preflight-only"] if preflight else []))
    return RS.main()


def test_prepared_command_passes_runtime_preflight_without_outputs(acquisition, monkeypatch):
    result, argv = prepared(acquisition)
    assert argv[argv.index("--input-preflight") + 1] == result["preflight_path"]
    assert invoke(monkeypatch, argv) is None
    assert not (Path(result["bundle_dir"]) / "work").exists()
    assert not (Path(result["bundle_dir"]) / "mosaic.png").exists()


@pytest.mark.parametrize("change", ["session", "raw", "profile", "scale", "remove_records"])
def test_changes_after_preparation_fail_before_creating_work(acquisition, monkeypatch, change):
    folder, session, captures = acquisition
    result, argv = prepared(acquisition)
    if change == "session":
        session["scan_config"]["objective"] = "changed-objective"
        save(folder, session, captures)
    elif change == "raw":
        # Valid new image and internally consistent updated acquisition journal;
        # still forbidden because the command was bound to the earlier inputs.
        from PIL import Image
        path = folder / captures[0]["filename"]
        Image.new("RGB", (1920, 1080), (100, 120, 140)).save(path)
        captures[0]["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        captures[0]["image_bytes"] = path.stat().st_size
        save(folder, session, captures)
    elif change == "profile":
        path = Path(result["profile_path"])
        value = json.loads(path.read_text())
        value["nominal_vectors"][0] = -1
        path.write_text(json.dumps(value))
    elif change == "scale":
        argv[argv.index("--out-scale") + 1] = "0.0625"
    else:
        (folder / "session.json").unlink()
        (folder / "events.jsonl").unlink()
    # Run the ordinary execution entry, not only --preflight-only. It must stop
    # before creating work even when later stages would execute.
    with pytest.raises(SystemExit):
        invoke(monkeypatch, argv, preflight=False)
    assert not (Path(result["bundle_dir"]) / "work").exists()
    assert not (Path(result["bundle_dir"]) / "mosaic.png").exists()


def test_direct_cli_checks_session_even_without_a_prepared_bundle(acquisition, tmp_path, monkeypatch):
    folder, session, captures = acquisition
    profile = {"schema_version": 1, "name": "test", "image_size": [1920, 1080],
               "nominal_vectors": [0, 900, 1750, 0], "geometry_source": "measured-test"}
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    session["scan_config"]["acquisition_contract"]["expected_image_size"] = [3840, 2160]
    save(folder, session, captures)
    work = tmp_path / "work"
    with pytest.raises(SystemExit):
        invoke(monkeypatch, ["run_stitch.py", "--data", str(folder), "--work", str(work),
                            "--stitch-profile", str(profile_path), "--no-ai"])
    assert not work.exists()


def test_changed_metadata_cannot_reuse_previous_work_binding(acquisition, tmp_path, monkeypatch):
    folder, session, captures = acquisition
    result, argv = prepared(acquisition)
    inventory = SP.inspect_tiles(SP.scan_grid_tiles(folder))
    profile = SP.load_profile(result["profile_path"])
    resolved = SP.resolve_profile(profile, inventory)
    metadata, _ = inspect_acquisition(folder, inventory, "flat-grid")
    work = Path(result["bundle_dir"]) / "work"
    work.mkdir()
    original_state = json.dumps({"run_binding": SP.make_run_binding(resolved, inventory, metadata)})
    (work / "state.json").write_text(original_state)
    session["scan_config"]["objective"] = "changed-but-still-matching-size"
    save(folder, session, captures)
    index = argv.index("--input-preflight")
    del argv[index:index + 2]
    argv.append("--force")
    with pytest.raises(SystemExit):
        invoke(monkeypatch, argv, preflight=False)
    assert (work / "state.json").read_text() == original_state
    assert not (work / "cache").exists()


def test_absent_metadata_preserves_previous_binding_structure():
    inventory = {"fingerprint": "test"}
    resolved = {"profile": {}, "scale_div": 8, "out_scale": 0.125, "full": False}
    old = SP.make_run_binding(resolved, inventory)
    assert SP.make_run_binding(resolved, inventory, {"present": False}) == old
    assert "acquisition_fingerprint" not in old


@pytest.mark.parametrize("flag", ["--plan", "--auto-layout"])
def test_bound_execution_cannot_be_reinterpreted_as_layout_planning(acquisition, monkeypatch, flag):
    result, argv = prepared(acquisition)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, argv + [flag], preflight=False)
    assert not (Path(result["bundle_dir"]) / "work").exists()
