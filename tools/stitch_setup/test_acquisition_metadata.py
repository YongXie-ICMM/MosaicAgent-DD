"""Verify scanner provenance against real synthetic PNG headers and byte hashes."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.stitch_setup import service


@pytest.fixture
def acquisition(tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    captures = []
    for row in range(2):
        for col in range(2):
            path = folder / f"mosaic_r{row}_c{col}.png"
            Image.new("RGB", (1920, 1080), (50 + row, 60 + col, 90)).save(path)
            payload = path.read_bytes()
            captures.append({"event": "capture_success", "session_id": "synthetic-test",
                             "filename": path.name, "row": row, "col": col,
                             "saved_image_resolution": [1920, 1080], "image_bytes": len(payload),
                             "image_sha256": hashlib.sha256(payload).hexdigest()})
    session = {"schema_version": 2, "session_id": "synthetic-test", "status": "completed",
               "camera": {"actual_resolution": [1920, 1080], "requested_resolution": [1920, 1080],
                          "backend": "opencv", "capture_api": "DSHOW", "resolution_source": "received_frame"},
               "planned_positions": 4, "photos_saved": 4,
               "scan_config": {"nx": 2, "ny": 2, "dx_steps": 143, "dy_steps": 76,
                               "order": "Y_first", "calibration_status": "legacy defaults",
                               "acquisition_contract": {"schema_version": 1, "expected_image_size": [1920, 1080],
                                 "actual_image_size": [1920, 1080], "verification_status": "verified_received_frame",
                                 "raw_images_resized": False, "calibration_status": "unverified_for_current_mode"}}}
    save(folder, session, captures)
    return folder, session, captures


def save(folder, session, captures):
    (folder / "session.json").write_text(json.dumps(session))
    (folder / "events.jsonl").write_text(''.join(json.dumps(e) + '\n' for e in captures))


def inspect(folder):
    return service.dispatch("inspect", {"data_dir": str(folder)})


def prepare_body(folder):
    result = inspect(folder)
    return {"data_dir": str(folder), "fingerprint": result["inspection"]["fingerprint"],
            "profile_kind": "custom", "nominal_vectors": [0, 900, 1750, 0],
            "calibration_note": "Synthetic fixture only, measured original-pixel displacement.",
            "scale_div": 8, "out_scale": 0.125}


def test_actual_1080_contract_is_preserved_without_inventing_calibration(acquisition):
    folder, session, captures = acquisition
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    result = inspect(folder)
    record = result["inspection"]["acquisition"]
    assert record["expected_image_size"] == record["actual_image_size"] == [1920, 1080]
    assert record["grid"]["selection_scope"] == "full_recorded_grid"
    assert record["steps"] == {"dx_steps": 143, "dy_steps": 76}
    assert record["matching_selected_capture_records"] == 4
    assert record["calibration_status"] == "unverified_for_current_mode"
    assert record["geometry_inferred"] is False
    assert record["physical_coverage_verified"] is False
    assert len(record["source_files"]) == 2
    assert "profile" not in result
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


def test_prepared_bundle_contains_metadata_bound_to_inspected_bytes(acquisition):
    folder, _, _ = acquisition
    result = service.dispatch("prepare", prepare_body(folder))
    bundle = Path(result["bundle_dir"])
    record = json.loads((bundle / "acquisition_metadata.json").read_text())
    preflight = json.loads((bundle / "preflight.json").read_text())
    assert record == preflight["inspection"]["acquisition"]
    assert record["source_files"][0]["sha256"] == hashlib.sha256((folder / "session.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("field", ["expected_image_size", "actual_image_size"])
def test_contract_dimensions_cannot_disagree_with_originals(acquisition, field):
    folder, session, captures = acquisition
    session["scan_config"]["acquisition_contract"][field] = [3840, 2160]
    save(folder, session, captures)
    with pytest.raises(ValueError, match="does not match"):
        inspect(folder)


@pytest.mark.parametrize("field", ["actual_resolution", "requested_resolution"])
def test_camera_dimensions_cannot_disagree_with_originals(acquisition, field):
    folder, session, captures = acquisition
    session["camera"][field] = [3840, 2160]
    save(folder, session, captures)
    with pytest.raises(ValueError, match="does not match"):
        inspect(folder)


@pytest.mark.parametrize("field,value,reason", [
    ("image_sha256", "0" * 64, "SHA-256 mismatch"),
    ("image_bytes", 1, "byte size differs"),
    ("saved_image_resolution", [3840, 2160], "dimensions differ"),
    ("session_id", "another-run", "different session"),
    ("row", 7, "coordinates conflict"),
])
def test_recorded_originals_are_checked_not_only_the_directory_name(acquisition, field, value, reason):
    folder, session, captures = acquisition
    captures[0][field] = value
    save(folder, session, captures)
    with pytest.raises(ValueError, match=reason):
        inspect(folder)


def test_completed_954_point_record_with_four_tiles_is_a_subset(acquisition):
    folder, session, captures = acquisition
    session["scan_config"].update(nx=18, ny=53)
    session.update(planned_positions=954, photos_saved=954)
    save(folder, session, captures)
    result = inspect(folder)
    assert result["inspection"]["acquisition"]["grid"] == {
        "nx": 18, "ny": 53, "planned_tiles": 954, "selected_tiles": 4, "selection_scope": "subset"}
    assert any("not the complete scan" in message for message in result["warnings"])


def test_unfinished_scan_is_not_promoted_to_completed_when_current_grid_is_full(acquisition):
    folder, session, captures = acquisition
    session["status"] = "running"
    save(folder, session, captures)
    result = inspect(folder)
    assert result["inspection"]["acquisition"]["session_status"] == "running"
    assert any("not marked completed" in message for message in result["warnings"])


@pytest.mark.parametrize("which", ["session", "events"])
def test_metadata_changes_require_reinspection_before_creating_bundle(acquisition, which):
    folder, session, captures = acquisition
    body = prepare_body(folder)
    if which == "session":
        session["scan_config"]["objective"] = "40X"
    else:
        captures.append({"event": "review_note", "note": "new information"})
    save(folder, session, captures)
    with pytest.raises(ValueError, match="changed since inspection"):
        service.dispatch("prepare", body)
    assert not (folder.parent / "_stitch_results").exists()


def test_legacy_session_with_driver_default_remains_inspectable_but_uncalibrated(acquisition):
    folder, session, captures = acquisition
    session["camera"]["requested_resolution"] = None
    session["scan_config"].pop("acquisition_contract")
    save(folder, session, captures)
    result = inspect(folder)
    record = result["inspection"]["acquisition"]
    assert record["expected_image_size"] is None
    assert record["calibration_status"] == "unverified_for_current_mode"
    assert any("calibration is still required" in w for w in result["warnings"])


def test_no_events_is_explicitly_weaker_evidence(acquisition):
    folder, _, _ = acquisition
    (folder / "events.jsonl").unlink()
    result = inspect(folder)
    assert result["inspection"]["acquisition"]["matching_selected_capture_records"] is None
    assert any("events.jsonl is absent" in w for w in result["warnings"])


def test_incomplete_or_foreign_journal_is_not_silently_ignored(acquisition):
    folder, session, captures = acquisition
    (folder / "events.jsonl").write_text('{"event":')
    with pytest.raises(ValueError, match="Cannot read"):
        inspect(folder)
    save(folder, session, captures[:-1])
    with pytest.raises(ValueError, match="lack capture_success"):
        inspect(folder)


def test_missing_checksum_is_reported_as_weaker_evidence(acquisition):
    folder, session, captures = acquisition
    captures[0].pop("image_sha256")
    save(folder, session, captures)
    result = inspect(folder)
    assert result["inspection"]["acquisition"]["checksum_verified_selected_tiles"] == 3
    assert any("no recorded acquisition checksum" in w for w in result["warnings"])


def test_selected_grid_cannot_exceed_recorded_saved_count(acquisition):
    folder, session, captures = acquisition
    session["photos_saved"] = 3
    save(folder, session, captures)
    with pytest.raises(ValueError, match="photos_saved"):
        inspect(folder)
