"""Local setup contracts with synthetic images; no models, devices or processes."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shlex
import socket
import subprocess
from types import SimpleNamespace
import zipfile

from PIL import Image
import pytest


SPEC = importlib.util.spec_from_file_location("stitch_setup_service_tested",
                                            Path(__file__).with_name("service.py"))
service = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(service)


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Setup must not start a process or connect to anything")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def png(size=(640, 360), color=(80, 100, 120)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def grid(tmp_path):
    folder = tmp_path / "raw tiles & 50% !"
    folder.mkdir()
    payload = png()
    for row in range(2):
        for col in range(2):
            (folder / f"mosaic_r{row}_c{col}.png").write_bytes(payload)
    return folder


def inspect(folder, **kwargs):
    return service.dispatch("inspect", {"data_dir": str(folder), **kwargs})


def preparation(folder, **kwargs):
    result = inspect(folder)
    return {"data_dir": str(folder), "fingerprint": result["inspection"]["fingerprint"],
            "profile_kind": "rescaled", "confirmations": {
                "same_optics": True, "same_field_of_view": True, "same_stage_steps": True},
            "scale_div": 8, "out_scale": 0.125, "full": False, **kwargs}


def hashes(folder):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.iterdir()}


def test_inspection_reads_all_headers_but_writes_nothing(grid):
    before = hashes(grid)
    result = inspect(grid)
    summary = result["inspection"]
    assert summary["tile_count"] == 4
    assert summary["column_count"] == 2
    assert summary["image_size"] == [640, 360]
    assert summary["layout_mode"] == "flat-grid"
    assert summary["file_mb_min"] == summary["file_mb_max"] == len(png()) / 1_000_000
    assert summary["default_scale_div"] == 8
    assert "input_inventory" not in result
    assert not (grid.parent / "_stitch_results").exists()
    assert hashes(grid) == before


def test_prepare_creates_unique_bound_bundles_without_touching_sources(grid):
    before = hashes(grid)
    body = preparation(grid)
    first = service.dispatch("prepare", body)
    second = service.dispatch("prepare", body)
    assert first["bundle_dir"] != second["bundle_dir"]
    assert first["status"] == "profile_prepared"
    assert first["stitch_executed"] is False
    bundle = Path(first["bundle_dir"])
    assert bundle.parent == grid.parent / "_stitch_results"
    assert {path.name for path in bundle.iterdir()} == {
        "profile.json", "preflight.json", "argv.json", "instructions.txt"}
    assert not (bundle / "work").exists()
    assert not (bundle / "mosaic.png").exists()
    argv = json.loads(Path(first["argv_path"]).read_text())
    assert "--no-ai" in argv and "--full" not in argv and "--layout" not in argv
    assert shlex.split(first["command"]) == argv
    assert argv[argv.index("--data") + 1] == str(grid)
    saved = json.loads(Path(first["preflight_path"]).read_text())
    assert saved["status"] == "profile_prepared" and saved["stitch_executed"] is False
    assert len(saved["input_inventory"]["tiles"]) == 4
    assert saved["inspection"]["fingerprint"] == body["fingerprint"]
    assert hashes(grid) == before


def test_changed_tile_invalidates_inspection_before_any_output(grid):
    body = preparation(grid)
    (grid / "mosaic_r0_c0.png").write_bytes(png(color=(10, 20, 30)))
    with pytest.raises(ValueError, match="changed since inspection"):
        service.dispatch("prepare", body)
    assert not (grid.parent / "_stitch_results").exists()


@pytest.mark.parametrize("confirmations", [None, {}, {"same_optics": "true",
                         "same_field_of_view": True, "same_stage_steps": True}])
def test_rescaling_requires_three_actual_booleans(grid, confirmations):
    with pytest.raises(ValueError, match="Confirm all three"):
        service.dispatch("prepare", preparation(grid, confirmations=confirmations))
    assert not (grid.parent / "_stitch_results").exists()


@pytest.mark.parametrize("settings", [
    {"profile_kind": "historical"},
    {"scale_div": True}, {"scale_div": "8"}, {"scale_div": 7},
    {"full": "false"}, {"out_scale": float("nan")},
    {"out_scale": 1, "full": False},
    {"profile_kind": "custom", "calibration_note": ""},
    {"profile_kind": "custom", "calibration_note": "measured", "nominal_vectors": [0, 500, 500, 0]},
])
def test_invalid_geometry_or_types_never_create_output(grid, settings):
    with pytest.raises(ValueError):
        service.dispatch("prepare", preparation(grid, **settings))
    assert not (grid.parent / "_stitch_results").exists()


def test_custom_profile_persists_measurement_note_and_full_render_choice(grid):
    result = service.dispatch("prepare", preparation(
        grid, profile_kind="custom", calibration_note="Measured on pair 1–2.\nSame objective.",
        nominal_vectors=[-4, 300, 550, 6], out_scale=1, full=True))
    profile = json.loads(Path(result["profile_path"]).read_text())
    assert profile["nominal_vectors"] == [-4, 300, 550, 6]
    assert "Measured on pair 1–2." in profile["geometry_source"]
    preflight = json.loads(Path(result["preflight_path"]).read_text())
    assert preflight["calibration_note"] == "Measured on pair 1–2.\nSame objective."
    assert "--full" in preflight["argv"]


def test_historical_profile_requires_and_accepts_original_dimensions(grid):
    for path in grid.iterdir():
        path.write_bytes(png((3840, 2160)))
    result = service.dispatch("prepare", preparation(grid, profile_kind="historical"))
    assert result["profile"]["image_size"] == [3840, 2160]
    assert result["profile"]["nominal_vectors"] == [-74.6, 1904.56, 3583.58, 153.4]


def test_output_cannot_be_inside_raw_directory_even_through_symlink(grid, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(grid, target_is_directory=True)
    before = hashes(grid)
    for parent in (grid, grid / "results", alias / "results"):
        with pytest.raises(ValueError, match="outside the raw"):
            service.dispatch("prepare", preparation(grid, output_parent=str(parent)))
    assert hashes(grid) == before


@pytest.mark.parametrize("kind", ["folders", "zip"])
def test_deterministic_column_names_are_saved_as_explicit_layout(tmp_path, kind):
    folder = tmp_path / "columns"
    folder.mkdir()
    for column, direction in ((1, "down"), (2, "up")):
        name = ("a_" if column == 1 else "") + f"L{column}_{direction}"
        if kind == "zip":
            with zipfile.ZipFile(folder / (name + ".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
                for index in (1, 2):
                    archive.writestr(f"native/{index:04d}.png", png())
        else:
            (folder / name).mkdir()
            for index in (1, 2):
                (folder / name / f"{index:04d}.png").write_bytes(png())
    summary = inspect(folder)["inspection"]
    assert summary["layout_mode"] == "named-columns" and summary["tile_count"] == 4
    assert summary["file_mb_min"] == len(png()) / 1_000_000
    result = service.dispatch("prepare", preparation(folder))
    layout_path = Path(result["bundle_dir"]) / "layout.json"
    assert json.loads(layout_path.read_text())["major_axis"] == "column"
    assert "--layout" in json.loads(Path(result["argv_path"]).read_text())


def test_row_major_layout_rejected_before_output(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps({**service.DEFAULT_LAYOUT, "major_axis": "row"}))
    with pytest.raises(ValueError, match="Row-major"):
        inspect(raw, layout_path=str(layout))


def test_changed_provided_layout_requires_new_inspection(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("L1_down", "L2_up"):
        (raw / name).mkdir()
        for index in (1, 2):
            (raw / name / f"{index:04d}.png").write_bytes(png())
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps({"layout": service.DEFAULT_LAYOUT}))
    checked = inspect(raw, layout_path=str(layout))
    assert checked["inspection"]["layout_mode"] == "provided-layout"
    body = {**preparation(raw), "layout_path": str(layout),
            "fingerprint": checked["inspection"]["fingerprint"]}
    layout.write_text(json.dumps({"layout": {**service.DEFAULT_LAYOUT,
                                            "reverse_direction_token": "down"}}))
    with pytest.raises(ValueError, match="changed since inspection"):
        service.dispatch("prepare", body)
    assert not (raw.parent / "_stitch_results").exists()


def test_mixed_resolution_and_mixed_sources_fail(grid):
    (grid / "mosaic_r0_c0.png").write_bytes(png((320, 180)))
    with pytest.raises(ValueError, match="Mixed tile resolutions"):
        inspect(grid)
    (grid / "mosaic_r0_c0.png").write_bytes(png())
    (grid / "L1_down").mkdir()
    with pytest.raises(ValueError, match="mixed"):
        inspect(grid)


def test_scanner_review_candidates_and_log_directories_are_explicitly_excluded(grid):
    review = grid / "review_candidates"
    review.mkdir()
    (review / "r0_c0_attempt0001.png").write_bytes(png())
    logs = grid / "history" / "console_logs"
    logs.mkdir(parents=True)
    (logs / "launch.txt").write_text("fixture only")
    result = inspect(grid)
    assert result["inspection"]["tile_count"] == 4
    assert str(review) in result["inspection"]["excluded_paths"]
    assert any(str(review) in warning for warning in result["warnings"])
    assert any(str(grid / "history") in warning for warning in result["warnings"])


def test_unexpected_nested_image_directory_is_not_silently_excluded(grid):
    unknown = grid / "another acquisition"
    unknown.mkdir()
    (unknown / "raw.png").write_bytes(png())
    with pytest.raises(ValueError, match="mixed|Unexpected nested image container"):
        inspect(grid)


def test_shell_arguments_roundtrip_including_quotes_and_metacharacters(monkeypatch):
    argv = ["/path with spaces/python", "run_stitch.py", "--data",
            "/raw O'Brien & 50% ! $HOME $(echo no) `x` \"quoted\"; data"]
    monkeypatch.setattr(service, "os", SimpleNamespace(name="posix"))
    command, label = service._command(argv)
    assert shlex.split(command) == argv and label == "POSIX shell"
    monkeypatch.setattr(service, "os", SimpleNamespace(name="nt"))
    command, label = service._command(argv)
    assert label == "PowerShell" and command.startswith("& ")
    # Parse the deliberately restricted PowerShell grammar: only single-quoted
    # literal arguments, separated by spaces, following the call operator.
    token = re.compile(r"'((?:[^']|'')*)'(?: |$)")
    cursor, parsed = 2, []
    while cursor < len(command):
        match = token.match(command, cursor)
        assert match is not None, command[cursor:]
        parsed.append(match[1].replace("''", "'"))
        cursor = match.end()
    assert parsed == argv


@pytest.mark.parametrize("action,body", [("inspect", []), ("inspect", {"data_dir": 4}),
                                         ("run", {}), ("prepare", {"fingerprint": "bad"})])
def test_invalid_requests_have_actionable_errors(action, body):
    with pytest.raises(ValueError):
        service.dispatch(action, body)
