"""Synthetic/offline integration checks for ordinary stitching's colour policy.

These assert execution boundaries and provenance, rather than retesting the gain
solver's arithmetic. No model, instrument, or research image is used.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_stitch as RS
import stitch_profile as SP
from tools.stitch_setup.acquisition_metadata import inspect_acquisition


@pytest.fixture
def flat_grid(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(2031)
    canvas = rng.integers(40, 190, (112, 160, 3), dtype=np.uint8)
    for col in range(2):
        for row in range(2):
            tile = canvas[row * 36:row * 36 + 64, col * 48:col * 48 + 96]
            Image.fromarray(tile).save(raw / f"mosaic_r{row}_c{col}.png")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "schema_version": 1, "name": "synthetic-colour-integration",
        "image_size": [96, 64], "nominal_vectors": [0, 36, 48, 0],
        "geometry_source": "synthetic-only", "scale_div": 1,
    }), encoding="utf-8")
    return raw, profile


def invoke(monkeypatch, raw, profile, work, *extra):
    monkeypatch.setattr(sys, "argv", [
        "run_stitch.py", "--data", str(raw), "--stitch-profile", str(profile),
        "--work", str(work), "--out", str(work.parent / "mosaic.png"),
        "--no-ai", *extra,
    ])
    return RS.main()


def sources(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.glob("*.png")}


def test_ordinary_cli_selects_auto_without_a_student_flag(flat_grid, tmp_path, monkeypatch):
    raw, profile = flat_grid
    original = sources(raw)
    seen = []

    def capture(args, work, cache_dir, tiles, pool, client, state, started):
        seen.append(args.colour_correction)
        assert pool is None and client is None
        assert state.d["preflight"]["resolved"]["colour_correction"] == "auto"

    monkeypatch.setattr(RS, "_run_pipeline", capture)
    invoke(monkeypatch, raw, profile, tmp_path / "work")
    assert seen == ["auto"]
    assert sources(raw) == original


def test_auto_preflight_only_does_not_compute_or_create_outputs(flat_grid, tmp_path, monkeypatch):
    raw, profile = flat_grid
    original = sources(raw)
    work, report = tmp_path / "never_created", tmp_path / "preflight_result.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only preflight reached processing or a model")

    monkeypatch.setattr(RS, "_run_pipeline", forbidden)
    monkeypatch.setattr(RS, "_ai_pool", forbidden)
    invoke(monkeypatch, raw, profile, work, "--preflight-only",
           "--preflight-report", str(report))
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["resolved"]["colour_correction"] == "auto"
    assert not work.exists()
    assert not (tmp_path / "mosaic.png").exists()
    assert sources(raw) == original


@pytest.mark.parametrize("force", [False, True])
def test_auto_and_explicit_off_cannot_share_cached_work(flat_grid, tmp_path, monkeypatch, force):
    raw, profile = flat_grid
    work = tmp_path / "bound"
    seen = []
    monkeypatch.setattr(RS, "_run_pipeline", lambda args, *a: seen.append(args.colour_correction))
    invoke(monkeypatch, raw, profile, work)
    before = (work / "state.json").read_bytes()
    before_preflight = (work / "preflight.json").read_bytes()
    flags = ["--colour-correction", "off"] + (["--force"] if force else [])
    with pytest.raises(SystemExit):
        invoke(monkeypatch, raw, profile, work, *flags)
    assert (work / "state.json").read_bytes() == before
    assert (work / "preflight.json").read_bytes() == before_preflight
    assert seen == ["auto"]
    invoke(monkeypatch, raw, profile, tmp_path / "explicit_off", "--colour-correction", "off")
    assert seen == ["auto", "off"]
    auto_binding = json.loads(before)["run_binding"]
    off_binding = json.loads((tmp_path / "explicit_off/state.json").read_text())["run_binding"]
    assert auto_binding["fingerprint"] != off_binding["fingerprint"]


def test_default_auto_rejects_work_created_before_colour_policy(flat_grid, tmp_path, monkeypatch):
    raw, profile = flat_grid
    inventory = SP.inspect_tiles(SP.scan_grid_tiles(raw))
    resolved = SP.resolve_profile(SP.load_profile(profile), inventory)
    resolved["grid_policy"] = "preserve"
    metadata, _ = inspect_acquisition(raw, inventory, "flat-grid")
    old = {"run_binding": SP.make_run_binding(resolved, inventory, metadata),
           "summary": {"tiles_used": 4}, "_done": ["render", "summary"]}
    work = tmp_path / "unlabelled"
    work.mkdir()
    state = work / "state.json"
    state.write_text(json.dumps(old), encoding="utf-8")
    before = state.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unlabelled cache reached processing")

    monkeypatch.setattr(RS, "_run_pipeline", forbidden)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, raw, profile, work, "--force")
    assert state.read_bytes() == before
    assert not (work / "cache").exists()


def corrected_manifest(folder):
    """Write the public colour_match_grid derived-dataset provenance format."""
    records = []
    for col in range(2):
        for row in range(2):
            path = folder / f"mosaic_r{row}_c{col}.png"
            records.append({
                "row": row, "source_col": col, "col": col,
                "filename": path.name, "kind": "hard_link_to_original",
                "gain_rgb": None, "flat_field": False, "clipped_fraction": None,
                "source_filename": path.name,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            })
    manifest = {"schema_version": 2, "kind": "colour_matched_grid",
                "built_at_utc": "2026-10-31T00:00:00+00:00",
                "source_dir": "synthetic-parent", "mirrored_columns": False,
                "column_mapping": "new_col = index(old_col)",
                "direction": {}, "flat_field": None,
                "gains": {"mode": "tile", "tiles_corrected": 0}, "tiles": records}
    path = folder / "colour_match_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_auto_recognizes_hash_bound_previously_corrected_source(flat_grid, tmp_path, monkeypatch):
    raw, profile = flat_grid
    path, _ = corrected_manifest(raw)
    original = sources(raw)
    seen = []

    def capture(args, work, cache_dir, tiles, pool, client, state, started):
        policy = state.d["preflight"]["resolved"]["colour_input"]
        seen.append(policy)
        assert policy["state"] == "already_corrected"
        assert policy["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(RS, "_run_pipeline", capture)
    invoke(monkeypatch, raw, profile, tmp_path / "work")
    assert len(seen) == 1
    assert sources(raw) == original


@pytest.mark.parametrize("damage", ["changed_tile", "missing_record", "duplicate_record", "invalid_json"])
def test_bad_colour_provenance_stops_before_work_or_model(flat_grid, tmp_path, monkeypatch, damage):
    raw, profile = flat_grid
    path, manifest = corrected_manifest(raw)
    if damage == "changed_tile":
        Image.new("RGB", (96, 64), (32, 64, 96)).save(raw / "mosaic_r0_c0.png")
    elif damage == "missing_record":
        manifest["tiles"].pop()
        path.write_text(json.dumps(manifest), encoding="utf-8")
    elif damage == "duplicate_record":
        manifest["tiles"][-1] = dict(manifest["tiles"][0])
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        path.write_text("{broken", encoding="utf-8")
    work = tmp_path / "never_created"

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid prior colour provenance reached execution")

    monkeypatch.setattr(RS, "_run_pipeline", forbidden)
    monkeypatch.setattr(RS, "_ai_pool", forbidden)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, raw, profile, work)
    assert not work.exists()
    assert not (tmp_path / "mosaic.png").exists()


def test_changed_prior_colour_manifest_cannot_reuse_work(flat_grid, tmp_path, monkeypatch):
    raw, profile = flat_grid
    path, manifest = corrected_manifest(raw)
    work = tmp_path / "bound"
    seen = []
    monkeypatch.setattr(RS, "_run_pipeline", lambda *a: seen.append(True))
    invoke(monkeypatch, raw, profile, work)
    before = (work / "state.json").read_bytes()
    # Pixels remain unchanged, but the source correction identity has changed.
    manifest["built_at_utc"] = "2026-10-31T01:00:00+00:00"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit):
        invoke(monkeypatch, raw, profile, work, "--force")
    assert (work / "state.json").read_bytes() == before
    assert seen == [True]


def mocked_pipeline(flat_grid, tmp_path, monkeypatch):
    """Keep real orchestration/state while replacing expensive stage algorithms."""
    import register as R
    import blend as B
    raw, _ = flat_grid
    tiles = SP.scan_grid_tiles(raw)
    positions = np.array([[0., 0.], [0., 36.], [48., 0.], [48., 36.]])
    monkeypatch.setattr(RS.T, "build_cache", lambda *a, **kw: {"cached": 0, "total": 4})
    monkeypatch.setattr(RS.T, "flag_tiles", lambda *a: {"conflicts": []})
    monkeypatch.setattr(RS, "step_qc", lambda *a: {tiles[1].tid})
    monkeypatch.setattr(RS, "step_conflict", lambda *a: set())
    monkeypatch.setattr(RS, "step_rescue", lambda *a, **kw: set())
    monkeypatch.setattr(RS, "step_inspect", lambda *a: [])
    monkeypatch.setattr(R, "build_edges", lambda *a, **kw: ([], {}))
    monkeypatch.setattr(R, "solve_positions", lambda *a: (positions.copy(),
                        {"residual_rms": 0., "n_components": 1}))
    monkeypatch.setattr(B, "estimate_flatfield", lambda *a, **kw: np.ones((64, 96, 3)))
    work = tmp_path / "processing"
    work.mkdir()
    state = RS.State(work / "state.json")
    state.d = {"run_binding": {"fingerprint": "synthetic-bound"},
               "summary": {"tiles_used": 99}, "render": {"n_placed": 99},
               "colour_correction": {"status": "old-success"},
               "_done": ["colour_correction", "render", "summary"]}
    args = SimpleNamespace(colour_correction="auto", colour_input={"state": "raw"},
                           grid_policy="legacy", scale_div=1, workers=1, votes=3,
                           out=str(work / "mosaic.png"), out_scale=1., full=False)
    return tiles, positions, args, work, state, B


def test_default_pipeline_corrects_registered_tiles_then_filters_render_gains(flat_grid, tmp_path, monkeypatch):
    tiles, positions, args, work, state, blend = mocked_pipeline(flat_grid, tmp_path, monkeypatch)
    gains = {tile.tid: [1.01 + i * .01, .98, 1.] for i, tile in enumerate(tiles)}
    calls = []

    def estimate(selected, solved, flatfield, **kwargs):
        calls.append("colour")
        assert [t.tid for t in selected] == [t.tid for t in tiles]
        np.testing.assert_array_equal(solved, positions)
        assert kwargs["binding"] == state.d["run_binding"]
        assert "summary" not in state.d and "render" not in state.d
        return gains, {"status": "ok", "gain_channel_order": "RGB"}

    def render(selected, solved, target, **kwargs):
        calls.append("render")
        expected = [tiles[i] for i in (0, 2, 3)]
        assert [t.tid for t in selected] == [t.tid for t in expected]
        np.testing.assert_array_equal(solved, positions[[0, 2, 3]])
        assert kwargs["tile_gains_rgb"] == {t.tid: gains[t.tid] for t in expected}
        return {"out_w": 144, "out_h": 100, "n_placed": 3,
                "rendered_tids": [t.tid for t in expected],
                "failed_load_tids": [], "missing_position_tids": []}

    monkeypatch.setattr(RS.SC, "solve_tile_gains", estimate)
    monkeypatch.setattr(blend, "render", render)
    RS._run_pipeline(args, work, str(work / "cache"), tiles, None, None, state, time.time())
    assert calls == ["colour", "render"]
    saved = json.loads(state.path.read_text())
    assert saved["summary"]["colour_correction"] == "applied_at_render"
    assert saved["summary"]["tiles_used"] == 3
    assert json.loads((work / "positions_tids.json").read_text()) == [t.tid for t in tiles]
    np.testing.assert_array_equal(np.load(work / "positions.npy"), positions)


@pytest.mark.parametrize("error_type", [ValueError, OSError, RuntimeError])
def test_colour_failure_prevents_render_and_removes_prior_success(flat_grid, tmp_path, monkeypatch, error_type):
    tiles, positions, args, work, state, blend = mocked_pipeline(flat_grid, tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise error_type("synthetic overlap evidence failure")

    def forbidden(*args, **kwargs):
        raise AssertionError("Colour failure must not fall through to uncorrected render")

    monkeypatch.setattr(RS.SC, "solve_tile_gains", fail)
    monkeypatch.setattr(blend, "render", forbidden)
    with pytest.raises(SystemExit):
        RS._run_pipeline(args, work, str(work / "cache"), tiles, None, None, state, time.time())
    saved = json.loads(state.path.read_text())
    assert saved["colour_correction"]["status"] == "failed"
    assert "summary" not in saved and "render" not in saved
    assert not {"colour_correction", "render", "summary"}.intersection(saved["_done"])
    report = json.loads((work / "colour_correction.json").read_text())
    assert report["status"] == "failed"
    assert "synthetic overlap evidence failure" in report["error"]


@pytest.mark.parametrize("policy,expected", [
    ("off", "disabled_by_explicit_request"),
    ("prior", "verified_prior_correction"),
])
def test_explicit_off_and_verified_prior_do_not_apply_new_tile_gains(flat_grid, tmp_path, monkeypatch, policy, expected):
    tiles, positions, args, work, state, blend = mocked_pipeline(flat_grid, tmp_path, monkeypatch)
    if policy == "off":
        args.colour_correction = "off"
    else:
        args.colour_input = {"state": "already_corrected", "manifest_sha256": "a" * 64}

    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit off/prior correction must not estimate a second RGB gain")

    monkeypatch.setattr(RS.SC, "solve_tile_gains", forbidden)
    assert RS.step_colour(args, work, tiles, positions, np.ones((64, 96, 3)), state) is None
    report = json.loads((work / "colour_correction.json").read_text())
    assert report["status"] == expected
    if policy == "prior":
        assert report["applied_again"] is False


def successful_gains(tiles):
    return {tile.tid: [1.01, .98, 1.] for tile in tiles}, {"status": "ok"}


def test_render_exception_cannot_leave_successful_colour_proof(flat_grid, tmp_path, monkeypatch):
    tiles, positions, args, work, state, blend = mocked_pipeline(flat_grid, tmp_path, monkeypatch)
    monkeypatch.setattr(RS.SC, "solve_tile_gains", lambda *a, **kw: successful_gains(tiles))

    def fail(*args, **kwargs):
        raise OSError("synthetic output write failed")

    monkeypatch.setattr(blend, "render", fail)
    with pytest.raises(OSError, match="synthetic output write failed"):
        RS._run_pipeline(args, work, str(work / "cache"), tiles, None, None, state, time.time())
    saved = json.loads(state.path.read_text())
    report = json.loads((work / "colour_correction.json").read_text())
    assert report["status"] == "render_failed"
    assert report["render_verified"] is False
    assert saved["colour_correction"]["status"] == "render_failed"
    assert "summary" not in saved
    assert not {"colour_correction", "render", "summary"}.intersection(saved["_done"])


@pytest.mark.parametrize("partly_rendered", [False, True])
def test_incomplete_render_cannot_leave_successful_colour_proof(flat_grid, tmp_path, monkeypatch, partly_rendered):
    tiles, positions, args, work, state, blend = mocked_pipeline(flat_grid, tmp_path, monkeypatch)
    args.grid_policy = "preserve"
    monkeypatch.setattr(RS.SC, "solve_tile_gains", lambda *a, **kw: successful_gains(tiles))

    def incomplete(selected, *args, **kwargs):
        rendered = [t.tid for t in selected] if partly_rendered else [t.tid for t in selected[:-1]]
        return {"out_w": 144, "out_h": 100, "n_placed": len(rendered),
                "rendered_tids": rendered,
                "failed_load_tids": [selected[-1].tid], "missing_position_tids": []}

    monkeypatch.setattr(blend, "render", incomplete)
    with pytest.raises(SystemExit):
        RS._run_pipeline(args, work, str(work / "cache"), tiles, None, None, state, time.time())
    saved = json.loads(state.path.read_text())
    report = json.loads((work / "colour_correction.json").read_text())
    assert report["status"] == "render_failed"
    assert report["render_verified"] is False
    assert "summary" not in saved
    assert not {"colour_correction", "render", "summary"}.intersection(saved["_done"])


def test_cache_failure_invalidates_standalone_colour_report_from_previous_run(flat_grid, tmp_path, monkeypatch):
    tiles, positions, args, work, state, blend = mocked_pipeline(flat_grid, tmp_path, monkeypatch)
    report_path = work / "colour_correction.json"
    report_path.write_text(json.dumps({"status": "applied_at_render", "render_verified": True}))

    def fail(*args, **kwargs):
        raise OSError("synthetic cache decode failed")

    monkeypatch.setattr(RS.T, "build_cache", fail)
    with pytest.raises(OSError, match="synthetic cache decode failed"):
        RS._run_pipeline(args, work, str(work / "cache"), tiles, None, None, state, time.time())
    saved = json.loads(state.path.read_text())
    report = json.loads(report_path.read_text())
    assert report["status"] == "pending"
    assert report["render_verified"] is False
    assert "summary" not in saved and "render" not in saved
    assert not saved["_done"]

