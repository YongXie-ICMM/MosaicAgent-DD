"""Offline CLI protection for complete recorded grids and work-cache binding."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time

from PIL import Image
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_stitch as RS
import stitch_profile as SP


def test_auto_only_trusts_complete_flat_grid():
    assert RS.resolve_grid_policy("auto", "flat-physical-grid") == "preserve"
    for source in ("validated-layout", "needs-layout", "historical"):
        assert RS.resolve_grid_policy("auto", source) == "legacy"
        with pytest.raises(SP.ProfileError, match="requires a complete"):
            RS.resolve_grid_policy("preserve", source)
    assert RS.resolve_grid_policy("legacy", "flat-physical-grid") == "legacy"


def fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for c in range(2):
        for r in range(2):
            Image.new("RGB", (64, 48), "gray").save(raw / f"mosaic_r{r}_c{c}.png")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema_version": 1, "name": "synthetic",
                                  "image_size": [64, 48], "nominal_vectors": [0, 36, 48, 0],
                                  "geometry_source": "synthetic", "scale_div": 1}))
    return raw, profile


def invoke(monkeypatch, raw, profile, work, policy):
    monkeypatch.setattr(sys, "argv", ["run_stitch.py", "--data", str(raw),
                        "--stitch-profile", str(profile), "--work", str(work),
                        "--no-ai", "--grid-policy", policy])
    RS.main()


def test_cli_selects_policy_and_rejects_switching_cached_work(tmp_path, monkeypatch):
    raw, profile = fixture(tmp_path)
    calls = []
    def capture(args, work, cache_dir, ts, pool, client, state, started):
        calls.append(args.grid_policy)
        assert state.d["grid_policy"] == args.grid_policy
        assert state.d["preflight"]["grid_policy"] == args.grid_policy
        assert pool is None and client is None
    monkeypatch.setattr(RS, "_run_pipeline", capture)
    work = tmp_path / "protected"
    invoke(monkeypatch, raw, profile, work, "auto")
    invoke(monkeypatch, raw, profile, work, "preserve")
    before = (work / "state.json").read_bytes()
    with pytest.raises(SystemExit):
        invoke(monkeypatch, raw, profile, work, "legacy")
    assert (work / "state.json").read_bytes() == before
    invoke(monkeypatch, raw, profile, tmp_path / "legacy", "legacy")
    assert calls == ["preserve", "preserve", "legacy"]


def test_legacy_binding_keeps_old_contract_and_protection_changes_it():
    resolved = {"profile": {}, "scale_div": 1, "out_scale": .125, "full": False}
    inventory = {"fingerprint": "fixture"}
    old = SP.make_run_binding(resolved, inventory)
    assert SP.make_run_binding({**resolved, "grid_policy": "legacy"}, inventory) == old
    protected = SP.make_run_binding({**resolved, "grid_policy": "preserve"}, inventory)
    assert protected["grid_policy"] == "preserve"
    assert protected["fingerprint"] != old["fingerprint"]


@pytest.mark.parametrize("partial", [False, True])
def test_pipeline_does_not_report_success_when_a_grid_tile_fails_to_render(tmp_path, monkeypatch, partial):
    import register as R
    import blend as B
    ts = [SimpleNamespace(tid="a"), SimpleNamespace(tid="b")]
    monkeypatch.setattr(RS.T, "build_cache", lambda *a, **kw: {"cached": 0, "total": 2})
    monkeypatch.setattr(RS.T, "flag_tiles", lambda *a: {"conflicts": []})
    monkeypatch.setattr(RS, "step_qc", lambda *a: {"b"})
    monkeypatch.setattr(RS, "step_conflict", lambda *a: set())
    def edges(*a, **kw):
        assert kw["row_policy"] == "recorded-grid"
        return [], {}
    monkeypatch.setattr(R, "build_edges", edges)
    monkeypatch.setattr(R, "solve_positions", lambda *a: (np.array([[0., 0.], [48., 0.]]),
                        {"residual_rms": 0, "n_components": 1}))
    monkeypatch.setattr(B, "estimate_flatfield", lambda *a, **kw: np.ones((2, 2, 3)))
    monkeypatch.setattr(B, "render", lambda *a, **kw: {"out_w": 100, "out_h": 48,
                        "rendered_tids": ["a", "b"] if partial else ["a"],
                        "n_placed": 2 if partial else 1, "failed_load_tids": ["b"],
                        "missing_position_tids": []})
    state = RS.State(tmp_path / "state.json")
    state.d = {"summary": {"tiles_used": 2}, "_done": ["render", "summary"]}
    args = SimpleNamespace(grid_policy="preserve", scale_div=1, workers=1, votes=3,
                           out=str(tmp_path / "partial.png"), out_scale=1, full=False)
    with pytest.raises(SystemExit):
        RS._run_pipeline(args, tmp_path, str(tmp_path / "cache"), ts, None, None, state, time.time())
    saved = json.loads(state.path.read_text())
    assert saved["rescue"] == ["b"]
    assert saved["render"]["incomplete_tids"] == ["b"]
    assert "summary" not in saved and "render" not in saved["_done"]


def test_pipeline_clears_success_before_any_stage_can_raise(tmp_path, monkeypatch):
    state = RS.State(tmp_path / "state.json")
    state.d = {"summary": {"tiles_used": 2}, "render": {"n_placed": 2},
               "_done": ["render", "summary"]}
    def fail(*a, **kw):
        raise OSError("synthetic decode failure")
    monkeypatch.setattr(RS.T, "build_cache", fail)
    args = SimpleNamespace(scale_div=1)
    with pytest.raises(OSError):
        RS._run_pipeline(args, tmp_path, str(tmp_path / "cache"), [], None, None, state, time.time())
    saved = json.loads(state.path.read_text())
    assert "render" not in saved and "summary" not in saved and not saved["_done"]
