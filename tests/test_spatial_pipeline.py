"""Opt-in spatial exports and cache integrity on synthetic local masks.

The renderer contract is stubbed here; numeric mapping is tested by spatial_maps.
No camera, segmentation checkpoint, or external model is used.
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flakepipeline"))
import stages
import orchestrator as O


@pytest.fixture
def data(tmp_path):
    # Unequal native classes: 1=2L, 2=1L, 3=TL. Support and exclusion
    # deliberately remove different pixels, including one background pixel.
    mask = np.array([[0, 1, 2, 2], [3, 2, 1, 3], [2, 3, 2, 0]], dtype=np.uint8)
    valid = np.ones_like(mask, dtype=bool)
    valid[0, 0] = False
    excluded = np.zeros_like(valid)
    excluded[1, 1] = True
    state = {}
    for name, values in (("mask", mask), ("valid_mask", valid), ("exclusion_mask", excluded)):
        path = tmp_path / (name + ".npy")
        np.save(path, values)
        state[name] = str(path)
    ctx = {"work": tmp_path, "sample": "S", "map_preview_div": 1}
    return ctx, state


@pytest.fixture
def exporter(monkeypatch):
    calls = []

    def export(mask, valid, exclusion, output, sample, **kwargs):
        output = Path(output)
        assert not output.exists()
        output.mkdir(parents=True)
        artifact = output / "tiles.csv"
        artifact.write_text("row,1L,2L,TL\n1,4,2,3\n", encoding="utf-8")
        manifest = output / "manifest.json"
        manifest.write_text('{"synthetic":true}', encoding="utf-8")
        calls.append((mask, valid, exclusion, output, sample, kwargs))
        return {"manifest": str(manifest), "outputs": [str(manifest), str(artifact)],
                "summary": {"synthetic": True}}

    module = SimpleNamespace(export_spatial_maps=export, SPATIAL_MAPS_VERSION="test-1",
                             implementation_fingerprint=lambda: "a" * 64)
    monkeypatch.setattr(stages, "_spatial_maps_module", lambda: module)
    # Numeric stats is real. Its legacy optional pie rendering is irrelevant to
    # these hook/cache tests; disabling it keeps the tests small and offline.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    return module, calls


def stats_spec(tmp_path):
    orch = O.Orchestrator({"work_root": str(tmp_path)})
    spec = next(spec for spec in orch._specs() if spec["name"] == "stats")
    return orch, spec


def record(orch, spec, ctx, state, result):
    manifest = {"stages": {}}
    orch._record(manifest, ctx["work"] / "manifest.json", "stats",
                 spec["key"](ctx, state), result, spec["outputs"](ctx, result.data),
                 [], O.Decision("accept" if result.ok else "escalate"))
    return manifest


def test_disabled_export_preserves_existing_stats_and_key(data, exporter):
    ctx, state = data
    orch, spec = stats_spec(ctx["work"])
    initial_key = spec["key"](ctx, state)
    result = spec["run"](ctx, state)
    assert result.ok
    assert result.data["counted_px"] == 9
    assert result.data["ratios_pct"] == {"2L": 22.22, "1L": 44.44, "TL": 33.33}
    assert "spatial_maps" not in result.data
    assert "spatial_maps" not in initial_key
    assert not (ctx["work"] / "04_stats/spatial_maps").exists()
    ctx["spatial_maps"] = {"enabled": False}
    assert spec["key"](ctx, state) == initial_key
    assert exporter[1] == []


def test_enabled_export_binds_native_sources_and_persists_outputs(data, exporter):
    ctx, state = data
    ctx["spatial_maps"] = {"enabled": True, "heatmap_grid": [4, 3], "combined_grid": [2, 3]}
    orch, spec = stats_spec(ctx["work"])
    result = spec["run"](ctx, state)
    call = exporter[1][0]
    assert result.ok
    assert call[:3] == (state["mask"], state["valid_mask"], state["exclusion_mask"])
    assert call[4] == "S"
    assert call[5] == {"heatmap_grid": (4, 3), "combined_grid": (2, 3)}
    assert call[3].parent == ctx["work"] / "04_stats/spatial_maps"
    stats = json.loads((ctx["work"] / "04_stats/S_stats.json").read_text(encoding="utf-8"))
    assert stats["spatial_maps"] == result.data["spatial_maps"]
    artifacts = spec["outputs"](ctx, result.data)
    assert len(artifacts) == len(set(artifacts)) == 3
    assert set(result.data["spatial_maps"]["outputs"]) <= set(artifacts)
    manifest = record(orch, spec, ctx, state, result)
    assert orch._cached(manifest, "stats", spec["key"](ctx, state)) is not None
    original = {path: Path(path).read_bytes() for path in result.data["spatial_maps"]["outputs"]}
    second = spec["run"](ctx, state)
    assert second.data["spatial_maps"]["manifest"] != result.data["spatial_maps"]["manifest"]
    assert all(Path(path).read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("change", ["heatmap_grid", "combined_grid", "implementation", "version", "disable"])
def test_spatial_config_and_implementation_invalidate_stats_cache(data, exporter, change):
    ctx, state = data
    ctx["spatial_maps"] = {"enabled": True}
    orch, spec = stats_spec(ctx["work"])
    result = spec["run"](ctx, state)
    manifest = record(orch, spec, ctx, state, result)
    if change in ("heatmap_grid", "combined_grid"):
        ctx["spatial_maps"][change] = [2, 2]
    elif change == "implementation":
        exporter[0].implementation_fingerprint = lambda: "b" * 64
    elif change == "version":
        exporter[0].SPATIAL_MAPS_VERSION = "test-2"
    else:
        ctx["spatial_maps"]["enabled"] = False
    assert orch._cached(manifest, "stats", spec["key"](ctx, state)) is None


@pytest.mark.parametrize("artifact_index", [0, 1])
@pytest.mark.parametrize("damage", ["delete", "tamper"])
def test_spatial_manifest_and_data_are_checked_before_cache_reuse(data, exporter, artifact_index, damage):
    ctx, state = data
    ctx["spatial_maps"] = {"enabled": True}
    orch, spec = stats_spec(ctx["work"])
    result = spec["run"](ctx, state)
    manifest = record(orch, spec, ctx, state, result)
    artifact = Path(result.data["spatial_maps"]["outputs"][artifact_index])
    if damage == "delete":
        artifact.unlink()
    else:
        content = artifact.read_bytes()
        artifact.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    assert orch._cached(manifest, "stats", spec["key"](ctx, state)) is None


@pytest.mark.parametrize("failure", ["exception", "missing_artifact", "outside_run"])
def test_export_failure_is_recorded_and_never_cached_as_success(data, exporter, failure):
    ctx, state = data
    ctx["spatial_maps"] = {"enabled": True}
    original = exporter[0].export_spatial_maps
    def fail(*args, **kwargs):
        if failure == "exception":
            raise RuntimeError("synthetic export failure")
        value = original(*args, **kwargs)
        if failure == "missing_artifact":
            Path(value["outputs"][-1]).unlink()
        else:
            value["outputs"].append(state["mask"])
        return value
    exporter[0].export_spatial_maps = fail
    orch, spec = stats_spec(ctx["work"])
    result = spec["run"](ctx, state)
    assert not result.ok and result.escalate
    assert "spatial_maps_error" in result.data and "spatial_maps" not in result.data
    assert result.data["counted_px"] == 9
    persisted = json.loads((ctx["work"] / "04_stats/S_stats.json").read_text(encoding="utf-8"))
    assert persisted["spatial_maps_error"] == result.data["spatial_maps_error"]
    manifest = record(orch, spec, ctx, state, result)
    assert orch._cached(manifest, "stats", spec["key"](ctx, state)) is None


@pytest.mark.parametrize("settings", [True, {"enabled": "yes"}, {"enabled": True, "heatmap_grid": [True, 2]},
                                     {"enabled": True, "combined_grid": [2, 0]},
                                     {"enabled": True, "heatmap_grid": [2.0, 3]}])
def test_bad_settings_fail_before_writing_stats(data, exporter, settings):
    ctx, state = data
    ctx["spatial_maps"] = settings
    with pytest.raises(ValueError, match="spatial_maps"):
        stages.stage_stats(ctx, state["mask"], state["valid_mask"], state["exclusion_mask"])
    assert not (ctx["work"] / "04_stats").exists()
    assert not exporter[1]


def test_cli_spatial_switch_only_enables_requested_exports(tmp_path, monkeypatch):
    calls = []
    module = ModuleType("orchestrator")
    class FakeOrchestrator:
        def __init__(self, cfg):
            calls.append(cfg)
            self.cfg = cfg
        def run(self, samples):
            return {"samples": [], "consistency": [], "config": self.cfg}
    module.Orchestrator = FakeOrchestrator
    monkeypatch.setitem(sys.modules, "orchestrator", module)
    spec = importlib.util.spec_from_file_location("_spatial_cli_test", ROOT / "flakepipeline/run.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    path = tmp_path / "config.json"
    config = {"work_root": str(tmp_path), "samples": [],
              "spatial_maps": {"enabled": False, "heatmap_grid": [3, 5]}}
    path.write_text(json.dumps(config), encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(path)])
    cli.main()
    assert calls[-1]["spatial_maps"]["enabled"] is False
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(path), "--spatial-maps"])
    cli.main()
    assert calls[-1]["spatial_maps"] == {"enabled": True, "heatmap_grid": [3, 5]}
    assert path.read_bytes() == before


def test_real_spatial_export_matches_native_stats_and_keeps_sources(data):
    ctx, state = data
    ctx["spatial_maps"] = {"enabled": True, "heatmap_grid": [4, 3], "combined_grid": [2, 3]}
    before = {path: Path(path).read_bytes() for path in state.values()}
    result = stages.stage_stats(ctx, state["mask"], state["valid_mask"], state["exclusion_mask"])
    assert result.ok, result.data.get("spatial_maps_error")
    spatial = result.data["spatial_maps"]
    whole = spatial["summary"]["global"]
    assert whole["counts_px"] == {"background": 1, "1L": 4, "2L": 2, "TL": 3}
    assert sum(whole["counts_px"][layer] for layer in ("1L", "2L", "TL")) == result.data["counted_px"]
    for layer in ("1L", "2L", "TL"):
        assert round(whole["of_crystal_pct"][layer], 2) == result.data["ratios_pct"][layer]
    assert all(Path(path).is_file() for path in spatial["outputs"])
    assert all(Path(path).read_bytes() == content for path, content in before.items())
    orch, spec = stats_spec(ctx["work"])
    saved = record(orch, spec, ctx, state, result)
    assert orch._cached(saved, "stats", spec["key"](ctx, state)) is not None
