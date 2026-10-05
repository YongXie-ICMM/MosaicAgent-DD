"""CLI binding checks using actual handoff geometry and a nonexecuting orchestrator."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def handoff(folder, scale=1):
    from tools.stitch_setup.service import _layer_input_contract
    folder.mkdir()
    profile = {"schema_version": 1, "name": "cli-synthetic",
               "image_size": [1920, 1080], "nominal_vectors": [0, 1000, 1800, 0],
               "geometry_source": "synthetic-test", "scale_div": 1, "out_scale": scale}
    inventory = {"image_size": [1920, 1080], "fingerprint": "1" * 64, "tile_count": 4}
    inspection = {"image_size": [1920, 1080], "fingerprint": "2" * 64,
                  "layout_mode": "flat-grid", "acquisition": {"present": False}}
    resolved = {"profile": profile, "scale_div": 1, "out_scale": scale, "full": True}
    preflight = {"schema_version": 1, "inspection": inspection,
                 "input_inventory": inventory, "resolved": resolved}
    preflight_text = json.dumps(preflight) + "\n"
    profile_text = json.dumps(profile) + "\n"
    value = _layer_input_contract(inspection, inventory, resolved, preflight_text, profile_text)
    (folder / "profile.json").write_text(profile_text)
    (folder / "preflight.json").write_text(preflight_text)
    path = folder / "layer_input_contract.json"
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def cli(tmp_path, monkeypatch):
    calls = []
    module = ModuleType("orchestrator")

    class NonexecutingOrchestrator:
        def __init__(self, cfg):
            self.cfg = cfg
            self.work_root = Path(cfg["work_root"])
            calls.append(("construct", cfg))
        def run(self, samples):
            calls.append(("run", samples))
            return {"samples": [], "consistency": [], "config": self.cfg}
        def plan(self, samples):
            calls.append(("plan", samples))
            return {"samples": samples}
        def print_plan(self, value):
            calls.append(("print_plan", value))

    module.Orchestrator = NonexecutingOrchestrator
    monkeypatch.setitem(sys.modules, "orchestrator", module)
    spec = importlib.util.spec_from_file_location("_layer_contract_cli_test", ROOT / "flakepipeline/run.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    config = {"work_root": str(tmp_path / "not-created"), "weights": "synthetic-not-loaded.pth",
              "samples": [{"name": "sample_a", "mosaic": "not-opened.png"}],
              "tile": 512, "overlap": 64}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    def invoke(*args):
        monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_path), *map(str, args)])
        return entry.main()
    return invoke, calls, config, config_path


def test_default_cli_keeps_existing_config_and_execution_path(cli):
    invoke, calls, config, path = cli
    original = path.read_bytes()
    invoke()
    assert calls == [("construct", {key: value for key, value in config.items() if key != "samples"}),
                     ("run", config["samples"])]
    assert "inference_geometry" not in calls[0][1]
    assert path.read_bytes() == original
    assert not Path(config["work_root"]).exists()


def test_cli_injects_bound_geometry_and_retains_existing_flags(cli, tmp_path):
    invoke, calls, config, path = cli
    contract = handoff(tmp_path / "bundle")
    invoke("--layer-input-contract", contract, "--reference-capture-size", 3840, 2160,
           "--same-fov-confirmed", "--no-ai", "--no-repair")
    cfg = calls[0][1]
    geometry = cfg["inference_geometry"]
    assert geometry["status"] == "ready"
    assert geometry["observed_native_size"] == [1920, 1080]
    assert (geometry["native_tile"], geometry["native_overlap"]) == (256, 32)
    assert geometry["model_tile"] == 512 and geometry["model_overlap"] == 64
    assert hashlib.sha256(contract.read_bytes()).hexdigest() in geometry["provenance"]["note"]
    assert cfg["no_ai"] is True and cfg["no_repair"] is True
    assert calls[1] == ("run", config["samples"])


def test_plan_mode_uses_configured_model_windows_without_execution(cli, tmp_path):
    invoke, calls, config, path = cli
    config.update(tile=256, overlap=32)
    path.write_text(json.dumps(config))
    contract = handoff(tmp_path / "bundle")
    invoke("--layer-input-contract", contract, "--reference-capture-size", 3840, 2160,
           "--same-fov-confirmed", "--plan")
    geometry = calls[0][1]["inference_geometry"]
    assert (geometry["model_tile"], geometry["model_overlap"]) == (256, 32)
    assert (geometry["native_tile"], geometry["native_overlap"]) == (128, 16)
    assert [name for name, _ in calls] == ["construct", "plan", "print_plan"]


@pytest.mark.parametrize("failure", ["confirmation", "downsample", "profile", "preflight", "native", "missing"])
def test_bad_handoff_fails_before_orchestrator_construction(cli, tmp_path, capsys, failure):
    invoke, calls, config, path = cli
    contract = handoff(tmp_path / "bundle", scale=0.125 if failure == "downsample" else 1)
    args = ["--layer-input-contract", contract, "--reference-capture-size", 3840, 2160]
    if failure != "confirmation":
        args.append("--same-fov-confirmed")
    if failure in ("profile", "preflight"):
        sibling = contract.with_name(failure + ".json")
        sibling.write_bytes(sibling.read_bytes() + b"\n")
    elif failure == "native":
        record = json.loads(contract.read_text())
        record["native_image_size"] = [3840, 2160]
        contract.write_text(json.dumps(record))
    elif failure == "missing":
        contract.unlink()
    with pytest.raises(SystemExit) as error:
        invoke(*args)
    assert error.value.code == 2
    assert calls == []
    assert not Path(config["work_root"]).exists()
    message = capsys.readouterr().err
    assert "error:" in message and "Traceback" not in message
    if failure == "downsample":
        assert "mosaic_output_scale_needs_explicit_mapping" in message


@pytest.mark.parametrize("args", [("--same-fov-confirmed",),
                                  ("--reference-capture-size", "3840", "2160"),
                                  ("--layer-input-contract", "missing.json")])
def test_geometry_options_cannot_be_partially_supplied(cli, args):
    invoke, calls, config, path = cli
    with pytest.raises(SystemExit) as error:
        invoke(*args)
    assert error.value.code == 2 and calls == []
