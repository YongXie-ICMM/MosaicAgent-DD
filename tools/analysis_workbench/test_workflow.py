"""Bound DD jobs: synthetic local fixtures, no devices/models/external calls."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import Mock, patch

import pytest

import stitch_profile as SP
from tools.analysis_workbench import workflow as W


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGOorq0HAALuAXgUVWfXAAAAAElFTkSuQmCC")


class FakeProcess:
    def __init__(self):
        self.pid = 954
        self.returncode = None

    def poll(self):
        return self.returncode


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@pytest.fixture
def prepared(tmp_path):
    repo, bundle, raw = tmp_path / "repo", tmp_path / "bundle", tmp_path / "raw"
    repo.mkdir()
    bundle.mkdir()
    raw.mkdir()
    (repo / "run_stitch.py").write_text("# fake local fixture; not an experiment\n")
    profile = {"schema_version": 1, "name": "synthetic", "image_size": [320, 180],
               "nominal_vectors": [0, 140, 270, 0], "geometry_source": "synthetic"}
    resolved = {"profile": profile, "scale_div": 4, "out_scale": 0.25, "full": False,
                "colour_correction": "auto", "colour_algorithm": "synthetic-test-only", "colour_input": {"state": "raw"}}
    inventory = {"fingerprint": "a" * 64, "tile_count": 4, "tiles": [{"tid": f"t{i}"} for i in range(4)]}
    inspection = {"layout_mode": "flat-grid", "data_dir": str(raw), "acquisition": {"present": False}}
    argv = [sys.executable, str(repo / "run_stitch.py"), "--data", str(raw), "--work", str(bundle / "work"),
            "--out", str(bundle / "mosaic.png"), "--stitch-profile", str(bundle / "profile.json"),
            "--scale-div", "4", "--out-scale", "0.25", "--input-preflight", str(bundle / "preflight.json"),
            "--colour-correction", "auto", "--no-ai"]
    preflight = {"status": "profile_prepared", "resolved": resolved, "input_inventory": inventory,
                 "inspection": inspection, "argv": argv, "layout": None}
    write(bundle / "argv.json", argv)
    write(bundle / "profile.json", profile)
    write(bundle / "preflight.json", preflight)
    result = {"bundle_dir": str(bundle), "argv_path": str(bundle / "argv.json"),
              "profile_path": str(bundle / "profile.json"), "preflight_path": str(bundle / "preflight.json")}
    manager = W.DDWorkflow(tmp_path / "runtime", repo=repo)
    return manager, result


def complete(manager, ticket, **changes):
    internal = manager.prepared[ticket["id"]]
    binding = internal["expected_binding"]
    colour = {"status": "applied_at_render", "render_verified": True, "run_binding": binding}
    output = Path(internal["expected_output"])
    state = {"run_binding": binding, "_done": ["register", "colour_correction", "render", "summary"],
             "colour_correction": colour,
             "render": {"path": str(output), "out_w": 1, "out_h": 1, "n_placed": 4,
                        "rendered_tids": [f"t{i}" for i in range(4)], "n_failed_loads": 0,
                        "failed_load_tids": [], "missing_position_tids": [], "incomplete_tids": [], "missing_selected_tids": []},
             "summary": {"tiles_selected": 4, "tiles_used": 4, "tiles_total": 4,
                         "render_failed": [], "out": str(output), "colour_correction": "applied_at_render"}}
    for section, values in changes.items():
        state[section].update(values)
    output.write_bytes(PNG)
    write(Path(internal["state_path"]), state)
    write(Path(internal["state_path"]).parent / "colour_correction.json", state["colour_correction"])
    return state


def launch_fake(manager, result):
    ticket = manager.register_prepared(result)
    process = FakeProcess()
    with patch.object(W.subprocess, "Popen", return_value=process) as popen, patch.object(W.threading, "Thread"):
        job = manager.start(ticket["id"])
    return ticket, job, process, popen


def finish_fake(manager, job, process, code=0):
    process.returncode = code
    manager._watch(job["id"])
    return manager.state()["jobs"][0]


def test_inspection_registration_and_state_never_launch_a_child(prepared):
    manager, result = prepared
    with patch.object(W.subprocess, "Popen", side_effect=AssertionError("No child")):
        ticket = manager.register_prepared(result)
        state = manager.state()
    assert ticket["kind"] == "stitch" and ticket["status"] == "prepared"
    assert state["prepared"][0]["id"] == ticket["id"]
    assert state["jobs"] == [] and state["active_job_id"] is None
    assert "argv" not in ticket


def test_launch_uses_generated_list_shell_false_and_duplicate_click_is_one_job(prepared):
    manager, result = prepared
    ticket, job, process, popen = launch_fake(manager, result)
    with patch.object(W.subprocess, "Popen", side_effect=AssertionError("Duplicate launch")):
        again = manager.start(ticket["id"])
    assert again["id"] == job["id"]
    assert popen.call_count == 1
    args, kwargs = popen.call_args
    assert isinstance(args[0], list) and kwargs["shell"] is False
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert Path(job["log_path"]).parent.is_relative_to(manager.directory)
    complete(manager, ticket)
    done = finish_fake(manager, job, process)
    assert done["status"] == "completed"
    assert done["render_verified"] and done["colour_status"] == "applied_at_render"
    assert done["source_dir"] == ticket["source_dir"]
    assert done["source_binding"]["input_fingerprint"] == "a" * 64
    assert done["coverage"]["fraction"] == 1.0
    assert len(done["artifacts"]) == 3


@pytest.mark.parametrize("field", ["argv_path", "profile_path", "preflight_path"])
def test_changed_prepared_control_refuses_launch(prepared, field):
    manager, result = prepared
    ticket = manager.register_prepared(result)
    Path(result[field]).write_text("{}")
    with patch.object(W.subprocess, "Popen", side_effect=AssertionError("No child")):
        with pytest.raises(ValueError, match="changed"):
            manager.start(ticket["id"])
    assert manager.state()["jobs"] == []


@pytest.mark.parametrize("case", ["script", "off", "arbitrary_flag", "outside_work"])
def test_untrusted_or_off_commands_are_not_registered(prepared, case):
    manager, result = prepared
    argv = W._read(result["argv_path"])
    if case == "script":
        argv[1] = "-c"
    elif case == "off":
        argv[argv.index("--colour-correction") + 1] = "off"
    elif case == "arbitrary_flag":
        argv += ["--force"]
    elif case == "outside_work":
        argv[argv.index("--work") + 1] = str(manager.repo / "unrelated")
    write(Path(result["argv_path"]), argv)
    preflight = W._read(result["preflight_path"])
    preflight["argv"] = argv
    write(Path(result["preflight_path"]), preflight)
    with pytest.raises(ValueError):
        manager.register_prepared(result)


def test_zero_exit_without_outputs_is_failure(prepared):
    manager, result = prepared
    ticket, job, process, _ = launch_fake(manager, result)
    done = finish_fake(manager, job, process)
    assert done["status"] == "failed" and done["return_code"] == 0
    assert done["output_path"] is None and not done["artifacts"]


@pytest.mark.parametrize("case", ["wrong_binding", "incomplete", "colour_unverified", "missing_grid_id", "bad_png", "different_colour_record"])
def test_unverified_results_cannot_be_marked_completed(prepared, case):
    manager, result = prepared
    ticket, job, process, _ = launch_fake(manager, result)
    state = complete(manager, ticket)
    state_path = Path(manager.prepared[ticket["id"]]["state_path"])
    if case == "wrong_binding":
        state["run_binding"] = {}
    elif case == "incomplete":
        state["render"]["failed_load_tids"] = ["t3"]
    elif case == "colour_unverified":
        state["colour_correction"]["render_verified"] = False
    elif case == "missing_grid_id":
        state["render"]["rendered_tids"][3] = "some-other-tile"
    elif case == "bad_png":
        Path(manager.prepared[ticket["id"]]["expected_output"]).write_bytes(PNG[:24])
    elif case == "different_colour_record":
        write(state_path.parent / "colour_correction.json", {})
    write(state_path, state)
    done = finish_fake(manager, job, process)
    assert done["status"] == "failed" and done["error"]


def test_process_error_and_launch_error_are_recorded(prepared):
    manager, result = prepared
    ticket, job, process, _ = launch_fake(manager, result)
    done = finish_fake(manager, job, process, code=7)
    assert done["status"] == "failed" and done["return_code"] == 7
    assert Path(job["log_path"]).is_file()


def test_restart_keeps_active_outcome_unknown_and_never_replays(prepared):
    manager, result = prepared
    ticket, job, _, _ = launch_fake(manager, result)
    with patch.object(W.subprocess, "Popen", side_effect=AssertionError("No replay")):
        restored = W.DDWorkflow(manager.directory.parent, repo=manager.repo)
        state = restored.state()
        duplicate = restored.start(ticket["id"])
    assert state["jobs"][0]["status"] == "attention"
    assert duplicate["id"] == job["id"] and duplicate["status"] == "attention"
    records = [json.loads(line) for line in (manager.directory / "events.jsonl").read_text().splitlines()]
    assert records[-1]["event"] == "restart_attention"
    assert records[-1]["job_id"] == job["id"]


def test_live_colour_and_stage_are_read_from_bound_state(prepared):
    manager, result = prepared
    ticket, _, _, _ = launch_fake(manager, result)
    complete(manager, ticket)
    live = manager.state()["jobs"][0]
    assert live["status"] == "running"
    assert live["render_verified"] and live["colour_status"] == "applied_at_render"
    assert live["progress"]["completed_steps"][-1] == "summary"


def test_real_background_child_writes_local_fixture_and_is_verified(prepared):
    manager, result = prepared
    # This fake CLI is under the fixed repo/run_stitch.py path and writes a
    # declared synthetic result; it cannot load a model, device or credentials.
    source = '''import json,sys,base64
from pathlib import Path
argv=sys.argv
work=Path(argv[argv.index('--work')+1]);out=Path(argv[argv.index('--out')+1])
template=json.loads((Path(__file__).parent/'synthetic_state.json').read_text())
work.mkdir(parents=True,exist_ok=True)
out.write_bytes(base64.b64decode(%r))
(work/'state.json').write_text(json.dumps(template))
(work/'colour_correction.json').write_text(json.dumps(template['colour_correction']))
print('Synthetic local child completed')
''' % base64.b64encode(PNG).decode()
    (manager.repo / "run_stitch.py").write_text(source)
    ticket = manager.register_prepared(result)
    template = complete(manager, ticket)
    Path(manager.prepared[ticket["id"]]["expected_output"]).unlink()
    state_path = Path(manager.prepared[ticket["id"]]["state_path"])
    state_path.unlink()
    (state_path.parent / "colour_correction.json").unlink()
    write(manager.repo / "synthetic_state.json", template)
    job = manager.start(ticket["id"])
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        current = manager.state()["jobs"][0]
        if current["status"] not in ("running", "starting"):
            break
        time.sleep(0.02)
    assert current["id"] == job["id"] and current["status"] == "completed", current
    assert "Synthetic local child" in Path(current["log_path"]).read_text()


def layer_plan(manager, tmp_path):
    script = manager.repo / "flakepipeline/run.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text("# Synthetic pipeline fixture only\n")
    output = tmp_path / "layers"
    config = output / "control/layers_config.json"
    write(config, {"synthetic": True})
    files = [{"path": str(path), "sha256": W._digest(path)} for path in (script, config)]
    return {"kind": "layers", "output_dir": str(output), "files": files,
            "source_binding": {"kind": "synthetic-only"},
            "expected": {"report_path": str(output / "run/report.json"), "samples": []},
            "argv": [sys.executable, str(script), "--config", str(config), "--no-ai", "--no-repair"]}


def test_generic_analysis_register_is_inert_and_fixed_command_is_bound(prepared, tmp_path):
    manager, _ = prepared
    plan = layer_plan(manager, tmp_path)
    with patch.object(W.subprocess, "Popen", side_effect=AssertionError("No child")):
        ticket = manager.register_plan(plan)
        assert ticket["kind"] == "layers"
        assert manager.state()["jobs"] == []
    Path(plan["files"][0]["path"]).write_text("# changed\n")
    with pytest.raises(ValueError, match="changed"):
        manager.start(ticket["id"])


def test_generic_analysis_needs_validated_outputs_and_normalizes_artifacts(prepared, tmp_path):
    manager, _ = prepared
    plan = layer_plan(manager, tmp_path)
    ticket = manager.register_plan(plan)
    process = FakeProcess()
    with patch.object(W.subprocess, "Popen", return_value=process), patch.object(W.threading, "Thread"):
        job = manager.start(ticket["id"])
    output = Path(plan["expected"]["report_path"])
    write(output, {"synthetic": True})
    from tools.analysis_workbench import workflow_analysis as analysis
    with patch.object(analysis, "validate_plan_result", return_value={
            "status": "validated", "output_path": str(output), "artifacts": [str(output)], "review_required": True}):
        done = finish_fake(manager, job, process)
    assert done["status"] == "completed" and done["review_required"]
    assert done["artifacts"][0]["name"] == "run/report.json"
    assert done["artifacts"][0]["sha256"] == W._digest(output)
    assert done["source_binding"] == plan["source_binding"]
    assert done["source_files"] == plan["files"]


@pytest.mark.parametrize("case", ["unvalidated", "outside_output"])
def test_generic_zero_exit_cannot_bypass_analysis_validation(prepared, tmp_path, case):
    manager, _ = prepared
    plan = layer_plan(manager, tmp_path)
    ticket = manager.register_plan(plan)
    process = FakeProcess()
    with patch.object(W.subprocess, "Popen", return_value=process), patch.object(W.threading, "Thread"):
        job = manager.start(ticket["id"])
    from tools.analysis_workbench import workflow_analysis as analysis
    if case == "unvalidated":
        validation = patch.object(analysis, "validate_plan_result", side_effect=ValueError("Manifest not bound"))
    else:
        outside = tmp_path / "outside.json"
        write(outside, {})
        validation = patch.object(analysis, "validate_plan_result", return_value={
            "output_path": str(outside), "artifacts": [str(outside)]})
    with validation:
        done = finish_fake(manager, job, process)
    assert done["status"] == "failed" and not done["artifacts"]


def test_spatial_plan_only_accepts_fixed_cli_and_grid_parameters(prepared, tmp_path):
    manager, _ = prepared
    script = manager.repo / "tools/layer_spatial_maps.py"
    script.parent.mkdir()
    script.write_text("# Synthetic CLI only\n")
    mask = tmp_path / "mask.npy"
    mask.write_bytes(b"synthetic source, not a model output")
    output = tmp_path / "new_spatial"
    plan = {"kind": "spatial", "output_dir": str(output),
            "files": [{"path": str(mask), "sha256": W._digest(mask)}],
            "source_binding": {}, "expected": {"manifest_path": str(output / "spatial_manifest.json")},
            "argv": [sys.executable, str(script), "--mask", str(mask), "--sample", "synthetic", "--output", str(output),
                     "--heatmap-grid", "8", "15", "--combined-grid", "6", "11"]}
    with patch.object(W.subprocess, "Popen", side_effect=AssertionError("No child")):
        ticket = manager.register_plan(plan)
    assert ticket["kind"] == "spatial" and not output.exists()
    plan["argv"] += ["--force"]
    with pytest.raises(ValueError, match="Unsupported"):
        manager.register_plan(plan)


def test_memory_journal_retains_preparation_execution_and_bound_output_refs(prepared):
    manager, result = prepared
    ticket, job, process, _ = launch_fake(manager, result)
    journal = manager.directory / "events.jsonl"
    before = journal.read_bytes()
    manager.state()
    assert journal.read_bytes() == before  # inspection/query is read-only
    complete(manager, ticket)
    finish_fake(manager, job, process)
    assert journal.read_bytes().startswith(before)
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["prepared", "start_requested", "started", "completed"]
    assert len({row["event_id"] for row in rows}) == 4
    assert all(row["ticket_id"] == ticket["id"] and row["input_binding_sha256"] for row in rows)
    assert rows[-1]["output_refs"] and rows[-1]["status"] == "completed"
    assert all("argv" not in row and "env" not in row and "config" not in row for row in rows)


def test_unwritable_memory_journal_blocks_process_launch(prepared):
    manager, result = prepared
    ticket = manager.register_prepared(result)
    # Replacing the journal with a directory reliably denies append even as a
    # privileged test user; no permissions or real source files are changed.
    journal = manager.directory / "events.jsonl"
    journal.unlink()
    journal.mkdir()
    with patch.object(W.subprocess, "Popen", side_effect=AssertionError("No child")) as child:
        job = manager.start(ticket["id"])
    assert child.call_count == 0
    assert job["status"] == "failed" and "Memory journal" in job["error"]


def test_different_scan_batches_keep_separate_public_source_identity(prepared, tmp_path):
    manager, result = prepared
    first = manager.register_prepared(result)
    new_bundle, new_raw = tmp_path / "second-bundle", tmp_path / "second-raw"
    new_bundle.mkdir()
    new_raw.mkdir()
    old_bundle = Path(result["bundle_dir"])
    preflight = W._read(result["preflight_path"])
    argv = [value.replace(str(old_bundle), str(new_bundle)) for value in preflight["argv"]]
    argv[argv.index("--data") + 1] = str(new_raw)
    preflight["argv"] = argv
    preflight["inspection"]["data_dir"] = str(new_raw)
    preflight["input_inventory"]["fingerprint"] = "b" * 64
    write(new_bundle / "argv.json", argv)
    write(new_bundle / "preflight.json", preflight)
    write(new_bundle / "profile.json", preflight["resolved"]["profile"])
    second = manager.register_prepared({"bundle_dir": str(new_bundle), "argv_path": str(new_bundle / "argv.json"),
                                        "preflight_path": str(new_bundle / "preflight.json"),
                                        "profile_path": str(new_bundle / "profile.json")})
    assert first["source_dir"] != second["source_dir"]
    assert first["source_binding"]["input_fingerprint"] != second["source_binding"]["input_fingerprint"]
    assert first["id"] != second["id"]
