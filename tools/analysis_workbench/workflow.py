"""Bound background jobs for the local DD workflow; no device or model imports.

Only server-created preparation records are registered. HTTP callers receive an
opaque ticket, never a command input. Inspection, registration and state queries
cannot launch a process. A successful exit still needs validated bound outputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import struct
import subprocess
import sys
import threading


REPO = Path(__file__).resolve().parents[2]
MAX_JSON = 32 * 1024 * 1024
_ACTIVE = {"starting", "running"}
_STITCH_VALUES = {"--data", "--work", "--out", "--stitch-profile", "--scale-div",
                  "--out-scale", "--input-preflight", "--colour-correction", "--layout"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _read(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_JSON:
        raise ValueError(f"Missing, linked or oversized workflow record: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Workflow input must be a regular file: {path}")
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _path(value):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("Workflow paths must be non-empty text")
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError("Workflow control paths cannot be symlinks")
    return path.resolve()


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _arguments(argv, values, switches=(), multiple=None):
    if not isinstance(argv, list) or len(argv) < 2 or len(argv) > 80 or any(
            not isinstance(value, str) or not value or "\x00" in value for value in argv):
        raise ValueError("A registered workflow needs a bounded generated argument list")
    if Path(argv[0]).expanduser().resolve() != Path(sys.executable).resolve():
        raise ValueError("Workflow must use the workbench Python executable")
    result, index = {}, 2
    while index < len(argv):
        name = argv[index]
        if name in result:
            raise ValueError(f"Repeated workflow argument: {name}")
        if name in switches:
            result[name], index = True, index + 1
        elif name in values or name in (multiple or {}):
            count = (multiple or {}).get(name, 1)
            if index + count >= len(argv):
                raise ValueError(f"Missing workflow argument value: {name}")
            result[name] = argv[index + 1] if count == 1 else argv[index + 1:index + 1 + count]
            index += count + 1
        else:
            raise ValueError(f"Unsupported workflow argument: {name}")
    return result


class DDWorkflow:
    def __init__(self, runtime, repo=REPO):
        self.repo = _path(repo)
        self.directory = _path(runtime) / "dd_workflow"
        (self.directory / "prepared").mkdir(parents=True, exist_ok=True)
        (self.directory / "jobs").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.prepared, self.jobs, self.processes = {}, {}, {}
        self.errors = []
        for path in sorted((self.directory / "prepared").glob("*.json"))[-1000:]:
            try:
                ticket = _read(path)
                if ticket.get("id") != path.stem:
                    raise ValueError("Prepared ticket ID differs from its filename")
                self.prepared[ticket["id"]] = ticket
            except (OSError, ValueError, AttributeError) as error:
                self.errors.append(str(error))
        for path in sorted((self.directory / "jobs").glob("*/job.json"))[-1000:]:
            try:
                job = _read(path)
                if job.get("id") != path.parent.name:
                    raise ValueError("Job ID differs from its directory")
                if job.get("status") in _ACTIVE:
                    job.update(status="attention", finished_at=_now(),
                               error="Workbench restarted while this job was active. Its outcome is unknown; no automatic replay was attempted.")
                    _atomic(path, job)
                    self._event("restart_attention", self.prepared.get(job.get("ticket_id")), job, strict=False)
                self.jobs[job["id"]] = job
            except (OSError, ValueError, AttributeError) as error:
                self.errors.append(str(error))

    def _ticket_path(self, ticket):
        return self.directory / "prepared" / (ticket["id"] + ".json")

    def _job_path(self, job):
        return self.directory / "jobs" / job["id"] / "job.json"

    def _files(self, paths):
        return [{"path": str(path), "sha256": _digest(path)} for path in sorted(set(map(_path, paths)), key=str)]

    def _event(self, event, ticket=None, job=None, *, strict=True):
        """Append a concise process-memory journal; never save config/env/keys."""
        ticket, job = ticket or {}, job or {}
        binding = ticket.get("expected_binding") or ticket.get("plan", {}).get("source_binding", {})
        binding_hash = (binding.get("fingerprint") if isinstance(binding, dict) else None) or hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        record = {"schema_version": 1, "event_id": secrets.token_hex(16), "timestamp_utc": _now(),
                  "event": event, "step": ticket.get("kind", job.get("kind")),
                  "ticket_id": ticket.get("id", job.get("ticket_id")), "job_id": job.get("id"),
                  "status": job.get("status", ticket.get("status")), "input_binding_sha256": binding_hash,
                  "input_refs": [item["path"] for item in ticket.get("files", [])[:50]],
                  "input_file_count": len(ticket.get("files", [])),
                  "output_dir": ticket.get("output_dir", job.get("output_dir")),
                  "output_refs": [item.get("path") for item in job.get("artifacts", [])[:50]],
                  "output_path": job.get("output_path"), "process_log": job.get("log_path"),
                  "source_dir": ticket.get("source_dir", job.get("source_dir")),
                  "return_code": job.get("return_code"), "error": str(job.get("error"))[:4000] if job.get("error") else None}
        try:
            with self.lock, (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, ValueError) as error:
            message = "DD workflow Memory journal could not be saved: " + str(error)
            self.errors.append(message)
            if job:
                job["memory_error"] = message
            if strict:
                raise ValueError(message) from error

    def _verify(self, ticket):
        for item in ticket["files"]:
            if _digest(item["path"]) != item["sha256"]:
                raise ValueError(f"Prepared input changed; inspect and prepare again: {item['path']}")

    def _public_ticket(self, ticket):
        return {key: _copy(ticket[key]) for key in (
            "id", "kind", "status", "bundle_dir", "output_dir", "created_at", "job_id",
            "input_count", "source_dir", "input_dir", "source_binding", "source_files") if key in ticket}

    def _save_ticket(self, ticket):
        fresh = ticket["id"] not in self.prepared
        _atomic(self._ticket_path(ticket), ticket)
        self.prepared[ticket["id"]] = ticket
        if fresh:
            self._event("prepared", ticket)
        return self._public_ticket(ticket)

    def register_prepared(self, result):
        """Register only the backend's stitch_setup.service.prepare result."""
        bundle = _path(result.get("bundle_dir"))
        paths = {name: _path(result.get(name)) for name in ("argv_path", "profile_path", "preflight_path")}
        expected_names = {"argv_path": "argv.json", "profile_path": "profile.json", "preflight_path": "preflight.json"}
        if any(path != bundle / expected_names[name] for name, path in paths.items()):
            raise ValueError("Prepared control files must belong to the registered bundle")
        argv = _read(paths["argv_path"])
        args = _arguments(argv, _STITCH_VALUES, ("--full", "--no-ai"))
        if _path(argv[1]) != self.repo / "run_stitch.py":
            raise ValueError("Stitch ticket does not select this repository's run_stitch.py")
        required = _STITCH_VALUES - {"--layout"}
        if not required <= args.keys() or args["--colour-correction"] != "auto":
            raise ValueError("Student stitching requires generated arguments with automatic colour correction")
        if (_path(args["--work"]) != bundle / "work" or _path(args["--out"]) != bundle / "mosaic.png"
                or _path(args["--stitch-profile"]) != paths["profile_path"]
                or _path(args["--input-preflight"]) != paths["preflight_path"]):
            raise ValueError("Stitch command escapes or disagrees with its generated bundle")
        preflight = _read(paths["preflight_path"])
        if preflight.get("status") != "profile_prepared" or preflight.get("argv") != argv:
            raise ValueError("Stitch arguments differ from the prepared preflight")
        resolved, inventory, inspection = preflight["resolved"], preflight["input_inventory"], preflight["inspection"]
        if resolved.get("colour_correction") != "auto" or _path(args["--data"]) != _path(inspection["data_dir"]):
            raise ValueError("Stitch source or colour policy differs from preparation")
        if _read(paths["profile_path"]) != resolved["profile"]:
            raise ValueError("Stitch profile differs from preparation")
        if (int(args["--scale-div"]) != resolved["scale_div"]
                or float(args["--out-scale"]) != resolved["out_scale"]
                or ("--full" in args) != resolved["full"]):
            raise ValueError("Stitch rendering scales differ from preparation")
        controls = list(paths.values())
        if "--layout" in args:
            layout = _path(args["--layout"])
            if layout != bundle / "layout.json" or _read(layout) != preflight.get("layout"):
                raise ValueError("Stitch layout differs from its preparation")
            controls.append(layout)
        layer_contract = bundle / "layer_input_contract.json"
        if layer_contract.is_file():
            controls.append(layer_contract)
        resolved = dict(resolved, grid_policy="preserve" if inspection["layout_mode"] == "flat-grid" else "legacy")
        # The same pure binding function is used by the child CLI. Import lazily;
        # opening the workbench itself does not load image or model libraries.
        profile_module = importlib.import_module("stitch_profile")
        expected = profile_module.make_run_binding(resolved, inventory, inspection.get("acquisition"))
        controls += [self.repo / name for name in ("run_stitch.py", "stitch_colour.py", "stitch_colour_policy.py",
                                                  "stitch_profile.py", "blend.py", "tiles.py", "register.py",
                                                  "tools/colour_match_grid.py", "kimi_agents.py", "focus_metrics.py")
                     if (self.repo / name).is_file()]
        ticket = {"id": secrets.token_hex(16), "kind": "stitch", "status": "prepared", "created_at": _now(),
                  "bundle_dir": str(bundle), "output_dir": str(bundle), "argv": argv,
                  "source_dir": str(_path(args["--data"])), "input_dir": str(_path(args["--data"])),
                  "source_binding": {"input_fingerprint": inventory["fingerprint"], "run_fingerprint": expected["fingerprint"],
                                     "acquisition_fingerprint": expected.get("acquisition_fingerprint")},
                  "files": self._files(controls), "input_count": inventory["tile_count"],
                  "expected_binding": expected, "expected_tids": [tile["tid"] for tile in inventory["tiles"]],
                  "expected_output": str(bundle / "mosaic.png"), "state_path": str(bundle / "work/state.json")}
        with self.lock:
            return self._save_ticket(ticket)

    def register_plan(self, plan):
        """Register a server-generated layer or spatial analysis plan, never body argv."""
        plan = _copy(plan)
        kind, argv, output = plan.get("kind"), plan.get("argv"), _path(plan.get("output_dir"))
        if kind == "layers":
            args = _arguments(argv, {"--config"}, ("--no-ai", "--no-repair"))
            if (_path(argv[1]) != self.repo / "flakepipeline/run.py" or set(args) != {"--config", "--no-ai", "--no-repair"}
                    or _path(args["--config"]) != output / "control/layers_config.json"):
                raise ValueError("Layer plan must use the fixed reviewed pipeline and generated config")
        elif kind == "spatial":
            args = _arguments(argv, {"--mask", "--valid-mask", "--exclusions", "--sample", "--output", "--mosaic"},
                              ("--auto-central-exclusion",),
                              multiple={"--heatmap-grid": 2, "--combined-grid": 2})
            if (_path(argv[1]) != self.repo / "tools/layer_spatial_maps.py" or _path(args.get("--output")) != output
                    or not {"--mask", "--sample", "--heatmap-grid", "--combined-grid"} <= args.keys()
                    or args["--heatmap-grid"] != ["8", "15"] or args["--combined-grid"] != ["6", "11"]):
                raise ValueError("Spatial plan must use the fixed reviewed grid-map command")
        else:
            raise ValueError("Unknown DD analysis plan")
        files = plan.get("files")
        if not isinstance(files, list) or not files or len(files) > 4096:
            raise ValueError("Analysis plan needs bounded source-file hashes")
        records = self._files([item["path"] for item in files] + [argv[1]])
        by_path = {item["path"]: item["sha256"] for item in records}
        if any(by_path[str(_path(item["path"]))] != item.get("sha256") for item in files):
            raise ValueError("Analysis sources changed during preparation")
        ticket = {"id": secrets.token_hex(16), "kind": kind, "status": "prepared", "created_at": _now(),
                  "output_dir": str(output), "argv": argv, "plan": plan, "files": records,
                  "source_binding": plan.get("source_binding", {}), "source_files": plan["files"]}
        with self.lock:
            return self._save_ticket(ticket)

    def start_stitch(self, ticket_id):
        return self.start(ticket_id)

    def start(self, ticket_id):
        """Launch one fixed argv in the background, once per prepared ticket."""
        with self.lock:
            if not isinstance(ticket_id, str) or ticket_id not in self.prepared:
                raise ValueError("Prepare this workflow step before starting it")
            ticket = self.prepared[ticket_id]
            if ticket.get("job_id") in self.jobs:
                return _copy(self.jobs[ticket["job_id"]])
            if any(job["status"] in _ACTIVE for job in self.jobs.values()):
                raise ValueError("Another DD workflow step is running; wait for it to finish")
            self._verify(ticket)
            if ticket["kind"] == "stitch" and (Path(ticket["expected_output"]).exists() or Path(ticket["state_path"]).exists()):
                raise ValueError("Prepared output already has a result; prepare a new run to preserve it")
            if ticket["kind"] == "stitch" and Path(ticket["state_path"]).parent.is_symlink():
                raise ValueError("Prepared work directory cannot be replaced with a symlink")
            job_id = secrets.token_hex(16)
            folder = self.directory / "jobs" / job_id
            folder.mkdir()
            job = {"id": job_id, "ticket_id": ticket_id, "kind": ticket["kind"], "status": "starting",
                   "output_dir": ticket["output_dir"], "created_at": _now(), "started_at": _now(),
                   "source_dir": ticket.get("source_dir"), "input_dir": ticket.get("input_dir"),
                   "source_binding": _copy(ticket.get("source_binding", {})),
                   "source_files": _copy(ticket.get("source_files", [])),
                   "finished_at": None, "log_path": str(folder / "process.log"), "output_path": None,
                   "error": None, "artifacts": [], "colour_status": "pending" if ticket["kind"] == "stitch" else None,
                   "render_verified": False, "progress": {"stage": "starting", "completed_steps": [], "detail": ""}}
            self.jobs[job_id] = job
            ticket.update(job_id=job_id, status="started")
            self._save_ticket(ticket)
            _atomic(self._job_path(job), job)
            try:
                self._event("start_requested", ticket, job)
                with Path(job["log_path"]).open("xb") as stream:
                    process = subprocess.Popen(ticket["argv"], cwd=str(self.repo), stdin=subprocess.DEVNULL,
                                               stdout=stream, stderr=subprocess.STDOUT, shell=False,
                                               env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"))
                self.processes[job_id] = process
                job.update(status="running", pid=process.pid)
                _atomic(self._job_path(job), job)
                self._event("started", ticket, job, strict=False)
                threading.Thread(target=self._watch, args=(job_id,), daemon=True,
                                 name="DD-" + job_id[:8]).start()
            except Exception as error:
                job.update(status="failed", error=str(error), finished_at=_now())
                _atomic(self._job_path(job), job)
                self._event("failed", ticket, job, strict=False)
            return _copy(job)

    def _progress(self, job, ticket):
        if ticket["kind"] != "stitch":
            expected = ticket["plan"].get("expected", {})
            completed = []
            if ticket["kind"] == "layers":
                for sample in expected.get("samples", [])[:100]:
                    try:
                        record = _read(sample["manifest_path"])
                        if record.get("sample") == sample.get("sample"):
                            completed.extend(sample["sample"] + ":" + name for name in record.get("stages", {}))
                    except (OSError, ValueError, TypeError, AttributeError):
                        pass
            job["progress"] = {"stage": completed[-1] if completed else "processing",
                               "completed_steps": completed[-30:],
                               "detail": "Intermediate outputs await final source and result validation."}
            return
        try:
            state = _read(ticket["state_path"])
            if state.get("run_binding") != ticket["expected_binding"]:
                return
            done = state.get("_done", [])
            if not isinstance(done, list):
                return
            colour = state.get("colour_correction", {})
            job.update(colour_status=colour.get("status", "pending"),
                       render_verified=colour.get("render_verified") is True,
                       progress={"stage": done[-1] if done else "processing", "completed_steps": done[-30:],
                                 "detail": colour.get("error", "")})
        except (OSError, ValueError, TypeError, AttributeError):
            # Child writes state.json between stages; transient partial reads are
            # pending evidence, never evidence of a successful result.
            pass

    def _validate_stitch(self, ticket):
        self._verify(ticket)
        state = _read(ticket["state_path"])
        if state.get("run_binding") != ticket["expected_binding"]:
            raise ValueError("Stitch result is not bound to the prepared input/profile")
        summary, render, colour = state.get("summary", {}), state.get("render", {}), state.get("colour_correction", {})
        selected = summary.get("tiles_selected")
        tids = render.get("rendered_tids", [])
        if (type(selected) is not int or selected < 1 or summary.get("tiles_used") != selected
                or render.get("n_placed") != selected or len(tids) != selected or len(set(tids)) != selected
                or summary.get("tiles_total") != ticket["input_count"]
                or any(render.get(key) for key in ("failed_load_tids", "missing_position_tids", "incomplete_tids", "missing_selected_tids"))
                or summary.get("render_failed") or render.get("n_failed_loads", 0)):
            raise ValueError("Stitch result does not verify complete selected-tile rendering")
        if ticket["expected_binding"].get("grid_policy") == "preserve" and set(tids) != set(ticket["expected_tids"]):
            raise ValueError("Protected-grid result is missing prepared source tiles")
        if (colour.get("render_verified") is not True or colour.get("status") not in ("applied_at_render", "verified_prior_correction")
                or colour.get("run_binding") != ticket["expected_binding"]
                or not {"render", "summary", "colour_correction"} <= set(state.get("_done", []))):
            raise ValueError("Stitch result has no verified automatic colour correction")
        diagnostic = Path(ticket["state_path"]).parent / "colour_correction.json"
        if _read(diagnostic) != colour:
            raise ValueError("Colour diagnostic and final stitch state disagree")
        output = _path(ticket["expected_output"])
        if _path(summary.get("out")) != output or _path(render.get("path")) != output or not output.is_file():
            raise ValueError("Expected stitched output is missing or the state names another result")
        with output.open("rb") as stream:
            header = stream.read(24)
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise ValueError("Stitched result is not a PNG image")
        if struct.unpack(">II", header[16:24]) != (render.get("out_w"), render.get("out_h")):
            raise ValueError("Stitched PNG dimensions disagree with the render record")
        from PIL import Image
        with Image.open(output) as image:
            image.verify()
        artifacts = [{"name": str(path.relative_to(ticket["output_dir"])), "path": str(path), "sha256": _digest(path)}
                     for path in (output, diagnostic, Path(ticket["state_path"]))]
        return {"output_path": str(output), "artifacts": artifacts,
                "colour_status": colour["status"], "render_verified": True,
                "coverage": {"selected": selected, "rendered": selected, "fraction": 1.0}}

    def _analysis_artifacts(self, ticket, result):
        """Normalise trusted validator output for the server's artifact endpoint."""
        output = _path(result.get("output_path"))
        root = Path(ticket["output_dir"])
        if not _inside(output, root) or not output.is_file():
            raise ValueError("Analysis result escaped or is missing from this job's output directory")
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts or len(artifacts) > 512:
            raise ValueError("Analysis validator returned an invalid output inventory")
        normalized, seen = [], set()
        for item in artifacts:
            path = _path(item.get("path") if isinstance(item, dict) else item)
            if not _inside(path, root) or path in seen or not path.is_file():
                raise ValueError("Analysis output is missing, duplicated or outside this job")
            seen.add(path)
            digest = _digest(path)
            if isinstance(item, dict) and item.get("sha256") not in (None, digest):
                raise ValueError("Analysis artifact changed after validation")
            normalized.append({"name": str(path.relative_to(root)), "path": str(path), "sha256": digest})
        return dict(result, output_path=str(output), artifacts=normalized)

    def _watch(self, job_id):
        process = self.processes[job_id]
        pause = threading.Event()
        while process.poll() is None:
            with self.lock:
                self._progress(self.jobs[job_id], self.prepared[self.jobs[job_id]["ticket_id"]])
            pause.wait(0.4)
        with self.lock:
            job = self.jobs[job_id]
            ticket = self.prepared[job["ticket_id"]]
            self._progress(job, ticket)
            job.update(return_code=process.returncode, finished_at=_now())
            try:
                if process.returncode != 0:
                    raise ValueError(f"Workflow command exited with code {process.returncode}; inspect its saved process log")
                if ticket["kind"] == "stitch":
                    result = self._validate_stitch(ticket)
                else:
                    self._verify(ticket)
                    # The server may be launched by absolute path from another
                    # cwd. Resolve its sibling helper without requiring cwd=root.
                    if str(self.repo) not in sys.path:
                        sys.path.insert(0, str(self.repo))
                    module = (__package__ + ".workflow_analysis") if __package__ else "workflow_analysis"
                    result = importlib.import_module(module).validate_plan_result(ticket["plan"])
                    result = self._analysis_artifacts(ticket, result)
                if not isinstance(result, dict) or not result.get("output_path"):
                    raise ValueError("Workflow result validator returned no confirmed output")
                job.update(result, status="completed", error=None)
                job["progress"]["stage"] = "completed"
            except Exception as error:
                job.update(status="failed", error=str(error), output_path=None, artifacts=[])
            _atomic(self._job_path(job), job)
            self._event("completed" if job["status"] == "completed" else "failed", ticket, job, strict=False)

    def state(self):
        """Snapshot current evidence only. Never infer success or launch a process."""
        with self.lock:
            for job in self.jobs.values():
                if job["status"] in _ACTIVE:
                    self._progress(job, self.prepared[job["ticket_id"]])
            jobs = sorted(self.jobs.values(), key=lambda item: item["created_at"], reverse=True)[:100]
            prepared = sorted(self.prepared.values(), key=lambda item: item["created_at"], reverse=True)[:100]
            return {"prepared": [self._public_ticket(ticket) for ticket in prepared], "jobs": _copy(jobs),
                    "active_job_id": next((job["id"] for job in jobs if job["status"] in _ACTIVE), None),
                    "updated_at": _now(), "errors": self.errors[-20:]}
