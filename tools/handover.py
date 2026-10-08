#!/usr/bin/env python3
"""Process ONE scan handover folder end to end; every step reads from and writes to the same folder.

The instrument computer packs a scan day with ``acquisition/Auto_Scan/pack_handover.py``
(runs copied intact, console logs, journals, camera and white-balance records,
``handover.json`` with hashes). This tool consumes that folder on the analysis computer::

    python tools/handover.py run <handover folder>          # all steps, done steps are skipped
    python tools/handover.py status <handover folder>
    python tools/handover.py run <folder> --from colour      # redo from a step onwards
    python tools/handover.py verify|assemble|incidents|colour|stitch|check|report <folder>

Steps and what they leave under ``<folder>/analysis/``:

1. ``verify``    - every packed file against ``handover.json``; tiles against the hashes the
                   scanner recorded; the scanner runtime files against this repository's
                   known manifest. ``verify.json``.
2. ``assemble``  - the runs that share one scan configuration are merged into one flat grid
                   by their original row/column numbers (a resumed run keeps the numbering):
                   ``flat_grid/`` holds hard links (or copies) of the chosen tiles with a
                   derived ``session.json`` / ``events.jsonl`` that record which run every tile
                   came from, duplicates, missing positions, and a registration check across
                   the seams where two runs meet. ``assemble.json``.
3. ``incidents`` - timeline of interruptions from the runs' events and the console transcripts
                   with elapsed time since each run started. ``incidents.csv`` / ``.md`` / ``.json``.
4. ``colour``    - ``tools/colour_match_grid.py`` on the flat grid: illumination field and
                   per-tile gains. ``colour_matched/`` (derived dataset, report, manifest).
5. ``stitch``    - ``run_stitch.py`` on the derived dataset (Kimi when configured).
                   ``colour_matched/mosaic.png``, ``mosaic_preview.jpg``, ``stitch_log.txt``.
6. ``check``     - the analysis colour check (reference substrate colour) on original and
                   corrected tiles of every run, plus the white-balance records. ``colour_check.json``.
7. ``report``    - ``REPORT_zh.md`` and ``report.json`` for the supervisor.

``handover_status.json`` links the steps: each records its inputs' fingerprint, outputs and
outcome; a step whose inputs did not change is skipped on the next run. Originals in
``runs/`` are never modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "tools", REPO / "acquisition" / "Auto_Scan"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

STEPS = ["verify", "assemble", "incidents", "colour", "stitch", "check", "report"]
SCHEMA_VERSION = 1
TILE_RE = re.compile(r"^mosaic_r(?P<row>\d+)_c(?P<col>\d+)\.(?:png|jpg|jpeg|tif|tiff|bmp)$", re.I)
INCIDENT_EVENTS = {"move_failed", "capture_failed", "acquisition_mode_mismatch", "resume", "log_resume_authorized",
                   "resume_camera_readback", "_unparseable_line"}
RC_LINE = re.compile(r"rc\s*=\s*-?\d+|Traceback|movement failed|失败|错误", re.I)
SESSION_LINE = re.compile(r"Photos will be saved to:\s*(.+)$")


class HandoverError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"event": "_unparseable_line", "line_number": n})
    return out


def parse_utc(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hard_link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


# ----------------------------------------------------------------------------
# context and status
# ----------------------------------------------------------------------------
class Context:
    def __init__(self, folder: Path, *, options: dict, log=print):
        self.folder = Path(folder).resolve()
        self.analysis = self.folder / "analysis"
        self.options = options
        self.log = log
        manifest_path = self.folder / "handover.json"
        if not manifest_path.is_file():
            raise HandoverError(f"{self.folder} has no handover.json; pack the scan with acquisition/Auto_Scan/pack_handover.py first")
        self.manifest = read_json(manifest_path)
        if self.manifest.get("kind") != "scan_handover":
            raise HandoverError("handover.json is not a scan_handover record")
        self.manifest_sha256 = sha256_file(manifest_path)
        self.analysis.mkdir(exist_ok=True)
        self.status_path = self.analysis / "handover_status.json"
        self.status = read_json(self.status_path) if self.status_path.is_file() else {
            "schema_version": SCHEMA_VERSION, "kind": "handover_status", "handover_sha256": self.manifest_sha256, "steps": {}}
        if self.status.get("handover_sha256") != self.manifest_sha256:
            self.log("[status] handover.json changed since the last run; all steps will run again")
            self.status = {"schema_version": SCHEMA_VERSION, "kind": "handover_status", "handover_sha256": self.manifest_sha256, "steps": {}}

    def save_status(self):
        self.status["updated_at_utc"] = now_utc()
        write_json(self.status_path, self.status)

    def result(self, step: str):
        rec = self.status["steps"].get(step)
        if not rec or rec.get("status") != "done":
            return None
        path = self.analysis / rec["result_file"]
        return read_json(path) if path.is_file() else None

    def runs(self) -> list[dict]:
        return sorted(self.manifest.get("runs", []), key=lambda r: (r.get("started_at_utc") or "", r["folder"]))

    def run_dir(self, run: dict) -> Path:
        return self.folder / run.get("handover_path", "runs/" + run["folder"])


def fingerprint(ctx: Context, step: str) -> str:
    parts = [ctx.manifest_sha256, step, json.dumps(ctx.options.get(step, {}), sort_keys=True)]
    if step == "verify":                    # a moved or re-copied folder is verified again
        parts.append(str(ctx.folder))
    idx = STEPS.index(step)
    if idx:
        prev = ctx.status["steps"].get(STEPS[idx - 1], {})
        parts.append(prev.get("finished_at_utc", ""))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# step 1: verify
# ----------------------------------------------------------------------------
def step_verify(ctx: Context) -> dict:
    m = ctx.manifest
    files = m.get("files", [])
    missing, mismatched, checked = [], [], 0
    t0 = time.time()
    for i, rec in enumerate(files, 1):
        p = ctx.folder / rec["path"]
        if not p.is_file():
            missing.append(rec["path"])
            continue
        if p.stat().st_size != rec["bytes"] or sha256_file(p) != rec["sha256"]:
            mismatched.append(rec["path"])
        checked += 1
        if i % 200 == 0:
            ctx.log(f"[verify] {i}/{len(files)} files ({time.time() - t0:.0f} s)")
    runtime = {"packed_check": m.get("scanner_runtime_check"), "matches_repository": None}
    repo_manifest = REPO / "acquisition" / "Auto_Scan" / "delivery_manifest.json"
    packed_manifest = ctx.folder / "scanner" / "delivery_manifest.json"
    if repo_manifest.is_file() and packed_manifest.is_file():
        ours = {f["path"]: f["sha256"] for f in read_json(repo_manifest).get("files", [])}
        theirs = {f["path"]: f["sha256"] for f in read_json(packed_manifest).get("files", [])}
        runtime["matches_repository"] = ours == theirs
        runtime["differing_files"] = sorted(k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k))
    runs = []
    for run in ctx.runs():
        d = ctx.run_dir(run)
        rec = {"folder": run["folder"], "present": d.is_dir(), "session_json": (d / "session.json").is_file(),
               "events_jsonl": (d / "events.jsonl").is_file()}
        if rec["events_jsonl"]:
            events = read_jsonl(d / "events.jsonl")
            captures = [e for e in events if e.get("event") == "capture_success" and e.get("image_sha256")]
            bad, absent = [], []
            for e in captures:
                name = e.get("filename") or Path(e.get("image_relative_path", "")).name
                p = d / name
                if not p.is_file():
                    absent.append(name)
                elif sha256_file(p) != e["image_sha256"]:
                    bad.append(name)
            rec.update(captures_recorded=len(captures), tiles_hash_mismatch=bad, tiles_missing=absent,
                       unparseable_lines=sum(1 for e in events if e.get("event") == "_unparseable_line"))
        runs.append(rec)
    ok = not missing and not mismatched and all(not r.get("tiles_hash_mismatch") and not r.get("tiles_missing") for r in runs)
    result = {"schema_version": SCHEMA_VERSION, "ok": ok, "files_listed": len(files), "files_checked": checked,
              "missing": missing, "mismatched": mismatched, "runs": runs, "scanner_runtime": runtime,
              "packed_at_utc": m.get("packed_at_utc"), "sample_id": m.get("sample_id"), "operator": m.get("operator")}
    ctx.log(f"[verify] {checked}/{len(files)} files OK, {len(missing)} missing, {len(mismatched)} mismatched; "
            f"runtime matches repository: {runtime['matches_repository']}")
    if not ok and not ctx.options.get("verify", {}).get("ignore"):
        raise HandoverError("Handover folder does not match its manifest (see analysis/verify.json); "
                            "re-copy the folder, or run with --ignore-verify to continue anyway")
    return result


# ----------------------------------------------------------------------------
# step 2: assemble
# ----------------------------------------------------------------------------
def _grid_key(run: dict) -> tuple:
    g = run.get("grid") or {}
    return (run.get("sample_id"), g.get("nx"), g.get("ny"), g.get("dx_steps"), g.get("dy_steps"), g.get("order"),
            g.get("invert_x"), g.get("invert_y"))


def _run_captures(ctx: Context, run: dict) -> dict:
    """(row, col) -> capture event (last capture_success wins within one run)."""
    d = ctx.run_dir(run)
    captures = {}
    if (d / "events.jsonl").is_file():
        for e in read_jsonl(d / "events.jsonl"):
            if e.get("event") == "capture_success" and e.get("row") is not None and e.get("col") is not None:
                name = e.get("filename") or Path(e.get("image_relative_path", "")).name
                if name and (d / name).is_file():
                    captures[(int(e["row"]), int(e["col"]))] = dict(e, _path=d / name)
    if not captures:                      # no usable journal: fall back to the files on disk
        for p in sorted(d.iterdir()):
            mt = TILE_RE.fullmatch(p.name)
            if mt:
                captures[(int(mt["row"]), int(mt["col"]))] = {"filename": p.name, "_path": p, "image_sha256": sha256_file(p),
                                                              "image_bytes": p.stat().st_size, "event_id": None}
    return captures


def step_assemble(ctx: Context) -> dict:
    runs = ctx.runs()
    if not runs:
        raise HandoverError("No runs in handover.json")
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        groups.setdefault(_grid_key(run), []).append(run)
    key = max(groups, key=lambda k: sum(r.get("tiles_on_disk", 0) for r in groups[k]))
    chosen = groups[key]
    skipped = [r["folder"] for k, rs in groups.items() if k != key for r in rs]
    prefer = ctx.options.get("assemble", {}).get("prefer", "latest")
    nx, ny = key[1], key[2]
    out = ctx.analysis / "flat_grid"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    selected: dict[tuple, dict] = {}
    duplicates = []
    for run in chosen:
        captures = _run_captures(ctx, run)
        for pos, cap in captures.items():
            if pos in selected:
                earlier = selected[pos]
                same = earlier["image_sha256"] == cap.get("image_sha256")
                duplicates.append({"row": pos[0], "col": pos[1], "runs": [earlier["_run"], run["folder"]], "identical": same,
                                   "kept": run["folder"] if prefer == "latest" else earlier["_run"]})
                if prefer != "latest":
                    continue
            selected[pos] = dict(cap, _run=run["folder"], _session_id=run.get("session_id"))
    missing = [[r, c] for c in range(nx or 0) for r in range(ny or 0) if (r, c) not in selected] if nx and ny else []
    tiles, methods = [], {}
    for pos in sorted(selected):
        cap = selected[pos]
        dst = out / f"mosaic_r{pos[0]}_c{pos[1]}{cap['_path'].suffix.lower()}"
        method = link_or_copy(cap["_path"], dst)
        methods[method] = methods.get(method, 0) + 1
        tiles.append({"row": pos[0], "col": pos[1], "filename": dst.name, "source_run": cap["_run"], "source_session_id": cap.get("_session_id"),
                      "source_event_id": cap.get("event_id"), "source_filename": cap["_path"].name,
                      "image_sha256": cap.get("image_sha256") or sha256_file(dst), "image_bytes": cap.get("image_bytes") or dst.stat().st_size,
                      "timestamp_utc": cap.get("timestamp_utc")})
    primary = chosen[0]
    primary_session = read_json(ctx.run_dir(primary) / "session.json") if (ctx.run_dir(primary) / "session.json").is_file() else {}
    session = dict(primary_session)
    session.update({"status": "assembled_from_runs", "photos_saved": len(tiles), "planned_positions": (nx or 0) * (ny or 0),
                    "positions_completed": len(tiles), "position_trusted": False,
                    "position_trust_basis": "assembled from several scan sessions by tools/handover.py; see assembled_from",
                    "assembled_from": [{"run": r["folder"], "session_id": r.get("session_id"), "status": r.get("status"),
                                        "tiles_used": sum(1 for t in tiles if t["source_run"] == r["folder"])} for r in chosen],
                    "assembled_at_utc": now_utc(), "duplicates_resolved": duplicates, "missing_positions": missing})
    events = [{"event": "session_started", "timestamp_utc": now_utc(), "session_id": session.get("session_id"),
               "event_id": hashlib.sha256(b"assembled-start").hexdigest()[:32], "assembled": True, "sequence": 1}]
    for i, t in enumerate(tiles, 2):
        events.append({"event": "capture_success", "timestamp_utc": t["timestamp_utc"] or now_utc(), "session_id": session.get("session_id"),
                       "event_id": hashlib.sha256(("assembled-" + t["filename"]).encode()).hexdigest()[:32], "sequence": i,
                       "row": t["row"], "col": t["col"], "filename": t["filename"], "image_relative_path": t["filename"],
                       "image_sha256": t["image_sha256"], "image_bytes": t["image_bytes"],
                       "assembled_from": {"run": t["source_run"], "session_id": t["source_session_id"], "event_id": t["source_event_id"],
                                          "filename": t["source_filename"]}})
    events.append({"event": "session_finished", "timestamp_utc": now_utc(), "session_id": session.get("session_id"), "status": "assembled_from_runs",
                   "photos_saved": len(tiles), "planned_positions": session["planned_positions"], "assembled": True,
                   "event_id": hashlib.sha256(b"assembled-finish").hexdigest()[:32], "sequence": len(events) + 1})
    session["event_count"] = len(events)
    write_json(out / "session.json", session)
    with (out / "events.jsonl").open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    # the stitcher's own acquisition check must accept the derived records
    acquisition = {"checked": False}
    try:
        import stitch_profile as SP
        from tools.stitch_setup import acquisition_metadata as am
        inventory = SP.inspect_tiles(SP.scan_grid_tiles(out))
        evidence, warnings = am.inspect_acquisition(out, inventory, "flat-grid")
        acquisition = {"checked": True, "present": evidence.get("present"), "checksum_verified": evidence.get("checksum_verified_selected_tiles"),
                       "warnings": warnings, "image_size": inventory.get("image_size")}
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        acquisition = {"checked": False, "error": str(exc)}
    seams = _cross_run_seams(ctx, out, tiles, missing)
    result = {"schema_version": SCHEMA_VERSION, "grid": {"nx": nx, "ny": ny, "sample_id": key[0], "dx_steps": key[3], "dy_steps": key[4],
                                                        "order": key[5], "invert_x": key[6], "invert_y": key[7]},
              "runs_assembled": [r["folder"] for r in chosen], "runs_skipped_other_grid": skipped, "prefer": prefer,
              "tiles": len(tiles), "complete": not missing, "missing_positions": missing, "duplicates": duplicates,
              "tiles_by_run": {r["folder"]: sum(1 for t in tiles if t["source_run"] == r["folder"]) for r in chosen},
              "link_method": methods, "flat_grid": str(out), "acquisition_check": acquisition, "cross_run_seams": seams,
              "tile_records": tiles}
    ctx.log(f"[assemble] {len(tiles)} tiles from {len(chosen)} run(s) -> flat_grid ({'complete' if not missing else f'{len(missing)} missing'}); "
            f"{len(duplicates)} duplicate position(s); cross-run seams: {seams.get('summary')}")
    return result


def _cross_run_seams(ctx: Context, out: Path, tiles: list[dict], missing: list) -> dict:
    """Register a few neighbouring pairs that come from different runs; a resumed run must attach where its numbers say."""
    try:
        import colour_match_grid as cm
    except ImportError as exc:
        return {"checked": False, "error": str(exc)}
    by_pos = {(t["row"], t["col"]): t for t in tiles}
    pairs = []
    for (r, c), t in by_pos.items():
        for other, kind in (((r, c + 1), "h"), ((r + 1, c), "v")):
            u = by_pos.get(other)
            if u and u["source_run"] != t["source_run"]:
                pairs.append((kind, (r, c), other))
    if not pairs:
        return {"checked": True, "pairs": 0, "summary": "single run" if len({t['source_run'] for t in tiles}) == 1 else "no adjacent pairs across runs"}
    try:
        inv = cm.inventory(out) if not missing else None
    except cm.ColourMatchError:
        inv = None
    direction = None
    if inv is not None:
        try:
            direction = cm.detect_direction(inv)
        except cm.ColourMatchError as exc:
            return {"checked": False, "pairs": len(pairs), "error": str(exc)}
    rng = np.random.default_rng(0)
    sample = [pairs[i] for i in sorted(rng.choice(len(pairs), min(8, len(pairs)), replace=False))]
    records = []
    for kind, a, b in sample:
        A, B = cm._load_rgb(out / by_pos[a]["filename"]), cm._load_rgb(out / by_pos[b]["filename"])
        side = (direction["next_column_side"] if direction else "left") if kind == "h" else "below"
        if kind == "h" and direction is None:
            best = max((cm.register(A, B, s) + (s,) for s in ("left", "right")), key=lambda x: x[2])
            dx, dy, score, side = best
        else:
            dx, dy, score = cm.register(A, B, side)
        expected = (direction["next_column_vector_dxdy"] if kind == "h" else direction["next_row_vector_dxdy"]) if direction else None
        dev = max(abs(dx - expected[0]) / A.shape[1], abs(dy - expected[1]) / A.shape[0]) if expected else None
        records.append({"kind": kind, "a": list(a), "b": list(b), "runs": [by_pos[a]["source_run"], by_pos[b]["source_run"]],
                        "side": side, "dx": dx, "dy": dy, "ncc": round(score, 3), "deviation_from_grid_vector": round(dev, 3) if dev is not None else None,
                        "ok": score >= 0.6 and (dev is None or dev <= 0.1)})
    n_ok = sum(1 for r in records if r["ok"])
    return {"checked": True, "pairs": len(pairs), "sampled": records, "summary": f"{n_ok}/{len(records)} sampled seams register at the expected offset",
            "ok": n_ok == len(records)}


# ----------------------------------------------------------------------------
# step 3: incidents
# ----------------------------------------------------------------------------
def step_incidents(ctx: Context) -> dict:
    rows = []
    runs_summary = []
    for run in ctx.runs():
        d = ctx.run_dir(run)
        events = read_jsonl(d / "events.jsonl") if (d / "events.jsonl").is_file() else []
        captures = [e for e in events if e.get("event") == "capture_success"]
        t_start = parse_utc(run.get("started_at_utc"))
        t_first = parse_utc(captures[0]["timestamp_utc"]) if captures else t_start
        planned = run.get("planned_positions") or ((run.get("grid") or {}).get("nx", 0) * (run.get("grid") or {}).get("ny", 0)) or None
        done_before = 0
        for e in events:
            if e.get("event") == "capture_success":
                done_before += 1
            if e.get("event") in INCIDENT_EVENTS or e.get("error"):
                t = parse_utc(e.get("timestamp_utc"))
                rows.append({"run": run["folder"], "source": "events.jsonl", "event": e.get("event"), "timestamp_utc": e.get("timestamp_utc"),
                             "elapsed_from_run_start_min": round((t - t_start).total_seconds() / 60, 1) if t and t_start else None,
                             "elapsed_from_first_capture_min": round((t - t_first).total_seconds() / 60, 1) if t and t_first else None,
                             "point": done_before, "planned": planned,
                             "fraction_of_run": round(done_before / planned, 3) if planned else None,
                             "row": e.get("row"), "col": e.get("col"), "error": e.get("error"), "detail": e.get("status") or e.get("next_point") or e.get("reason"),
                             "verification": e.get("verification")})
        runs_summary.append({"run": run["folder"], "status": run.get("status"), "started_at_utc": run.get("started_at_utc"),
                             "finished_at_utc": run.get("finished_at_utc"), "first_capture_utc": captures[0]["timestamp_utc"] if captures else None,
                             "last_capture_utc": captures[-1]["timestamp_utc"] if captures else None, "captures": len(captures), "planned": planned,
                             "duration_min": round((parse_utc(run.get("finished_at_utc")) - t_start).total_seconds() / 60, 1)
                             if t_start and parse_utc(run.get("finished_at_utc")) else None,
                             "camera": run.get("camera")})
    # restart gaps between consecutive runs
    gaps = []
    for a, b in zip(runs_summary, runs_summary[1:]):
        ta, tb = parse_utc(a.get("last_capture_utc") or a.get("finished_at_utc")), parse_utc(b.get("first_capture_utc") or b.get("started_at_utc"))
        if ta and tb:
            gaps.append({"from_run": a["run"], "to_run": b["run"], "gap_min": round((tb - ta).total_seconds() / 60, 1)})
    # console transcripts
    logs = []
    log_dir = ctx.folder / "console_logs"
    run_dirs = {run["folder"]: run for run in ctx.runs()}
    if log_dir.is_dir():
        for p in sorted(log_dir.glob("*.txt")):
            text = p.read_text(encoding="utf-8", errors="replace")
            attributed = None
            for line in text.splitlines():
                m = SESSION_LINE.search(line)
                if m:
                    for stamp in run_dirs:
                        if stamp in m.group(1):
                            attributed = stamp
            flagged = [{"line": i, "text": line.strip()[:200]} for i, line in enumerate(text.splitlines(), 1) if RC_LINE.search(line)]
            logs.append({"file": p.name, "run": attributed, "lines": len(text.splitlines()), "flagged": flagged})
            for f in flagged:
                rows.append({"run": attributed, "source": p.name, "event": "console", "timestamp_utc": None, "elapsed_from_run_start_min": None,
                             "elapsed_from_first_capture_min": None, "point": None, "planned": None, "fraction_of_run": None,
                             "row": None, "col": None, "error": f["text"], "detail": f"line {f['line']}", "verification": None})
    result = {"schema_version": SCHEMA_VERSION, "runs": runs_summary, "restart_gaps": gaps, "incidents": rows, "console_logs": logs,
              "missing_console_logs": [r["folder"] for r in ctx.runs() if not any(l["run"] == r["folder"] for l in logs)]}
    with (ctx.analysis / "incidents.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "source", "event", "timestamp_utc", "elapsed_from_run_start_min", "elapsed_from_first_capture_min",
                                          "point", "planned", "fraction_of_run", "row", "col", "error", "detail"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    (ctx.analysis / "incidents_zh.md").write_text(_incidents_md(result), encoding="utf-8")
    fatal = [r for r in rows if r["event"] in ("move_failed", "capture_failed", "acquisition_mode_mismatch")]
    ctx.log(f"[incidents] {len(rows)} records ({len(fatal)} fatal) across {len(runs_summary)} run(s); console logs: {len(logs)}")
    return result


def _incidents_md(res: dict) -> str:
    lines = ["# 扫描中断与错误时间线", ""]
    lines.append("| 轮次 | 状态 | 开始 (UTC) | 结束 (UTC) | 拍到 / 计划 | 用时 (min) | 相机增益 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in res["runs"]:
        lines.append(f"| {r['run']} | {r['status']} | {r['started_at_utc']} | {r['finished_at_utc']} | {r['captures']} / {r['planned']} | "
                     f"{r['duration_min']} | {_readback((r.get('camera') or {}).get('gain'))} |")
    if res["restart_gaps"]:
        lines += ["", "重启间隔：" + "；".join(f"{g['from_run']} → {g['to_run']}: {g['gap_min']} min" for g in res["restart_gaps"])]
    lines += ["", "| 轮次 | 来源 | 事件 | 距开始 (min) | 第几点 / 计划 | 行,列 | 内容 |", "|---|---|---|---|---|---|---|"]
    for i in res["incidents"]:
        lines.append(f"| {i['run']} | {i['source']} | {i['event']} | {i['elapsed_from_first_capture_min']} | {i['point']} / {i['planned']} | "
                     f"{i['row']},{i['col']} | {(i['error'] or i['detail'] or '')} |")
    if res["missing_console_logs"]:
        lines += ["", "**缺少控制台记录的轮次**：" + ", ".join(res["missing_console_logs"]) + "（请学生补交 history/console_logs/launch_*.txt）"]
    lines += ["", "读取 `rc=-1` 的重试与致命的移动命令 `rc=-1` 要分开看：前者程序自动恢复，后者按安全策略停止。是否与运行时长有关，看“距开始”一列是否集中在后段。"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# step 4-5: colour and stitch
# ----------------------------------------------------------------------------
def step_colour(ctx: Context) -> dict:
    assembled = ctx.result("assemble")
    if not assembled:
        raise HandoverError("assemble step has not run")
    if not assembled["complete"]:
        return {"schema_version": SCHEMA_VERSION, "skipped": True,
                "reason": f"flat grid incomplete ({len(assembled['missing_positions'])} missing positions); colour matching needs every position"}
    import colour_match_grid as cm
    opts = ctx.options.get("colour", {})
    out = ctx.analysis / "colour_matched"
    if out.exists():
        shutil.rmtree(out)
    argv = ["--data", str(ctx.analysis / "flat_grid"), "--out", str(out), "--workers", str(opts.get("workers", 4)),
            "--scale-div", str(opts.get("scale_div", 4)), "--out-scale", str(opts.get("out_scale", 0.25))]
    if opts.get("no_flat_field"):
        argv.append("--no-flat-field")
    if opts.get("gain_mode"):
        argv += ["--gain-mode", opts["gain_mode"]]
    if opts.get("no_ai"):
        argv.append("--no-ai")
    if opts.get("full"):
        argv.append("--full")
    log_path = ctx.analysis / "colour_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        code = _run_with_log(cm, argv, f)
    if code != 0 or not (out / "colour_match_report.json").is_file():
        raise HandoverError("colour matching failed; see analysis/colour_log.txt")
    report = read_json(out / "colour_match_report.json")
    gains = report["gains"]
    result = {"schema_version": SCHEMA_VERSION, "skipped": False, "dataset": str(out), "argv": argv,
              "flat_field": report.get("flat_field"), "edges_used": gains.get("edges_used"),
              "edge_rms_before": gains.get("edge_rms_before"), "edge_rms_after": gains.get("edge_rms_after"),
              "column_steps": gains.get("column_steps"), "column_level_gain_rgb": gains.get("column_level_gain_rgb"),
              "reference_segment": gains.get("reference_segment"), "tile_deviation_max_abs": gains.get("tile_deviation_max_abs"),
              "flagged_tiles": gains.get("flagged_tiles"), "stitch_command_file": str(out / "stitch_command.txt")}
    ctx.log(f"[colour] steps {[(s['col_a'], s['col_b']) for s in gains.get('column_steps', [])]}, edge RMS {gains.get('edge_rms_before')} -> {gains.get('edge_rms_after')}")
    return result


def _run_with_log(module, argv, log_file) -> int:
    """Run module.main(argv) with every print copied into log_file as well as the console."""
    import builtins
    original = builtins.print

    def logged_print(*args, **kwargs):
        line = " ".join(str(a) for a in args)
        log_file.write(line + "\n")
        log_file.flush()
        original(line, flush=True)
    builtins.print = logged_print
    try:
        return int(module.main(argv))
    finally:
        builtins.print = original


def step_stitch(ctx: Context) -> dict:
    colour = ctx.result("colour")
    if not colour:
        raise HandoverError("colour step has not run")
    if colour.get("skipped"):
        return {"schema_version": SCHEMA_VERSION, "skipped": True, "reason": "no colour-matched dataset: " + colour.get("reason", "")}
    import colour_match_grid as cm
    out = Path(colour["dataset"])
    opts = ctx.options.get("stitch", {})
    cmd = cm.stitch_command(out, out / "stitch_profile.json", no_ai=bool(opts.get("no_ai")), workers=int(opts.get("workers", 4)),
                            extra=["--full"] if opts.get("full") else None)
    if opts.get("dry_run"):
        return {"schema_version": SCHEMA_VERSION, "skipped": True, "reason": "dry run", "command": cmd}
    log_path = ctx.analysis / "stitch_log.txt"
    ctx.log("[stitch] " + " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as f:
        code = subprocess.call(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    mosaic = out / "mosaic.png"
    if code != 0 or not mosaic.is_file():
        raise HandoverError(f"run_stitch.py exited with {code}; see analysis/stitch_log.txt")
    state = read_json(out / "_stitch_work" / "state.json") if (out / "_stitch_work" / "state.json").is_file() else {}
    summary = state.get("summary", {})
    diag = (state.get("register") or {}).get("diag", {})
    preview = _preview(mosaic, ctx.analysis / "mosaic_preview.jpg", 8)
    placement = _placement_check(out / "_stitch_work", ctx.result("assemble") or {})
    result = {"schema_version": SCHEMA_VERSION, "skipped": False, "command": cmd, "mosaic": str(mosaic), "mosaic_sha256": sha256_file(mosaic),
              "placement_check": placement,
              "preview": str(preview), "tiles_total": summary.get("tiles_total"), "tiles_used": summary.get("tiles_used"),
              "dropped": summary.get("dropped"), "rescued": len(summary.get("rescued") or []), "residual_rms_px": diag.get("residual_rms"),
              "n_components": diag.get("n_components"), "seconds": summary.get("seconds"), "kimi": summary.get("kimi"),
              "seam_inspection": state.get("inspect"), "render": state.get("render")}
    ctx.log(f"[stitch] {summary.get('tiles_used')}/{summary.get('tiles_total')} tiles, residual {diag.get('residual_rms')} px, {summary.get('seconds')} s")
    return result


def _placement_check(work: Path, assembled: dict, *, limit_frac: float = 0.25) -> dict:
    """Solved tile positions against a rigid grid model (origin + col*h + row*v): a tile placed rows away
    from where the stage numbering puts it shows up here without opening the mosaic."""
    pos_path, tid_path = work / "positions.npy", work / "positions_tids.json"
    if not pos_path.is_file():
        return {"checked": False, "reason": "positions.npy missing"}
    pos = np.load(pos_path)
    grid = assembled.get("grid") or {}
    nx, ny = grid.get("nx"), grid.get("ny")
    if tid_path.is_file():
        tids = json.loads(tid_path.read_text(encoding="utf-8"))
        rc = [re.search(r"mosaic_r(\d+)_c(\d+)", t) for t in tids]
        if len(tids) != len(pos) or any(m is None for m in rc):
            return {"checked": False, "reason": "tile ids do not match positions"}
        rows = np.array([int(m.group(1)) for m in rc]); cols = np.array([int(m.group(2)) for m in rc])
        names = [t.split("/")[-1] for t in tids]
        order = "positions_tids.json"
    elif nx and ny and len(pos) == nx * ny:
        cols = np.repeat(np.arange(nx), ny); rows = np.tile(np.arange(ny), nx)          # scan_dataset order: column-major
        names = [f"mosaic_r{r}_c{c}.png" for r, c in zip(rows, cols)]
        order = "assumed column-major (complete grid, no tile ids saved)"
    else:
        return {"checked": False, "reason": "no tile ids and the grid is not complete"}
    A = np.stack([np.ones(len(pos)), cols, rows], axis=1).astype(float)
    coef, *_ = np.linalg.lstsq(A, pos, rcond=None)
    res = pos - A @ coef
    dev = np.hypot(res[:, 0], res[:, 1])
    tile_w, tile_h = 1920, 1080
    sample_w = (assembled.get("acquisition_check") or {}).get("image_size")
    if sample_w and len(sample_w) == 2:
        tile_w, tile_h = sample_w
    limit = limit_frac * min(tile_w, tile_h)
    bad = np.flatnonzero(dev > limit)
    outliers = [{"tile": names[i], "stitched_col": int(cols[i]), "row": int(rows[i]), "deviation_px": round(float(dev[i]), 1),
                 "dx": round(float(res[i, 0]), 1), "dy": round(float(res[i, 1]), 1), "rows_off": round(float(res[i, 1] / coef[2][1]), 2) if coef[2][1] else None}
                for i in sorted(bad, key=lambda i: -dev[i])]
    return {"checked": True, "order": order, "tiles": int(len(pos)), "grid_model": {"origin": coef[0].round(1).tolist(), "h": coef[1].round(1).tolist(), "v": coef[2].round(1).tolist()},
            "deviation_px": {"median": round(float(np.median(dev)), 1), "p95": round(float(np.percentile(dev, 95)), 1), "max": round(float(dev.max()), 1)},
            "limit_px": round(limit, 1), "outliers": outliers, "ok": len(bad) == 0}


def _preview(mosaic: Path, target: Path, div: int) -> Path:
    import cv2
    m = cv2.imread(str(mosaic), cv2.IMREAD_COLOR)
    if m is None:
        raise HandoverError(f"cannot decode {mosaic}")
    small = cv2.resize(m, (max(1, m.shape[1] // div), max(1, m.shape[0] // div)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(target), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return target


# ----------------------------------------------------------------------------
# step 6: colour check
# ----------------------------------------------------------------------------
def step_check(ctx: Context) -> dict:
    from flakepipeline import color_diagnostics as cd
    reference = cd.load_reference(REPO / "configs" / "reference_substrate_colour.json")
    assembled = ctx.result("assemble") or {}
    colour = ctx.result("colour") or {}
    corrected_dir = Path(colour["dataset"]) if colour and not colour.get("skipped") else None
    manifest = read_json(corrected_dir / "colour_match_manifest.json") if corrected_dir and (corrected_dir / "colour_match_manifest.json").is_file() else None
    corrected_name = {}
    if manifest:
        for t in manifest["tiles"]:
            corrected_name[t["source_filename"]] = t["filename"]
    per_run = []
    sample_n = int(ctx.options.get("check", {}).get("samples", 12))
    for run in ctx.runs():
        d = ctx.run_dir(run)
        tiles = sorted(p for p in d.iterdir() if TILE_RE.fullmatch(p.name))
        if not tiles:
            per_run.append({"run": run["folder"], "checked": 0})
            continue
        idx = np.linspace(0, len(tiles) - 1, min(sample_n, len(tiles))).round().astype(int)
        originals, corrected = [], []
        for i in sorted(set(idx.tolist())):
            p = tiles[i]
            rec = cd.colour_check(p, reference)
            originals.append({"tile": p.name, "verdict": rec.get("verdict"), "gain_rgb": rec.get("gain_rgb"),
                              "observed_median_rgb": rec["observed"]["median_rgb"], "method": rec["observed"]["method"]})
            if corrected_dir and p.name in corrected_name and (corrected_dir / corrected_name[p.name]).is_file():
                rec2 = cd.colour_check(corrected_dir / corrected_name[p.name], reference)
                corrected.append({"tile": corrected_name[p.name], "verdict": rec2.get("verdict"), "gain_rgb": rec2.get("gain_rgb"),
                                  "observed_median_rgb": rec2["observed"]["median_rgb"]})
        def split_gain(items):
            gains = np.array([o["gain_rgb"] for o in items if o.get("gain_rgb")], np.float64)
            if not len(gains):
                return None
            med = np.median(gains, axis=0)
            brightness = float(np.exp(np.mean(np.log(med))))                 # common factor of the three channels
            balance = med / brightness
            return {"median_gain_rgb": med.round(4).tolist(), "brightness_gain": round(brightness, 4),
                    "balance_gain_rgb": balance.round(4).tolist(), "balance_max_abs_deviation": round(float(np.max(np.abs(balance - 1))), 4)}
        per_run.append({"run": run["folder"], "checked": len(originals), "original": cd.summarize([{"verdict": o["verdict"], "gain_rgb": o["gain_rgb"]} for o in originals]),
                        "original_split": split_gain(originals), "original_tiles": originals,
                        "corrected": cd.summarize([{"verdict": o["verdict"], "gain_rgb": o["gain_rgb"]} for o in corrected]) if corrected else None,
                        "corrected_split": split_gain(corrected) if corrected else None, "corrected_tiles": corrected})
    calibrations = ctx.manifest.get("colour_calibration") or []
    result = {"schema_version": SCHEMA_VERSION, "reference": {"mean_rgb": reference["mean_rgb"], "tolerance_fraction": reference["tolerance_fraction"]} if reference else None,
              "runs": per_run, "white_balance_records": calibrations,
              "note": "Verdicts compare the bare-substrate colour with the reference of the capture mode the layer model was reviewed on; "
                      "a corrected tile within tolerance is still a diagnostic preview, not a validated measurement."}
    ctx.log("[check] " + "; ".join(f"{r['run']}: {(r.get('original') or {}).get('verdict')}" for r in per_run if r.get("checked")))
    return result


# ----------------------------------------------------------------------------
# step 7: report
# ----------------------------------------------------------------------------
def _readback(value):
    """A camera readback as a short string: plain numbers as they are, {'value','status'} records by their value."""
    if isinstance(value, dict):
        if value.get("value") is None:
            return value.get("status", "unavailable")
        return f"{value['value']} ({value.get('status', '')})".replace(" ()", "")
    return value


def _fmt(value, nd=4):
    """Round floats (also inside lists / dicts) for the human report."""
    if isinstance(value, float):
        return round(value, nd)
    if isinstance(value, (list, tuple)):
        return [_fmt(v, nd) for v in value]
    if isinstance(value, dict):
        return {k: _fmt(v, nd) for k, v in value.items()}
    return value


def step_report(ctx: Context) -> dict:
    m = ctx.manifest
    verify, assembled, incidents = ctx.result("verify") or {}, ctx.result("assemble") or {}, ctx.result("incidents") or {}
    colour, stitch, check = ctx.result("colour") or {}, ctx.result("stitch") or {}, ctx.result("check") or {}
    L = []
    L += [f"# 扫描交接分析报告：{m.get('sample_id')}", "",
          f"打包：{m.get('packed_at_utc')}，操作者：{m.get('operator') or '未填'}，仪器电脑：{(m.get('instrument') or {}).get('hostname')}。"
          f"备注：{m.get('notes') or '无'}。分析：{now_utc()}。", ""]
    L += ["## 1. 资料核验", ""]
    if verify:
        rt = verify.get("scanner_runtime") or {}
        L.append(f"- 文件：{verify.get('files_checked')}/{verify.get('files_listed')} 个与清单一致，缺失 {len(verify.get('missing', []))}，不一致 {len(verify.get('mismatched', []))}。")
        pc = rt.get("packed_check") or {}
        detail = ""
        if pc and not pc.get("ok"):
            detail = f"（清单 {pc.get('release')}；与清单不符 {pc.get('mismatched')}，缺 {len(pc.get('missing') or [])} 个）"
        L.append(f"- 扫描程序运行文件（{pc.get('files_checked', '?')} 个）：仪器端自检 {'通过' if pc.get('ok') else '未通过'}{detail}；"
                 f"与仓库的 954-1080p-A1 记录{'一致' if rt.get('matches_repository') else '不一致'}"
                 + (f"（不同的文件 {len(rt.get('differing_files') or [])} 个）" if rt.get('differing_files') else "") + "。")
        for r in verify.get("runs", []):
            L.append(f"- 轮次 {r['folder']}：记录 {r.get('captures_recorded')} 张，哈希不符 {len(r.get('tiles_hash_mismatch', []))}，缺图 {len(r.get('tiles_missing', []))}。")
    L += ["", "## 2. 轮次与中断", ""]
    for r in incidents.get("runs", []):
        L.append(f"- {r['run']}：{r['status']}，{r['captures']}/{r['planned']} 张，{r['duration_min']} min，相机增益 {_readback((r.get('camera') or {}).get('gain'))}。")
    for g in incidents.get("restart_gaps", []):
        L.append(f"- 重启间隔 {g['from_run']} → {g['to_run']}：{g['gap_min']} min。")
    fatal = [i for i in incidents.get("incidents", []) if i["event"] in ("move_failed", "capture_failed", "acquisition_mode_mismatch")]
    for i in fatal:
        L.append(f"- **{i['event']}** 在 {i['run']} 第 {i['point']}/{i['planned']} 点（{i['fraction_of_run']}，距开始 {i['elapsed_from_first_capture_min']} min，行 {i['row']} 列 {i['col']}）：{i['error']}")
    if incidents.get("missing_console_logs"):
        L.append(f"- 缺少控制台记录：{', '.join(incidents['missing_console_logs'])}。")
    L.append("- 详细时间线：`analysis/incidents_zh.md`、`analysis/incidents.csv`。")
    L += ["", "## 3. 多轮拼合", ""]
    if assembled:
        g = assembled["grid"]
        L.append(f"- 网格 {g['nx']}×{g['ny']}（invert_x={g['invert_x']}），{assembled['tiles']} 张来自 {len(assembled['runs_assembled'])} 轮：{assembled['tiles_by_run']}。")
        L.append(f"- {'完整' if assembled['complete'] else '缺 ' + str(len(assembled['missing_positions'])) + ' 个位置：' + str(assembled['missing_positions'][:20])}；重复位置 {len(assembled['duplicates'])}（保留 {assembled['prefer']}）。")
        seams = assembled.get("cross_run_seams") or {}
        L.append(f"- 跨轮接缝配准：{seams.get('summary')}。" + ("" if seams.get("ok", True) else
                 " 没对上的样本通常是重叠区没有可配准的纹理（整片裸衬底），或续扫轮次的位置比原轮次偏了一些；拼接程序用全局配准处理这些，"
                 "看第 5 节的配准残差。若续扫列整体错位，请核对两轮的起点。"))
        if assembled.get("runs_skipped_other_grid"):
            L.append(f"- 扫描参数不同、未拼合的轮次：{assembled['runs_skipped_other_grid']}。")
    L += ["", "## 4. 颜色与照明校正", ""]
    if colour.get("skipped"):
        L.append(f"- 跳过：{colour.get('reason')}")
    elif colour:
        ff = colour.get("flat_field") or {}
        L.append(f"- 照明场：左/右 {_fmt(ff.get('edge_to_edge_ratio_left_over_right'))}，上/下 {_fmt(ff.get('edge_to_edge_ratio_top_over_bottom'))}。")
        L.append(f"- 相邻图失配（log RMS）：{colour.get('edge_rms_before')} → {colour.get('edge_rms_after')}，{colour.get('edges_used')} 条边；单张偏差最大 {colour.get('tile_deviation_max_abs')}。")
        L.append(f"- 列台阶：{[(s['col_a'], s['col_b'], _fmt(s['relative_gain_b_over_a'])) for s in colour.get('column_steps', [])]}；基准列 {colour.get('reference_segment')}。")
        L.append("- 列水平增益：" + "；".join(f"c{c} {_fmt(v, 3)}" for c, v in (colour.get("column_level_gain_rgb") or {}).items()))
    L += ["", "## 5. 拼接", ""]
    if stitch.get("skipped"):
        L.append(f"- 跳过：{stitch.get('reason')}")
    elif stitch:
        L.append(f"- {stitch.get('tiles_used')}/{stitch.get('tiles_total')} 张参与，丢弃 {len(stitch.get('dropped') or [])}，救回 {stitch.get('rescued')}，配准残差 RMS {_fmt(stitch.get('residual_rms_px'), 2)} px，连通块 {stitch.get('n_components')}，{_fmt(stitch.get('seconds'), 0)} s。")
        L.append(f"- Kimi：{stitch.get('kimi')}。")
        pc = stitch.get("placement_check") or {}
        if not pc and stitch.get("mosaic"):
            pc = _placement_check(Path(stitch["mosaic"]).parent / "_stitch_work", assembled)
        if pc.get("checked"):
            L.append(f"- 位置核对（与刚性网格模型比）：偏差中位 {pc['deviation_px']['median']} px，95 分位 {pc['deviation_px']['p95']} px，最大 {pc['deviation_px']['max']} px；"
                     + ("**没有**超过 {} px 的瓦片。".format(pc['limit_px']) if pc.get("ok") else
                        f"**{len(pc['outliers'])} 张超过 {pc['limit_px']} px**：" + "；".join(f"{o['tile']} 偏 {o['deviation_px']} px（约 {o['rows_off']} 行）" for o in pc['outliers'][:6])
                        + "——这些瓦片在拼接图上会错位、留洞或出现半透明重影，请看预览图相应位置。"))
        L.append(f"- 拼接图：`{Path(stitch['mosaic']).relative_to(ctx.folder)}`（sha256 {stitch.get('mosaic_sha256', '')[:12]}…），预览 `analysis/mosaic_preview.jpg`。")
    L += ["", "## 6. 颜色检查（参考衬底颜色）", ""]
    if check:
        ref = check.get("reference") or {}
        L.append(f"- 参考 {ref.get('mean_rgb')} ± {ref.get('tolerance_fraction')}。")
        for r in check.get("runs", []):
            o, c = r.get("original") or {}, r.get("corrected") or {}
            L.append(f"- {r['run']}：原图 {o.get('verdict')}" + (f"，校正后 {c.get('verdict')}" if c else "") + f"（抽查 {r.get('checked')} 张）。")
            sp = r.get("original_split")
            if sp:
                tol = (check.get("reference") or {}).get("tolerance_fraction")
                g = sp["brightness_gain"]
                pct = (1 - 1 / g) * 100 if g > 1 else (1 / g - 1) * 100
                L.append(f"  - 拆开看：整体亮度需 ×{g}（裸衬底比参考{'暗' if g > 1 else '亮'} {pct:.0f} %），"
                         f"色彩平衡 {sp['balance_gain_rgb']}，平衡偏差 {sp['balance_max_abs_deviation'] * 100:.1f} %"
                         + (f"（{'在' if tol and sp['balance_max_abs_deviation'] <= tol else '超出'} {tol * 100:.0f} % 容差）" if tol else "") + "。")
        for w in check.get("white_balance_records", []):
            L.append(f"- 白平衡记录 {w.get('folder')}：calibrated={w.get('calibrated')}，{w.get('verdict')}，读数 {w.get('observed_mean_rgb')}。")
    L += ["", "## 7. 文件", "",
          "- `analysis/flat_grid/`：拼合后的平铺网格（硬链接，原图不动）；`analysis/colour_matched/`：校正后的派生数据集、报告、清单、拼接图；",
          "- `analysis/verify.json`、`assemble.json`、`incidents.*`、`colour_check.json`、`stitch_log.txt`、`handover_status.json`。",
          "", "## 8. 注意", "",
          "- 校正与拼接用于目视和验收；层数识别请对原图或校正后的瓦片单独跑并看颜色检查结论；增益不是相机标定。",
          "- 原始轮次目录 `runs/` 未被修改；要重做某一步：`python tools/handover.py run <文件夹> --from <步骤>`。"]
    text = "\n".join(L) + "\n"
    (ctx.analysis / "REPORT_zh.md").write_text(text, encoding="utf-8")
    result = {"schema_version": SCHEMA_VERSION, "report_md": "analysis/REPORT_zh.md", "verify_ok": verify.get("ok"),
              "runs": [r["folder"] for r in ctx.runs()], "tiles": assembled.get("tiles"), "complete": assembled.get("complete"),
              "fatal_incidents": len(fatal), "colour_skipped": colour.get("skipped"), "stitch_skipped": stitch.get("skipped"),
              "mosaic": stitch.get("mosaic"), "colour_check": {r["run"]: (r.get("original") or {}).get("verdict") for r in check.get("runs", [])}}
    ctx.log("[report] analysis/REPORT_zh.md")
    return result


STEP_FUNCTIONS = {"verify": step_verify, "assemble": step_assemble, "incidents": step_incidents, "colour": step_colour,
                  "stitch": step_stitch, "check": step_check, "report": step_report}
RESULT_FILES = {"check": "colour_check.json"}


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------
def run_step(ctx: Context, step: str, *, force: bool = False) -> dict:
    fp = fingerprint(ctx, step)
    rec = ctx.status["steps"].get(step)
    if rec and rec.get("status") == "done" and rec.get("fingerprint") == fp and not force:
        ctx.log(f"[{step}] already done ({rec.get('finished_at_utc')}); skipped")
        return rec
    rec = {"status": "running", "started_at_utc": now_utc(), "fingerprint": fp, "result_file": RESULT_FILES.get(step, f"{step}.json")}
    ctx.status["steps"][step] = rec
    ctx.save_status()
    try:
        result = STEP_FUNCTIONS[step](ctx)
    except Exception as exc:
        rec.update(status="failed", finished_at_utc=now_utc(), error=str(exc))
        stale = ctx.analysis / rec["result_file"]
        if stale.exists():
            stale.unlink()
        ctx.save_status()
        raise
    write_json(ctx.analysis / rec["result_file"], result)
    rec.update(status="done", finished_at_utc=now_utc(), skipped=bool(result.get("skipped")))
    ctx.save_status()
    return rec


def run_all(ctx: Context, *, start: str | None = None, force_steps: set[str] | None = None) -> None:
    begin = STEPS.index(start) if start else 0
    for step in STEPS[begin:]:
        run_step(ctx, step, force=bool(force_steps and step in force_steps) or (start is not None and STEPS.index(step) >= begin))


def print_status(ctx: Context) -> None:
    for step in STEPS:
        rec = ctx.status["steps"].get(step)
        if not rec:
            ctx.log(f"  {step:10s} -")
        else:
            extra = " (skipped: no input)" if rec.get("skipped") else ""
            ctx.log(f"  {step:10s} {rec['status']}{extra}  {rec.get('finished_at_utc') or rec.get('started_at_utc')}  {rec.get('error', '')}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=["run", "status"] + STEPS)
    parser.add_argument("folder", type=Path, help="the handover folder (contains handover.json)")
    parser.add_argument("--from", dest="start", choices=STEPS, default=None, help="run: redo from this step onwards")
    parser.add_argument("--ignore-verify", action="store_true", help="continue even when files differ from the manifest")
    parser.add_argument("--prefer", choices=("latest", "earliest"), default="latest", help="which run wins a duplicated position")
    parser.add_argument("--no-ai", action="store_true", help="stitch without Kimi")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scale-div", type=int, default=4)
    parser.add_argument("--out-scale", type=float, default=0.25)
    parser.add_argument("--full", action="store_true", help="render the mosaic from the original tiles (needed when --out-scale > 1/scale-div)")
    parser.add_argument("--no-flat-field", action="store_true")
    parser.add_argument("--gain-mode", choices=("tile", "column"), default=None)
    parser.add_argument("--dry-run-stitch", action="store_true", help="prepare the stitch command but do not run run_stitch.py")
    parser.add_argument("--check-samples", type=int, default=12)
    args = parser.parse_args(argv)
    log = lambda m: print(m, flush=True)  # noqa: E731
    options = {"verify": {"ignore": args.ignore_verify}, "assemble": {"prefer": args.prefer},
               "colour": {"workers": args.workers, "scale_div": args.scale_div, "out_scale": args.out_scale, "full": args.full,
                          "no_flat_field": args.no_flat_field, "gain_mode": args.gain_mode, "no_ai": args.no_ai},
               "stitch": {"no_ai": args.no_ai, "workers": args.workers, "full": args.full, "dry_run": args.dry_run_stitch},
               "check": {"samples": args.check_samples}}
    try:
        ctx = Context(args.folder, options=options, log=log)
        if args.action == "status":
            print_status(ctx)
            return 0
        if args.action == "run":
            run_all(ctx, start=args.start)
        else:
            run_step(ctx, args.action, force=True)
        print_status(ctx)
        failed = [s for s, r in ctx.status["steps"].items() if r.get("status") == "failed"]
        return 1 if failed else 0
    except HandoverError as exc:
        log(f"Handover processing stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
