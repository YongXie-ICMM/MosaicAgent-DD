#!/usr/bin/env python3
"""Pack everything the analysis side needs from a scan day into ONE handover folder.

Students used to copy tiles by hand, merge runs and drop the records that explain an
interruption. This script collects, without editing anything:

* every scan session of the chosen day (or the chosen sessions), each ``mosaic_photos/<stamp>/``
  copied intact: raw tiles, ``session.json``, ``events.jsonl``, ``sharpness.csv``,
  ``program_snapshot/``, recovery receipts, review candidates - runs are never merged here;
* the console transcripts ``history/console_logs/launch_*.txt`` of that day;
* the ``shared_history/`` journals, the ``camera_history/`` connection records and the
  ``colour_calibration/`` white-balance records of that day;
* the scanner's ``delivery_manifest.json`` and the result of checking the 16 runtime files
  against it.

It writes ``handover.json`` (one record per run with its grid, counts, camera readback,
event statistics and incidents such as ``move_failed``; the SHA-256 of every packed file)
and a short ``README_zh.txt``. Tiles are verified against the hashes the scanner recorded
in ``events.jsonl``. Optionally the folder is zipped (ZIP64, stored, not recompressed).

The analysis side (``tools/handover.py``) consumes this folder and writes its own results
into ``<folder>/analysis/``; nothing here depends on the analysis environment (standard
library only). The scanner runtime files are not touched.

Usage (on the instrument computer, from ``acquisition/Auto_Scan``):

    python pack_handover.py                  # interactive: today's sessions, asks for operator / notes
    python pack_handover.py --today --zip    # unattended
    python pack_handover.py --session 20261006_140512_123456 --session 20261006_170210_654321
    python pack_handover.py --since 2026-10-01 --out E:\\handover
    python pack_handover.py --today --link   # same disk, no extra space: hard links instead of copies
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import sys
import zipfile

HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
PACKER_VERSION = "2026-10-07"
MOSAIC_DIR = "mosaic_photos"
RUNTIME_MANIFEST = "delivery_manifest.json"
RC_LINE = re.compile(r"rc\s*=\s*-?\d+|Traceback|movement failed|failed:|Error|错误|失败", re.I)
INCIDENT_EVENTS = ("move_failed", "capture_failed", "acquisition_mode_mismatch", "resume",
                   "log_resume_authorized", "resume_camera_readback", "session_finished")


class PackError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def local_date(dt: datetime | None):
    return dt.astimezone().date() if dt else None


def stamp_to_local(stamp: str) -> datetime | None:
    """mosaic_photos / console log stamps are local time: YYYYmmdd_HHMMSS[_ffffff]."""
    m = re.match(r"(\d{8})_(\d{6})(?:_(\d{1,6}))?", stamp)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").astimezone()
    except ValueError:
        return None


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(str(path)).free


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


# ----------------------------------------------------------------------------
# runtime check
# ----------------------------------------------------------------------------
def check_runtime(scan_dir: Path) -> dict:
    manifest_path = scan_dir / RUNTIME_MANIFEST
    if not manifest_path.is_file():
        return {"ok": False, "error": "delivery_manifest.json missing"}
    manifest = read_json(manifest_path)
    mismatched, missing = [], []
    for entry in manifest.get("files", []):
        p = scan_dir / entry["path"]
        if not p.is_file():
            missing.append(entry["path"])
        elif sha256_file(p) != entry["sha256"]:
            mismatched.append(entry["path"])
    return {"ok": not mismatched and not missing, "release": manifest.get("release"),
            "source_repo_commit": manifest.get("source_repo_commit"), "files_checked": len(manifest.get("files", [])),
            "mismatched": mismatched, "missing": missing, "manifest_sha256": sha256_file(manifest_path)}


# ----------------------------------------------------------------------------
# sessions
# ----------------------------------------------------------------------------
def list_sessions(scan_dir: Path) -> list[dict]:
    root = scan_dir / MOSAIC_DIR
    sessions = []
    if not root.is_dir():
        return sessions
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        session_path = folder / "session.json"
        if not session_path.is_file():
            continue
        try:
            session = read_json(session_path)
        except (ValueError, OSError) as exc:
            sessions.append({"folder": folder, "stamp": folder.name, "error": f"session.json unreadable: {exc}"})
            continue
        config = session.get("scan_config") or {}
        started = parse_utc(session.get("started_at_utc")) or stamp_to_local(folder.name)
        sessions.append({"folder": folder, "stamp": folder.name, "session": session, "started": started,
                         "finished": parse_utc(session.get("finished_at_utc")),
                         "date": local_date(started), "sample_id": config.get("sample_id"),
                         "nx": config.get("nx"), "ny": config.get("ny"), "status": session.get("status"),
                         "photos_saved": session.get("photos_saved"), "planned_positions": session.get("planned_positions")})
    return sessions


def describe_run(entry: dict) -> dict:
    """Summary of one session for the manifest: grid, counts, camera, events, incidents, tile hash check."""
    folder: Path = entry["folder"]
    session = entry["session"]
    config = session.get("scan_config") or {}
    camera = session.get("camera") or {}
    events = read_jsonl(folder / "events.jsonl") if (folder / "events.jsonl").is_file() else []
    counts: dict[str, int] = {}
    for e in events:
        counts[e.get("event", "?")] = counts.get(e.get("event", "?"), 0) + 1
    captures = [e for e in events if e.get("event") == "capture_success"]
    first_capture = parse_utc(captures[0]["timestamp_utc"]) if captures else None
    t0 = first_capture or entry.get("started")
    incidents = []
    for e in events:
        if e.get("event") in INCIDENT_EVENTS or e.get("event") == "_unparseable_line":
            t = parse_utc(e.get("timestamp_utc"))
            incidents.append({"event": e.get("event"), "row": e.get("row"), "col": e.get("col"),
                              "timestamp_utc": e.get("timestamp_utc"),
                              "elapsed_min": round((t - t0).total_seconds() / 60, 1) if t and t0 else None,
                              "error": e.get("error"), "status": e.get("status"), "next_point": e.get("next_point"),
                              "verification": e.get("verification")})
    tiles = sorted(p.name for p in folder.iterdir() if re.match(r"^mosaic_r\d+_c\d+\.(png|jpg|jpeg|tif|tiff|bmp)$", p.name, re.I))
    rows = [int(re.match(r"mosaic_r(\d+)_c(\d+)", t).group(1)) for t in tiles]
    cols = [int(re.match(r"mosaic_r(\d+)_c(\d+)", t).group(2)) for t in tiles]
    recorded = {}
    for e in captures:
        name = e.get("filename") or Path(e.get("image_relative_path", "")).name
        if name and e.get("image_sha256"):
            recorded[name] = e["image_sha256"]
    readback = camera.get("readback") if isinstance(camera.get("readback"), dict) else {}
    camera_summary = {k: camera.get(k) for k in ("backend", "model", "display_name", "actual_resolution", "capture_api") if k in camera}
    for key in ("gain", "exposure", "exposure_us", "exposure_ms", "white_balance", "wb_temperature", "temperature", "tint", "auto_exposure", "auto_white_balance"):
        if key in readback:
            camera_summary[key] = readback[key]
    return {"folder": folder.name, "session_id": session.get("session_id"), "started_at_utc": session.get("started_at_utc"),
            "finished_at_utc": session.get("finished_at_utc"), "status": session.get("status"),
            "release": config.get("release"), "sample_id": config.get("sample_id"), "objective": config.get("objective"),
            "grid": {k: config.get(k) for k in ("nx", "ny", "dx_steps", "dy_steps", "order", "invert_x", "invert_y",
                                                  "overlap_x_steps", "overlap_y_steps", "origin_logical_steps", "fov_steps_x", "fov_steps_y")},
            "planned_positions": session.get("planned_positions"), "positions_completed": session.get("positions_completed"),
            "photos_saved": session.get("photos_saved"), "tiles_on_disk": len(tiles),
            "rows": [min(rows), max(rows)] if rows else None, "cols": [min(cols), max(cols)] if cols else None,
            "first_capture_utc": captures[0]["timestamp_utc"] if captures else None,
            "last_capture_utc": captures[-1]["timestamp_utc"] if captures else None,
            "camera": camera_summary, "event_counts": counts, "event_count_recorded": session.get("event_count"),
            "incidents": incidents, "recorded_tile_hashes": recorded, "tile_names": tiles,
            "stage_simulated": session.get("stage_simulated")}


# ----------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------
def select_sessions(sessions: list[dict], *, today=False, since=None, names=None, all_sessions=False) -> list[dict]:
    valid = [s for s in sessions if "session" in s]
    if names:
        chosen = [s for s in valid if s["stamp"] in names]
        missing = set(names) - {s["stamp"] for s in chosen}
        if missing:
            raise PackError("Unknown session folder(s): " + ", ".join(sorted(missing)))
        return chosen
    if all_sessions:
        return valid
    if since is not None:
        return [s for s in valid if s["date"] and s["date"] >= since]
    if today:
        d = datetime.now().astimezone().date()
        return [s for s in valid if s["date"] == d]
    return []


def companions_for(scan_dir: Path, runs: list[dict]) -> dict:
    """Console logs, shared_history journals, colour calibrations of the runs' days (+/- 1 day margin)."""
    dates = {s["date"] for s in runs if s.get("date")}
    if not dates:
        return {"console_logs": [], "shared_history": [], "colour_calibration": []}
    lo, hi = min(dates) - timedelta(days=1), max(dates) + timedelta(days=1)

    def in_range(dt):
        return dt is not None and lo <= dt.astimezone().date() <= hi

    logs = []
    log_dir = scan_dir / "history" / "console_logs"
    if log_dir.is_dir():
        for p in sorted(log_dir.glob("launch_*.txt")):
            if in_range(stamp_to_local(p.name[len("launch_"):])):
                logs.append(p)
    journals = []
    sh = scan_dir / "shared_history"
    if sh.is_dir():
        for p in sorted(q for q in sh.iterdir() if q.is_dir()):
            m = re.match(r"(\d{8})T(\d{6})_\d+Z_", p.name)
            dt = None
            if m:
                try:
                    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    dt = None
            if in_range(dt):
                journals.append(p)
    calibs = []
    cc = scan_dir / "colour_calibration"
    if cc.is_dir():
        for p in sorted(q for q in cc.iterdir() if q.is_dir() and q.name != "simulation"):
            m = re.match(r"(\d{8})T(\d{6})Z", p.name)
            dt = None
            if m:
                try:
                    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    dt = None
            if in_range(dt):
                calibs.append(p)
    return {"console_logs": logs, "shared_history": journals, "colour_calibration": calibs}


# ----------------------------------------------------------------------------
# packing
# ----------------------------------------------------------------------------
LINK_MODE = {"link": False}       # --link: hard links instead of copies (same volume only; falls back to copying)


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if LINK_MODE["link"]:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, files: list[dict], rel_root: Path, log) -> None:
    for root, _dirs, names in os.walk(src):
        for name in sorted(names):
            s = Path(root) / name
            if s.is_symlink():
                continue
            d = dst / s.relative_to(src)
            _place(s, d)
            files.append({"path": d.relative_to(rel_root).as_posix(), "bytes": d.stat().st_size, "sha256": sha256_file(d)})


def _copy_file(src: Path, dst: Path, files: list[dict], rel_root: Path) -> dict:
    _place(src, dst)
    rec = {"path": dst.relative_to(rel_root).as_posix(), "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}
    files.append(rec)
    return rec


def tree_size(path: Path) -> int:
    total = 0
    for root, _d, names in os.walk(path):
        for n in names:
            try:
                total += (Path(root) / n).stat().st_size
            except OSError:
                pass
    return total


def console_log_summary(path: Path, text: str) -> dict:
    lines = text.splitlines()
    hits = [{"line": i, "text": line.strip()[:200]} for i, line in enumerate(lines, 1) if RC_LINE.search(line)]
    return {"name": path.name, "lines": len(lines), "started": lines[0][:120] if lines else None,
            "flagged_lines": len(hits), "flagged": hits[:200]}


def pack(scan_dir: Path, runs: list[dict], out_root: Path, *, operator: str = "", notes: str = "",
         make_zip: bool = False, link: bool = False, log=print) -> Path:
    if not runs:
        raise PackError("No scan session selected")
    LINK_MODE["link"] = bool(link)
    runs = sorted(runs, key=lambda s: s["stamp"])
    sample = next((s.get("sample_id") for s in runs if s.get("sample_id")), None) or "scan"
    sample_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample))[:40]
    date = (runs[0]["date"] or datetime.now().astimezone().date()).strftime("%Y%m%d")
    folder = out_root / f"handover_{sample_slug}_{date}"
    n = 2
    while folder.exists():
        folder = out_root / f"handover_{sample_slug}_{date}_{n}"
        n += 1
    companions = companions_for(scan_dir, runs)
    needed = sum(tree_size(s["folder"]) for s in runs) + sum(p.stat().st_size for p in companions["console_logs"]) \
        + sum(tree_size(p) for p in companions["shared_history"]) + sum(tree_size(p) for p in companions["colour_calibration"])
    out_root.mkdir(parents=True, exist_ok=True)
    free = free_bytes(out_root)
    if link:
        needed_now = needed * (1.05 if make_zip else 0.0)          # links take no space; the zip still does
    else:
        needed_now = needed * (2.1 if make_zip else 1.05)
    if free < needed_now + (200 << 20):
        raise PackError(f"Not enough free space at {out_root}: need about {human(needed_now)}, free {human(free)}")
    log(f"[pack] {len(runs)} run(s), {human(needed)} to {'link' if link else 'copy'} -> {folder}")
    folder.mkdir()
    files: list[dict] = []
    run_records = []
    for s in runs:
        log(f"[pack] run {s['stamp']}: sample {s.get('sample_id')}, {s.get('nx')}x{s.get('ny')}, status {s.get('status')}, "
            f"{s.get('photos_saved')} photos")
        rec = describe_run(s)
        dst = folder / "runs" / s["stamp"]
        before = len(files)
        _copy_tree(s["folder"], dst, files, folder, log)
        copied = {Path(f["path"]).name: f["sha256"] for f in files[before:] if Path(f["path"]).parent == dst.relative_to(folder)}
        mismatched = [name for name, h in rec["recorded_tile_hashes"].items() if name in copied and copied[name] != h]
        unrecorded = [name for name in rec["tile_names"] if name not in rec["recorded_tile_hashes"]]
        missing_files = [name for name in rec["recorded_tile_hashes"] if name not in copied]
        rec["tile_check"] = {"recorded_in_events": len(rec["recorded_tile_hashes"]), "copied": len(rec["tile_names"]),
                             "hash_mismatch": mismatched, "on_disk_but_not_recorded": unrecorded,
                             "recorded_but_missing": missing_files}
        del rec["recorded_tile_hashes"], rec["tile_names"]
        rec["handover_path"] = dst.relative_to(folder).as_posix()
        rec["bytes"] = sum(f["bytes"] for f in files[before:])
        run_records.append(rec)
        if mismatched or missing_files:
            log(f"[pack] WARNING run {s['stamp']}: {len(mismatched)} tile(s) differ from the recorded hash, "
                f"{len(missing_files)} recorded tile(s) missing")
    logs = []
    for p in companions["console_logs"]:
        rec = _copy_file(p, folder / "console_logs" / p.name, files, folder)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        logs.append(dict(console_log_summary(p, text), bytes=rec["bytes"], sha256=rec["sha256"]))
    journals = []
    for p in companions["shared_history"]:
        before = len(files)
        _copy_tree(p, folder / "shared_history" / p.name, files, folder, log)
        meta = {}
        if (p / "history.json").is_file():
            try:
                h = read_json(p / "history.json")
                meta = {k: h.get(k) for k in ("history_id", "created_at_utc", "updated_at_utc", "status", "event_count")}
            except (ValueError, OSError):
                meta = {"error": "history.json unreadable"}
        journals.append(dict(folder=p.name, files=len(files) - before, **meta))
    camera_history = []
    ch = scan_dir / "camera_history"
    if ch.is_dir():
        for p in sorted(q for q in ch.iterdir() if q.is_file()):
            camera_history.append(_copy_file(p, folder / "camera_history" / p.name, files, folder))
    calibrations = []
    for p in companions["colour_calibration"]:
        before = len(files)
        _copy_tree(p, folder / "colour_calibration" / p.name, files, folder, log)
        rec = {"folder": p.name, "files": len(files) - before}
        settings = p / "camera_colour_settings.json"
        if settings.is_file():
            try:
                c = read_json(settings)
                rec.update({k: c.get(k) for k in ("recorded_at_utc", "calibrated", "verdict", "mode", "accepted_by")})
                rec["observed_mean_rgb"] = (c.get("observed") or {}).get("median_rgb") or (c.get("observed") or {}).get("mean_rgb")
            except (ValueError, OSError):
                rec["error"] = "camera_colour_settings.json unreadable"
        calibrations.append(rec)
    latest = scan_dir / "colour_calibration" / "latest.json"
    if latest.is_file():
        _copy_file(latest, folder / "colour_calibration" / "latest.json", files, folder)
    runtime = check_runtime(scan_dir)
    if (scan_dir / RUNTIME_MANIFEST).is_file():
        _copy_file(scan_dir / RUNTIME_MANIFEST, folder / "scanner" / RUNTIME_MANIFEST, files, folder)
    manifest = {"schema_version": SCHEMA_VERSION, "kind": "scan_handover", "packed_at_utc": datetime.now(timezone.utc).isoformat(),
                "packer": {"script": Path(__file__).name, "version": PACKER_VERSION, "sha256": sha256_file(Path(__file__))},
                "instrument": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
                               "scan_dir": str(scan_dir)},
                "operator": operator, "notes": notes, "sample_id": sample,
                "copy_method": "hard_links_where_possible" if link else "copies",
                "scanner_runtime_check": runtime, "runs": run_records, "console_logs": logs, "shared_history": journals,
                "camera_history": camera_history, "colour_calibration": calibrations,
                "totals": {"files": len(files), "bytes": sum(f["bytes"] for f in files)}, "files": files,
                "layout": {"runs/<stamp>/": "one scan session, copied intact (never merged)",
                           "console_logs/": "launch_*.txt transcripts of the day", "shared_history/": "acquisition journals",
                           "camera_history/": "camera connection records", "colour_calibration/": "white-balance records",
                           "analysis/": "written later by tools/handover.py on the analysis computer"}}
    (folder / "handover.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (folder / "README_zh.txt").write_text(readme_text(manifest), encoding="utf-8")
    log(f"[pack] {len(files)} files, {human(manifest['totals']['bytes'])}; manifest handover.json written")
    if make_zip:
        zip_path = folder.with_suffix(".zip")
        log(f"[pack] zipping -> {zip_path.name} (stored, no recompression)")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for root, _d, names in os.walk(folder):
                for name in sorted(names):
                    p = Path(root) / name
                    z.write(p, (folder.name / p.relative_to(folder)).as_posix())
        digest = sha256_file(zip_path)
        zip_path.with_suffix(".zip.sha256").write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
        log(f"[pack] zip sha256 {digest}")
    return folder


def readme_text(manifest: dict) -> str:
    runs = "\n".join(f"  - runs/{r['folder']}: 样品 {r.get('sample_id')}，{r['grid'].get('nx')}×{r['grid'].get('ny')}，"
                     f"状态 {r.get('status')}，保存 {r.get('photos_saved')} 张，列 {r.get('cols')}，行 {r.get('rows')}，"
                     f"事件里的异常 {len([i for i in r['incidents'] if i['event'] not in ('session_finished',)])} 条"
                     for r in manifest["runs"])
    return (f"扫描交接文件夹（{manifest['sample_id']}，打包于 {manifest['packed_at_utc']}）\n\n"
            "这个文件夹是分析电脑需要的全部资料，请整个文件夹（或同名 zip）一起交给老师，不要改名、不要删文件。\n\n"
            f"包含的扫描轮次：\n{runs}\n\n"
            "其它：console_logs/（控制台记录）、shared_history/（采集日志）、camera_history/（相机连接记录）、\n"
            "colour_calibration/（白平衡记录）、scanner/（扫描程序清单）。handover.json 列出每个文件的 SHA-256。\n\n"
            "分析电脑上：双击 04_process_handover，选择这个文件夹；结果写入这个文件夹的 analysis/ 子目录。\n")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _prompt(text: str, default: str = "") -> str:
    try:
        value = input(text).strip()
    except EOFError:
        return default
    return value or default


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scan-dir", type=Path, default=HERE, help="the Auto_Scan directory (default: this script's folder)")
    parser.add_argument("--out", type=Path, default=None, help="where the handover folder is created (default: <scan-dir>/handover)")
    parser.add_argument("--today", action="store_true", help="pack today's sessions (default when no other selection)")
    parser.add_argument("--since", type=str, default=None, help="pack sessions started on or after YYYY-MM-DD")
    parser.add_argument("--session", action="append", default=None, help="pack this mosaic_photos/<stamp> (repeatable)")
    parser.add_argument("--all", action="store_true", help="pack every session")
    parser.add_argument("--list", action="store_true", help="only list the sessions and exit")
    parser.add_argument("--operator", default=None, help="who scanned (asked interactively when omitted)")
    parser.add_argument("--notes", default=None, help="free text about the day (asked interactively when omitted)")
    parser.add_argument("--zip", action="store_true", help="also create <folder>.zip beside the folder")
    parser.add_argument("--link", action="store_true", help="hard-link instead of copying when the destination is on the same disk "
                                                             "(no extra space; the folder then shares the originals' bytes - zip it or copy it to another disk to hand it over)")
    parser.add_argument("--yes", action="store_true", help="no interactive questions")
    args = parser.parse_args(argv)
    log = lambda m: print(m, flush=True)  # noqa: E731
    scan_dir = args.scan_dir.resolve()
    sessions = list_sessions(scan_dir)
    if args.list or not sessions:
        if not sessions:
            log(f"No scan sessions under {scan_dir / MOSAIC_DIR}")
            return 1
        for s in sessions:
            if "session" in s:
                log(f"  {s['stamp']}  {s['date']}  sample={s.get('sample_id')}  {s.get('nx')}x{s.get('ny')}  "
                    f"status={s.get('status')}  photos={s.get('photos_saved')}/{s.get('planned_positions')}")
            else:
                log(f"  {s['stamp']}  ({s['error']})")
        if args.list:
            return 0
    try:
        since = datetime.strptime(args.since, "%Y-%m-%d").date() if args.since else None
        explicit = bool(args.session or args.all or since or args.today)
        runs = select_sessions(sessions, today=args.today or not explicit, since=since, names=args.session, all_sessions=args.all)
        if not runs and not explicit and not args.yes:
            log("No session started today. Sessions on disk:")
            for s in sessions:
                if "session" in s:
                    log(f"  {s['stamp']}  {s['date']}  sample={s.get('sample_id')}  status={s.get('status')}  photos={s.get('photos_saved')}")
            picked = _prompt("Type the session stamp(s) to pack, separated by spaces (Enter = cancel): ")
            if not picked:
                return 1
            runs = select_sessions(sessions, names=picked.split())
        if not runs:
            raise PackError("No session selected (use --today, --since, --session or --all)")
        operator = args.operator if args.operator is not None else ("" if args.yes else _prompt("Operator name / 操作者: "))
        notes = args.notes if args.notes is not None else ("" if args.yes else _prompt("Notes for the teacher (interruptions, settings) / 备注: "))
        out_root = (args.out or (scan_dir / "handover")).resolve()
        folder = pack(scan_dir, runs, out_root, operator=operator, notes=notes, make_zip=args.zip, link=args.link, log=log)
        log(f"[done] {folder}")
        return 0
    except PackError as exc:
        log(f"Packing stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
