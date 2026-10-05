"""Launch the existing scanner UI and read its evidence, without importing it.

Opening the native window does not connect devices or start acquisition. The
scanner owns its hardware and journal; this adapter never stops its processes.

Two layouts are recognised: a Git checkout (``<repo>/Auto_Scan/launch_scan.py``)
and the delivered student package, which is flat (``<folder>/launch_scan.py``
beside ``delivery_manifest.json``, no ``.git``). Revision notes live in
``known_scanner_revisions.json`` next to this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import threading
import uuid


_PROCESSES: dict[str, subprocess.Popen] = {}
_LAUNCH_LOCK = threading.Lock()
_MAX_BYTES = 128 * 1024 * 1024        # 954 points wrote a 24.5 MB snapshot; keep a bound
_MAX_EVENTS = 50000
_MAX_FOLDERS = 2000
_MANIFEST_MAX = 2 * 1024 * 1024
_MAIN_PROGRAM = "03Auto_Snake_Scan_Camera_v3.py"
_REVISIONS_FILE = Path(__file__).with_name("known_scanner_revisions.json")
STILL_ACTIVE = 259


def _text(zh: str, en: str) -> dict:
    return {"zh": zh, "en": en}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------- layout
def scanner_folder(repo: Path) -> Path | None:
    """Folder holding launch_scan.py: ``<repo>/Auto_Scan`` for a checkout, ``<repo>``
    for the delivered student package; None when neither exists."""
    repo = Path(repo).expanduser().resolve()
    for folder in (repo / "Auto_Scan", repo):
        if (folder / "launch_scan.py").is_file():
            return folder
    return None


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True,
            timeout=3, check=False, shell=False,
            env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_revision(root: Path) -> tuple[str | None, str | None]:
    """HEAD and porcelain status of ``root`` only when ``root`` itself is a Git
    top level. A student package extracted inside some other repository must not
    inherit that repository's revision."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if not top:
        return None, None
    try:
        # git prints forward slashes and may differ in drive-letter case on Windows.
        if os.path.normcase(str(Path(top).resolve())) != os.path.normcase(str(Path(root).resolve())):
            return None, None
    except OSError:
        return None, None
    revision = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=normal") if revision else None
    return revision, status


def _sha256_file(path: Path, limit: int = 16 * 1024 * 1024) -> str | None:
    try:
        if path.stat().st_size > limit:
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def read_delivery_manifest(folder: Path) -> dict | None:
    """Bounded, read-only summary of the student package manifest, or None when the
    folder is not a delivered package. ``main_program_matches_manifest`` is True,
    False, or None when it could not be checked."""
    path = Path(folder) / "delivery_manifest.json"
    if not path.is_file():
        return None
    summary = {"path": str(path), "error": None, "release": None, "source_repo_commit": None,
               "local_changes": None, "hardware_validated": None, "external_model_calls": None,
               "built_at_utc": None, "main_program_matches_manifest": None,
               "package_files_match_manifest": None, "file_check_failures": [],
               "checked_files": 0}
    try:
        if path.stat().st_size > _MANIFEST_MAX:
            raise ValueError("delivery manifest exceeds the read limit")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("delivery manifest is not a JSON object")
    except (OSError, ValueError) as exc:
        summary["error"] = str(exc)
        return summary
    for key in ("release", "source_repo_commit", "local_changes", "hardware_validated",
                "external_model_calls", "built_at_utc"):
        value = data.get(key)
        summary[key] = value if isinstance(value, (str, bool, int)) else None
    # Helpers (camera, geometry, history) affect acquisition as much as the main
    # script. A package is consistent only when every listed file matches.
    records = data.get("files")
    if not isinstance(records, list) or not records or len(records) > 1024:
        summary["error"] = "Delivery manifest needs a bounded, non-empty files list"
        return summary
    seen = set()
    root = Path(folder).resolve()
    failures = summary["file_check_failures"]
    for item in records:
        relative = item.get("path") if isinstance(item, dict) else None
        digest = item.get("sha256") if isinstance(item, dict) else None
        normalized = relative.replace("\\", "/") if isinstance(relative, str) else ""
        if (not normalized or ":" in normalized
                or PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts
                or normalized in seen or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in digest)):
            summary["error"] = "Delivery manifest contains an invalid or duplicate file record"
            return summary
        seen.add(normalized)
        target = root / normalized
        try:
            target.resolve().relative_to(root)
        except (OSError, ValueError):
            failures.append(relative + ": outside package")
            continue
        actual = _sha256_file(target)
        matches = actual is not None and actual == digest.lower()
        summary["checked_files"] += 1
        if normalized == _MAIN_PROGRAM:
            summary["main_program_matches_manifest"] = matches
        if not matches:
            failures.append(relative + ": missing, unreadable, oversized, or checksum mismatch")
    if not {_MAIN_PROGRAM, "launch_scan.py"}.issubset(seen):
        summary["error"] = "Delivery manifest must list the main program and launch_scan.py"
        return summary
    summary["package_files_match_manifest"] = not failures
    return summary


def _revision_table() -> dict:
    try:
        data = json.loads(_REVISIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    entries = [e for e in (data.get("revisions") or []) if isinstance(e, dict) and isinstance(e.get("prefixes"), list)]
    unknown = data.get("unknown") if isinstance(data.get("unknown"), dict) else {}
    missing = data.get("no_revision") if isinstance(data.get("no_revision"), dict) else {}
    return {"revisions": entries,
            "unknown": _text(str(unknown.get("zh", "当前版本尚未归入已核对发布版本。")),
                             str(unknown.get("en", "This checkout is not a recognized reviewed release."))),
            "no_revision": _text(str(missing.get("zh", "无法确认扫描程序版本。")),
                                 str(missing.get("en", "Scanner version cannot be confirmed.")))}


def _note_for(revision: str | None, table: dict) -> dict:
    if not revision:
        return dict(table["no_revision"])
    for entry in table["revisions"]:
        if any(isinstance(p, str) and p and revision.startswith(p) for p in entry["prefixes"]):
            return _text(str(entry.get("zh", "")), str(entry.get("en", "")))
    return dict(table["unknown"])


def inspect_scanner(repo: Path) -> dict:
    """Describe the actual checkout or package; unknown Git state is never called clean."""
    repo = Path(repo).expanduser().resolve()
    folder = scanner_folder(repo)
    entrypoint = (folder or repo / "Auto_Scan") / "launch_scan.py"
    table = _revision_table()
    package = read_delivery_manifest(folder) if folder else None
    revision, status, source = None, None, None
    if folder:
        git_root = folder.parent if folder.name == "Auto_Scan" else folder
        revision, status = _git_revision(git_root)
        if revision:
            source = "git"
        elif package and isinstance(package.get("source_repo_commit"), str) and package["source_repo_commit"]:
            revision, source = package["source_repo_commit"], "delivery_manifest"
    note = _note_for(revision, table)
    if source == "delivery_manifest":
        release = package.get("release") or "student package"
        match = package.get("main_program_matches_manifest")
        verdict_zh = "一致" if match else "不一致" if match is False else "未核对"
        verdict_en = "matches" if match else "does NOT match" if match is False else "not checked against"
        note = _text(f"{note['zh']} 学生包：{release}；主程序与交付清单{verdict_zh}。",
                     f"{note['en']} Student package: {release}; main program {verdict_en} the delivery manifest.")
        if package.get("local_changes") is True:
            note = _text(note["zh"] + " 交付清单记录打包时源码有本地修改。",
                         note["en"] + " The manifest records local source changes at packaging time.")
    package_blocked = bool(package and (package.get("error") or package.get("package_files_match_manifest") is not True))
    if package:
        if package_blocked:
            detail = package.get("error") or "; ".join(package.get("file_check_failures", [])[:3])
            note = _text(note["zh"] + " 交付包核对未通过，禁止启动；请重新解压完整原包。" + str(detail),
                         note["en"] + " Package verification failed; launch is blocked. Re-extract the complete original package. " + str(detail))
        else:
            note = _text(note["zh"] + f" 清单中 {package['checked_files']} 个文件均一致。",
                         note["en"] + f" All {package['checked_files']} manifest-listed files match.")
        if package.get("hardware_validated") is False:
            note = _text(note["zh"] + " 此交付版本尚未完成实机验证。",
                         note["en"] + " Instrument validation of this delivery is pending.")
    if status:
        note = _text(note["zh"] + " 工作目录还有本地修改。", note["en"] + " Local changes are present.")
    return {
        "exists": entrypoint.is_file(), "revision": revision, "revision_source": source,
        "dirty": bool(status) if status is not None else None,
        "entrypoint": str(entrypoint), "folder": str(folder) if folder else None,
        "layout": None if folder is None else ("checkout" if folder.name == "Auto_Scan" else "package"),
        "platform_supported": platform.system() == "Windows",
        "version_note": note, "package": package, "launch_blocked": package_blocked,
    }


# ----------------------------------------------------------------- processes
def pid_alive(pid: int) -> bool | None:
    """True/False when the operating system answers, None when it cannot be checked.
    Never signals the process: on Windows ``os.kill`` terminates, so ctypes is used."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel32.GetExitCodeProcess.restype = ctypes.c_int
            kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            kernel32.CloseHandle.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel32.OpenProcess(0x1000, 0, pid)      # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return True if ctypes.get_last_error() == 5 else False   # ERROR_ACCESS_DENIED: exists
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return None
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def describe_lock(lock: Path) -> str:
    """Explain an existing scanner lock without touching it."""
    pid = None
    try:
        with lock.open("rb") as stream:
            text = stream.read(64).decode("ascii", errors="replace").strip()
        pid = int(text) if text.isdigit() else None
    except OSError:
        pass
    if pid is None:
        return ("The scanner lock exists but holds no readable PID. Check the existing scanner window; "
                "this adapter will not remove its lock: " + str(lock))
    alive = pid_alive(pid)
    if alive:
        return (f"The scanner lock exists and its process (PID {pid}) appears to be running; use the "
                "existing scanner window. This adapter never removes the lock.")
    if alive is False:
        return (f"The scanner lock exists but its process (PID {pid}) is not running (stale lock after a "
                "crash or power loss). Confirm no scanner window is open, then remove it by hand: "
                f"{lock}. This adapter never removes it.")
    return (f"The scanner lock exists (PID {pid}); the process state could not be checked. Check the "
            "existing scanner window; this adapter will not remove its lock: " + str(lock))


def _write_json(path: Path, value: dict) -> None:
    try:
        temp = path.with_name(path.name + "." + uuid.uuid4().hex[:8] + ".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        pass


def log_tail(path: Path, lines: int = 20, limit: int = 16384) -> list[str]:
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            raw = stream.read()
    except OSError:
        return []
    return raw.decode("utf-8", errors="replace").splitlines()[-lines:]


def last_launch(log_dir: Path) -> dict | None:
    """The most recent launch record written by this adapter, or None."""
    path = Path(log_dir).expanduser() / "active_scanner.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _watch(process, record: dict, logs: Path, on_exit) -> None:
    try:
        returncode = process.wait()
    except Exception as exc:  # noqa: BLE001 - a watcher must report, never crash silently
        returncode, error = None, str(exc)
    else:
        error = None
    ended = dict(record, status="exited", returncode=returncode, ended_at_utc=_now(),
                 wait_error=error)
    _write_json(logs / "active_scanner.json", ended)
    try:
        on_exit(dict(ended, log_tail=log_tail(Path(record["log_path"]))))
    except Exception:  # noqa: BLE001 - the callback belongs to the caller
        pass


def launch_scanner(repo: Path, log_dir: Path, python_executable: str, on_exit=None) -> dict:
    """Open the existing Windows UI, retaining its supervisor without a shutdown hook.

    No simulation switch, automatic connect, scan, restart, or termination is
    provided. A returned PID confirms process creation, not a ready GUI or scan.
    ``on_exit`` (optional) receives the launch record plus ``returncode`` and
    ``log_tail`` when the launched process ends, from a daemon thread.
    """
    if platform.system() != "Windows":
        raise RuntimeError("Real scanner launch requires Windows; no process or hardware was opened.")
    repo = Path(repo).expanduser().resolve()
    folder = scanner_folder(repo)
    if folder is None:
        raise FileNotFoundError(f"Scanner entrypoint not found under: {repo} (expected Auto_Scan/launch_scan.py or launch_scan.py)")
    entrypoint = folder / "launch_scan.py"
    package = read_delivery_manifest(folder)
    if package and (package.get("error") or package.get("package_files_match_manifest") is not True):
        raise RuntimeError("Student package integrity check failed. Re-extract the complete original package; no process was launched.")
    if not isinstance(python_executable, str) or not python_executable.strip():
        raise ValueError("A configured Python executable is required.")
    key = str(entrypoint)
    with _LAUNCH_LOCK:
        previous = _PROCESSES.get(key)
        if previous is not None and previous.poll() is None:
            raise RuntimeError("The managed scanner is already running; use its existing window.")
        lock = folder / ".mosaic_stage.lock"
        if lock.exists():
            raise RuntimeError(describe_lock(lock))
        logs = Path(log_dir).expanduser().resolve()
        logs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = logs / f"scanner_launch_{stamp}_{uuid.uuid4().hex[:8]}.log"
        with path.open("xb") as output:
            process = subprocess.Popen(
                [python_executable, "-u", str(entrypoint)], cwd=str(folder),
                env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1"),
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                shell=False,
            )
        _PROCESSES[key] = process
        record = {"pid": process.pid, "log_path": str(path), "entrypoint": str(entrypoint),
                  "started_at_utc": _now(), "status": "running"}
        _write_json(logs / "active_scanner.json", record)
        if on_exit is not None:
            threading.Thread(target=_watch, args=(process, record, logs, on_exit),
                             name="scanner-exit-watch", daemon=True).start()
        return dict(record)


# ----------------------------------------------------------------- histories
def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise ValueError(f"History exceeds {_MAX_BYTES // (1024 * 1024)} MiB read limit")
    return raw


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON value: {value}")


def _json(raw: bytes):
    return json.loads(raw, parse_constant=_reject_constant)


def _validate_events(events, history_id: str) -> None:
    if not isinstance(events, list) or not events or len(events) > _MAX_EVENTS:
        raise ValueError("History events missing or exceed the read limit")
    seen = set()
    for number, event in enumerate(events, 1):
        if not isinstance(event, dict) or event.get("schema_version") != 1:
            raise ValueError("Unsupported history event schema")
        if type(event.get("sequence")) is not int or event["sequence"] != number:
            raise ValueError("History event sequence is not contiguous")
        identity = event.get("event_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Missing or duplicate event identity")
        seen.add(identity)
        if event.get("history_id") != history_id:
            raise ValueError("History identity mismatch")
        if not isinstance(event.get("event"), str) or not isinstance(event.get("timestamp_utc"), str):
            raise ValueError("Event name or timestamp missing")
        if number > 1 and event["event"] == "history_started":
            raise ValueError("Duplicate history start")
        if number < len(events) and event["event"] == "history_closed":
            raise ValueError("Events follow a closed history")
    if events[0]["event"] != "history_started":
        raise ValueError("History does not begin with history_started")


def _history_events(path: Path) -> tuple[list, list[str]]:
    snapshot = _json(_read_bytes(path))
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        raise ValueError("Unsupported history snapshot schema")
    history_id = snapshot.get("history_id")
    if not isinstance(history_id, str) or not history_id:
        raise ValueError("Snapshot history identity missing")
    events = snapshot.get("events")
    _validate_events(events, history_id)
    closed = events[-1]["event"] == "history_closed"
    if (type(snapshot.get("event_count")) is not int or snapshot["event_count"] != len(events)
            or type(snapshot.get("snapshot_sequence")) is not int
            or snapshot["snapshot_sequence"] != len(events)
            or snapshot.get("is_closed") is not closed):
        raise ValueError("Snapshot summary does not match its events")
    warnings = []
    journal = path.with_name("events.jsonl")
    if not journal.is_file():
        return events, ["journal_unavailable_snapshot_only"]
    raw = _read_bytes(journal)
    complete = raw.splitlines(keepends=True)
    if complete and not complete[-1].endswith(b"\n"):
        complete.pop()
        warnings.append("uncommitted_trailing_line")
    if len(complete) > _MAX_EVENTS:
        raise ValueError("Journal exceeds event read limit")
    journal_events = [_json(line) for line in complete]
    _validate_events(journal_events, history_id)
    if len(journal_events) < len(events) or journal_events[:len(events)] != events:
        raise ValueError("Snapshot does not match journal prefix")
    if len(journal_events) > len(events):
        warnings.append(f"snapshot_lag={len(journal_events) - len(events)}; using complete journal records")
    return journal_events, warnings


def _error(path: Path, reason: str) -> dict:
    return {"at": "", "kind": "scanner_history_error", "path": str(path),
            "message": _text("采集历史待核对：" + reason, "Acquisition history needs review: " + reason)}


def _acquisition_summary(events: list) -> tuple[dict | None, dict]:
    """Read the latest scan's contract; earlier sessions cannot verify a new scan."""
    latest = None
    camera = None
    for event in events:
        if event["event"] == "camera_connected" and isinstance(event.get("camera"), dict):
            camera = event["camera"]
        if event["event"] == "session_started":
            latest = None
            session = event.get("session")
            if isinstance(session, dict):
                config = session.get("scan_config")
                latest = config.get("acquisition_contract") if isinstance(config, dict) else None
                if isinstance(session.get("camera"), dict):
                    camera = session["camera"]
        elif event["event"] in {"acquisition_mode_verified", "acquisition_mode_mismatch"}:
            latest = event.get("acquisition_contract")

    def size(value):
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(type(v) is int and v > 0 for v in value):
            return f"{value[0]} × {value[1]}"
        return "unknown"

    if not isinstance(latest, dict) or latest.get("schema_version") != 1:
        actual = size(camera.get("actual_resolution")) if isinstance(camera, dict) else "unknown"
        if actual != "unknown":
            return None, _text(f" 历史相机记录为 {actual} 像素；未记录本次尺寸约束验证。",
                               f" Historical camera record: {actual} pixels; no acquisition-contract verification was recorded for this scan.")
        return None, _text("", "")
    expected = size(latest.get("expected_image_size"))
    actual = size(latest.get("actual_image_size"))
    status = str(latest.get("verification_status", "unknown"))[:60]
    phase = str(latest.get("verification_phase", "unknown"))[:40]
    verified = (status == "verified_received_frame" and expected != "unknown" and actual == expected)
    zh = "收到帧尺寸已核对" if verified else "尺寸不一致或验证失败" if status == "mismatch" else "尚未通过尺寸验证"
    en = "received-frame size verified" if verified else "size mismatch or failed verification" if status == "mismatch" else "frame-size verification pending or invalid"
    calibration = str(latest.get("calibration_status", "unknown"))[:80]
    return latest, _text(f" 最近扫描尺寸记录：目标 {expected}，实收 {actual} 像素；{zh}（{phase}）。物理标定状态：{calibration}。",
                         f" Latest scan mode record: expected {expected}, received {actual} pixels; {en} ({phase}). Physical calibration: {calibration}.")


def _summary(path: Path) -> dict:
    events, warnings = _history_events(path)
    last = events[-1]
    closed = last["event"] == "history_closed"
    endings = [event for event in events if event["event"] == "session_finished"]
    accepted = sum(event["event"] == "candidate_accepted" for event in events)
    captures = sum(event["event"] == "capture_success" for event in events)
    human = sum(event["event"] == "candidate_accepted" and event.get("human_confirmed") is True for event in events)
    end = endings[-1] if endings else None
    status = str(end.get("status", "unknown"))[:80] if end else "not_recorded"
    zh = (f"应用历史{'已关闭' if closed else '未关闭'}；最近扫描结束状态：{status}。"
          f"记录接受 {accepted} 张、采集成功事件 {captures} 条、明确人工接受 {human} 张。"
          "这些是日志记录，未在此核对原图、清晰度或空间覆盖。")
    en = (f"Application history {'closed' if closed else 'open'}; latest scan finish: {status}. "
          f"Recorded accepted images: {accepted}; capture-success events: {captures}; explicit human acceptances: {human}. "
          "These are log records; original images, sharpness and spatial coverage have not been verified here.")
    contract, mode_message = _acquisition_summary(events)
    zh += mode_message["zh"]
    en += mode_message["en"]
    if contract and contract.get("verification_status") == "mismatch":
        warnings.append("acquisition_mode_mismatch")
    metadata = events[0].get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("stage_simulated") is True:
        zh += " XY 位移台为仿真；相机可能为真实设备。"
        en += " XY stage was simulated; the camera may have been real."
    recording_error = last.get("recording_error") if closed else None
    if recording_error:
        warnings.append("recording_error=" + str(recording_error)[:300])
    if warnings:
        detail = "; ".join(warnings)
        zh += " 读取提示：" + detail
        en += " Read diagnostics: " + detail
    return {"at": last["timestamp_utc"], "kind": "scanner_history_warning" if warnings else "scanner_history",
            "message": _text(zh, en), "path": str(path), "acquisition_contract": contract}


def history_root(repo: Path) -> Path:
    """Where the scanner writes ``shared_history`` for this layout."""
    repo = Path(repo).expanduser().resolve()
    return (scanner_folder(repo) or repo / "Auto_Scan") / "shared_history"


def _history_paths(root: Path, limit: int) -> tuple[list[Path], bool]:
    paths = []
    overflow = False
    with os.scandir(root) as folders:
        for number, folder in enumerate(folders):
            if number >= _MAX_FOLDERS:
                overflow = True
                break
            if folder.is_dir(follow_symlinks=False):
                path = Path(folder.path) / "history.json"
                if path.exists() or path.with_name("events.jsonl").exists():
                    paths.append(path)
    # Recorder directory names begin with UTC timestamps; do not trust mutable mtimes.
    return sorted(paths, key=lambda item: item.parent.name, reverse=True)[:limit], overflow


def history_signature(repo: Path, limit: int = 5) -> tuple:
    """Cheap change detector for the newest histories: names, sizes and mtimes of
    their snapshot and journal. No file content is read."""
    root = history_root(repo)
    if not root.exists():
        return ()
    try:
        paths, overflow = _history_paths(root, limit)
    except OSError as exc:
        return ("error", str(exc))
    items = []
    for path in paths:
        for candidate in (path, path.with_name("events.jsonl")):
            try:
                stat = candidate.stat()
                items.append((str(candidate), stat.st_size, stat.st_mtime_ns))
            except OSError:
                items.append((str(candidate), None, None))
    return (overflow, tuple(items))


def scanner_history(repo: Path, limit=5) -> list:
    """Summarize recent application histories, bounded and strictly read-only.

    No image-count or filename inference, acquisition writes, history repair on
    disk, or imported scanner code. Invalid snapshots are visible error rows.
    """
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("limit must be an integer between 1 and 20")
    root = history_root(repo)
    if not root.exists():
        return []
    try:
        paths, overflow = _history_paths(root, limit)
    except OSError as exc:
        return [_error(root, str(exc))]
    rows = []
    if overflow:
        rows.append(_error(root, f"Directory listing capped at {_MAX_FOLDERS}; newest history is not guaranteed"))
    for path in paths[:limit - len(rows)]:
        try:
            rows.append(_summary(path))
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            rows.append(_error(path, str(exc)))
    return rows
