"""Verify the complete DD student trial kit without importing a device SDK.

The optional trial marker binds a separate, frozen C1 scanner. Existing teacher
workbench settings are handled by student_demo and are never reset here. Checks
read files and PE headers only; they never open a camera, controller or GUI.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import struct
import sys


MARKER = "trial_kit.json"
C1_RELEASE = "954-1080p-C1-candidate"
DLLS = ("amcam.dll", "drivers/bindy.dll", "drivers/libjximc.dll", "drivers/libximc.dll", "drivers/xiwrapper.dll")
# Exact path inventory of the independently frozen C1 delivery. A future scanner
# release needs an explicit new reviewed inventory; a marker cannot add code.
C1_FILES = frozenset("""
00_学生操作与故障恢复手册.pdf
00_学生操作说明.html
00_本次1080p试跑说明.pdf
00_本次日志更新说明.txt
00_设备通信日志使用说明.txt
01_setup.bat
02_start_scan.bat
03Auto_Snake_Scan_Camera_v3.py
Camera_v2.py
SHARED_HISTORY.md
STUDENT_SCAN_V3.md
acquisition_contract.py
amcam.dll
amcam.py
camera_observability.py
camera_session_history.py
collect_device_diagnostics.py
device_diagnostics.py
drivers/bindy.dll
drivers/libjximc.dll
drivers/libximc.dll
drivers/xiwrapper.dll
gui_language.py
gui_language_catalog.json
gui_theme.py
history_viewer.py
launch_scan.py
offline_validation.txt
requirements_scan.txt
scan_event_schema.py
scan_log_resume.py
scan_preview.py
scan_review.py
shared_history.py
test_acquisition_contract.py
test_camera_observability.py
test_collect_device_diagnostics.py
test_device_communication_scope.py
test_device_diagnostics.py
test_guard_scope.py
test_gui_language.py
test_history_integration.py
test_log_resume_gui.py
test_scan_event_schema.py
test_scan_review.py
test_scan_v3_offline.py
test_stage_communications.py
test_stage_read_retry.py
test_stage_readback.py
validation/20260930_mappingtest_v3/README.md
validation/20260930_mappingtest_v3/student_core_fingerprint.json
validation/20260930_mappingtest_v3/summary.json
validation/20261005_1080p_contract/README.md
validation/20261005_1080p_contract/protected_baseline.json
validation/20261025_log_categories/README.md
validation/20261025_log_categories/protected_baseline.json
validation/20261031_device_communication/03Auto_Snake_Scan_Camera_v3.py.before
validation/20261031_device_communication/Camera_v2.py.before
validation/20261031_device_communication/acquisition_contract.py.before
validation/20261031_device_communication/camera_session_history.py.before
validation/20261031_device_communication/gui_language.py.before
validation/20261031_device_communication/gui_language_catalog.json.before
validation/20261031_device_communication/gui_theme.py.before
validation/20261031_device_communication/history_viewer.py.before
validation/20261031_device_communication/launch_scan.py.before
validation/20261031_device_communication/prior_candidate_files.json
validation/20261031_device_communication/scan_event_schema.py.before
validation/20261031_device_communication/scan_log_resume.py.before
validation/20261031_device_communication/scan_preview.py.before
validation/20261031_device_communication/scan_review.py.before
validation/20261031_device_communication/shared_history.py.before
给学生的消息.txt
试跑记录模板.txt
""".split())


def _require(condition, message):
    if not condition:
        raise ValueError("DD trial kit: " + message)


def _relative(value):
    _require(isinstance(value, str) and value and len(value) <= 512
             and "\\" not in value and ":" not in value and not re.search(r"[\x00-\x1f]", value),
             "invalid relative path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and all(part not in (".", "..") and not part.startswith(".") for part in path.parts)
             and path.as_posix() == value, "path escapes or is not canonical")
    return path


def _within(root, relative):
    path = root
    for part in _relative(relative).parts:
        path = path / part
        _require(not path.is_symlink(), "linked control or scanner paths are not accepted")
    _require(path.resolve().is_relative_to(root), "scanner path escapes this kit")
    return path


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, limit):
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
             "missing, linked or oversized record: " + path.name)
    def invalid(value):
        raise ValueError("Nonfinite trial record value: " + value)
    data = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid)
    _require(isinstance(data, dict), "record must be a JSON object")
    return data


def _pe64(path):
    with path.open("rb") as source:
        header = source.read(64)
        _require(len(header) == 64 and header[:2] == b"MZ", "invalid DLL header: " + path.name)
        offset = struct.unpack_from("<I", header, 0x3c)[0]
        _require(64 <= offset <= min(path.stat().st_size - 26, 1024 * 1024), "invalid PE offset: " + path.name)
        source.seek(offset)
        pe = source.read(26)
    _require(pe[:4] == b"PE\0\0" and struct.unpack_from("<H", pe, 4)[0] == 0x8664
             and struct.unpack_from("<H", pe, 24)[0] == 0x20b,
             "scanner needs its original Windows x64 DLLs: " + path.name)


def initial_config(root):
    """Validate a fresh kit and return its portable, freshly resolved settings."""
    root = Path(root).resolve()
    marker = _json(root / MARKER, 32 * 1024)
    _require(type(marker.get("schema_version")) is int and marker["schema_version"] == 1
             and marker.get("kind") == "dd_student_trial", "unsupported trial marker")
    scanner = marker.get("scanner")
    _require(isinstance(scanner, dict), "missing scanner binding")
    folder = _within(root, scanner.get("path"))
    _require(folder != root and folder.is_dir(), "missing separate C1 scanner folder")
    manifest_path = _within(folder, "delivery_manifest.json")
    expected_sha = scanner.get("manifest_sha256")
    _require(isinstance(expected_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha)
             and manifest_path.is_file() and _sha(manifest_path) == expected_sha, "C1 manifest checksum differs; re-extract the complete kit")
    manifest = _json(manifest_path, 2 * 1024 * 1024)
    _require(manifest.get("release") == C1_RELEASE, "scanner is not the reviewed communication-log C1 delivery")
    files = manifest.get("files")
    _require(isinstance(files, list) and len(files) == len(C1_FILES), "incomplete C1 file inventory")
    seen = set()
    for record in files:
        _require(isinstance(record, dict), "invalid C1 file record")
        relative = record.get("path")
        _require(isinstance(relative, str) and relative in C1_FILES and relative not in seen, "unexpected or duplicate C1 file path")
        seen.add(relative)
        path = _within(folder, relative)
        digest = record.get("sha256")
        _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                 and path.is_file() and _sha(path) == digest, "C1 file checksum differs: " + relative)
    _require(seen == C1_FILES, "C1 files are missing")
    for relative in DLLS:
        _pe64(_within(folder, relative))
    return {"project_root": str(root), "scan_repo": str(folder), "inference_results": ""}


def check_environment(*, python_only=False, check_system_runtime=True):
    """No GUI, camera, controller, vendor SDK or network access is performed."""
    checks, errors = {}, []
    windows = platform.system() == "Windows"
    checks["tk_required_for_scanner"] = windows
    checks["python_version"] = list(sys.version_info[:3])
    checks["python_executable"] = sys.executable
    checks["python_64bit"] = struct.calcsize("P") == 8
    if sys.version_info[:2] < (3, 9):
        errors.append("Python 3.9 or newer, 64-bit, is required. The instrument's Python 3.9.18 is supported; no Python upgrade is needed.")
    if not checks["python_64bit"]:
        errors.append("A 64-bit Python is required by the supplied Windows x64 scanner DLLs.")
    for name in (("tkinter",) if python_only else ("tkinter", "numpy", "cv2")):
        try:
            module = importlib.import_module(name)
            checks[name] = {"available": True, "version": str(getattr(module, "__version__", getattr(module, "TkVersion", "stdlib")))}
        except Exception as error:
            checks[name] = {"available": False, "error": str(error)}
            if name != "tkinter" or windows:
                errors.append("Install Python with Tcl/Tk support." if name == "tkinter" else "Run 01_install.bat to install " + name + ".")
    if not python_only and check_system_runtime and windows:
        import ctypes
        # Absolute Windows System32 paths only. Never load a vendor/device DLL
        # here; XIMC initialization remains in its diagnostic scanner process.
        system = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32"
        for name in ("MSVCR120.dll", "MSVCP120.dll"):
            try:
                path = system / name
                if not path.is_file():
                    raise FileNotFoundError(str(path))
                ctypes.WinDLL(str(path))
                checks[name] = {"available": True, "path": str(path)}
            except (OSError, AttributeError) as error:
                checks[name] = {"available": False, "error": str(error)}
                errors.append("Install the official Microsoft Visual C++ 2013 x64 runtime required by the XIMC SDK (" + name + ").")
    checks["hardware_checked"] = False
    checks["network_used"] = False
    return {"ok": not errors, "checks": checks, "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-python", action="store_true", help="Pre-install: Python version, x64 and Tk only")
    group.add_argument("--check-environment", action="store_true", help="Post-install: Python/Tk/NumPy/OpenCV and Windows system runtimes")
    args = parser.parse_args(argv)
    result = check_environment(python_only=args.check_python)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
