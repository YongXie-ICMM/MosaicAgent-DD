"""Complete trial-kit startup: synthetic files only, no devices or SDK loads."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import dd_trial as trial
from tools import student_demo as demo


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def pe64():
    data = bytearray(154)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3c, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<H", data, 132, 0x8664)
    struct.pack_into("<H", data, 152, 0x20b)
    return bytes(data)


@pytest.fixture
def kit(tmp_path):
    root = tmp_path / "DD kit 中文 with spaces & 100%"
    scanner = root / "scanner_C1"
    scanner.mkdir(parents=True)
    records = []
    for relative in sorted(trial.C1_FILES):
        path = scanner / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pe64() if relative in trial.DLLS else b"synthetic fixture only\n")
        records.append({"path": relative, "sha256": digest(path)})
    manifest = scanner / "delivery_manifest.json"
    write_json(manifest, {"release": trial.C1_RELEASE, "files": records})
    write_json(root / trial.MARKER, {"schema_version": 1, "kind": "dd_student_trial",
                                    "scanner": {"path": "scanner_C1", "manifest_sha256": digest(manifest)}})
    return root


def rebind_marker(root):
    marker = json.loads((root / trial.MARKER).read_text())
    marker["scanner"]["manifest_sha256"] = digest(root / "scanner_C1/delivery_manifest.json")
    write_json(root / trial.MARKER, marker)


def test_fresh_trial_automatically_selects_verified_c1_without_hardware(kit, monkeypatch):
    child = Mock(side_effect=AssertionError("No child process"))
    monkeypatch.setattr(demo.subprocess, "Popen", child)
    monkeypatch.setattr(demo.subprocess, "run", child)
    monkeypatch.setattr(ctypes, "WinDLL", Mock(side_effect=AssertionError("No SDK DLL load")), raising=False)
    config_path = demo.prepare_workbench_config(kit)
    config = json.loads(config_path.read_text())
    assert config == {"project_root": str(kit.resolve()), "scan_repo": str((kit / "scanner_C1").resolve()), "inference_results": ""}
    assert not child.called
    assert not (kit / "acquisition").exists()


def test_existing_teacher_settings_are_byte_preserved_even_with_bad_trial_marker(kit):
    path = kit / "outputs/workbench/workbench.local.json"
    original = b'{"project_root":"D:/real experiment", "scan_repo":"D:/approved C1", "inference_results":"D:/review"}\n'
    path.parent.mkdir(parents=True)
    path.write_bytes(original)
    (kit / trial.MARKER).write_text("malformed marker, not read for an existing teacher config")
    assert demo.prepare_workbench_config(kit) == path.resolve()
    assert path.read_bytes() == original


@pytest.mark.parametrize("case", ["malformed", "wrong_kind", "schema_bool", "absolute", "parent", "windows_drive", "backslash", "manifest_hash", "missing_feature", "changed_feature", "unexpected_secret", "duplicate", "bad_dll"])
def test_invalid_marker_or_scanner_blocks_without_old_scanner_fallback(kit, case):
    marker_path = kit / trial.MARKER
    marker = json.loads(marker_path.read_text())
    manifest_path = kit / "scanner_C1/delivery_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if case == "malformed":
        marker_path.write_text("{")
    elif case == "wrong_kind":
        marker["kind"] = "other"
    elif case == "schema_bool":
        marker["schema_version"] = True
    elif case in ("absolute", "parent", "windows_drive", "backslash"):
        marker["scanner"]["path"] = {"absolute": "/elsewhere", "parent": "../scanner", "windows_drive": "C:/scanner", "backslash": "scanner_C1\\drivers"}[case]
    elif case == "manifest_hash":
        marker["scanner"]["manifest_sha256"] = "0" * 64
    elif case == "missing_feature":
        (kit / "scanner_C1/device_diagnostics.py").unlink()
    elif case == "changed_feature":
        (kit / "scanner_C1/Camera_v2.py").write_text("changed after delivery")
    elif case == "unexpected_secret":
        manifest["files"][0]["path"] = ".env"
        write_json(manifest_path, manifest)
    elif case == "duplicate":
        manifest["files"][1] = manifest["files"][0]
        write_json(manifest_path, manifest)
    elif case == "bad_dll":
        dll = kit / "scanner_C1/drivers/libximc.dll"
        data = bytearray(dll.read_bytes())
        struct.pack_into("<H", data, 132, 0x14c)
        dll.write_bytes(data)
        next(item for item in manifest["files"] if item["path"] == "drivers/libximc.dll")["sha256"] = digest(dll)
        write_json(manifest_path, manifest)
    if case != "malformed":
        write_json(marker_path, marker)
    if case in ("unexpected_secret", "duplicate", "bad_dll"):
        rebind_marker(kit)
    with pytest.raises(demo.DemoError, match="validation failed"):
        demo.prepare_workbench_config(kit)
    assert not (kit / "outputs/workbench/workbench.local.json").exists()


def test_missing_marker_retains_original_fresh_checkout_behavior(tmp_path):
    root = tmp_path / "ordinary analysis checkout"
    root.mkdir()
    path = demo.prepare_workbench_config(root)
    assert json.loads(path.read_text())["scan_repo"] == str(root / "acquisition")


def test_relative_marker_is_rebased_after_usb_copy_before_initial_launch(kit, tmp_path):
    import shutil
    moved = tmp_path / "second USB folder 中文"
    shutil.copytree(kit, moved)
    settings = trial.initial_config(moved)
    assert settings["scan_repo"] == str((moved / "scanner_C1").resolve())
    assert str(kit) not in json.dumps(settings)


def test_marker_symlink_is_not_followed(kit):
    marker = kit / trial.MARKER
    real = kit / "copied_marker.json"
    marker.rename(real)
    marker.symlink_to(real)
    with pytest.raises(ValueError, match="linked"):
        trial.initial_config(kit)


def test_scanner_symlink_is_not_followed(kit):
    scanner = kit / "scanner_C1"
    real = kit / "moved_scanner"
    scanner.rename(real)
    scanner.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="linked"):
        trial.initial_config(kit)


def fake_import(name):
    assert name in ("tkinter", "numpy", "cv2")
    return SimpleNamespace(__version__="synthetic", TkVersion=8.6)


def test_pre_pip_check_uses_only_python_and_tk_no_sdk_or_array_import(monkeypatch):
    names = []
    def load(name):
        names.append(name)
        return fake_import(name)
    monkeypatch.setattr(trial.importlib, "import_module", load)
    monkeypatch.setattr(trial.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ctypes, "WinDLL", Mock(side_effect=AssertionError("No DLLs before pip")), raising=False)
    result = trial.check_environment(python_only=True)
    assert result["ok"] and names == ["tkinter"]
    assert result["checks"]["hardware_checked"] is False


def test_python38_or_32bit_reports_actionable_failure(monkeypatch):
    monkeypatch.setattr(trial.importlib, "import_module", fake_import)
    monkeypatch.setattr(trial.sys, "version_info", (3, 8, 18))
    monkeypatch.setattr(trial.struct, "calcsize", lambda _: 4)
    result = trial.check_environment(python_only=True)
    assert not result["ok"]
    assert any("3.9" in error for error in result["errors"])
    assert any("64-bit" in error for error in result["errors"])


def test_instrument_python3918_64bit_passes_without_upgrade(monkeypatch):
    monkeypatch.setattr(trial.importlib, "import_module", fake_import)
    monkeypatch.setattr(trial.sys, "version_info", (3, 9, 18))
    monkeypatch.setattr(trial.struct, "calcsize", lambda _: 8)
    result = trial.check_environment(python_only=True)
    assert result["ok"] and result["errors"] == []
    assert result["checks"]["python_version"] == [3, 9, 18]


def test_windows_post_install_loads_only_system_vc_runtime_not_scanner_dlls(tmp_path, monkeypatch):
    system = tmp_path / "Windows/System32"
    system.mkdir(parents=True)
    for name in ("MSVCR120.dll", "MSVCP120.dll"):
        (system / name).write_bytes(b"synthetic OS runtime; never executed")
    monkeypatch.setenv("SystemRoot", str(system.parent))
    monkeypatch.setattr(trial.platform, "system", lambda: "Windows")
    monkeypatch.setattr(trial.importlib, "import_module", fake_import)
    load = Mock()
    monkeypatch.setattr(ctypes, "WinDLL", load, raising=False)
    result = trial.check_environment()
    assert result["ok"]
    assert [Path(call.args[0]) for call in load.call_args_list] == [system / "MSVCR120.dll", system / "MSVCP120.dll"]
    assert all("ximc" not in call.args[0].lower() and "amcam" not in call.args[0].lower() for call in load.call_args_list)


def test_missing_windows_system_runtime_is_explicit_not_a_motion_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("SystemRoot", str(tmp_path / "missing Windows"))
    monkeypatch.setattr(trial.platform, "system", lambda: "Windows")
    monkeypatch.setattr(trial.importlib, "import_module", fake_import)
    result = trial.check_environment()
    assert not result["ok"]
    assert any("Visual C++ 2013 x64" in error for error in result["errors"])
    assert result["checks"]["hardware_checked"] is False


def test_macos_analysis_does_not_require_scanner_tk_or_vc(monkeypatch):
    def load(name):
        if name == "tkinter":
            raise ImportError("Optional macOS Tk absent")
        return fake_import(name)
    monkeypatch.setattr(trial.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(trial.importlib, "import_module", load)
    monkeypatch.setattr(ctypes, "WinDLL", Mock(side_effect=AssertionError("No Windows DLL on macOS")), raising=False)
    result = trial.check_environment()
    assert result["ok"] and not result["checks"]["tk_required_for_scanner"]
    assert result["checks"]["tkinter"]["available"] is False


def test_cli_returns_structured_success_and_failure(monkeypatch, capsys):
    monkeypatch.setattr(trial, "check_environment", lambda **_: {"ok": True, "checks": {}, "errors": []})
    assert trial.main(["--check-python"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    monkeypatch.setattr(trial, "check_environment", lambda **_: {"ok": False, "checks": {}, "errors": ["fixture"]})
    assert trial.main(["--check-environment"]) == 1
    assert json.loads(capsys.readouterr().out)["errors"] == ["fixture"]
