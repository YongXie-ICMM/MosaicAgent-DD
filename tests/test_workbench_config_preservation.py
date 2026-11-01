"""Relaunch must retain teacher-selected project/C1 paths, never reset settings."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import student_demo as demo


def config_path(root):
    path = root / 'outputs/workbench/workbench.local.json'
    path.parent.mkdir(parents=True)
    return path


@pytest.mark.parametrize('bom', [False, True])
def test_relaunch_preserves_teacher_c1_project_choices_unknown_fields_and_exact_bytes(tmp_path, monkeypatch, bom):
    path = config_path(tmp_path)
    selected = {'project_root': 'D:\\DD_project',
                'scan_repo': 'D:\\C1_communication_log_scanner',
                'inference_results': 'D:\\reviewed_results',
                'future_preference': {'language': 'zh', 'layout': 'teacher-selected'}}
    original = ('\ufeff' if bom else '') + json.dumps(selected, ensure_ascii=False, indent=3) + '\r\n\r\n'
    path.write_bytes(original.encode('utf-8'))
    before = path.stat()
    # Even a newly written demo pointer must not supersede explicit real inputs.
    latest = tmp_path / 'outputs/demo/latest.json'; latest.parent.mkdir(parents=True)
    latest.write_text('{broken demo pointer should not be inspected')
    def forbidden(*args, **kwargs):
        pytest.fail('Existing settings must not be rewritten or model code loaded')
    monkeypatch.setattr(demo, 'atomic_json', forbidden)
    monkeypatch.setattr(demo, 'load_stages', forbidden)
    assert demo.prepare_workbench_config(tmp_path) == path
    assert path.read_bytes() == original.encode('utf-8')
    assert path.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize('payload', [
    b'{broken', b'\xff\xfe', b'[]', b'null', b'{}',
    b'{"project_root":"p","scan_repo":null,"inference_results":""}',
    b'{"project_root":"p","scan_repo":"c1","inference_results":[]}',
    b'{"project_root":"p","scan_repo":"c1","inference_results":"","extra":NaN}',
])
def test_invalid_existing_settings_fail_before_port_or_launch_without_overwrite(tmp_path, monkeypatch, payload):
    path = config_path(tmp_path); path.write_bytes(payload)
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid settings must not write defaults, bind a port or launch')
    monkeypatch.setattr(demo, 'atomic_json', forbidden)
    monkeypatch.setattr(demo, 'available_port', forbidden)
    monkeypatch.setattr(demo.subprocess, 'call', forbidden)
    with pytest.raises(demo.DemoError, match='原文件已保留'):
        demo.open_workbench(tmp_path, no_browser=True)
    assert path.read_bytes() == payload


def test_existing_config_directory_is_not_overwritten(tmp_path):
    path = config_path(tmp_path); path.mkdir()
    with pytest.raises(demo.DemoError, match='原文件已保留'):
        demo.prepare_workbench_config(tmp_path)
    assert path.is_dir()


def test_broken_existing_config_link_is_preserved_instead_of_replaced(tmp_path):
    path = config_path(tmp_path)
    target = tmp_path / 'missing-teacher-settings.json'
    path.symlink_to(target)
    with pytest.raises(demo.DemoError, match='原文件已保留'):
        demo.prepare_workbench_config(tmp_path)
    assert path.is_symlink() and path.readlink() == target
    assert not target.exists()


def test_fresh_default_settings_remain_portable_and_empty_without_a_demo(tmp_path):
    path = demo.prepare_workbench_config(tmp_path)
    assert json.loads(path.read_text()) == {'project_root': str(tmp_path.resolve()),
        'scan_repo': str(tmp_path.resolve() / 'acquisition'), 'inference_results': ''}


def test_latest_demo_is_used_only_when_creating_settings_for_the_first_time(tmp_path):
    results = tmp_path / 'outputs/demo/synthetic-run/results'; results.mkdir(parents=True)
    (results / 'run_manifest.json').write_text('{}')
    latest = tmp_path / 'outputs/demo/latest.json'
    latest.write_text(json.dumps({'run': 'outputs/demo/synthetic-run'}))
    path = demo.prepare_workbench_config(tmp_path)
    assert json.loads(path.read_text())['inference_results'] == str(results.parent.resolve())
    before = path.read_bytes()
    other = tmp_path / 'outputs/demo/newer-run/results'; other.mkdir(parents=True)
    (other / 'run_manifest.json').write_text('{}')
    latest.write_text(json.dumps({'run': 'outputs/demo/newer-run'}))
    demo.prepare_workbench_config(tmp_path)
    assert path.read_bytes() == before


def test_open_workbench_passes_the_preserved_config_to_current_python(tmp_path, monkeypatch):
    path = config_path(tmp_path)
    original = b'{"project_root":"teacher-project","scan_repo":"C1","inference_results":""}\n'
    path.write_bytes(original)
    calls = []
    monkeypatch.setattr(demo, 'available_port', lambda: 8898)
    monkeypatch.setattr(demo.subprocess, 'call', lambda command, **kw: calls.append((command, kw)) or 0)
    assert demo.open_workbench(tmp_path, no_browser=True) == 0
    command, options = calls[0]
    assert command[0] == sys.executable
    assert command[command.index('--config') + 1] == str(path)
    assert command[command.index('--port') + 1] == '8898'
    assert options['cwd'] == tmp_path.resolve()
    assert path.read_bytes() == original
