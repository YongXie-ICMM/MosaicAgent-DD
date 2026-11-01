"""DD server integration boundaries using synthetic records and mocked launchers."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.analysis_workbench import server as S
from tools.analysis_workbench.test_workflow_intake import damage_scan, source_bytes
from tools.stitch_setup.test_acquisition_metadata import acquisition

HERE = Path(__file__).resolve().parent


@pytest.fixture
def wb(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    def forbidden(*args, **kwargs):
        raise AssertionError('Tests must not launch devices, models or external programs')
    monkeypatch.setattr(S.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(S.subprocess, 'run', forbidden)
    monkeypatch.setattr(S.urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(S.webbrowser, 'open', forbidden)
    project = tmp_path / 'project'; project.mkdir()
    scanner = tmp_path / 'scanner'; scanner.mkdir()
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'project_root': str(project), 'scan_repo': str(scanner), 'inference_results': ''}))
    workbench = S.Workbench(config)
    monkeypatch.setattr(workbench, '_inspect_scanner', lambda *a: {'package': {'package_files_match_manifest': False}})
    return workbench


def fake_manager(wb, payload):
    manager = SimpleNamespace(directory=wb.memory_root / 'DD_workflow', state=lambda: copy.deepcopy(payload))
    manager.directory.mkdir(parents=True, exist_ok=True)
    wb._dd_workflow = manager
    return manager


def job(identity, kind, source, output, date, *, status='completed', sources=None):
    return {'id': identity, 'kind': kind, 'status': status, 'source_dir': str(source),
            'output_dir': str(output), 'output_path': str(output / 'mosaic.png'),
            'source_files': sources or [], 'created_at': f'2026-10-31T{date}:00+00:00', 'artifacts': []}


def test_memory_lives_in_configured_project_and_scan_project_uses_sibling(wb, acquisition):
    assert wb.memory_root == wb.project / 'Memory'
    assert wb.dd_workflow.directory.is_relative_to(wb.memory_root)
    folder, _, _ = acquisition
    before = source_bytes(folder)
    wb.config['project_root'] = str(folder)
    wb._memory_store = wb._dd_workflow = None
    assert wb.memory_root == folder.parent / 'Memory'
    assert wb.dd_workflow.directory.is_relative_to(folder.parent / 'Memory')
    assert source_bytes(folder) == before and not (folder / 'Memory').exists()


def test_active_batch_keeps_its_descendants_but_excludes_other_jobs_and_prepared(wb, tmp_path):
    scan_a, scan_b = tmp_path / 'scan-A', tmp_path / 'scan-B'
    stitch_out, layer_out = tmp_path / 'stitch-A', tmp_path / 'layers-A'
    rows = [job('spatial-A', 'spatial', '', tmp_path / 'spatial-A', '14:00', sources=[{'path': str(layer_out / 'results/mask.png')}]),
            job('layer-A', 'layers', '', layer_out, '13:00', sources=[{'path': str(stitch_out / 'mosaic.png')}]),
            job('other', 'stitch', scan_b, tmp_path / 'stitch-B', '12:00'),
            job('prefix-collision', 'layers', '', tmp_path / 'irrelevant', '11:30', sources=[{'path': str(tmp_path / 'stitch-A-other/mosaic.png')}]),
            job('stitch-A', 'stitch', scan_a, stitch_out, '11:00')]
    payload = {'jobs': rows, 'prepared': [{'id': 'A', 'kind': 'stitch', 'source_dir': str(scan_a)},
                                         {'id': 'B', 'kind': 'stitch', 'source_dir': str(scan_b)}]}
    manager = fake_manager(wb, payload)
    S.atomic_json(manager.directory / 'handover_intake.json', {'status': 'verified', 'path': str(scan_a)})
    state = wb.workflow_state()
    assert [row['id'] for row in state['active_jobs']] == ['spatial-A', 'layer-A', 'stitch-A']
    assert [row['id'] for row in state['active_prepared']] == ['A']
    assert len(state['jobs']) == 5  # Other batches remain in history, not current-phase proof.
    S.atomic_json(manager.directory / 'handover_intake.json', {'status': 'verified', 'path': str(scan_b)})
    state = wb.workflow_state()
    assert [row['id'] for row in state['active_jobs']] == ['other']
    assert [row['id'] for row in state['active_prepared']] == ['B']


@pytest.mark.parametrize('matches,features,allowed', [
    (False, True, False), (True, False, False), (True, True, True),
])
def test_scanner_requires_both_verified_manifest_and_communication_features(wb, tmp_path, monkeypatch, matches, features, allowed):
    package = tmp_path / 'package'; package.mkdir()
    names = ['device_diagnostics.py', 'camera_observability.py', 'collect_device_diagnostics.py'] if features else ['03Auto_Snake_Scan_Camera_v3.py']
    (package / 'delivery_manifest.json').write_text(json.dumps({'files': [{'path': name} for name in names]}))
    monkeypatch.setattr(wb, '_inspect_scanner', lambda *a: {'folder': str(package), 'package': {'package_files_match_manifest': matches}})
    launch = Mock(return_value={'ok': True, 'message': 'synthetic launcher only'})
    monkeypatch.setattr(wb, 'action', launch)
    if allowed:
        assert wb.workflow_action('scanner', {})['ok'] is True
        launch.assert_called_once_with('scan', None)
    else:
        with pytest.raises(ValueError, match='通信日志版'):
            wb.workflow_action('scanner', {})
        launch.assert_not_called()


@pytest.mark.parametrize('latest_a_status,expected_name', [('completed', 'new-A'), ('failed', 'old-A')])
def test_layers_binds_newest_completed_stitch_of_matching_batch(wb, tmp_path, monkeypatch, latest_a_status, expected_name):
    scan_a, scan_b = tmp_path / 'scan-A', tmp_path / 'scan-B'
    rows = [job('new-B', 'stitch', scan_b, tmp_path / 'new-B', '15:00'),
            job('new-A', 'stitch', scan_a, tmp_path / 'new-A', '14:00', status=latest_a_status),
            job('old-A', 'stitch', scan_a, tmp_path / 'old-A', '13:00')]
    manager = fake_manager(wb, {'jobs': rows, 'prepared': []})
    manager.register_plan = Mock(return_value={'id': 'teacher-plan'})
    manager.start = Mock(return_value={'id': 'layer-job', 'kind': 'layers', 'status': 'starting'})
    S.atomic_json(manager.directory / 'handover_intake.json', {'status': 'verified', 'path': str(scan_a)})
    analysis = importlib.import_module('workflow_analysis')
    prepare = Mock(return_value={'kind': 'layers', 'fixture': True})
    monkeypatch.setattr(analysis, 'prepare_layers', prepare)
    result = wb.workflow_action('layers', {'config_path': str(tmp_path / 'teacher.json')})
    assert result['job']['id'] == 'layer-job'
    assert prepare.call_args.kwargs['expected_mosaic'] == str(tmp_path / expected_name / 'mosaic.png')
    manager.register_plan.assert_called_once_with({'kind': 'layers', 'fixture': True})
    manager.start.assert_called_once_with('teacher-plan')


def test_completed_artifact_is_readable_but_traversal_hidden_and_unfinished_are_not(wb, tmp_path):
    output = wb.memory_root / 'outputs'; output.mkdir(parents=True)
    (output / 'plot.svg').write_text('<svg/>')
    (output / '.env').write_text('dummy test data')
    outside = tmp_path / 'outside.png'; outside.write_bytes(b'not an artifact')
    declared = [{'path': str(output / 'plot.svg'), 'sha256': hashlib.sha256(b'<svg/>').hexdigest()}]
    manager = fake_manager(wb, {'jobs': [{'id': 'ready', 'status': 'completed', 'output_dir': str(output), 'artifacts': declared},
                                      {'id': 'running', 'status': 'running', 'output_dir': str(output)}]})
    assert wb.workflow_artifact('ready', 'plot.svg')[0] == b'<svg/>'
    for identity, name in [('ready', '../outside.png'), ('ready', str(outside)),
                           ('ready', '.env'), ('running', 'plot.svg'), ('missing', 'plot.svg')]:
        with pytest.raises(ValueError):
            wb.workflow_artifact(identity, name)
    link = output / 'outside.png'
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        wb.workflow_artifact('ready', link.name)


def test_completed_output_changed_after_validation_is_not_served_as_verified(wb):
    output = wb.memory_root / 'completed'; output.mkdir(parents=True)
    target = output / 'plot.svg'; target.write_bytes(b'<svg/>')
    declared = [{'path': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}]
    fake_manager(wb, {'jobs': [{'id': 'ready', 'status': 'completed', 'output_dir': str(output), 'artifacts': declared}]})
    assert wb.workflow_artifact('ready', target.name)[0] == b'<svg/>'
    target.write_bytes(b'<svg><text>changed after validation</text></svg>')
    with pytest.raises(ValueError, match='hash|SHA-256|verified|declared|校验|核验|清单|改变'):
        wb.workflow_artifact('ready', target.name)


def test_new_undeclared_json_in_completed_output_is_not_a_verified_artifact(wb):
    output = wb.memory_root / 'completed'; output.mkdir(parents=True)
    target = output / 'unvalidated.json'; target.write_text('{"not":"an output of this job"}')
    fake_manager(wb, {'jobs': [{'id': 'ready', 'status': 'completed', 'output_dir': str(output), 'artifacts': []}]})
    with pytest.raises(ValueError, match='hash|verified|declared|校验|核验|清单'):
        wb.workflow_artifact('ready', target.name)


def test_state_links_only_current_hash_verified_declared_outputs(wb):
    output = wb.memory_root / 'completed'; output.mkdir(parents=True)
    real = output / 'plot.svg'; real.write_text('<svg/>')
    stale = output / 'stale.png'; stale.write_bytes(b'changed since completion')
    undeclared = output / 'unvalidated.png'; undeclared.write_bytes(b'not a declared result')
    records = [{'path': str(real), 'sha256': hashlib.sha256(real.read_bytes()).hexdigest()},
               {'path': str(stale), 'sha256': '0' * 64}]
    fake_manager(wb, {'jobs': [{'id': 'ready', 'status': 'completed', 'output_dir': str(output), 'artifacts': records}], 'prepared': []})
    state = wb.workflow_state()
    assert [row['name'] for row in state['jobs'][0]['artifacts']] == ['plot.svg']
    assert 'unvalidated.png' not in json.dumps(state)


def test_job_log_is_bounded_and_cannot_escape_memory(wb, tmp_path):
    log = wb.memory_root / 'jobs/run/process.log'; log.parent.mkdir(parents=True)
    log.write_bytes(b'line\n' * 100_000)
    outside = tmp_path / 'outside.log'; outside.write_text('outside fixture')
    fake_manager(wb, {'jobs': [{'id': 'run', 'log_path': str(log)},
                               {'id': 'outside', 'log_path': str(outside)}]})
    data = wb.workflow_log('run')
    assert data.endswith(b'line\n') and len(data) <= 256 * 1024
    for identity in ('outside', '../run', 'missing'):
        with pytest.raises(ValueError):
            wb.workflow_log(identity)
    log.unlink(); log.symlink_to(outside)
    with pytest.raises(ValueError):
        wb.workflow_log('run')


def test_verified_intake_snapshots_logs_to_memory_and_preserves_source(wb, acquisition, monkeypatch):
    folder, _, _ = acquisition
    before = source_bytes(folder)
    memory = wb.memory
    snapshot = Mock(wraps=memory.snapshot_acquisition)
    monkeypatch.setattr(memory, 'snapshot_acquisition', snapshot)
    result = wb.workflow_action('handover', {'path': str(folder)})
    assert result['ok'] is True and result['handover']['status'] == 'verified'
    snapshot.assert_called_once_with(str(folder.resolve()))
    saved = S.read_json(wb.dd_workflow.directory / 'handover_intake.json')
    assert saved['input_fingerprint'] == result['handover']['input_fingerprint']
    assert 'memory_snapshot' in saved
    assert source_bytes(folder) == before
    assert not list(wb.memory_root.rglob('mosaic_r*.png'))


@pytest.mark.parametrize('damage', ['running', 'missing_events', 'changed_pixels'])
def test_rejected_intake_does_not_snapshot_or_replace_previous_verified_state(wb, acquisition, monkeypatch, damage):
    folder, session, captures = acquisition
    damage_scan(folder, session, captures, damage)
    previous = wb.dd_workflow.directory / 'handover_intake.json'
    S.atomic_json(previous, {'status': 'verified', 'path': 'earlier-source', 'input_fingerprint': 'earlier-binding'})
    original = previous.read_bytes()
    memory = wb.memory
    snapshot = Mock(side_effect=AssertionError('Invalid intake must not enter Memory snapshot'))
    record = Mock(wraps=memory.record)
    monkeypatch.setattr(memory, 'snapshot_acquisition', snapshot)
    monkeypatch.setattr(memory, 'record', record)
    before = source_bytes(folder)
    with pytest.raises(ValueError):
        wb.workflow_action('handover', {'path': str(folder)})
    snapshot.assert_not_called(); record.assert_not_called()
    assert previous.read_bytes() == original
    assert source_bytes(folder) == before
