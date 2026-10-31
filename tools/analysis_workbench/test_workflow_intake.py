"""Real synthetic PNG/log intake; no sockets, devices, models or source rewrites."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.analysis_workbench.workflow_intake import inspect_transfer
from tools.stitch_setup.test_acquisition_metadata import acquisition, save

REPO = Path(__file__).resolve().parents[2]


def source_bytes(folder):
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob('*') if p.is_file()}


def damage_scan(folder, session, captures, damage):
    if damage == 'running':
        session['status'] = 'running'; save(folder, session, captures)
    elif damage in {'missing_session', 'missing_events'}:
        (folder / ('session.json' if damage == 'missing_session' else 'events.jsonl')).unlink()
    elif damage == 'changed_pixels':
        Image.new('RGB', (1920, 1080), (32, 64, 96)).save(folder / captures[0]['filename'])
    elif damage == 'changed_capture_hash':
        captures[0]['image_sha256'] = '0' * 64; save(folder, session, captures)
    elif damage == 'wrong_photo_count':
        session['photos_saved'] = 3; save(folder, session, captures)
    elif damage == 'missing_capture':
        save(folder, session, captures[:-1])
    else:
        raise AssertionError(damage)


def test_intake_verifies_real_headers_hashes_and_independent_capture_records(acquisition):
    folder, session, captures = acquisition
    before = source_bytes(folder)
    result = inspect_transfer(folder, REPO)
    assert result['status'] == 'verified'
    assert result['path'] == str(folder.resolve())
    assert result['session_id'] == session['session_id']
    assert result['photos'] == result['capture_success_records'] == 4
    assert result['source_images_modified'] is False and result['archives_created'] is False
    assert result['inspection']['acquisition']['matching_selected_capture_records'] == 4
    assert result['inspection']['acquisition']['physical_coverage_verified'] is False
    assert {Path(row['path']).name for row in result['files']} == {'session.json', 'events.jsonl'}
    for row in result['files']:
        assert row['sha256'] == hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
    assert source_bytes(folder) == before


@pytest.mark.parametrize('damage', ['running', 'missing_session', 'missing_events',
    'changed_pixels', 'changed_capture_hash', 'wrong_photo_count', 'missing_capture'])
def test_incomplete_or_changed_scan_cannot_return_verified(acquisition, damage):
    folder, session, captures = acquisition
    damage_scan(folder, session, captures, damage)
    before = source_bytes(folder)
    with pytest.raises(ValueError):
        inspect_transfer(folder, REPO)
    assert source_bytes(folder) == before
    assert not (folder / 'Memory').exists()


def test_one_packed_run_selects_its_explicit_session_without_merging(acquisition, tmp_path):
    folder, _, _ = acquisition
    handover = tmp_path / 'handover.json'
    handover.write_text(json.dumps({'kind': 'scan_handover', 'runs': [{'folder': folder.name}]}))
    before = source_bytes(folder)
    result = inspect_transfer(tmp_path, REPO)
    assert result['path'] == str(folder.resolve()) and result['photos'] == 4
    assert source_bytes(folder) == before


@pytest.mark.parametrize('runs', [[], [{'folder': 'raw'}, {'folder': 'another_scan'}], [{'folder': '../outside'}]])
def test_packed_multiple_missing_or_escaping_runs_are_not_silently_selected(acquisition, tmp_path, runs):
    (tmp_path / 'handover.json').write_text(json.dumps({'kind': 'scan_handover', 'runs': runs}))
    with pytest.raises(ValueError):
        inspect_transfer(tmp_path, REPO)
