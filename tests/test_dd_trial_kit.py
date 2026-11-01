"""Independent trial ZIP integrity checks; synthetic headers and inert code only."""
import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import struct
import sys
import warnings
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import verify_dd_trial_kit as V


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode()


def rows(files):
    return [{'path': name, 'sha256': sha(data), 'bytes': len(data)} for name, data in sorted(files.items())]


@pytest.fixture
def kit(tmp_path, monkeypatch):
    analysis = {name: b'# inert synthetic source\n' for name in (
        '01_install.bat', '01_install.command', '03_open_workbench.bat', '03_open_workbench.command',
        'tools/student_demo.py', 'tools/dd_trial.py', 'tools/verify_dd_trial_kit.py',
        'tools/analysis_workbench/server.py', 'tools/analysis_workbench/static/workflow.html')}
    frozen = {f'runtime_{i:02d}.py': b'# inert frozen runtime\n' for i in range(16)}
    frozen_manifest = encoded({'schema_version': 1, 'files': rows(frozen)})
    analysis[V.FROZEN] = frozen_manifest
    analysis.update({'acquisition/Auto_Scan/' + name: data for name, data in frozen.items()})
    model = b'fake checkpoint; must never deserialize'
    analysis[V.MODEL] = model
    image_manifest = []
    image_zip = io.BytesIO()
    with zipfile.ZipFile(image_zip, 'w') as archive:
        for i in range(2):
            # Header fixture only, not a real measurement or a raster decoder test.
            payload = b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\rIHDR' + struct.pack('>II', 1920, 1080) + bytes([i])
            member = f'synthetic_{i}.png'; archive.writestr(member, payload)
            image_manifest.append({'member': member, 'sha256': sha(payload), 'width': 1920, 'height': 1080})
    analysis[V.IMAGES] = image_zip.getvalue()
    analysis[V.ASSETS] = encoded({'schema_version': 1, 'weights': {'path': V.MODEL, 'sha256': sha(model)},
        'archive': {'path': V.IMAGES, 'sha256': sha(analysis[V.IMAGES])}, 'samples': image_manifest})
    scanner = {name: b'# inert communication fixture\n' for name in (
        'device_diagnostics.py', 'camera_observability.py', 'collect_device_diagnostics.py')}
    scanner.update({f'fixture_{i:02d}.py': b'# not a device\n' for i in range(70)})
    scanner_manifest = encoded({'release': '954-1080p-C1-candidate', 'files': rows(scanner)})
    files = {**analysis, **{'scanner_C1/' + name: data for name, data in scanner.items()}, V.SCANNER: scanner_manifest,
        V.MARKER: encoded({'schema_version': 1, 'kind': 'dd_student_trial',
                          'scanner': {'path': 'scanner_C1', 'manifest_sha256': sha(scanner_manifest)}}),
        V.QUICKSTART: b'%PDF-1.4\nsynthetic manual fixture\n'}
    monkeypatch.setattr(V, 'FROZEN_SHA', sha(frozen_manifest))
    monkeypatch.setattr(V, 'C1_SHA', sha(scanner_manifest))
    monkeypatch.setattr(V, 'MODEL_SHA', sha(model))
    monkeypatch.setattr(V, 'IMAGES_SHA', sha(analysis[V.IMAGES]))
    counter = [0]

    def build(changes=None, *, refresh_inner=True, outer_edit=None, extras=()):
        current = copy.deepcopy(files)
        if changes:
            for name, value in changes.items():
                if value is None: current.pop(name, None)
                else: current[name] = value
        selected = {name: current[name] for name in analysis if name in current}
        inner = {'schema_version': 1, 'kind': 'student_bundle_manifest', 'top_level_folder': V.TOP,
            'credentials_included': False, 'source_commit': 'synthetic-test-only',
            'files': rows(selected), 'file_count': len(selected), 'total_bytes': sum(map(len, selected.values()))}
        if not refresh_inner:
            original_inner = {name: files[name] for name in analysis}
            inner.update(files=rows(original_inner), file_count=len(original_inner), total_bytes=sum(map(len, original_inner.values())))
        current[V.ANALYSIS] = encoded(inner)
        outer = {'schema_version': 1, 'analysis_manifest': V.ANALYSIS, 'scanner_manifest': V.SCANNER,
            'quickstart': V.QUICKSTART, 'files': rows(current), 'file_count': len(current),
            'total_bytes': sum(map(len, current.values()))}
        if outer_edit: outer_edit(outer, current)
        current[V.OUTER] = encoded(outer)
        counter[0] += 1
        path = tmp_path / f'kit {counter[0]}.zip'
        with warnings.catch_warnings(), zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
            warnings.simplefilter('ignore', UserWarning)
            for name, data in current.items(): archive.writestr(V.TOP + '/' + name, data)
            for name, data, mode in extras:
                info = zipfile.ZipInfo(name); info.external_attr = mode << 16
                archive.writestr(info, data)
        return path
    return build, files


def test_valid_trial_checks_three_manifests_assets_and_pdf_without_extracting(kit, monkeypatch):
    build, _ = kit
    def forbidden(*args, **kwargs):
        pytest.fail('Verifier must never extract or execute a bundled program')
    monkeypatch.setattr(zipfile.ZipFile, 'extract', forbidden)
    monkeypatch.setattr(zipfile.ZipFile, 'extractall', forbidden)
    path = build()
    report = V.check_trial_kit(path)
    assert report['ok'] is True and report['credentials_included'] is False
    assert report['scanner_file_count'] == 73 and report['frozen_runtime_file_count'] == 16
    assert report['sha256'] == sha(path.read_bytes())
    assert report['quickstart'] == V.QUICKSTART


@pytest.mark.parametrize('name', [V.MODEL, V.QUICKSTART, 'tools/dd_trial.py',
    'scanner_C1/device_diagnostics.py', 'acquisition/Auto_Scan/runtime_00.py'])
def test_removed_required_component_cannot_be_reported_complete(kit, name):
    build, _ = kit
    with pytest.raises(V.TrialKitError): V.check_trial_kit(build({name: None}))


def test_byte_change_fails_even_when_the_zip_crc_is_valid(kit):
    build, _ = kit
    def alter(outer, current): current['tools/student_demo.py'] = b'# changed after manifest\n'
    with pytest.raises(V.TrialKitError, match='count mismatch|SHA-256 mismatch'):
        V.check_trial_kit(build(outer_edit=alter))


def test_analysis_manifest_is_checked_independently_of_resigned_outer_records(kit):
    build, _ = kit
    with pytest.raises(V.TrialKitError, match='Analysis manifest mismatch'):
        V.check_trial_kit(build({'tools/student_demo.py': b'# changed\n'}, refresh_inner=False))


def test_analysis_code_cannot_be_listed_only_in_the_outer_manifest(kit):
    build, _ = kit
    with pytest.raises(V.TrialKitError, match='outside the analysis manifest'):
        V.check_trial_kit(build({'tools/unbound_extra.py': b'# unbound source\n'}))


def test_resigning_frozen_scanner_manifest_does_not_bypass_release_anchor(kit):
    build, original = kit
    frozen = json.loads(original[V.FROZEN])
    replacement = b'# changed frozen runtime\n'
    row = frozen['files'][0]
    row.update(sha256=sha(replacement), bytes=len(replacement))
    with pytest.raises(V.TrialKitError, match='frozen scanner manifest changed'):
        V.check_trial_kit(build({V.FROZEN: encoded(frozen), 'acquisition/Auto_Scan/' + row['path']: replacement}))


def test_replacing_model_and_resigning_both_manifests_does_not_bypass_original0409_anchor(kit):
    build, original = kit
    asset = json.loads(original[V.ASSETS]); replacement = b'wrong checkpoint'
    asset['weights']['sha256'] = sha(replacement)
    with pytest.raises(V.TrialKitError, match='0409/demo asset binding changed'):
        V.check_trial_kit(build({V.MODEL: replacement, V.ASSETS: encoded(asset)}))


def test_c1_runtime_change_is_caught_independently_of_outer_and_analysis_manifests(kit):
    build, _ = kit
    with pytest.raises(V.TrialKitError, match='C1 scanner file mismatch'):
        V.check_trial_kit(build({'scanner_C1/device_diagnostics.py': b'# changed C1\n'}))


@pytest.mark.parametrize('path', [V.TOP + '/../escape.py', '/absolute.py', 'C:/drive.py',
    V.TOP + '/bad\\path.py', V.TOP + '/CON.txt', V.TOP + '/endswithdot.', V.TOP + '/a//b.py'])
def test_unsafe_and_nonportable_paths_are_rejected_before_extraction(kit, path):
    build, _ = kit
    with pytest.raises(V.TrialKitError, match='path|folder'):
        V.check_trial_kit(build(extras=[(path, b'inert', stat.S_IFREG | 0o644)]))


@pytest.mark.parametrize('path', ['tools/student_demo.py', 'TOOLS/STUDENT_DEMO.PY'])
def test_duplicate_or_windows_case_collision_is_rejected(kit, path):
    build, _ = kit
    with pytest.raises(V.TrialKitError, match='Duplicate or case-colliding'):
        V.check_trial_kit(build(extras=[(V.TOP + '/' + path, b'inert', stat.S_IFREG | 0o644)]))


def test_zip_symlink_is_rejected(kit):
    build, _ = kit
    with pytest.raises(V.TrialKitError, match='Symlink'):
        V.check_trial_kit(build(extras=[(V.TOP + '/linked.py', b'../../outside', stat.S_IFLNK | 0o777)]))


@pytest.mark.parametrize('path', ['.env', '.env.prod', 'config.local.json', 'Memory/events.jsonl',
    'scanner_C1/mosaic_photos/mosaic_r0_c0.png', 'device_diagnostics/run/communications.jsonl',
    'nested/credentials.json'])
def test_private_files_or_real_runtime_data_are_never_accepted_even_if_manifested(kit, path):
    build, _ = kit
    with pytest.raises(V.TrialKitError, match='Private/runtime data'):
        V.check_trial_kit(build({path: b'not a real secret or experiment'}))


def test_unlisted_extra_file_is_rejected(kit):
    build, _ = kit
    with pytest.raises(V.TrialKitError, match='Outer manifest inventory mismatch'):
        V.check_trial_kit(build(extras=[(V.TOP + '/extra.txt', b'inert', stat.S_IFREG | 0o644)]))


@pytest.mark.parametrize('field', ['file_count', 'total_bytes'])
def test_wrong_outer_counts_are_not_trusted(kit, field):
    build, _ = kit
    with pytest.raises(V.TrialKitError, match='count mismatch'):
        V.check_trial_kit(build(outer_edit=lambda outer, _: outer.update({field: outer[field] + 1})))


def test_cli_has_explicit_success_and_failure(kit, capsys):
    build, _ = kit
    assert V.main(['--check', str(build())]) == 0
    assert json.loads(capsys.readouterr().out)['ok'] is True
    assert V.main(['--check', 'nonexistent-trial.zip']) == 1
    assert 'verification failed' in capsys.readouterr().err
