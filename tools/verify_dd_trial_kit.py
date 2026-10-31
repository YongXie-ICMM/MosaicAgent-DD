#!/usr/bin/env python3
"""Independently verify the DD student trial ZIP without extraction or devices.

Only the standard library is used. Release anchors intentionally bind the known
0409 assets, frozen 16-file DD scanner export and separate 73-file C1 candidate.
Updating those components requires reviewing and updating these anchors.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import unicodedata
import zipfile

TOP = 'MosaicAgent-DD'
OUTER = 'trial_kit_manifest.json'
ANALYSIS = 'bundle_manifest.json'
MARKER = 'trial_kit.json'
SCANNER = 'scanner_C1/delivery_manifest.json'
QUICKSTART = '00_学生试跑极简说明.pdf'
FROZEN = 'acquisition/Auto_Scan/delivery_manifest.json'
ASSETS = 'data/demo/assets.json'
MODEL = 'weights/model_0409_all.pth'
IMAGES = 'data/demo/source_images.zip'
MODEL_SHA = 'b771fc9cfaa8fdc1eba9d680d7c3b678890a903ddf12f2073b5a8c957b64a022'
IMAGES_SHA = 'bae41a3cd7e9536eb5ef4e9f80d3396bba6717df3f58ed16646e304bbe829ae5'
FROZEN_SHA = 'a963ad72cfabb794b9f142d55ccbdb6e0099bb2f55dfb7fad944ec497800115f'
C1_SHA = '32cfb50520b8981ee96c2420184bbc9e1bef50e47db4f9ebab179a720fe5bee1'
MAX_FILES = 10000
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
HASH = re.compile(r'[0-9a-fA-F]{64}\Z')
RESERVED = {'con', 'prn', 'aux', 'nul', *{f'com{i}' for i in range(1, 10)}, *{f'lpt{i}' for i in range(1, 10)}}
PRIVATE_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'memory',
    'outputs', 'dist', 'workbench_history', 'mosaic_photos', 'device_diagnostics',
    'camera_history', 'shared_history', 'review_candidates', 'scan_runs',
    'kimi_cache', '_kimi_cache', '_stitch_work', 'node_modules'}
PRIVATE_NAMES = {'credentials.json', 'service_account.json', 'token.json', 'id_rsa', 'id_ed25519'}


class TrialKitError(ValueError):
    """The trial archive is incomplete, mixed, unsafe or differs from its records."""


def _require(condition, message):
    if not condition:
        raise TrialKitError(message)


def _safe_relative(value):
    _require(isinstance(value, str) and value and '\\' not in value and not value.startswith('/'),
             'Unsafe archive path')
    parts = value.split('/')
    _require(all(part and part not in ('.', '..') for part in parts), f'Unsafe archive path: {value}')
    for part in parts:
        _require(not any(ord(c) < 32 or ord(c) == 127 or c in '<>:"|?*' for c in part)
                 and not part.endswith((' ', '.')) and part.split('.')[0].casefold() not in RESERVED,
                 f'Nonportable Windows archive path: {value}')
    return value


def _key(value):
    return unicodedata.normalize('NFC', value).casefold()


def _private_path(relative):
    parts = PurePosixPath(relative).parts
    name = parts[-1].casefold()
    if any(part.casefold() in PRIVATE_DIRS for part in parts[:-1]):
        return True
    if name != '.env.example' and (name == '.env' or name.startswith('.env.') or name.endswith('.env')):
        return True
    return name.endswith('.local.json') or name in PRIVATE_NAMES or name.endswith(('.pem', '.key'))


def _json(data, label):
    _require(len(data) <= MAX_JSON_BYTES, f'Oversized JSON record: {label}')
    def invalid(value):
        raise ValueError('Nonfinite JSON value')
    try:
        value = json.loads(data.decode('utf-8-sig'), parse_constant=invalid)
    except (ValueError, UnicodeError) as exc:
        raise TrialKitError(f'Invalid JSON record: {label}') from exc
    _require(isinstance(value, dict), f'Expected JSON object: {label}')
    return value


def _records(manifest, label):
    rows = manifest.get('files')
    _require(isinstance(rows, list) and bool(rows), f'Missing file records: {label}')
    result, identities = {}, set()
    for row in rows:
        _require(isinstance(row, dict), f'Invalid file record: {label}')
        name = _safe_relative(row.get('path'))
        _require(_key(name) not in identities, f'Duplicate manifest path: {label}/{name}')
        identities.add(_key(name))
        _require(isinstance(row.get('sha256'), str) and HASH.fullmatch(row['sha256']) is not None,
                 f'Invalid SHA-256: {label}/{name}')
        _require(type(row.get('bytes')) is int and row['bytes'] >= 0,
                 f'Invalid byte count: {label}/{name}')
        result[name] = row
    if 'file_count' in manifest:
        _require(type(manifest['file_count']) is int and manifest['file_count'] == len(result),
                 f'File count mismatch: {label}')
    if 'total_bytes' in manifest:
        _require(type(manifest['total_bytes']) is int and manifest['total_bytes'] == sum(r['bytes'] for r in result.values()),
                 f'Total byte count mismatch: {label}')
    return result


def _digest_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def check_trial_kit(zip_path):
    """Return a verification report; never extract, import a model or run code."""
    path = Path(zip_path)
    _require(path.is_file(), 'Trial ZIP does not exist')
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            _require(len(infos) <= MAX_FILES, 'Too many archive members')
            names, identities, files, total = set(), set(), {}, 0
            for info in infos:
                name = info.filename[:-1] if info.is_dir() else info.filename
                _safe_relative(name)
                _require(name == TOP or name.startswith(TOP + '/'), 'Unexpected top-level archive folder')
                _require(_key(name) not in identities, f'Duplicate or case-colliding ZIP member: {name}')
                identities.add(_key(name)); names.add(info.filename)
                mode = stat.S_IFMT(info.external_attr >> 16)
                _require(mode in (0, stat.S_IFREG, stat.S_IFDIR) and not stat.S_ISLNK(info.external_attr >> 16),
                         f'Symlink or special ZIP member: {name}')
                _require(not (info.flag_bits & 1), 'Encrypted archives are not supported')
                _require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), 'Unsupported ZIP compression')
                _require(info.file_size <= MAX_MEMBER_BYTES, f'Oversized archive member: {name}')
                total += info.file_size
                _require(total <= MAX_TOTAL_BYTES, 'Oversized trial archive')
                if info.is_dir():
                    continue
                _require(name != TOP, 'Top-level folder is a file')
                relative = name[len(TOP) + 1:]
                _require(not _private_path(relative), f'Private/runtime data must not be shipped: {relative}')
                files[relative] = info
            _require(OUTER in files, 'Missing outer trial manifest')
            _require(archive.testzip() is None, 'ZIP CRC check failed')

            def data(relative):
                _require(relative in files, f'Missing required file: {relative}')
                return archive.read(files[relative])
            def record(relative):
                return _json(data(relative), relative)

            outer = record(OUTER)
            _require(type(outer.get('schema_version')) is int and outer['schema_version'] == 1,
                     'Unsupported trial manifest schema')
            for key, expected in [('analysis_manifest', ANALYSIS), ('scanner_manifest', SCANNER), ('quickstart', QUICKSTART)]:
                _require(outer.get(key) == expected, f'Unexpected {key} in trial manifest')
            _require(type(outer.get('file_count')) is int and type(outer.get('total_bytes')) is int,
                     'Outer manifest file/byte counts are required')
            outer_rows = _records(outer, OUTER)
            _require(set(outer_rows) == set(files) - {OUTER}, 'Outer manifest inventory mismatch (missing or unlisted files)')
            digests = {}
            for relative, row in outer_rows.items():
                _require(files[relative].file_size == row['bytes'], f'Byte count mismatch: {relative}')
                with archive.open(files[relative]) as stream:
                    digests[relative] = _digest_stream(stream)
                _require(digests[relative] == row['sha256'].lower(), f'SHA-256 mismatch: {relative}')

            inner = record(ANALYSIS)
            _require(inner.get('schema_version') == 1 and inner.get('kind') == 'student_bundle_manifest',
                     'Invalid analysis bundle manifest')
            _require(inner.get('top_level_folder') == TOP and inner.get('credentials_included') is False,
                     'Analysis bundle must be credential-free')
            analysis_rows = _records(inner, ANALYSIS)
            _require(type(inner.get('file_count')) is int and inner['file_count'] == len(analysis_rows),
                     'Analysis manifest file count mismatch')
            for relative, row in analysis_rows.items():
                _require(relative in outer_rows and relative not in (OUTER, ANALYSIS, MARKER, QUICKSTART)
                         and not relative.startswith('scanner_C1/'), 'Invalid analysis manifest subset')
                _require(row['sha256'].lower() == digests[relative] and row['bytes'] == files[relative].file_size,
                         f'Analysis manifest mismatch: {relative}')
            outer_only = {ANALYSIS, MARKER, QUICKSTART, '00_开始这里.txt'}
            _require(all(name in analysis_rows or name in outer_only or name.startswith('scanner_C1/')
                         for name in outer_rows), 'Analysis files outside the analysis manifest')
            for required in ('01_install.bat', '01_install.command', '03_open_workbench.bat', '03_open_workbench.command',
                'tools/student_demo.py', 'tools/dd_trial.py', 'tools/verify_dd_trial_kit.py',
                'tools/analysis_workbench/server.py', 'tools/analysis_workbench/static/workflow.html', ASSETS, MODEL, IMAGES, FROZEN):
                _require(required in analysis_rows, f'Analysis bundle missing required file: {required}')

            marker = record(MARKER)
            _require(marker.get('schema_version') == 1 and marker.get('kind') == 'dd_student_trial', 'Invalid trial marker')
            scanner = marker.get('scanner')
            _require(isinstance(scanner, dict) and scanner.get('path') == 'scanner_C1'
                     and scanner.get('manifest_sha256') == digests[SCANNER], 'Trial scanner marker mismatch')
            _require(digests[SCANNER] == C1_SHA, 'C1 scanner manifest differs from the frozen candidate')
            c1 = record(SCANNER)
            _require(c1.get('release') == '954-1080p-C1-candidate', 'Wrong C1 scanner release')
            c1_rows = _records(c1, SCANNER)
            _require(len(c1_rows) == 73, 'C1 must contain the frozen 73 manifest files')
            selected_c1 = {relative[len('scanner_C1/'):] for relative in files if relative.startswith('scanner_C1/')}
            _require(selected_c1 == set(c1_rows) | {'delivery_manifest.json'}, 'C1 scanner inventory mismatch')
            for relative, row in c1_rows.items():
                name = 'scanner_C1/' + relative
                _require(digests[name] == row['sha256'].lower() and files[name].file_size == row['bytes'],
                         f'C1 scanner file mismatch: {relative}')
            _require({'device_diagnostics.py', 'camera_observability.py', 'collect_device_diagnostics.py'} <= c1_rows.keys(),
                     'C1 communication diagnostics modules missing')

            _require(digests[FROZEN] == FROZEN_SHA, 'DD frozen scanner manifest changed')
            frozen = record(FROZEN)
            frozen_rows = _records(frozen, FROZEN)
            _require(len(frozen_rows) == 16, 'DD frozen scanner must contain 16 runtime files')
            for relative, row in frozen_rows.items():
                name = 'acquisition/Auto_Scan/' + relative
                _require(name in analysis_rows and digests[name] == row['sha256'].lower()
                         and files[name].file_size == row['bytes'], f'DD frozen scanner runtime changed: {relative}')

            assets = record(ASSETS)
            _require(assets.get('schema_version') == 1, 'Invalid trusted asset manifest')
            for key, relative, anchor in [('weights', MODEL, MODEL_SHA), ('archive', IMAGES, IMAGES_SHA)]:
                item = assets.get(key)
                _require(isinstance(item, dict) and item.get('path') == relative
                         and item.get('sha256') == anchor and digests[relative] == anchor,
                         f'Original 0409/demo asset binding changed: {relative}')
            samples = assets.get('samples')
            _require(isinstance(samples, list) and len(samples) == 2, 'Expected two original demo images')
            with zipfile.ZipFile(io.BytesIO(data(IMAGES))) as source:
                members = source.infolist()
                _require(len(members) == 2 and len({i.filename for i in members}) == 2,
                         'Demo image ZIP inventory mismatch')
                expected = {row.get('member') for row in samples if isinstance(row, dict)}
                _require({i.filename for i in members} == expected, 'Demo image members do not match trusted samples')
                for sample in samples:
                    payload = source.read(sample['member'])
                    _require(hashlib.sha256(payload).hexdigest() == sample.get('sha256'), 'Original demo image hash mismatch')
                    _require(payload.startswith(b'\x89PNG\r\n\x1a\n') and len(payload) >= 24,
                             'Original demo image must be PNG')
                    size = list(struct.unpack('>II', payload[16:24]))
                    _require(size == [sample.get('width'), sample.get('height')] == [1920, 1080],
                             'Original demo image dimensions changed')
            _require(data(QUICKSTART).startswith(b'%PDF-'), 'Quick-start PDF is missing or invalid')
            report = {'ok': True, 'bundle': str(path.resolve()), 'bytes': path.stat().st_size,
                'file_count': len(files), 'analysis_file_count': len(analysis_rows),
                'scanner_file_count': len(c1_rows), 'frozen_runtime_file_count': len(frozen_rows),
                'source_commit': inner.get('source_commit'), 'credentials_included': False,
                'quickstart': QUICKSTART, 'quickstart_sha256': digests[QUICKSTART],
                'model_sha256': digests[MODEL], 'scanner_manifest_sha256': digests[SCANNER]}
    except TrialKitError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile, RuntimeError) as exc:
        raise TrialKitError('Cannot verify trial ZIP: ' + type(exc).__name__) from exc
    with path.open('rb') as stream:
        report['sha256'] = _digest_stream(stream)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--check', required=True, type=Path, metavar='ZIP')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(check_trial_kit(args.check), ensure_ascii=False, indent=2))
        return 0
    except TrialKitError as exc:
        print('Trial kit verification failed: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
