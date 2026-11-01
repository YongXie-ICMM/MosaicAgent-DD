"""Append-only DD workflow memory and bounded, content-addressed scan-log copies.

Raw images, DLLs, credentials and experimental source directories are never moved
or copied wholesale. Session histories remain distinct; shared process diagnostics
are copied once per content hash. A snapshot is a bounded live-file observation,
not an atomic capture, a failure finding or proof of physical travel.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import threading
import uuid

MIB = 1024 * 1024
MAX_STRUCTURED_FILE_BYTES = 16 * MIB
MAX_STRUCTURED_TOTAL_BYTES = 128 * MIB
MAX_SDK_TAIL_BYTES = 32 * MIB
MAX_POINTER_BYTES = 64 * 1024
MAX_FILES = 500
MAX_DIRECTORY_ENTRIES = 2000
MAX_INDEX_BYTES = 8 * MIB
SESSION_FILES = (
    'session.json', 'events.jsonl', 'sharpness.csv', 'device_diagnostics.json',
    'auto_scan_log.csv', 'device_fault_checkpoint.json', 'fault_recovery_checkpoint.json',
    'sharpness_statistics.csv', 'sharpness_statistics_summary.txt',
)
SNAPSHOT_FILES = (
    '03Auto_Snake_Scan_Camera_v3.py', 'Camera_v2.py', 'camera_session_history.py',
    'scan_review.py', 'scan_preview.py', 'shared_history.py', 'history_viewer.py',
    'scan_log_resume.py', 'acquisition_contract.py', 'scan_event_schema.py',
    'device_diagnostics.py', 'camera_observability.py', 'collect_device_diagnostics.py',
    'delivery_manifest.json',
)
DIAGNOSTIC_NAME = re.compile(r'(?:communications\.jsonl(?:\.[1-4])?|incident_[A-Za-z0-9_-]+\.json|incident_[A-Za-z0-9_-]+\.post\.jsonl)\Z')
SENSITIVE = re.compile(r'(?:^\.|secret|credential|password|api[_-]?key|token)', re.I)
IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$')
_LOCKS, _LOCK_GUARD = {}, threading.Lock()


class MemoryStoreError(ValueError):
    pass


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def _absolute(path):
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _has_symlink(path):
    for value in (_absolute(path), *_absolute(path).parents):
        try:
            if stat.S_ISLNK(value.lstat().st_mode):
                return True
        except FileNotFoundError:
            continue
    return False


def _path_check(path):
    value = _absolute(path)
    if '..' in Path(os.fspath(path)).parts or _has_symlink(value):
        raise MemoryStoreError('Symlink paths and path traversal are not stored / 不接受符号链接或越界路径')
    return value


def _state(st):
    return {'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'ctime_ns': st.st_ctime_ns,
            'device': st.st_dev, 'inode': st.st_ino}


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_segment(path, limit, tail=False):
    """Bound reads to initial fd size; capture replacements/appends explicitly."""
    path = _path_check(path)
    before_path = path.stat()
    if not stat.S_ISREG(before_path.st_mode):
        raise MemoryStoreError('Only regular files are copied')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (before_path.st_dev, before_path.st_ino):
            raise MemoryStoreError('Source changed before opening')
        if before.st_size > limit and not tail:
            raise MemoryStoreError('Structured file exceeds the per-file or total size limit')
        offset = max(0, before.st_size - limit) if tail else 0
        expected = min(limit, max(0, before.st_size - offset))
        stream.seek(offset)
        pieces, remaining = [], expected
        while remaining:
            part = stream.read(min(65536, remaining))
            if not part:
                break
            pieces.append(part)
            remaining -= len(part)
        after = os.fstat(stream.fileno())
    data = b''.join(pieces)
    try:
        after_path = path.stat()
        replaced = (before.st_dev, before.st_ino) != (after_path.st_dev, after_path.st_ino)
    except OSError:
        replaced = True
    changed = _state(before) != _state(after) or replaced or len(data) != expected
    return data, {'source': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
                  'source_before': _state(before), 'source_after': _state(after),
                  'source_changed_during_copy': changed, 'copy_unstable': changed,
                  'path_replaced_during_copy': replaced, 'snapshot_atomic': False,
                  'source_size_at_open': before.st_size, 'copied_bytes': len(data),
                  'start_offset_bytes': offset, 'end_offset_bytes_exclusive': offset + len(data),
                  'truncated': bool(offset or len(data) != before.st_size)}


def _small_json(path, limit=MAX_POINTER_BYTES):
    data, _ = _read_segment(path, limit)
    try:
        record = json.loads(data)
    except (ValueError, UnicodeError) as error:
        raise MemoryStoreError('Invalid JSON record') from error
    if not isinstance(record, dict):
        raise MemoryStoreError('Expected a JSON object')
    return record


def _clean(value, depth=0):
    """Keep operational metadata bounded; never serialize arrays/bytes or keys."""
    if depth > 6:
        return '<depth limit>'
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float('inf') else None
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        return value[:8000]
    if isinstance(value, dict):
        return {str(key)[:160]: '<redacted>' if SENSITIVE.search(str(key)) else _clean(item, depth + 1)
                for key, item in list(value.items())[:128]}
    if isinstance(value, (list, tuple)):
        return [_clean(item, depth + 1) for item in value[:256]]
    return '<non-metadata value omitted>'


def _identifier(value, prefix):
    if isinstance(value, str) and IDENTIFIER.fullmatch(value) and value not in ('.', '..'):
        return value
    return prefix + '-' + hashlib.sha256(str(value).encode()).hexdigest()[:20]


def _pointer_path(value, base=None):
    """Resolve only recorded native paths and defined relative bases; no relocation."""
    if not isinstance(value, str) or not value:
        return None, 'path_not_recorded'
    if '\x00' in value:
        return None, 'invalid_path_rejected'
    if '..' in value.replace('\\', '/').split('/'):
        return None, 'path_traversal_rejected'
    windows = PureWindowsPath(value)
    if windows.drive and not windows.is_absolute():
        return None, 'ambiguous_drive_relative_path_rejected'
    if os.name != 'nt' and (windows.drive or value.startswith('\\')):
        return None, 'remote_windows_path_unavailable'
    path = Path(value).expanduser()
    if not path.is_absolute():
        if base is None:
            return None, 'relative_path_without_recorded_base'
        path = base / path
    try:
        path = _path_check(path)
    except (OSError, ValueError):
        return None, 'symlink_or_unsafe_path_rejected'
    if not path.exists():
        return None, 'recorded_path_unavailable'
    return path, None


class MemoryStore:
    def __init__(self, root):
        self.root = _path_check(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise MemoryStoreError('Memory root must be a directory')
        with _LOCK_GUARD:
            self.lock = _LOCKS.setdefault(str(self.root), threading.RLock())
        with self._locked():
            if not (self.root / 'index.json').exists():
                self._save_index({'schema_version': 1, 'kind': 'DD_workflow_memory',
                                  'created_at_utc': _utc(), 'event_count': 0,
                                  'steps': {}, 'acquisitions': {}, 'device_runs': {},
                                  'legacy_references': {}})
            else:
                self._index()

    @contextmanager
    def _locked(self):
        with self.lock:
            path = _path_check(self.root / '.memory.lock')
            fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise MemoryStoreError('Invalid memory lock')
                if os.name == 'nt':
                    import msvcrt
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b'0')
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                if os.name == 'nt':
                    import msvcrt
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def _index(self):
        record = _small_json(self.root / 'index.json', MAX_INDEX_BYTES)
        if (record.get('kind') != 'DD_workflow_memory'
                or type(record.get('schema_version')) is not int or record['schema_version'] != 1
                or type(record.get('event_count')) is not int or record['event_count'] < 0
                or not all(isinstance(record.get(key), dict) for key in
                           ('steps', 'acquisitions', 'device_runs', 'legacy_references'))):
            raise MemoryStoreError('Existing Memory index is invalid; it was not replaced')
        return record

    def _save_index(self, record):
        path = _path_check(self.root / 'index.json')
        temp = self.root / ('index.' + uuid.uuid4().hex + '.tmp')
        try:
            with temp.open('x', encoding='utf-8') as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def record(self, step, status, **fields):
        if not all(isinstance(v, str) and 0 < len(v) <= 120 and not re.search(r'[\x00-\x1f]', v)
                   for v in (step, status)):
            raise MemoryStoreError('Step/status must be short text labels')
        record = {'event_id': uuid.uuid4().hex, 'timestamp_utc': _utc(),
                  'step': step, 'status': status, 'fields': _clean(fields)}
        encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        if len(encoded) > 128 * 1024:
            raise MemoryStoreError('Memory event metadata exceeds128KiB')
        with self._locked():
            index = self._index()
            path = _path_check(self.root / 'events.jsonl')
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
            with os.fdopen(os.open(path, flags, 0o600), 'ab') as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise MemoryStoreError('Memory events must be a regular file')
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            index['event_count'] += 1
            index['updated_at_utc'] = record['timestamp_utc']
            index['steps'][step] = {'status': status, 'event_id': record['event_id'],
                                    'timestamp_utc': record['timestamp_utc']}
            self._save_index(index)
        return record

    def _store_blob(self, directory, name, data):
        digest = hashlib.sha256(data).hexdigest()
        path = directory / (digest + '__' + name[:120])
        with self._locked():
            _path_check(directory)
            directory.mkdir(parents=True, exist_ok=True)
            _path_check(path)
            reused = path.exists()
            if reused:
                if not path.is_file() or path.stat().st_size != len(data) or _sha_file(path) != digest:
                    raise MemoryStoreError('Existing Memory object changed; refusing to overwrite history')
            else:
                with path.open('xb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
        return {'path': str(path), 'relative_path': str(path.relative_to(self.root)),
                'sha256': digest, 'bytes': len(data), 'reused': reused}

    def _store_manifest(self, folder, record):
        # Observation timestamps are kept in events; the immutable snapshot key is
        # determined by sources, copied content and availability, not by click time.
        key_record = record
        def stable(value):
            if isinstance(value, dict):
                return {k: stable(v) for k, v in value.items() if k not in ('reused',)}
            if isinstance(value, list):
                return [stable(v) for v in value]
            return value
        fingerprint = hashlib.sha256(json.dumps(stable(key_record), sort_keys=True,
                                                ensure_ascii=False).encode()).hexdigest()
        path = folder / 'snapshots' / (fingerprint + '.json')
        with self._locked():
            _path_check(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open('x', encoding='utf-8') as stream:
                    json.dump({**record, 'schema_version': 1, 'created_at_utc': _utc(),
                               'snapshot_fingerprint': fingerprint}, stream, ensure_ascii=False, indent=2)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
            else:
                existing = _small_json(path, MAX_INDEX_BYTES)
                payload = {k: v for k, v in existing.items()
                           if k not in ('schema_version', 'created_at_utc', 'snapshot_fingerprint')}
                existing_fingerprint = hashlib.sha256(json.dumps(stable(payload), sort_keys=True,
                                                        ensure_ascii=False).encode()).hexdigest()
                if existing.get('snapshot_fingerprint') != fingerprint or existing_fingerprint != fingerprint:
                    raise MemoryStoreError('Existing Memory snapshot changed; refusing to overwrite')
        return path

    def snapshot_acquisition(self, scan_folder):
        scan = _path_check(scan_folder)
        if not scan.is_dir():
            raise MemoryStoreError('Scan folder must be an existing directory')
        warnings, pointer, session = [], {}, {}
        for name, target in [('session.json', session), ('device_diagnostics.json', pointer)]:
            try:
                target.update(_small_json(scan / name))
            except (OSError, ValueError) as error:
                warnings.append({'kind': 'metadata_unavailable', 'source': str(scan / name), 'reason': str(error)})
        session_id = _identifier(session.get('session_id', str(scan)), 'unidentified-session')
        session_base = self.root / 'acquisition' / session_id
        structured_total, file_count = 0, 0

        def copy_file(source, group, name, *, tail=False):
            nonlocal structured_total, file_count
            record = {'source': str(source), 'role': 'sdk_raw_tail' if tail else 'structured_log'}
            if file_count >= MAX_FILES:
                record.update(status='unavailable', reason='file_limit_reached')
                return record
            file_count += 1
            try:
                limit = MAX_SDK_TAIL_BYTES if tail else min(MAX_STRUCTURED_FILE_BYTES,
                                                           MAX_STRUCTURED_TOTAL_BYTES - structured_total)
                data, details = _read_segment(source, max(0, limit), tail=tail)
                stored = self._store_blob(group / 'objects', name, data)
                record.update(details, **stored, status='included')
                if not tail:
                    structured_total += len(data)
                else:
                    record.update(copy_kind='bounded_byte_tail', first_record_may_be_partial=bool(details['start_offset_bytes']))
            except (OSError, ValueError) as error:
                record.update(status='unavailable', reason=str(error))
            return record

        files = []
        for name in SESSION_FILES:
            source = scan / name
            if source.exists() or source.is_symlink():
                files.append(copy_file(source, session_base, name))
            elif name in SESSION_FILES[:4]:
                warnings.append({'kind': 'session_log_unavailable', 'source': str(source),
                                 'reason': 'required_session_log_not_found'})
        for name in SNAPSHOT_FILES:
            source = scan / 'program_snapshot' / name
            if source.exists() or source.is_symlink():
                files.append(copy_file(source, session_base / 'program_snapshot', name))
        run_identity = (pointer.get('run_id') or pointer.get('diagnostics_directory')
                        or f'unknown-for-session:{session_id}:{scan}')
        run_id = _identifier(run_identity, 'unidentified-run')
        process_key = run_id + '-' + hashlib.sha256(str(pointer.get('diagnostics_directory', '')).encode()).hexdigest()[:12]
        process_base = self.root / 'device_runs' / process_key
        process_files = []
        diagnostics, reason = _pointer_path(pointer.get('diagnostics_directory'), base=scan)
        if reason:
            warnings.append({'kind': 'diagnostics_unavailable', 'source': pointer.get('diagnostics_directory'), 'reason': reason})
        elif not diagnostics.is_dir():
            warnings.append({'kind': 'diagnostics_unavailable', 'source': str(diagnostics), 'reason': 'not_a_directory'})
        else:
            try:
                candidates = []
                with os.scandir(diagnostics) as entries:
                    for count, entry in enumerate(entries):
                        if count >= MAX_DIRECTORY_ENTRIES:
                            warnings.append({'kind': 'diagnostics_directory_limit', 'reason': 'Only2000entries inspected'})
                            break
                        if DIAGNOSTIC_NAME.fullmatch(entry.name) and not SENSITIVE.search(entry.name):
                            candidates.append(entry.name)
                for name in sorted(candidates):
                    process_files.append(copy_file(diagnostics / name, process_base, name))
            except OSError as error:
                warnings.append({'kind': 'diagnostics_unavailable', 'source': str(diagnostics), 'reason': str(error)})
        sdk_base = None
        if pointer.get('sdk_raw_log_base_cwd'):
            sdk_base, base_reason = _pointer_path(pointer['sdk_raw_log_base_cwd'])
            if base_reason:
                warnings.append({'kind': 'sdk_base_unavailable', 'source': pointer['sdk_raw_log_base_cwd'], 'reason': base_reason})
        sdk, reason = _pointer_path(pointer.get('sdk_raw_log'), base=sdk_base)
        if reason:
            warnings.append({'kind': 'sdk_raw_log_unavailable', 'source': pointer.get('sdk_raw_log'), 'reason': reason})
        elif SENSITIVE.search(sdk.name) or sdk.suffix.lower() not in ('.log', '.txt'):
            warnings.append({'kind': 'sdk_raw_log_unavailable', 'source': str(sdk), 'reason': 'filename_not_allowed'})
        else:
            process_files.append(copy_file(sdk, process_base, 'ximc_sdk_raw.tail.log', tail=True))
        for item in files + process_files:
            if item['status'] != 'included':
                warnings.append({'kind': 'file_unavailable', 'source': item['source'],
                                 'reason': item['reason']})
        process_record = {'kind': 'shared_device_process_snapshot', 'run_id': run_id,
                          'identity_basis': ('recorded_run_id' if pointer.get('run_id') else
                                             'recorded_diagnostics_directory' if pointer.get('diagnostics_directory') else
                                             'session_only_process_identity_unavailable'),
                          'source': {'diagnostics_directory': pointer.get('diagnostics_directory'),
                                     'sdk_raw_log': pointer.get('sdk_raw_log')},
                          'files': process_files, 'warnings': [w for w in warnings if w['kind'].startswith(('diagnostics', 'sdk'))],
                          'snapshot_atomic': False, 'images_included': False, 'device_sdk_calls': 0}
        process_manifest = self._store_manifest(process_base, process_record)
        acquisition_record = {'kind': 'acquisition_log_snapshot', 'session_id': session_id,
                              'source': str(scan), 'files': files, 'warnings': warnings,
                              'diagnostics_pointer': _clean(pointer), 'process_run_id': run_id,
                              'process_manifest_path': str(process_manifest), 'process_manifest_sha256': _sha_file(process_manifest),
                              'snapshot_atomic': False, 'images_included': False, 'device_sdk_calls': 0,
                              'limits': {'structured_file_bytes': MAX_STRUCTURED_FILE_BYTES,
                                         'structured_total_bytes': MAX_STRUCTURED_TOTAL_BYTES,
                                         'sdk_tail_bytes': MAX_SDK_TAIL_BYTES, 'max_files': MAX_FILES}}
        manifest = self._store_manifest(session_base, acquisition_record)
        with self._locked():
            index = self._index()
            index['acquisitions'][session_id] = {'source': str(scan), 'latest_manifest_path': str(manifest),
                                                'sha256': _sha_file(manifest), 'updated_at_utc': _utc()}
            index['device_runs'][process_key] = {'run_id': run_id, 'latest_manifest_path': str(process_manifest),
                                                'sha256': _sha_file(process_manifest), 'updated_at_utc': _utc()}
            self._save_index(index)
        result = {'session_id': session_id, 'manifest_path': str(manifest), 'files': files,
                  'process_run_id': run_id, 'process_manifest_path': str(process_manifest),
                  'process_files': process_files, 'warnings': warnings,
                  'snapshot_atomic': False, 'images_included': False, 'structured_bytes': structured_total,
                  'status': 'partial' if warnings else 'recorded',
                  'included_file_count': sum(f['status'] == 'included' for f in files + process_files),
                  'unavailable_file_count': sum(f['status'] != 'included' for f in files + process_files)}
        self.record('acquisition_snapshot', result['status'], session_id=session_id,
                    manifest_path=str(manifest), process_manifest_path=str(process_manifest),
                    warnings=warnings, images_included=False)
        return result

    def legacy_reference(self, path, label):
        source = _path_check(path)
        if not source.exists():
            raise MemoryStoreError('Legacy history location does not exist')
        if not isinstance(label, str) or not label.strip() or len(label) > 240:
            raise MemoryStoreError('Legacy label must be a short description')
        key = hashlib.sha256(str(source).encode()).hexdigest()
        record = {'reference_id': key, 'path': str(source), 'label': label,
                  'read_only_reference': True, 'copied': False, 'moved': False,
                  'kind': 'directory' if source.is_dir() else 'file'}
        if source.is_file():
            record['bytes'] = source.stat().st_size
            record['sha256'] = _sha_file(source) if source.stat().st_size <= MAX_STRUCTURED_FILE_BYTES else None
        with self._locked():
            index = self._index()
            previous = index['legacy_references'].get(key, {})
            labels = previous.get('labels', [])
            if label not in labels:
                labels.append(label)
            index['legacy_references'][key] = {**record, 'labels': labels, 'updated_at_utc': _utc()}
            self._save_index(index)
        self.record('legacy_reference', 'registered', **record)
        return record

    def summary(self):
        with self._locked():
            index = self._index()
            path = self.root / 'events.jsonl'
            recent = []
            if path.exists():
                data, _ = _read_segment(path, 16 * MIB, tail=True)
                lines = data.splitlines()
                # A byte-tail may start inside the first record; recent complete
                # records remain available without pretending to parse that prefix.
                for line in reversed(lines):
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if isinstance(event, dict):
                        recent.append(event)
                    if len(recent) >= 80:
                        break
                recent.reverse()
            files, seen, warnings = [], set(), []
            snapshots = [*list(index['acquisitions'].values())[-80:],
                         *list(index['device_runs'].values())[-80:]]
            for item in snapshots:
                try:
                    manifest_path = _path_check(item['latest_manifest_path'])
                    try:
                        manifest_path.relative_to(self.root)
                    except ValueError as error:
                        raise MemoryStoreError('Snapshot manifest escaped Memory root') from error
                    if _sha_file(manifest_path) != item['sha256']:
                        raise MemoryStoreError('Retained snapshot manifest changed')
                    manifest = _small_json(manifest_path, MAX_INDEX_BYTES)
                    for record in manifest.get('files', []):
                        if record.get('status') == 'included' and record['path'] not in seen:
                            stored_path = _path_check(record['path'])
                            try:
                                stored_path.relative_to(self.root)
                            except ValueError as error:
                                raise MemoryStoreError('Stored file escaped Memory root') from error
                            seen.add(record['path'])
                            if len(files) < 2000:
                                files.append({k: record[k] for k in ('path', 'sha256', 'bytes', 'source', 'role')})
                except (OSError, ValueError, KeyError) as error:
                    warnings.append({'kind': 'snapshot_manifest_unavailable', 'reason': str(error)})
            return {'root': str(self.root), 'index_path': str(self.root / 'index.json'),
                    'events_path': str(path), 'event_count': index['event_count'],
                    'latest_events': recent, 'steps': index['steps'],
                    'acquisitions': list(index['acquisitions'].values()),
                    'device_runs': list(index['device_runs'].values()),
                    'legacy_references': list(index['legacy_references'].values()),
                    'files': files, 'files_list_limited': len(seen) > 2000 or len(index['acquisitions']) > 80 or len(index['device_runs']) > 80,
                    'warnings': warnings,
                    'history_policy': 'Append-only events; immutable content-addressed copies; originals never moved',
                    'device_sdk_calls': 0, 'raw_images_included': False}
