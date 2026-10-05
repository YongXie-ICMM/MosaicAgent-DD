"""Camera discovery and auditable scan records; no hardware access on import.

OpenCV indices are connection hints, not persistent camera identities. UVC
capture cards may not expose the microscope camera's model or serial number.
"""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import platform
import threading
import uuid
from importlib import metadata
from shared_history import _clean, _preserve_evidence, _restore_live_journal


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime_versions():
    versions = {}
    for name in ('opencv-python', 'opencv-contrib-python', 'opencv-python-headless', 'numpy'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w', encoding='utf-8') as f:
            json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(clean(value), ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _read(obj, name):
    try:
        value = getattr(obj, name)
        return clean(value() if callable(value) else value)
    except Exception:
        return None


def describe_camera(cam, frame=None):
    backend = getattr(cam, 'backend', None)
    sdk = backend in ('amcam', 'toupcam')
    dev = getattr(cam, '_dev_info', None)
    handle = getattr(cam, 'hcam', None)
    cap = getattr(cam, '_cv_cap', None)
    supported = getattr(cam, '_uvc_supported', {}) or {}
    dimensions = None
    if frame is not None and len(getattr(frame, 'shape', ())) >= 2:
        dimensions = [int(frame.shape[1]), int(frame.shape[0])]
    info = {
        'recorded_at_utc': now(), 'backend': backend,
        'camera_index': getattr(cam, 'camera_index', None),
        'requested_resolution': getattr(cam, '_requested_resolution', None),
        'actual_resolution': dimensions or _read(cam, 'get_resolution'),
        'resolution_source': 'received_frame' if dimensions else 'driver_readback',
        'display_name': _read(dev, 'displayname') if sdk else None,
        'model': _read(getattr(dev, 'model', None), 'name') if sdk else None,
        'device_id': _read(dev, 'id') if sdk else None,
        'serial_number': _read(handle, 'SerialNumber') if sdk else None,
        'firmware_version': _read(handle, 'FwVersion') if sdk else None,
        'capture_api': _read(cap, 'getBackendName') if not sdk else backend,
        'identity_status': 'sdk_reported' if sdk else 'not_exposed_by_opencv',
        'microscope_view_confirmed': False,
        'readback': {},
        'notes': [
            'Device index can change after reconnecting. Confirm the microscope view.',
            'Model and serial remain null if the capture interface does not expose them.',
            'Image dimensions do not establish sample scale in micrometres per pixel.',
        ],
    }
    for key, method, uvc_key, unit in [
        ('exposure', 'get_exposure_time', 'exposure', 'microseconds' if sdk else 'driver_native'),
        ('gain', 'get_gain', 'gain', 'SDK_native' if sdk else 'driver_native'),
        ('white_balance', 'get_temp_tint', 'white_balance', 'temperature_tint' if sdk else 'driver_native'),
    ]:
        can_read = sdk or uvc_key in supported
        value = _read(cam, method) if can_read else None
        info['readback'][key] = {'value': value, 'unit': unit,
                                'status': 'reported' if value is not None else 'unavailable'}
    return clean(info)


def open_detected_camera(CameraClass, history_dir, bits=24, resolution=None,
                         indices=range(4), exclude_indices=()):
    """Try the last successful connection then bounded automatic candidates.

    The caller must run this off the GUI thread. A received frame, rather than
    an opened handle alone, is required. No previous exposure setting is applied.
    """
    from acquisition_contract import (AcquisitionModeError, contract,
                                      require_frame_size, preserve_diagnostic_frame)
    history = Path(history_dir)
    last = {}
    try:
        last = json.loads((history / 'last_camera.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        pass
    excluded = set(exclude_indices)
    candidates = []
    previous_index = last.get('camera_index')
    previous_backend = last.get('backend')
    if isinstance(previous_index, int) and previous_index not in excluded:
        if previous_backend in ('auto', 'amcam', 'toupcam', 'opencv'):
            # Camera_v2 resolves its SDK globally; do not force a different SDK name.
            candidates.append(('opencv' if previous_backend == 'opencv' else 'auto', previous_index))
    for i in indices:
        if i not in excluded:
            # Explicit fallback also handles missing SDK DLLs raising OSError.
            candidates.extend([('auto', i), ('opencv', i)])
    attempts = []
    for backend, index in dict.fromkeys(candidates):
        cam = None
        try:
            cam = CameraClass(camera_index=index, bits=bits, backend=backend, resolution=resolution)
            cam.open()
            cam.start_live()
            frame = None
            for _ in range(3):
                frame = cam.grab_frame(timeout=1.0)
                if frame is not None and getattr(frame, 'size', 0) > 0:
                    break
            if frame is None or getattr(frame, 'size', 0) == 0:
                raise RuntimeError('Camera opened but did not deliver a valid frame')
            info = describe_camera(cam, frame)
            if resolution is not None:
                try:
                    actual = require_frame_size(frame, resolution, 'connection')
                except AcquisitionModeError as mismatch:
                    mismatch.camera_index = index
                    mismatch.backend_requested = backend
                    try:
                        diagnostic = preserve_diagnostic_frame(
                            frame, history / 'diagnostic_frames', CameraClass.save_image, 'connection')
                        mismatch.args = (str(mismatch) + ' Diagnostic: ' + str(diagnostic),)
                    except Exception as save_error:
                        mismatch.args = (str(mismatch) + ' Diagnostic save failed: ' + str(save_error),)
                    raise
                info['acquisition_contract'] = contract(
                    resolution, actual, phase='connection', verified=True)
            info['discovery_attempts'] = attempts + [{'backend_requested': backend, 'index': index, 'status': 'received_frame'}]
            info['previous_configuration_reused'] = False
            info['previous_connection_hint_tried'] = bool(last)
            # Persist the current readback, not stale model or settings from last time.
            append_jsonl(history / 'camera_connections.jsonl', info)
            atomic_json(history / 'last_camera.json', info)
            return cam, info
        except Exception as exc:
            attempts.append({'backend_requested': backend, 'index': index,
                             'status': 'failed', 'error': str(exc)})
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass
            if isinstance(exc, AcquisitionModeError):
                # Do not silently select a different (possibly non-microscope)
                # camera merely because the first received frame has the wrong mode.
                append_jsonl(history / 'camera_connections.jsonl',
                             {'recorded_at_utc': now(), 'status': 'resolution_mismatch',
                              'expected_image_size': list(resolution),
                              'actual_image_size': exc.actual, 'attempts': attempts})
                raise
    append_jsonl(history / 'camera_connections.jsonl',
                 {'recorded_at_utc': now(), 'status': 'all_candidates_failed', 'attempts': attempts})
    raise RuntimeError('No camera delivered a frame. Close other camera software, check connection, and retry. ' +
                       '; '.join(f"{a['backend_requested']}:{a['index']}: {a['error']}" for a in attempts))


class ScanSession:
    """Append-only event history plus atomic summary in a unique scan folder."""
    def __init__(self, folder, camera_info, scan_config, source_paths, simulated=False,
                 shared_history=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        if (self.folder / 'session.json').exists():
            raise FileExistsError('This folder already contains a session; use a new folder.')
        self._lock = threading.Lock()
        self._shared_history = shared_history
        self.history_error = None
        self._events = []
        self._journal_count = 0
        self._recovery_id = None
        self._unrecoverable_error = None
        sources = []
        snapshot = self.folder / 'program_snapshot'
        snapshot.mkdir(exist_ok=True)
        source_paths = list(source_paths)
        if source_paths:
            manifest = Path(source_paths[0]).resolve().parent / 'delivery_manifest.json'
            if manifest.exists() and manifest not in [Path(p) for p in source_paths]:
                source_paths.append(manifest)
        for source in source_paths:
            path = Path(source)
            if not path.is_file():
                sources.append({'name': path.name, 'status': 'missing'})
                continue
            data = path.read_bytes()
            target = snapshot / path.name
            if target.exists():
                raise FileExistsError(f'Duplicate source basename: {path.name}')
            target.write_bytes(data)
            sources.append({'name': path.name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
        self.summary = clean({
            'schema_version': 2, 'session_id': uuid.uuid4().hex,
            'started_at_utc': now(), 'status': 'running',
            'stage_simulated': bool(simulated),
            'camera_may_be_real_in_stage_simulation': bool(simulated),
            'camera': camera_info, 'scan_config': scan_config,
            'program_sources': sources,
            'environment': {'os': platform.platform(), 'python': platform.python_version(),
                            'packages': runtime_versions()},
            'evidence_scope': 'Acquisition log, not independent scientific accuracy or an LLM run.',
            'event_count': 0,
        })
        if shared_history is not None:
            self.summary['shared_history'] = {
                'history_id': shared_history.history_id,
                'path': Path(os.path.relpath(shared_history.path, self.folder)).as_posix(),
            }
        atomic_json(self.folder / 'session.json', self.summary)
        self.record('session_started', session=self.summary.copy())

    def record(self, event, **fields):
        with self._lock:
            return self._record_locked(event, fields)

    def _record_locked(self, event, fields):
        if self.history_error is not None:
            raise RuntimeError('Acquisition history is incomplete; use explicit recovery after checking storage: '
                               + self.history_error)
        try:
            return self._write_record_locked(event, fields)
        except Exception as exc:
            # Ordinary recording stays blocked; only recover_history may retry
            # the same retained payload with its original identifiers.
            self.history_error = str(exc)
            raise

    def _write_record_locked(self, event, fields):
        reserved = {'event', 'event_id', 'session_id', 'sequence', 'timestamp_utc'}
        if reserved.intersection(fields):
            raise ValueError('Event identity fields are assigned by the recorder.')
        if event == 'resume':
            self.summary['status'] = 'running'
            self.summary.pop('finished_at_utc', None)
        data = clean(dict(fields, event=str(event), timestamp_utc=now(),
                          session_id=self.summary['session_id'],
                          event_id=uuid.uuid4().hex, sequence=len(self._events) + 1))
        filepath = fields.get('filepath')
        try:
            if filepath and Path(filepath).is_file():
                path = Path(filepath)
                raw = path.read_bytes()
                data['image_sha256'] = hashlib.sha256(raw).hexdigest()
                data['image_bytes'] = len(raw)
                data['image_relative_path'] = Path(os.path.relpath(path, self.folder)).as_posix()
            elif event in ('candidate_saved', 'candidate_accepted', 'capture_success'):
                raise ValueError('A saved-image event requires the original image file.')
        except Exception as exc:
            self._unrecoverable_error = 'Original image identity was not recorded: ' + str(exc)
            raise
        self._events.append(data)
        append_jsonl(self.folder / 'events.jsonl', data)
        self._journal_count = len(self._events)
        self.summary['event_count'] = data['sequence']
        self._forward_record(data)
        atomic_json(self.folder / 'session.json', self.summary)
        return copy.deepcopy(data)

    def _forward_record(self, data):
        if self._shared_history is None:
            return
        reserved = {'event', 'event_id', 'session_id', 'sequence', 'timestamp_utc'}
        forwarded = {k: v for k, v in data.items() if k not in reserved}
        if 'image_relative_path' in forwarded:
            forwarded['scan_image_relative_path'] = forwarded.pop('image_relative_path')
        fields = dict(forwarded, scan_session_id=data['session_id'],
                      source_event_id=data['event_id'], source_sequence=data['sequence'],
                      source_timestamp_utc=data['timestamp_utc'], scan_folder=str(self.folder.resolve()))
        existing = self._shared_history.get_source_event(data['event_id'])
        if existing is not None:
            expected = _clean(fields, self._shared_history.folder)
            if existing['event'] != data['event'] or any(existing.get(key) != value
                                                       for key, value in expected.items()):
                raise ValueError('Shared mirror conflicts with its original scan event.')
            return
        self._shared_history.record(data['event'], **fields)

    def events(self):
        """Copy retained events; pending events become durable only after recovery."""
        with self._lock:
            return copy.deepcopy(self._events)

    def find_event(self, event, **fields):
        with self._lock:
            for record in reversed(self._events):
                if record['event'] == event and all(record.get(key) == value
                                                  for key, value in fields.items()):
                    return copy.deepcopy(record)
        return None

    def _validate_image_evidence(self):
        promotions = {}
        for event in self._events:
            if event['event'] in ('candidate_promotion_requested', 'candidate_accepted'):
                source = event.get('candidate_path') or event.get('candidate_original_path')
                target = event.get('filepath')
                if source and target:
                    promotions[str(Path(source).resolve())] = Path(target)
        for event in self._events:
            if 'image_sha256' not in event:
                continue
            path = self.folder / event['image_relative_path']
            if not path.is_file():
                path = promotions.get(str(path.resolve()), path)
            if not path.is_file():
                raise ValueError('Recorded image is missing; recovery refused: ' + str(path))
            raw = path.read_bytes()
            if len(raw) != event['image_bytes'] or hashlib.sha256(raw).hexdigest() != event['image_sha256']:
                raise ValueError('Recorded image changed; recovery refused: ' + str(path))

    def recover_history(self):
        """Recover writes from this live session without taking or moving a photo.

        File hashes, complete journal records and existing mirrors must agree.
        Neither this method nor its receipt claims that the stage stayed still;
        the caller must independently check connection and position continuity.
        """
        with self._lock:
            self._recovery_id = self._recovery_id or uuid.uuid4().hex
            recovery_id = self._recovery_id
            previous_error = self.history_error
            pending = copy.deepcopy(self._events[self._journal_count:])
            try:
                if self._unrecoverable_error:
                    raise ValueError(self._unrecoverable_error)
                self._validate_image_evidence()
                journal_backup = _restore_live_journal(self.folder / 'events.jsonl', self._events,
                                                       self._journal_count, recovery_id)
                if self._shared_history is not None:
                    self._shared_history.recover_history()
                    for event in self._events:
                        self._forward_record(event)
                self.summary['event_count'] = len(self._events)
                snapshot_backup = _preserve_evidence(self.folder / 'session.json', recovery_id)
                atomic_json(self.folder / 'session.json', self.summary)
                receipt = {'recovery_id': recovery_id, 'timestamp_utc': now(),
                           'session_id': self.summary['session_id'],
                           'method': 'same_process_retained_events', 'previous_error': previous_error,
                           'event_count': len(self._events), 'recovered_events': pending,
                           'journal_backup': journal_backup, 'snapshot_backup': snapshot_backup,
                           'stage_position_verified': False, 'image_hashes_verified': True}
                atomic_json(self.folder / 'recovery_receipts' / (recovery_id + '.json'), receipt)
            except Exception as exc:
                self.history_error = str(exc)
                raise
            self._journal_count = len(self._events)
            self.history_error = None
            self._recovery_id = None
            return receipt

    def finish(self, status, **fields):
        with self._lock:
            if self.history_error is not None:
                raise RuntimeError('Acquisition history is incomplete; recover it before finishing: '
                                   + self.history_error)
            if {'session_id', 'event_count', 'schema_version'}.intersection(fields):
                raise ValueError('Session identity cannot be changed by finish.')
            self.summary.update(clean(fields))
            self.summary.update({'status': str(status), 'finished_at_utc': now()})
            self._record_locked('session_finished', dict(fields, status=str(status)))
