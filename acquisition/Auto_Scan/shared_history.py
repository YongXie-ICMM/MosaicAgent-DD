"""One acquisition writer, many read-only agents, and separate agent notes.

``events.jsonl`` is the acquisition journal. Ordinary writes append only; explicit
same-process repair preserves its exact pre-repair bytes before replacement.
``history.json`` is an atomically replaced, human-readable snapshot. Agents must
not edit either file. This module
does not connect to an instrument or call a model, including from its CLI.

Recorded path references use '/' separators. Absolute source paths are made
relative to the history folder when they are on the same drive. References on
another Windows drive remain absolute and are not portable without that drive.
An explicitly named ``scan_image_relative_path`` is relative to its event's
``scan_folder``; other supplied relative paths retain the caller's stated base.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import os
import threading
import time
import uuid


SCHEMA_VERSION = 1
EVIDENCE_SCOPE = (
    'Recorded acquisition operations and supplied observations. A requested move '
    'does not establish controller completion or physical displacement. Controller '
    'readback does not independently establish image freshness or physical travel. '
    'Agent notes are separate interpretations, not acquisition evidence.'
)
_RESERVED = {'event_id', 'sequence', 'timestamp_utc', 'history_id', 'schema_version'}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _path_text(value):
    return str(value).replace(os.sep, '/')


def _clean(value, folder=None, key=''):
    if isinstance(value, Path):
        value = value.resolve()
        try:
            return _path_text(os.path.relpath(value, folder) if folder is not None else value)
        except ValueError:  # Different Windows drives cannot be relative.
            return _path_text(value)
    if isinstance(value, str) and folder is not None:
        if key == 'filepath' or key.endswith(('_path', '_folder')):
            if os.path.isabs(value):
                try:
                    return _path_text(os.path.relpath(Path(value).resolve(), folder))
                except ValueError:  # Different Windows drives cannot be relative.
                    return _path_text(value)
            return _path_text(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, dict):
        return {str(k): _clean(v, folder, str(k)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v, folder, key) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers/antivirus may briefly hold a handle without delete
        # sharing. Retry only this atomic promotion, never append an event twice.
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (2 ** attempt))
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_event(path, event):
    # One writer owns this journal. Consumers only read it.
    encoded = (json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    with Path(path).open('ab') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _preserve_evidence(path, recovery_id):
    """Keep exact pre-repair bytes before any recovery replacement."""
    path = Path(path)
    if not path.exists():
        return None
    raw = path.read_bytes()
    directory = path.parent / 'recovery_backups'
    directory.mkdir(exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()
    target = directory / (recovery_id + '_' + digest[:16] + '_' + path.name)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError('Recovery evidence backup does not match original bytes.')
    else:
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.read_bytes() != raw:
                raise OSError('Recovery evidence backup was not written completely.')
            # A partial backup must never occupy its final, content-addressed
            # name: once storage is repaired, the same recovery can try again.
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {'path': target.relative_to(path.parent).as_posix(),
            'sha256': digest, 'bytes': len(raw)}


def _restore_live_journal(path, events, committed_count, recovery_id):
    """Repair only a live writer's exact prefix, never guess missing events.

    Complete on-disk records must equal retained records. The only tolerated
    incomplete tail is a byte prefix of the next retained event. Original bytes
    are retained in a durable backup before an atomic journal replacement.
    """
    path = Path(path)
    raw = path.read_bytes() if path.exists() else b''
    lines = raw.splitlines(keepends=True)
    count = 0
    tail = b''
    for line in lines:
        if not line.endswith(b'\n'):
            tail = line
            break
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError('Committed journal data changed or is malformed; recovery refused.') from exc
        if count >= len(events) or value != events[count]:
            raise ValueError('Committed journal data differs from this live session; recovery refused.')
        count += 1
    if count < committed_count:
        raise ValueError('Previously committed journal records are missing; recovery refused.')
    if tail:
        if count >= len(events):
            raise ValueError('Journal has an unknown trailing record; recovery refused.')
        expected = (json.dumps(events[count], ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        if not expected.startswith(tail) and not (tail.endswith(b'\r') and expected[:-1] == tail[:-1]):
            raise ValueError('Interrupted journal tail differs from retained event; recovery refused.')
    backup = None
    if count != len(events) or tail:
        backup = _preserve_evidence(path, recovery_id)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('xb') as stream:
                for event in events:
                    stream.write((json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    # A failed fsync may have left a complete line. Re-sync it even if no rewrite
    # was needed; its original identifier must not be appended a second time.
    with path.open('ab') as stream:
        stream.flush()
        os.fsync(stream.fileno())
    return backup


def _snapshot(events):
    if not events or events[0]['event'] != 'history_started':
        raise ValueError('The journal must begin with history_started.')
    first = events[0]
    final = events[-1]
    closed = final['event'] == 'history_closed'
    return {
        'schema_version': SCHEMA_VERSION,
        'history_id': first['history_id'],
        'created_at_utc': first['timestamp_utc'],
        'updated_at_utc': final['timestamp_utc'],
        'status': final.get('status', 'closed') if closed else 'open',
        'is_closed': closed,
        'event_count': len(events),
        'snapshot_sequence': final['sequence'],
        'metadata': copy.deepcopy(first.get('metadata', {})),
        'evidence_scope': EVIDENCE_SCOPE,
        'path_base': 'directory_containing_history_json',
        'events': copy.deepcopy(events),
    }


class SharedHistory:
    """Application-session history; instantiate lazily before the first action.

    Only this object writes acquisition records. Multiple threads may call
    ``record``. Different applications get different directories, so they never
    compete over a shared index or journal. Reading and agent notes need no lock.
    An append failure blocks ordinary writes until explicit same-process recovery.
    A snapshot failure leaves the durable journal recoverable and can be retried
    by a subsequent record without reusing the committed event's sequence.
    """
    def __init__(self, root_dir, metadata=None):
        self.history_id = uuid.uuid4().hex
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.folder = Path(root_dir).resolve() / (stamp + '_' + self.history_id[:10])
        self.folder.mkdir(parents=True, exist_ok=False)
        self.path = self.folder / 'history.json'
        self.journal_path = self.folder / 'events.jsonl'
        self._lock = threading.Lock()
        self._events = []
        self._closed = False
        self._append_failed = False
        self._journal_count = 0
        self._last_error = None
        self._recovery_id = None
        self.record('history_started', metadata=_clean(metadata or {}, self.folder))

    def _record_locked(self, event, fields):
        if self._closed:
            raise RuntimeError('History is closed; start a new application history.')
        if self._append_failed:
            raise RuntimeError('Journal append previously failed; stop logging to this journal.')
        if not isinstance(event, str) or not event.strip():
            raise ValueError('event must be a nonempty string.')
        conflicts = _RESERVED.intersection(fields)
        if conflicts:
            raise ValueError('Reserved history fields: ' + ', '.join(sorted(conflicts)))
        record = _clean(fields, self.folder)
        record.update({
            'event': event, 'schema_version': SCHEMA_VERSION,
            'history_id': self.history_id, 'event_id': uuid.uuid4().hex,
            'sequence': len(self._events) + 1, 'timestamp_utc': _now(),
        })
        record.setdefault('actor', 'acquisition_program')
        # Retain the original identifier, timestamp and complete payload before
        # I/O: an interrupted write is retried as the same observed event.
        self._events.append(record)
        try:
            _append_event(self.journal_path, record)
        except Exception as exc:
            self._append_failed = True
            self._last_error = str(exc)
            raise
        self._journal_count = len(self._events)
        if event == 'history_closed':
            self._closed = True
        try:
            _atomic_json(self.path, _snapshot(self._events))
        except Exception as exc:
            self._last_error = str(exc)
            raise
        self._last_error = None
        return copy.deepcopy(record)

    def get_source_event(self, source_event_id):
        """Read a retained scan mirror, including an unflushed pending record."""
        with self._lock:
            matches = [event for event in self._events
                       if event.get('source_event_id') == source_event_id]
            if len(matches) > 1:
                raise ValueError('Duplicate source event in shared history.')
            return copy.deepcopy(matches[0]) if matches else None

    def recover_history(self):
        """Flush this live object's original events; never reopen a prior run."""
        with self._lock:
            self._recovery_id = self._recovery_id or uuid.uuid4().hex
            recovery_id = self._recovery_id
            previous_error = self._last_error
            try:
                backup = _restore_live_journal(self.journal_path, self._events,
                                               self._journal_count, recovery_id)
                snapshot_backup = _preserve_evidence(self.path, recovery_id)
                _atomic_json(self.path, _snapshot(self._events))
                receipt = {'recovery_id': recovery_id, 'timestamp_utc': _now(),
                           'history_id': self.history_id, 'method': 'same_process_retained_events',
                           'previous_error': previous_error, 'event_count': len(self._events),
                           'journal_backup': backup, 'snapshot_backup': snapshot_backup}
                receipts = self.folder / 'recovery_receipts'
                receipts.mkdir(exist_ok=True)
                _atomic_json(receipts / (recovery_id + '.json'), receipt)
            except Exception as exc:
                self._last_error = str(exc)
                self._append_failed = True
                raise
            self._journal_count = len(self._events)
            self._append_failed = False
            self._closed = bool(self._events and self._events[-1]['event'] == 'history_closed')
            self._last_error = None
            self._recovery_id = None
            return receipt

    def record(self, event, **fields):
        """Append one event; reserved identifiers cannot be supplied by callers."""
        if event in ('history_closed',):
            raise ValueError('Use close() to finish a history.')
        if event == 'history_started' and self._events:
            raise ValueError('history_started is reserved for initialization.')
        with self._lock:
            return self._record_locked(event, fields)

    def close(self, status='closed', **fields):
        """Mark only this application history closed, not a scan successful."""
        if not isinstance(status, str) or not status.strip() or status == 'open':
            raise ValueError('A closed history needs a non-open status.')
        with self._lock:
            if self._closed:
                return copy.deepcopy(self._events[-1])
            return self._record_locked('history_closed', dict(fields, status=status))


def _history_path(path):
    path = Path(path).resolve()
    return path / 'history.json' if path.is_dir() else path


def _validate_events(events, history_id=None):
    if not isinstance(events, list) or not events:
        raise ValueError('History must contain a nonempty events list.')
    seen = set()
    for sequence, event in enumerate(events, 1):
        if not isinstance(event, dict):
            raise ValueError('Each history event must be an object.')
        if event.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported event schema version.')
        if event.get('sequence') != sequence or isinstance(event.get('sequence'), bool):
            raise ValueError('History sequence is not contiguous.')
        if not isinstance(event.get('event_id'), str) or not event['event_id']:
            raise ValueError('Missing event_id.')
        if event['event_id'] in seen:
            raise ValueError('Duplicate event_id.')
        seen.add(event['event_id'])
        history_id = history_id or event.get('history_id')
        if not isinstance(history_id, str) or not history_id or event.get('history_id') != history_id:
            raise ValueError('History identity mismatch.')
        if not isinstance(event.get('timestamp_utc'), str) or not isinstance(event.get('event'), str):
            raise ValueError('Event name and timestamp are required.')
        if sequence > 1 and event['event'] == 'history_started':
            raise ValueError('Duplicate history_started event.')
        if sequence < len(events) and event['event'] == 'history_closed':
            raise ValueError('Events follow history_closed.')
    _snapshot(events)


def _read_journal(path):
    raw = Path(path).read_bytes()
    lines = raw.splitlines(keepends=True)
    events = []
    warning = None
    for index, line in enumerate(lines):
        # A last line without newline is not committed, even if valid JSON.
        if not line.endswith(b'\n'):
            if index != len(lines) - 1:
                raise ValueError('Incomplete line inside acquisition journal.')
            warning = 'uncommitted_trailing_line'
            break
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError('Malformed committed journal line %d.' % (index + 1)) from exc
        events.append(event)
    _validate_events(events)
    return events, warning


def read_history(path, recover=False):
    """Read and validate a snapshot without changing acquisition files.

    ``read_diagnostics`` reports snapshot lag and any incomplete journal tail.
    With ``recover=True`` return a journal-derived view, still without writing.
    A missing journal is reported explicitly (for standalone snapshot exports).
    If the snapshot is missing or invalid, recovery requires ``recover=True``.
    """
    path = _history_path(path)
    snapshot = None
    snapshot_error = None
    try:
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(snapshot, dict) or snapshot.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported history snapshot schema.')
        _validate_events(snapshot.get('events'), snapshot.get('history_id'))
        expected = _snapshot(snapshot['events'])
        if snapshot != expected:
            raise ValueError('Snapshot summary is inconsistent with its events.')
    except (OSError, ValueError) as exc:
        snapshot_error = str(exc)
        if not recover:
            raise
    journal = path.with_name('events.jsonl')
    if not journal.exists():
        if snapshot is None or snapshot_error:
            raise ValueError('Cannot recover without the acquisition journal.')
        result = copy.deepcopy(snapshot)
        result['read_diagnostics'] = {
            'journal_available': False, 'snapshot_lag': None,
            'warning': 'journal_unavailable_snapshot_only', 'recovered_in_memory': False,
        }
        return result
    events, warning = _read_journal(journal)
    if snapshot is not None and not snapshot_error:
        n = snapshot['event_count']
        if len(events) < n or events[:n] != snapshot['events']:
            raise ValueError('Snapshot does not match the append-only journal prefix.')
        lag = len(events) - n
    else:
        lag = None
    result = _snapshot(events) if recover else copy.deepcopy(snapshot)
    result['read_diagnostics'] = {
        'journal_available': True, 'journal_sequence': len(events),
        'snapshot_lag': lag, 'warning': warning,
        'snapshot_error': snapshot_error, 'recovered_in_memory': bool(recover),
    }
    return result


def rebuild_history(path, output=None):
    """Recover in memory, optionally save a NEW file outside the source folder.

    This never overwrites the active history or its journal. A crash leaves an
    open history open; recovery does not invent a successful completion event.
    """
    source = _history_path(path)
    result = read_history(source, recover=True)
    if output is not None:
        output = Path(output).resolve()
        if output.parent == source.parent or output.exists():
            raise ValueError('Recovery output must be new and outside the source history folder.')
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids overwriting a file created after the check.
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    return result


def list_histories(root_dir):
    """Discover sessions from their own files, without a mutable shared index."""
    result = []
    root = Path(root_dir).resolve()
    if not root.exists():
        return result
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not ((folder / 'history.json').exists() or (folder / 'events.jsonl').exists()):
            continue
        path = folder / 'history.json'
        try:
            history = read_history(path, recover=True)
            result.append({
                'history_id': history['history_id'], 'history_path': str(path),
                'created_at_utc': history['created_at_utc'], 'status': history['status'],
                'is_closed': history['is_closed'], 'event_count': history['event_count'],
                'metadata': history['metadata'], 'read_diagnostics': history['read_diagnostics'],
            })
        except (OSError, ValueError) as exc:
            result.append({'history_path': str(path), 'status': 'unreadable', 'error': str(exc)})
    return result


def write_agent_note(path, agent_id, note, related_event_ids=(), kind='observation'):
    """Append a separate immutable-by-API note; never amend source history.

    These are ordinary local files, not access-control or cryptographic seals.
    An agent's observation or proposal does not execute an instrument action.
    """
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError('agent_id must be a nonempty string.')
    if kind not in ('observation', 'proposal', 'analysis'):
        raise ValueError('kind must be observation, proposal, or analysis.')
    source = _history_path(path)
    history = read_history(source, recover=True)
    ids = list(related_event_ids)
    valid = {event['event_id'] for event in history['events']}
    if any(event_id not in valid for event_id in ids):
        raise ValueError('related_event_ids must refer to this acquisition history.')
    value = {
        'schema_version': SCHEMA_VERSION, 'note_id': uuid.uuid4().hex,
        'agent_id': agent_id, 'kind': kind, 'timestamp_utc': _now(),
        'history_id': history['history_id'],
        'source_snapshot_sequence': history['snapshot_sequence'],
        'source_history_path': '../history.json',
        'source_read_diagnostics': history['read_diagnostics'],
        'related_event_ids': ids, 'note': _clean(note),
        'evidence_scope': 'Agent interpretation only; does not alter acquisition records or execute actions.',
    }
    notes = source.parent / 'agent_notes'
    notes.mkdir(exist_ok=True)
    target = notes / (value['note_id'] + '.json')
    # New unique name plus atomic rename means readers never see a partial note.
    _atomic_json(target, value)
    return target


def read_agent_notes(path):
    """Read peer-agent notes and report every invalid note instead of hiding it.

    Note filenames are captured before reading the journal. A concurrently
    published note will be visible on the next call, and cannot be falsely
    rejected because it references an event newer than our history read.
    """
    source = _history_path(path)
    paths = sorted((source.parent / 'agent_notes').glob('*.json'))
    history = read_history(source, recover=True)
    event_sequences = {event['event_id']: event['sequence'] for event in history['events']}
    result = {
        'history_id': history['history_id'],
        'source_snapshot_sequence': history['snapshot_sequence'],
        'source_read_diagnostics': history['read_diagnostics'],
        'notes': [], 'invalid_notes': [],
    }
    for note_path in paths:
        relative_path = note_path.relative_to(source.parent).as_posix()
        try:
            value = json.loads(note_path.read_text(encoding='utf-8'))
            if not isinstance(value, dict) or value.get('schema_version') != SCHEMA_VERSION:
                raise ValueError('Unsupported agent note schema.')
            if value.get('history_id') != history['history_id']:
                raise ValueError('Agent note history_id does not match the source history.')
            if value.get('note_id') != note_path.stem:
                raise ValueError('Agent note_id does not match its filename.')
            for key in ('agent_id', 'timestamp_utc'):
                if not isinstance(value.get(key), str) or not value[key].strip():
                    raise ValueError('Agent note requires a nonempty ' + key + '.')
            if value.get('kind') not in ('observation', 'proposal', 'analysis'):
                raise ValueError('Unsupported agent note kind.')
            sequence = value.get('source_snapshot_sequence')
            if type(sequence) is not int or not 1 <= sequence <= history['snapshot_sequence']:
                raise ValueError('Agent note source sequence is outside this history.')
            refs = value.get('related_event_ids')
            if not isinstance(refs, list) or any(
                    not isinstance(ref, str) or ref not in event_sequences
                    or event_sequences[ref] > sequence for ref in refs):
                raise ValueError('Agent note references an unknown or newer acquisition event.')
            if value.get('source_history_path') != '../history.json' or 'note' not in value:
                raise ValueError('Agent note is missing its source reference or note content.')
            result['notes'].append(dict(value, note_path=relative_path))
        except (OSError, ValueError) as exc:
            result['invalid_notes'].append({'note_path': relative_path, 'error': str(exc)})
    result['notes'].sort(key=lambda value: (value['timestamp_utc'], value['note_id']))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list', help='List shared acquisition histories.')
    listing.add_argument('--root', default='shared_history')
    show = commands.add_parser('show', help='Read a validated JSON history.')
    show.add_argument('path')
    show.add_argument('--recover', action='store_true', help='Use complete journal entries, without modifying files.')
    notes = commands.add_parser('note', help='Save a separate agent observation/proposal.')
    notes.add_argument('path')
    notes.add_argument('--agent', required=True)
    notes.add_argument('--text', required=True)
    notes.add_argument('--kind', choices=('observation', 'proposal', 'analysis'), default='observation')
    notes.add_argument('--event-id', action='append', default=[])
    peer_notes = commands.add_parser('notes', help='Read peer-agent notes and report invalid references.')
    peer_notes.add_argument('path')
    args = parser.parse_args(argv)
    if args.command == 'list':
        result = list_histories(args.root)
    elif args.command == 'show':
        result = read_history(args.path, recover=args.recover)
    elif args.command == 'notes':
        result = read_agent_notes(args.path)
    else:
        result = {'note_path': str(write_agent_note(args.path, args.agent, args.text, args.event_id, args.kind))}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
