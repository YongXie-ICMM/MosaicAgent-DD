"""Fault-injection checks for live-process recovery; no device imports/calls."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest import mock

import camera_session_history as scan_module
import shared_history as shared_module
from camera_session_history import ScanSession
from shared_history import SharedHistory, read_history


class LiveHistoryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shared = SharedHistory(self.root / 'shared')
        self.scan = ScanSession(self.root / 'scan', {}, {}, [], shared_history=self.shared)

    @staticmethod
    def partial_write(path, event):
        encoded = json.dumps(event, ensure_ascii=False, allow_nan=False).encode('utf-8')
        with Path(path).open('ab') as stream:
            stream.write(encoded[:max(1, len(encoded) // 2)])
        raise OSError('disk full midway through a record')

    def scan_events(self):
        return [json.loads(line) for line in (self.scan.folder / 'events.jsonl').read_bytes().splitlines()]

    def mirrors(self):
        return [event for event in read_history(self.shared.path)['events']
                if event.get('scan_session_id') == self.scan.summary['session_id']]

    def test_partial_shared_append_retries_original_id_and_archives_exact_bytes(self):
        with mock.patch.object(shared_module, '_append_event', side_effect=self.partial_write):
            with self.assertRaises(OSError):
                self.shared.record('manual_move_attempt', dx_steps=40)
        pending = copy.deepcopy(self.shared._events[-1])
        before = self.shared.journal_path.read_bytes()
        result = self.shared.recover_history()
        self.assertEqual(read_history(self.shared.path)['events'][-1], pending)
        self.assertEqual((self.shared.folder / result['journal_backup']['path']).read_bytes(), before)
        self.shared.record('recovery_operator_confirmed', no_motion_command_sent=True)
        self.assertEqual(len({e['event_id'] for e in read_history(self.shared.path)['events']}), 4)

    def test_partial_scan_append_retries_original_id_time_and_sequence(self):
        with mock.patch.object(scan_module, 'append_jsonl', side_effect=self.partial_write):
            with self.assertRaises(OSError):
                self.scan.record('move_attempt', row=0, col=1)
        pending = self.scan.events()[-1]
        before = (self.scan.folder / 'events.jsonl').read_bytes()
        receipt = self.scan.recover_history()
        self.assertIsNone(self.scan.history_error)
        self.assertEqual(self.scan_events()[-1], pending)
        self.assertEqual((self.scan.folder / receipt['journal_backup']['path']).read_bytes(), before)
        self.assertEqual(self.mirrors()[-1]['source_event_id'], pending['event_id'])
        self.assertEqual(self.mirrors()[-1]['source_timestamp_utc'], pending['timestamp_utc'])
        self.assertFalse(receipt['stage_position_verified'])

    def test_complete_append_followed_by_fsync_failure_is_not_duplicated(self):
        original = scan_module.append_jsonl
        def write_then_fail(path, event):
            original(path, event)
            raise OSError('fsync failed after complete bytes')
        with mock.patch.object(scan_module, 'append_jsonl', side_effect=write_then_fail):
            with self.assertRaises(OSError):
                self.scan.record('position_checked', x_steps=40)
        pending = self.scan.events()[-1]
        self.scan.recover_history()
        self.scan.recover_history()
        self.assertEqual(self.scan_events(), self.scan.events())
        self.assertEqual(sum(e['event_id'] == pending['event_id'] for e in self.scan_events()), 1)
        self.assertEqual(sum(e['source_event_id'] == pending['event_id'] for e in self.mirrors()), 1)

    def test_shared_mirror_partial_failure_does_not_duplicate_scan_or_mirror(self):
        with mock.patch.object(shared_module, '_append_event', side_effect=self.partial_write):
            with self.assertRaises(OSError):
                self.scan.record('move_completed', x_steps=40)
        scan_event = self.scan.events()[-1]
        shared_event = self.shared.get_source_event(scan_event['event_id'])
        self.scan.recover_history()
        self.assertEqual(self.shared.get_source_event(scan_event['event_id']), shared_event)
        self.assertEqual(len(self.scan_events()), 2)
        self.assertEqual(len(self.mirrors()), 2)

    def test_shared_snapshot_failure_reuses_already_forwarded_event(self):
        with mock.patch.object(shared_module, '_atomic_json', side_effect=OSError('snapshot locked')):
            with self.assertRaises(OSError):
                self.scan.record('move_completed', x_steps=40)
        original = self.shared.get_source_event(self.scan.events()[-1]['event_id'])
        self.scan.recover_history()
        self.assertEqual(self.mirrors()[-1], original)
        self.assertEqual(len(self.mirrors()), 2)

    def test_scan_snapshot_failure_recovers_event_count_without_duplicate(self):
        with mock.patch.object(scan_module, 'atomic_json', side_effect=OSError('summary locked')):
            with self.assertRaises(OSError):
                self.scan.record('position_checked', x_steps=40)
        self.assertEqual(json.loads((self.scan.folder / 'session.json').read_text())['event_count'], 1)
        self.scan.recover_history()
        self.assertEqual(json.loads((self.scan.folder / 'session.json').read_text())['event_count'], 2)
        self.assertEqual(len(self.scan_events()), 2)

    def test_promoted_candidate_hash_resolves_to_final_file(self):
        candidate = self.scan.folder / 'candidate.png'
        final = self.scan.folder / 'mosaic_r0_c0.png'
        candidate.write_bytes(b'original candidate bytes')
        saved = self.scan.record('candidate_saved', filepath=str(candidate))
        self.scan.record('human_review', decision='accept', filepath=str(candidate))
        self.scan.record('candidate_promotion_requested', candidate_path=str(candidate), filepath=str(final))
        candidate.replace(final)
        with mock.patch.object(scan_module, 'append_jsonl', side_effect=self.partial_write):
            with self.assertRaises(OSError):
                self.scan.record('candidate_accepted', candidate_original_path=str(candidate), filepath=str(final))
        self.scan.recover_history()
        self.assertEqual(self.scan_events()[-1]['image_sha256'], saved['image_sha256'])
        self.assertEqual(final.read_bytes(), b'original candidate bytes')
        self.assertIsNone(self.scan.find_event('capture_success'))

    def test_missing_or_changed_photo_refuses_recovery_and_does_not_change_journal(self):
        photo = self.scan.folder / 'candidate.png'
        photo.write_bytes(b'original candidate bytes')
        self.scan.record('candidate_saved', filepath=str(photo))
        before = (self.scan.folder / 'events.jsonl').read_bytes()
        photo.write_bytes(b'altered image bytes')
        with self.assertRaisesRegex(ValueError, 'image changed'):
            self.scan.recover_history()
        photo.unlink()
        with self.assertRaisesRegex(ValueError, 'image is missing'):
            self.scan.recover_history()
        self.assertEqual((self.scan.folder / 'events.jsonl').read_bytes(), before)
        self.assertTrue(self.scan.history_error)

    def test_changed_committed_event_is_not_overwritten_from_memory(self):
        journal = self.scan.folder / 'events.jsonl'
        data = json.loads(journal.read_text())
        data['session']['scan_config'] = {'tampered': True}
        journal.write_text(json.dumps(data) + '\n')
        before = journal.read_bytes()
        with self.assertRaisesRegex(ValueError, 'differs from this live session'):
            self.scan.recover_history()
        self.assertEqual(journal.read_bytes(), before)

    def test_missing_committed_journal_refuses_recovery(self):
        (self.scan.folder / 'events.jsonl').unlink()
        with self.assertRaisesRegex(ValueError, 'committed journal records are missing'):
            self.scan.recover_history()
        self.assertFalse((self.scan.folder / 'events.jsonl').exists())

    def test_unknown_partial_tail_is_not_silently_replaced(self):
        journal = self.scan.folder / 'events.jsonl'
        with journal.open('ab') as stream:
            stream.write(b'{"unknown":')
        before = journal.read_bytes()
        with self.assertRaisesRegex(ValueError, 'unknown trailing record'):
            self.scan.recover_history()
        self.assertEqual(journal.read_bytes(), before)

    def test_repeated_repair_failure_keeps_same_pending_event_and_blocks_normal_writes(self):
        with mock.patch.object(scan_module, 'append_jsonl', side_effect=self.partial_write):
            with self.assertRaises(OSError):
                self.scan.record('move_attempt', x_steps=40)
        pending = self.scan.events()[-1]
        with mock.patch.object(shared_module.os, 'replace', side_effect=PermissionError('still locked')):
            with self.assertRaises(PermissionError):
                self.scan.recover_history()
        with self.assertRaisesRegex(RuntimeError, 'history is incomplete'):
            self.scan.record('move_completed')
        self.assertEqual(self.scan.events()[-1], pending)
        self.scan.recover_history()
        self.assertEqual(self.scan_events()[-1], pending)
        self.assertIsNone(self.scan.history_error)

    def test_partial_backup_failure_preserves_source_and_can_retry_same_recovery(self):
        with mock.patch.object(scan_module, 'append_jsonl', side_effect=self.partial_write):
            with self.assertRaises(OSError):
                self.scan.record('move_attempt', x_steps=40)
        journal = self.scan.folder / 'events.jsonl'
        before = journal.read_bytes()
        original_open = Path.open

        class PartialBackup:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def write(self, raw):
                self.stream.write(raw[:max(1, len(raw) // 2)])
                self.stream.flush()
                raise OSError('disk full midway through evidence backup')

        def open_with_partial_backup(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            if path.parent.name == 'recovery_backups' and args and args[0] == 'xb':
                return PartialBackup(stream)
            return stream

        with mock.patch.object(Path, 'open', open_with_partial_backup):
            with self.assertRaisesRegex(OSError, 'midway through evidence backup'):
                self.scan.recover_history()
        recovery_id = self.scan._recovery_id
        self.assertEqual(journal.read_bytes(), before)
        self.assertTrue(self.scan.history_error)
        self.assertEqual(list((self.scan.folder / 'recovery_backups').iterdir()), [])
        receipt = self.scan.recover_history()
        self.assertEqual(receipt['recovery_id'], recovery_id)
        backup = self.scan.folder / receipt['journal_backup']['path']
        self.assertEqual(backup.read_bytes(), before)
        self.assertEqual(len(self.scan_events()), 2)
        self.assertIsNone(self.scan.history_error)

    def test_failed_finish_does_not_modify_summary_when_an_error_is_already_latched(self):
        with mock.patch.object(scan_module, 'append_jsonl', side_effect=OSError('full')):
            with self.assertRaises(OSError):
                self.scan.record('move_attempt')
        before = copy.deepcopy(self.scan.summary)
        with self.assertRaisesRegex(RuntimeError, 'history is incomplete'):
            self.scan.finish('completed', photos_saved=920)
        self.assertEqual(self.scan.summary, before)

    def test_recovery_receipt_failure_stays_latched_then_retry_is_idempotent(self):
        with mock.patch.object(scan_module, 'append_jsonl', side_effect=OSError('full')):
            with self.assertRaises(OSError):
                self.scan.record('move_attempt')
        original = scan_module.atomic_json
        def fail_receipt(path, value):
            if Path(path).parent.name == 'recovery_receipts':
                raise OSError('receipt write failed')
            return original(path, value)
        with mock.patch.object(scan_module, 'atomic_json', side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError, 'receipt write failed'):
                self.scan.recover_history()
        recovery_id = self.scan._recovery_id
        self.assertTrue(self.scan.history_error)
        result = self.scan.recover_history()
        self.assertEqual(result['recovery_id'], recovery_id)
        self.assertEqual(len(self.scan_events()), 2)
        self.assertEqual(len(self.mirrors()), 2)

    def test_saved_event_without_original_hash_is_not_fabricated_on_recovery(self):
        with self.assertRaisesRegex(ValueError, 'original image file'):
            self.scan.record('candidate_saved', filepath=str(self.scan.folder / 'missing.png'))
        (self.scan.folder / 'missing.png').write_bytes(b'new unrelated bytes')
        with self.assertRaisesRegex(ValueError, 'identity was not recorded'):
            self.scan.recover_history()
        self.assertEqual(len(self.scan_events()), 1)

    def test_copy_accessors_cannot_mutate_live_evidence(self):
        event = self.scan.record('test_event', nested={'value': 1})
        event['nested']['value'] = 2
        found = self.scan.find_event('test_event')
        found['nested']['value'] = 3
        events = self.scan.events()
        events[-1]['nested']['value'] = 4
        self.assertEqual(self.scan.find_event('test_event')['nested']['value'], 1)


if __name__ == '__main__':
    unittest.main()
