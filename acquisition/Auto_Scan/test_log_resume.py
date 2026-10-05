"""Offline regressions for recovering a saved image without a second capture."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib

import camera_session_history as scan_history
from camera_session_history import ScanSession
from scan_log_resume import recover_scan_checkpoint
from shared_history import SharedHistory


def png_bytes():
    """Produce a small valid PNG without requiring a camera or image library."""
    def chunk(kind, data):
        return (struct.pack('!I', len(data)) + kind + data
                + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('!2I5B', 2, 2, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\0\xff\0\0\0\xff\0\0\0\0\xff\xff\xff\xff'))
            + chunk(b'IEND', b''))


class LogResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shared = SharedHistory(self.root / 'shared')
        self.session = ScanSession(self.root / 'scan', {'backend': 'fake'},
                                   {'nx': 2, 'ny': 1}, [], simulated=True,
                                   shared_history=self.shared)
        self.image = self.root / 'scan' / 'mosaic_r0_c0.png'
        self.raw = png_bytes()
        self.image.write_bytes(self.raw)
        self.checkpoint = {
            'phase': 'promoted', 'filepath': str(self.image),
            'image_sha256': hashlib.sha256(self.raw).hexdigest(),
            'image_bytes': len(self.raw), 'point': 1, 'row': 0, 'col': 0,
            'sharpness': 45.0, 'retries': 0, 'human_confirmed': True,
            'photo_count_before': 0, 'human_count_before': 0,
            'sharpness_len_before': 0, 'reconciled': False,
        }

    def reconciliation_events(self):
        return [event for event in self.session.events()
                if event['event'] == 'recovery_point_reconciled']

    def test_promoted_image_reconciles_once_and_never_fabricates_capture_success(self):
        with mock.patch.object(self.session, 'recover_history',
                               wraps=self.session.recover_history) as recover:
            first = recover_scan_checkpoint(self.session, self.checkpoint)
            second = recover_scan_checkpoint(self.session, self.checkpoint)
        self.assertEqual(recover.call_count, 2)
        self.assertEqual(first['action'], 'reuse_saved_image')
        self.assertEqual(first['reconciliation_event_id'], second['reconciliation_event_id'])
        self.assertEqual(len(self.reconciliation_events()), 1)
        self.assertEqual(self.image.read_bytes(), self.raw)
        self.assertTrue(self.checkpoint['reconciled'])
        self.assertFalse(any(event['event'] == 'capture_success' for event in self.session.events()))
        event = self.reconciliation_events()[0]
        self.assertEqual(event['original_acceptance_basis'], 'live_checkpoint_after_os_replace')
        self.assertFalse(event['physical_position_verified'])

    def test_candidate_checkpoint_does_not_reuse_even_if_final_image_exists(self):
        self.checkpoint['phase'] = 'candidate_saved'
        self.checkpoint['image_sha256'] = 'intentionally not trusted'
        result = recover_scan_checkpoint(self.session, self.checkpoint)
        self.assertEqual(result['action'], 'retry_current')
        self.assertEqual(self.reconciliation_events(), [])
        self.assertEqual(self.image.read_bytes(), self.raw)

    def test_no_checkpoint_repairs_pending_log_without_claiming_a_photo(self):
        original = scan_history.append_jsonl
        def fail(path, event):
            if event['event'] == 'move_completed':
                raise OSError('disk unavailable after movement')
            return original(path, event)
        with mock.patch.object(scan_history, 'append_jsonl', side_effect=fail):
            with self.assertRaises(OSError):
                self.session.record('move_completed', x_steps=40, y_steps=0)
        result = recover_scan_checkpoint(self.session, None)
        self.assertEqual(result['action'], 'retry_current')
        self.assertIsNone(self.session.history_error)
        self.assertIsNotNone(self.session.find_event('move_completed'))
        self.assertEqual(self.reconciliation_events(), [])

    def test_changed_or_missing_image_blocks_before_repairing_logs(self):
        for changed in (b'altered raw image', None):
            with self.subTest(changed=changed):
                if changed is None:
                    self.image.unlink()
                else:
                    self.image.write_bytes(changed)
                with mock.patch.object(self.session, 'recover_history') as recover:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        recover_scan_checkpoint(self.session, self.checkpoint)
                    recover.assert_not_called()
                self.assertFalse(self.checkpoint['reconciled'])

    def test_invalid_or_undecodable_image_blocks_before_repair(self):
        with mock.patch.object(self.session, 'recover_history') as recover:
            with self.assertRaisesRegex(ValueError, 'decoded'):
                recover_scan_checkpoint(self.session, self.checkpoint,
                                        image_decoder=lambda path: False)
            recover.assert_not_called()

    def test_real_decoder_accepts_valid_png_and_reports_original_bytes(self):
        import cv2
        def decode(path):
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            return image is not None and image.shape == (2, 2, 3)
        result = recover_scan_checkpoint(self.session, self.checkpoint, image_decoder=decode)
        self.assertEqual(result['image_bytes'], len(self.raw))
        self.assertEqual(result['image_sha256'], hashlib.sha256(self.raw).hexdigest())

    def test_image_changed_during_decode_is_rejected(self):
        def decoder(path):
            path.write_bytes(b'changed during decode')
            return True
        with mock.patch.object(self.session, 'recover_history') as recover:
            with self.assertRaisesRegex(ValueError, 'during validation'):
                recover_scan_checkpoint(self.session, self.checkpoint, image_decoder=decoder)
            recover.assert_not_called()

    def test_failed_reconciliation_mirror_is_restored_once_with_original_event_id(self):
        original = self.shared.record
        def fail(event, **fields):
            if event == 'recovery_point_reconciled':
                raise OSError('shared mirror unavailable')
            return original(event, **fields)
        with mock.patch.object(self.shared, 'record', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'shared mirror'):
                recover_scan_checkpoint(self.session, self.checkpoint)
        self.assertFalse(self.checkpoint['reconciled'])
        pending_id = self.reconciliation_events()[0]['event_id']
        result = recover_scan_checkpoint(self.session, self.checkpoint)
        self.assertEqual(result['reconciliation_event_id'], pending_id)
        self.assertEqual(len(self.reconciliation_events()), 1)
        forwarded = [event for event in self.shared._events
                     if event.get('source_event_id') == pending_id]
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(self.image.read_bytes(), self.raw)

    def test_partial_reconciliation_journal_retry_preserves_bytes_and_event_identity(self):
        original = scan_history.append_jsonl
        damaged = []
        def partial(path, event):
            if event['event'] == 'recovery_point_reconciled':
                prefix = (json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')[:83]
                damaged.append(prefix)
                with Path(path).open('ab') as stream:
                    stream.write(prefix)
                raise OSError('disk full in reconciliation append')
            return original(path, event)
        with mock.patch.object(scan_history, 'append_jsonl', side_effect=partial):
            with self.assertRaises(OSError):
                recover_scan_checkpoint(self.session, self.checkpoint)
        journal_before = (self.session.folder / 'events.jsonl').read_bytes()
        self.assertTrue(journal_before.endswith(damaged[0]))
        pending_id = self.reconciliation_events()[0]['event_id']
        result = recover_scan_checkpoint(self.session, self.checkpoint)
        self.assertEqual(result['reconciliation_event_id'], pending_id)
        self.assertEqual(len(self.reconciliation_events()), 1)
        backups = list((self.session.folder / 'recovery_backups').rglob('*'))
        self.assertTrue(any(path.is_file() and path.read_bytes() == journal_before for path in backups))
        self.assertEqual(self.image.read_bytes(), self.raw)

    def test_unresolved_storage_failure_keeps_checkpoint_unreconciled(self):
        with mock.patch.object(self.session, 'recover_history',
                               side_effect=OSError('still out of disk space')):
            with self.assertRaises(OSError):
                recover_scan_checkpoint(self.session, self.checkpoint)
        self.assertFalse(self.checkpoint['reconciled'])
        self.assertEqual(self.reconciliation_events(), [])

    def test_a_reconciled_flag_without_durable_event_is_not_accepted(self):
        self.checkpoint['reconciled'] = True
        with self.assertRaisesRegex(ValueError, 'audit event is missing'):
            recover_scan_checkpoint(self.session, self.checkpoint)
        self.assertEqual(self.reconciliation_events(), [])

    def test_reused_reconciliation_identity_with_different_point_is_rejected(self):
        recover_scan_checkpoint(self.session, self.checkpoint)
        self.checkpoint['point'] = 2
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            recover_scan_checkpoint(self.session, self.checkpoint)
        self.assertEqual(len(self.reconciliation_events()), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
