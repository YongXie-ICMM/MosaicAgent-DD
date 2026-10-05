"""Offline persistence and multi-reader checks; no camera or stage is imported."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import threading
import unittest
from unittest import mock

import shared_history as history_module
from shared_history import (SharedHistory, list_histories, read_agent_notes, read_history,
                            rebuild_history, write_agent_note)


class SharedHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.history = SharedHistory(self.root / 'shared_history',
                                     {'stage_simulated': True, 'operator': '离线测试'})

    def tearDown(self):
        self.temp.cleanup()

    def test_events_have_stable_identifiers_and_json_safe_values(self):
        evidence = {'position': [1, 2], 'unknown': float('nan')}
        event = self.history.record('move_attempt', evidence=evidence)
        evidence['position'][0] = 900
        event['evidence']['position'][1] = 800
        value = read_history(self.history.path)
        self.assertEqual(value['event_count'], 2)
        recorded = value['events'][-1]
        self.assertEqual(recorded['evidence'], {'position': [1, 2], 'unknown': None})
        self.assertEqual(recorded['history_id'], self.history.history_id)
        self.assertEqual(recorded['actor'], 'acquisition_program')
        self.assertEqual(recorded['sequence'], 2)
        self.assertTrue(recorded['event_id'])
        self.assertTrue(value['metadata']['stage_simulated'])
        self.assertEqual(value['status'], 'open')
        self.assertFalse(value['is_closed'])

    def test_reserved_fields_cannot_be_overwritten(self):
        for field in ('event_id', 'history_id', 'sequence', 'timestamp_utc', 'schema_version'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.history.record('move_attempt', **{field: 'overwrite'})
        self.assertEqual(read_history(self.history.path)['event_count'], 1)

    def test_paths_are_relative_and_remain_linked_after_folder_move(self):
        scan = self.root / 'mosaic_photos' / 'scan001'
        scan.mkdir(parents=True)
        image = scan / 'mosaic_r1_c1.png'
        image.write_bytes(b'not an actual camera image')
        self.history.record('capture_success', filepath=str(image),
                            scan_folder=str(scan), nested={'source_path': image})
        value = read_history(self.history.path)
        event = value['events'][-1]
        self.assertFalse(Path(event['filepath']).is_absolute())
        self.assertEqual((self.history.folder / event['filepath']).resolve(), image.resolve())
        self.assertEqual(event['filepath'], '../../mosaic_photos/scan001/mosaic_r1_c1.png')
        self.assertEqual(event['nested']['source_path'], event['filepath'])
        self.assertEqual((self.history.folder / event['scan_folder']).resolve(), scan.resolve())

    def test_windows_path_separators_and_different_drives(self):
        # Exercise the Windows separator/relpath outputs on this offline host.
        image = self.root / 'mosaic_photos' / 'scan001' / 'frame.png'
        with mock.patch.object(history_module.os, 'sep', '\\'):
            with mock.patch.object(history_module.os.path, 'relpath',
                                   return_value='..\\..\\mosaic_photos\\scan001\\frame.png'):
                for value in (image, str(image.resolve())):
                    converted = history_module._clean(value, self.history.folder, 'filepath')
                    self.assertEqual(converted, '../../mosaic_photos/scan001/frame.png')
            self.assertEqual(history_module._clean('images\\frame.png', self.history.folder,
                                                   'scan_image_relative_path'), 'images/frame.png')
            with mock.patch.object(history_module.os.path, 'relpath', side_effect=ValueError('different drives')):
                self.assertEqual(history_module._clean(image, self.history.folder), str(image.resolve()))
                with mock.patch.object(history_module.os.path, 'isabs', return_value=True):
                    self.assertEqual(history_module._clean('D:\\data\\frame.png', self.history.folder,
                                                           'filepath'), 'D:/data/frame.png')

    def test_concurrent_readers_never_observe_partial_snapshot(self):
        errors = []
        finished = threading.Event()
        observations = []

        def reader():
            while not finished.is_set():
                try:
                    value = read_history(self.history.path)
                    self.assertEqual(value['event_count'], len(value['events']))
                    self.assertEqual(value['snapshot_sequence'], value['events'][-1]['sequence'])
                    observations.append(value['event_count'])
                except Exception as exc:
                    errors.append(exc)
                    return

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for thread in readers:
            thread.start()
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                list(executor.map(lambda number: self.history.record('move_completed', number=number), range(30)))
        finally:
            finished.set()
            for thread in readers:
                thread.join(timeout=5)
        self.assertFalse(errors, errors)
        self.assertTrue(observations)
        final = read_history(self.history.path)
        self.assertEqual(final['event_count'], 31)
        self.assertEqual([e['sequence'] for e in final['events']], list(range(1, 32)))
        self.assertEqual(len({e['event_id'] for e in final['events']}), 31)

    def test_snapshot_write_failure_is_detected_and_no_sequence_reused(self):
        with mock.patch.object(history_module, '_atomic_json', side_effect=OSError('disk snapshot failure')):
            with self.assertRaises(OSError):
                self.history.record('candidate_saved', scan_session_id='s001')
        old = read_history(self.history.path)
        self.assertEqual(old['event_count'], 1)
        self.assertEqual(old['read_diagnostics']['snapshot_lag'], 1)
        recovered = read_history(self.history.path, recover=True)
        self.assertEqual(recovered['events'][-1]['event'], 'candidate_saved')
        self.assertEqual(recovered['event_count'], 2)
        self.history.record('human_review', decision='accept')
        current = read_history(self.history.path)
        self.assertEqual([e['sequence'] for e in current['events']], [1, 2, 3])
        self.assertEqual(current['read_diagnostics']['snapshot_lag'], 0)

    def test_transient_windows_replace_lock_does_not_duplicate_events(self):
        replace = history_module.os.replace
        attempts = []

        def briefly_locked(source, destination):
            attempts.append((source, destination))
            if len(attempts) < 3:
                raise PermissionError('simulated transient Windows sharing violation')
            return replace(source, destination)

        with mock.patch.object(history_module.os, 'replace', side_effect=briefly_locked), \
                mock.patch.object(history_module.time, 'sleep') as sleep:
            self.history.record('candidate_saved')
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)
        value = read_history(self.history.path)
        self.assertEqual(value['event_count'], 2)
        self.assertEqual(value['read_diagnostics']['snapshot_lag'], 0)
        self.assertEqual(len(self.history.journal_path.read_text().splitlines()), 2)

    def test_append_failure_blocks_further_writes(self):
        with mock.patch.object(history_module, '_append_event', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.history.record('move_attempt')
        with self.assertRaisesRegex(RuntimeError, 'append previously failed'):
            self.history.record('move_completed')
        self.assertEqual(read_history(self.history.path)['event_count'], 1)

    def test_incomplete_tail_is_reported_without_changing_raw_data(self):
        with self.history.journal_path.open('ab') as stream:
            stream.write(b'{"event":"capture_')
        before = self.history.journal_path.read_bytes()
        value = read_history(self.history.path)
        recovered = rebuild_history(self.history.path)
        self.assertEqual(value['read_diagnostics']['warning'], 'uncommitted_trailing_line')
        self.assertEqual(recovered['event_count'], 1)
        self.assertEqual(self.history.journal_path.read_bytes(), before)
        self.assertFalse(recovered['is_closed'])

    def test_valid_json_without_newline_is_still_uncommitted(self):
        pending = copy.deepcopy(read_history(self.history.path)['events'][0])
        pending.update(event='move_attempt', sequence=2, event_id='new-event')
        with self.history.journal_path.open('ab') as stream:
            stream.write(json.dumps(pending).encode('utf-8'))
        recovered = read_history(self.history.path, recover=True)
        self.assertEqual(recovered['event_count'], 1)
        self.assertEqual(recovered['read_diagnostics']['warning'], 'uncommitted_trailing_line')

    def test_corrupt_committed_journal_line_raises(self):
        with self.history.journal_path.open('ab') as stream:
            stream.write(b'{broken}\n')
        with self.assertRaisesRegex(ValueError, 'Malformed committed journal line 2'):
            read_history(self.history.path)

    def test_tampered_snapshot_is_rejected_and_recovery_does_not_rewrite(self):
        value = json.loads(self.history.path.read_text())
        value['status'] = 'success'
        self.history.path.write_text(json.dumps(value))
        original = self.history.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            read_history(self.history.path)
        recovered = read_history(self.history.path, recover=True)
        self.assertEqual(recovered['status'], 'open')
        self.assertTrue(recovered['read_diagnostics']['snapshot_error'])
        self.assertEqual(self.history.path.read_bytes(), original)

    def test_missing_snapshot_can_be_recovered_without_completion_claim(self):
        self.history.path.unlink()
        with self.assertRaises(FileNotFoundError):
            read_history(self.history.path)
        value = rebuild_history(self.history.path)
        self.assertEqual(value['status'], 'open')
        self.assertFalse(value['is_closed'])
        self.assertFalse(self.history.path.exists())

    def test_snapshot_without_journal_has_explicit_limitation(self):
        self.history.journal_path.unlink()
        value = read_history(self.history.path)
        self.assertFalse(value['read_diagnostics']['journal_available'])
        self.assertIsNone(value['read_diagnostics']['snapshot_lag'])

    def test_recovery_output_cannot_overwrite_acquisition_files(self):
        for output in (self.history.path, self.history.journal_path,
                       self.history.folder / 'recovered.json'):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                rebuild_history(self.history.path, output=output)
        output = self.root / 'review' / 'recovered.json'
        recovered = rebuild_history(self.history.path, output=output)
        self.assertEqual(json.loads(output.read_text()), recovered)
        with self.assertRaises(ValueError):
            rebuild_history(self.history.path, output=output)

    def test_agent_notes_keep_acquisition_byte_identical(self):
        event = self.history.record('capture_success', scan_session_id='scan001', image_sha256='abc')
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (self.history.path, self.history.journal_path)}
        with ThreadPoolExecutor(max_workers=4) as executor:
            notes = list(executor.map(
                lambda number: write_agent_note(self.history.path, 'agent-%d' % number,
                                               {'finding': 'await independent reference'},
                                               [event['event_id']]), range(8)))
        self.assertEqual(len(set(notes)), 8)
        for path in notes:
            value = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(value['history_id'], self.history.history_id)
            self.assertEqual(value['source_snapshot_sequence'], 2)
            self.assertEqual(value['related_event_ids'], [event['event_id']])
            self.assertEqual(value['source_history_path'], '../history.json')
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_agent_note_rejects_unknown_source_event(self):
        with self.assertRaisesRegex(ValueError, 'refer to this acquisition history'):
            write_agent_note(self.history.path, 'qa', 'bad reference', ['different-history-event'])
        self.assertFalse((self.history.folder / 'agent_notes').exists())

    def test_peer_notes_concurrent_read_and_invalid_notes_are_visible(self):
        import io
        event = self.history.record('candidate_saved')
        originals = {path: path.read_bytes() for path in (self.history.path, self.history.journal_path)}

        def publish_and_read(number):
            write_agent_note(self.history.path, 'agent-%d' % number, 'Review the candidate.',
                             [event['event_id']])
            view = read_agent_notes(self.history.path)
            self.assertFalse(view['invalid_notes'])
            return view

        with ThreadPoolExecutor(max_workers=4) as executor:
            views = list(executor.map(publish_and_read, range(8)))
        self.assertTrue(all(view['notes'] for view in views))
        valid = read_agent_notes(self.history.path)
        self.assertEqual(len(valid['notes']), 8)
        self.assertEqual({note['agent_id'] for note in valid['notes']},
                         {'agent-%d' % number for number in range(8)})
        invalid = write_agent_note(self.history.path, 'bad-history', 'Deliberately invalid test fixture.')
        value = json.loads(invalid.read_text())
        value['history_id'] = 'another-history'
        invalid.write_text(json.dumps(value))
        invalid_ref = write_agent_note(self.history.path, 'bad-ref', 'Invalid reference test fixture.')
        value = json.loads(invalid_ref.read_text())
        value['related_event_ids'] = ['not-an-acquisition-event']
        invalid_ref.write_text(json.dumps(value))
        (invalid.parent / 'malformed.json').write_text('{broken JSON')
        result = read_agent_notes(self.history.path)
        self.assertEqual(len(result['notes']), 8)
        self.assertEqual(len(result['invalid_notes']), 3)
        self.assertTrue(all(note['note_path'].startswith('agent_notes/')
                            for note in result['notes'] + result['invalid_notes']))
        output = io.StringIO()
        with mock.patch('sys.stdout', output):
            self.assertEqual(history_module.main(['notes', str(self.history.path)]), 0)
        self.assertEqual(len(json.loads(output.getvalue())['invalid_notes']), 3)
        for path, data in originals.items():
            self.assertEqual(path.read_bytes(), data)

    def test_unique_application_histories_and_discovery(self):
        second = SharedHistory(self.root / 'shared_history', {'stage_simulated': False})
        second.close(status='operator_closed')
        rows = list_histories(self.root / 'shared_history')
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row['history_id'] for row in rows}), 2)
        self.assertEqual({row['status'] for row in rows}, {'open', 'operator_closed'})
        self.assertEqual(list_histories(self.root / 'missing'), [])

    def test_close_is_idempotent_and_prevents_later_events(self):
        event = self.history.close(reason='operator exited')
        self.assertEqual(self.history.close(), event)
        value = read_history(self.history.path)
        self.assertTrue(value['is_closed'])
        self.assertEqual(value['status'], 'closed')
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.history.record('move_attempt')

    def test_scan_provenance_is_preserved_in_flat_event_fields(self):
        self.history.record('move_completed', scan_session_id='scan-001',
                            source_event_id='original-event', source_sequence=23,
                            source_timestamp_utc='2026-09-29T10:00:00+00:00',
                            verification={'basis': 'controller_counter_only', 'physical_travel_verified': False})
        event = read_history(self.history.path)['events'][-1]
        self.assertEqual(event['scan_session_id'], 'scan-001')
        self.assertEqual(event['source_sequence'], 23)
        self.assertFalse(event['verification']['physical_travel_verified'])

    def test_cli_show_returns_json_and_does_not_change_files(self):
        import io
        output = io.StringIO()
        before = self.history.path.read_bytes()
        with mock.patch('sys.stdout', output):
            self.assertEqual(history_module.main(['show', str(self.history.path)]), 0)
        self.assertEqual(json.loads(output.getvalue())['history_id'], self.history.history_id)
        self.assertEqual(self.history.path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
