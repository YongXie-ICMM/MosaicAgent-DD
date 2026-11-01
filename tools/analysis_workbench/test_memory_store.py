"""Bounded memory collection tests using synthetic saved logs only."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('memory_store_under_test', Path(__file__).with_name('memory_store.py'))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.memory = M.MemoryStore(self.root / 'Memory')
        self.device = self.root / 'device_diagnostics/run-shared'
        self.device.mkdir(parents=True)
        for name in ['communications.jsonl', 'communications.jsonl.1', 'incident_one.json', 'incident_one.post.jsonl']:
            (self.device / name).write_text('{"event":"fixture"}\n')
        self.sdk = self.device / 'ximc_sdk_raw.log'
        self.sdk.write_bytes(b'0123456789abcdef')

    def scan(self, name='scan-A', session_id='session-A', **pointer_changes):
        root = self.root / name
        root.mkdir()
        (root / 'session.json').write_text(json.dumps({'session_id': session_id, 'status': 'completed'}))
        (root / 'events.jsonl').write_text('{"event":"session_finished"}\n')
        (root / 'sharpness.csv').write_text('row,col,sharpness\n0,0,200\n')
        pointer = {'schema_version': 1, 'run_id': 'same-process',
                   'diagnostics_directory': str(self.device), 'sdk_raw_log': str(self.sdk)}
        pointer.update(pointer_changes)
        (root / 'device_diagnostics.json').write_text(json.dumps(pointer))
        snapshots = root / 'program_snapshot'
        snapshots.mkdir()
        (snapshots / 'Camera_v2.py').write_text('# fixture source\n')
        (snapshots / 'delivery_manifest.json').write_text('{"fixture":true}\n')
        (snapshots / '.env').write_text('API_KEY=must-not-copy')
        (snapshots / 'libximc.dll').write_bytes(b'not copied')
        (snapshots / 'arbitrary.py').write_text('# not allowed\n')
        (root / 'mosaic_r0_c0.png').write_bytes(b'raw pixels must not copy')
        (root / '.env').write_text('SECRET=must-not-copy')
        return root

    def test_append_only_events_redact_keys_and_summary_returns_last80(self):
        first = self.memory.record('scan', 'completed', api_key='not-logged', details={'password': 'no'}, raw=b'no bytes')
        prefix = (self.memory.root / 'events.jsonl').read_bytes()
        for i in range(85):
            self.memory.record('review', 'pending', point=i)
        self.assertTrue((self.memory.root / 'events.jsonl').read_bytes().startswith(prefix))
        summary = self.memory.summary()
        self.assertEqual(summary['event_count'], 86)
        self.assertEqual(len(summary['latest_events']), 80)
        self.assertEqual(first['fields']['api_key'], '<redacted>')
        self.assertEqual(first['fields']['raw'], '<non-metadata value omitted>')
        self.assertNotIn(b'not-logged', (self.memory.root / 'events.jsonl').read_bytes())

    def test_independent_store_instances_preserve_concurrent_event_index(self):
        other = M.MemoryStore(self.memory.root)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: (self.memory if i % 2 else other).record('job', 'done', number=i), range(32)))
        summary = other.summary()
        self.assertEqual(summary['event_count'], 32)
        self.assertEqual(len({e['event_id'] for e in summary['latest_events']}), 32)

    def test_snapshot_allowlist_never_moves_or_copies_raw_secrets_or_dlls(self):
        scan = self.scan()
        (self.device / 'communications.jsonl.5').write_text('excluded rotation')
        (self.device / 'incident_secret.json').write_text('must not copy')
        before = {p: p.read_bytes() for p in scan.rglob('*') if p.is_file()}
        result = self.memory.snapshot_acquisition(scan)
        copied = [r for r in result['files'] + result['process_files'] if r['status'] == 'included']
        sources = {Path(r['source']).name for r in copied}
        self.assertEqual(sources, {'session.json', 'events.jsonl', 'sharpness.csv', 'device_diagnostics.json',
                                  'Camera_v2.py', 'delivery_manifest.json', 'communications.jsonl',
                                  'communications.jsonl.1', 'incident_one.json', 'incident_one.post.jsonl', 'ximc_sdk_raw.log'})
        for record in copied:
            self.assertEqual(sha(record['path']), record['sha256'])
            self.assertFalse(record['snapshot_atomic'])
            self.assertIn('source_before', record)
            self.assertIn('source_after', record)
        self.assertEqual(before, {p: p.read_bytes() for p in scan.rglob('*') if p.is_file()})
        self.assertFalse(result['images_included'])
        self.assertEqual(len(self.memory.summary()['files']), len(copied))

    def test_shared_process_logs_deduplicate_across_two_distinct_scans(self):
        scan1 = self.scan()
        scan2 = self.scan('scan-B', 'session-B')
        a = self.memory.snapshot_acquisition(scan1)
        b = self.memory.snapshot_acquisition(scan2)
        self.assertNotEqual(a['manifest_path'], b['manifest_path'])
        self.assertEqual(a['process_manifest_path'], b['process_manifest_path'])
        self.assertTrue(all(r['reused'] for r in b['process_files'] if r['status'] == 'included'))
        self.assertEqual(len(self.memory.summary()['device_runs']), 1)
        self.assertEqual(len(self.memory.summary()['acquisitions']), 2)
        self.assertEqual({r['path'] for r in a['process_files']}, {r['path'] for r in b['process_files']})

    def test_unchanged_snapshot_reuses_but_changed_file_keeps_old_revision(self):
        scan = self.scan()
        a = self.memory.snapshot_acquisition(scan)
        b = self.memory.snapshot_acquisition(scan)
        self.assertEqual(a['manifest_path'], b['manifest_path'])
        old_manifest = Path(a['manifest_path']).read_bytes()
        old_file = next(r for r in a['files'] if Path(r['source']).name == 'events.jsonl')['path']
        (scan / 'events.jsonl').write_text('{"event":"later"}\n')
        c = self.memory.snapshot_acquisition(scan)
        self.assertNotEqual(a['manifest_path'], c['manifest_path'])
        self.assertEqual(Path(a['manifest_path']).read_bytes(), old_manifest)
        self.assertIn(b'session_finished', Path(old_file).read_bytes())

    def test_changed_shared_log_creates_new_process_and_acquisition_snapshots(self):
        scan = self.scan()
        a = self.memory.snapshot_acquisition(scan)
        old = Path(a['process_manifest_path']).read_bytes()
        with (self.device / 'communications.jsonl').open('a') as stream:
            stream.write('{"new":true}\n')
        b = self.memory.snapshot_acquisition(scan)
        self.assertNotEqual(a['process_manifest_path'], b['process_manifest_path'])
        self.assertNotEqual(a['manifest_path'], b['manifest_path'])
        self.assertEqual(Path(a['process_manifest_path']).read_bytes(), old)

    def test_foreign_windows_pointer_is_explicit_unavailable_not_guessed(self):
        scan = self.scan(diagnostics_directory=r'D:\user\device_diagnostics\same-process', sdk_raw_log=r'D:\user\device_diagnostics\same-process\ximc.log')
        result = self.memory.snapshot_acquisition(scan)
        self.assertEqual(result['process_files'], [])
        self.assertEqual({w['reason'] for w in result['warnings']}, {'remote_windows_path_unavailable'})
        self.assertTrue(Path(result['manifest_path']).is_file())

    def test_missing_process_identity_does_not_merge_distinct_scan_sessions(self):
        scan1 = self.scan()
        scan2 = self.scan('scan-B', 'session-B')
        for scan in [scan1, scan2]:
            (scan / 'device_diagnostics.json').unlink()
        a = self.memory.snapshot_acquisition(scan1)
        b = self.memory.snapshot_acquisition(scan2)
        self.assertNotEqual(a['process_run_id'], b['process_run_id'])
        self.assertEqual(len(self.memory.summary()['device_runs']), 2)
        self.assertEqual(a['status'], 'partial')

    def test_relative_sdk_needs_recorded_base_and_traversal_is_rejected(self):
        scan = self.scan(sdk_raw_log='ximc_sdk_raw.log')
        a = self.memory.snapshot_acquisition(scan)
        self.assertTrue(any(w['reason'] == 'relative_path_without_recorded_base' for w in a['warnings']))
        pointer = json.loads((scan / 'device_diagnostics.json').read_text())
        pointer['sdk_raw_log_base_cwd'] = str(self.device)
        (scan / 'device_diagnostics.json').write_text(json.dumps(pointer))
        b = self.memory.snapshot_acquisition(scan)
        self.assertTrue(any(r['role'] == 'sdk_raw_tail' and r['status'] == 'included' for r in b['process_files']))
        pointer['sdk_raw_log'] = '../secret.log'
        (scan / 'device_diagnostics.json').write_text(json.dumps(pointer))
        c = self.memory.snapshot_acquisition(scan)
        self.assertTrue(any(w['reason'] == 'path_traversal_rejected' for w in c['warnings']))

    def test_symlink_sources_and_ancestors_are_not_followed(self):
        scan = self.scan()
        actual = scan / 'events.jsonl'
        actual.unlink()
        outside = self.root / 'outside.jsonl'
        outside.write_text('do not follow')
        actual.symlink_to(outside)
        result = self.memory.snapshot_acquisition(scan)
        record = next(r for r in result['files'] if r['source'] == str(actual))
        self.assertEqual(record['status'], 'unavailable')
        link = self.root / 'scan-link'
        link.symlink_to(scan, target_is_directory=True)
        with self.assertRaisesRegex(M.MemoryStoreError, '符号链接'):
            self.memory.snapshot_acquisition(link)

    def test_structured_size_limits_explicit_and_sdk_tail_has_offsets(self):
        scan = self.scan()
        with patch.object(M, 'MAX_STRUCTURED_FILE_BYTES', 8), patch.object(M, 'MAX_SDK_TAIL_BYTES', 5):
            result = self.memory.snapshot_acquisition(scan)
        self.assertTrue(any(r['status'] == 'unavailable' and 'size limit' in r['reason'] for r in result['files']))
        sdk = next(r for r in result['process_files'] if r['role'] == 'sdk_raw_tail')
        self.assertEqual(Path(sdk['path']).read_bytes(), b'bcdef')
        self.assertEqual(sdk['start_offset_bytes'], 11)
        self.assertEqual(sdk['end_offset_bytes_exclusive'], 16)
        self.assertTrue(sdk['truncated'] and sdk['first_record_may_be_partial'])
        self.assertEqual(result['status'], 'partial')
        self.assertTrue(result['unavailable_file_count'])

    def test_total_structured_budget_is_shared_across_session_and_device_files(self):
        scan = self.scan()
        with patch.object(M, 'MAX_STRUCTURED_TOTAL_BYTES', 70):
            result = self.memory.snapshot_acquisition(scan)
        self.assertLessEqual(result['structured_bytes'], 70)
        included = [r for r in result['files'] + result['process_files']
                    if r['status'] == 'included' and r['role'] != 'sdk_raw_tail']
        self.assertEqual(sum(r['bytes'] for r in included), result['structured_bytes'])
        self.assertEqual(result['status'], 'partial')

    def test_summary_does_not_follow_edited_index_manifest_outside_memory(self):
        scan = self.scan()
        self.memory.snapshot_acquisition(scan)
        outside = self.root / 'outside.json'
        outside.write_text('{"private":"DO_NOT_INSPECT_74717"}')
        path = self.memory.root / 'index.json'
        index = json.loads(path.read_text())
        index['acquisitions']['session-A']['latest_manifest_path'] = str(outside)
        index['acquisitions']['session-A']['sha256'] = sha(outside)
        path.write_text(json.dumps(index))
        result = self.memory.summary()
        self.assertTrue(any('escaped Memory root' in w['reason'] for w in result['warnings']))
        self.assertNotIn('DO_NOT_INSPECT_74717', json.dumps(result))

    def test_source_changed_during_copy_is_recorded_and_never_called_atomic(self):
        source = self.root / 'live.log'
        source.write_bytes(b'abcdef')
        original = M.os.fstat
        calls = [0]
        def changed(fd):
            result = original(fd)
            calls[0] += 1
            if calls[0] == 2:
                with source.open('ab') as stream:
                    stream.write(b'new')
                result = original(fd)
            return result
        with patch.object(M.os, 'fstat', side_effect=changed):
            data, record = M._read_segment(source, 100)
        self.assertEqual(data, b'abcdef')
        self.assertTrue(record['source_changed_during_copy'])
        self.assertFalse(record['snapshot_atomic'])

    def test_memory_object_tampering_never_overwrites_retained_history(self):
        scan = self.scan()
        first = self.memory.snapshot_acquisition(scan)
        saved = Path(next(r for r in first['files'] if Path(r['source']).name == 'events.jsonl')['path'])
        saved.write_text('tampered')
        second = self.memory.snapshot_acquisition(scan)
        current = next(r for r in second['files'] if Path(r['source']).name == 'events.jsonl')
        self.assertEqual(current['status'], 'unavailable')
        self.assertIn('refusing to overwrite', current['reason'])
        self.assertEqual(saved.read_text(), 'tampered')

    def test_legacy_references_make_one_index_node_without_moving_sources(self):
        legacy = self.root / 'old history'
        legacy.mkdir()
        (legacy / 'a.txt').write_text('retain')
        a = self.memory.legacy_reference(legacy, 'Previous scan logs')
        b = self.memory.legacy_reference(legacy, 'Earlier logs')
        self.assertEqual(a['reference_id'], b['reference_id'])
        refs = self.memory.summary()['legacy_references']
        self.assertEqual(len(refs), 1)
        self.assertEqual(len(refs[0]['labels']), 2)
        self.assertEqual((legacy / 'a.txt').read_text(), 'retain')
        self.assertFalse(any(p.name == 'a.txt' for p in self.memory.root.rglob('*')))


if __name__ == '__main__':
    unittest.main()
