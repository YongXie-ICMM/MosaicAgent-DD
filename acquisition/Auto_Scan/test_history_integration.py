"""Windowless acquisition/history integration; fake camera and stage only."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

from camera_session_history import ScanSession
from test_scan_review import load_logic, Stage, Camera, Widget
from test_scan_v3_offline import Value


class HistoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        from shared_history import SharedHistory
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = SharedHistory(self.root / 'shared_history', metadata={'stage_simulated': True})
        self.ns = load_logic()
        self.gui = object.__new__(self.ns['MosaicNavigatorGUI'])
        g = self.gui
        g._history = self.history
        g._shared_history = self.history
        g._history_error = None
        g._history_help_shown = False
        g._history_root = self.root / 'shared_history'
        g._ensure_history = lambda: self.history
        g.stage = Stage()
        g.stage.connected = True
        g.stage.last_move_evidence = {'status': 'counter_checked', 'physical_displacement_verified': False}
        g.stage.last_position_evidence = {'status': 'counter_checked'}
        g.cam = Camera(g.stage)
        g.connected_camera = True
        g.auto_running = False
        g._camera_connecting = False
        g._run_auto_capture = True
        # These legacy tests isolate review/history using non-image byte fixtures.
        # test_acquisition_contract exercises the real new guards with PNGs.
        g._verify_acquisition_mode = lambda *a, **kw: None
        g._check_candidate_dimensions = lambda *a, **kw: [1920, 1080]
        g._run_review_mode = 'step_confirm'
        g._camera_lock = threading.Lock()
        g._cancel_event = threading.Event()
        g._capture_attempts = g._capture_failures = g._photo_count = 0
        g._human_review_count = g._scan_done_count = 0
        g._sharpness_records = []
        g._last_accepted_photo = None
        g._scan_origin = (0, 0)
        g._worker_error = None
        g._planned_total = 2
        g._scan_params = None
        g._worker = None
        g._close_preview = lambda: None
        g._start_preview = lambda: None
        g._update_gui_status = lambda *args: None
        g._refresh_pos = lambda: None
        self.gui_callbacks = []
        def after(delay, callback, *args):
            if threading.current_thread() is threading.main_thread() and not delay:
                callback(*args)
            else:
                self.gui_callbacks.append((callback, args))
        g.after = after
        g.var_sample_id = Value('SP_HISTORY_TEST')
        g.var_objective = Value('20X')
        g.var_dx = Value('40')
        g.var_dy = Value('40')
        for name in ('var_status', 'var_history_status', 'var_progress', 'btn_auto', 'btn_resume'):
            setattr(g, name, Widget())
        self.errors = []
        def showerror(title, message, **kwargs):
            self.assertIs(threading.current_thread(), threading.main_thread())
            self.errors.append((title, message))
        self.ns['messagebox'] = SimpleNamespace(showwarning=lambda *args: None,
                                                 showerror=showerror)
        self.scan = self.root / 'mosaic_photos' / 'scan01'
        g._session_folder = str(self.scan)
        g._session_log = ScanSession(self.scan, {'backend': 'fake'},
                                     {'sample_id': 'SP_HISTORY_TEST'}, [],
                                     simulated=True, shared_history=self.history)

    def events(self):
        return json.loads(self.history.path.read_text(encoding='utf-8'))['events']

    def scan_events(self):
        return [json.loads(line) for line in (self.scan / 'events.jsonl').read_text(encoding='utf-8').splitlines()]

    def wait_manual(self):
        # The production method starts a thread; join the saved reference because
        # its completion callback can set the GUI attribute back to None.
        worker = self.gui._worker
        if worker is not None:
            worker.join(2)
            self.assertFalse(worker.is_alive())
        callbacks, self.gui_callbacks = self.gui_callbacks, []
        for callback, args in callbacks:
            callback(*args)

    def test_manual_move_records_intent_before_command_and_checked_result_afterward(self):
        g = self.gui
        move = g.stage.move_relative
        observed = []
        def checked_move(dx, dy, cancel):
            attempt = self.events()[-1]
            observed.append(attempt)
            self.assertEqual(attempt['event'], 'manual_move_attempt')
            self.assertEqual(attempt['dx_steps'], 40)
            self.assertEqual(attempt['sample_id'], 'SP_HISTORY_TEST')
            move(dx, dy, cancel)
        g.stage.move_relative = checked_move
        g.manual_move(1, 0)
        self.wait_manual()
        self.assertEqual(len(observed), 1)
        self.assertEqual(g.stage.moves, [(40, 0)])
        completed = self.events()[-1]
        self.assertEqual(completed['event'], 'manual_move_completed')
        self.assertEqual(completed['x_steps'], 40)
        self.assertFalse(completed['verification']['physical_displacement_verified'])

    def test_manual_move_does_not_dispatch_when_intent_cannot_be_recorded(self):
        g = self.gui
        def fail(*args, **kwargs):
            raise OSError('history disk unavailable')
        g._record_history = fail
        g.manual_move(1, 0)
        self.wait_manual()
        self.assertEqual(g.stage.moves, [])
        self.assertFalse(g.auto_running)
        self.assertIn('history disk unavailable', g.var_status.value)

    def test_failed_manual_move_keeps_requested_delta_and_controller_evidence(self):
        g = self.gui
        def fail(dx, dy, cancel):
            g.stage.last_move_evidence = {'status': 'counter_mismatch', 'observed_dx': 0}
            raise RuntimeError('controller did not advance')
        g.stage.move_relative = fail
        g.manual_move(1, 0)
        self.wait_manual()
        failed = self.events()[-1]
        self.assertEqual(failed['event'], 'manual_move_failed')
        attempt = self.events()[-2]
        self.assertEqual(attempt['event'], 'manual_move_attempt')
        self.assertEqual(attempt['dx_steps'], 40)
        self.assertEqual(failed['verification']['observed_dx'], 0)
        self.assertIn('controller did not advance', failed['error'])

    def test_stop_precedes_logging_and_logging_failure_does_not_prevent_stop(self):
        g = self.gui
        calls = []
        def stop(wait):
            self.assertTrue(g._cancel_event.is_set())
            calls.append('stop')
        def fail(event, **fields):
            calls.append(event)
            raise OSError('history disk unavailable')
        g.stage.emergency_stop = stop
        g._record_history = fail
        g.on_emergency_stop()
        self.assertEqual(calls, ['stop', 'emergency_stop_requested'])
        self.assertFalse(g.stage.position_trusted)
        self.assertIsNone(g._scan_params)
        self.assertIn('历史写入失败', g.var_history_status.value)

    def test_retake_accept_move_lifecycle_is_shared_without_changing_image_identity(self):
        g = self.gui
        choices = iter(['retake', 'accept', 'accept'])
        g._request_review = lambda *args, **kwargs: next(choices)
        g._auto_scan_worker(2, 1, 40, 40, 'X_first')
        self.assertIsNone(g._worker_error)
        self.assertEqual(g._photo_count, 2)
        self.assertEqual(g.cam.reads, [(0, 0), (0, 0), (40, 0)])
        self.assertEqual(g.stage.moves, [(40, 0)])
        source = self.scan_events()
        forwarded = [event for event in self.events() if 'source_event_id' in event]
        self.assertEqual([e['source_event_id'] for e in forwarded], [e['event_id'] for e in source])
        self.assertEqual([e['source_sequence'] for e in forwarded], list(range(1, len(source) + 1)))
        self.assertEqual([e['decision'] for e in forwarded if e['event'] == 'human_review'],
                         ['retake', 'accept', 'accept'])
        first = self.scan / 'mosaic_r0_c0.png'
        self.assertEqual(first.read_bytes(), b'RAW_CANDIDATE_2')
        saved_hash = hashlib.sha256(first.read_bytes()).hexdigest()
        acceptance = next(e for e in forwarded if e['event'] == 'candidate_accepted')
        self.assertEqual(acceptance['image_sha256'], saved_hash)
        reviewed = [e for e in forwarded if e['event'] == 'candidate_saved']
        self.assertEqual(reviewed[1]['image_sha256'], saved_hash)
        self.assertTrue((self.scan / 'review_candidates' / 'r0_c0_attempt0001.png').is_file())
        g._session_log.finish('completed', photos_saved=2, spatial_coverage_verified=False)
        self.assertEqual(self.events()[-1]['status'], 'completed')
        self.assertFalse(self.events()[-1]['spatial_coverage_verified'])

    def test_shared_history_failure_after_promotion_preserves_raw_image_and_stops_progress(self):
        g = self.gui
        g._request_review = lambda *args, **kwargs: 'accept'
        original = self.history.record
        def fail_at_accept(event, **fields):
            if event == 'candidate_accepted':
                raise OSError('shared history unavailable after promotion')
            return original(event, **fields)
        self.history.record = fail_at_accept
        g._auto_scan_worker(2, 1, 40, 40, 'X_first')
        self.assertIn('shared history unavailable', str(g._worker_error))
        self.assertTrue(g._cancel_event.is_set())
        self.assertEqual(g._scan_done_count, 0)
        self.assertEqual(g.stage.moves, [])
        self.assertEqual((self.scan / 'mosaic_r0_c0.png').read_bytes(), b'RAW_CANDIDATE_1')
        self.assertFalse((self.scan / 'mosaic_r0_c1.png').exists())
        self.assertEqual(self.scan_events()[-1]['event'], 'candidate_accepted')
        self.assertFalse(any(e['event'] == 'capture_success' for e in self.events()))

    def test_transient_shared_log_failure_cannot_reenable_resume_at_finalization(self):
        g = self.gui
        g._scan_params = (2, 1, 40, 40, 'X_first')
        g._request_review = lambda *args, **kwargs: 'accept'
        original = self.history.record
        failed_once = []
        def transient_failure(event, **fields):
            if event == 'candidate_accepted' and not failed_once:
                failed_once.append(True)
                raise OSError('one failed history write')
            return original(event, **fields)
        self.history.record = transient_failure
        g._auto_scan_worker(2, 1, 40, 40, 'X_first')
        self.assertTrue(failed_once)
        g._finalize_auto()
        self.assertEqual(g.btn_resume.options['state'], 'disabled')
        self.assertIsNone(g._scan_params)
        self.assertTrue(g._history_error)
        self.assertEqual((self.scan / 'mosaic_r0_c0.png').read_bytes(), b'RAW_CANDIDATE_1')
        self.assertEqual(len(self.errors), 1)
        self.assertIn('02_start_scan.bat', self.errors[0][1])
        self.assertIn('不会自动回到原起点', self.errors[0][1])
        self.assertIn(str(self.scan), self.errors[0][1])
        self.assertEqual(g.btn_auto.options['state'], 'disabled')

    def assert_failed_resume_is_idle_and_no_operation_dispatched(self):
        g = self.gui
        self.assertFalse(g.auto_running)
        self.assertIsNone(g._worker)
        self.assertIsNone(g._scan_params)
        self.assertEqual(g.stage.moves, [])
        self.assertEqual(g.cam.reads, [])
        self.assertEqual(g.btn_resume.options['state'], 'disabled')
        self.assertTrue(g._history_error)
        self.assertIn('未开始续扫', g.var_status.value)
        self.assertIn('历史写入失败', g.var_history_status.value)
        self.assertEqual(len(self.errors), 1)
        self.assertIn('不续接旧的部分扫描', self.errors[0][1])

    def test_resume_camera_readback_logging_failure_returns_idle_without_dispatch(self):
        g = self.gui
        g._scan_params = (2, 1, 40, 40, 'X_first')
        self.ns['describe_camera'] = lambda cam: {'backend': 'fake', 'camera_index': 0}
        original = self.history.record
        def fail_readback(event, **fields):
            if event == 'resume_camera_readback':
                raise OSError('camera readback history write failed')
            return original(event, **fields)
        self.history.record = fail_readback
        g.resume_auto()
        self.assert_failed_resume_is_idle_and_no_operation_dispatched()
        self.assertEqual(self.scan_events()[-1]['event'], 'resume_camera_readback')

    def test_resume_event_logging_failure_releases_busy_state_before_worker_start(self):
        g = self.gui
        g._scan_params = (2, 1, 40, 40, 'X_first')
        g._run_auto_capture = False
        g.var_nx = Value('2')
        g.var_ny = Value('1')
        original = self.history.record
        def fail_resume(event, **fields):
            if event == 'resume':
                raise OSError('resume history write failed')
            return original(event, **fields)
        self.history.record = fail_resume
        g.resume_auto()
        self.assert_failed_resume_is_idle_and_no_operation_dispatched()
        self.assertEqual(self.scan_events()[-1]['event'], 'resume')

    def test_manual_result_history_failure_offers_recovery_on_main_thread(self):
        g = self.gui
        original = self.history.record
        def fail_result(event, **fields):
            if event == 'manual_move_completed':
                raise OSError('disk full after manual movement')
            return original(event, **fields)
        self.history.record = fail_result
        g.manual_move(1, 0)
        self.wait_manual()
        self.assertEqual(g.stage.moves, [(40, 0)])
        self.assertFalse(g.auto_running)
        self.assertIsNone(g._worker)
        self.assertEqual(g.btn_auto.options['state'], 'disabled')
        self.assertEqual(g.btn_resume.options['state'], 'disabled')
        self.assertEqual(len(self.errors), 1)
        self.assertIn('disk full after manual movement', self.errors[0][1])

    def test_recovery_is_shown_once_and_history_button_can_reopen_it(self):
        g = self.gui
        g._history_error = 'disk full'
        viewer_calls = []
        self.ns['show_history'] = lambda *args: viewer_calls.append(args)
        g._show_history_recovery()
        g._show_history_recovery()
        self.assertEqual(len(self.errors), 1)
        g.on_show_history()
        self.assertEqual(len(self.errors), 2)
        self.assertEqual(len(viewer_calls), 1)
        self.assertEqual(g.stage.moves, [])
        self.assertEqual(g.cam.reads, [])
        self.assertEqual(g.var_history_status.value, '历史写入失败·恢复步骤')

    def test_recovery_waits_until_worker_and_device_connection_are_idle(self):
        g = self.gui
        g._history_error = 'cannot write'
        g._worker = SimpleNamespace(is_alive=lambda: True)
        g._show_history_recovery()
        self.assertFalse(self.errors)
        self.assertEqual(len(self.gui_callbacks), 1)
        g._worker = None
        callback, args = self.gui_callbacks.pop()
        callback(*args)
        self.assertEqual(len(self.errors), 1)

    def test_worker_recovery_request_never_opens_dialog_in_worker(self):
        g = self.gui
        g._history_error = 'cannot write'
        thread = threading.Thread(target=g._show_history_recovery)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.errors)
        self.assertEqual(len(self.gui_callbacks), 1)
        callback, args = self.gui_callbacks.pop()
        callback(*args)
        self.assertEqual(len(self.errors), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
