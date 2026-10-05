"""Windowless fault-injection tests for the live scan recovery coordinator."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np


class InlineThread:
    """Run the hardware-free test worker synchronously; no Tk event loop."""
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args
    def start(self):
        self.target(*self.args)
    def is_alive(self):
        return False


class GUIRecoveryTests(unittest.TestCase):
    def setUp(self):
        # Reuse setup only, without inheriting and rerunning its test cases.
        from test_history_integration import HistoryIntegrationTests
        HistoryIntegrationTests.setUp(self)
        g = self.gui
        self.ns['CAMERA_RESOLUTION'] = (4, 4)  # Explicit miniature fixture contract.
        self.ns['cv2'] = cv2
        self.ns['_HAS_CV2'] = True
        def save(frame, path, **kwargs):
            pixels = np.zeros((4, 4, 3), dtype=np.uint8)
            pixels[:, :, 0] = len(g.cam.reads) * 40
            ok, encoded = cv2.imencode('.png', pixels)
            self.assertTrue(ok)
            Path(path).write_bytes(encoded.tobytes())
        self.ns['Camera'].save_image = save
        self.camera_info = {'backend': 'opencv', 'camera_index': 0,
                            'actual_resolution': [4, 4], 'readback': {'exposure': -3}}
        self.ns['describe_camera'] = lambda camera: copy.deepcopy(self.camera_info)
        g._verify_acquisition_mode = lambda *a, **kw: copy.deepcopy(self.camera_info)
        g._session_camera_signature = g._camera_signature(self.camera_info)
        g._session_recovery_readback = copy.deepcopy(self.camera_info['readback'])
        g._device_epoch = 0
        g._log_resume_plan = (2, 1, 40, 40, 'X_first')
        g._scan_params = g._log_resume_plan
        g._log_resume_devices = (g.stage, g.cam, 0)
        g._log_recovery_cancelled = False
        g._log_recovery_busy = False
        g._log_recovery_review = False
        g._log_pause_position = (0, 0)
        g._point_checkpoint = None
        self.reviews = []
        def review(*args, **kwargs):
            self.reviews.append((args, kwargs))
            return 'accept'
        g._request_review = review
        self.preview_starts = []
        g._start_preview = lambda: self.preview_starts.append(True)
        self.questions = []
        self.ns['messagebox'].askyesno = lambda *args, **kwargs: self.questions.append(args) or True

    def scan_events(self):
        return self.gui._session_log.events()

    def fail_at_event(self, event_name):
        original = self.history.record
        def fail(event, **fields):
            if event == event_name:
                raise OSError('injected ' + event_name + ' storage failure')
            return original(event, **fields)
        return mock.patch.object(self.history, 'record', side_effect=fail)

    def run_first_scan(self):
        self.gui._auto_scan_worker(2, 1, 40, 40, 'X_first')

    def dispatch(self, result):
        self.gui.auto_running = True
        self.gui._log_recovery_busy = True
        with mock.patch.object(self.ns['threading'], 'Thread', InlineThread):
            self.gui._dispatch_log_resume(result)

    def test_post_promotion_logging_failures_reuse_exact_image_and_count_once(self):
        for failure in ('candidate_accepted', 'capture_success'):
            with self.subTest(failure=failure):
                # Each phase gets an independent live session and camera.
                if failure != 'candidate_accepted':
                    self.setUp()
                g = self.gui
                with self.fail_at_event(failure):
                    self.run_first_scan()
                self.assertIsNotNone(g._worker_error)
                original = (self.scan / 'mosaic_r0_c0.png').read_bytes()
                self.assertEqual(g._scan_done_count, 0)
                g._finalize_auto()
                self.assertEqual(g.btn_resume.options['text'], '检查并继续')
                result = g._prepare_log_resume()
                self.assertEqual(result['action'], 'reuse_saved_image')
                self.assertEqual(g.cam.reads, [(0, 0)])
                self.assertEqual(g.stage.moves, [])
                # A second check has no counting or acquisition side effects.
                self.assertEqual(g._prepare_log_resume()['reconciliation_event_id'],
                                 result['reconciliation_event_id'])
                self.dispatch(result)
                self.assertEqual(g._scan_done_count, 2)
                self.assertEqual(g._photo_count, 2)
                self.assertEqual(g._human_review_count, 2)
                self.assertEqual(len(g._sharpness_records), 2)
                self.assertEqual(g.cam.reads, [(0, 0), (40, 0)])
                self.assertEqual(g.stage.moves, [(40, 0)])
                self.assertEqual((self.scan / 'mosaic_r0_c0.png').read_bytes(), original)
                self.assertEqual(len([e for e in self.scan_events()
                                      if e['event'] == 'recovery_point_reconciled']), 1)

    def test_unreviewed_candidate_failure_retries_same_position_and_preserves_candidate(self):
        g = self.gui
        with self.fail_at_event('candidate_saved'):
            self.run_first_scan()
        original_candidate = self.scan / 'review_candidates' / 'r0_c0_attempt0001.png'
        original = original_candidate.read_bytes()
        self.assertEqual(self.reviews, [])
        result = g._prepare_log_resume()
        self.assertEqual(result['action'], 'retry_current')
        self.dispatch(result)
        self.assertEqual(g.cam.reads, [(0, 0), (0, 0), (40, 0)])
        self.assertEqual(g.stage.moves, [(40, 0)])
        self.assertEqual(g._photo_count, 2)
        self.assertEqual(g._scan_done_count, 2)
        self.assertEqual(original_candidate.read_bytes(), original)
        self.assertEqual(len(self.reviews), 2)

    def test_completed_movement_with_failed_log_is_not_dispatched_twice(self):
        g = self.gui
        with self.fail_at_event('move_completed'):
            self.run_first_scan()
        self.assertEqual(g.stage.moves, [(40, 0)])
        self.assertEqual(g.cam.reads, [(0, 0)])
        result = g._prepare_log_resume()
        self.assertEqual(result['action'], 'retry_current')
        self.dispatch(result)
        self.assertEqual(g.stage.moves, [(40, 0)])
        self.assertEqual(g.cam.reads, [(0, 0), (40, 0)])
        self.assertEqual(g._photo_count, 2)

    def test_acceptance_before_promotion_failure_requires_fresh_review(self):
        for failure in ('human_review', 'candidate_promotion_requested'):
            with self.subTest(failure=failure):
                if failure != 'human_review':
                    self.setUp()
                g = self.gui
                with self.fail_at_event(failure):
                    self.run_first_scan()
                self.assertFalse((self.scan / 'mosaic_r0_c0.png').exists())
                self.assertEqual(len(self.reviews), 1)
                result = g._prepare_log_resume()
                self.assertEqual(result['action'], 'retry_current')
                self.dispatch(result)
                self.assertEqual(g.cam.reads, [(0, 0), (0, 0), (40, 0)])
                self.assertEqual(len(self.reviews), 3)
                self.assertEqual(g._human_review_count, 2)
                self.assertEqual(g._scan_done_count, 2)


    def test_final_summary_failure_recovers_without_more_capture_or_movement(self):
        g = self.gui
        self.run_first_scan()
        original_moves, original_reads = list(g.stage.moves), list(g.cam.reads)
        with self.fail_at_event('session_finished'):
            g._finalize_auto()
        self.assertTrue(g._history_error)
        result = g._prepare_log_resume()
        self.dispatch(result)
        self.assertEqual(g.stage.moves, original_moves)
        self.assertEqual(g.cam.reads, original_reads)
        self.assertEqual(g._photo_count, 2)
        self.assertEqual(g._scan_done_count, 2)
        self.assertEqual(g._session_log.summary['status'], 'completed')
        self.assertIsNone(g._history_error)

    def test_changed_device_epoch_blocks_before_log_repair(self):
        g = self.gui
        g._device_epoch += 1
        with mock.patch.object(g._session_log, 'recover_history') as recover:
            with self.assertRaisesRegex(RuntimeError, '重连'):
                g._prepare_log_resume()
            recover.assert_not_called()
        self.assertEqual(g.stage.moves, [])
        self.assertEqual(g.cam.reads, [])

    def test_external_counter_change_blocks_before_log_repair(self):
        g = self.gui
        g.stage.controller_x += 1
        with mock.patch.object(g._session_log, 'recover_history') as recover:
            with self.assertRaisesRegex(RuntimeError, 'counter changed'):
                g._prepare_log_resume()
            recover.assert_not_called()
        self.assertFalse(g.stage.position_trusted)
        self.assertEqual(g.stage.moves, [])

    def test_changed_camera_identity_or_settings_blocks(self):
        for field, value in (('actual_resolution', [8, 8]), ('readback', {'exposure': -4})):
            with self.subTest(field=field):
                original = copy.deepcopy(self.camera_info)
                self.camera_info[field] = value
                with mock.patch.object(self.gui._session_log, 'recover_history') as recover:
                    with self.assertRaises(RuntimeError):
                        self.gui._prepare_log_resume()
                    recover.assert_not_called()
                self.camera_info = original

    def test_disconnected_stage_or_camera_blocks_before_log_repair(self):
        for attribute_owner, attribute in ((self.gui.stage, 'connected'),
                                            (self.gui, 'connected_camera')):
            with self.subTest(attribute=attribute):
                setattr(attribute_owner, attribute, False)
                with mock.patch.object(self.gui._session_log, 'recover_history') as recover:
                    with self.assertRaises(RuntimeError):
                        self.gui._prepare_log_resume()
                    recover.assert_not_called()
                setattr(attribute_owner, attribute, True)

    def test_emergency_stop_invalidates_recovery_and_dispatch(self):
        g = self.gui
        stopped = []
        g.stage.emergency_stop = lambda **kwargs: stopped.append(True)
        g.on_emergency_stop()
        self.assertTrue(stopped)
        self.assertFalse(g._can_check_log_resume())
        g._dispatch_log_resume({'action': 'retry_current'})
        self.assertEqual(g.stage.moves, [])
        self.assertEqual(g.cam.reads, [])
        self.assertFalse(g.auto_running)

    def test_repeated_click_during_check_dispatches_no_second_worker(self):
        g = self.gui
        g._log_recovery_busy = True
        with mock.patch.object(self.ns['threading'], 'Thread') as thread:
            g.on_check_and_resume()
            g.on_check_and_resume()
            thread.assert_not_called()
        self.assertEqual(g.stage.moves, [])
        self.assertEqual(g.cam.reads, [])

    def test_declined_confirmation_leaves_motion_stopped_and_recovery_available(self):
        g = self.gui
        g.auto_running = True
        g._log_recovery_busy = True
        self.ns['messagebox'].askyesno = lambda *args, **kwargs: False
        with mock.patch.object(self.ns['threading'], 'Thread') as thread:
            g._log_resume_prepared({'action': 'retry_current'}, None)
            thread.assert_not_called()
        self.assertFalse(g.auto_running)
        self.assertFalse(g._log_recovery_busy)
        self.assertTrue(g._can_check_log_resume())
        self.assertEqual(g.btn_resume.options['state'], 'normal')
        self.assertEqual(g.stage.moves, [])
        self.assertEqual(g.cam.reads, [])

    def test_storage_still_failing_never_dispatches_scan(self):
        g = self.gui
        with mock.patch.object(g._session_log, 'recover_history',
                               side_effect=OSError('disk still full')):
            with mock.patch.object(self.ns['threading'], 'Thread', InlineThread):
                g.on_check_and_resume()
        self.assertEqual(g.stage.moves, [])
        self.assertEqual(g.cam.reads, [])
        self.assertFalse(g.auto_running)
        self.assertTrue(g._history_error)
        self.assertEqual(g.btn_auto.options['state'], 'disabled')

    def test_position_change_during_confirmation_is_rechecked_before_authorization(self):
        g = self.gui
        def confirm(*args, **kwargs):
            g.stage.controller_x += 1
            return True
        self.ns['messagebox'].askyesno = confirm
        g.auto_running = True
        g._log_recovery_busy = True
        with mock.patch.object(self.ns['threading'], 'Thread', InlineThread):
            g._log_resume_prepared({'action': 'retry_current'}, None)
        self.assertEqual(g.stage.moves, [])
        self.assertEqual(g.cam.reads, [])
        self.assertFalse(g.auto_running)
        self.assertIsNone(g._session_log.find_event('log_resume_authorized'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
