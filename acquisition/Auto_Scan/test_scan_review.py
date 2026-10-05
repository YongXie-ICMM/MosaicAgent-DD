"""Offline regressions for candidate review: no SDK, Tk window or hardware."""
import ast
import base64
import ctypes
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

from scan_review import ReviewGate


SOURCE = Path(__file__).with_name('03Auto_Snake_Scan_Camera_v3.py')


def load_logic():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    ns = dict(os=os, Path=Path, ctypes=ctypes, time=time, threading=threading,
              base64=base64, ReviewGate=ReviewGate, __file__=str(SOURCE),
              tk=SimpleNamespace(Tk=object, PhotoImage=lambda **kw: object()))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    ns[target.id] = value
    names = {'_handle_value', '_is_invalid_handle', '_interruptible_sleep',
             'capture_and_save', 'safe_int', 'safe_float', 'MosaicNavigatorGUI'}
    nodes = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    ns['SETTLE_TIME_S'] = 0
    ns['_HAS_CV2'] = False
    ns['compare_views'] = lambda *args: {'status': 'changed', 'message': 'changed'}
    ns['_sharpness_score'] = lambda frame: 100.0
    ns['_grab_frame_flushed'] = lambda cam, **kw: cam.grab_frame(**kw)
    ns['Camera'] = SimpleNamespace(save_image=lambda frame, path, **kw: Path(path).write_bytes(frame))
    return ns


class Widget:
    def __init__(self):
        self.options = {}
        self.value = None

    def config(self, **kwargs):
        self.options.update(kwargs)

    def set(self, value):
        self.value = value


class Log:
    def __init__(self):
        self.events = []

    def record(self, event, **fields):
        self.events.append((event, fields))


class Stage:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.controller_x = 0
        self.controller_y = 0
        self.position_trusted = True
        self.moves = []

    def verify_position(self):
        if not self.position_trusted:
            raise RuntimeError('stage position untrusted')
        if (self.controller_x, self.controller_y) != (self.x, self.y):
            self.position_trusted = False
            raise RuntimeError('controller counter changed outside the scan')

    def move_relative(self, dx, dy, cancel):
        if cancel.is_set():
            raise RuntimeError('canceled')
        self.moves.append((dx, dy))
        self.x += dx
        self.y += dy
        self.controller_x += dx
        self.controller_y += dy


class Camera:
    backend = 'opencv'
    camera_index = 0

    def __init__(self, stage):
        self.stage = stage
        self.reads = []
        self.last_capture_metadata = None

    def grab_frame(self, **kwargs):
        self.reads.append((self.stage.x, self.stage.y))
        self.last_capture_metadata = {'host_read_sequence': len(self.reads),
                                      'timestamp_source': 'host_read_not_sensor_exposure'}
        return ('RAW_CANDIDATE_%d' % len(self.reads)).encode('ascii')


class ReviewGateTests(unittest.TestCase):
    def test_duplicate_and_stale_token_clicks_cannot_choose_new_request(self):
        gate = ReviewGate()
        old = gate.begin()
        self.assertTrue(gate.choose(old, 'retake'))
        self.assertFalse(gate.choose(old, 'accept'))
        current = gate.begin()
        self.assertFalse(gate.choose(old, 'accept'))
        self.assertTrue(gate.choose(current, 'accept'))
        self.assertEqual(gate.wait(current, threading.Event()), 'accept')

    def test_stale_waiter_cannot_deactivate_new_request(self):
        gate = ReviewGate()
        old = gate.begin()
        current = gate.begin()
        self.assertEqual(gate.wait(old, threading.Event()), 'stop')
        self.assertTrue(gate.choose(current, 'accept'))

    def test_cancel_wakes_waiter_and_prevents_later_acceptance(self):
        gate = ReviewGate()
        token = gate.begin()
        cancel = threading.Event()
        started = threading.Event()
        results = []

        def wait_for_review():
            started.set()
            results.append(gate.wait(token, cancel))

        worker = threading.Thread(target=wait_for_review, daemon=True)
        worker.start()
        self.assertTrue(started.wait(1.0))
        cancel.set()
        worker.join(1.0)
        self.assertFalse(worker.is_alive(), 'cancel must wake review without user input')
        self.assertEqual(results, ['stop'])
        self.assertFalse(gate.choose(token, 'accept'))


class ScanReviewTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_logic()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cls = self.ns['MosaicNavigatorGUI']
        self.gui = object.__new__(cls)
        g = self.gui
        g.stage = Stage()
        g.cam = Camera(g.stage)
        g.connected_camera = True
        g._run_auto_capture = True
        # These legacy tests isolate review/history using non-image byte fixtures.
        # test_acquisition_contract exercises the real new guards with PNGs.
        g._verify_acquisition_mode = lambda *a, **kw: None
        g._check_candidate_dimensions = lambda *a, **kw: [1920, 1080]
        g._run_review_mode = 'step_confirm'
        g._session_folder = str(self.root)
        g._session_log = Log()
        g._camera_lock = threading.Lock()
        g._cancel_event = threading.Event()
        g._capture_attempts = g._capture_failures = g._photo_count = 0
        g._human_review_count = g._scan_done_count = 0
        g._sharpness_records = []
        g._last_accepted_photo = None
        g._scan_origin = (0, 0)
        g._worker_error = None
        g._update_gui_status = lambda *args: None
        g._refresh_pos = lambda: None
        g.after = lambda delay, callback, *args: callback(*args)
        g._review_gate = ReviewGate()
        g._review_token = None
        g._review_active = False
        g._review_block_accept = False
        g._close_preview = lambda: None
        g._start_preview = lambda: None
        for name in ('var_status', 'var_review', '_reference_label', '_candidate_label',
                     'btn_accept', 'btn_retake', 'btn_review_stop'):
            setattr(g, name, Widget())
        self.reviewed = []
        self.questions = []
        self.ns['messagebox'] = SimpleNamespace(
            askyesno=lambda *args: self.questions.append(args) or True)
        self.ns['cv2'] = SimpleNamespace(
            imread=lambda path: SimpleNamespace(shape=(16, 16, 3)),
            resize=lambda frame, size: frame,
            imencode=lambda ext, frame: (True, SimpleNamespace(tobytes=lambda: b'preview')))

    def decisions(self, choices):
        sequence = iter(choices)

        def request(candidate, previous, summary, **kwargs):
            self.reviewed.append({'candidate': Path(candidate), 'bytes': Path(candidate).read_bytes(),
                                  'previous': previous, 'summary': summary,
                                  'position': (self.gui.stage.x, self.gui.stage.y), **kwargs})
            return next(sequence)

        self.gui._request_review = request

    def event_fields(self, name):
        return [fields for event, fields in self.gui._session_log.events if event == name]

    def test_retake_stays_at_same_point_preserves_rejected_candidate_and_adds_no_move(self):
        self.decisions(['retake', 'accept', 'accept'])
        self.gui._auto_scan_worker(2, 1, 143, 76, 'X_first')
        self.assertIsNone(self.gui._worker_error)
        self.assertEqual(self.gui.cam.reads, [(0, 0), (0, 0), (143, 0)])
        self.assertEqual(self.gui.stage.moves, [(143, 0)])
        self.assertEqual(self.gui._scan_done_count, 2)
        self.assertEqual(self.gui._photo_count, 2)
        self.assertEqual(self.reviewed[0]['candidate'].read_bytes(), b'RAW_CANDIDATE_1')
        self.assertEqual((self.root / 'mosaic_r0_c0.png').read_bytes(), b'RAW_CANDIDATE_2')
        self.assertEqual([f['decision'] for f in self.event_fields('human_review')],
                         ['retake', 'accept', 'accept'])

    def test_accept_publishes_exact_reviewed_bytes_without_another_camera_read(self):
        # Fallback sharpness paths return a NumPy scalar; its comparison must
        # become a JSON boolean, never the string "False".
        self.ns['_sharpness_score'] = lambda frame: np.float64(100.0)
        self.decisions(['accept'])
        self.assertTrue(self.gui._take_photo_at_point(1, 1, 0, 0))
        self.assertEqual((self.root / 'mosaic_r0_c0.png').read_bytes(), self.reviewed[0]['bytes'])
        self.assertEqual(len(self.gui.cam.reads), 1)
        self.assertEqual(self.gui._photo_count, 1)
        self.assertEqual(len(self.event_fields('capture_success')), 1)
        self.assertIs(self.event_fields('capture_success')[0]['below_blur_threshold'], False)

    def test_stop_leaves_zero_accepted_points_and_preserves_candidate(self):
        self.decisions(['stop'])
        self.gui._auto_scan_worker(2, 1, 143, 76, 'X_first')
        self.assertEqual(self.gui._scan_done_count, 0)
        self.assertEqual(self.gui._photo_count, 0)
        self.assertEqual(self.gui.stage.moves, [])
        self.assertTrue(self.gui._cancel_event.is_set())
        self.assertFalse(self.gui.stage.position_trusted)
        self.assertEqual(self.reviewed[0]['candidate'].read_bytes(), b'RAW_CANDIDATE_1')
        self.assertFalse((self.root / 'mosaic_r0_c0.png').exists())
        self.assertEqual(self.event_fields('capture_success'), [])

    def test_same_view_in_continuous_mode_requires_review_and_blocks_accept(self):
        self.gui._run_review_mode = 'continuous'
        self.ns['compare_views'] = lambda *args: {'status': 'same_view', 'message': 'no change'}
        self.decisions(['stop'])
        self.assertFalse(self.gui._take_photo_at_point(1, 1, 0, 0))
        self.assertTrue(self.reviewed[0]['block_accept'])
        self.assertEqual(self.gui._photo_count, 0)

    def test_blurry_continuous_capture_requests_human_review(self):
        self.gui._run_review_mode = 'continuous'
        self.ns['_sharpness_score'] = lambda frame: 0.0
        self.decisions(['retake', 'stop'])
        self.assertFalse(self.gui._take_photo_at_point(1, 1, 0, 0))
        self.assertEqual(len(self.reviewed), 2)
        self.assertTrue(all(review['warning'] for review in self.reviewed))
        self.assertEqual(self.gui.cam.reads, [(0, 0), (0, 0)])
        self.assertEqual(self.gui._photo_count, 0)

    def test_cancellation_after_review_cannot_publish_or_increment_count(self):
        def cancel_then_accept(*args, **kwargs):
            self.gui._cancel_event.set()
            return 'accept'

        self.gui._request_review = cancel_then_accept
        self.assertFalse(self.gui._take_photo_at_point(1, 1, 0, 0))
        self.assertEqual(self.gui._photo_count, 0)
        self.assertFalse((self.root / 'mosaic_r0_c0.png').exists())
        self.assertEqual(len(list((self.root / 'review_candidates').glob('*.png'))), 1)

    def test_external_counter_change_during_review_prevents_promotion_and_next_move(self):
        def move_outside_scan_then_accept(candidate, previous, summary, **kwargs):
            self.assertEqual(Path(candidate).read_bytes(), b'RAW_CANDIDATE_1')
            # A controller/joystick change does not update scanner logical x/y.
            self.gui.stage.controller_x += 1
            return 'accept'

        self.gui._request_review = move_outside_scan_then_accept
        self.gui._auto_scan_worker(2, 1, 143, 76, 'X_first')
        self.assertRegex(str(self.gui._worker_error), 'controller counter changed')
        self.assertFalse(self.gui.stage.position_trusted)
        self.assertTrue(self.gui._cancel_event.is_set())
        self.assertEqual(self.gui._scan_done_count, 0)
        self.assertEqual(self.gui._photo_count, 0)
        self.assertEqual(self.gui.stage.moves, [])
        self.assertEqual(self.gui.cam.reads, [(0, 0)])
        self.assertFalse((self.root / 'mosaic_r0_c0.png').exists())
        candidates = list((self.root / 'review_candidates').glob('*.png'))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].read_bytes(), b'RAW_CANDIDATE_1')
        self.assertEqual(self.event_fields('candidate_accepted'), [])
        self.assertEqual(self.event_fields('capture_success'), [])

    def show_review(self, block_accept=False, warning=False):
        token = self.gui._review_gate.begin()
        self.gui._show_review(token, str(self.root / 'candidate.png'), None,
                              'inspect current image', warning, block_accept)
        return token

    def test_same_view_disabled_accept_is_also_guarded_by_handler(self):
        token = self.show_review(block_accept=True)
        self.assertEqual(self.gui.btn_accept.options['state'], 'disabled')
        self.gui._choose_review(token, 'accept')
        self.assertTrue(self.gui._review_gate.choose(token, 'retake'),
                        'blocked accept must leave the request awaiting retake/stop')

    def test_canceled_review_accept_cannot_choose_or_prompt(self):
        token = self.show_review(warning=True)
        self.gui._cancel_event.set()
        self.gui._choose_review(token, 'accept', warning=True)
        self.assertNotEqual(self.gui._review_gate._decision, 'accept')
        self.assertEqual(self.questions, [])

    def test_cancel_during_modal_warning_cannot_choose_accept(self):
        token = self.show_review(warning=True)

        def cancel_during_prompt(*args):
            # Tk message boxes run a nested event loop, which may process Stop.
            self.gui._cancel_event.set()
            return True

        self.ns['messagebox'].askyesno = cancel_during_prompt
        self.gui._choose_review(token, 'accept', warning=True)
        self.assertNotEqual(self.gui._review_gate._decision, 'accept')

    def test_review_render_error_blocks_accept_but_retains_retake_and_stop(self):
        def fail_to_render(frame, size):
            raise RuntimeError('image resize failed')

        self.ns['cv2'].resize = fail_to_render
        token = self.show_review()
        self.assertEqual(self.gui.btn_accept.options['state'], 'disabled')
        self.assertEqual(self.gui.btn_retake.options['state'], 'normal')
        self.assertEqual(self.gui.btn_review_stop.options['state'], 'normal')
        self.gui._choose_review(token, 'retake')
        self.assertEqual(self.gui._review_gate.wait(token, self.gui._cancel_event), 'retake')

    def test_stale_accept_click_cannot_prompt_or_hide_new_request(self):
        old = self.show_review(warning=True)
        current = self.show_review(warning=True)
        self.gui._choose_review(old, 'accept', warning=True)
        self.assertEqual(self.questions, [])
        self.assertTrue(self.gui._review_active)
        self.assertTrue(self.gui._review_gate.choose(current, 'retake'))

    def test_queued_review_after_cancellation_cannot_reactivate_controls(self):
        token = self.gui._review_gate.begin()
        self.gui._cancel_event.set()
        self.gui._show_review(token, str(self.root / 'candidate.png'), None,
                              'canceled review', False, False)
        self.assertFalse(self.gui._review_active)
        self.assertNotEqual(self.gui.btn_accept.options.get('state'), 'normal')


if __name__ == '__main__':
    unittest.main(verbosity=2)
