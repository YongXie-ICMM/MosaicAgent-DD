"""Offline preview lifecycle checks: fake camera only; no Tk/SDK/hardware."""
import queue
import threading
import time
from types import SimpleNamespace
import unittest

from scan_preview import PreviewPump
from scan_review import ReviewGate
from test_scan_review import load_logic, Widget


class ControlledCamera:
    """A deterministic fake native read which only finishes when released."""
    def __init__(self):
        self.entered = queue.Queue()
        self.replies = queue.Queue()
        self.calls = []

    def grab_frame(self, **kwargs):
        self.calls.append((kwargs, threading.get_ident()))
        self.entered.put(len(self.calls))
        reply = self.replies.get(timeout=2)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def wait_read(self):
        return self.entered.get(timeout=1)


class PreviewPumpTests(unittest.TestCase):
    def setUp(self):
        self.camera_lock = threading.Lock()
        self.pump = PreviewPump(self.camera_lock, interval_s=0.001)
        self.cameras = []

    def camera(self):
        camera = ControlledCamera()
        self.cameras.append(camera)
        return camera

    def tearDown(self):
        self.pump.stop()
        for camera in self.cameras:
            camera.replies.put(None)
        if self.pump._thread is not None:
            self.pump._thread.join(timeout=1)
            self.assertFalse(self.pump._thread.is_alive(), 'fake reader must exit')

    def test_repeated_start_uses_one_worker_and_never_reads_on_calling_thread(self):
        camera = self.camera()
        self.assertTrue(self.pump.start(camera))
        camera.wait_read()
        first_worker = self.pump._thread
        for _ in range(5):
            self.assertTrue(self.pump.start(camera))
            self.assertIs(self.pump._thread, first_worker)
        self.assertEqual(len(camera.calls), 1)
        arguments, worker_id = camera.calls[0]
        self.assertEqual(arguments, {'timeout': 0.5, 'flush': 0})
        self.assertNotEqual(worker_id, threading.get_ident())

    def test_stop_and_poll_return_while_native_read_is_still_blocked(self):
        camera = self.camera()
        self.pump.start(camera)
        camera.wait_read()
        start = time.monotonic()
        self.pump.stop()
        self.assertIsNone(self.pump.poll())
        self.assertLess(time.monotonic() - start, 0.2)
        self.assertTrue(self.pump._thread.is_alive())
        camera.replies.put('stale frame')
        self.pump._thread.join(timeout=1)
        self.assertIsNone(self.pump.poll(), 'late frame cannot reappear after stop')

    def test_restart_waits_for_old_native_read_before_opening_new_reader(self):
        first, second = self.camera(), self.camera()
        self.pump.start(first)
        first.wait_read()
        old_thread = self.pump._thread
        self.pump.stop()
        self.assertFalse(self.pump.start(second))
        self.assertIs(self.pump._thread, old_thread)
        self.assertEqual(second.calls, [])
        first.replies.put('old frame')
        old_thread.join(timeout=1)
        self.assertTrue(self.pump.start(second))
        second.wait_read()
        self.assertIsNone(self.pump.poll())
        second.replies.put('new frame')
        second.wait_read()
        self.assertEqual(self.pump.poll(), ('frame', 'new frame'))

    def test_camera_switch_invalidates_old_frame_even_without_explicit_stop(self):
        first, second = self.camera(), self.camera()
        self.pump.start(first)
        first.wait_read()
        old_thread = self.pump._thread
        self.assertFalse(self.pump.start(second))
        first.replies.put('wrong camera')
        old_thread.join(timeout=1)
        self.assertIsNone(self.pump.poll())
        self.assertTrue(self.pump.start(second))
        second.wait_read()
        self.assertNotEqual(first.calls[0][1], threading.get_ident())

    def test_latest_mailbox_replaces_older_frames_and_poll_consumes_once(self):
        camera = self.camera()
        self.pump.start(camera)
        camera.wait_read()
        camera.replies.put('frame 1')
        camera.wait_read()
        camera.replies.put('frame 2')
        camera.wait_read()
        self.assertEqual(self.pump.poll(), ('frame', 'frame 2'))
        self.assertIsNone(self.pump.poll())

    def test_stop_clears_already_published_frame_and_inflight_replacement(self):
        camera = self.camera()
        self.pump.start(camera)
        camera.wait_read()
        camera.replies.put('queued frame')
        camera.wait_read()
        self.pump.stop()
        self.assertIsNone(self.pump.poll())
        camera.replies.put('late replacement')
        self.pump._thread.join(timeout=1)
        self.assertIsNone(self.pump.poll())

    def test_busy_capture_lock_is_skipped_and_stop_does_not_wait_for_it(self):
        camera = self.camera()
        self.camera_lock.acquire()
        try:
            self.pump.start(camera)
            with self.assertRaises(queue.Empty):
                camera.entered.get(timeout=0.04)
            self.assertIsNone(self.pump.poll())
            self.pump.stop()
            self.pump._thread.join(timeout=1)
            self.assertFalse(self.pump._thread.is_alive())
            self.assertEqual(camera.calls, [])
        finally:
            self.camera_lock.release()

    def test_error_is_delivered_before_retry_and_camera_lock_is_released(self):
        camera = self.camera()
        self.pump.start(camera)
        camera.wait_read()
        camera.replies.put(RuntimeError('camera disconnected'))
        worker = self.pump._thread
        worker.join(timeout=1)
        self.assertFalse(self.pump.start(camera), 'do not swallow failure on timer restart')
        self.assertIs(self.pump._thread, worker)
        self.assertEqual(self.pump.poll(), ('error', 'camera disconnected'))
        self.assertIsNone(self.pump.poll())
        self.assertTrue(self.camera_lock.acquire(blocking=False))
        self.camera_lock.release()
        self.assertTrue(self.pump.start(camera))
        camera.wait_read()


class FakePump:
    def __init__(self):
        self.starts = []
        self.stops = 0
        self.polls = 0
        self.results = []

    def start(self, camera):
        self.starts.append(camera)
        return True

    def stop(self):
        self.stops += 1

    def poll(self):
        self.polls += 1
        return self.results.pop(0) if self.results else None


class IntegratedPreviewTests(unittest.TestCase):
    """Exercise real GUI methods with a deterministic, windowless Tk timer."""
    def setUp(self):
        self.ns = load_logic()
        self.ns['_HAS_CV2'] = True
        self.now = 100.0
        self.ns['time'] = SimpleNamespace(monotonic=lambda: self.now)
        self.gui = object.__new__(self.ns['MosaicNavigatorGUI'])
        g = self.gui
        g.auto_running = False
        g._review_active = False
        g._preview_running = False
        g._preview_generation = 0
        g._preview_after_id = None
        g._preview_pump = FakePump()
        g._live_last_update = None
        g.connected_camera = True
        g.cam = object()
        g._cancel_event = threading.Event()
        g._review_gate = ReviewGate()
        g._review_token = None
        for name in ('var_live_status', 'var_status', 'var_review', '_live_panel',
                     '_reference_label', '_candidate_label', 'btn_accept',
                     'btn_retake', 'btn_review_stop'):
            setattr(g, name, Widget())
        g._reference_label.image_source = 'accepted image'
        g._candidate_label.image_source = 'candidate image'
        self.displayed = []

        def display(label, frame):
            self.displayed.append((label, frame))
            label.image_source = frame

        g._display_frame = display
        self.pending = {}
        self.next_id = 0

        def after(delay, callback, *args):
            self.next_id += 1
            timer = 'after-%d' % self.next_id
            self.pending[timer] = (callback, args)
            return timer

        g.after = after
        g.after_cancel = lambda timer: self.pending.pop(timer, None)

    def tick(self):
        timer = self.gui._preview_after_id
        callback, args = self.pending.pop(timer)
        callback(*args)

    def test_repeated_start_creates_one_timer_loop(self):
        g = self.gui
        g._start_preview()
        first = g._preview_after_id
        g._start_preview()
        self.assertEqual(g._preview_after_id, first)
        self.assertEqual(len(self.pending), 1)
        self.assertEqual(len(g._preview_pump.starts), 1)
        self.tick()
        self.assertEqual(len(self.pending), 1)
        self.assertNotEqual(g._preview_after_id, first)

    def test_canceled_callback_cannot_revive_or_cancel_new_preview(self):
        g = self.gui
        g._start_preview()
        stale_callback, stale_args = self.pending[g._preview_after_id]
        g._close_preview()
        self.assertEqual(self.pending, {})
        g._start_preview()
        current = g._preview_after_id
        stale_callback(*stale_args)
        self.assertEqual(g._preview_after_id, current)
        self.assertEqual(list(self.pending), [current])
        self.assertEqual(g._preview_pump.polls, 0)
        self.assertTrue(g._preview_running)

    def test_preview_cannot_start_during_movement_or_capture(self):
        g = self.gui
        g.auto_running = True
        g._review_active = False
        g._start_preview()
        self.assertFalse(g._preview_running)
        self.assertEqual(g._preview_pump.starts, [])
        self.assertEqual(self.pending, {})

    def test_existing_preview_stops_if_acquisition_begins_before_tick(self):
        g = self.gui
        g._start_preview()
        g.auto_running = True
        self.tick()
        self.assertFalse(g._preview_running)
        self.assertEqual(g._preview_pump.polls, 0)
        self.assertEqual(g._preview_pump.stops, 1)
        self.assertEqual(self.pending, {})

    def test_live_frame_changes_only_live_panel_during_candidate_review(self):
        g = self.gui
        g.auto_running = True
        g._review_active = True
        g._preview_pump.results = [('frame', 'refocused live image')]
        g._start_preview()
        self.tick()
        self.assertEqual(g._live_panel.image_source, 'refocused live image')
        self.assertEqual(g._candidate_label.image_source, 'candidate image')
        self.assertEqual(g._reference_label.image_source, 'accepted image')
        self.assertEqual(self.displayed, [(g._live_panel, 'refocused live image')])

    def test_failure_stops_timer_and_explicitly_marks_retained_frame(self):
        g = self.gui
        g._preview_pump.results = [('error', 'unplugged camera')]
        g._start_preview()
        self.tick()
        self.assertFalse(g._preview_running)
        self.assertEqual(self.pending, {})
        self.assertIn('失败', g.var_live_status.value)
        self.assertIn('上一帧', g.var_live_status.value)
        self.assertEqual(g._candidate_label.image_source, 'candidate image')

    def test_retained_frame_loses_live_status_until_new_frame_arrives(self):
        g = self.gui
        g._preview_pump.results = [('frame', 'frame 1')]
        g._start_preview()
        self.tick()
        self.assertIn('实时', g.var_live_status.value)
        self.now = 103.5
        self.tick()
        self.assertIn('上次更新', g.var_live_status.value)
        self.assertEqual(g._live_panel.image_source, 'frame 1')
        self.now = 104.0
        g._preview_pump.results = [('frame', 'frame 2')]
        self.tick()
        self.assertIn('实时', g.var_live_status.value)
        self.assertEqual(g._live_panel.image_source, 'frame 2')

    def test_disconnect_before_scheduled_tick_stops_without_polling(self):
        g = self.gui
        g._start_preview()
        g.connected_camera = False
        self.tick()
        self.assertEqual(g._preview_pump.polls, 0)
        self.assertEqual(g._preview_pump.stops, 1)
        self.assertFalse(g._preview_running)
        self.assertEqual(self.pending, {})

    def test_show_review_starts_live_view_and_hide_disables_review_actions(self):
        g = self.gui
        self.ns['cv2'] = SimpleNamespace(imread=lambda path: 'disk:' + path)
        g.auto_running = True
        token = g._review_gate.begin()
        g._show_review(token, 'candidate.png', 'accepted.png', 'Check image', False, False)
        self.assertTrue(g._preview_running)
        self.assertEqual(g._candidate_label.image_source, 'disk:candidate.png')
        self.assertEqual(g._reference_label.image_source, 'disk:accepted.png')
        self.assertEqual(g.btn_accept.options['state'], 'normal')
        g._hide_review(token)
        self.assertFalse(g._preview_running)
        self.assertFalse(g._review_active)
        self.assertEqual(self.pending, {})
        for button in (g.btn_accept, g.btn_retake, g.btn_review_stop):
            self.assertEqual(button.options['state'], 'disabled')
        self.assertEqual(g._candidate_label.image_source, 'disk:candidate.png')


if __name__ == '__main__':
    unittest.main(verbosity=2)
