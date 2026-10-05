"""Offline camera regressions; no SDK import, dependencies, GUI or hardware.

The DirectShow fake models its one pending sample: grab() does not consume it,
whereas read() consumes it and allows a sample at the current stage position.
This tests the actual flush contract without treating host times as exposure
timestamps or claiming to validate the physical capture pipeline.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest


SOURCE = Path(__file__).with_name('Camera_v2.py')


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


def load_camera(clock):
    """Execute the production Camera class without importing hardware modules."""
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Camera')
    ns = dict(threading=threading, time=clock, _sdk_name='opencv')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns['Camera']


class PendingDirectShow:
    def __init__(self, clock, position=0, read_seconds=0.01):
        self.clock = clock
        self.position = position
        self.pending_position = position
        self.read_seconds = read_seconds
        self.read_calls = 0
        self.grab_calls = 0
        self.fail_at = None
        self.empty_at = None
        self.raise_at = None

    def move_to(self, position):
        self.position = position
        # DirectShow retains its previous sample until read/retrieve consumes it.

    def grab(self):
        self.grab_calls += 1
        return True

    def read(self):
        self.read_calls += 1
        self.clock.now += self.read_seconds
        if self.read_calls == self.raise_at:
            raise OSError('driver read failed')
        if self.read_calls == self.fail_at:
            return False, None
        if self.read_calls == self.empty_at:
            return True, SimpleNamespace(size=0)
        position = self.pending_position
        self.pending_position = self.position
        return True, SimpleNamespace(size=3, position=position)


class CameraFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.camera = load_camera(self.clock)(backend='opencv')
        self.capture = PendingDirectShow(self.clock)
        self.camera._cv_cap = self.capture

    def test_fake_reproduces_one_position_lag_with_grab_only_flush(self):
        self.capture.move_to(1)
        for _ in range(10):
            self.capture.grab()
        ok, frame = self.capture.read()
        self.assertTrue(ok)
        self.assertEqual(frame.position, 0)

    def test_consuming_flush_returns_correct_position_through_three_by_three(self):
        positions = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1),
                     (1, 0), (2, 0), (2, 1), (2, 2)]
        received = []
        for position in positions:
            self.capture.move_to(position)
            frame = self.camera.grab_frame(timeout=5.0, flush=10)
            received.append(frame.position)
        self.assertEqual(received, positions)
        self.assertEqual(self.capture.read_calls, 9 * 11)
        self.assertEqual(self.capture.grab_calls, 0)

    def test_one_consuming_discard_releases_pending_sample(self):
        self.capture.move_to(1)
        frame = self.camera.grab_frame(flush=1)
        self.assertEqual(frame.position, 1)
        self.assertEqual(self.capture.read_calls, 2)

    def test_zero_flush_preview_performs_only_one_read(self):
        self.capture.move_to(1)
        frame = self.camera.grab_frame(timeout=0.05, flush=0)
        self.assertEqual(self.capture.read_calls, 1)
        # Zero flush intentionally permits a pending sample for fast preview.
        self.assertEqual(frame.position, 0)
        self.assertEqual(self.camera.last_capture_metadata['discarded_frames'], 0)

    def test_failed_discard_aborts_without_returning_old_or_final_frame(self):
        self.camera.grab_frame(flush=0)
        self.capture.fail_at = 3
        self.assertIsNone(self.camera.grab_frame(flush=10))
        self.assertEqual(self.capture.read_calls, 3)
        self.assertIsNone(self.camera.last_capture_metadata)

    def test_failed_final_read_cannot_fall_back_to_last_discard(self):
        self.capture.fail_at = 3
        self.assertIsNone(self.camera.grab_frame(flush=2))
        self.assertEqual(self.capture.read_calls, 3)
        self.assertIsNone(self.camera.last_capture_metadata)

    def test_empty_successful_read_is_failure(self):
        self.capture.empty_at = 1
        self.assertIsNone(self.camera.grab_frame(flush=2))
        self.assertEqual(self.capture.read_calls, 1)
        self.assertIsNone(self.camera.last_capture_metadata)

    def test_exception_propagates_without_previous_capture_metadata(self):
        self.camera.grab_frame(flush=0)
        self.capture.raise_at = 2
        with self.assertRaisesRegex(OSError, 'driver read failed'):
            self.camera.grab_frame(flush=1)
        self.assertIsNone(self.camera.last_capture_metadata)

    def test_total_deadline_applies_to_discard_sequence(self):
        self.capture.read_seconds = 0.25
        self.assertIsNone(self.camera.grab_frame(timeout=0.5, flush=10))
        self.assertEqual(self.capture.read_calls, 2)
        self.assertIsNone(self.camera.last_capture_metadata)

    def test_late_final_read_is_rejected_even_when_driver_blocks_past_deadline(self):
        self.capture.read_seconds = 1.0
        self.assertIsNone(self.camera.grab_frame(timeout=0.1, flush=0))
        self.assertEqual(self.capture.read_calls, 1)
        self.assertGreater(self.clock.now, 100.1)
        self.assertIsNone(self.camera.last_capture_metadata)

    def test_zero_or_negative_timeout_does_not_start_driver_read(self):
        for timeout in (0, -1):
            self.assertIsNone(self.camera.grab_frame(timeout=timeout, flush=2))
        self.assertEqual(self.capture.read_calls, 0)

    def test_metadata_counts_consumed_samples_and_labels_host_times(self):
        self.camera.grab_frame(flush=0)
        self.capture.read_seconds = 0.25
        self.camera.grab_frame(timeout=2.0, flush=2)
        meta = self.camera.last_capture_metadata
        self.assertEqual(meta['host_read_sequence'], 4)
        self.assertEqual(meta['discarded_frames'], 2)
        self.assertAlmostEqual(meta['host_read_started_monotonic_s'], 100.51)
        self.assertAlmostEqual(meta['host_read_completed_monotonic_s'], 100.76)
        self.assertEqual(meta['timestamp_source'], 'host_read_not_sensor_exposure')

    def test_missing_capture_clears_previous_metadata(self):
        self.camera.grab_frame(flush=0)
        self.camera._cv_cap = None
        self.assertIsNone(self.camera.grab_frame())
        self.assertIsNone(self.camera.last_capture_metadata)

    def test_sdk_still_waits_for_new_callback_frame_without_opencv_reads(self):
        self.camera._backend = 'amcam'
        self.camera.hcam = object()
        next_frame = object()
        camera = self.camera
        calls = []

        class CallbackCondition:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def wait_for(self, predicate, timeout):
                calls.append(timeout)
                if predicate():
                    raise AssertionError('must wait for a new SDK callback')
                camera._latest_frame = next_frame
                camera._frame_seq += 1
                return predicate()

        self.camera._cond = CallbackCondition()
        self.assertIs(self.camera.grab_frame(timeout=1.5, flush=10), next_frame)
        self.assertEqual(calls, [1.5])
        self.assertEqual(self.capture.read_calls, 0)
        self.assertIsNone(self.camera.last_capture_metadata)


if __name__ == '__main__':
    unittest.main()
