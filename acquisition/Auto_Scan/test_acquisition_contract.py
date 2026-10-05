"""1080p acquisition gates with synthetic pixels only; no Tk or hardware I/O."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np

from acquisition_contract import (AcquisitionModeError, contract, frame_size,
                                  require_frame_size, validate_saved_image)
from camera_session_history import describe_camera, open_detected_camera


SIZE = (1920, 1080)


def pixels(size=SIZE, value=75):
    return np.full((size[1], size[0], 3), value, dtype=np.uint8)


def save(frame, path, **kwargs):
    ok, image = cv2.imencode('.png', frame)
    if not ok:
        raise ValueError('Fake image encoder failed')
    Path(path).write_bytes(image.tobytes())


class ContractTests(unittest.TestCase):
    def test_dimensions_require_received_pixels_not_a_setting(self):
        self.assertEqual(require_frame_size(pixels(), SIZE, 'test'), [1920, 1080])
        for frame in (None, np.zeros((0, 5)), pixels((640, 480))):
            with self.subTest(actual=frame_size(frame)), self.assertRaises(AcquisitionModeError):
                require_frame_size(frame, SIZE, 'test')

    def test_real_png_and_supplied_bytes_are_checked_without_resizing(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / '照片.png'
            save(pixels(), path)
            raw = path.read_bytes()
            self.assertEqual(validate_saved_image(path, SIZE), list(SIZE))
            save(pixels((640, 480)), path)
            with self.assertRaises(AcquisitionModeError):
                validate_saved_image(path, SIZE)
            self.assertEqual(validate_saved_image(path, SIZE, raw_bytes=raw), list(SIZE))
            path.write_bytes(b'not an image')
            with self.assertRaises(AcquisitionModeError):
                validate_saved_image(path, SIZE)

    def test_contract_does_not_claim_physical_calibration(self):
        data = contract(SIZE, SIZE, phase='pre_scan', verified=True)
        self.assertEqual(data['verification_status'], 'verified_received_frame')
        self.assertEqual(data['calibration_status'], 'unverified_for_current_mode')
        self.assertIs(data['raw_images_resized'], False)
        self.assertEqual(json.loads(json.dumps(data))['expected_image_size'], [1920, 1080])


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.created = []
        outer = self
        class Camera:
            save_image = staticmethod(save)
            frame_size = SIZE
            def __init__(self, camera_index, bits, backend, resolution):
                self.camera_index, self.backend = camera_index, backend
                self._requested_resolution = resolution
                self.closed = False
                outer.created.append(self)
            def open(self): pass
            def start_live(self): pass
            def close(self): self.closed = True
            def get_resolution(self): return (640, 480)
            def grab_frame(self, **kwargs): return pixels(self.frame_size)
        self.Camera = Camera

    def test_success_records_actual_frame_even_when_driver_readback_is_stale(self):
        with TemporaryDirectory() as directory:
            cam, info = open_detected_camera(self.Camera, directory, resolution=SIZE)
            self.assertEqual(cam._requested_resolution, SIZE)
            self.assertEqual(info['actual_resolution'], list(SIZE))
            self.assertEqual(info['resolution_source'], 'received_frame')
            self.assertEqual(info['acquisition_contract']['verification_phase'], 'connection')
            self.assertEqual(info['acquisition_contract']['verification_status'], 'verified_received_frame')

    def test_mismatch_preserves_frame_closes_camera_and_does_not_try_other_camera(self):
        self.Camera.frame_size = (640, 480)
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AcquisitionModeError, 'Diagnostic:'):
                open_detected_camera(self.Camera, directory, resolution=SIZE)
            self.assertEqual(len(self.created), 1)
            self.assertTrue(self.created[0].closed)
            root = Path(directory)
            self.assertFalse((root / 'last_camera.json').exists())
            images = list((root / 'diagnostic_frames').glob('*.png'))
            self.assertEqual(len(images), 1)
            self.assertEqual(validate_saved_image(images[0], (640, 480)), [640, 480])
            event = json.loads((root / 'camera_connections.jsonl').read_text())
            self.assertEqual(event['status'], 'resolution_mismatch')
            self.assertEqual(event['actual_image_size'], [640, 480])

    def test_explicit_next_camera_skips_failed_index_without_silent_fallback(self):
        from test_scan_review import load_logic, Widget
        ns = load_logic()
        cls = ns['MosaicNavigatorGUI']
        gui = object.__new__(cls)
        gui.cam = None
        gui._camera_cycle_excluded = set()
        gui._history_error = None
        gui.var_status = Widget()
        history = []
        gui._record_history_best_effort = lambda event, **fields: history.append((event, fields))
        ns['messagebox'] = SimpleNamespace(showerror=lambda *a: None)
        self.Camera.grab_frame = lambda cam, **kwargs: pixels((640, 480) if cam.camera_index == 0 else SIZE)
        with TemporaryDirectory() as directory:
            for _ in range(2):  # Ordinary Connect does not silently skip camera 0.
                with self.assertRaises(AcquisitionModeError) as caught:
                    open_detected_camera(self.Camera, directory, resolution=SIZE, indices=[0, 1])
                gui._camera_connect_failed(caught.exception)
                self.assertEqual(caught.exception.backend_requested, 'auto')
                self.assertEqual(caught.exception.camera_index, 0)
            self.assertEqual([c.camera_index for c in self.created], [0, 0])
            self.assertEqual(gui._failed_camera_index, 0)
            self.assertEqual(history[-1][1]['camera_index'], 0)
            def connect(exclude_indices=()):
                gui.cam, info = open_detected_camera(self.Camera, directory, resolution=SIZE,
                                                     indices=[0, 1], exclude_indices=exclude_indices)
            gui.on_connect_camera = connect
            gui.on_next_camera()
            self.assertEqual(gui.cam.camera_index, 1)
            self.assertEqual([c.camera_index for c in self.created], [0, 0, 1])

    def test_diagnostic_storage_failure_still_blocks_connection(self):
        self.Camera.frame_size = (640, 480)
        self.Camera.save_image = staticmethod(mock.Mock(side_effect=OSError('disk full')))
        with TemporaryDirectory() as directory, self.assertRaisesRegex(AcquisitionModeError, 'Diagnostic save failed'):
            open_detected_camera(self.Camera, directory, resolution=SIZE)
        self.assertTrue(self.created[0].closed)
        self.assertEqual(len(self.created), 1)


class ScannerGuardTests(unittest.TestCase):
    def setUp(self):
        from test_history_integration import HistoryIntegrationTests
        HistoryIntegrationTests.setUp(self)
        g = self.gui
        # Undo only the legacy fixture mocks: use the actual production guards.
        del g._verify_acquisition_mode
        del g._check_candidate_dimensions
        self.assertEqual(self.ns['CAMERA_RESOLUTION'], SIZE)
        self.ns['describe_camera'] = describe_camera
        self.ns['Camera'].save_image = save
        self.ns['_HAS_CV2'] = True
        self.ns['cv2'] = cv2
        self.ns['ENABLE_BLUR_CHECK'] = False
        self.frame_sizes = iter([SIZE] * 20)
        def grab(**kwargs):
            g.cam.reads.append((g.stage.x, g.stage.y))
            return pixels(next(self.frame_sizes), value=min(len(g.cam.reads) * 15, 255))
        g.cam.grab_frame = grab
        g.cam.get_resolution = lambda: (640, 480)  # stale readback must not pass/fail gate
        g.cam._requested_resolution = SIZE
        g._camera_info = {'backend': 'opencv', 'camera_index': 0, 'actual_resolution': list(SIZE)}
        g._session_camera_signature = g._camera_signature(g._camera_info)
        g._request_review = lambda *a, **kw: 'accept'

    def run_scan(self, count=3, skip=0):
        self.gui._auto_scan_worker(count, 1, 40, 40, 'X_first', skip)

    def events(self):
        return self.gui._session_log.events()

    def test_success_verifies_before_first_capture_and_persists_contract(self):
        self.run_scan()
        g = self.gui
        self.assertIsNone(g._worker_error)
        self.assertEqual(g._photo_count, 3)
        self.assertEqual(g.stage.moves, [(40, 0), (40, 0)])
        events = self.events()
        verified = next(i for i, e in enumerate(events) if e['event'] == 'acquisition_mode_verified')
        capture = next(i for i, e in enumerate(events) if e['event'] == 'capture_attempt')
        self.assertLess(verified, capture)
        data = json.loads((Path(g._session_folder) / 'session.json').read_text())
        self.assertEqual(data['scan_config']['acquisition_contract']['actual_image_size'], list(SIZE))
        self.assertEqual(data['camera']['resolution_source'], 'received_frame')
        self.assertEqual(data['camera']['actual_resolution'], list(SIZE))
        self.assertEqual(len(list(Path(g._session_folder).glob('mosaic_*.png'))), 3)

    def test_pre_scan_mismatch_blocks_movement_and_keeps_diagnostic(self):
        self.frame_sizes = iter([(640, 480)])
        self.run_scan()
        g = self.gui
        self.assertIsInstance(g._worker_error, AcquisitionModeError)
        self.assertEqual(g.stage.moves, [])
        self.assertEqual(g._photo_count, 0)
        self.assertEqual(len(list(Path(g._session_folder).glob('review_candidates/acquisition_pre_scan_*.png'))), 1)
        self.assertEqual(g._session_log.summary['scan_config']['acquisition_contract']['verification_status'], 'mismatch')

    def test_candidate_mode_change_never_reaches_review_or_promotion(self):
        self.frame_sizes = iter([SIZE, (640, 480)])
        self.gui._request_review = mock.Mock(side_effect=AssertionError('wrong image must never reach review'))
        self.run_scan()
        g = self.gui
        self.assertIsInstance(g._worker_error, AcquisitionModeError)
        self.assertEqual(g.stage.moves, [])
        self.assertEqual(g._photo_count, 0)
        self.assertEqual(len(list(Path(g._session_folder).glob('review_candidates/r0_c0_*.png'))), 1)
        self.assertFalse(list(Path(g._session_folder).glob('mosaic_*.png')))
        g._request_review.assert_not_called()

    def test_mid_scan_mode_change_stops_at_current_point(self):
        self.frame_sizes = iter([SIZE, SIZE, (640, 480)])
        self.run_scan()
        self.assertEqual(self.gui._scan_done_count, 1)
        self.assertEqual(self.gui._photo_count, 1)
        self.assertEqual(self.gui.stage.moves, [(40, 0)])
        self.assertEqual(self.gui._point_checkpoint['phase'], 'before_capture')

    def test_candidate_changed_during_review_is_rechecked_before_promotion(self):
        def change(candidate, *a, **kw):
            save(pixels((640, 480)), candidate)
            return 'accept'
        self.gui._request_review = change
        self.run_scan()
        self.assertIsInstance(self.gui._worker_error, AcquisitionModeError)
        self.assertEqual(self.gui._photo_count, 0)
        self.assertFalse(list(Path(self.gui._session_folder).glob('mosaic_*.png')))

    def test_preflight_log_failure_blocks_all_capture_and_motion(self):
        original = self.gui._session_log.record
        def record(event, **fields):
            if event == 'acquisition_mode_verified':
                raise OSError('disk full')
            return original(event, **fields)
        self.gui._session_log.record = record
        self.run_scan()
        self.assertIsInstance(self.gui._worker_error, OSError)
        self.assertEqual(self.gui.stage.moves, [])
        self.assertEqual(self.gui._photo_count, 0)
        self.assertEqual(len(self.gui.cam.reads), 1)

    def test_mismatch_log_failure_does_not_promote_or_destroy_candidate(self):
        self.frame_sizes = iter([SIZE, (640, 480)])
        original = self.gui._session_log.record
        def record(event, **fields):
            if event == 'acquisition_mode_mismatch':
                raise OSError('disk full')
            return original(event, **fields)
        self.gui._session_log.record = record
        self.run_scan()
        self.assertEqual(self.gui._point_checkpoint['phase'], 'before_capture')
        self.assertFalse(list(Path(self.gui._session_folder).glob('mosaic_*.png')))
        self.assertEqual(len(list(Path(self.gui._session_folder).glob('review_candidates/*.png'))), 1)

    def test_wrong_resolution_promoted_checkpoint_cannot_be_reused_by_log_recovery(self):
        g = self.gui
        path = Path(g._session_folder) / 'mosaic_r0_c0.png'
        save(pixels((640, 480)), path)
        raw = path.read_bytes()
        g._point_checkpoint = {'phase': 'promoted', 'filepath': str(path),
                               'image_sha256': hashlib.sha256(raw).hexdigest(), 'image_bytes': len(raw)}
        # Device safety gate is tested above; this test isolates the saved-image
        # guard on the actual recovery path, after its hash check.
        g._validate_log_resume_devices = lambda: None
        with mock.patch.object(g._session_log, 'recover_history') as recover:
            with self.assertRaises(AcquisitionModeError):
                g._prepare_log_resume()
            recover.assert_not_called()
        self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(g.stage.moves, [])

    def test_resume_gate_uses_received_frame_before_any_move(self):
        self.frame_sizes = iter([(640, 480)])
        self.run_scan(skip=1)
        self.assertEqual(self.gui.stage.moves, [])
        self.assertEqual(self.gui._photo_count, 0)
        self.assertEqual(self.gui._worker_error.phase, 'resume')

    def test_resume_entry_accepts_stale_driver_size_then_checks_real_frame_before_move(self):
        from test_log_resume_gui import InlineThread
        from test_scan_v3_offline import Value
        g = self.gui
        g._scan_params = (2, 1, 40, 40, 'X_first')
        g._scan_done_count = g._photo_count = 1
        g._camera_connecting = False
        g.var_nx, g.var_ny = Value('2'), Value('1')
        warnings = []
        self.ns['messagebox'].showwarning = lambda *a, **kw: warnings.append(a)
        with mock.patch.object(self.ns['threading'], 'Thread', InlineThread):
            g.resume_auto()
        self.assertEqual(warnings, [])
        self.assertIsNone(g._worker_error)
        self.assertEqual(g._scan_done_count, 2)
        self.assertEqual(g.stage.moves, [(40, 0)])
        events = self.events()
        verified = next(i for i, e in enumerate(events) if e['event'] == 'acquisition_mode_verified')
        movement = next(i for i, e in enumerate(events) if e['event'] == 'move_attempt')
        self.assertLess(verified, movement)
        self.assertEqual(events[verified]['acquisition_contract']['verification_phase'], 'resume')

    def test_fresh_frame_gate_rejects_changed_camera_identity_before_resume_move(self):
        self.gui.cam.camera_index = 2
        self.run_scan(skip=1)
        self.assertIn('Camera identity changed', str(self.gui._worker_error))
        self.assertEqual(self.gui.stage.moves, [])
        self.assertEqual(self.gui._photo_count, 0)

    def test_cancelled_preflight_never_reads_camera(self):
        self.gui._cancel_event.set()
        self.run_scan()
        self.assertEqual(self.gui.cam.reads, [])
        self.assertEqual(self.gui.stage.moves, [])


if __name__ == '__main__':
    unittest.main()
