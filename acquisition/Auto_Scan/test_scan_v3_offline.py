"""Pure offline regression tests: AST-load logic; never import camera SDKs or open a GUI."""
import ast
import ctypes
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest

SOURCE = Path(__file__).with_name('03Auto_Snake_Scan_Camera_v3.py')


def load_logic():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    ns = dict(os=os, ctypes=ctypes, time=time, threading=threading,
              tk=SimpleNamespace(Tk=object), __file__=str(SOURCE))
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
             '_write_lockfile', '_remove_lockfile', 'capture_and_save',
             'safe_int', 'safe_float', 'XIMCStage', 'MosaicNavigatorGUI'}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    ns['SETTLE_TIME_S'] = 0
    return ns


class Value:
    def __init__(self, value=None): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class Log:
    def __init__(self): self.events = []; self.finished = None
    def record(self, event, **fields): self.events.append((event, fields))
    def finish(self, status, **fields): self.finished = (status, fields)


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_logic()
        self.gui = self.ns['MosaicNavigatorGUI']

    def test_checkout_sdk_is_discoverable_without_importing_or_opening_it(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        index = next(i for i, n in enumerate(tree.body)
                     if isinstance(n, ast.Assign) and any(
                         isinstance(t, ast.Name) and t.id == '_sdk_parent' for t in n.targets))
        code = compile(ast.Module(body=tree.body[index:index + 2], type_ignores=[]),
                       str(SOURCE), 'exec')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            child = root / 'Auto_Scan'
            child.mkdir()
            paths = [str(child)]
            ns = dict(Path=Path, __file__=str(child / SOURCE.name),
                      sys=SimpleNamespace(path=paths))
            exec(code, ns)
            self.assertEqual(paths, [str(child)])
            (root / 'amcam.py').write_text('raise AssertionError("must not import SDK")')
            exec(code, ns)
            self.assertEqual(paths, [str(child), str(root)])
            exec(code, ns)
            self.assertEqual(paths.count(str(root)), 1)

    def scanner(self):
        stage = self.ns['XIMCStage'](simulate=True)
        stage.connect()
        obj = SimpleNamespace(stage=stage, _run_auto_capture=False, _cancel_event=threading.Event(),
                              _scan_origin=(0, 0), _scan_done_count=0,
                              _worker_error=None, _session_log=Log(),
                              _refresh_pos=lambda: None, after=lambda *a: None)
        obj._build_snake_path = lambda *a: self.gui._build_snake_path(obj, *a)
        obj._take_photo_at_point = lambda *a, **kw: True
        return obj

    def run_scan(self, obj, skip=0):
        self.gui._auto_scan_worker(obj, 2, 2, 143, 76, 'X_first', skip)

    def test_both_snake_orders_cover_grid_once(self):
        for order in ('X_first', 'Y_first'):
            path = self.gui._build_snake_path(None, 3, 3, 143, 76, order)
            self.assertEqual(len(path), 9)
            self.assertEqual(len({(r, c) for _, _, r, c in path}), 9)

    def test_capture_failure_never_counts_or_advances(self):
        obj = self.scanner()
        obj._take_photo_at_point = lambda *a, **kw: False
        self.run_scan(obj)
        self.assertEqual(obj._scan_done_count, 0)
        self.assertEqual((obj.stage.x, obj.stage.y), (0, 0))
        self.assertIsNotNone(obj._worker_error)

    def test_failed_second_capture_resumes_same_position(self):
        obj = self.scanner()
        calls = []
        def capture(i, *args, **kw):
            calls.append((i, obj.stage.x, obj.stage.y))
            return i != 2
        obj._take_photo_at_point = capture
        self.run_scan(obj)
        self.assertEqual(obj._scan_done_count, 1)
        self.assertEqual((obj.stage.x, obj.stage.y), (143, 0))
        obj._cancel_event.clear()
        obj._worker_error = None
        obj._take_photo_at_point = lambda i, *a, **kw: calls.append((i, obj.stage.x, obj.stage.y)) or True
        self.run_scan(obj, skip=1)
        self.assertEqual(calls[:3], [(1, 0, 0), (2, 143, 0), (2, 143, 0)])
        self.assertEqual(obj._scan_done_count, 4)
        self.assertEqual((obj.stage.x, obj.stage.y), (0, 76))

    def test_missing_camera_cannot_silently_dwell(self):
        obj = SimpleNamespace(_run_auto_capture=True, connected_camera=False, cam=None)
        with self.assertRaisesRegex(RuntimeError, 'connected camera'):
            self.gui._take_photo_at_point(obj, 1, 4, 0, 0)

    def test_start_rejects_missing_camera_before_session_or_movement(self):
        warnings = []
        self.ns['messagebox'] = SimpleNamespace(showwarning=lambda *a: warnings.append(a))
        obj = SimpleNamespace(stage=SimpleNamespace(connected=True, position_trusted=True),
                              auto_running=False, _camera_connecting=False,
                              var_auto_capture=Value(True), connected_camera=False, cam=None)
        self.gui.start_auto(obj)
        self.assertEqual(warnings[0][0], 'Camera required')

    def test_motion_timeout_stops_and_invalidates_position(self):
        from _stage_test_fixture import connected_stage
        stage, dll = connected_stage()
        dll.move_behavior = 'never_stops'
        stage.MOVE_DEADLINE_S = 0.005
        stage.POLL_INTERVAL_S = 0.001
        with self.assertRaisesRegex(RuntimeError, 'no photo taken'):
            stage.move_relative(143, 0)
        self.assertFalse(stage.position_trusted)
        self.assertEqual(stage.x, 0)
        self.assertEqual(dll.stops, [11, 22])
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            stage.move_relative(143, 0)

    def test_move_failure_prevents_next_photo(self):
        obj = self.scanner()
        captures = []
        obj._take_photo_at_point = lambda i, *a, **kw: captures.append(i) or True
        def fail(*args): raise RuntimeError('motor fault')
        obj.stage.move_relative = fail
        self.run_scan(obj)
        self.assertEqual(captures, [1])
        self.assertEqual(obj._scan_done_count, 1)
        self.assertEqual(obj._session_log.events[-1][0], 'move_failed')

    def test_existing_lock_is_not_removed_or_process_killed(self):
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / '.lock'
            lock.write_text('another-owner')
            self.ns['_LOCKFILE'] = str(lock)
            self.ns['_LOCK_OWNED'] = False
            with self.assertRaises(RuntimeError): self.ns['_write_lockfile']()
            self.ns['_remove_lockfile']()
            self.assertEqual(lock.read_text(), 'another-owner')

    def test_owned_lock_can_be_released(self):
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / '.lock'
            self.ns['_LOCKFILE'] = str(lock)
            self.ns['_LOCK_OWNED'] = False
            self.ns['_write_lockfile']()
            self.assertEqual(lock.read_text(), str(os.getpid()))
            self.ns['_remove_lockfile']()
            self.assertFalse(lock.exists())

    def test_capture_success_preserves_existing_raw_file(self):
        self.ns['_grab_frame_flushed'] = lambda *a, **kw: object()
        self.ns['_sharpness_score'] = lambda frame: 100.0
        self.ns['_HAS_CV2'] = False
        self.ns['Camera'] = SimpleNamespace(save_image=lambda frame, path, **kw: Path(path).write_bytes(b'RAW'))
        with tempfile.TemporaryDirectory() as td:
            photo = Path(td) / 'mosaic_r0_c0.png'
            ok, _, _ = self.ns['capture_and_save'](None, str(photo))
            self.assertTrue(ok)
            self.assertEqual(photo.read_bytes(), b'RAW')
            ok, _, _ = self.ns['capture_and_save'](None, str(photo))
            self.assertFalse(ok)
            self.assertEqual(photo.read_bytes(), b'RAW')

    def test_cancel_before_capture_never_reads_camera(self):
        event = threading.Event(); event.set()
        self.ns['_grab_frame_flushed'] = lambda *a, **kw: self.fail('camera must not be read')
        self.assertFalse(self.ns['capture_and_save'](None, 'unused.png', event)[0])

    def test_failed_summary_is_not_marked_completed_or_aborted(self):
        obj = SimpleNamespace(auto_running=True, _worker=None, _planned_total=4,
                              btn_auto=SimpleNamespace(config=lambda **kw: None),
                              btn_resume=SimpleNamespace(config=lambda **kw: None),
                              after=lambda *a: None, _refresh_pos=lambda: None,
                              _worker_error=RuntimeError('camera failed'),
                              _scan_done_count=1, _scan_params=(2, 2, 143, 76, 'X_first'),
                              stage=SimpleNamespace(position_trusted=True),
                              _cancel_event=threading.Event(), _run_auto_capture=True,
                              _session_log=Log(), _sharpness_records=[],
                              _session_folder='stub', _photo_count=1,
                              var_status=Value(), var_progress=Value())
        obj._cancel_event.set()
        self.gui._finalize_auto(obj)
        self.assertEqual(obj._session_log.finished[0], 'failed')
        self.assertEqual(obj._session_log.finished[1]['error'], 'camera failed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
