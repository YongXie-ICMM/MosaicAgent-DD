"""Offline tests: all camera interfaces are fakes; never open real devices."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

from camera_session_history import ScanSession, describe_camera, open_detected_camera


class FakeCamera:
    created = []
    failing = set()
    blank = set()

    def __init__(self, camera_index, bits, backend, resolution):
        self.camera_index = camera_index
        self.backend = backend
        self._requested_resolution = resolution
        self.closed = False
        self._uvc_supported = {}
        self.created.append(self)

    def open(self):
        if (self.backend, self.camera_index) in self.failing:
            raise OSError('SDK DLL absent')

    def start_live(self):
        pass

    def grab_frame(self, timeout):
        if (self.backend, self.camera_index) in self.blank:
            return None
        return SimpleNamespace(size=1920*1080*3, shape=(1080,1920,3))

    def get_resolution(self):
        return (640,480)  # Deliberately stale readback: received frame must win.

    def get_exposure_time(self):
        return 0

    def close(self):
        self.closed = True


class HistoryTests(unittest.TestCase):
    def setUp(self):
        FakeCamera.created = []
        FakeCamera.failing = set()
        FakeCamera.blank = set()

    def test_sdk_oserror_falls_back_to_verified_uvc_frame(self):
        FakeCamera.failing = {('auto',0)}
        with TemporaryDirectory() as d:
            cam, info = open_detected_camera(FakeCamera,d,indices=[0])
            self.assertEqual(cam.backend,'opencv')
            self.assertTrue(FakeCamera.created[0].closed)
            self.assertEqual(info['actual_resolution'],[1920,1080])
            self.assertIsNone(info['model'])
            self.assertIsNone(info['readback']['exposure']['value'])

    def test_open_handle_without_frame_is_not_success(self):
        FakeCamera.blank = {('auto',0),('opencv',0)}
        with TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'No camera delivered'):
                open_detected_camera(FakeCamera,d,indices=[0])
            self.assertTrue(all(c.closed for c in FakeCamera.created))
            self.assertFalse((Path(d)/'last_camera.json').exists())

    def test_previous_identity_not_carried_to_replaced_camera(self):
        with TemporaryDirectory() as d:
            (Path(d)/'last_camera.json').write_text(json.dumps({'backend':'opencv','camera_index':2,'model':'old model'}))
            cam, info=open_detected_camera(FakeCamera,d,indices=[0,1,2])
            self.assertEqual(cam.camera_index,2)
            self.assertIsNone(info['model'])
            self.assertFalse(info['previous_configuration_reused'])

    def test_next_camera_excludes_old_index_even_with_history(self):
        with TemporaryDirectory() as d:
            (Path(d)/'last_camera.json').write_text(json.dumps({'backend':'opencv','camera_index':0}))
            cam,_=open_detected_camera(FakeCamera,d,indices=[0,1],exclude_indices=[0])
            self.assertEqual(cam.camera_index,1)

    def test_corrupt_history_recovers_and_append_preserves_connections(self):
        with TemporaryDirectory() as d:
            (Path(d)/'last_camera.json').write_text('{broken')
            for _ in range(2):
                cam,_=open_detected_camera(FakeCamera,d,indices=[0]);cam.close()
            self.assertEqual(len((Path(d)/'camera_connections.jsonl').read_text().splitlines()),2)

    def test_session_preserves_code_and_image_hash_and_does_not_overwrite(self):
        with TemporaryDirectory() as d:
            root=Path(d); source=root/'source.py';source.write_text('print(1)')
            folder=root/'run';s=ScanSession(folder,{'backend':'opencv'},{'nx':3,'ny':3},[source],True)
            photo=folder/'mosaic_r0_c0.png';photo.write_bytes(b'fake image')
            s.record('capture',status='saved',filepath=str(photo),row=0,col=0)
            s.record('capture',status='failed',row=0,col=1)
            s.finish('failed',saved_images=1)
            records=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
            self.assertEqual(len(records[1]['image_sha256']),64)
            state=json.loads((folder/'session.json').read_text())
            self.assertEqual(state['status'],'failed')
            self.assertTrue(state['stage_simulated'])
            self.assertEqual((folder/'program_snapshot/source.py').read_text(),'print(1)')
            with self.assertRaises(FileExistsError):ScanSession(folder,{}, {}, [])


if __name__=='__main__':
    unittest.main()
