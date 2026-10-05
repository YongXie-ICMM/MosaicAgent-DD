"""Offline tests for the white-balance calibration helper: simulated camera, no hardware,
no SDK import, no stage, no network. Records are written to temporary folders only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wb_calibrate as wb  # noqa: E402

REFERENCE = {"schema_version": 1, "kind": "reference_substrate_colour",
             "mean_rgb": [219.2, 170.9, 170.0], "std_rgb": [9.5, 8.3, 8.1], "tolerance_fraction": 0.05}
QUIET = lambda *_: None  # noqa: E731


def within(record_or_measurement):
    comp = record_or_measurement["comparison"]
    return comp["verdict"] == "within_tolerance"


class SimulatedCameraContract(unittest.TestCase):
    def test_true_settings_render_the_reference_colour(self):
        cam = wb.SimulatedCamera(exposure=20000, temp=5200, tint=1250)
        measurement = wb.measure_frame(cam.grab_frame(), REFERENCE)
        self.assertTrue(measurement["gate"]["clean"], measurement["gate"]["reasons"])
        self.assertTrue(within(measurement))
        np.testing.assert_allclose(measurement["observed"]["median_rgb"], REFERENCE["mean_rgb"], atol=2)

    def test_shifted_start_is_detected_as_different(self):
        cam = wb.SimulatedCamera(exposure=9000, temp=7600, tint=720)
        measurement = wb.measure_frame(cam.grab_frame(), REFERENCE)
        self.assertNotEqual(measurement["comparison"]["verdict"], "within_tolerance")

    def test_frames_are_bgr_like_the_real_camera(self):
        cam = wb.SimulatedCamera(exposure=20000, temp=5200, tint=1250)
        frame = cam.grab_frame()
        median = np.median(frame.reshape(-1, 3), axis=0)
        self.assertGreater(median[2], median[0])      # red (index 2 in BGR) is the brightest channel of the substrate


class CleanSubstrateGate(unittest.TestCase):
    def test_crystals_in_the_field_block_recording(self):
        cam = wb.SimulatedCamera(exposure=20000, temp=5200, tint=1250, clean=False)
        measurement = wb.measure_frame(cam.grab_frame(), REFERENCE)
        self.assertFalse(measurement["gate"]["clean"])
        self.assertTrue(any("plateau" in r for r in measurement["gate"]["reasons"]))
        hints = wb.guidance(measurement, REFERENCE)
        self.assertIn("clean bare-substrate", hints[0])

    def test_saturation_and_darkness_are_exposure_findings_not_cleanliness(self):
        bright = wb.SimulatedCamera(exposure=80000, temp=5200, tint=1250)
        measurement = wb.measure_frame(bright.grab_frame(), REFERENCE)
        self.assertTrue(measurement["gate"]["clean"])
        self.assertEqual(measurement["gate"]["exposure"], "saturated")
        dark = wb.SimulatedCamera(exposure=2000, temp=5200, tint=1250)
        measurement = wb.measure_frame(dark.grab_frame(), REFERENCE)
        self.assertEqual(measurement["gate"]["exposure"], "too_dark")

    def test_guidance_names_the_axis_that_is_off(self):
        # with temp_sign=+1 a temperature below the true value renders bluer
        too_blue = wb.SimulatedCamera(exposure=20000, temp=3600, tint=1250)
        hints = " ".join(wb.guidance(wb.measure_frame(too_blue.grab_frame(), REFERENCE), REFERENCE))
        self.assertIn("Too blue", hints)
        too_red = wb.SimulatedCamera(exposure=20000, temp=6400, tint=1250)
        hints = " ".join(wb.guidance(wb.measure_frame(too_red.grab_frame(), REFERENCE), REFERENCE))
        self.assertIn("Too red", hints)
        not_green = wb.SimulatedCamera(exposure=20000, temp=5200, tint=720)
        hints = " ".join(wb.guidance(wb.measure_frame(not_green.grab_frame(), REFERENCE), REFERENCE))
        self.assertIn("Not enough green", hints)
        dark = wb.SimulatedCamera(exposure=14000, temp=5200, tint=1250)
        hints = " ".join(wb.guidance(wb.measure_frame(dark.grab_frame(), REFERENCE), REFERENCE))
        self.assertIn("Too dark", hints)


class AutomaticAdjustment(unittest.TestCase):
    def run_auto(self, **camera_kwargs):
        cam = wb.SimulatedCamera(**camera_kwargs)
        session = wb.Session(cam, REFERENCE, settle_s=0.0, log=QUIET, sleep=lambda s: None)
        report = session.auto_adjust()
        return cam, session, report

    def test_converges_from_the_shifted_capture_mode(self):
        cam, session, report = self.run_auto(exposure=9000, temp=7600, tint=720)
        self.assertEqual(report["outcome"], "within_tolerance")
        self.assertTrue(within(session.last_measurement))
        self.assertLess(cam.frames_delivered, 90)                       # bounded number of frames
        self.assertFalse(cam.auto_exposure)                             # automatics were switched off
        self.assertIn("auto_exposure_off", report["automatics"])

    def test_direction_conventions_are_discovered_not_assumed(self):
        for temp_sign, tint_sign in [(+1, +1), (-1, +1), (+1, -1), (-1, -1)]:
            with self.subTest(temp_sign=temp_sign, tint_sign=tint_sign):
                _, session, report = self.run_auto(exposure=9000, temp=7600, tint=720,
                                                   temp_sign=temp_sign, tint_sign=tint_sign)
                self.assertEqual(report["outcome"], "within_tolerance")

    def test_no_channel_is_clipped_on_the_way(self):
        # a strong imbalance must not be 'fixed' by driving the dominant channel to 255
        cam, session, report = self.run_auto(exposure=9000, temp=9000, tint=600)
        self.assertEqual(report["outcome"], "within_tolerance")
        self.assertTrue(all(step["median_rgb"][0] < 250 for step in session.steps[1:]), session.steps)

    def test_missing_tint_control_is_reported_as_uncorrectable(self):
        _, session, report = self.run_auto(exposure=9000, temp=7600, tint=720, controls=("exposure", "temp"))
        self.assertNotEqual(report["outcome"], "within_tolerance")
        self.assertIn("green", report["uncorrectable_axes"])

    def test_dirty_field_is_never_adjusted_into_a_record(self):
        cam, session, report = self.run_auto(exposure=20000, temp=7600, tint=720, clean=False)
        self.assertEqual(report["outcome"], "field_not_clean")
        self.assertEqual(cam.set_log, [])                               # nothing was written to the camera

    def test_every_step_is_logged_with_controls_and_reading(self):
        _, session, _ = self.run_auto(exposure=9000, temp=7600, tint=720)
        self.assertGreaterEqual(len(session.steps), 3)
        for step in session.steps:
            self.assertEqual(set(step), {"at_utc", "note", "controls", "median_rgb", "gain_rgb", "verdict", "clean"})
            self.assertEqual(set(step["controls"]), {"exposure", "temp", "tint"})


class ManualMode(unittest.TestCase):
    def test_camera_without_controls_gets_guidance_only(self):
        cam = wb.SimulatedCamera(exposure=20000, temp=7600, tint=1250, controls=())
        session = wb.Session(cam, REFERENCE, settle_s=0.0, log=QUIET, sleep=lambda s: None)
        self.assertEqual(session.controls.kind, "simulated")
        self.assertEqual(session.controls.available, {})
        decision, ready = wb.manual_loop(session, window=False, max_seconds=0.3, accept_when_ready=True, log=QUIET)
        self.assertEqual((decision, ready), ("timeout", False))
        self.assertEqual(cam.set_log, [])

    def test_manual_accepts_as_soon_as_the_operator_reaches_tolerance(self):
        cam = wb.SimulatedCamera(exposure=20000, temp=5200, tint=1250, controls=())
        session = wb.Session(cam, REFERENCE, settle_s=0.0, log=QUIET, sleep=lambda s: None)
        decision, ready = wb.manual_loop(session, window=False, max_seconds=5, accept_when_ready=True, log=QUIET)
        self.assertEqual((decision, ready), ("accepted", True))


class Recording(unittest.TestCase):
    def test_record_binds_frame_reference_and_settings(self):
        cam = wb.SimulatedCamera(exposure=9000, temp=7600, tint=720)
        session = wb.Session(cam, REFERENCE, settle_s=0.0, log=QUIET, sleep=lambda s: None)
        adjustment = session.auto_adjust()
        with tempfile.TemporaryDirectory() as temp:
            reference_path = Path(temp) / "reference.json"
            reference_path.write_text(json.dumps(REFERENCE), encoding="utf-8")
            record = wb.write_record(Path(temp) / "out", session=session, reference_path=reference_path, mode="auto",
                                     accepted_by="automatic_within_tolerance", adjustment=adjustment, operator=None,
                                     simulated=False)
            folder = Path(record["folder"])
            self.assertTrue(record["calibrated"])
            self.assertEqual(record["verdict"], "within_tolerance")
            self.assertEqual(record["reference"]["sha256"], wb.sha256_file(reference_path))
            frame_path = folder / record["frame"]["path"]
            self.assertEqual(record["frame"]["sha256"], wb.sha256_file(frame_path))
            self.assertEqual(set(record["controls_readback"]), {"exposure", "temp", "tint"})
            self.assertEqual(len((folder / "calibration_steps.jsonl").read_text(encoding="utf-8").splitlines()),
                             len(session.steps))
            latest = json.loads((Path(temp) / "out" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["folder"], folder.name)
            self.assertFalse(latest["simulated"])
            # the saved PNG decodes back to the camera's RGB reading
            try:
                import cv2
                decoded = cv2.imread(str(frame_path))[:, :, ::-1]
            except ImportError:
                from PIL import Image
                decoded = np.asarray(Image.open(frame_path).convert("RGB"))
            np.testing.assert_allclose(np.median(decoded.reshape(-1, 3), axis=0), record["observed"]["median_rgb"], atol=2)

    def test_simulated_records_stay_in_the_simulation_folder_without_latest(self):
        cam = wb.SimulatedCamera(exposure=20000, temp=5200, tint=1250)
        session = wb.Session(cam, REFERENCE, settle_s=0.0, log=QUIET, sleep=lambda s: None)
        session.read("start")
        with tempfile.TemporaryDirectory() as temp:
            reference_path = Path(temp) / "reference.json"
            reference_path.write_text(json.dumps(REFERENCE), encoding="utf-8")
            record = wb.write_record(Path(temp) / "out", session=session, reference_path=reference_path, mode="manual",
                                     accepted_by="operator_key", adjustment=None, operator=None, simulated=True)
            self.assertIn("simulation", Path(record["folder"]).parts)
            self.assertTrue(record["simulated"])
            self.assertFalse((Path(temp) / "out" / "latest.json").exists())

    def test_forced_record_outside_tolerance_is_not_called_calibrated(self):
        cam = wb.SimulatedCamera(exposure=20000, temp=7600, tint=1250, controls=())
        session = wb.Session(cam, REFERENCE, settle_s=0.0, log=QUIET, sleep=lambda s: None)
        session.read("manual")
        with tempfile.TemporaryDirectory() as temp:
            reference_path = Path(temp) / "reference.json"
            reference_path.write_text(json.dumps(REFERENCE), encoding="utf-8")
            record = wb.write_record(Path(temp) / "out", session=session, reference_path=reference_path, mode="manual",
                                     accepted_by="operator_forced_outside_tolerance", adjustment=None, operator=None,
                                     simulated=False)
            self.assertFalse(record["calibrated"])
            self.assertEqual(record["verdict"], "colour_balance_differs_from_reference")


class CommandLine(unittest.TestCase):
    def test_simulated_auto_run_exits_zero_and_records_a_simulation(self):
        with tempfile.TemporaryDirectory() as temp:
            reference_path = Path(temp) / "reference.json"
            reference_path.write_text(json.dumps(REFERENCE), encoding="utf-8")
            code = wb.main(["--auto", "--simulate-camera", "--no-prompts", "--out", str(Path(temp) / "out"),
                            "--reference", str(reference_path)])
            self.assertEqual(code, wb.EXIT_OK)
            folders = list((Path(temp) / "out" / "simulation").iterdir())
            self.assertEqual(len(folders), 1)
            self.assertFalse((Path(temp) / "out" / "latest.json").exists())

    def test_dirty_simulated_field_exits_without_a_record(self):
        with tempfile.TemporaryDirectory() as temp:
            reference_path = Path(temp) / "reference.json"
            reference_path.write_text(json.dumps(REFERENCE), encoding="utf-8")
            code = wb.main(["--auto", "--simulate-camera", "--simulate-dirty", "--no-prompts",
                            "--out", str(Path(temp) / "out"), "--reference", str(reference_path)])
            self.assertEqual(code, wb.EXIT_NOT_RECORDED)
            self.assertFalse((Path(temp) / "out").exists())

    def test_missing_or_invalid_reference_is_an_error_before_any_camera(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(wb.main(["--auto", "--simulate-camera", "--reference", str(Path(temp) / "none.json")]),
                             wb.EXIT_ERROR)
            broken = Path(temp) / "broken.json"
            broken.write_text(json.dumps(dict(REFERENCE, mean_rgb=[1, 2])), encoding="utf-8")
            self.assertEqual(wb.main(["--auto", "--simulate-camera", "--reference", str(broken)]), wb.EXIT_ERROR)

    def test_shipped_reference_is_the_default(self):
        self.assertEqual(wb.DEFAULT_REFERENCE.name, "reference_substrate_colour.json")
        self.assertTrue(wb.DEFAULT_REFERENCE.is_file())
        self.assertIsNotNone(wb.cd.load_reference(wb.DEFAULT_REFERENCE))


class ScannerRuntimeUntouched(unittest.TestCase):
    def test_helper_is_not_part_of_the_hash_locked_runtime(self):
        manifest = json.loads((HERE / "delivery_manifest.json").read_text(encoding="utf-8"))
        listed = {item["path"] for item in manifest["files"]}
        self.assertNotIn("wb_calibrate.py", listed)
        self.assertNotIn("00_white_balance_then_scan.bat", listed)
        self.assertIn("02_start_scan.bat", listed)

    def test_module_import_opens_no_hardware_modules(self):
        self.assertNotIn("Camera_v2", sys.modules)
        self.assertNotIn("amcam", sys.modules)
        self.assertNotIn("toupcam", sys.modules)
        self.assertNotIn("ximc", sys.modules)


if __name__ == "__main__":
    unittest.main()
