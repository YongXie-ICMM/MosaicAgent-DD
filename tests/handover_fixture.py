"""Build a realistic fake ``Auto_Scan`` tree for handover tests.

Records are written with the scanner's own ``ScanSession`` / ``SharedHistory`` classes
(unchanged, hash-locked runtime files), so the formats are the real ones. Tiles come from
the synthetic specimen of ``test_colour_match_grid`` so the colour and stitch steps have
something to register. Two runs reproduce the 2026-10-06 situation: run 1 covers
columns 0..NCOLS-2 and stops with ``move_failed``; run 2 (brighter, different gain)
covers the last column and re-shoots the last tile of run 1's final column.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCAN_SRC = ROOT / "acquisition" / "Auto_Scan"
sys.path.insert(0, str(SCAN_SRC))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import test_colour_match_grid as T  # noqa: E402

NCOLS, NROWS = 5, 4
CAMERA_INFO_BASE = {"backend": "opencv", "camera_index": 0, "requested_resolution": [T.TILE_W, T.TILE_H],
                    "actual_resolution": [T.TILE_W, T.TILE_H], "display_name": "fixture camera", "model": "fixture",
                    "identity_status": "fixture", "microscope_view_confirmed": True}


def scan_config(sample_id="FIXTURE_5mg"):
    return {"acquisition_contract": {"schema_version": 1, "expected_image_size": [T.TILE_W, T.TILE_H],
                                     "actual_image_size": [T.TILE_W, T.TILE_H], "verification_status": "verified_received_frame",
                                     "verification_phase": "pre_scan", "calibration_status": "unverified_for_current_mode",
                                     "raw_images_resized": False},
            "sample_id": sample_id, "objective": "20x", "device_x": "sim", "device_y": "sim", "overlap_x_steps": 8, "overlap_y_steps": 8,
            "nx": NCOLS, "ny": NROWS, "dx_steps": 143, "dy_steps": 76, "order": "Y_first", "auto_capture": True,
            "release": "954-1080p-A1", "review_mode": "step_confirm", "origin_logical_steps": [0, 0], "fullstep_um": 2.5,
            "px_per_step": 25.6, "calibration_status": "legacy defaults", "fov_steps_x": 151.04, "fov_steps_y": 84.375,
            "invert_x": True, "invert_y": False, "stage_simulated": True, "camera_source": "connected camera"}


def _tile_array(canvas, vig, r, c, factor):
    x_right = canvas.shape[1] - T.TILE_W - 10 - (NROWS - 1) * T.ROW_VEC[0]
    x0 = x_right + c * T.COL_VEC[0]
    y0 = 10 + r * T.ROW_VEC[1] + c * T.COL_VEC[1]
    x = x0 + r * T.ROW_VEC[0]
    tile = canvas[y0:y0 + T.TILE_H, x:x + T.TILE_W] * vig * factor
    return np.clip(np.rint(tile), 0, 255).astype(np.uint8)


def _write_run(scan_dir: Path, stamp: str, points, *, factor, gain_readback, history, fail_after=None,
               status="completed", sample_id="FIXTURE_5mg"):
    """points: list of (row, col) in capture order. Returns the session folder."""
    from camera_session_history import ScanSession
    canvas = T.specimen(0)
    vig = T.vignette()
    folder = scan_dir / "mosaic_photos" / stamp
    camera = dict(CAMERA_INFO_BASE, readback={"gain": gain_readback, "exposure_ms": 20.0, "auto_exposure": False})
    log = ScanSession(folder, camera, scan_config(sample_id), [SCAN_SRC / "acquisition_contract.py", SCAN_SRC / "delivery_manifest.json"],
                      simulated=True, shared_history=history)
    log.record("acquisition_mode_verified", acquisition_contract={"verification_status": "verified_received_frame"})
    photos = 0
    for i, (r, c) in enumerate(points, 1):
        log.record("move_attempt", row=r, col=c, target_x_steps=c * 143, target_y_steps=r * 76, dx_steps=143, dy_steps=0)
        if fail_after is not None and i > fail_after:
            log.record("move_failed", row=r, col=c, error="Y movement failed: rc=-1; no photo taken",
                       verification={"before": {"x": c * 143, "y": r * 76}, "after": None})
            break
        log.record("move_completed", row=r, col=c, x_steps=c * 143, y_steps=r * 76, verification={"ok": True})
        arr = _tile_array(canvas, vig, r, c, factor)
        name = f"mosaic_r{r}_c{c}.png"
        Image.fromarray(arr).save(folder / name)
        log.record("capture_success", sharpness=120.0, retries=0, below_blur_threshold=False, camera_backend="opencv",
                   camera_index=0, row=r, col=c, point=i, x_steps=c * 143, y_steps=r * 76, filename=name,
                   filepath=str(folder / name))
        photos += 1
    (folder / "sharpness.csv").write_text("row,col,sharpness\n" + "".join(f"{r},{c},120.0\n" for r, c in points[:photos]), encoding="utf-8")
    log.finish(status, positions_completed=photos, photos_saved=photos, planned_positions=len(points))
    return folder


def build_scan_dir(tmp_path: Path, *, with_runtime_files=True) -> dict:
    """Fake Auto_Scan with two runs, console logs, journals, camera and colour records."""
    from shared_history import SharedHistory
    scan_dir = tmp_path / "Auto_Scan"
    scan_dir.mkdir()
    if with_runtime_files:
        manifest = json.loads((SCAN_SRC / "delivery_manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            shutil.copy2(SCAN_SRC / entry["path"], scan_dir / entry["path"])
        shutil.copy2(SCAN_SRC / "delivery_manifest.json", scan_dir / "delivery_manifest.json")
    stamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = SharedHistory(scan_dir / "shared_history", metadata={"application": "fixture", "release": "954-1080p-A1"})
    # run 1: snake over columns 0..NCOLS-2, fails on the way to the last point of column NCOLS-2
    points1 = []
    for c in range(NCOLS - 1):
        rows = range(NROWS) if c % 2 == 0 else range(NROWS - 1, -1, -1)
        points1.extend((r, c) for r in rows)
    run1 = _write_run(scan_dir, stamp_base + "_000001", points1, factor=np.ones(3, np.float32), gain_readback=21,
                      history=history, fail_after=len(points1) - 1, status="stopped_on_error")
    # run 2: brighter / bluer camera session, last column plus the point run 1 did not get
    gain = (0.94, 0.955, 0.92)
    factor2 = (np.ones(3, np.float32) / np.array(gain, np.float32))
    missing = points1[-1]
    points2 = [missing] + [(r, NCOLS - 1) for r in range(NROWS)]
    run2 = _write_run(scan_dir, stamp_base + "_000002", points2, factor=factor2, gain_readback=27, history=history)
    history.close("closed", source="fixture")
    logs = scan_dir / "history" / "console_logs"
    logs.mkdir(parents=True)
    (logs / f"launch_{stamp_base}_000001.txt").write_text(
        "Console history started: fixture\n[session] Photos will be saved to: " + str(run1) + "\n"
        "[scan] point 15/16\nY movement failed: rc=-1; no photo taken\n[stop] soft stop issued\n", encoding="utf-8")
    (logs / f"launch_{stamp_base}_000002.txt").write_text(
        "Console history started: fixture\n[session] Photos will be saved to: " + str(run2) + "\n[scan] done\n", encoding="utf-8")
    ch = scan_dir / "camera_history"
    ch.mkdir()
    (ch / "camera_connections.jsonl").write_text(json.dumps({"recorded_at_utc": "2026-10-06T06:00:00+00:00", "backend": "opencv",
                                                            "readback": {"gain": 21}}) + "\n" +
                                                 json.dumps({"recorded_at_utc": "2026-10-06T09:00:00+00:00", "backend": "opencv",
                                                            "readback": {"gain": 27}}) + "\n", encoding="utf-8")
    cal_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_fixt01"
    cal = scan_dir / "colour_calibration" / cal_stamp
    cal.mkdir(parents=True)
    Image.fromarray(np.full((T.TILE_H, T.TILE_W, 3), T.SUBSTRATE, np.uint8)).save(cal / "colour_reference_frame.png")
    (cal / "camera_colour_settings.json").write_text(json.dumps({"schema_version": 1, "kind": "camera_colour_calibration",
                                                                 "recorded_at_utc": "2026-10-06T05:55:00+00:00", "calibrated": True,
                                                                 "verdict": "within_tolerance", "mode": "auto", "accepted_by": "auto",
                                                                 "observed": {"median_rgb": [219.0, 171.0, 170.0]}}), encoding="utf-8")
    (cal / "calibration_steps.jsonl").write_text("{}\n", encoding="utf-8")
    (scan_dir / "colour_calibration" / "latest.json").write_text(json.dumps({"folder": cal_stamp, "calibrated": True}), encoding="utf-8")
    return {"scan_dir": scan_dir, "run1": run1, "run2": run2, "stamp1": run1.name, "stamp2": run2.name, "gain": gain,
            "missing_point": missing, "points1": points1, "points2": points2}


def ignore_sigpipe():
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError, OSError):
        pass


def count_files(path: Path) -> int:
    return sum(len(names) for _r, _d, names in os.walk(path))
