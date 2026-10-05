"""White-balance and exposure calibration on a clean bare-substrate field (camera only).

Why: the 2026-10-05 controlled comparison (docs/diagnostics/20261005_colour_balance)
showed that the layer model reads the new capture mode "one layer thicker" because its
colour balance differs from the reference substrate colour, not because the pixel count
changed. The agreed first remedy is to bring the camera back to the reference at
acquisition time and to record the settings. This helper does exactly that and nothing
else:

* it opens the camera through the unchanged ``Camera_v2`` / ``camera_session_history``
  discovery (same 1920 x 1080 frame contract as the scanner), never the XY stage;
* it measures the bare-substrate colour with the same estimator the analysis uses
  (``flakepipeline.color_diagnostics.background_colour``) and compares it with
  ``configs/reference_substrate_colour.json``;
* with ``--auto`` it adjusts exposure, then colour temperature / tint (SDK) or the
  white-balance temperature (UVC) by measurement-driven secant steps, assuming no
  direction convention; without controls it shows live guidance and the student adjusts
  the camera's own menu;
* it records ``colour_reference_frame.png`` and ``camera_colour_settings.json`` under
  ``colour_calibration/<timestamp>/`` (plus ``latest.json``), and refuses to call the
  result calibrated when the reading is outside tolerance.

It does not change the scanner runtime, which keeps reading and recording camera
settings without writing them. The analysis-side colour check on the acquired images
remains the gate; this record is evidence of how the camera was set, not a reflectance
calibration. A VLM opinion on "is this clean bare substrate" is deliberately not part of
this tool: the deterministic flatness checks below decide, and nothing here sets camera
parameters from any model output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
try:
    from flakepipeline import color_diagnostics as cd
except ImportError as exc:                       # pragma: no cover - packaging error
    raise SystemExit("flakepipeline/color_diagnostics.py was not found beside acquisition/. "
                     "Run this helper from the complete student package, not from a copied Auto_Scan folder. "
                     f"({exc})")

SCHEMA_VERSION = 1
DEFAULT_REFERENCE = REPO / "configs" / "reference_substrate_colour.json"
DEFAULT_OUTPUT = HERE / "colour_calibration"
CAMERA_HISTORY = HERE / "camera_history"
EXPECTED_SIZE = (1920, 1080)
LUMA = np.array([0.299, 0.587, 0.114], np.float64)

# Clean bare-substrate gate (geometry of the field, independent of exposure):
CLEAN_MIN_FLAT_FRACTION = 0.50      # most of the field must be locally flat
CLEAN_MIN_PLATEAU_SHARE = 0.85      # one luminance plateau (allowing a vignetting gradient) holds the flat pixels
CLEAN_MAX_STD = 12.0                # per-channel spread inside that plateau
SATURATED_LEVEL = 250
MAX_SATURATED_FRACTION = 0.005
MIN_MEDIAN_LUMINANCE = 40.0

EXIT_OK, EXIT_ERROR, EXIT_NOT_WITHIN_TOLERANCE, EXIT_NOT_RECORDED = 0, 1, 2, 3


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def save_png(frame_bgr: np.ndarray, path: Path) -> None:
    """Lossless PNG of the camera frame (BGR as delivered by Camera_v2)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        if not cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]):
            raise RuntimeError("cv2.imwrite returned False")
        return
    except ImportError:
        pass
    from PIL import Image
    Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1])).save(path, format="PNG", compress_level=0)


# ----------------------------------------------------------------------------
# Measurement on one frame
# ----------------------------------------------------------------------------
def luminance(rgb) -> float:
    return float(np.dot(np.asarray(rgb, np.float64), LUMA))


def clean_substrate_check(observed: dict, rgb: np.ndarray) -> dict:
    """Deterministic 'is this a clean bare-substrate field' decision.

    ``clean`` concerns the geometry of the field (flat, one plateau, low spread) and is
    what gates recording. ``exposure`` reports saturation / darkness separately so the
    exposure loop can fix a clean but badly exposed field.
    """
    reasons = []
    if observed["flat_fraction"] < CLEAN_MIN_FLAT_FRACTION:
        reasons.append(f"only {observed['flat_fraction']:.0%} of the field is flat (need >= {CLEAN_MIN_FLAT_FRACTION:.0%}): "
                       "crystals, particles, edges or focus problems / 平坦区域不足：有晶体、颗粒、边缘或失焦")
    selected = observed.get("selected_plateau")
    plateaus = observed.get("plateaus") or []
    if selected is None:
        reasons.append("no dominant luminance plateau / 没有占主导的亮度台阶")
    else:
        merged = set(selected.get("merged_centres") or [selected["centre"]])
        share = sum(p["share_of_flat"] for p in plateaus if p["centre"] in merged)
        others = [p for p in plateaus if p["centre"] not in merged]
        if share < CLEAN_MIN_PLATEAU_SHARE:
            reasons.append(f"the brightest plateau holds only {share:.0%} of the flat pixels (need >= {CLEAN_MIN_PLATEAU_SHARE:.0%}) "
                           "/ 最亮台阶占比不足，说明视野里还有别的材料")
        if others:
            reasons.append(f"{len(others)} further luminance plateau(s) present: material other than bare substrate in the field "
                           "/ 视野里存在其它亮度台阶（有材料）")
    if max(observed["std_rgb"]) > CLEAN_MAX_STD:
        reasons.append(f"colour spread inside the plateau {max(observed['std_rgb']):.1f} > {CLEAN_MAX_STD}: "
                       "not uniform / 台阶内颜色离散过大，不均匀")
    saturated = float(np.mean(np.any(rgb >= SATURATED_LEVEL, axis=2)))
    lum = luminance(observed["median_rgb"])
    exposure = "ok"
    if saturated > MAX_SATURATED_FRACTION:
        exposure = "saturated"
    elif lum < MIN_MEDIAN_LUMINANCE:
        exposure = "too_dark"
    return {"clean": not reasons, "reasons": reasons, "exposure": exposure,
            "saturated_fraction": round(saturated, 5), "median_luminance": round(lum, 2)}


def measure_frame(frame_bgr: np.ndarray, reference: dict) -> dict:
    """Bare-substrate colour of one BGR frame, its comparison with the reference and the gate."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR colour frame")
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    observed = cd.background_colour(rgb)
    comparison = cd.compare_to_reference(observed, reference)
    gate = clean_substrate_check(observed, rgb)
    return {"observed": observed, "comparison": comparison, "gate": gate,
            "frame_size_wh": [int(rgb.shape[1]), int(rgb.shape[0])]}


def guidance(measurement: dict, reference: dict) -> list[str]:
    """Human-readable hints for a student adjusting the camera's own menu (EN / 中文)."""
    hints = []
    gate = measurement["gate"]
    if not gate["clean"]:
        hints.append("Move to a clean bare-substrate area (no crystals, dust or edges), refocus, then read again "
                     "/ 先移到干净裸衬底（无晶体、颗粒、边缘），对好焦再读数")
        hints.extend("  - " + r for r in gate["reasons"])
        return hints
    comp = measurement["comparison"]
    if comp["verdict"] == "within_tolerance":
        hints.append("Within tolerance: accept and record / 已在容差内：可以接受并记录")
        return hints
    obs = np.asarray(measurement["observed"]["median_rgb"], np.float64)
    ref = np.asarray(reference["mean_rgb"], np.float64)
    tol = reference["tolerance_fraction"]
    lum_ratio = luminance(obs) / luminance(ref)
    if gate["exposure"] == "saturated":
        hints.append("Overexposed (clipped pixels): shorten exposure or lower lamp brightness / 过曝：缩短曝光或调低灯光")
    elif lum_ratio < 1 - tol:
        hints.append(f"Too dark ({lum_ratio:.0%} of reference brightness): lengthen exposure or raise lamp brightness "
                     "/ 偏暗：加长曝光或调高灯光")
    elif lum_ratio > 1 + tol:
        hints.append(f"Too bright ({lum_ratio:.0%} of reference brightness): shorten exposure or lower lamp brightness "
                     "/ 偏亮：缩短曝光或调低灯光")
    rb_obs, rb_ref = obs[0] / obs[2], ref[0] / ref[2]
    if rb_obs < rb_ref * (1 - tol):
        hints.append(f"Too blue (R/B {rb_obs:.2f}, reference {rb_ref:.2f}): change the colour-temperature setting until R/B rises "
                     "/ 偏蓝：调色温直到 R/B 升到参考值")
    elif rb_obs > rb_ref * (1 + tol):
        hints.append(f"Too red (R/B {rb_obs:.2f}, reference {rb_ref:.2f}): change the colour-temperature setting until R/B falls "
                     "/ 偏红：调色温直到 R/B 降到参考值")
    g_obs, g_ref = obs[1] / math.sqrt(obs[0] * obs[2]), ref[1] / math.sqrt(ref[0] * ref[2])
    if g_obs < g_ref * (1 - tol):
        hints.append(f"Not enough green (G/√(RB) {g_obs:.2f}, reference {g_ref:.2f}): move the tint setting towards green "
                     "/ 绿色不足：把色调(tint)向绿色方向调")
    elif g_obs > g_ref * (1 + tol):
        hints.append(f"Too green (G/√(RB) {g_obs:.2f}, reference {g_ref:.2f}): move the tint setting towards magenta "
                     "/ 偏绿：把色调(tint)向洋红方向调")
    return hints


def format_reading(measurement: dict, reference: dict) -> str:
    obs = measurement["observed"]["median_rgb"]
    ref = reference["mean_rgb"]
    parts = []
    for name, o, r in zip("RGB", obs, ref):
        parts.append(f"{name} {o:5.1f}->{r:5.1f} ({(o / r - 1) * 100:+.1f}%)" if r else f"{name} {o:5.1f}")
    gate = "clean" if measurement["gate"]["clean"] else "NOT clean"
    return "  ".join(parts) + f"  | field: {gate}, exposure: {measurement['gate']['exposure']}, verdict: {measurement['comparison']['verdict']}"


# ----------------------------------------------------------------------------
# Camera control adapters
# ----------------------------------------------------------------------------
class Controls:
    """Uniform access to the controls a backend exposes; absent controls are reported, not faked."""

    def __init__(self, cam, kind: str, available: dict):
        self.cam = cam
        self.kind = kind                 # 'sdk' | 'uvc' | 'simulated' | 'none'
        self.available = dict(available)  # name -> {'bounds': (lo, hi), 'step': float, 'integer': bool}

    def has(self, name: str) -> bool:
        return name in self.available

    def bounds(self, name: str):
        return self.available[name]["bounds"]

    def step(self, name: str):
        return self.available[name]["step"]

    def integer(self, name: str) -> bool:
        return self.available[name]["integer"]

    def get(self, name: str):
        cam = self.cam
        if name == "exposure":
            return cam.get_exposure_time()
        if name == "gain":
            return cam.get_gain()
        if name in ("temp", "tint"):
            temp, tint = cam.get_temp_tint()
            return temp if name == "temp" else tint
        raise KeyError(name)

    def set(self, name: str, value):
        cam = self.cam
        if self.integer(name):
            value = int(round(value))
        lo, hi = self.bounds(name)
        value = min(max(value, lo), hi)
        if name == "exposure":
            cam.set_exposure_time(value)
        elif name == "gain":
            cam.set_gain(value)
        elif name == "temp":
            _, tint = cam.get_temp_tint()
            cam.set_temp_tint(value, tint if self.has("tint") else None)
        elif name == "tint":
            temp, _ = cam.get_temp_tint()
            cam.set_temp_tint(temp, value)
        else:
            raise KeyError(name)
        return value

    def disable_automatics(self) -> list[str]:
        done = []
        if self.kind in ("sdk", "uvc", "simulated"):
            try:
                self.cam.set_auto_exposure(False)
                done.append("auto_exposure_off")
            except Exception as exc:        # a driver may refuse; the measurement loop still works
                done.append(f"auto_exposure_unchanged:{exc}")
        if self.kind == "uvc" and getattr(self.cam, "_has_uvc", lambda n: False)("auto_wb"):
            try:
                self.cam._set_uvc("auto_wb", 0)
                done.append("auto_white_balance_off")
            except Exception as exc:
                done.append(f"auto_white_balance_unchanged:{exc}")
        return done

    def readback(self) -> dict:
        values = {}
        for name in ("exposure", "gain", "temp", "tint"):
            if self.has(name):
                try:
                    values[name] = self.get(name)
                except Exception as exc:
                    values[name] = f"unreadable:{exc}"
        return values


def controls_for(cam) -> Controls:
    backend = getattr(cam, "backend", None)
    if backend == "simulated":
        return Controls(cam, "simulated", cam.control_table())
    if backend in ("amcam", "toupcam") and getattr(cam, "hcam", None) is not None:
        available = {}
        try:
            emin, emax, _ = cam.get_exposure_range()
            available["exposure"] = {"bounds": (max(int(emin), 1), int(emax)), "step": None, "integer": True}
        except Exception:
            pass
        try:
            gmin, gmax, _ = cam.get_gain_range()
            if gmax > gmin:
                available["gain"] = {"bounds": (int(gmin), int(gmax)), "step": max(1, int((gmax - gmin) / 20)), "integer": True}
        except Exception:
            pass
        try:
            cam.get_temp_tint()
            # SDK documented ranges: temperature 2000..15000 K, tint 200..2500 (default 6503 / 1000)
            available["temp"] = {"bounds": (2000, 15000), "step": 400, "integer": True}
            available["tint"] = {"bounds": (200, 2500), "step": 120, "integer": True}
        except Exception:
            pass
        return Controls(cam, "sdk", available)
    if backend == "opencv":
        supported = getattr(cam, "_uvc_supported", {}) or {}
        available = {}
        if "exposure" in supported:
            available["exposure"] = {"bounds": (-13, -1), "step": 1, "integer": True}       # DirectShow log2 steps
        if "gain" in supported:
            available["gain"] = {"bounds": (0, 255), "step": 16, "integer": True}
        if "white_balance" in supported:
            available["temp"] = {"bounds": (2000, 10000), "step": 400, "integer": True}     # kelvin, no tint on UVC
        return Controls(cam, "uvc", available)
    return Controls(cam, "none", {})


# ----------------------------------------------------------------------------
# Simulated camera (offline tests and --simulate-camera self-test; never an instrument record)
# ----------------------------------------------------------------------------
class SimulatedCamera:
    """A bare-substrate field rendered from exposure, temperature and tint settings.

    The response directions are parameters (``temp_sign``, ``tint_sign``) so tests can
    prove that the controller discovers them instead of assuming a convention.
    """
    backend = "simulated"
    camera_index = 0
    _requested_resolution = EXPECTED_SIZE

    def __init__(self, *, exposure=12000, temp=6500, tint=1000, clean=True, size=(480, 270),
                 temp_sign=+1.0, tint_sign=+1.0, controls=("exposure", "temp", "tint"), noise=0.6, seed=0,
                 reference_rgb=(219.2, 170.9, 170.0), true_settings=(20000, 5200, 1250)):
        self.exposure, self.temp, self.tint = exposure, temp, tint
        self.auto_exposure = True
        self.clean, self.size, self.noise = clean, size, noise
        self.temp_sign, self.tint_sign = temp_sign, tint_sign
        self.controls = tuple(controls)
        self.reference_rgb = np.asarray(reference_rgb, np.float64)
        self.true_exposure, self.true_temp, self.true_tint = true_settings
        self.rng = np.random.default_rng(seed)
        self.frames_delivered = 0
        self.closed = False
        self.set_log = []

    # --- Camera_v2-like surface used by the helper ---
    def open(self):
        return None

    def close(self):
        self.closed = True

    def is_open(self):
        return not self.closed

    def start_live(self):
        return None

    def stop_live(self):
        return None

    def get_resolution(self):
        return self.size

    def set_auto_exposure(self, enable):
        self.auto_exposure = bool(enable)

    def get_exposure_time(self):
        return self.exposure

    def set_exposure_time(self, value):
        self.exposure = int(value)
        self.set_log.append(("exposure", self.exposure))

    def get_exposure_range(self):
        return (100, 500000, 12000)

    def get_gain(self):
        return 100

    def set_gain(self, value):
        self.set_log.append(("gain", value))

    def get_gain_range(self):
        return (100, 100, 100)

    def set_temp_tint(self, temp, tint=None):
        self.temp = int(temp)
        if tint is not None:
            self.tint = int(tint)
        self.set_log.append(("temp_tint", self.temp, self.tint))

    def get_temp_tint(self):
        return (self.temp, self.tint)

    @staticmethod
    def save_image(frame, filepath, quality=100):
        save_png(frame, Path(filepath))

    def control_table(self):
        table = {"exposure": {"bounds": (100, 500000), "step": None, "integer": True},
                 "temp": {"bounds": (2000, 15000), "step": 400, "integer": True},
                 "tint": {"bounds": (200, 2500), "step": 120, "integer": True}}
        return {k: v for k, v in table.items() if k in self.controls}

    # --- rendering ---
    def channel_gains(self):
        r = math.exp(self.temp_sign * 0.12 * (self.temp - self.true_temp) / 1000.0)
        b = math.exp(-self.temp_sign * 0.12 * (self.temp - self.true_temp) / 1000.0)
        g = math.exp(self.tint_sign * 0.18 * (self.tint - self.true_tint) / 1000.0)
        return np.array([r, g, b], np.float64)

    def grab_frame(self, timeout=5.0, flush=3):
        self.frames_delivered += 1
        width, height = self.size
        base = self.reference_rgb * (self.exposure / self.true_exposure) * self.channel_gains()
        image = np.empty((height, width, 3), np.float64)
        image[:] = base
        gradient = np.linspace(-6.0, 6.0, height)[:, None, None]       # vignetting-like illumination gradient
        image += gradient
        if not self.clean:
            for k in range(6):                                           # crystals occupy ~30 % of the field
                y, x = (k * 37) % (height - 60), (k * 71) % (width - 90)
                image[y:y + 60, x:x + 90] = base * np.array([0.9, 0.75, 1.0]) + gradient[y:y + 60]
        image += self.rng.normal(0.0, self.noise, image.shape)
        rgb = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb[:, :, ::-1])                     # BGR like Camera_v2


# ----------------------------------------------------------------------------
# Measurement-driven adjustment
# ----------------------------------------------------------------------------
class Session:
    """One camera, one reference; collects every step for the record."""

    def __init__(self, cam, reference: dict, *, settle_s=0.25, frames_per_reading=3, log=print, sleep=time.sleep):
        self.cam = cam
        self.reference = reference
        self.controls = controls_for(cam)
        self.settle_s = settle_s
        self.frames_per_reading = max(1, int(frames_per_reading))
        self.log = log
        self.sleep = sleep
        self.steps = []
        self.last_frame = None
        self.last_measurement = None

    # -- reading --
    def read(self, note: str = "") -> dict:
        """Median-of-frames reading after the camera has had time to apply a change."""
        if self.settle_s:
            self.sleep(self.settle_s)
        readings, frame = [], None
        for _ in range(self.frames_per_reading):
            frame = self.cam.grab_frame(timeout=3.0, flush=3)
            if frame is None or getattr(frame, "size", 0) == 0:
                raise RuntimeError("The camera stopped delivering frames / 相机没有送出图像")
            readings.append(measure_frame(frame, self.reference))
        rgb = np.median([r["observed"]["median_rgb"] for r in readings], axis=0)
        measurement = readings[-1]
        measurement["observed"]["median_rgb"] = [round(float(v), 2) for v in rgb]
        measurement["comparison"] = cd.compare_to_reference(measurement["observed"], self.reference)
        measurement["frames_in_reading"] = len(readings)
        self.last_frame, self.last_measurement = frame, measurement
        self.steps.append({"at_utc": now_utc(), "note": note, "controls": self.controls.readback(),
                           "median_rgb": measurement["observed"]["median_rgb"],
                           "gain_rgb": measurement["comparison"].get("gain_rgb"),
                           "verdict": measurement["comparison"]["verdict"], "clean": measurement["gate"]["clean"]})
        self.log(f"[reading] {note:<22} {format_reading(measurement, self.reference)}")
        return measurement

    # -- error metrics (log ratios; 0 = on reference) --
    def errors(self, measurement: dict) -> dict:
        obs = np.asarray(measurement["observed"]["median_rgb"], np.float64)
        ref = np.asarray(self.reference["mean_rgb"], np.float64)
        obs = np.maximum(obs, 1e-3)
        g = ref / obs
        # 'peak' puts the brightest observed channel on the brightest reference channel: with a
        # strong colour imbalance a luminance target would clip the dominant channel first.
        return {"luminance": math.log(luminance(ref) / max(luminance(obs), 1e-3)),
                "peak": math.log(float(ref.max()) / float(obs.max())),
                "red_blue": math.log(g[0] / g[2]),
                "green": math.log(g[1] / math.sqrt(g[0] * g[2]))}

    def secant(self, control: str, metric: str, *, tol: float, max_iter: int, note: str) -> dict:
        """Drive one control until |metric| <= tol using secant steps; no direction convention.

        Keeps the best value seen and restores it at the end. Reports a control as
        ineffective when changing it does not move the metric.
        """
        ctl = self.controls
        lo, hi = ctl.bounds(control)
        value = ctl.get(control)
        err = self.errors(self.last_measurement)[metric]
        best = (abs(err), value, err)
        history = [(value, err)]
        step = ctl.step(control) or max(abs(value) * 0.25, 1.0)
        if control == "exposure" and ctl.step(control) is None:
            # exposure in microseconds: brightness is roughly proportional, so the first
            # step can be the proportional guess itself (damped)
            trial = value * math.exp(0.8 * err)
        else:
            trial = value + step
        outcome = "max_iterations"
        for _ in range(max_iter):
            if abs(err) <= tol:
                outcome = "within_tolerance"
                break
            trial = min(max(trial, lo), hi)
            if ctl.integer(control):
                trial = int(round(trial))
            if trial == value:
                # pinned at a bound or quantisation limit
                outcome = "at_bound" if trial in (lo, hi) else "quantised"
                break
            ctl.set(control, trial)
            err_new = self.errors(self.read(f"{note} {control}={trial}"))[metric]
            history.append((trial, err_new))
            if abs(err_new) < best[0]:
                best = (abs(err_new), trial, err_new)
            slope = (err_new - err) / (trial - value)
            value, err = trial, err_new
            if abs(slope) < 1e-9:
                if len(history) >= 3 and abs(history[-1][1] - history[-3][1]) < 0.004:
                    outcome = "control_ineffective"
                    break
                trial = value + (step if trial >= value else -step) * 2
                continue
            jump = -err / slope
            limit = 3 * (step if ctl.step(control) else abs(value) * 0.5 + 1.0)
            jump = max(-limit, min(limit, jump))
            trial = value + jump
        if abs(err) <= tol:
            outcome = "within_tolerance"
        elif best[1] != value:
            ctl.set(control, best[1])
            self.read(f"{note} restore {control}={best[1]}")
        return {"control": control, "metric": metric, "outcome": outcome, "final_value": ctl.get(control),
                "final_error": round(self.errors(self.last_measurement)[metric], 4), "history": history}

    def auto_adjust(self, *, max_rounds: int = 3) -> dict:
        """Exposure first, then colour temperature and tint, repeated until all channels fit."""
        tol = self.reference["tolerance_fraction"]
        log_tol = math.log(1 + tol) * 0.5          # margin so that every channel passes the per-channel test
        report = {"mode": "auto", "controls": self.controls.kind, "available": sorted(self.controls.available),
                  "automatics": self.controls.disable_automatics(), "rounds": []}
        self.read("start")
        if not self.last_measurement["gate"]["clean"]:
            report["outcome"] = "field_not_clean"
            return report
        for round_index in range(max_rounds):
            actions = []
            if self.controls.has("exposure"):
                actions.append(self.secant("exposure", "peak", tol=log_tol, max_iter=8, note=f"r{round_index}"))
            if self.controls.has("gain") and abs(self.errors(self.last_measurement)["peak"]) > log_tol:
                actions.append(self.secant("gain", "peak", tol=log_tol, max_iter=6, note=f"r{round_index}"))
            if self.controls.has("temp"):
                actions.append(self.secant("temp", "red_blue", tol=log_tol, max_iter=8, note=f"r{round_index}"))
            if self.controls.has("tint"):
                actions.append(self.secant("tint", "green", tol=log_tol, max_iter=8, note=f"r{round_index}"))
            report["rounds"].append(actions)
            verdict = self.last_measurement["comparison"]["verdict"]
            if verdict == "within_tolerance" or not actions:
                break
            if not self.last_measurement["gate"]["clean"]:
                # a dark start can hide crystals; once the field is bright enough they show up
                report["outcome"] = "field_not_clean"
                report["detected"] = "during_adjustment"
                return report
        report["outcome"] = self.last_measurement["comparison"]["verdict"]
        missing = [m for m, c in (("green", "tint"), ("red_blue", "temp"), ("brightness", "exposure")) if not self.controls.has(c)]
        if report["outcome"] != "within_tolerance" and missing:
            report["uncorrectable_axes"] = missing
        return report


# ----------------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------------
def camera_description(cam, frame) -> dict:
    if getattr(cam, "backend", None) == "simulated":
        return {"backend": "simulated", "simulated": True, "actual_resolution": list(cam.get_resolution())}
    try:
        from camera_session_history import describe_camera
        return describe_camera(cam, frame)
    except Exception as exc:              # pragma: no cover - defensive, record what we can
        return {"backend": getattr(cam, "backend", None), "describe_error": str(exc)}


def write_record(out_dir: Path, *, session: Session, reference_path: Path, mode: str, accepted_by: str,
                 adjustment: dict | None, operator: dict | None, simulated: bool) -> dict:
    measurement = session.last_measurement
    frame = session.last_frame
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    folder = Path(out_dir) / ("simulation" if simulated else "") / stamp
    folder.mkdir(parents=True, exist_ok=False)
    frame_path = folder / "colour_reference_frame.png"
    type(session.cam).save_image(frame, str(frame_path))
    comparison = measurement["comparison"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "camera_colour_calibration",
        "recorded_at_utc": now_utc(),
        "simulated": bool(simulated),
        "mode": mode,
        "accepted_by": accepted_by,
        "calibrated": comparison["verdict"] == "within_tolerance" and measurement["gate"]["clean"],
        "verdict": comparison["verdict"],
        "clean_substrate_check": measurement["gate"],
        "observed": measurement["observed"],
        "comparison": comparison,
        "reference": {"path": str(reference_path), "sha256": sha256_file(reference_path),
                      "mean_rgb": session.reference["mean_rgb"], "tolerance_fraction": session.reference["tolerance_fraction"]},
        "camera": camera_description(session.cam, frame),
        "controls_kind": session.controls.kind,
        "controls_available": sorted(session.controls.available),
        "controls_readback": session.controls.readback(),
        "adjustment": adjustment,
        "operator": operator or {},
        "frame": {"path": frame_path.name, "sha256": sha256_file(frame_path), "bytes": frame_path.stat().st_size,
                  "size_wh": measurement["frame_size_wh"], "channel_order_on_disk": "PNG RGB (camera delivered BGR)"},
        "steps_file": "calibration_steps.jsonl",
        "environment": {"os": platform.platform(), "python": platform.python_version()},
        "limits": [
            "A colour-balance match to the reference substrate colour, not a reflectance or spectral calibration.",
            "The scanner runtime reads and records camera settings; it does not apply this file.",
            "Settings can be lost when the camera is reopened or power-cycled: re-read before each scan.",
            "The analysis colour check on the acquired images remains the gate for every run.",
        ],
    }
    atomic_json(folder / "camera_colour_settings.json", record)
    with (folder / "calibration_steps.jsonl").open("w", encoding="utf-8") as stream:
        for step in session.steps:
            stream.write(json.dumps(step, ensure_ascii=False) + "\n")
    if not simulated:
        latest = dict(record, folder=folder.name)
        atomic_json(Path(out_dir) / "latest.json", latest)
    record["folder"] = str(folder)
    return record


# ----------------------------------------------------------------------------
# Live guidance window (manual mode)
# ----------------------------------------------------------------------------
def manual_loop(session: Session, *, window: bool, max_seconds: float, accept_when_ready: bool, log=print) -> tuple[str, bool]:
    """Read repeatedly and show guidance; returns (decision, within_tolerance).

    decision: 'accepted' (within tolerance), 'forced' (recorded outside tolerance),
    'quit' or 'timeout'. With ``window`` a preview with keys a / f / q is shown.
    """
    cv2 = None
    if window:
        try:
            import cv2 as _cv2
            cv2 = _cv2
            cv2.namedWindow("White balance check (a=accept f=force-record q=quit)", cv2.WINDOW_NORMAL)
        except Exception as exc:
            log(f"[window] preview unavailable ({exc}); console guidance only / 预览窗口不可用，只用控制台提示")
            cv2 = None
    started = time.monotonic()
    decision = "timeout"
    while time.monotonic() - started < max_seconds:
        measurement = session.read("manual")
        for hint in guidance(measurement, session.reference):
            log("          " + hint)
        ready = measurement["gate"]["clean"] and measurement["comparison"]["verdict"] == "within_tolerance"
        if ready and accept_when_ready:
            decision = "accepted"
            break
        if cv2 is not None:
            frame = session.last_frame
            scale = 960 / max(frame.shape[1], 1)
            view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
            panel = np.full((150, view.shape[1], 3), 32, np.uint8)
            lines = [format_reading(measurement, session.reference)] + \
                    [h.split(" / ")[0] for h in guidance(measurement, session.reference)][:4]
            for i, text in enumerate(lines):
                cv2.putText(panel, text[:120], (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (80, 220, 80) if ready and i == 0 else (230, 230, 230), 1, cv2.LINE_AA)
            cv2.imshow("White balance check (a=accept f=force-record q=quit)", np.vstack([view, panel]))
            key = cv2.waitKey(400) & 0xFF
            if key == ord("a") and ready:
                decision = "accepted"
                break
            if key == ord("f"):
                decision = "forced"
                break
            if key in (ord("q"), 27):
                decision = "quit"
                break
    if cv2 is not None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return decision, bool(session.last_measurement and session.last_measurement["gate"]["clean"]
                          and session.last_measurement["comparison"]["verdict"] == "within_tolerance")


# ----------------------------------------------------------------------------
# Camera opening (real hardware only here)
# ----------------------------------------------------------------------------
def open_real_camera(*, camera_index: int | None, backend: str, log=print):
    """Use the scanner's own discovery and 1920 x 1080 frame contract; records the connection
    in camera_history exactly like a scan connection. Never touches the stage."""
    from Camera_v2 import Camera                      # imports the SDK / OpenCV only now
    from camera_session_history import open_detected_camera
    indices = range(4) if camera_index is None else [camera_index]
    if backend != "auto":
        # A forced backend bypasses discovery but keeps the frame-size contract.
        from acquisition_contract import require_frame_size
        cam = Camera(camera_index=indices[0], bits=24, backend=backend, resolution=EXPECTED_SIZE)
        cam.open()
        cam.start_live()
        frame = cam.grab_frame(timeout=3.0)
        if frame is None:
            cam.close()
            raise RuntimeError("Camera opened but delivered no frame / 相机已打开但没有图像")
        require_frame_size(frame, EXPECTED_SIZE, "colour_calibration")
        return cam, {"backend_requested": backend, "camera_index": indices[0]}
    cam, info = open_detected_camera(Camera, CAMERA_HISTORY, bits=24, resolution=EXPECTED_SIZE, indices=indices)
    log(f"[camera] {info.get('backend')} index {info.get('camera_index')} at {info.get('actual_resolution')}")
    return cam, info


def ask_operator(interactive: bool) -> dict:
    """Lamp brightness and objective are not readable from the camera; ask once."""
    fields = {"lamp_brightness": None, "objective": None, "notes": None}
    if not interactive or not sys.stdin or not sys.stdin.isatty():
        return fields
    prompts = {"lamp_brightness": "Lamp brightness setting (scale value) / 灯光亮度刻度 [Enter to skip]: ",
               "objective": "Objective (e.g. 50x) / 物镜 [Enter to skip]: ",
               "notes": "Notes / 备注 [Enter to skip]: "}
    for key, prompt in prompts.items():
        try:
            value = input(prompt).strip()
        except EOFError:
            value = ""
        fields[key] = value or None
    return fields


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auto", action="store_true",
                        help="adjust exposure, colour temperature and tint automatically when the backend exposes them")
    parser.add_argument("--manual", action="store_true", help="live guidance only; the operator adjusts the camera menu")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--backend", default="auto", choices=("auto", "opencv", "amcam", "toupcam"))
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-seconds", type=float, default=600.0, help="manual mode time limit")
    parser.add_argument("--no-window", action="store_true", help="console guidance only (no OpenCV preview window)")
    parser.add_argument("--accept-when-ready", action="store_true",
                        help="manual mode: record automatically as soon as the reading is within tolerance")
    parser.add_argument("--force-record", action="store_true",
                        help="record the final state even when outside tolerance (marked calibrated=false)")
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    parser.add_argument("--frames-per-reading", type=int, default=3)
    parser.add_argument("--simulate-camera", action="store_true",
                        help="self-test with a rendered field; records go to colour_calibration/simulation/ only")
    parser.add_argument("--simulate-dirty", action="store_true", help="with --simulate-camera: a field with crystals")
    parser.add_argument("--no-prompts", action="store_true", help="do not ask for lamp brightness / objective")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    log = lambda message: print(message, flush=True)       # noqa: E731
    try:
        reference = cd.load_reference(args.reference)
    except ValueError as exc:
        log(f"Reference file invalid / 参考文件无效: {args.reference}: {exc}")
        return EXIT_ERROR
    if reference is None:
        log(f"Reference substrate colour not found / 找不到参考衬底颜色文件: {args.reference}")
        return EXIT_ERROR
    log(f"Reference substrate RGB {reference['mean_rgb']} +- {reference['tolerance_fraction']:.0%} "
        f"({args.reference.name}) / 参考衬底颜色")

    simulated = bool(args.simulate_camera)
    cam = None
    try:
        if simulated:
            # start from the colour balance of the current (shifted) capture mode
            cam = SimulatedCamera(clean=not args.simulate_dirty, exposure=9000, temp=7600, tint=720)
            log("[camera] SIMULATED field; this is a self-test, not an instrument record / 模拟相机自测")
        else:
            cam, _ = open_real_camera(camera_index=args.camera_index, backend=args.backend, log=log)
        session = Session(cam, reference, settle_s=0.0 if simulated else args.settle_seconds,
                          frames_per_reading=args.frames_per_reading, log=log)
        if session.controls.kind == "none" and args.auto:
            log("This camera path exposes no exposure / white-balance controls (HDMI capture or limited driver). "
                "Adjust in the camera's own menu while watching the reading / 此采集路径没有可写的曝光和白平衡控制："
                "请在相机自带菜单里调，同时看读数")
        mode = "auto" if (args.auto and session.controls.kind != "none" and not args.manual) else "manual"
        adjustment = None
        if mode == "auto":
            adjustment = session.auto_adjust()
            log(f"[auto] outcome: {adjustment['outcome']}" +
                (f" (not controllable here: {adjustment['uncorrectable_axes']})" if adjustment.get("uncorrectable_axes") else ""))
            ready = adjustment["outcome"] == "within_tolerance"
            if ready:
                decision = "auto"
            elif adjustment["outcome"] == "field_not_clean":
                for hint in guidance(session.last_measurement, reference):
                    log("          " + hint)
                decision = "forced" if args.force_record else "quit"
            else:
                for hint in guidance(session.last_measurement, reference):
                    log("          " + hint)
                decision = "forced" if args.force_record else "quit"
        else:
            decision, ready = manual_loop(session, window=not args.no_window and not simulated,
                                          max_seconds=min(args.max_seconds, 3.0) if simulated else args.max_seconds,
                                          accept_when_ready=args.accept_when_ready or simulated, log=log)
            if decision in ("timeout", "quit") and args.force_record and session.last_measurement is not None:
                decision = "forced"
        if decision in ("quit", "timeout"):
            log("Nothing recorded / 未记录。" + (" Time limit reached." if decision == "timeout" else ""))
            return EXIT_NOT_RECORDED
        if decision == "forced" and not session.last_measurement["gate"]["clean"]:
            log("Refusing to record a field that is not clean bare substrate / 视野不是干净裸衬底，拒绝记录")
            return EXIT_NOT_RECORDED
        operator = ask_operator(interactive=not args.no_prompts and not simulated)
        record = write_record(args.out, session=session, reference_path=args.reference, mode=mode,
                              accepted_by={"auto": "automatic_within_tolerance", "accepted": "operator_key",
                                           "forced": "operator_forced_outside_tolerance"}[decision],
                              adjustment=adjustment, operator=operator, simulated=simulated)
        log(f"Recorded / 已记录: {record['folder']}")
        log(f"  frame sha256 {record['frame']['sha256'][:16]}…  calibrated={record['calibrated']}  verdict={record['verdict']}")
        if record["calibrated"]:
            log("Camera colour balance is on the reference. Keep lamp, exposure and white balance unchanged and start the scan "
                "/ 白平衡已回到参考：保持灯光、曝光、白平衡不动，开始扫描")
            return EXIT_OK
        log("Recorded, but NOT within tolerance: the analysis colour check will flag these images for review "
            "/ 已记录但未达到参考：分析时颜色检查会提示并标为待复核")
        return EXIT_NOT_WITHIN_TOLERANCE
    except KeyboardInterrupt:
        log("Interrupted; nothing recorded / 已中断，未记录")
        return EXIT_NOT_RECORDED
    except Exception as exc:
        log(f"Error / 错误: {exc}")
        return EXIT_ERROR
    finally:
        if cam is not None:
            try:
                cam.stop_live()
                cam.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
