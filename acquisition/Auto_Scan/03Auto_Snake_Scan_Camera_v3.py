# -*- coding: utf-8 -*-
"""
XY Mosaic Navigator + Auto Camera Capture v3 (student verification build)
XY 蛇形扫描 + 自动拍照

Merged from:
  - 02Auto_Snake_Scan.py (XY stage control, snake scan GUI)
  - auto_scan_v8.py (camera capture, frame flush, blur detection)

Features:
  - Connect / Disconnect XY stage
  - Four-direction manual move buttons (one photo step per click)
  - Overlap-based dx/dy auto calculation
  - Auto snake scan with NX x NY grid
  - AUTOMATIC photo capture at each point (no manual shutter needed)
  - Frame flush to clear HDMI buffer lag
  - Blur detection + wait-and-recapture (no active Z-axis autofocus)
  - Embedded live preview and frozen review in the same window
  - Photos saved to timestamped session folder
  - Emergency stop button
  - Background worker thread (GUI stays responsive during scan)
  - Lockfile to prevent COM port conflicts
"""

import ctypes
import time
import os
import sys
import platform
import threading
import signal
import atexit
from pathlib import Path
from camera_session_history import open_detected_camera, describe_camera, ScanSession
import argparse
import base64
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from scan_review import ReviewGate, compare_views
from scan_preview import PreviewPump
from shared_history import SharedHistory
from history_viewer import show_history

import numpy as np

# The Git checkout keeps the vendor SDK one directory above Auto_Scan;
# the standalone student package keeps it beside this script. Search either
# layout before Camera_v2 performs its SDK detection. This opens no device.
_sdk_parent = Path(__file__).resolve().parent.parent
if any((_sdk_parent / name).is_file() for name in ("amcam.py", "toupcam.py")):
    if str(_sdk_parent) not in sys.path:
        sys.path.append(str(_sdk_parent))

# Try to import Camera and OpenCV
_HAS_CAMERA = False
_HAS_CV2 = False
try:
    from Camera_v2 import Camera
    _HAS_CAMERA = True
except ImportError:
    try:
        from Camera import Camera
        _HAS_CAMERA = True
        print("[warn] Using legacy Camera module (Camera_v2 not found).")
    except ImportError:
        print("[warn] Camera module not found. Put Camera_v2.py in this folder.")

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    print("[warn] OpenCV not installed. Preview/blur detection disabled.")

# ============================================================
# Configuration / 配置区
# ============================================================

# -- XY Stage --
DEVICE_X = b"xi-com:\\\\.\\COM7"
DEVICE_Y = b"xi-com:\\\\.\\COM10"

# Stage calibration
FULLSTEP_UM = 2.5
PX_PER_STEP = 25.6  # legacy default, not verified for the currently connected camera

# Microstep mode (1=full step, 0x09=1/256 for ICMM)
MICROSTEP_MODE = 1

# Field of view in steps (for overlap calculation)
FOV_STEPS_X = 151.04
FOV_STEPS_Y = 84.375

# Direction conventions
INVERT_X = True
INVERT_Y = False

# Internal step gap (0 = no delay between individual steps)
STEP_GAP_S = 0.0

# Simulation mode (no hardware needed)
SIMULATE_STAGE = False

# -- Camera --
CAMERA_INDEX = 0
CAMERA_BACKEND = "auto"
BITS = 24
CAMERA_RESOLUTION = (1920, 1080)  # 954-1080p-A1: request and verify received pixels

SAVE_FOLDER = "mosaic_photos"
SAVE_FORMAT = ".png"
JPEG_QUALITY = 95

# Settle time before capture (let stage vibrations damp)
SETTLE_TIME_S = 0.5

# Frame flush (discard old buffered frames from HDMI capture card)
ENABLE_FRAME_FLUSH = True
BUFFER_FLUSH_FRAMES = 10

# Blur detection
ENABLE_BLUR_CHECK = True
BLUR_THRESHOLD = 37.0
FOCUS_MAX_RETRIES = 3
FOCUS_RETRY_DELAY_S = 0.8

# Exposure, gain and white balance are read and recorded, not blindly written.
# Keep microscope illumination and hardware controls stable during a scan.

# ============================================================
# DLL search (aligned with main pipeline)
# ============================================================

def _register_dll_dir(dll_path):
    """Register the directory containing the DLL for Windows DLL search."""
    dll_dir = os.path.dirname(os.path.abspath(dll_path))
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(dll_dir)
        except Exception:
            pass
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    return dll_path


def _default_ximc_dll_path():
    """Find libximc.dll in drivers/ subfolder next to this program."""
    _script = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(_script, "drivers", "libximc.dll")
    if os.path.exists(candidate):
        return _register_dll_dir(candidate)
    return ""


# ============================================================
# Handle utilities (aligned with main pipeline)
# ============================================================

def _handle_value(h):
    """Extract integer from a ctypes handle (c_void_p or c_int)."""
    if h is None:
        return 0
    if hasattr(h, "value"):
        v = h.value
        return int(v) if v is not None else 0
    return int(h)


def _is_invalid_handle(h):
    """Check if an XIMC device handle is invalid."""
    v = _handle_value(h)
    return v in {0, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF}


# ============================================================
# Utility functions
# ============================================================

def compute_dxdy(overlap_x, overlap_y):
    dx = int(round(FOV_STEPS_X - overlap_x))
    dy = int(round(FOV_STEPS_Y - overlap_y))
    return dx, dy

def steps_to_um(s): return s * FULLSTEP_UM
def steps_to_px(s): return s * PX_PER_STEP

def sgn(v):
    if v > 0: return 1
    if v < 0: return -1
    return 0

def safe_int(v, default=0):
    try: return int(float(v))
    except Exception: return default

def safe_float(v, default=0.5):
    try: return float(v)
    except Exception: return default


def _interruptible_sleep(seconds, cancel_event=None):
    """Sleep in small chunks so cancel_event can interrupt."""
    if seconds <= 0:
        return
    chunk = 0.05
    elapsed = 0.0
    while elapsed < seconds:
        if cancel_event and cancel_event.is_set():
            return
        remaining = seconds - elapsed
        time.sleep(min(chunk, remaining))
        elapsed += chunk


# ============================================================
# Lockfile (prevent COM port conflicts)
# ============================================================

_LOCKFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mosaic_stage.lock")


_LOCK_OWNED = False


def _write_lockfile():
    """Exclusive ownership; never terminate another process or silently steal its lock."""
    global _LOCK_OWNED
    try:
        fd = os.open(_LOCKFILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "Another scanner may be running. Close it first. If it crashed, "
            "confirm it has stopped before manually removing: " + _LOCKFILE) from exc
    with os.fdopen(fd, "w") as stream:
        stream.write(str(os.getpid()))
    _LOCK_OWNED = True


def _remove_lockfile():
    global _LOCK_OWNED
    if not _LOCK_OWNED:
        return
    try:
        with open(_LOCKFILE, "r") as stream:
            owner = stream.read().strip()
        if owner == str(os.getpid()):
            os.remove(_LOCKFILE)
    except FileNotFoundError:
        pass
    finally:
        _LOCK_OWNED = False


# ============================================================
# XIMC Stage (with proper handle safety)
# ============================================================

class XIMCStage:
    """Verify controller step counters, never infer physical travel from them.

    ctypes declarations below were checked against official libximc 3.0.2
    lowlevel/_lowlevel.py from https://pypi.org/project/libximc/3.0.2/.
    Wheel SHA256: a92b2e52dff361ff92cc304957bb3231c877849038e99509d3affa779a8732fd.
    The bundled win64 DLL is byte-identical to that wheel's DLL (SHA256:
    2f99e09a4ea2cea72ec3d595fab9cefcc3eddac36d350aa7e39b2c8742f200e9).
    The earlier local 2.8.8 ximc.h agrees on these C signatures. In particular,
    device_t is int, close_device takes int*, and long_t is a 64-bit integer.
    A matching SDK is required before reading structures from the real DLL.
    """

    SDK_HEADER_VERSION = "3.0.2"
    POLL_INTERVAL_S = 0.01
    MOVE_DEADLINE_S = 30.0
    # Long mapping stability: retry read-only SDK queries after transient rc != 0.
    # This does NOT retry movement commands (command_movr).
    READ_RETRIES = 3                 # retries after the initial read (4 attempts max)
    READ_RETRY_BASE_DELAY_S = 0.10   # 0.10 s, 0.20 s, 0.30 s backoff
    MVCMD_RUNNING = 0x80
    MVCMD_ERROR = 0x40
    MOVE_STATE_MOVING = 0x01
    STATUS_FAULT_MASK = 0x1B3FFC7  # 3.0.2 STATE_SECUR plus ERRC/ERRD/ERRV

    class Position(ctypes.Structure):
        _fields_ = [("Position", ctypes.c_int), ("uPosition", ctypes.c_int),
                    ("EncPosition", ctypes.c_longlong)]

    class EngineSettings(ctypes.Structure):
        _fields_ = [("NomVoltage", ctypes.c_uint), ("NomCurrent", ctypes.c_uint),
                    ("NomSpeed", ctypes.c_uint), ("uNomSpeed", ctypes.c_uint),
                    ("EngineFlags", ctypes.c_uint), ("Antiplay", ctypes.c_int),
                    ("MicrostepMode", ctypes.c_uint), ("StepsPerRev", ctypes.c_uint)]

    class EngineType(ctypes.Structure):
        _fields_ = [("EngineType", ctypes.c_uint), ("DriverType", ctypes.c_uint)]

    class Status(ctypes.Structure):
        _fields_ = [("MoveSts", ctypes.c_uint), ("MvCmdSts", ctypes.c_uint),
                    ("PWRSts", ctypes.c_uint), ("EncSts", ctypes.c_uint),
                    ("WindSts", ctypes.c_uint), ("CurPosition", ctypes.c_int),
                    ("uCurPosition", ctypes.c_int), ("EncPosition", ctypes.c_longlong),
                    ("CurSpeed", ctypes.c_int), ("uCurSpeed", ctypes.c_int),
                    ("Ipwr", ctypes.c_int), ("Upwr", ctypes.c_int),
                    ("Iusb", ctypes.c_int), ("Uusb", ctypes.c_int),
                    ("CurT", ctypes.c_int), ("Flags", ctypes.c_uint),
                    ("GPIOFlags", ctypes.c_uint), ("CmdBufFreeSpace", ctypes.c_uint)]

    def __init__(self, simulate=False):
        self.lib = None
        self.dev_x = None
        self.dev_y = None
        self.x = 0
        self.y = 0
        self._simulate = simulate
        self._connected = False
        # This flag means continuity of controller counters, not encoder truth.
        self.position_trusted = False
        self._motion_lock = threading.RLock()
        # Serialize individual SDK calls and stop/dispatch ordering, never a
        # whole wait loop. Stop sets its event before waiting for this lock.
        self._command_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._last_snapshot = None
        # Engine type/settings are static during a scan. Read them once at connect
        # and reuse them for runtime snapshots instead of querying every point.
        self._static_axis_config = None
        self.last_move_evidence = None
        self.last_position_evidence = None
        self.stage_info = {"mode": "simulation" if simulate else "hardware",
                           "verification_basis": "simulated" if simulate else "controller_step_counter_only",
                           "physical_travel_verified": False,
                           "sdk_header_version": self.SDK_HEADER_VERSION}

    @staticmethod
    def _valid_device(dev):
        if dev is None:
            return False
        value = _handle_value(dev)
        return 0 <= value <= 0x7FFFFFFF

    @staticmethod
    def _fields(value):
        return {name: int(getattr(value, name)) for name, _ in value._fields_}

    def _bind_sdk(self):
        signatures = {
            "open_device": ([ctypes.c_char_p], ctypes.c_int),
            "close_device": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "command_movr": ([ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_int),
            "command_sstp": ([ctypes.c_int], ctypes.c_int),
            "get_position": ([ctypes.c_int, ctypes.POINTER(self.Position)], ctypes.c_int),
            "get_status": ([ctypes.c_int, ctypes.POINTER(self.Status)], ctypes.c_int),
            "get_engine_settings": ([ctypes.c_int, ctypes.POINTER(self.EngineSettings)], ctypes.c_int),
            "get_entype_settings": ([ctypes.c_int, ctypes.POINTER(self.EngineType)], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result

    def _read(self, name, dev, kind):
        """Read a controller structure with bounded retries for transient SDK errors.

        Only read-only SDK calls come through this helper. Movement commands are
        deliberately NOT retried here, so the original motion-safety policy is kept.
        """
        if not self._valid_device(dev):
            raise RuntimeError("Invalid controller device handle")

        last_rc = None
        total_attempts = 1 + int(self.READ_RETRIES)
        for attempt in range(1, total_attempts + 1):
            value = kind()
            with self._command_lock:
                rc = getattr(self.lib, name)(_handle_value(dev), ctypes.byref(value))
            if rc == 0:
                if attempt > 1:
                    print(f"[stage] {name} recovered on attempt {attempt}/{total_attempts}")
                return self._fields(value)

            last_rc = rc
            if attempt < total_attempts:
                delay = self.READ_RETRY_BASE_DELAY_S * attempt
                print(f"[stage] WARNING: {name} failed: rc={rc}; "
                      f"retry {attempt}/{self.READ_RETRIES} after {delay:.2f}s")
                time.sleep(delay)

        raise RuntimeError(
            f"{name} failed after {total_attempts} attempts: rc={last_rc}")

    def _status(self, dev, require_stopped=False):
        status = self._read("get_status", dev, self.Status)
        if status["Flags"] & self.STATUS_FAULT_MASK:
            raise RuntimeError(f"Controller fault flags: 0x{status['Flags']:X}")
        if status["MvCmdSts"] & self.MVCMD_ERROR:
            raise RuntimeError(f"Controller movement error: MvCmdSts=0x{status['MvCmdSts']:X}")
        if require_stopped and (status["MvCmdSts"] & self.MVCMD_RUNNING or
                                status["MoveSts"] & self.MOVE_STATE_MOVING):
            raise RuntimeError("Controller is already moving; counter baseline is unavailable")
        return status

    def _snapshot(self, refresh_static=False):
        """Read a controller snapshot.

        Engine settings/type are treated as static for one connection and are
        queried only during connect (or an explicit refresh). Runtime snapshots
        still read live status and position for every safety/continuity check.
        """
        if refresh_static or self._static_axis_config is None:
            static = {}
            for axis, dev in (("x", self.dev_x), ("y", self.dev_y)):
                engine = self._read("get_engine_settings", dev, self.EngineSettings)
                engine_type = self._read("get_entype_settings", dev, self.EngineType)
                if engine_type["EngineType"] != 3:
                    raise RuntimeError(
                        f"{axis.upper()} is not configured as a stepper motor; step units unknown")
                mode = engine["MicrostepMode"]
                if mode not in range(1, 10):
                    raise RuntimeError(f"{axis.upper()} unknown microstep mode: {mode}")
                static[axis] = {
                    "engine_settings": engine,
                    "engine_type": engine_type,
                    "microsteps_per_step": 1 << (mode - 1),
                }
            self._static_axis_config = static

        snapshot = {}
        for axis, dev in (("x", self.dev_x), ("y", self.dev_y)):
            status = self._status(dev, require_stopped=True)
            config = self._static_axis_config[axis]
            # Copy dicts so evidence snapshots do not share mutable containers.
            engine = dict(config["engine_settings"])
            engine_type = dict(config["engine_type"])
            divisor = int(config["microsteps_per_step"])
            position = self._read("get_position", dev, self.Position)
            if abs(position["uPosition"]) >= divisor:
                raise RuntimeError(f"{axis.upper()} invalid fractional step readback: {position}")
            snapshot[axis] = {"position": position, "engine_settings": engine,
                              "engine_type": engine_type, "microsteps_per_step": divisor,
                              "counter_microsteps": position["Position"] * divisor + position["uPosition"],
                              "status": status}
            self._status(dev, require_stopped=True)
        return snapshot

    def _check_continuity(self, snapshot, expected_deltas=None):
        if self._last_snapshot is None:
            raise RuntimeError("Controller counter baseline unavailable")
        expected_deltas = expected_deltas or {}
        for axis in ("x", "y"):
            before, after = self._last_snapshot[axis], snapshot[axis]
            if (after["engine_settings"] != before["engine_settings"] or
                    after["engine_type"] != before["engine_type"]):
                raise RuntimeError(f"{axis.upper()} controller settings changed; units must be checked")
            expected = expected_deltas.get(axis, 0) * before["microsteps_per_step"]
            observed = after["counter_microsteps"] - before["counter_microsteps"]
            if observed != expected:
                raise RuntimeError(f"{axis.upper()} counter mismatch: expected {expected} microsteps, "
                                   f"observed {observed}; physical field must be checked")

    def _check_cancelled(self, cancel_event=None):
        if self._stop_event.is_set() or (cancel_event and cancel_event.is_set()):
            raise RuntimeError("Movement interrupted; actual position must be checked")

    def _close_handles(self):
        for axis in ("x", "y"):
            dev = getattr(self, "dev_" + axis)
            if not self._valid_device(dev):
                continue
            handle = ctypes.c_int(_handle_value(dev))
            try:
                with self._command_lock:
                    rc = self.lib.close_device(ctypes.byref(handle))
                if rc != 0:
                    raise RuntimeError(f"rc={rc}")
            except Exception as exc:
                print(f"[stage] Could not close {axis.upper()} controller: {exc}")
            finally:
                setattr(self, "dev_" + axis, None)

    def connect(self):
        if self._connected:
            return
        self.position_trusted = False
        self._stop_event.clear()
        self._static_axis_config = None
        if self._simulate:
            print("[stage] SIMULATION MODE — no hardware")
            self._connected = True
            self.position_trusted = True
            self.x = self.y = 0
            return

        if platform.system() != "Windows":
            raise RuntimeError("XIMC stage requires Windows")

        dll_path = _default_ximc_dll_path()
        if not dll_path:
            raise RuntimeError(
                "libximc.dll not found.\n"
                "Put the verified libximc SDK DLLs in drivers/ beside this script."
            )

        import hashlib
        with open(dll_path, "rb") as dll_file:
            self.stage_info["dll_sha256"] = hashlib.sha256(dll_file.read()).hexdigest()
        self.lib = ctypes.WinDLL(dll_path)
        # Refuse an unreviewed ABI before passing output buffers to the DLL.
        version = ctypes.create_string_buffer(64)
        self.lib.ximc_version.argtypes = [ctypes.POINTER(ctypes.c_char)]
        self.lib.ximc_version.restype = None
        self.lib.ximc_version(version)
        sdk_version = version.value.decode("ascii", errors="replace")
        self.stage_info.update(dll_path=dll_path, sdk_version=sdk_version)
        if sdk_version != self.SDK_HEADER_VERSION:
            raise RuntimeError(f"Unverified libximc version {sdk_version!r}; bindings match "
                               f"SDK {self.SDK_HEADER_VERSION}. Review matching SDK before use.")
        self._bind_sdk()

        # Try opening with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.dev_x = self.lib.open_device(DEVICE_X)
                self.dev_y = self.lib.open_device(DEVICE_Y)
            except Exception:
                self._close_handles()
                raise

            if self._valid_device(self.dev_x) and self._valid_device(self.dev_y):
                break

            # Clean up partial opens
            self._close_handles()
            self.dev_x = self.dev_y = None

            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                print(f"[stage] COM ports may be locked. Waiting {wait}s before retry... ({attempt+1}/{max_retries})")
                time.sleep(wait)

        if not self._valid_device(self.dev_x) or not self._valid_device(self.dev_y):
            raise RuntimeError(
                f"Failed to open stage devices after {max_retries} attempts.\n"
                f"X handle: {_handle_value(self.dev_x)}, Y handle: {_handle_value(self.dev_y)}\n"
                f"Check COM ports and power supply."
            )

        try:
            # Full static configuration is read once here and cached for this connection.
            self._last_snapshot = self._snapshot(refresh_static=True)
        except Exception:
            self._close_handles()
            raise
        self.stage_info["initial_controller_snapshot"] = self._last_snapshot
        self.x = self.y = 0
        self._connected = True
        self.position_trusted = True
        print("[stage] Connected; controller step counters verified. Physical travel is unverified.")

    def disconnect(self):
        self.position_trusted = False
        if self._simulate:
            self._connected = False
            print("[stage] Simulation disconnected")
            return
        # Stop motors first
        self.emergency_stop(wait=True)
        # Wait for our motion worker to leave the DLL before closing its handles.
        with self._motion_lock:
            self._close_handles()
        self.dev_x = self.dev_y = None
        self._static_axis_config = None
        self._connected = False
        print("[stage] Disconnected")

    def emergency_stop(self, wait=False):
        """Stop both motors immediately."""
        self._stop_event.set()
        self.position_trusted = False
        if self._simulate or not self.lib:
            return
        with self._command_lock:
            for name, dev in [("X", self.dev_x), ("Y", self.dev_y)]:
                if not self._valid_device(dev):
                    continue
                try:
                    rc = self.lib.command_sstp(_handle_value(dev))
                    if rc != 0:
                        raise RuntimeError(f"command_sstp rc={rc}")
                except Exception as e:
                    print(f"[stage] Failed to stop {name}: {e}")
        if wait:
            time.sleep(0.3)

    def _wait_for_stop(self, dev, cancel_event=None, evidence=None):
        """Poll command status with a separate application deadline.

        command_wait_for_stop's second argument is a poll interval, NOT a
        timeout. Native SDK calls themselves must return for cancellation or
        this application deadline to be serviced.
        """
        deadline = time.monotonic() + self.MOVE_DEADLINE_S
        while True:
            self._check_cancelled(cancel_event)
            status = self._status(dev)
            if evidence is not None:
                evidence["last_status"] = status
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Movement deadline exceeded ({self.MOVE_DEADLINE_S:g}s)")
            if not (status["MvCmdSts"] & self.MVCMD_RUNNING or
                    status["MoveSts"] & self.MOVE_STATE_MOVING):
                return status
            self._stop_event.wait(min(self.POLL_INTERVAL_S, max(0.0, deadline - time.monotonic())))

    def verify_position(self):
        """Check stationary counter continuity before capture; no motor commands."""
        with self._motion_lock:
            evidence = {"mode": self.stage_info["mode"],
                        "verification_basis": self.stage_info["verification_basis"],
                        "physical_travel_verified": False, "result": "unverified"}
            self.last_position_evidence = evidence
            try:
                if not self._connected or not self.position_trusted:
                    raise RuntimeError("Stage position uncertain; reconnect and check the field")
                self._check_cancelled()
                if self._simulate:
                    evidence["result"] = "simulated"
                    return evidence
                snapshot = self._snapshot()
                evidence["controller_snapshot"] = snapshot
                self._check_continuity(snapshot)
                self._check_cancelled()
                evidence["result"] = "controller_counter_verified"
                return evidence
            except Exception as exc:
                evidence.update(result="failed", error=str(exc))
                self.position_trusted = False
                self.emergency_stop(wait=False)
                raise

    def move_relative(self, dx_steps, dy_steps, cancel_event=None):
        """Advance logical coordinates only after exact controller delta readback.

        An unchanged/wrong counter, controller fault or unavailable readback
        aborts the scan. Never retry a move or change engine settings here.
        EncPosition is recorded as raw evidence; its physical meaning is not
        established by this program.
        """
        with self._motion_lock:
            evidence = {"mode": self.stage_info["mode"],
                        "verification_basis": self.stage_info["verification_basis"],
                        "physical_travel_verified": False,
                        "requested_logical_steps": {"x": dx_steps, "y": dy_steps},
                        "axes": [], "result": "unverified"}
            self.last_move_evidence = evidence
            try:
                if not self._connected or not self.position_trusted:
                    raise RuntimeError("Stage position uncertain. Reconnect and check the field before a NEW scan")
                self._check_cancelled(cancel_event)
                for delta in (dx_steps, dy_steps):
                    if int(delta) != delta or abs(delta) > 0x7FFFFFFF:
                        raise ValueError("Movement requires whole steps within the signed 32-bit range")
                dx_steps, dy_steps = int(dx_steps), int(dy_steps)
                if self._simulate:
                    self.x += dx_steps
                    self.y += dy_steps
                    evidence["result"] = "simulated"
                    return
                before = self._snapshot()
                evidence["before"] = before
                self._check_continuity(before)
                for axis, delta, invert, dev in (
                    ("x", dx_steps, INVERT_X, self.dev_x),
                    ("y", dy_steps, INVERT_Y, self.dev_y),
                ):
                    self._check_cancelled(cancel_event)
                    if delta == 0:
                        continue
                    command = -delta if invert else delta
                    axis_evidence = {"axis": axis, "command_steps": command,
                                     "before": before[axis],
                                     "expected_delta_microsteps": command * before[axis]["microsteps_per_step"]}
                    evidence["axes"].append(axis_evidence)
                    with self._command_lock:
                        self._check_cancelled(cancel_event)
                        rc = self.lib.command_movr(_handle_value(dev), command, 0)
                    axis_evidence["command_result"] = rc
                    if rc != 0:
                        raise RuntimeError(f"{axis.upper()} movement failed: rc={rc}")
                    self._wait_for_stop(dev, cancel_event, axis_evidence)
                    self._check_cancelled(cancel_event)
                    after = self._snapshot()
                    evidence["after"] = after
                    axis_evidence["after"] = after[axis]
                    axis_evidence["observed_delta_microsteps"] = (
                        after[axis]["counter_microsteps"] - before[axis]["counter_microsteps"])
                    self._check_continuity(after, {axis: command})
                    self._check_cancelled(cancel_event)
                    self._last_snapshot = before = after
                    setattr(self, axis, getattr(self, axis) + delta)
                    axis_evidence["result"] = "controller_counter_verified"
                evidence["result"] = "controller_counter_verified"
            except Exception as exc:
                evidence.update(result="failed", error=str(exc))
                self.position_trusted = False
                self.emergency_stop(wait=False)
                raise RuntimeError(f"{exc}; no photo taken") from exc

    @property
    def connected(self):
        return self._connected


# ============================================================
# Camera helper functions
# ============================================================

def _grab_frame_flushed(cam, timeout=5.0, flush_frames=0):
    """Grab a frame, optionally flushing old buffered frames first."""
    flush_frames = int(max(0, flush_frames))
    if flush_frames <= 0:
        return cam.grab_frame(timeout=timeout)

    # Try native flush parameter
    try:
        return cam.grab_frame(timeout=timeout, flush=flush_frames)
    except TypeError:
        pass

    # Fallback: manually drain old frames
    deadline = time.monotonic() + timeout
    for _ in range(flush_frames):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or cam.grab_frame(timeout=remaining) is None:
            return None
    remaining = deadline - time.monotonic()
    return cam.grab_frame(timeout=remaining) if remaining > 0 else None


def _sharpness_score(frame):
    """Measure sharpness of center structure via Sobel gradient."""
    if not _HAS_CV2 or frame is None:
        return float("inf")

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray = gray.astype(np.float64)
    h, w = gray.shape

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    full_gmax = grad.max()

    if len(frame.shape) != 3:
        return full_gmax

    # Detect structures via saturation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) > 2000]
    if not big:
        return full_gmax

    cx_img = w // 2
    # Cache moments to avoid double computation
    moments = [(c, cv2.moments(c)) for c in big]
    best_c, _ = min(moments, key=lambda cm: abs(
        int(cm[1]['m10'] / max(cm[1]['m00'], 1)) - cx_img))

    cmask = np.zeros((h, w), np.uint8)
    cv2.drawContours(cmask, [best_c], -1, 255, -1)
    k3 = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(cmask, k3, iterations=3)
    eroded = cv2.erode(cmask, k3, iterations=3)
    edge = (dilated - eroded) > 0

    if edge.sum() < 10:
        return full_gmax

    return float(grad[edge].max())


def capture_and_save(cam, filepath, cancel_event=None, skip_settle=False):
    """Capture a frame with flush + blur retry, save to filepath.

    Returns (success, sharpness, retry_count).
    """
    if not skip_settle:
        _interruptible_sleep(SETTLE_TIME_S, cancel_event)

    if cancel_event and cancel_event.is_set():
        return False, 0, 0
    flush = BUFFER_FLUSH_FRAMES if ENABLE_FRAME_FLUSH else 0
    frame = _grab_frame_flushed(cam, timeout=5.0, flush_frames=flush)
    if frame is None:
        print(f"[camera] ERROR: got None frame")
        return False, 0, 0

    sharpness = _sharpness_score(frame)
    retry = 0

    if ENABLE_BLUR_CHECK and _HAS_CV2:
        while sharpness < BLUR_THRESHOLD and retry < FOCUS_MAX_RETRIES:
            if cancel_event and cancel_event.is_set():
                break
            retry += 1
            print(f"[focus] Blurry (sharpness={sharpness:.1f} < {BLUR_THRESHOLD}), "
                  f"retry {retry}/{FOCUS_MAX_RETRIES}...")
            _interruptible_sleep(FOCUS_RETRY_DELAY_S, cancel_event)
            frame = _grab_frame_flushed(cam, timeout=5.0, flush_frames=flush)
            if frame is None:
                return False, sharpness, retry
            sharpness = _sharpness_score(frame)

        if sharpness < BLUR_THRESHOLD:
            print(f"[focus] WARNING: still blurry after {FOCUS_MAX_RETRIES} retries")

    if cancel_event and cancel_event.is_set():
        return False, sharpness, retry
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        if os.path.exists(filepath):
            raise RuntimeError("Refusing to overwrite an existing raw image: " + filepath)
        temp_path = filepath + ".partial" + (os.path.splitext(filepath)[1] or ".png")
        Camera.save_image(frame, temp_path, quality=JPEG_QUALITY)
        os.replace(temp_path, filepath)
    except Exception as e:
        print(f"[camera] Save failed: {e}")
        return False, sharpness, retry
    return True, sharpness, retry


# ============================================================
# GUI
# ============================================================

class MosaicNavigatorGUI(tk.Tk):
    def __init__(self, simulate=False):
        super().__init__()
        self.title("显微扫描工作台 v3.4 / 自动走位 + 人工确认 + 共享历史")
        width = min(1280, self.winfo_screenwidth() - 60)
        height = min(900, self.winfo_screenheight() - 90)
        self.geometry(f"{width}x{height}")
        self.minsize(980, 610)

        self.stage = XIMCStage(simulate=simulate)
        self.cam = None
        self._camera_lock = threading.Lock()  # serialize all camera access
        self.connected_camera = False
        self._camera_connecting = False
        self._disconnect_error = None
        self._camera_info = {}
        self._camera_cycle_excluded = set()
        self._session_log = None
        self._shared_history = None
        self._history_error = None
        self._history_help_shown = False
        self._device_epoch = 0
        self._log_resume_plan = None
        self._log_resume_devices = None
        self._log_pause_position = None
        self._point_checkpoint = None
        self._log_recovery_busy = False
        self._log_recovery_review = False
        self._log_recovery_cancelled = False
        self._history_root = Path(__file__).resolve().parent / "shared_history"
        self._scan_origin = (0, 0)
        self._run_auto_capture = True
        self.auto_running = False
        self._worker = None
        self._worker_error = None  # store worker exception
        self._cancel_event = threading.Event()
        self._preview_after_id = None
        self._preview_running = False
        self._preview_generation = 0
        self._live_last_update = None
        self._preview_pump = PreviewPump(self._camera_lock)
        self._review_gate = ReviewGate()
        self._review_active = False
        self._review_token = None
        self._run_review_mode = "step_confirm"
        self._last_accepted_photo = None
        self._human_review_count = 0

        # Session folder for photos
        self._session_folder = None
        self._photo_count = 0
        self._sharpness_records = []


        # Resume state: track where we stopped
        self._scan_done_count = 0      # how many points completed
        self._scan_params = None       # (nx, ny, dx_step, dy_step, order) for resume

        # -- Variables --
        self.var_sample_id = tk.StringVar(value="SP_TEST")
        self.var_objective = tk.StringVar(value="")
        self.var_photo_dwell = tk.StringVar(value="0.50")
        self.var_overlap_x = tk.StringVar(value="8")
        self.var_overlap_y = tk.StringVar(value="8")
        self.var_dx = tk.StringVar(value="40")
        self.var_dy = tk.StringVar(value="40")
        self.var_nx = tk.StringVar(value="2")
        self.var_ny = tk.StringVar(value="1")
        self.var_order = tk.StringVar(value="X_first")
        self.var_auto_capture = tk.BooleanVar(value=True)
        self.var_review_mode = tk.StringVar(value="step_confirm")

        self.var_status = tk.StringVar(value="IDLE")
        self.var_progress = tk.StringVar(value="")
        self.var_history_status = tk.StringVar(value="查看历史")
        self.var_pos_steps = tk.StringVar()
        self.var_pos_um = tk.StringVar()
        self.var_pos_px = tk.StringVar()

        self._build_ui()
        from gui_language import install_language_switch
        self._language_ui = install_language_switch(self)
        self._refresh_pos()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        """One window: compact controls, live field, and frozen review images."""
        self.configure(background="#edf2f5")
        style = ttk.Style(self)
        style.theme_use("clam")
        font = "Microsoft YaHei UI" if platform.system() == "Windows" else "PingFang SC"
        style.configure("TFrame", background="#edf2f5")
        style.configure("TLabel", background="#edf2f5", foreground="#243c49", font=(font, 10))
        style.configure("TLabelframe", background="#edf2f5", bordercolor="#cbd8df")
        style.configure("TLabelframe.Label", background="#edf2f5", foreground="#284858", font=(font, 10, "bold"))
        style.configure("TButton", font=(font, 10), padding=(7, 3))
        style.configure("TRadiobutton", background="#edf2f5", font=(font, 10))
        style.configure("Accent.TButton", background="#14747b", foreground="white", padding=(10, 5))
        style.map("Accent.TButton", background=[("disabled", "#cbdadc"), ("active", "#105b62")],
                  foreground=[("disabled", "#657d82")])
        style.configure("Stop.TButton", background="#a83536", foreground="white")
        style.map("Stop.TButton", background=[("active", "#822829")])
        style.configure("Header.TLabel", font=(font, 16, "bold"))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(14, 10, 14, 6))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="显微扫描工作台", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="v3.4 · 看图确认后再走下一点").pack(side=tk.LEFT, padx=12)
        ttk.Button(header, textvariable=self.var_history_status, command=self.on_show_history).pack(side=tk.LEFT)
        self.btn_estop = ttk.Button(header, text="紧急停止", style="Stop.TButton", command=self.on_emergency_stop)
        self.btn_estop.pack(side=tk.RIGHT)
        self.btn_resume = ttk.Button(header, text="检查并继续", command=self.resume_auto, state="disabled")
        self.btn_resume.pack(side=tk.RIGHT, padx=8)
        self.btn_auto = ttk.Button(header, text="开始扫描", style="Accent.TButton", command=self.start_auto)
        self.btn_auto.pack(side=tk.RIGHT)

        body = ttk.Frame(self, padding=(12, 2, 12, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        rail = ttk.Frame(body, width=286)
        rail.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        rail.grid_propagate(False)
        rail.columnconfigure(0, weight=1)
        rail.rowconfigure(3, weight=1)

        conn = ttk.Labelframe(rail, text="1  连接设备", padding=8)
        conn.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for c in (0, 1): conn.columnconfigure(c, weight=1)
        ttk.Button(conn, text="连接相机", command=self.on_connect_camera).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(conn, text="连接位移台", command=self.on_connect_stage).grid(row=0, column=1, sticky="ew")
        ttk.Button(conn, text="换一个相机", command=self.on_next_camera).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(4, 0))
        ttk.Button(conn, text="断开设备", command=self.on_disconnect).grid(row=1, column=1, sticky="ew", pady=(4, 0))
        self.var_connection = tk.StringVar(value="相机未连接 · 位移台未连接")
        ttk.Label(conn, textvariable=self.var_connection, wraplength=246).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))

        params = ttk.Labelframe(rail, text="2  样品与扫描", padding=8)
        params.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        params.columnconfigure(1, weight=1)
        for r, title, variable in ((0, "样品编号", self.var_sample_id), (1, "物镜", self.var_objective)):
            ttk.Label(params, text=title).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(params, textvariable=variable, width=19).grid(row=r, column=1, sticky="ew", padx=(8, 0), pady=2)
        grid = ttk.Frame(params)
        grid.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        for r, items in enumerate((("列 NX", self.var_nx, "行 NY", self.var_ny),
                                   ("X 步长", self.var_dx, "Y 步长", self.var_dy))):
            for c, title, variable in ((0, items[0], items[1]), (2, items[2], items[3])):
                ttk.Label(grid, text=title).grid(row=r, column=c, sticky="w", pady=3)
                ttk.Entry(grid, textvariable=variable, width=5).grid(row=r, column=c+1, padx=(4, 8), pady=3)
        order = ttk.Frame(params); order.grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(order, text="先沿 X", variable=self.var_order, value="X_first").pack(side=tk.LEFT)
        ttk.Radiobutton(order, text="先沿 Y", variable=self.var_order, value="Y_first").pack(side=tk.LEFT, padx=8)
        mode = ttk.Frame(params); mode.grid(row=4,column=0,columnspan=2,sticky="w",pady=(4,0))
        ttk.Radiobutton(mode, text="逐点确认", variable=self.var_review_mode,
                        value="step_confirm").pack(side=tk.LEFT)
        ttk.Radiobutton(mode, text="连续采集", variable=self.var_review_mode,
                        value="continuous").pack(side=tk.LEFT,padx=8)

        overlap = ttk.Frame(params)
        overlap.grid(row=5,column=0,columnspan=2,sticky="ew",pady=(7,0))
        ttk.Label(overlap,text="重叠 X").pack(side=tk.LEFT)
        ttk.Entry(overlap,textvariable=self.var_overlap_x,width=4).pack(side=tk.LEFT,padx=3)
        ttk.Label(overlap,text="Y").pack(side=tk.LEFT,padx=(5,0))
        ttk.Entry(overlap,textvariable=self.var_overlap_y,width=4).pack(side=tk.LEFT,padx=3)
        ttk.Button(overlap,text="应用",command=self.on_apply_overlap).pack(side=tk.RIGHT)

        manual = ttk.Labelframe(rail, text="起点调整 · 不会自动拍照", padding=7)
        manual.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for c in range(2): manual.columnconfigure(c, weight=1)
        for r, c, title, sx, sy in ((0,0,"← X−",-1,0),(0,1,"X+ →",1,0),
                                  (1,0,"↑ Y+",0,1),(1,1,"↓ Y−",0,-1)):
            ttk.Button(manual, text=title, width=6, command=lambda x=sx,y=sy:self.manual_move(x,y)).grid(row=r,column=c,sticky="ew",padx=2,pady=2)

        ttk.Label(rail, textvariable=self.var_pos_steps, wraplength=276).grid(row=5, column=0, sticky="w", pady=(5, 0))
        ttk.Label(rail, textvariable=self.var_pos_um, wraplength=276, foreground="#647a84").grid(row=6, column=0, sticky="w")

        images = ttk.Frame(body)
        images.grid(row=0, column=1, sticky="nsew")
        images.columnconfigure(0, weight=1)
        images.rowconfigure(0, weight=3, minsize=180)
        images.rowconfigure(1, weight=2, minsize=150)

        live = ttk.Frame(images)
        live.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        live.columnconfigure(0, weight=1); live.rowconfigure(1, weight=1)
        livebar = ttk.Frame(live); livebar.grid(row=0,column=0,sticky="ew",pady=(0,4))
        ttk.Label(livebar, text="实时画面", font=(font,11,"bold")).pack(side=tk.LEFT)
        self.var_live_status = tk.StringVar(value="连接相机后在这里看显微镜画面")
        ttk.Label(livebar, textvariable=self.var_live_status, foreground="#647a84").pack(side=tk.LEFT,padx=12)
        ttk.Button(livebar, text="预览 / 暂停", command=self.toggle_preview).pack(side=tk.RIGHT)
        self._live_panel = self._make_image_label(live, "请先连接相机\n实时画面会直接显示在本窗口")
        self._live_panel.master.grid(row=1, column=0, sticky="nsew")

        frozen = ttk.Frame(images)
        frozen.grid(row=1,column=0,sticky="nsew")
        frozen.columnconfigure(0,weight=1,uniform="review"); frozen.columnconfigure(1,weight=1,uniform="review")
        frozen.rowconfigure(1,weight=1)
        ttk.Label(frozen,text="上一张 · 已确认",font=(font,10,"bold")).grid(row=0,column=0,sticky="w",pady=(0,4))
        ttk.Label(frozen,text="当前候选 · 确认后保存这张",font=(font,10,"bold"),foreground="#14747b").grid(row=0,column=1,sticky="w",padx=(8,0),pady=(0,4))
        self._reference_label = self._make_image_label(frozen,"还没有已确认照片")
        self._reference_label.master.grid(row=1,column=0,sticky="nsew",padx=(0,4))
        self._candidate_label = self._make_image_label(frozen,"开始扫描后在这里确认照片")
        self._candidate_label.master.grid(row=1,column=1,sticky="nsew",padx=(4,0))
        self.var_review = tk.StringVar(value="实时画面用于看焦点；下方候选是将要保存的照片。")
        review_message = ttk.Label(images,textvariable=self.var_review,wraplength=770)
        review_message.grid(row=2,column=0,sticky="ew",pady=(8,5))
        images.bind("<Configure>",lambda e:review_message.configure(wraplength=max(300,e.width-8)))
        choices = ttk.Frame(images); choices.grid(row=3,column=0,sticky="ew")
        for c in range(3): choices.columnconfigure(c,weight=1)
        self.btn_accept=ttk.Button(choices,text="确认保存，下一点",style="Accent.TButton",state="disabled")
        self.btn_accept.grid(row=0,column=0,sticky="ew",padx=(0,5))
        self.btn_retake=ttk.Button(choices,text="原位重拍",state="disabled")
        self.btn_retake.grid(row=0,column=1,sticky="ew",padx=5)
        self.btn_review_stop=ttk.Button(choices,text="停止，检查问题",state="disabled")
        self.btn_review_stop.grid(row=0,column=2,sticky="ew",padx=(5,0))

        footer=ttk.Frame(self,padding=(14,5,14,9));footer.grid(row=2,column=0,sticky="ew")
        ttk.Label(footer,textvariable=self.var_status,wraplength=900).pack(side=tk.LEFT)
        ttk.Label(footer,textvariable=self.var_progress,font=(font,11,"bold")).pack(side=tk.RIGHT)

    def _make_image_label(self, parent, placeholder):
        container=ttk.Frame(parent)
        container.grid_propagate(False)
        container.rowconfigure(0,weight=1);container.columnconfigure(0,weight=1)
        label=tk.Label(container,text=placeholder,background="#142632",foreground="#b7ced8",
                       borderwidth=0,font=("Arial",12),width=1,height=1)
        label.grid(row=0,column=0,sticky="nsew")
        label.image_source=None
        label.bind("<Configure>",lambda event:self._resize_display_image(label))
        return label

    def _display_frame(self, label, frame):
        label.image_source=frame
        self._resize_display_image(label)

    def _resize_display_image(self, label):
        frame=getattr(label,"image_source",None)
        if frame is None:
            return
        height,width=frame.shape[:2]
        area_w=max(40,getattr(label,"winfo_width",lambda:400)()-4)
        area_h=max(40,getattr(label,"winfo_height",lambda:235)()-4)
        factor=min(area_w/width,area_h/height)
        resized=cv2.resize(frame,(max(1,round(width*factor)),max(1,round(height*factor))))
        encoded=cv2.imencode('.png',resized)[1].tobytes()
        photo=tk.PhotoImage(data=base64.b64encode(encoded))
        label.config(image=photo,text="")
        label.image=photo

    # --------------------------------------------------------
    # Shared operation history (one recorder, read-only agent consumers)
    # --------------------------------------------------------
    def _history_context(self):
        # Capture Tk variables on the GUI thread, before launching a worker.
        return {"sample_id": self.var_sample_id.get().strip(),
                "objective": self.var_objective.get().strip() or None}

    def _ensure_history(self):
        session_error = getattr(self._session_log, "history_error", None)
        if session_error:
            self._history_error = session_error
        if self._history_error:
            raise RuntimeError("历史写入失败，已停止新增操作；请保留数据并按恢复步骤检查：" + self._history_error)
        if self._shared_history is None:
            self._shared_history = SharedHistory(self._history_root, metadata={
                "application": "AmScope acquisition executor", "release": "954-1080p-A1",
                "stage_simulated": bool(self.stage._simulate),
                "position_basis": "controller_step_counter_only_not_physical_metrology",
                "session_scope": "one_application_run_including_manual_moves_and_scans",
            })
        return self._shared_history

    def _record_history(self, event, **fields):
        try:
            return self._ensure_history().record(event, **fields)
        except Exception as exc:
            self._history_error = str(exc)
            raise RuntimeError("历史记录未能写入；已停止后续采集。已有照片保留。 " + str(exc)) from exc

    def _record_history_best_effort(self, event, **fields):
        # Only for operations that must remain possible if storage fails (stop,
        # disconnect). Ordinary movement/capture must persist its intent first.
        try:
            self._record_history(event, **fields)
            return True
        except Exception as exc:
            self.var_history_status.set("历史写入失败")
            print("[history] " + str(exc))
            return False

    def on_show_history(self):
        if self._history_error:
            self._show_history_recovery(force=True)
        show_history(self, self._history_root,
                     self._shared_history.path if self._shared_history else None)

    def _history_recovery_text(self):
        """Read-only guidance; restarting must never pretend to restore position."""
        base = Path(__file__).resolve().parent
        photo_folder = getattr(self, "_session_folder", None) or "本次尚未建立照片目录"
        if self._can_check_log_resume():
            return (
                "照片已保留，扫描已暂停。通常不用关闭程序。\n\n"
                "1. 保持样品、焦点和设备连接不变，不点移动按钮。\n"
                "2. 检查磁盘空间，关闭占用日志的编辑器；不删除或修改实验数据。\n"
                "3. 点顶部“检查并继续”。程序会核对照片、日志、相机及平台计数。\n"
                "4. 核对通过后，看显微镜或实时画面，确认样品和视野没变，再同意继续。\n\n"
                "已保存且核对一致的照片会复用；尚未完成的点在原位重新取候选。"
                "如果设备已重连、急停或位置改变，则不能从旧轮次续扫。\n"
                "详细说明见包内 PDF 第6–7页。\n"
                f"照片目录：{photo_folder}\n报错：{self._history_error}"
            )
        return (
            "照片可能已保存，但日志不完整。本次已禁止继续扫描；不要删除照片或手工补改 JSON。\n\n"
            "1. 看位移台是否停稳；若仍在移动或不安全，先点“紧急停止”。\n"
            "2. 截图当前报错；保留整个解压程序文件夹（照片、历史和程序都保留）。\n"
            "3. 等当前操作结束，点“断开设备”，再关闭窗口。断开报错时不要重复开程序，请联系老师。\n"
            "4. 检查磁盘空间、目录写入权限。需要换位置时，退出后重新解压到可写的本地目录，保留原目录。\n"
            "5. 重新双击 02_start_scan.bat，重新连接位移台和相机。通常只重启程序，不用重启电脑。\n"
            "6. 人工找回并确认起点、方向和焦点；程序不会自动回到原起点。先做 NX=2、NY=1 的新扫描。\n"
            "7. 新扫描保存到新的时间戳目录；不续接旧的部分扫描。再次报同样错误就停止，交回原目录和截图。\n\n"
            "详细步骤见：00_学生操作与故障恢复手册.pdf 第 6–7 页\n"
            f"程序目录：{base}\n照片目录：{photo_folder}\n"
            f"报错：{self._history_error}"
        )

    def _show_history_recovery(self, force=False):
        """Show once on the GUI thread, after workers release the instruments."""
        if threading.current_thread() is not threading.main_thread():
            self.after(0, self._show_history_recovery, force)
            return
        if not self._history_error:
            self._history_error = getattr(self._session_log, "history_error", None)
        if not self._history_error:
            return
        self._scan_params = None
        self.btn_auto.config(state="disabled")
        self.btn_resume.config(state="normal" if self._can_check_log_resume() else "disabled",
                               text="检查并继续")
        self.var_history_status.set("历史写入失败·恢复步骤")
        if self._camera_connecting or (self._worker and self._worker.is_alive()):
            self.after(100, self._show_history_recovery, force)
            return
        if getattr(self, "_history_help_shown", False) and not force:
            return
        self._history_help_shown = True
        messagebox.showerror("日志写入失败：按步骤恢复", self._history_recovery_text(), parent=self)

    def _can_check_log_resume(self):
        return bool(getattr(self, "_log_resume_plan", None)
                    and getattr(self, "_log_resume_devices", None)
                    and getattr(self, "_session_log", None)
                    and not getattr(self, "_log_recovery_cancelled", False))

    def _validate_log_resume_devices(self):
        """Read only: never reconnect, home, jog or clear an emergency stop."""
        if not self._can_check_log_resume():
            raise RuntimeError("没有本次运行的完整续扫状态；请保留资料，重新确认起点后新建扫描。")
        stage, camera, epoch = self._log_resume_devices
        if (stage is not self.stage or camera is not self.cam
                or epoch != getattr(self, "_device_epoch", 0)):
            raise RuntimeError("设备已经重连或执行过其他移动，不能按原位置续扫。")
        if not self.stage.connected or not self.stage.position_trusted:
            raise RuntimeError("平台连接或位置已改变，不能续扫；请重新核对起点。")
        if tuple(self._log_pause_position or ()) != (self.stage.x, self.stage.y):
            raise RuntimeError("暂停位置已改变，不能续扫。")
        self.stage.verify_position()
        if self._run_auto_capture:
            if not self.connected_camera or self.cam is None:
                raise RuntimeError("相机已断开，不能续扫。")
            current = self._verify_acquisition_mode("log_recovery", record=False)
            if self._camera_signature(current) != self._session_camera_signature:
                raise RuntimeError("相机或图像尺寸改变，不能混入本轮扫描。")
            expected_readback = getattr(self, "_session_recovery_readback", None)
            if expected_readback is not None and current.get("readback") != expected_readback:
                raise RuntimeError("相机采集设置改变，请新建扫描。")
        if getattr(self, "_log_recovery_cancelled", False):
            raise RuntimeError("已取消恢复，设备保持停止。")

    def _prepare_log_resume(self):
        from scan_log_resume import recover_scan_checkpoint
        self._validate_log_resume_devices()
        def decode(path):
            if not _HAS_CV2:
                return False
            from acquisition_contract import validate_saved_image
            validate_saved_image(path, CAMERA_RESOLUTION, "log_recovery_saved_image")
            return True
        result = recover_scan_checkpoint(self._session_log,
                                         getattr(self, "_point_checkpoint", None), image_decoder=decode)
        self._validate_log_resume_devices()
        return result

    def on_check_and_resume(self):
        if (self.auto_running or self._camera_connecting
                or getattr(self, "_log_recovery_busy", False)
                or (self._worker and self._worker.is_alive())):
            return
        if not self._can_check_log_resume():
            self._show_history_recovery(force=True)
            return
        self._log_recovery_busy = True
        self.auto_running = True
        self.btn_auto.config(state="disabled")
        self.btn_resume.config(state="disabled")
        self._close_preview()
        self.var_status.set("正在核对照片、日志和暂停位置；不会移动平台…")
        def worker():
            try:
                result = self._prepare_log_resume()
            except Exception as exc:
                self.after(0, self._log_resume_prepared, None, str(exc))
            else:
                self.after(0, self._log_resume_prepared, result, None)
        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _log_resume_failed(self, error):
        self._worker = None
        self.auto_running = False
        self._log_recovery_busy = False
        self._log_recovery_review = False
        self._history_error = str(error)
        self._close_preview()
        self.btn_auto.config(state="disabled")
        self.btn_resume.config(state="normal" if self._can_check_log_resume() else "disabled", text="检查并继续")
        self.var_status.set("未续扫：" + str(error))
        messagebox.showerror("尚不能继续：照片已保留", str(error) + "\n\n修正保存问题后可再点“检查并继续”。"
                             "设备或位置改变时，请正常退出、核对起点后新建扫描。", parent=self)

    def _log_resume_prepared(self, result, error):
        self._worker = None
        if error:
            self._log_resume_failed(error)
            return
        if not self._can_check_log_resume():
            self._log_resume_failed("恢复期间已停止或关闭；不会继续扫描。")
            return
        self._log_recovery_review = True
        self._start_preview()
        action = "复用已保存的当前照片，不重复拍摄" if result['action'] == 'reuse_saved_image' else "从尚未完成的点继续，必要时原位重拍"
        confirmed = messagebox.askyesno(
            "记录已修复：请确认现场未变",
            "请看显微镜或实时画面：样品、视野、焦点均未改变，且没有手动移动或重连设备吗？\n\n"
            + action + "。继续使用原扫描参数和原文件夹。\n"
            "选“是”后程序会再次核对位置，再继续；不确定请选“否”。", parent=self)
        self._log_recovery_review = False
        self._close_preview()
        if not confirmed:
            self.auto_running = False
            self._log_recovery_busy = False
            self.var_status.set("记录已核对，尚未续扫；确认现场后可再点“检查并继续”。")
            self.btn_resume.config(state="normal" if self._can_check_log_resume() else "disabled", text="检查并继续")
            return
        # Recheck after the modal: it pumps Tk events, including Emergency Stop.
        def worker():
            try:
                result = self._prepare_log_resume()
                self._session_log.record("log_resume_authorized", actor="human_operator",
                                         field_sample_focus_unchanged_confirmed=True,
                                         confirmation_basis="operator_confirmation_not_independent_metrology",
                                         verification=getattr(self.stage, "last_position_evidence", {}))
                self._validate_log_resume_devices()
                next_point = (result['point'] if result['action'] == 'reuse_saved_image'
                              else self._scan_done_count) + 1
                if next_point <= self._planned_total:
                    self._session_log.record("resume", next_point=next_point, reason="verified_log_recovery")
            except Exception as exc:
                self.after(0, self._log_resume_failed, str(exc))
            else:
                self.after(0, self._dispatch_log_resume, result)
        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _dispatch_log_resume(self, result):
        self._worker = None
        if not self._can_check_log_resume() or not self.stage.position_trusted:
            self._log_resume_failed("恢复期间位置状态失效；不会移动。")
            return
        checkpoint = getattr(self, "_point_checkpoint", None)
        if result['action'] == 'reuse_saved_image':
            self._photo_count = checkpoint['photo_count_before'] + 1
            self._human_review_count = checkpoint['human_count_before'] + int(checkpoint['human_confirmed'])
            self._sharpness_records = self._sharpness_records[:checkpoint['sharpness_len_before']]
            self._sharpness_records.append({key: checkpoint[key] for key in ('row', 'col', 'sharpness', 'retries', 'filepath')})
            self._last_accepted_photo = checkpoint['filepath']
            self._scan_done_count = checkpoint['point']
        self._history_error = None
        self._history_help_shown = False
        self._worker_error = None
        self._log_recovery_busy = False
        self._scan_params = self._log_resume_plan
        self.var_history_status.set("查看历史")
        self.btn_resume.config(text="检查并继续", state="disabled")
        self._cancel_event.clear()
        if self._scan_done_count >= self._planned_total:
            self._finalize_auto()
            return
        self.var_status.set(f"记录修复完成，继续第 {self._scan_done_count + 1}/{self._planned_total} 点")
        self._worker = threading.Thread(target=self._auto_scan_worker,
                                       args=(*self._scan_params, self._scan_done_count), daemon=True)
        self._worker.start()
        self.after(100, self._poll_auto_done)

    def _resume_history_failed(self, error):
        self._history_error = str(error)
        self._scan_params = None
        self.auto_running = False
        self.btn_auto.config(state="normal")
        self.btn_resume.config(state="disabled")
        self.var_history_status.set("历史写入失败")
        self.var_status.set("未开始续扫；历史写入失败，照片保留，请检查保存位置并按恢复步骤处理。")
        self._show_history_recovery()

    # --------------------------------------------------------
    # Connection
    # --------------------------------------------------------
    def on_connect_stage(self):
        if self.auto_running or self._camera_connecting:
            return
        self._device_epoch = getattr(self, "_device_epoch", 0) + 1
        try:
            self._record_history("stage_connect_requested", **self._history_context())
            if self.stage.connected:
                self.stage.disconnect()
            self.stage.connect()
            self._record_history("stage_connected", stage=getattr(self.stage, "stage_info", {}),
                                 x_steps=self.stage.x, y_steps=self.stage.y,
                                 stage_simulated=bool(self.stage._simulate))
            self._disconnect_error = None
            self._update_status()
        except Exception as e:
            self._record_history_best_effort("stage_connect_failed", error=str(e))
            if self._history_error:
                self._show_history_recovery()
            else:
                messagebox.showerror("Stage Connect Failed", str(e))
            self.var_status.set("STAGE ERROR")

    def on_connect_camera(self, exclude_indices=()):
        if self.auto_running or self._camera_connecting:
            return
        self._device_epoch = getattr(self, "_device_epoch", 0) + 1
        if not _HAS_CAMERA:
            messagebox.showerror("Camera", "Camera_v2.py or its dependencies are unavailable.")
            return
        try:
            self._record_history("camera_connect_requested", excluded_indices=list(exclude_indices),
                                 **self._history_context())
        except Exception as exc:
            self._show_history_recovery()
            return
        self._close_preview()
        self._camera_connecting = True
        self.var_status.set("Detecting camera / 正在读取设备...")
        def connect_worker():
            try:
                with self._camera_lock:
                    if self.cam is not None:
                        self.cam.close()
                    self.cam = None
                    self.connected_camera = False
                    cam, info = open_detected_camera(
                        Camera, Path(__file__).resolve().parent / "camera_history",
                        bits=BITS, resolution=CAMERA_RESOLUTION,
                        indices=range(4), exclude_indices=exclude_indices)
                    self.cam = cam
                    self._camera_info = info
                    self.connected_camera = True
                self.after(0, self._camera_connected)
            except Exception as exc:
                self.after(0, self._camera_connect_failed, exc)
        threading.Thread(target=connect_worker, daemon=True).start()

    def _camera_connected(self):
        self._failed_camera_index = None
        self._camera_connecting = False
        self._disconnect_error = None
        self._record_history_best_effort("camera_connected", camera=self._camera_info)
        self._update_status()
        self._start_preview()
        # Camera settings are read automatically. Do not change an unknown UVC
        # camera's exposure mode: the legacy wrapper uses device-specific values.
        self.var_status.set("相机已连接：请在右侧确认显微镜画面")
        if self._history_error:
            self.var_status.set("相机已连接，但历史写入失败；请查看恢复步骤。")
            self._show_history_recovery()

    def _camera_connect_failed(self, error):
        self._failed_camera_index = getattr(error, "camera_index", None)
        self._camera_connecting = False
        self.connected_camera = False
        self._record_history_best_effort("camera_connect_failed", error=str(error),
                                         camera_index=self._failed_camera_index,
                                         backend_requested=getattr(error, "backend_requested", None))
        self.var_status.set("CAMERA ERROR")
        if self._history_error:
            self._show_history_recovery()
        else:
            messagebox.showerror("Camera Connect Failed", str(error))

    def on_next_camera(self):
        if self.cam is not None:
            self._camera_cycle_excluded.add(self.cam.camera_index)
        elif type(getattr(self, "_failed_camera_index", None)) is int:
            # Explicit user action only: ordinary Connect never auto-skips it.
            self._camera_cycle_excluded.add(self._failed_camera_index)
        if len(self._camera_cycle_excluded) >= 4:
            self._camera_cycle_excluded.clear()
        self.on_connect_camera(exclude_indices=tuple(self._camera_cycle_excluded))


    def on_disconnect(self):
        if self.auto_running or self._camera_connecting:
            messagebox.showwarning("设备忙", "请先停止扫描，等待当前操作结束。")
            return
        self._device_epoch = getattr(self, "_device_epoch", 0) + 1
        self._close_preview()
        self._camera_connecting = True
        self.var_status.set("正在断开设备…")
        self._disconnect_error = None

        def disconnect_worker():
            errors = []
            try:
                with self._camera_lock:
                    if self.cam is not None:
                        self.cam.close()
                    self.cam = None
                    self.connected_camera = False
            except Exception as exc:
                errors.append(str(exc))
            try:
                if self.stage.connected:
                    self.stage.disconnect()
            except Exception as exc:
                errors.append(str(exc))
            self.after(0, self._disconnect_finished, "; ".join(errors))

        threading.Thread(target=disconnect_worker, daemon=True).start()

    def _disconnect_finished(self, error):
        self._camera_connecting = False
        self._disconnect_error = error
        self._record_history_best_effort("devices_disconnected" if not error else "disconnect_failed", error=error or None)
        self._update_status()
        self.var_live_status.set("设备已断开 · 保留上一帧" if not error else "断开失败 · 画面非实时")
        self.var_status.set("设备已断开" if not error else "断开设备失败：" + error)
        if self._history_error:
            self._show_history_recovery()

    def on_emergency_stop(self):
        """Do not offer unsafe relative-coordinate resume after an emergency stop."""
        self._cancel_event.set()
        self._log_recovery_cancelled = True
        self._log_resume_plan = None
        self.stage.position_trusted = False
        self.stage.emergency_stop(wait=False)
        # Stop is issued BEFORE file I/O. This records a request, never a claim
        # that a mechanical stop or final position has been independently seen.
        self._record_history_best_effort("emergency_stop_requested", actor="human_operator",
                                         wait_for_controller=False, physical_stop_verified=False,
                                         x_steps=self.stage.x, y_steps=self.stage.y)
        self._scan_params = None
        self.var_status.set("STOPPING — reconnect and check position before a NEW scan")
        self.btn_auto.config(state="disabled")
        self.btn_resume.config(state="disabled")
        if self._history_error and not self.auto_running:
            self._show_history_recovery()

    def _update_status(self):
        parts = []
        if self.stage.connected:
            parts.append("Stage OK")
        if self.connected_camera:
            parts.append("Camera OK")
        self.var_connection.set("相机已连接" if self.connected_camera else "相机未连接")
        self.var_connection.set(self.var_connection.get() + (" · 位移台已连接" if self.stage.connected else " · 位移台未连接"))
        if parts:
            self.var_status.set(" | ".join(parts))
        else:
            self.var_status.set("IDLE")

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------
    def toggle_preview(self):
        if self._preview_running:
            self._close_preview()
        else:
            self._start_preview()

    def _start_preview(self):
        if self.auto_running and not self._review_active and not getattr(self, "_log_recovery_review", False):
            return
        if not self.connected_camera or not _HAS_CV2:
            return
        if self._preview_running:
            return
        self._preview_running = True
        self._preview_generation += 1
        generation = self._preview_generation
        self._live_last_update = None
        self.var_live_status.set("正在等待新画面…")
        self._preview_pump.start(self.cam)
        self._preview_after_id = self.after(0, self._preview_tick, generation)

    def _preview_tick(self, generation):
        # A canceled callback may already be queued; never revive an old loop.
        if generation != self._preview_generation or not self._preview_running:
            return
        self._preview_after_id = None
        if (not self.connected_camera or self.cam is None
                or (self.auto_running and not self._review_active and not getattr(self, "_log_recovery_review", False))):
            self._close_preview()
            return
        try:
            self._preview_pump.start(self.cam)
            result = self._preview_pump.poll()
            if result is not None:
                kind, payload = result
                if kind == "error":
                    raise RuntimeError(payload)
                self._display_frame(self._live_panel, payload)
                self._live_last_update = time.monotonic()
                self.var_live_status.set("实时 · 下方候选保持不变")
            if self._live_last_update is None:
                self.var_live_status.set("正在等待新画面…")
            elif time.monotonic() - self._live_last_update > 2:
                age = int(time.monotonic() - self._live_last_update)
                self.var_live_status.set(f"等待新画面 · 上次更新 {age} 秒前")
        except Exception as exc:
            print(f"[preview] Error: {exc}")
            self._close_preview()
            self.var_live_status.set("预览读取失败 · 显示的是上一帧")
            return
        self._preview_after_id = self.after(50, self._preview_tick, generation)

    def _close_preview(self):
        self._preview_running = False
        self._preview_generation += 1
        if self._preview_after_id is not None:
            try:
                self.after_cancel(self._preview_after_id)
            except Exception:
                pass
            self._preview_after_id = None
        self._preview_pump.stop()
        if hasattr(self, 'var_live_status'):
            self.var_live_status.set("预览已暂停 · 保留上一帧")

    # --------------------------------------------------------
    # Photo capture at each point
    # --------------------------------------------------------
    def _init_session(self):
        """Create timestamped session folder for photos."""
        history = self._ensure_history()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._session_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), SAVE_FOLDER, ts)
        os.makedirs(self._session_folder, exist_ok=False)
        self._photo_count = 0
        self._capture_attempts = 0
        self._capture_failures = 0
        self._sharpness_records = []
        self._last_accepted_photo = None
        self._human_review_count = 0
        nx, ny, dx, dy, order = self._scan_params
        if self.cam is not None and self.connected_camera:
            with self._camera_lock:
                current = describe_camera(self.cam)
            current["actual_resolution"] = self._camera_info.get("actual_resolution")
            current["resolution_source"] = "connection_frame_pending_pre_scan"
            current["connection_record"] = {k: v for k, v in self._camera_info.items()
                                            if k != "connection_record"}
            current["microscope_view_confirmed"] = True
            current["confirmation_basis"] = "operator_confirmed_at_scan_start"
            self._camera_info = current
        from acquisition_contract import contract
        acquisition = contract(CAMERA_RESOLUTION, self._camera_info.get("actual_resolution"),
                               phase="pre_scan")
        config = {"acquisition_contract": acquisition,
                  "sample_id": self.var_sample_id.get().strip(),
                  "objective": self.var_objective.get().strip() or None,
                  "device_x": DEVICE_X.decode(), "device_y": DEVICE_Y.decode(),
                  "overlap_x_steps": safe_int(self.var_overlap_x.get(), 8),
                  "overlap_y_steps": safe_int(self.var_overlap_y.get(), 8),
                  "nx": nx, "ny": ny, "dx_steps": dx, "dy_steps": dy,
                  "order": order, "auto_capture": self._run_auto_capture,
                  "release": "954-1080p-A1", "review_mode": self._run_review_mode,
                  "origin_logical_steps": self._scan_origin,
                  "fullstep_um": FULLSTEP_UM, "px_per_step": PX_PER_STEP,
                  "calibration_status": "legacy defaults; must verify current microscope and camera",
                  "fov_steps_x": FOV_STEPS_X, "fov_steps_y": FOV_STEPS_Y,
                  "invert_x": INVERT_X, "invert_y": INVERT_Y,
                  "stage_position_basis": "controller-counter-checked logical coordinates; not physical encoder verification",
                  "stage_readback": getattr(self.stage, "stage_info", {}),
                  "stage_simulated": self.stage._simulate,
                  "camera_source": "connected camera" if self.connected_camera else "none",
                  "settle_s": SETTLE_TIME_S, "flush_frames": BUFFER_FLUSH_FRAMES,
                  "blur_threshold": BLUR_THRESHOLD, "blur_max_retries": FOCUS_MAX_RETRIES}
        self._session_camera_signature = self._camera_signature(self._camera_info)
        self._session_recovery_readback = self._camera_info.get("readback")
        base = Path(__file__).resolve().parent
        self._session_log = ScanSession(
            self._session_folder, self._camera_info, config,
            [Path(__file__), base / "Camera_v2.py", base / "camera_session_history.py", base / "scan_review.py", base / "scan_preview.py",
             base / "shared_history.py", base / "history_viewer.py", base / "scan_log_resume.py",
             base / "acquisition_contract.py"],
            simulated=self.stage._simulate, shared_history=history)
        self._log_resume_plan = self._scan_params
        self._log_resume_devices = (self.stage, self.cam, getattr(self, "_device_epoch", 0))
        self._log_pause_position = None
        self._point_checkpoint = None
        self._log_recovery_cancelled = False
        self.var_history_status.set("查看历史")
        print(f"[session] Photos will be saved to: {self._session_folder}")

    def _verify_acquisition_mode(self, phase, record=True):
        """Fresh-frame gate, called from background workers before scan movement."""
        from acquisition_contract import (AcquisitionModeError, contract,
                                          require_frame_size, preserve_diagnostic_frame)
        if self._cancel_event.is_set() and phase != "log_recovery":
            raise RuntimeError("Acquisition verification cancelled; no movement performed.")
        with self._camera_lock:
            frame = _grab_frame_flushed(self.cam, timeout=5.0,
                                       flush_frames=BUFFER_FLUSH_FRAMES if ENABLE_FRAME_FLUSH else 0)
            current = describe_camera(self.cam, frame)
        try:
            actual = require_frame_size(frame, CAMERA_RESOLUTION, phase)
        except AcquisitionModeError as mismatch:
            diagnostic = None
            if frame is not None:
                try:
                    diagnostic = preserve_diagnostic_frame(
                        frame, Path(self._session_folder) / "review_candidates",
                        Camera.save_image, phase)
                    mismatch.args = (str(mismatch) + " Diagnostic: " + str(diagnostic),)
                except Exception as save_error:
                    mismatch.args = (str(mismatch) + " Diagnostic save failed: " + str(save_error),)
            if record:
                failed = contract(CAMERA_RESOLUTION, mismatch.actual, phase=phase)
                failed["verification_status"] = "mismatch"
                self._session_log.summary["scan_config"]["acquisition_contract"] = failed
                self._session_log.record("acquisition_mode_mismatch", acquisition_contract=failed,
                                         diagnostic_path=str(diagnostic) if diagnostic else None,
                                         error=str(mismatch))
            raise
        if record:
            if self._camera_signature(current) != self._session_camera_signature:
                raise RuntimeError("Camera identity changed; start a new scan after confirming the microscope view.")
            verified = contract(CAMERA_RESOLUTION, actual, phase=phase, verified=True)
            current["microscope_view_confirmed"] = True
            current["confirmation_basis"] = "operator_confirmed_at_scan_start"
            current["connection_record"] = self._camera_info.get("connection_record", self._camera_info)
            self._camera_info = current
            self._session_log.summary["camera"] = current
            self._session_log.summary["scan_config"]["acquisition_contract"] = verified
            self._session_log.record("acquisition_mode_verified", acquisition_contract=verified)
        return current

    def _check_candidate_dimensions(self, candidate, phase, raw_bytes=None):
        from acquisition_contract import AcquisitionModeError, contract, validate_saved_image
        try:
            return validate_saved_image(candidate, CAMERA_RESOLUTION, phase, raw_bytes=raw_bytes)
        except AcquisitionModeError as mismatch:
            failed = contract(CAMERA_RESOLUTION, mismatch.actual, phase=phase)
            failed["verification_status"] = "mismatch"
            self._session_log.summary["scan_config"]["acquisition_contract"] = failed
            self._session_log.record("acquisition_mode_mismatch", acquisition_contract=failed,
                                     filepath=str(candidate), error=str(mismatch))
            raise

    @staticmethod
    def _camera_signature(info):
        return (info.get("backend"), info.get("camera_index"), info.get("device_id"),
                info.get("serial_number"), tuple(info.get("actual_resolution") or ()))

    def _take_photo_at_point(self, photo_idx, total, row, col):
        """Return success; a missing/failed capture must never become a completed point."""
        if not self._run_auto_capture:
            self._session_log.record("manual_dwell", row=row, col=col,
                                     note="Position visited; no image captured or verified")
            _interruptible_sleep(safe_float(self.var_photo_dwell.get(), 0.5), self._cancel_event)
            return not self._cancel_event.is_set()
        if not self.connected_camera or self.cam is None:
            raise RuntimeError("Automatic capture requires a connected camera. No dwell-only fallback.")
        if self._cancel_event.is_set():
            return False
        filename = f"mosaic_r{row}_c{col}{SAVE_FORMAT}"
        filepath = os.path.join(self._session_folder, filename)
        fields = {"row": row, "col": col, "point": photo_idx,
                  "x_steps": self.stage.x, "y_steps": self.stage.y,
                  "filename": filename, "filepath": filepath}
        while not self._cancel_event.is_set():
            if hasattr(self.stage, "verify_position"):
                self.stage.verify_position()
                self._session_log.record("position_checked", phase="before_capture", row=row, col=col,
                                         verification=getattr(self.stage, "last_position_evidence", {}))
            self._capture_attempts += 1
            candidate = os.path.join(self._session_folder, "review_candidates",
                                     f"r{row}_c{col}_attempt{self._capture_attempts:04d}{SAVE_FORMAT}")
            self._session_log.record("capture_attempt", **dict(fields, filepath=candidate))
            self._update_gui_status(f"正在取图 {photo_idx}/{total}", f"{photo_idx}/{total}")
            try:
                _interruptible_sleep(SETTLE_TIME_S, self._cancel_event)
                with self._camera_lock:
                    ok, sharpness, retries = capture_and_save(self.cam, candidate, self._cancel_event, skip_settle=True)
                    frame_metadata = getattr(self.cam, "last_capture_metadata", None)
                if not ok:
                    raise RuntimeError("未取得并保存新图像，已停在当前点；检查相机后再继续。")
                self._check_candidate_dimensions(candidate, phase="candidate")
                diagnostic = compare_views(self._last_accepted_photo, candidate)
            except Exception as exc:
                self._capture_failures += 1
                self._session_log.record("capture_failed", error=str(exc), **dict(fields, filepath=candidate))
                raise
            low_sharpness = bool(sharpness < BLUR_THRESHOLD)
            self._session_log.record("candidate_saved", sharpness=sharpness, retries=retries,
                                     frame_read=frame_metadata, image_diagnostic=diagnostic,
                                     **dict(fields, filepath=candidate))
            needs_review = (self._run_review_mode == "step_confirm" or low_sharpness
                            or diagnostic['status'] != 'changed')
            decision = "accept"
            if needs_review:
                summary = f"第 {photo_idx}/{total} 点：{diagnostic['message']}"
                if low_sharpness:
                    summary += " 清晰度偏低，请先调焦并原位重拍。"
                decision = self._request_review(candidate, self._last_accepted_photo, summary,
                                                warning=low_sharpness or diagnostic['status'] == 'unverified',
                                                block_accept=diagnostic['status'] == 'same_view')
                self._session_log.record("human_review", decision=decision,
                                         decision_source="cancellation" if self._cancel_event.is_set() else "operator_review",
                                         rule_warning=low_sharpness or diagnostic['status'] != 'changed',
                                         **dict(fields, filepath=candidate))
            if decision == "retake":
                continue  # stay at this exact point; never send an extra move
            if decision != "accept" or self._cancel_event.is_set():
                self.stage.position_trusted = False
                self._cancel_event.set()
                return False
            if os.path.exists(filepath):
                raise RuntimeError("Refusing to overwrite accepted raw image: " + filepath)
            if hasattr(self.stage, "verify_position"):
                # Detect external controller movement while the operator reviewed
                # the frozen candidate. Counter continuity is not physical proof.
                self.stage.verify_position()
                self._session_log.record("position_checked", phase="before_accept", row=row, col=col,
                                         verification=getattr(self.stage, "last_position_evidence", {}))
            if self._cancel_event.is_set():
                return False
            # Save exactly the reviewed bytes, not a newly grabbed frame.
            import hashlib
            candidate_bytes = Path(candidate).read_bytes()
            saved_dimensions = self._check_candidate_dimensions(
                candidate, phase="before_promotion", raw_bytes=candidate_bytes)
            checkpoint = getattr(self, "_point_checkpoint", None)
            if checkpoint is not None:
                checkpoint.update(filepath=filepath, candidate_path=candidate,
                                  image_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
                                  image_bytes=len(candidate_bytes), point=photo_idx, row=row, col=col,
                                  sharpness=sharpness, retries=retries, human_confirmed=needs_review)
            self._session_log.record("candidate_promotion_requested", candidate_path=candidate,
                                     saved_image_resolution=saved_dimensions,
                                     expected_image_resolution=list(CAMERA_RESOLUTION),
                                     human_confirmed=needs_review, **fields)
            if Path(candidate).read_bytes() != candidate_bytes:
                raise RuntimeError("Candidate changed after dimension verification; photo preserved, scan stopped.")
            os.replace(candidate, filepath)
            if checkpoint is not None:
                checkpoint['phase'] = 'promoted'
            self._session_log.record("candidate_accepted", candidate_original_path=candidate,
                                     human_confirmed=needs_review, frame_read=frame_metadata,
                                     image_diagnostic=diagnostic, **fields)
            self._human_review_count += int(needs_review)
            self._last_accepted_photo = filepath
            break
        else:
            return False
        self._photo_count += 1
        self._sharpness_records.append({"row": row, "col": col, "sharpness": sharpness,
                                       "retries": retries, "filepath": filepath})
        self._session_log.record("capture_success", sharpness=sharpness, retries=retries,
                                 below_blur_threshold=bool(sharpness < BLUR_THRESHOLD),
                                 camera_backend=self.cam.backend,
                                 camera_index=self.cam.camera_index,
                                 saved_image_resolution=saved_dimensions, **fields)
        print(f"[photo] {photo_idx}/{total}: {filename} sharp={sharpness:.1f}")
        return True

    def _request_review(self, candidate, previous, summary, warning=False, block_accept=False):
        token = self._review_gate.begin()
        self.after(0, self._show_review, token, candidate, previous, summary, warning, block_accept)
        decision = self._review_gate.wait(token, self._cancel_event)
        self.after(0, self._hide_review, token)
        return decision

    def _show_review(self, token, candidate, previous, summary, warning, block_accept):
        if self._cancel_event.is_set() or not self._review_gate.is_active(token):
            return
        self._review_token = token
        self._review_active = True
        self.var_status.set("等待你确认：平台保持不动")
        self.var_review.set(summary + " 左：上一张已确认；右：当前待确认。接受后保存的就是右侧这张。")
        for label, path, title in ((self._reference_label, previous, "上一张已确认"),
                                   (self._candidate_label, candidate, "当前待确认")):
            if not path:
                label.config(image="", text="起点，没有上一张照片")
                label.image = None
                label.image_source = None
                continue
            try:
                frame = cv2.imread(path)
                if frame is None:
                    raise RuntimeError("无法读取照片")
                self._display_frame(label, frame)
            except Exception as exc:
                label.config(image="", text="照片显示失败：请原位重拍或停止检查")
                label.image = None
                label.image_source = None
                self.var_review.set("照片显示失败，不能确认保存。请原位重拍或停止检查。 " + str(exc))
                block_accept = True
        self._review_block_accept = block_accept
        self.btn_accept.config(state="disabled" if block_accept else "normal",
                               command=lambda: self._choose_review(token, "accept", warning))
        self.btn_retake.config(state="normal", command=lambda: self._choose_review(token, "retake"))
        self.btn_review_stop.config(state="normal", command=lambda: self._choose_review(token, "stop"))
        self._start_preview()

    def _choose_review(self, token, choice, warning=False):
        if (token != self._review_token or not self._review_gate.is_active(token)
                or self._cancel_event.is_set()):
            return
        if choice == "accept" and self._review_block_accept:
            return
        if choice == "accept" and warning and not messagebox.askyesno(
                "请核实当前照片", "自动检查未通过或不确定。你已确认画面确实移动且图像可用吗？\n"
                "若仍模糊，请取消后选择原位重拍。人工确认将保留在日志中。"):
            return
        # Modal dialogs process Tk events; Stop may have arrived while open.
        if (token != self._review_token or not self._review_gate.is_active(token)
                or self._cancel_event.is_set()):
            return
        if self._review_gate.choose(token, choice):
            self._hide_review(token)

    def _hide_review(self, token):
        if self._review_token != token:
            return
        self._review_active = False
        self._close_preview()
        for button in (self.btn_accept, self.btn_retake, self.btn_review_stop):
            button.config(state="disabled")

    def _update_gui_status(self, status_text, progress_text):
        """Thread-safe GUI status update via StringVar (thread-safe in CPython)."""
        # StringVar.set() is safe from threads under CPython's GIL.
        # self.after() from worker threads also works on CPython/Windows.
        try:
            self.var_status.set(status_text)
            self.var_progress.set(progress_text)
        except Exception:
            pass

    # --------------------------------------------------------
    # Overlap
    # --------------------------------------------------------
    def on_apply_overlap(self):
        ox = safe_int(self.var_overlap_x.get(), 8)
        oy = safe_int(self.var_overlap_y.get(), 8)
        dx, dy = compute_dxdy(ox, oy)
        self.var_dx.set(str(dx))
        self.var_dy.set(str(dy))
        self.var_status.set("dx/dy updated")

    # --------------------------------------------------------
    # Manual move (no auto photo)
    # --------------------------------------------------------
    def manual_move(self, sx, sy):
        if not self.stage.connected:
            messagebox.showwarning("位移台未连接", "请先连接位移台。")
            return
        if self.auto_running or self._camera_connecting:
            return
        dx = safe_int(self.var_dx.get(), 0) * sx
        dy = safe_int(self.var_dy.get(), 0) * sy
        context = self._history_context()
        try:
            self._ensure_history()
        except Exception as exc:
            self._history_error = str(exc)
            self._show_history_recovery()
            return
        self._device_epoch = getattr(self, "_device_epoch", 0) + 1
        self._close_preview()
        self._cancel_event.clear()
        self._scan_params = None
        self.auto_running = True
        self.btn_auto.config(state="disabled")
        self.btn_resume.config(state="disabled")
        self.var_status.set("正在调整起点…")

        def move_worker():
            error = None
            try:
                self._record_history("manual_move_attempt", dx_steps=dx, dy_steps=dy,
                                     x_steps=self.stage.x, y_steps=self.stage.y,
                                     actor="human_operator", **context)
                self.stage.move_relative(dx, dy, self._cancel_event)
                self._record_history("manual_move_completed", x_steps=self.stage.x, y_steps=self.stage.y,
                                     verification=getattr(self.stage, "last_move_evidence", {}), **context)
            except Exception as exc:
                error = str(exc)
                # Retain any available controller evidence without interpreting
                # a partial/failed move as a completed physical translation.
                try:
                    self._record_history("manual_move_failed", error=error,
                                         verification=getattr(self.stage, "last_move_evidence", {}), **context)
                except Exception as history_exc:
                    error += "；" + str(history_exc)
            self.after(0, self._finish_manual_move, error)

        self._worker = threading.Thread(target=move_worker, daemon=True)
        self._worker.start()

    def _finish_manual_move(self, error):
        self.auto_running = False
        self._worker = None
        self.btn_auto.config(state="normal")
        self._refresh_pos()
        if error:
            self.var_status.set("移动未通过检查：" + error)
        else:
            self.var_status.set("起点已调整：请看实时画面确认位置")
        self._start_preview()
        if self._history_error:
            self._show_history_recovery()

    # --------------------------------------------------------
    # Auto snake scan (background thread)
    # --------------------------------------------------------
    def start_auto(self):
        if getattr(self, "_history_error", None):
            self._show_history_recovery(force=True)
            return
        if not self.stage.connected:
            messagebox.showwarning("Not connected", "Connect stage first")
            return
        if self.auto_running or self._camera_connecting:
            return
        if not self.stage.position_trusted:
            messagebox.showwarning("Position uncertain", "Reconnect the stage, check the field, then start a NEW scan.")
            return
        if self.var_auto_capture.get() and (not self.connected_camera or self.cam is None):
            messagebox.showwarning("Camera required", "Click Connect Camera and confirm the microscope preview first.")
            return
        if self.var_auto_capture.get() and not messagebox.askyesno(
                "开始前确认", "当前预览是显微镜样品，已确认起点、方向和焦点吗？\n"
                "默认每点人工确认；未确认前不会自动进入下一点。"):
            return

        nx = max(1, safe_int(self.var_nx.get(), 1))
        ny = max(1, safe_int(self.var_ny.get(), 1))
        dx_step = safe_int(self.var_dx.get(), 0)
        dy_step = safe_int(self.var_dy.get(), 0)

        # Validate step sizes
        if nx > 1 and dx_step == 0:
            messagebox.showwarning("Invalid", "dx = 0 but NX > 1. Set overlap or dx first.")
            return
        if ny > 1 and dy_step == 0:
            messagebox.showwarning("Invalid", "dy = 0 but NY > 1. Set overlap or dy first.")
            return

        order = self.var_order.get()

        self._cancel_event.clear()
        self._log_resume_plan = None
        self._point_checkpoint = None
        self.auto_running = True
        self.btn_auto.config(state="disabled")
        self.btn_resume.config(state="disabled")
        self._scan_done_count = 0
        self._scan_params = (nx, ny, dx_step, dy_step, order)

        self._scan_origin = (self.stage.x, self.stage.y)
        self._planned_total = nx * ny
        self._run_auto_capture = bool(self.var_auto_capture.get())
        self._run_review_mode = self.var_review_mode.get()
        self._worker_error = None
        self._close_preview()
        # Record provenance before any movement. A logging failure prevents the scan.
        try:
            self._init_session()
        except Exception as exc:
            self.auto_running = False
            self._history_error = str(exc)
            self.var_status.set("扫描尚未开始：记录创建失败；请查看恢复步骤。")
            self._show_history_recovery()
            return

        self._worker = threading.Thread(
            target=self._auto_scan_worker,
            args=(nx, ny, dx_step, dy_step, order, 0),
            daemon=True,
        )
        self._worker.start()
        self.after(100, self._poll_auto_done)

    def resume_auto(self):
        """Resume scan from where it was stopped. Jumps directly to next point."""
        if getattr(self, "_history_error", None) and self._can_check_log_resume():
            self.on_check_and_resume()
            return
        if not self.stage.connected:
            messagebox.showwarning("Not connected", "Connect stage first")
            return
        if self.auto_running or self._scan_params is None or self._camera_connecting:
            return
        try:
            self._ensure_history()
        except Exception as exc:
            self._resume_history_failed(exc)
            return
        if not self.stage.position_trusted:
            messagebox.showwarning("Position uncertain", "Reconnect and check the field, then start a NEW scan.")
            return
        if self._run_auto_capture and (not self.connected_camera or self.cam is None):
            messagebox.showwarning("Camera required", "Reconnect the camera before RESUME.")
            return

        if self._run_auto_capture:
            with self._camera_lock:
                current_camera = describe_camera(self.cam)
            try:
                self._session_log.record("resume_camera_readback", camera=current_camera)
            except Exception as exc:
                self._resume_history_failed(exc)
                return
            # Driver dimensions may be stale. Check identity here; the worker
            # validates received pixels and the full signature before any move.
            if self._camera_signature(current_camera)[:-1] != self._session_camera_signature[:-1]:
                messagebox.showwarning("Camera changed", "The camera or image size changed. Start a NEW scan; do not mix this session.")
                return
        nx, ny, dx_step, dy_step, order = self._scan_params
        skip = self._scan_done_count  # skip already-completed points
        total = nx * ny

        if skip >= total:
            messagebox.showinfo("Done", "All points already completed.")
            self._scan_params = None
            self.btn_resume.config(state="disabled")
            return

        # Warn if user changed parameters
        cur_nx = max(1, safe_int(self.var_nx.get(), 1))
        cur_ny = max(1, safe_int(self.var_ny.get(), 1))
        if cur_nx != nx or cur_ny != ny:
            if not messagebox.askyesno("Parameters Changed",
                    f"NX/NY changed from {nx}x{ny} to {cur_nx}x{cur_ny}.\n"
                    f"Resume with ORIGINAL {nx}x{ny}?"):
                return

        self._cancel_event.clear()
        self.auto_running = True
        self.btn_auto.config(state="disabled")
        self.btn_resume.config(state="disabled")

        self._close_preview()
        try:
            self._session_log.record("resume", next_point=skip + 1)
        except Exception as exc:
            self._resume_history_failed(exc)
            return
        print(f"[resume] Resuming from point {skip + 1}/{total}")
        self.var_status.set(f"RESUMING from {skip + 1}/{total}...")

        self._worker = threading.Thread(
            target=self._auto_scan_worker,
            args=(nx, ny, dx_step, dy_step, order, skip),
            daemon=True,
        )
        self._worker.start()
        self.after(100, self._poll_auto_done)

    def _poll_auto_done(self):
        """Check if worker thread finished."""
        if self._worker and self._worker.is_alive():
            self.after(100, self._poll_auto_done)
        else:
            self._finalize_auto()

    def _finalize_auto(self):
        self.auto_running = False
        self._worker = None
        self.btn_auto.config(state="normal")
        self.after(0, self._refresh_pos)

        total = self._planned_total
        worker_error = self._worker_error

        if worker_error is not None:
            # Worker crashed — show error, allow resume
            err_msg = str(self._worker_error)
            self.var_status.set(f"ERROR at {self._scan_done_count}/{total}: {err_msg}")
            if self._scan_params is not None and self.stage.position_trusted:
                self.btn_resume.config(state="normal")
            self._worker_error = None
        elif self._cancel_event.is_set():
            remaining = total - self._scan_done_count
            self.var_status.set(f"STOPPED at {self._scan_done_count}/{total} — check stage position before continuing")
            if self._scan_params is not None and self.stage.position_trusted:
                self.btn_resume.config(state="normal")
        else:
            self.var_status.set(f"采集结束：{self._photo_count} 张已保存；仍需核对整幅覆盖")
            self._scan_params = None  # clear resume state
            self.btn_resume.config(state="disabled")

        outcome = "completed" if self._scan_done_count == total else "failed"
        if self._cancel_event.is_set() and worker_error is None and outcome != "completed":
            outcome = "aborted"
        if outcome == "completed" and not self._run_auto_capture:
            outcome = "manual_completed"
        if self._session_log and getattr(self._session_log, "history_error", None):
            self._history_error = self._session_log.history_error
            self._scan_params = None
            self.var_status.set("记录写入失败，扫描暂停；检查保存位置后点“检查并继续”。")
        elif self._session_log:
            try:
                self._session_log.finish(
                    outcome, positions_completed=self._scan_done_count,
                    photos_saved=self._photo_count, planned_positions=total,
                    capture_attempts=getattr(self, "_capture_attempts", 0),
                    capture_failures=getattr(self, "_capture_failures", 0),
                    position_trusted=self.stage.position_trusted,
                    position_trust_basis="controller_step_counter_only_not_physical_metrology",
                    human_confirmed_points=getattr(self, '_human_review_count', 0),
                    spatial_coverage_verified=False,
                    error=None if worker_error is None else str(worker_error))
            except Exception as exc:
                self._history_error = str(exc)
                self._scan_params = None
                self.btn_resume.config(state="disabled")
                self.var_status.set("照片保留，但历史写入不完整；请保留整个文件夹并检查磁盘。")
                self.var_history_status.set("历史写入失败")
        # Save sharpness summary
        if self._sharpness_records and self._session_folder:
            self._save_sharpness_csv()

        self.var_progress.set(f"{self._scan_done_count}/{total}")
        print(f"\n[done] {self._photo_count} photos saved to: {self._session_folder}")
        if getattr(self, "_history_error", None):
            self._log_pause_position = (self.stage.x, self.stage.y)
            self._show_history_recovery()

    def _build_snake_path(self, nx, ny, dx_step, dy_step, order):
        """Pre-compute the full snake scan path as a list of (move_dx, move_dy, row, col).
        First entry has move=(0,0) for the starting point."""
        path = [(0, 0, 0, 0)]  # starting point: no movement

        if order == "X_first":
            for row_idx in range(ny):
                dirx = +1 if row_idx % 2 == 0 else -1
                for cx in range(nx - 1):
                    col = cx + 1 if dirx > 0 else nx - 2 - cx
                    path.append((dirx * dx_step, 0, row_idx, col))
                if row_idx < ny - 1:
                    col = nx - 1 if row_idx % 2 == 0 else 0
                    path.append((0, +dy_step, row_idx + 1, col))
        else:
            for col_idx in range(nx):
                diry = +1 if col_idx % 2 == 0 else -1
                for ry in range(ny - 1):
                    row = ry + 1 if diry > 0 else ny - 2 - ry
                    path.append((0, diry * dy_step, row, col_idx))
                if col_idx < nx - 1:
                    row = ny - 1 if col_idx % 2 == 0 else 0
                    path.append((+dx_step, 0, row, col_idx + 1))
        return path

    def _auto_scan_worker(self, nx, ny, dx_step, dy_step, order, skip=0):
        """Resume a failed photo at its recorded target, without a second relative move."""
        path = self._build_snake_path(nx, ny, dx_step, dy_step, order)
        total = len(path)
        target_x, target_y = self._scan_origin
        try:
            if self._run_auto_capture:
                self._verify_acquisition_mode('resume' if skip else 'pre_scan')
            for i, (mdx, mdy, row, col) in enumerate(path):
                target_x += mdx
                target_y += mdy
                if i < skip:
                    continue
                if self._cancel_event.is_set():
                    return
                self._point_checkpoint = {
                    'phase': 'before_capture', 'point': i + 1, 'row': row, 'col': col,
                    'photo_count_before': getattr(self, '_photo_count', 0),
                    'human_count_before': getattr(self, '_human_review_count', 0),
                    'sharpness_len_before': len(getattr(self, '_sharpness_records', [])),
                }
                dx = target_x - self.stage.x
                dy = target_y - self.stage.y
                if dx or dy:
                    self._session_log.record("move_attempt", row=row, col=col,
                                             target_x_steps=target_x, target_y_steps=target_y,
                                             dx_steps=dx, dy_steps=dy)
                    try:
                        self.stage.move_relative(dx, dy, self._cancel_event)
                    except Exception as exc:
                        self._session_log.record("move_failed", row=row, col=col, error=str(exc),
                                                 verification=getattr(self.stage, "last_move_evidence", {}))
                        raise
                    self._session_log.record("move_completed", row=row, col=col,
                                             x_steps=self.stage.x, y_steps=self.stage.y,
                                             verification=getattr(self.stage, "last_move_evidence", {}))
                    self.after(0, self._refresh_pos)
                if self._cancel_event.is_set():
                    return
                if not self._take_photo_at_point(i + 1, total, row=row, col=col):
                    if self._cancel_event.is_set():
                        return
                    raise RuntimeError("Point did not complete; no point skipped")
                self._scan_done_count = i + 1
        except Exception as exc:
            self._worker_error = exc
            self._log_pause_position = (self.stage.x, self.stage.y)
            print(f"[scan] stopped: {exc}")
            self._cancel_event.set()

    def _save_sharpness_csv(self):
        """Save sharpness records to CSV."""
        import csv
        csv_path = os.path.join(self._session_folder, "sharpness.csv")
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["row", "col", "sharpness", "retries", "filepath"])
                for rec in self._sharpness_records:
                    writer.writerow([rec["row"], rec["col"],
                                     f"{rec['sharpness']:.2f}", rec["retries"],
                                     rec["filepath"]])
            print(f"[report] Sharpness CSV: {csv_path}")
        except OSError as e:
            print(f"[report] Failed to save CSV: {e}")

    # --------------------------------------------------------
    # Position display
    # --------------------------------------------------------
    def _refresh_pos(self):
        x, y = self.stage.x, self.stage.y
        self.var_pos_steps.set(f"控制器核对的逻辑步数：X={x}, Y={y}")
        self.var_pos_um.set(f"按当前标定换算 um：X={steps_to_um(x):.2f}, Y={steps_to_um(y):.2f}")
        self.var_pos_px.set("控制器计数不等于真实机械位移；请结合图像地标确认。")

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------
    def _on_close(self):
        if self.auto_running:
            if not messagebox.askyesno("Confirm Exit", "Stop the scan and exit?"):
                return
            self.on_emergency_stop()
        self._finish_close_when_idle()

    def _finish_close_when_idle(self):
        if self._camera_connecting or (self._worker and self._worker.is_alive()):
            self.after(100, self._finish_close_when_idle)
            return
        self.auto_running = False
        self._close_preview()
        if self._disconnect_error:
            self.var_status.set("断开失败，请检查设备后重试：" + self._disconnect_error)
            return
        if self.connected_camera or self.stage.connected:
            self.on_disconnect()
            self.after(100, self._finish_close_when_idle)
            return
        if self._shared_history is not None:
            try:
                self._shared_history.close("closed", source="window_close", devices_disconnected=True,
                                           recording_error=self._history_error)
            except Exception as exc:
                print("[history] Could not record clean close: " + str(exc))
        self.destroy()


# ============================================================
# main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="XY Mosaic Navigator + Auto Camera")
    p.add_argument("--sim", action="store_true", help="Simulate STAGE ONLY; camera stays real if connected. Not hardware validation.")
    return p.parse_args()


_app_ref = None  # global ref for signal handler

def _safe_shutdown():
    """Stop motors, close devices, remove lockfile."""
    global _app_ref
    if _app_ref is not None:
        try:
            _app_ref.stage.emergency_stop(wait=False)
        except Exception:
            pass
        try:
            _app_ref.stage.disconnect()
        except Exception:
            pass
    _remove_lockfile()


if __name__ == "__main__":
    args = parse_args()

    _write_lockfile()
    atexit.register(_safe_shutdown)

    def _sigint_handler(sig, frame):
        print("\n[Ctrl+C] Shutting down...")
        _safe_shutdown()
        sys.exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    app = MosaicNavigatorGUI(simulate=args.sim)
    _app_ref = app
    app.mainloop()
