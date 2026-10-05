# -*- coding: utf-8 -*-
"""
Universal Camera Controller v2
-------------------------------
Supports AmScope / ToupTek SDK cameras AND UVC cameras (including
YW MS2300D) with full OpenCV-based property control.

Supports multiple backends (auto-detected in order):
  1. amcam   — AmScope SDK  (amcam.py + amcam.dll)
  2. toupcam — ToupTek SDK  (toupcam.py + toupcam.dll)
  3. opencv  — OpenCV with UVC controls for USB cameras (e.g. YW MS2300D)

Tested cameras:
  - AmScope AF408N (4K Auto-Focus HDMI)
  - AmScope MU1803 (USB microscope camera)
  - YW MS2300D (2MP UVC microscope camera, 1600x1200)
  - Any ToupTek OEM camera
  - Any UVC camera via OpenCV

v2 Changes from Camera.py:
  - OpenCV backend now supports exposure, gain, brightness, contrast,
    saturation, white balance, gamma, and sharpness control via UVC/DirectShow
  - Added YW MS2300D camera profile with optimal defaults
  - Enhanced print_capabilities() for UVC cameras
  - preview_and_tune() keyboard controls work for UVC cameras too

Usage:
    from Camera_v2 import Camera

    cam = Camera()                   # auto-detect backend
    cam.open()
    cam.print_capabilities()         # show what this camera supports
    cam.preview_and_tune()           # interactive preview window
    frame = cam.snap_image()         # capture one frame (numpy BGR)
    cam.save_image(frame, "pic.jpg")
    cam.close()
"""

# =========================================================================
# ★★★ STUDENT CONFIGURATION AREA / 学生配置区 ★★★
# =========================================================================
#
# >>> ONLY MODIFY THE VALUES BELOW — DO NOT CHANGE CODE AFTER THIS SECTION <<<
# >>> 只修改下面的值 — 不要修改本区域之后的任何代码 <<<
#
# -------------------------------------------------------------------------
# Camera selection / 相机选择
# -------------------------------------------------------------------------
# CAMERA_INDEX : Which camera to use (0 = first camera found).
#                使用哪个相机（0 = 第一个检测到的相机）。
#                If you have multiple cameras, try 0, 1, 2, ...
#                如果有多台相机，依次尝试 0, 1, 2, ...
CAMERA_INDEX = 0
#
# BACKEND : Camera connection type / 相机连接方式
#   "auto"   — Auto-detect (try SDK first, then OpenCV)
#              自动检测（优先 SDK，其次 OpenCV）
#   "opencv" — Force OpenCV (for UVC cameras, e.g. YW MS2300D, AF408N)
#              强制使用 OpenCV（用于 UVC 相机，如 YW MS2300D、AF408N）
#   "amcam"  — Force AmScope SDK (for USB cameras, e.g. MU1803)
#              强制使用 AmScope SDK（用于 USB 相机，如 MU1803）
BACKEND = "auto"
#
# -------------------------------------------------------------------------
# Image settings / 图像设置
# -------------------------------------------------------------------------
# BITS : Color depth / 色彩深度
#   24 — Color image (BGR) / 彩色图像
#    8 — Grayscale image / 灰度图像
BITS = 24
#
# RESOLUTION : Desired image resolution / 期望的图像分辨率
#   None         — Use camera default / 使用相机默认分辨率
#   (3840, 2160) — 4K
#   (1920, 1080) — 1080p (Full HD)
#   (1600, 1200) — UXGA (YW MS2300D max)
#   (1280, 720)  — 720p
RESOLUTION = None
#
# -------------------------------------------------------------------------
# Auto-save settings / 自动存图设置
# -------------------------------------------------------------------------
# SAVE_FOLDER : Where to save captured images / 图片保存路径
#   Use "." for current folder / 使用 "." 表示当前文件夹
SAVE_FOLDER = "."
#
# SAVE_FORMAT : Image format / 图片格式
#   ".png"  — Lossless (larger file) / 无损（文件较大）
#   ".jpg"  — Compressed (smaller file) / 有损压缩（文件较小）
#   ".tiff" — Lossless (for scientific use) / 无损（科研用途）
SAVE_FORMAT = ".png"
#
# JPEG_QUALITY : JPEG quality 1-100 (only used when SAVE_FORMAT=".jpg")
#                JPEG 质量 1-100（仅在格式为 .jpg 时有效）
JPEG_QUALITY = 95
#
# =========================================================================
# ★★★ END OF STUDENT CONFIGURATION / 学生配置区结束 ★★★
# =========================================================================
# >>> DO NOT MODIFY ANYTHING BELOW THIS LINE <<<
# >>> 请勿修改以下任何内容 <<<
# =========================================================================

import ctypes
import os
import sys
import threading
import time
import numpy as np

# Resolve SAVE_FOLDER relative to the program directory (not cwd)
_PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(SAVE_FOLDER):
    SAVE_FOLDER = os.path.join(_PROGRAM_DIR, SAVE_FOLDER)

# =========================================================================
# Backend auto-detection
# =========================================================================
_sdk = None        # the imported SDK module (amcam or toupcam)
_sdk_name = None   # "amcam" | "toupcam" | "opencv"
_FLAG = {}         # SDK flag constants mapped to generic names
_EVENT = {}        # SDK event constants mapped to generic names

def _try_import_sdk():
    """Try amcam -> toupcam -> opencv, set module-level _sdk / _sdk_name."""
    global _sdk, _sdk_name, _FLAG, _EVENT

    # --- Try amcam ---
    try:
        import amcam as _m
        _m.Amcam.EnumV2  # verify it loaded properly
        _sdk = _m
        _sdk_name = "amcam"
        _FLAG = {k.replace("AMCAM_FLAG_", ""): getattr(_m, k)
                 for k in dir(_m) if k.startswith("AMCAM_FLAG_")}
        _EVENT = {k.replace("AMCAM_EVENT_", ""): getattr(_m, k)
                  for k in dir(_m) if k.startswith("AMCAM_EVENT_")}
        return
    except Exception:
        pass

    # --- Try toupcam ---
    try:
        import toupcam as _m
        _m.Toupcam.EnumV2
        _sdk = _m
        _sdk_name = "toupcam"
        _FLAG = {k.replace("TOUPCAM_FLAG_", ""): getattr(_m, k)
                 for k in dir(_m) if k.startswith("TOUPCAM_FLAG_")}
        _EVENT = {k.replace("TOUPCAM_EVENT_", ""): getattr(_m, k)
                  for k in dir(_m) if k.startswith("TOUPCAM_EVENT_")}
        return
    except Exception:
        pass

    # --- Fallback: OpenCV ---
    _sdk = None
    _sdk_name = "opencv"

_try_import_sdk()

# OpenCV — always try to import (used for preview/save even with SDK backend)
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _sdk_class():
    """Return the SDK camera class (Amcam or Toupcam)."""
    if _sdk_name == "amcam":
        return _sdk.Amcam
    elif _sdk_name == "toupcam":
        return _sdk.Toupcam
    return None


def _event(name: str):
    """Lookup a generic event constant by short name (e.g. 'IMAGE')."""
    return _EVENT.get(name)


def _flag(name: str):
    """Lookup a generic flag constant by short name (e.g. 'TEC')."""
    return _FLAG.get(name, 0)


def _opencv_api():
    """Return the platform-appropriate OpenCV capture API."""
    if not _HAS_CV2:
        return None
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


# =========================================================================
# UVC Property Helper
# =========================================================================
# OpenCV CAP_PROP IDs for UVC camera controls via DirectShow / V4L2.
# These allow controlling exposure, gain, brightness, etc. on UVC cameras
# like the YW MS2300D without needing a proprietary SDK.

_UVC_PROPS = {
    "brightness":     cv2.CAP_PROP_BRIGHTNESS     if _HAS_CV2 else 10,
    "contrast":       cv2.CAP_PROP_CONTRAST       if _HAS_CV2 else 11,
    "saturation":     cv2.CAP_PROP_SATURATION     if _HAS_CV2 else 12,
    "gain":           cv2.CAP_PROP_GAIN           if _HAS_CV2 else 14,
    "exposure":       cv2.CAP_PROP_EXPOSURE       if _HAS_CV2 else 15,
    "white_balance":  cv2.CAP_PROP_WB_TEMPERATURE if _HAS_CV2 else 45,
    "auto_exposure":  cv2.CAP_PROP_AUTO_EXPOSURE  if _HAS_CV2 else 21,
    "auto_wb":        cv2.CAP_PROP_AUTO_WB        if _HAS_CV2 else 44,
    "sharpness":      cv2.CAP_PROP_SHARPNESS      if _HAS_CV2 else 20,
    "gamma":          cv2.CAP_PROP_GAMMA          if _HAS_CV2 else 22,
    "backlight":      cv2.CAP_PROP_BACKLIGHT      if _HAS_CV2 else 32,
    "focus":          cv2.CAP_PROP_FOCUS          if _HAS_CV2 else 28,
    "auto_focus":     cv2.CAP_PROP_AUTOFOCUS      if _HAS_CV2 else 39,
}


def _uvc_get(cap, prop_name):
    """Read a UVC property. Returns None if unsupported."""
    prop_id = _UVC_PROPS.get(prop_name)
    if prop_id is None or cap is None:
        return None
    val = cap.get(prop_id)
    # OpenCV returns 0.0 for unsupported props — but 0.0 can also be valid.
    # We return it and let the caller interpret.
    return val


def _uvc_set(cap, prop_name, value):
    """Set a UVC property. Returns True if the driver accepted it."""
    prop_id = _UVC_PROPS.get(prop_name)
    if prop_id is None or cap is None:
        return False
    return cap.set(prop_id, value)


def _uvc_probe(cap):
    """Probe which UVC properties this camera actually supports.
    Returns dict of {prop_name: current_value} for properties that respond."""
    supported = {}
    if cap is None:
        return supported
    for name, prop_id in _UVC_PROPS.items():
        try:
            val = cap.get(prop_id)
            # Try setting it to same value — if it doesn't raise, it's supported
            if cap.set(prop_id, val):
                supported[name] = val
        except Exception:
            pass
    return supported


# =========================================================================
# Unified Camera class
# =========================================================================
class Camera:
    """
    Universal camera controller (v2).

    When an amcam/toupcam SDK is available the native SDK is used for
    full hardware control.  When no SDK is found (or backend="opencv"),
    uses OpenCV with UVC property controls for cameras like the YW MS2300D.
    """

    def __init__(self, camera_index: int = 0, bits: int = 24, backend: str = "auto",
                 resolution: tuple = None):
        """
        Parameters
        ----------
        camera_index : int
            Index of the camera to open (0 = first found).
        bits : int
            Bits per pixel for SDK capture (24 = BGR, 8 = gray). Ignored for OpenCV.
        backend : str
            "auto" (default), "amcam", "toupcam", or "opencv".
            "auto" tries SDK first, falls back to OpenCV if no SDK camera found.
        resolution : tuple (width, height) or None
            Request specific resolution (e.g. (1600, 1200) for YW MS2300D max).
            For SDK backend: selects the closest matching resolution index.
            For OpenCV backend: sets CAP_PROP width/height.
        """
        self.camera_index = camera_index
        self.bits = bits
        self._requested_backend = backend
        self._backend = backend if backend != "auto" else _sdk_name
        self._requested_resolution = resolution

        # SDK backend state
        self.hcam = None
        self._dev_info = None      # AmcamDeviceV2 / ToupcamDeviceV2
        self._buf = None
        self._width = 0
        self._height = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest_frame = None
        self._running = False
        self._frame_seq = 0
        self._in_callback = 0      # refcount: how many callbacks are in-flight
        self._disconnected = False
        self._last_error = None

        # OpenCV backend state
        self._cv_cap = None
        self._uvc_supported = {}   # {prop_name: current_value} — probed on open
        self._opencv_read_sequence = 0
        # Diagnostics for the last returned OpenCV frame, cleared on failure.
        # These are host read times/counts, NOT sensor exposure times/frame IDs.
        self.last_capture_metadata = None

    @property
    def backend(self) -> str:
        return self._backend

    # ==================================================================
    # Core: open / close
    # ==================================================================
    def open(self):
        if self.is_open():
            raise RuntimeError("[Camera] Already open. Call close() first.")
        self._disconnected = False
        self._last_error = None
        if self._backend in ("amcam", "toupcam"):
            try:
                self._open_sdk()
            except RuntimeError as e:
                if self._requested_backend == "auto":
                    print(f"[Camera] SDK failed ({e}), falling back to OpenCV...")
                    self._backend = "opencv"
                    self._open_opencv()
                else:
                    raise
        else:
            self._open_opencv()

    def close(self):
        if self._backend in ("amcam", "toupcam"):
            self._close_sdk()
        else:
            self._close_opencv()

    def is_open(self) -> bool:
        if self._backend in ("amcam", "toupcam"):
            return self.hcam is not None
        return self._cv_cap is not None and self._cv_cap.isOpened()

    # ------------------------------------------------------------------
    # SDK open / close
    # ------------------------------------------------------------------
    def _open_sdk(self):
        cls = _sdk_class()
        devices = cls.EnumV2()
        if len(devices) == 0:
            raise RuntimeError(f"[{self._backend}] No camera found.")
        if self.camera_index >= len(devices):
            raise RuntimeError(
                f"[{self._backend}] Camera index {self.camera_index} out of range "
                f"(found {len(devices)} camera(s))."
            )

        self._dev_info = devices[self.camera_index]
        dev = self._dev_info
        print(f"[Camera] Backend: {self._backend}")
        print(f"[Camera] Found: {dev.displayname}")
        print(f"[Camera] Model: {dev.model.name}")
        print(f"[Camera] Resolutions:")
        for i, r in enumerate(dev.model.res):
            print(f"         [{i}] {r.width} x {r.height}")

        self.hcam = cls.Open(dev.id)
        if self.hcam is None:
            raise RuntimeError(f"[{self._backend}] Failed to open camera.")

        # Select requested resolution if specified
        if self._requested_resolution is not None:
            rw, rh = self._requested_resolution
            best_idx, best_diff = 0, float('inf')
            for i, r in enumerate(dev.model.res):
                diff = abs(r.width - rw) + abs(r.height - rh)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            self.hcam.put_eSize(best_idx)
            print(f"[Camera] Selected resolution index [{best_idx}]")

        self._width, self._height = self.hcam.get_Size()
        self._alloc_buf()
        print(f"[Camera] Opened at {self._width} x {self._height}")

    def _close_sdk(self):
        self.stop_live()
        with self._cond:
            if self.hcam is None:
                return
            # Wait for any in-flight callback to finish
            while self._in_callback > 0:
                self._cond.wait(timeout=2.0)
            self.hcam.Close()
            self.hcam = None
            self._buf = None
            self._dev_info = None
            self._latest_frame = None
        print("[Camera] Closed.")

    # ------------------------------------------------------------------
    # OpenCV open / close
    # ------------------------------------------------------------------
    def _open_opencv(self):
        if not _HAS_CV2:
            raise RuntimeError("OpenCV (cv2) is not installed.")
        print("[Camera] Backend: opencv (UVC)")

        # Try platform-preferred API first, then default
        api = _opencv_api()
        self._cv_cap = cv2.VideoCapture(self.camera_index, api)
        if not self._cv_cap.isOpened():
            self._cv_cap = cv2.VideoCapture(self.camera_index)
        if not self._cv_cap.isOpened():
            raise RuntimeError("[opencv] Could not open camera.")

        # Set MJPEG codec first — some devices need this before high-res negotiation
        self._cv_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Set requested resolution (important for YW MS2300D max 1600x1200)
        if self._requested_resolution is not None:
            rw, rh = self._requested_resolution
            self._cv_cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
            self._cv_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
            print(f"[Camera] Requested resolution: {rw} x {rh}")

        self._width = int(self._cv_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Opened at {self._width} x {self._height}")

        # Warn if actual resolution differs from requested
        if self._requested_resolution is not None:
            rw, rh = self._requested_resolution
            if self._width != rw or self._height != rh:
                print(f"[Camera] WARNING: got {self._width}x{self._height} "
                      f"instead of requested {rw}x{rh}")

        # Probe UVC controls
        self._uvc_supported = _uvc_probe(self._cv_cap)
        if self._uvc_supported:
            props = ", ".join(sorted(self._uvc_supported.keys()))
            print(f"[Camera] UVC controls available: {props}")
        else:
            print("[Camera] No UVC controls detected (HDMI capture or limited driver).")

    def _close_opencv(self):
        if self._cv_cap is not None:
            self._cv_cap.release()
            self._cv_cap = None
            self._uvc_supported = {}
            print("[Camera] Closed.")

    # ==================================================================
    # UVC control helpers (OpenCV backend)
    # ==================================================================
    def _has_uvc(self, prop_name: str) -> bool:
        """Check if a UVC property is available."""
        return prop_name in self._uvc_supported

    def _get_uvc(self, prop_name: str):
        """Get a UVC property value. Returns None if unavailable."""
        if not self._has_uvc(prop_name):
            return None
        return _uvc_get(self._cv_cap, prop_name)

    def _set_uvc(self, prop_name: str, value) -> bool:
        """Set a UVC property. Returns True on success."""
        if self._cv_cap is None:
            return False
        ok = _uvc_set(self._cv_cap, prop_name, value)
        if ok:
            self._uvc_supported[prop_name] = value
        return ok

    # ==================================================================
    # Capabilities query
    # ==================================================================
    def _has_flag(self, name: str) -> bool:
        if self._dev_info is None:
            return False
        return bool(self._dev_info.model.flag & _flag(name))

    def print_capabilities(self):
        """Print detected camera capabilities based on hardware flags or UVC probing."""
        if self._backend == "opencv":
            print(f"\n=== Camera Capabilities (OpenCV/UVC) ===")
            print(f"  Resolution: {self._width} x {self._height}")
            fourcc_int = int(self._cv_cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
            fps = self._cv_cap.get(cv2.CAP_PROP_FPS)
            print(f"  Codec: {fourcc_str}")
            print(f"  FPS: {fps:.1f}")

            if self._uvc_supported:
                print(f"\n  --- UVC Controls ---")
                uvc_labels = {
                    "brightness":    "Brightness",
                    "contrast":      "Contrast",
                    "saturation":    "Saturation",
                    "gain":          "Gain",
                    "exposure":      "Exposure",
                    "white_balance": "White Balance Temperature",
                    "auto_exposure": "Auto Exposure",
                    "auto_wb":       "Auto White Balance",
                    "sharpness":     "Sharpness",
                    "gamma":         "Gamma",
                    "backlight":     "Backlight Compensation",
                    "focus":         "Focus",
                    "auto_focus":    "Auto Focus",
                }
                for prop_name in sorted(self._uvc_supported.keys()):
                    label = uvc_labels.get(prop_name, prop_name)
                    val = _uvc_get(self._cv_cap, prop_name)
                    print(f"  [+] {label}: {val}")
            else:
                print("  (No UVC controls — HDMI capture or limited driver)")
            print()
            return

        print(f"\n=== Camera Capabilities: {self._dev_info.displayname} ===")
        caps = [
            ("CMOS",               "CMOS sensor"),
            ("CCD_PROGRESSIVE",    "Progressive CCD sensor"),
            ("CCD_INTERLACED",     "Interlaced CCD sensor"),
            ("MONO",               "Monochromatic"),
            ("USB30",              "USB 3.0"),
            ("USB30_OVER_USB20",   "USB 3.0 on USB 2.0 port (!)"),
            ("ROI_HARDWARE",       "Hardware ROI"),
            ("BINSKIP_SUPPORTED",  "Bin/Skip mode"),
            ("TEC",                "Thermoelectric Cooler"),
            ("TEC_ONOFF",          "TEC on/off control"),
            ("GETTEMPERATURE",     "Temperature readout"),
            ("FAN",                "Cooling fan"),
            ("TRIGGER_SOFTWARE",   "Software trigger"),
            ("TRIGGER_EXTERNAL",   "External trigger"),
            ("TRIGGER_SINGLE",     "Single trigger only"),
            ("AUTO_FOCUS",         "Auto focus"),
            ("FOCUSMOTOR",         "Focus motor"),
            ("BLACKLEVEL",         "Black level control"),
            ("BUFFER",             "Frame buffer"),
            ("DDR",                "DDR frame buffer"),
            ("CG",                 "Conversion Gain (HCG/LCG)"),
            ("CGHDR",              "Conversion Gain HDR"),
            ("GLOBALSHUTTER",      "Global shutter"),
            ("ISP",                "ISP chip"),
            ("HEAT",               "Anti-fog heater"),
            ("LOW_NOISE",          "Low noise mode"),
            ("PRECISE_FRAMERATE",  "Precise framerate control"),
            ("RAW8",               "RAW 8-bit"),
            ("RAW10",              "RAW 10-bit"),
            ("RAW12",              "RAW 12-bit"),
            ("RAW14",              "RAW 14-bit"),
            ("RAW16",              "RAW 16-bit"),
            ("RGB888",             "RGB 888"),
        ]
        for flag_name, desc in caps:
            if self._has_flag(flag_name):
                print(f"  [+] {desc}")

        # Exposure range
        try:
            emin, emax, edef = self.get_exposure_range()
            print(f"  Exposure range: {emin} ~ {emax} us (default {edef})")
        except Exception:
            pass

        # Gain range
        try:
            gmin, gmax, gdef = self.get_gain_range()
            print(f"  Gain range: {gmin} ~ {gmax} % (default {gdef})")
        except Exception:
            pass
        print()

    # ==================================================================
    # Resolution
    # ==================================================================
    def set_resolution(self, index: int = None, width: int = None, height: int = None):
        """
        Set resolution by index (SDK) or by width/height (both backends).

        For YW MS2300D, max resolution is 1600x1200:
            cam.set_resolution(width=1600, height=1200)
        """
        if not self.is_open():
            raise RuntimeError("[Camera] Camera is not open.")
        if self._backend in ("amcam", "toupcam"):
            with self._lock:
                was_running = self._running
            self.stop_live()
            try:
                if index is not None:
                    self.hcam.put_eSize(index)
                elif width is not None and height is not None:
                    dev = self._dev_info
                    best_idx, best_diff = 0, float('inf')
                    for i, r in enumerate(dev.model.res):
                        diff = abs(r.width - width) + abs(r.height - height)
                        if diff < best_diff:
                            best_diff = diff
                            best_idx = i
                    self.hcam.put_eSize(best_idx)
                self._width, self._height = self.hcam.get_Size()
                self._alloc_buf()
            finally:
                if was_running:
                    self.start_live()
        else:
            if width is not None and height is not None:
                self._cv_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cv_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._width = int(self._cv_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._height = int(self._cv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if self._width != width or self._height != height:
                    print(f"[Camera] WARNING: requested {width}x{height} "
                          f"but got {self._width}x{self._height}.")
            else:
                print("[Camera] OpenCV backend: use width/height instead of index.")
                return
        print(f"[Camera] Resolution: {self._width} x {self._height}")

    def get_resolution(self):
        return self._width, self._height

    # ==================================================================
    # Exposure
    # ==================================================================
    def set_auto_exposure(self, enable: bool):
        """Enable/disable auto-exposure.
        SDK: uses native auto-exposure.
        OpenCV/UVC: uses DirectShow auto-exposure control.
          - auto_exposure=0.75 means manual, 0.25 means auto (DirectShow convention)
        """
        if self._backend == "opencv":
            if self._has_uvc("auto_exposure"):
                # DirectShow: 0.25 = auto, 0.75 = manual
                val = 0.25 if enable else 0.75
                self._set_uvc("auto_exposure", val)
                print(f"[Camera] Auto-exposure: {'ON' if enable else 'OFF'}")
            else:
                print("[Camera] Auto-exposure not supported by this camera/driver.")
            return
        self.hcam.put_AutoExpoEnable(1 if enable else 0)

    def get_exposure_time(self) -> float:
        """Get exposure time.
        SDK: returns microseconds (int).
        OpenCV/UVC: returns DirectShow exposure value (log2 scale, e.g. -5 = 1/32s).
        """
        if self._backend == "opencv":
            val = self._get_uvc("exposure")
            return val if val is not None else 0
        return self.hcam.get_ExpoTime()

    def set_exposure_time(self, value):
        """Set exposure time.
        SDK: value in microseconds.
        OpenCV/UVC: DirectShow log2 value (e.g. -1, -2, ..., -13).
            Common values for YW MS2300D:
              -1 = 1/2 s,  -5 = 1/32 s,  -9 = 1/512 s,  -13 = 1/8192 s
        """
        if self._backend == "opencv":
            # Disable auto-exposure first
            if self._has_uvc("auto_exposure"):
                self._set_uvc("auto_exposure", 0.75)  # manual mode
            if self._has_uvc("exposure"):
                self._set_uvc("exposure", value)
                print(f"[Camera] Exposure set to {value}")
            else:
                print("[Camera] Exposure control not supported by this camera/driver.")
            return
        prev_auto = self.hcam.get_AutoExpoEnable()
        self.hcam.put_AutoExpoEnable(0)
        try:
            self.hcam.put_ExpoTime(value)
        except Exception:
            self.hcam.put_AutoExpoEnable(prev_auto)
            raise

    def get_exposure_range(self):
        """Get exposure range.
        SDK: returns (min_us, max_us, default_us).
        OpenCV/UVC: returns typical DirectShow range (-13, -1, -5).
        """
        if self._backend == "opencv":
            if self._has_uvc("exposure"):
                return (-13, -1, -5)
            return (0, 0, 0)
        return self.hcam.get_ExpTimeRange()

    # ==================================================================
    # Gain
    # ==================================================================
    def get_gain(self) -> float:
        if self._backend == "opencv":
            val = self._get_uvc("gain")
            return val if val is not None else 0
        return self.hcam.get_ExpoAGain()

    def set_gain(self, value):
        """Set gain.
        SDK: value is percentage (0-100).
        OpenCV/UVC: value depends on driver (typically 0-255 or 0-100).
        """
        if self._backend == "opencv":
            if self._has_uvc("gain"):
                self._set_uvc("gain", value)
                print(f"[Camera] Gain set to {value}")
            else:
                print("[Camera] Gain control not supported by this camera/driver.")
            return
        self.hcam.put_ExpoAGain(value)

    def get_gain_range(self):
        if self._backend == "opencv":
            if self._has_uvc("gain"):
                return (0, 255, 64)
            return (0, 0, 0)
        return self.hcam.get_ExpoAGainRange()

    # ==================================================================
    # White balance
    # ==================================================================
    def auto_white_balance(self):
        """Trigger auto white balance.
        SDK: uses native AWB.
        OpenCV/UVC: enables auto WB (one-shot not available, toggles auto mode).
        """
        if self._backend == "opencv":
            if self._has_uvc("auto_wb"):
                self._set_uvc("auto_wb", 1)
                print("[Camera] Auto white balance enabled.")
            else:
                print("[Camera] AWB not supported by this camera/driver.")
            return
        if self._has_flag("MONO"):
            print("[Camera] Monochrome camera — AWB skipped.")
            return
        self.hcam.AwbOnce()

    def set_temp_tint(self, temp: int, tint: int = None):
        """Set white balance temperature.
        SDK: uses native temp/tint control.
        OpenCV/UVC: sets WB temperature (tint is ignored, temp in Kelvin).
            Typical range for UVC: 2800-6500K
        """
        if self._backend == "opencv":
            # Disable auto WB first
            if self._has_uvc("auto_wb"):
                self._set_uvc("auto_wb", 0)
            if self._has_uvc("white_balance"):
                self._set_uvc("white_balance", temp)
                print(f"[Camera] White balance: {temp} K")
            else:
                print("[Camera] White balance control not supported.")
            return
        self.hcam.put_TempTint(temp, tint)

    def get_temp_tint(self):
        """Get white balance.
        SDK: returns (temp, tint).
        OpenCV/UVC: returns (wb_temperature, 0).
        """
        if self._backend == "opencv":
            val = self._get_uvc("white_balance")
            return (int(val) if val is not None else 0, 0)
        return self.hcam.get_TempTint()

    # ==================================================================
    # Temperature (SDK, cameras with TEC)
    # ==================================================================
    def get_temperature(self) -> float:
        if self._backend == "opencv" or not self._has_flag("GETTEMPERATURE"):
            return float('nan')
        raw = self.hcam.get_Temperature()
        return raw / 10.0

    def set_tec_target(self, celsius: float):
        if self._backend == "opencv" or not self._has_flag("TEC_ONOFF"):
            print("[Camera] TEC control not available.")
            return
        self.hcam.put_Temperature(int(celsius * 10))

    # ==================================================================
    # ROI (SDK, cameras with ROI_HARDWARE)
    # ==================================================================
    def set_roi(self, x: int, y: int, w: int, h: int):
        if self._backend == "opencv":
            print("[Camera] Hardware ROI not available in OpenCV backend. "
                  "Use software cropping instead.")
            return
        self.hcam.put_Roi(x, y, w, h)
        print(f"[Camera] ROI set to ({x}, {y}, {w}, {h})")

    def get_roi(self):
        if self._backend == "opencv":
            return (0, 0, self._width, self._height)
        return self.hcam.get_Roi()

    # ==================================================================
    # Trigger (SDK, cameras with TRIGGER_SOFTWARE)
    # ==================================================================
    def software_trigger(self, count: int = 1):
        if self._backend == "opencv" or not self._has_flag("TRIGGER_SOFTWARE"):
            print("[Camera] Software trigger not available.")
            return
        self.hcam.Trigger(count)

    # ==================================================================
    # Image enhancement
    # ==================================================================
    def set_brightness(self, val):
        """Set brightness. Works on both SDK and UVC cameras."""
        if self._backend == "opencv":
            if self._has_uvc("brightness"):
                self._set_uvc("brightness", val)
                print(f"[Camera] Brightness set to {val}")
            else:
                print("[Camera] Brightness control not supported.")
            return
        self.hcam.put_Brightness(val)

    def get_brightness(self):
        if self._backend == "opencv":
            val = self._get_uvc("brightness")
            return val if val is not None else 0
        return 0

    def set_contrast(self, val):
        """Set contrast. Works on both SDK and UVC cameras."""
        if self._backend == "opencv":
            if self._has_uvc("contrast"):
                self._set_uvc("contrast", val)
                print(f"[Camera] Contrast set to {val}")
            else:
                print("[Camera] Contrast control not supported.")
            return
        self.hcam.put_Contrast(val)

    def get_contrast(self):
        if self._backend == "opencv":
            val = self._get_uvc("contrast")
            return val if val is not None else 0
        return 0

    def set_gamma(self, val):
        """Set gamma. Works on both SDK and UVC cameras."""
        if self._backend == "opencv":
            if self._has_uvc("gamma"):
                self._set_uvc("gamma", val)
                print(f"[Camera] Gamma set to {val}")
            else:
                print("[Camera] Gamma control not supported.")
            return
        self.hcam.put_Gamma(val)

    def get_gamma(self):
        if self._backend == "opencv":
            val = self._get_uvc("gamma")
            return val if val is not None else 0
        return 0

    def set_saturation(self, val):
        """Set saturation. Works on both SDK and UVC cameras."""
        if self._backend == "opencv":
            if self._has_uvc("saturation"):
                self._set_uvc("saturation", val)
                print(f"[Camera] Saturation set to {val}")
            else:
                print("[Camera] Saturation control not supported.")
            return
        self.hcam.put_Saturation(val)

    def get_saturation(self):
        if self._backend == "opencv":
            val = self._get_uvc("saturation")
            return val if val is not None else 0
        return 0

    def set_sharpness(self, val):
        """Set sharpness (UVC cameras only, not available on SDK cameras)."""
        if self._backend == "opencv":
            if self._has_uvc("sharpness"):
                self._set_uvc("sharpness", val)
                print(f"[Camera] Sharpness set to {val}")
            else:
                print("[Camera] Sharpness control not supported.")
            return
        print("[Camera] Sharpness not available via SDK (use post-processing).")

    def get_sharpness(self):
        if self._backend == "opencv":
            val = self._get_uvc("sharpness")
            return val if val is not None else 0
        return 0

    def flip(self, horizontal: bool = False, vertical: bool = False):
        if self._backend == "opencv":
            print("[Camera] Flip not available via UVC — use software flip after capture.")
            return
        self.hcam.put_HFlip(horizontal)
        self.hcam.put_VFlip(vertical)

    # ==================================================================
    # Image capture
    # ==================================================================
    def start_live(self):
        if self._backend in ("amcam", "toupcam"):
            with self._lock:
                if self._running:
                    return
                self._running = True
            try:
                self.hcam.StartPullModeWithCallback(self._on_event, self)
            except Exception:
                with self._lock:
                    self._running = False
                raise
        # OpenCV is always "live" when cap is opened

    def stop_live(self):
        if self._backend in ("amcam", "toupcam"):
            with self._lock:
                if not self._running or self.hcam is None:
                    self._running = False
                    return
                self._running = False
            self.hcam.Stop()

    @property
    def is_streaming(self) -> bool:
        if self._backend == "opencv":
            return self._cv_cap is not None and self._cv_cap.isOpened()
        with self._lock:
            return self._running

    @staticmethod
    def _on_event(event, ctx):
        ev_image = _event("IMAGE")
        ev_disconnected = _event("DISCONNECTED")
        ev_error = _event("ERROR")
        if event == ev_image:
            ctx._pull_image()
        elif event == ev_disconnected:
            print("[Camera] WARNING: camera disconnected!")
            with ctx._cond:
                ctx._disconnected = True
                ctx._last_error = "Camera disconnected"
                ctx._cond.notify_all()
        elif event == ev_error:
            print("[Camera] ERROR event received.")
            with ctx._cond:
                ctx._last_error = "SDK error event"
                ctx._cond.notify_all()

    def _pull_image(self):
        with self._cond:
            hcam = self.hcam
            buf = self._buf
            width = self._width
            height = self._height
            if hcam is None or buf is None:
                return
            self._in_callback += 1
        try:
            hcam.PullImageWithRowPitchV2(buf, self.bits, -1, None)
            if self.bits == 24:
                frame = np.frombuffer(buf, dtype=np.uint8
                    ).reshape(height, width, 3).copy()
            else:
                frame = np.frombuffer(buf, dtype=np.uint8
                    ).reshape(height, width).copy()
            with self._cond:
                self._latest_frame = frame
                self._frame_seq += 1
                self._last_error = None
                self._cond.notify_all()
        except Exception as ex:
            print(f"[Camera] Pull image failed: {ex}")
            with self._cond:
                self._last_error = str(ex)
                self._cond.notify_all()
        finally:
            with self._cond:
                self._in_callback -= 1
                self._cond.notify_all()

    def grab_frame(self, timeout: float = 5.0, flush: int = 3):
        """
        Return the next frame as a numpy array (BGR/gray).

        Args:
            timeout: Max wait time (seconds). OpenCV checks an elapsed deadline
                     between reads and rejects a frame received after it. A
                     blocking driver read cannot be interrupted by this limit.
            flush: Number of frames to discard before reading (OpenCV only).
                   Set to 0 for live preview to avoid latency.
                   Set to 10+ for capture after stage movement.

        For OpenCV, last_capture_metadata describes the successful returned
        frame using host read times/counts, not sensor exposure timestamps.
        A consumed buffer is not proof of exposure after stage settling.
        """
        if self._backend == "opencv":
            self.last_capture_metadata = None
            if self._cv_cap is None:
                return None
            flush = max(0, int(flush))
            deadline = time.monotonic() + max(0.0, float(timeout))
            for read_index in range(flush + 1):
                read_started = time.monotonic()
                if read_started >= deadline:
                    return None
                # DirectShow grab() can be a no-op: retrieve()/read() consumes
                # the pending sample and resets its event. Discard full reads
                # so a pre-move sample cannot survive every "flush" iteration.
                ret, frame = self._cv_cap.read()
                read_completed = time.monotonic()
                if not ret or frame is None or frame.size == 0:
                    return None
                self._opencv_read_sequence += 1
                if read_completed >= deadline:
                    return None
                if read_index == flush:
                    self.last_capture_metadata = {
                        "backend": "opencv",
                        "host_read_sequence": self._opencv_read_sequence,
                        "host_read_started_monotonic_s": read_started,
                        "host_read_completed_monotonic_s": read_completed,
                        "discarded_frames": flush,
                        "timestamp_source": "host_read_not_sensor_exposure",
                    }
                    return frame
            return None

        # SDK path
        with self._cond:
            if self.hcam is None:
                raise RuntimeError("[Camera] Camera is not open.")
            if self._disconnected:
                raise RuntimeError("[Camera] Camera disconnected.")
            start_seq = self._frame_seq
            if self._cond.wait_for(
                lambda: self._frame_seq != start_seq or self._disconnected or self._last_error,
                timeout=timeout
            ):
                if self._disconnected:
                    raise RuntimeError("[Camera] Camera disconnected.")
                if self._last_error and self._frame_seq == start_seq:
                    raise RuntimeError(f"[Camera] {self._last_error}")
                if self._latest_frame is not None:
                    return self._latest_frame
        print("[Camera] Timeout waiting for frame.")
        return None

    def snap_image(self, timeout: float = 5.0):
        """Capture one frame. Starts live mode if not already running."""
        if self._backend in ("amcam", "toupcam"):
            with self._lock:
                running = self._running
            if not running:
                self.start_live()
        return self.grab_frame(timeout)

    # ==================================================================
    # Saving
    # ==================================================================
    @staticmethod
    def save_image(frame, filepath: str, quality: int = 100):
        if not _HAS_CV2:
            raise RuntimeError("OpenCV (cv2) is required for saving images.")
        ext = os.path.splitext(filepath)[1].lower()
        params = []
        if ext in ('.jpg', '.jpeg'):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        elif ext == '.png':
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
        ok = cv2.imwrite(filepath, frame, params)
        if not ok:
            raise RuntimeError(f"[Camera] Failed to save image: {filepath}")

    # ==================================================================
    # Interactive preview & tuning
    # ==================================================================
    def preview_and_tune(self):
        """
        Open a live preview window for interactive tuning.

        Keys (all backends):
          q / ESC  — accept settings and close
          r        — print current camera info

        Keys (SDK backend):
          w        — one-shot auto white balance
          e        — toggle auto-exposure
          +/-      — increase/decrease exposure (manual mode)

        Keys (UVC/OpenCV backend):
          w        — toggle auto white balance
          e        — toggle auto-exposure
          +/-      — adjust exposure (manual mode)
          b/B      — decrease/increase brightness
          c/C      — decrease/increase contrast
          g/G      — decrease/increase gain
        """
        if not _HAS_CV2:
            print("[Camera] OpenCV not available, skipping preview.")
            return
        was_running = self.is_streaming
        is_sdk = self._backend in ("amcam", "toupcam")
        is_uvc = self._backend == "opencv" and bool(self._uvc_supported)
        if is_sdk:
            self.start_live()

        if is_sdk:
            win_name = "Camera Preview (q=OK, w=AWB, e=AE, +/-=expo, r=info)"
        elif is_uvc:
            win_name = "Camera Preview (q=OK, w=AWB, e=AE, +/-=expo, b/B=brt, c/C=con, g/G=gain)"
        else:
            win_name = "Camera Preview (q=OK, r=info)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        if is_sdk:
            try:
                auto_expo = bool(self.hcam.get_AutoExpoEnable())
            except Exception:
                auto_expo = True
        elif is_uvc:
            ae_val = self._get_uvc("auto_exposure")
            auto_expo = (ae_val is not None and ae_val < 0.5)  # 0.25 = auto
        else:
            auto_expo = True

        print("[Camera] Preview opened.")
        if is_sdk:
            print("  q/ESC=accept  w=AWB  e=auto-expo  +/-=exposure  r=info")
        elif is_uvc:
            print("  q/ESC=accept  w=AWB  e=auto-expo  +/-=exposure")
            print("  b/B=brightness  c/C=contrast  g/G=gain  r=info")
        else:
            print("  q/ESC=accept  r=info")
            print("  (HDMI/OpenCV: camera controls are handled by the camera itself)")

        while True:
            try:
                frame = self.grab_frame(timeout=2.0, flush=0 if is_uvc else 3)
            except RuntimeError as ex:
                print(f"[Camera] Preview stopped: {ex}")
                break
            if frame is not None:
                cv2.imshow(win_name, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('w'):
                if is_sdk:
                    self.auto_white_balance()
                    print("[Camera] Auto white balance triggered.")
                elif is_uvc:
                    if self._has_uvc("auto_wb"):
                        cur_awb = self._get_uvc("auto_wb")
                        new_awb = 0 if cur_awb else 1
                        self._set_uvc("auto_wb", new_awb)
                        print(f"[Camera] Auto WB: {'ON' if new_awb else 'OFF'}")
            elif key == ord('e'):
                if is_sdk or is_uvc:
                    auto_expo = not auto_expo
                    self.set_auto_exposure(auto_expo)
                    print(f"[Camera] Auto-exposure: {'ON' if auto_expo else 'OFF'}")
            elif key in (ord('+'), ord('=')):
                if not auto_expo:
                    if is_sdk:
                        cur = self.get_exposure_time()
                        self.set_exposure_time(int(cur * 1.5) + 1)
                        print(f"[Camera] Exposure: {self.get_exposure_time()} us")
                    elif is_uvc and self._has_uvc("exposure"):
                        cur = int(self._get_uvc("exposure"))
                        new_val = min(cur + 1, -1)
                        self._set_uvc("exposure", new_val)
                        print(f"[Camera] Exposure: {new_val}")
            elif key == ord('-'):
                if not auto_expo:
                    if is_sdk:
                        cur = self.get_exposure_time()
                        self.set_exposure_time(max(1, int(cur / 1.5)))
                        print(f"[Camera] Exposure: {self.get_exposure_time()} us")
                    elif is_uvc and self._has_uvc("exposure"):
                        cur = int(self._get_uvc("exposure"))
                        new_val = max(cur - 1, -13)
                        self._set_uvc("exposure", new_val)
                        print(f"[Camera] Exposure: {new_val}")
            elif key == ord('b') and is_uvc:
                if self._has_uvc("brightness"):
                    cur = self._get_uvc("brightness")
                    self._set_uvc("brightness", cur - 5)
                    print(f"[Camera] Brightness: {cur - 5}")
            elif key == ord('B') and is_uvc:
                if self._has_uvc("brightness"):
                    cur = self._get_uvc("brightness")
                    self._set_uvc("brightness", cur + 5)
                    print(f"[Camera] Brightness: {cur + 5}")
            elif key == ord('c') and is_uvc:
                if self._has_uvc("contrast"):
                    cur = self._get_uvc("contrast")
                    self._set_uvc("contrast", cur - 5)
                    print(f"[Camera] Contrast: {cur - 5}")
            elif key == ord('C') and is_uvc:
                if self._has_uvc("contrast"):
                    cur = self._get_uvc("contrast")
                    self._set_uvc("contrast", cur + 5)
                    print(f"[Camera] Contrast: {cur + 5}")
            elif key == ord('g') and is_uvc:
                if self._has_uvc("gain"):
                    cur = self._get_uvc("gain")
                    self._set_uvc("gain", max(0, cur - 5))
                    print(f"[Camera] Gain: {max(0, cur - 5)}")
            elif key == ord('G') and is_uvc:
                if self._has_uvc("gain"):
                    cur = self._get_uvc("gain")
                    self._set_uvc("gain", cur + 5)
                    print(f"[Camera] Gain: {cur + 5}")
            elif key == ord('r'):
                self.print_capabilities()

        cv2.destroyWindow(win_name)
        if not was_running:
            self.stop_live()
        print("[Camera] Preview closed.")

    # ==================================================================
    # Internal
    # ==================================================================
    def _alloc_buf(self):
        channels = 3 if self.bits == 24 else 1
        bufsize = self._width * channels * self._height
        self._buf = ctypes.create_string_buffer(bufsize)

    # ==================================================================
    # Context manager
    # ==================================================================
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.stop_live()
        self.close()

    def __del__(self):
        try:
            self.stop_live()
        except Exception:
            pass
        try:
            self.close()
        except Exception:
            pass

    # ==================================================================
    # Static utilities
    # ==================================================================
    @staticmethod
    def list_cameras():
        """List all detected cameras (SDK + OpenCV)."""
        print("=== SDK Cameras ===")
        cls = _sdk_class()
        if cls is not None:
            try:
                devices = cls.EnumV2()
                if len(devices) == 0:
                    print("  (none found)")
                for i, dev in enumerate(devices):
                    print(f"  [{i}] {dev.displayname} ({dev.model.name})")
                    for j, r in enumerate(dev.model.res):
                        print(f"       res[{j}]: {r.width} x {r.height}")
            except Exception as e:
                print(f"  SDK error: {e}")
        else:
            print(f"  (no SDK available)")

        print("\n=== OpenCV/UVC Cameras ===")
        if _HAS_CV2:
            api = _opencv_api()
            for idx in range(5):
                cap = cv2.VideoCapture(idx, api)
                try:
                    if cap.isOpened():
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        name = cap.getBackendName() if hasattr(cap, 'getBackendName') else "unknown"
                        print(f"  [{idx}] {w} x {h} ({name})")
                finally:
                    cap.release()
        else:
            print("  (OpenCV not installed)")
        print()


# =========================================================================
# Backward-compatible aliases
# =========================================================================
AmScopeCamera = Camera
