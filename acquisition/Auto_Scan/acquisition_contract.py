"""Dimension contract for raw scan images; no hardware or file access on import.

Pixel dimensions are an acquisition invariant, not a physical pixel calibration.
A failed frame is never resized or accepted as a stitching tile.
"""
from pathlib import Path
import uuid


class AcquisitionModeError(RuntimeError):
    def __init__(self, expected, actual, phase):
        self.expected = list(expected)
        self.actual = list(actual) if actual else None
        self.phase = phase
        self.camera_index = None
        self.backend_requested = None
        super().__init__(
            f"Image size mismatch at {phase}: expected {self.expected}, received {self.actual}. "
            "Acquisition stopped; no resizing or automatic acceptance. Check the camera mode, "
            "reconnect if necessary, and verify the microscope view before starting a new scan. "
            "If this is the wrong device, click Next camera.\n"
            f"图像尺寸不一致：要求 {self.expected}，实际 {self.actual}。程序已停止，不会缩放或自动采用。"
            "请检查相机输出设置；如选错设备，请点击“换一个相机”。重新连接后确认显微镜画面，再开始新扫描。")


def frame_size(frame):
    shape = getattr(frame, 'shape', ())
    if len(shape) < 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
        return None
    return [int(shape[1]), int(shape[0])]


def require_frame_size(frame, expected, phase):
    actual = frame_size(frame)
    if actual != list(expected):
        raise AcquisitionModeError(expected, actual, phase)
    return actual


def contract(expected, actual=None, *, phase, verified=False):
    return {
        'schema_version': 1,
        'expected_image_size': list(expected),
        'actual_image_size': list(actual) if actual else None,
        'verification_status': 'verified_received_frame' if verified else 'pending_pre_scan',
        'verification_phase': phase,
        'calibration_status': 'unverified_for_current_mode',
        'raw_images_resized': False,
    }


def validate_saved_image(path, expected, phase='candidate', *, raw_bytes=None):
    # Decode the file actually saved, not a driver's reported mode. np.frombuffer
    # also supports non-ASCII Windows paths, unlike some cv2.imread builds.
    import cv2
    import numpy as np
    data = Path(path).read_bytes() if raw_bytes is None else raw_bytes
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return require_frame_size(frame, expected, phase)


def preserve_diagnostic_frame(frame, folder, save_image, phase):
    """Keep a unique failed-mode frame outside the accepted mosaic tile set."""
    directory = Path(folder)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('acquisition_' + phase + '_' + uuid.uuid4().hex + '.png')
    save_image(frame, str(path))
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError('Diagnostic frame was not saved: ' + str(path))
    return path
