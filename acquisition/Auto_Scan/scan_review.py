"""Offline image diagnostics and a cancelable, request-scoped human review gate.

Image matching is a warning, not encoder feedback or physical ground truth.
"""
import threading


class ReviewGate:
    def __init__(self):
        self._condition = threading.Condition()
        self._token = 0
        self._decision = None
        self._active = False

    def begin(self):
        with self._condition:
            self._token += 1
            self._decision = None
            self._active = True
            return self._token

    def choose(self, token, decision):
        if decision not in {'accept', 'retake', 'stop'}:
            raise ValueError('Unknown review decision')
        with self._condition:
            if not self._active or token != self._token:
                return False
            self._active = False
            self._decision = decision
            self._condition.notify_all()
            return True

    def is_active(self, token):
        with self._condition:
            return self._active and token == self._token

    def wait(self, token, cancel):
        with self._condition:
            while token == self._token and self._decision is None and not cancel.is_set():
                self._condition.wait(.05)
            if cancel.is_set() or token != self._token:
                if token == self._token:
                    self._active = False
                return 'stop'
            return self._decision


def compare_views(previous_path, candidate_path):
    """Conservative near-zero-motion screening, using image coordinates only."""
    import cv2
    import numpy as np
    if not previous_path:
        return {'status': 'first_point', 'message': '起点：请确认样品、视野和焦点。'}
    images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in (previous_path, candidate_path)]
    if any(im is None for im in images):
        return {'status': 'unverified', 'message': '图像无法比较，请人工核对。'}
    features = []
    scale = 960.0 / images[0].shape[1]
    detector = cv2.SIFT_create(nfeatures=2500)
    for im in images:
        features.append(detector.detectAndCompute(cv2.resize(im, None, fx=scale, fy=scale), None))
    (ka, da), (kb, db) = features
    if da is None or db is None:
        return {'status': 'unverified', 'message': '地标不足，请人工核对位置和清晰度。'}
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .7 * p[1].distance]
    if len(good) < 12:
        return {'status': 'unverified', 'message': '共同地标不足，请核对重叠、方向和焦点。'}
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2)
    if matrix is None:
        return {'status': 'unverified', 'message': '配准不确定，请人工核对。'}
    keep = mask.ravel().astype(bool)
    if keep.sum() < 12 or keep.mean() < .35 or abs(np.hypot(matrix[0, 0], matrix[1, 0])-1) > .03:
        return {'status': 'unverified', 'message': '配准证据不足，请人工核对。'}
    shift = np.median(dst[keep] - src[keep], axis=0) / scale
    near = bool(np.linalg.norm(shift) < 15)
    return {'status': 'same_view' if near else 'changed', 'inliers': int(keep.sum()),
            'feature_shift_px': shift.tolist(), 'basis': 'image_matching_not_stage_metrology',
            'message': '疑似仍是上一位置：先原位重拍；仍未变化就停止检查。' if near
                       else '检测到地标变化；这不等于位置和图像质量已经全部验收。'}
