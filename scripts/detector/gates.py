"""
detector/gates.py
-----------------
Detects red and blue slalom / GS gate poles in each frame using HSV colour
segmentation, then tracks each gate across frames with a simple Kalman filter.

Returns a dict  {gate_id: (cx, cy)}  per frame.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── HSV colour ranges ─────────────────────────────────────────────────────────
# Red wraps around 0° so we need two intervals.
RED_LOWER1 = np.array([0,   130, 60],  dtype=np.uint8)
RED_UPPER1 = np.array([10,  255, 255], dtype=np.uint8)
RED_LOWER2 = np.array([165, 130, 60],  dtype=np.uint8)
RED_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

BLUE_LOWER = np.array([95,  120, 60],  dtype=np.uint8)
BLUE_UPPER = np.array([135, 255, 255], dtype=np.uint8)

# ── Tunable constants ─────────────────────────────────────────────────────────
MIN_AREA_PX   = 150    # minimum contour area to be a pole candidate
MAX_AREA_PX   = 15000  # ignore very large blobs (background noise)
MATCH_DIST_PX = 90     # max pixel distance for Kalman nearest-neighbour match
MAX_MISSING   = 10     # frames a tracker can be unseen before being dropped


@dataclass
class _GateTracker:
    gate_id: str
    color:   str          # "R" or "B"
    kf:      cv2.KalmanFilter
    missed:  int = 0
    last_cx: float = 0.0
    last_cy: float = 0.0


def _make_kalman(cx: float, cy: float) -> cv2.KalmanFilter:
    """Constant-velocity Kalman filter (state: x, y, vx, vy; meas: x, y)."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix  = np.eye(2, 4, dtype=np.float32)
    kf.transitionMatrix   = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 5e-1
    kf.statePre           = np.array([[cx], [cy], [0.], [0.]], dtype=np.float32)
    kf.statePost          = kf.statePre.copy()
    kf.errorCovPre        = np.eye(4, dtype=np.float32)
    return kf


class GateDetector:
    """
    Stateful detector that keeps Kalman-tracked identities across frames.

    Usage:
        det = GateDetector()
        for frame in frames:
            gate_centroids = det.update(frame)   # {gate_id: (cx, cy)}
    """

    def __init__(self):
        self._trackers: Dict[str, _GateTracker] = {}
        self._next_id  = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Process one BGR frame.

        Returns
        -------
        dict  {gate_id: (cx_px, cy_px)}
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_mask  = (cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
                   | cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
        blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

        detections: List[Tuple[float, float, str]] = []  # (cx, cy, color)
        for mask, color in [(red_mask, "R"), (blue_mask, "B")]:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    np.ones((9, 9), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                detections.append((cx, cy, color))

        # Advance Kalman predictions
        predictions: Dict[str, np.ndarray] = {}
        for gid, trk in self._trackers.items():
            pred = trk.kf.predict()   # shape (4,1)
            predictions[gid] = pred[:2].flatten()

        # Greedy nearest-neighbour matching (same color only)
        matched_det_idxs: set = set()
        matched_trk_ids:  set = set()
        for gid, trk in self._trackers.items():
            pred_xy = predictions[gid]
            best_dist, best_i = MATCH_DIST_PX, -1
            for i, (cx, cy, col) in enumerate(detections):
                if col != trk.color or i in matched_det_idxs:
                    continue
                d = float(np.linalg.norm([cx - pred_xy[0], cy - pred_xy[1]]))
                if d < best_dist:
                    best_dist, best_i = d, i
            if best_i >= 0:
                cx, cy, _ = detections[best_i]
                trk.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
                trk.last_cx = cx
                trk.last_cy = cy
                trk.missed  = 0
                matched_det_idxs.add(best_i)
                matched_trk_ids.add(gid)
            else:
                trk.missed += 1

        # Spawn new trackers for unmatched detections
        for i, (cx, cy, col) in enumerate(detections):
            if i not in matched_det_idxs:
                gid = f"{col}{self._next_id}"
                self._next_id += 1
                self._trackers[gid] = _GateTracker(
                    gate_id=gid, color=col,
                    kf=_make_kalman(cx, cy),
                    last_cx=cx, last_cy=cy)

        # Drop stale trackers
        self._trackers = {
            gid: trk for gid, trk in self._trackers.items()
            if trk.missed <= MAX_MISSING
        }

        return {gid: (int(trk.last_cx), int(trk.last_cy))
                for gid, trk in self._trackers.items()}

    def reset(self):
        self._trackers.clear()
        self._next_id = 0