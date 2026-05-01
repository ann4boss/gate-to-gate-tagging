"""
detector/skier.py
-----------------
Detects the skier bounding box in every frame using YOLO26 (person class),
then stabilises the track across frames with a Kalman filter + IoU-based
association (ByteTrack-style).

Why a tracker?
  Raw YOLO detections flicker: the model may miss a frame, detect a
  spectator instead, or jump between two overlapping people.  A Kalman
  filter predicts where the skier *should* be in the next frame; we then
  match that prediction to new detections via IoU.  If no detection matches
  we coast on the prediction for up to MAX_COAST_FRAMES before resetting.

Usage (stateful — one instance per video):
    det = SkierDetector()
    for frame in frames:
        bbox = det.detect(frame)   # (x1, y1, x2, y2) or None
"""

import numpy as np
from typing import Optional, Tuple
from ultralytics import YOLO

BBox = Tuple[int, int, int, int]   # (x1, y1, x2, y2)

# DEFAULT_MODEL   = "yolo26s.pt"
DEFAULT_MODEL ="runs/detect/runs/detect/skier_epfl/weights/best.pt"
PERSON_CLASS    = 0          # COCO index for "person"
IOU_THRESHOLD   = 0.25       # min IoU to accept a detection as the same skier
MAX_COAST_FRAMES = 10        # frames we keep predicting without a detection


# ── Kalman filter state ───────────────────────────────────────────────────────
# State vector:  [cx, cy, w, h, vx, vy, vw, vh]  (centre + size + velocities)
# Measurement:   [cx, cy, w, h]

def _make_kalman() -> "cv2.KalmanFilter":
    import cv2
    kf = cv2.KalmanFilter(8, 4)

    # Transition matrix  (constant-velocity model)
    kf.transitionMatrix = np.array([
        [1,0,0,0, 1,0,0,0],
        [0,1,0,0, 0,1,0,0],
        [0,0,1,0, 0,0,1,0],
        [0,0,0,1, 0,0,0,1],
        [0,0,0,0, 1,0,0,0],
        [0,0,0,0, 0,1,0,0],
        [0,0,0,0, 0,0,1,0],
        [0,0,0,0, 0,0,0,1],
    ], dtype=np.float32)

    # Measurement matrix  (we observe cx, cy, w, h directly)
    kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
    kf.measurementMatrix[0, 0] = 1
    kf.measurementMatrix[1, 1] = 1
    kf.measurementMatrix[2, 2] = 1
    kf.measurementMatrix[3, 3] = 1

    # Noise covariances — tuned for ~30 fps broadcast footage
    kf.processNoiseCov     = np.eye(8, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kf.errorCovPost        = np.eye(8, dtype=np.float32)

    return kf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _xyxy_to_cxywh(x1, y1, x2, y2):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = float(x2 - x1)
    h  = float(y2 - y1)
    return cx, cy, w, h


def _cxywh_to_xyxy(cx, cy, w, h):
    return (
        int(cx - w / 2),
        int(cy - h / 2),
        int(cx + w / 2),
        int(cy + h / 2),
    )


def _iou(a: BBox, b: BBox) -> float:
    """Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# ── Main class ────────────────────────────────────────────────────────────────

class SkierDetector:
    """
    YOLO26 person detector stabilised by a Kalman filter tracker.

    The tracker is initialised on the first high-confidence detection and
    then maintained across frames via IoU matching.  It coasts (uses the
    Kalman prediction) for up to MAX_COAST_FRAMES frames when no matching
    detection is found, then resets.

    Parameters
    ----------
    model_path : str
        Ultralytics model name or path to fine-tuned .pt weights.
    conf : float
        YOLO detection confidence threshold.
    device : str
        'cpu', 'cuda', 'mps', or '' (auto).
    iou_threshold : float
        Minimum IoU between Kalman prediction and a new detection to
        treat them as the same skier.
    max_coast_frames : int
        How many frames to coast on the Kalman prediction alone before
        declaring the track lost and resetting.
    """

    def __init__(
        self,
        model_path:      str   = DEFAULT_MODEL,
        conf:            float = 0.40,
        device:          str   = "",
        iou_threshold:   float = IOU_THRESHOLD,
        max_coast_frames: int  = MAX_COAST_FRAMES,
    ):
        self.model           = YOLO(model_path)
        self.conf            = conf
        self.device          = device or ("cuda" if self._cuda_available() else "cpu")
        self.iou_threshold   = iou_threshold
        self.max_coast_frames = max_coast_frames

        # Tracker state
        self._kf:           Optional[object] = None   # cv2.KalmanFilter
        self._coast_count:  int = 0
        self._last_bbox:    Optional[BBox] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Optional[BBox]:
        """
        Run detection + tracking on one BGR frame.

        Returns
        -------
        (x1, y1, x2, y2) for the tracked skier, or None if the track
        has not been initialised or has been lost.
        """
        detections = self._run_yolo(frame)

        # ── Case 1: tracker not yet initialised ──────────────────────────────
        if self._kf is None:
            if not detections:
                return None
            # Pick the most confident detection to seed the tracker
            best = max(detections, key=lambda d: d[1])
            self._init_tracker(best[0])
            self._last_bbox = best[0]
            return best[0]

        # ── Case 2: tracker active — predict next position ───────────────────
        predicted_bbox = self._kalman_predict()

        # Find the detection with highest IoU against the prediction
        best_det, best_iou = None, 0.0
        for det_bbox, _ in detections:
            iou = _iou(predicted_bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_det = det_bbox

        if best_det is not None and best_iou >= self.iou_threshold:
            # Good match — correct the Kalman filter
            self._kalman_correct(best_det)
            self._coast_count = 0
            self._last_bbox   = best_det
            return best_det
        else:
            # No matching detection — coast on prediction
            self._coast_count += 1
            if self._coast_count > self.max_coast_frames:
                # Track lost — reset
                self.reset()
                return None
            return predicted_bbox

    def reset(self):
        """Reset the tracker (call between runs/athletes)."""
        self._kf          = None
        self._coast_count = 0
        self._last_bbox   = None

    # ── Internal: YOLO ────────────────────────────────────────────────────────

    def _run_yolo(self, frame: np.ndarray):
        """Return list of (BBox, confidence) for all person detections."""
        results = self.model(
            frame,
            task="detect",
            conf=self.conf,
            classes=[PERSON_CLASS],
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        out = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i])
            # Small vertical pad to include skis
            h   = y2 - y1
            pad = int(h * 0.10)
            y2  = min(frame.shape[0], y2 + pad)
            out.append(((int(x1), int(y1), int(x2), int(y2)), conf))
        return out

    # ── Internal: Kalman ──────────────────────────────────────────────────────

    def _init_tracker(self, bbox: BBox):
        import cv2  # imported here so the file is importable without cv2 at module level
        self._kf = _make_kalman()
        cx, cy, w, h = _xyxy_to_cxywh(*bbox)
        self._kf.statePre  = np.array([[cx],[cy],[w],[h],[0],[0],[0],[0]], dtype=np.float32)
        self._kf.statePost = self._kf.statePre.copy()
        self._coast_count  = 0

    def _kalman_predict(self) -> BBox:
        pred = self._kf.predict()          # advances internal state
        cx, cy, w, h = pred[0,0], pred[1,0], pred[2,0], pred[3,0]
        # Clamp width/height to positive values
        w = max(w, 10.0)
        h = max(h, 10.0)
        return _cxywh_to_xyxy(cx, cy, w, h)

    def _kalman_correct(self, bbox: BBox):
        cx, cy, w, h = _xyxy_to_cxywh(*bbox)
        meas = np.array([[cx],[cy],[w],[h]], dtype=np.float32)
        self._kf.correct(meas)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False