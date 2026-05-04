"""
detector/gates.py
-----------------
Detects gate poles in each frame using a fine-tuned YOLO26 model,
then tracks each pole across frames with a Kalman filter.


Classes detected:
    0  gate_contact  — the pole the skier hits (GS inner pole or SL panel)
    1  gate_outer    — the outer rigid pole of a GS gate (GS only)

Each detected pole is tracked across frames using a constant-velocity
Kalman filter with greedy IoU-based matching — the same approach used
in the skier detector.  This gives stable pole IDs across frames even
when the pole is briefly occluded or YOLO misses a frame.

Output per frame:
    {
        gate_id: {
            "cx":    int,           # centroid x in original frame coords
            "cy":    int,           # centroid y in original frame coords
            "bbox":  (x1,y1,x2,y2),# bounding box
            "class": int,           # 0=gate_contact, 1=gate_outer
            "label": str,           # "gate_contact" or "gate_outer"
            "conf":  float,         # YOLO detection confidence
        }
    }

Usage
-----
    det = GateDetector("runs/detect/gate_poles/weights/best.pt")
    for frame in frames:
        gates = det.update(frame)
        for gid, info in gates.items():
            cx, cy = info["cx"], info["cy"]
            label  = info["label"]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO

# ── Class definitions ─────────────────────────────────────────────────────────
CLASS_NAMES   = {0: "gate_contact", 1: "gate_outer"}
CLASS_CONTACT = 0
CLASS_OUTER   = 1

# ── Tracker constants ─────────────────────────────────────────────────────────
IOU_THRESHOLD  = 0.15   # lower than skier because poles are small thin objects
MAX_MISSING    = 60     # frames a tracker survives without a detection
DEFAULT_MODEL  = "runs/detect/gate_poles/weights/best.pt"


# ── Kalman filter (4-state: cx, cy, vx, vy) ──────────────────────────────────

def _make_kalman(cx: float, cy: float):
    import cv2
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix  = np.eye(2, 4, dtype=np.float32)
    kf.transitionMatrix   = np.array(
        [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32
    )
    kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 5e-1
    kf.statePre  = np.array([[cx],[cy],[0.],[0.]], dtype=np.float32)
    kf.statePost = kf.statePre.copy()
    kf.errorCovPre = np.eye(4, dtype=np.float32)
    return kf


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _bbox_centre(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _iou(a, b) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def _predicted_bbox(kf, last_w: float, last_h: float):
    """Return a bbox from Kalman prediction using last known width/height."""
    pred = kf.predict()
    cx, cy = pred[0,0], pred[1,0]
    return (
        int(cx - last_w/2), int(cy - last_h/2),
        int(cx + last_w/2), int(cy + last_h/2),
    )


# ── Pole tracker ──────────────────────────────────────────────────────────────

@dataclass
class _PoleTracker:
    gate_id: str
    cls:     int          # 0=gate_contact, 1=gate_outer
    kf:      object       # cv2.KalmanFilter
    last_cx: float
    last_cy: float
    last_w:  float
    last_h:  float
    last_conf: float
    missed:  int = 0


# ── Main class ────────────────────────────────────────────────────────────────

class GateDetector:
    """
    YOLO26-based gate pole detector with Kalman filter tracking.

    Parameters
    ----------
    model_path : str
        Path to fine-tuned YOLO26 weights from train_gate_detector.py.
    conf : float
        YOLO detection confidence threshold.
    device : str
        'cpu', 'cuda', 'mps', or '' (auto).
    iou_threshold : float
        Minimum IoU to match a detection to an existing track.
    max_missing : int
        Frames a track survives without a matching detection.
    contact_only : bool
        If True, only return gate_contact poles (class 0).
        Useful for feature vector computation where gate_outer is not needed.
    """

    def __init__(
        self,
        model_path:    str   = DEFAULT_MODEL,
        conf:          float = 0.35,
        device:        str   = "",
        iou_threshold: float = IOU_THRESHOLD,
        max_missing:   int   = MAX_MISSING,
        contact_only:  bool  = False,
    ):
        self.model         = YOLO(model_path)
        self.conf          = conf
        self.device        = device or ("cuda" if self._cuda_available() else "cpu")
        self.iou_threshold = iou_threshold
        self.max_missing   = max_missing
        self.contact_only  = contact_only

        self._trackers: Dict[str, _PoleTracker] = {}
        self._next_id = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> Dict[str, dict]:
        """
        Process one BGR frame.

        Returns
        -------
        dict keyed by gate_id, each value:
            {"cx", "cy", "bbox", "class", "label", "conf"}
        Only includes poles whose track is currently active.
        """
        detections = self._run_yolo(frame)

        # Filter to contact-only if requested
        if self.contact_only:
            detections = [d for d in detections if d["class"] == CLASS_CONTACT]

        # ── Predict next position for all active trackers ─────────────────────
        predictions = {}
        for gid, trk in self._trackers.items():
            predictions[gid] = _predicted_bbox(trk.kf, trk.last_w, trk.last_h)

        # ── Greedy IoU matching ───────────────────────────────────────────────
        matched_det_idxs: set = set()
        matched_trk_ids:  set = set()

        for gid, trk in self._trackers.items():
            pred_bbox  = predictions[gid]
            best_iou   = self.iou_threshold
            best_i     = -1

            for i, det in enumerate(detections):
                if i in matched_det_idxs:
                    continue
                # Only match same class
                if det["class"] != trk.cls:
                    continue
                iou = _iou(pred_bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_i   = i

            if best_i >= 0:
                det = detections[best_i]
                cx, cy = _bbox_centre(det["bbox"])
                x1,y1,x2,y2 = det["bbox"]
                trk.kf.correct(np.array([[cx],[cy]], dtype=np.float32))
                trk.last_cx   = cx
                trk.last_cy   = cy
                trk.last_w    = float(x2 - x1)
                trk.last_h    = float(y2 - y1)
                trk.last_conf = det["conf"]
                trk.missed    = 0
                matched_det_idxs.add(best_i)
                matched_trk_ids.add(gid)
            else:
                trk.missed += 1
                # Update position from Kalman prediction
                pred = predictions[gid]
                trk.last_cx = (pred[0] + pred[2]) / 2.0
                trk.last_cy = (pred[1] + pred[3]) / 2.0

        # ── Spawn new trackers for unmatched detections ───────────────────────
        for i, det in enumerate(detections):
            if i not in matched_det_idxs:
                cx, cy = _bbox_centre(det["bbox"])
                x1,y1,x2,y2 = det["bbox"]
                gid = f"{'C' if det['class']==CLASS_CONTACT else 'O'}{self._next_id}"
                self._next_id += 1
                self._trackers[gid] = _PoleTracker(
                    gate_id   = gid,
                    cls       = det["class"],
                    kf        = _make_kalman(cx, cy),
                    last_cx   = cx,
                    last_cy   = cy,
                    last_w    = float(x2 - x1),
                    last_h    = float(y2 - y1),
                    last_conf = det["conf"],
                )

        # ── Drop stale trackers ───────────────────────────────────────────────
        self._trackers = {
            gid: trk for gid, trk in self._trackers.items()
            if trk.missed <= self.max_missing
        }

        # ── Build output ──────────────────────────────────────────────────────
        return {
            gid: {
                "cx":    int(trk.last_cx),
                "cy":    int(trk.last_cy),
                "bbox":  _predicted_bbox(trk.kf, trk.last_w, trk.last_h),
                "class": trk.cls,
                "label": CLASS_NAMES[trk.cls],
                "conf":  trk.last_conf,
            }
            for gid, trk in self._trackers.items()
        }

    def reset(self):
        """Reset all trackers — call between runs."""
        self._trackers.clear()
        self._next_id = 0

    def contact_poles(self, gates: dict) -> dict:
        """Filter output of update() to gate_contact poles only."""
        return {gid: info for gid, info in gates.items()
                if info["class"] == CLASS_CONTACT}

    def nearest_contact_pole(
        self,
        gates: dict,
        skier_cx: float,
        skier_cy: float,
    ) -> Optional[dict]:
        """
        Return the gate_contact pole closest to the skier position.
        Useful for feature vector computation (feature[4], feature[7]).
        Returns None if no contact poles are tracked.
        """
        contacts = self.contact_poles(gates)
        if not contacts:
            return None
        return min(
            contacts.values(),
            key=lambda g: (g["cx"] - skier_cx)**2 + (g["cy"] - skier_cy)**2
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_yolo(self, frame: np.ndarray) -> List[dict]:
        results = self.model(
            frame,
            task    = "detect",
            conf    = self.conf,
            device  = self.device,
            verbose = False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        out = []
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf  = float(boxes.conf[i])
            cls   = int(boxes.cls[i])
            out.append({
                "bbox":  (int(x1), int(y1), int(x2), int(y2)),
                "conf":  conf,
                "class": cls,
                "label": CLASS_NAMES.get(cls, f"cls_{cls}"),
            })
        return out

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False