"""
detector/pose.py
----------------
Estimates 2-D body + ski + pole keypoints for the skier in every frame
using a YOLO26-pose model fine-tuned on the EPFL Ski-2DPose dataset.

Keypoint index reference — Ski-2DPose 24 keypoints
---------------------------------------------------
COCO body (0-16):
    0  nose             5  left shoulder    10  right wrist
    1  left eye         6  right shoulder   11  left hip
    2  right eye        7  left elbow       12  right hip
    3  left ear         8  right elbow      13  left knee
    4  right ear        9  left wrist       14  right knee
                                            15  left ankle
                                            16  right ankle
Ski-specific (17-23):
    17  left ski tip        20  right ski tip
    18  left ski tail       21  right ski tail
    19  left pole grip      22  right pole grip
    23  left pole tip

These indices are used directly in the feature vector:
    feature[0,1] = hip midpoint        (kpts 11 + 12)
    feature[2,3] = ankle midpoint      (kpts 15 + 16)
    feature[4]   = pole grip distance  (kpt 19 or 22, closest to inner pole)
    feature[6]   = pole touch flag     (derived from feature[4])

Usage
-----
    est = PoseEstimator()                           # pretrained COCO 17 kpts
    est = PoseEstimator("runs/pose/.../best.pt")    # fine-tuned 24 kpts

    keypoints = est.estimate(frame, skier_bbox)
    # → list of (x, y, conf) in original frame coordinates, length = n_kpts
"""

import numpy as np
from typing import List, Optional, Tuple
from ultralytics import YOLO

# (x, y, conf) tuple
Keypoint = Tuple[float, float, float]

# ── Keypoint index constants ──────────────────────────────────────────────────
# Body
KP_NOSE          = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER= 6
KP_LEFT_ELBOW    = 7
KP_RIGHT_ELBOW   = 8
KP_LEFT_WRIST    = 9
KP_RIGHT_WRIST   = 10
KP_LEFT_HIP      = 11
KP_RIGHT_HIP     = 12
KP_LEFT_KNEE     = 13
KP_RIGHT_KNEE    = 14
KP_LEFT_ANKLE    = 15
KP_RIGHT_ANKLE   = 16
# Ski-specific (only available after fine-tuning on Ski-2DPose)
KP_LEFT_SKI_TIP   = 17
KP_LEFT_SKI_TAIL  = 18
KP_LEFT_POLE_GRIP = 19
KP_RIGHT_SKI_TIP  = 20
KP_RIGHT_SKI_TAIL = 21
KP_RIGHT_POLE_GRIP= 22
KP_LEFT_POLE_TIP  = 23

# Pretrained COCO model (17 kpts) — auto-downloaded by Ultralytics
DEFAULT_MODEL = "yolo26s-pose.pt"

# Minimum keypoint confidence to treat a keypoint as valid
MIN_KP_CONF = 0.20


class PoseEstimator:
    """
    YOLO26-pose wrapper for single-skier 2-D pose estimation.

    Works with both:
      - yolo26s-pose.pt   (17 COCO keypoints, no fine-tuning needed)
      - fine-tuned .pt    (24 Ski-2DPose keypoints, after train_pose.py)

    The number of keypoints is detected automatically from the model output.

    Parameters
    ----------
    model_path : str
        Ultralytics model name or path to fine-tuned .pt weights.
    conf : float
        Minimum detection confidence for the person bounding box.
    device : str
        'cpu', 'cuda', 'mps', or '' (auto).
    min_kp_conf : float
        Keypoints below this confidence are zeroed out (x=0, y=0, conf=0).
    """

    def __init__(
        self,
        model_path:  str   = DEFAULT_MODEL,
        conf:        float = 0.35,
        device:      str   = "",
        min_kp_conf: float = MIN_KP_CONF,
    ):
        self.model       = YOLO(model_path)
        self.conf        = conf
        self.device      = device or ("cuda" if self._cuda_available() else "cpu")
        self.min_kp_conf = min_kp_conf

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(
        self,
        frame:       np.ndarray,
        skier_bbox:  Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Keypoint]:
        """
        Run pose estimation on one BGR frame.

        If skier_bbox (x1, y1, x2, y2) is provided the frame is cropped
        first — faster inference and fewer false positives from spectators.

        Returns
        -------
        List of (x, y, conf) in original frame coordinates.
        Length = number of keypoints the model was trained on (17 or 24).
        Returns an empty list if no person is detected.
        """
        if skier_bbox is not None:
            x1 = max(0, int(skier_bbox[0]))
            y1 = max(0, int(skier_bbox[1]))
            x2 = min(frame.shape[1], int(skier_bbox[2]))
            y2 = min(frame.shape[0], int(skier_bbox[3]))
            crop   = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            crop   = frame
            offset = (0, 0)

        if crop.size == 0:
            return []

        results = self.model(
            crop,
            task    = "pose",
            conf    = self.conf,
            device  = self.device,
            verbose = False,
        )

        if not results or results[0].keypoints is None:
            return []

        kp_data  = results[0].keypoints.data  # (N, K, 3)
        conf_arr = results[0].boxes.conf       # (N,)

        if len(conf_arr) == 0:
            return []

        # Pick the highest-confidence detection = the skier
        best_idx = int(conf_arr.argmax())
        kpts_raw = kp_data[best_idx].cpu().numpy()  # (K, 3)

        ox, oy = offset
        keypoints: List[Keypoint] = []
        for x, y, c in kpts_raw:
            if float(c) < self.min_kp_conf:
                # Low-confidence keypoint — zero it out so downstream code
                # can safely skip it with a simple `if conf > 0` check
                keypoints.append((0.0, 0.0, 0.0))
            else:
                keypoints.append((float(x + ox), float(y + oy), float(c)))

        return keypoints

    # ── Convenience accessors ────────────────────────────────────────────────

    def hip_midpoint(
        self, keypoints: List[Keypoint]
    ) -> Optional[Tuple[float, float]]:
        """Return (x, y) midpoint of left and right hip, or None if not visible."""
        return self._midpoint(keypoints, KP_LEFT_HIP, KP_RIGHT_HIP)

    def ankle_midpoint(
        self, keypoints: List[Keypoint]
    ) -> Optional[Tuple[float, float]]:
        """Return (x, y) midpoint of left and right ankle, or None if not visible."""
        return self._midpoint(keypoints, KP_LEFT_ANKLE, KP_RIGHT_ANKLE)

    def pole_grip_positions(
        self, keypoints: List[Keypoint]
    ) -> Tuple[Optional[Tuple[float,float]], Optional[Tuple[float,float]]]:
        """
        Return (left_pole_grip_xy, right_pole_grip_xy).
        Only available when using the 24-keypoint fine-tuned model.
        Returns (None, None) if keypoints are not available or not confident.
        """
        if len(keypoints) <= KP_RIGHT_POLE_GRIP:
            return None, None
        left  = self._kp_xy(keypoints, KP_LEFT_POLE_GRIP)
        right = self._kp_xy(keypoints, KP_RIGHT_POLE_GRIP)
        return left, right

    def wrist_positions(
        self, keypoints: List[Keypoint]
    ) -> Tuple[Optional[Tuple[float,float]], Optional[Tuple[float,float]]]:
        """
        Return (left_wrist_xy, right_wrist_xy).
        Falls back to COCO wrist keypoints (available even without fine-tuning).
        """
        return (
            self._kp_xy(keypoints, KP_LEFT_WRIST),
            self._kp_xy(keypoints, KP_RIGHT_WRIST),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _kp_xy(
        self, keypoints: List[Keypoint], idx: int
    ) -> Optional[Tuple[float, float]]:
        if idx >= len(keypoints):
            return None
        x, y, c = keypoints[idx]
        return (x, y) if c > 0 else None

    def _midpoint(
        self, keypoints: List[Keypoint], idx_a: int, idx_b: int
    ) -> Optional[Tuple[float, float]]:
        a = self._kp_xy(keypoints, idx_a)
        b = self._kp_xy(keypoints, idx_b)
        if a is None or b is None:
            return None
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False