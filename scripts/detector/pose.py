"""
detector/pose.py
----------------
Detects the skier and estimates 2-D body keypoints in every frame using
Ultralytics YOLO26 with the pose task.

Model used: yolo26s-pose.pt  (auto-downloaded on first run)

YOLO26-pose predicts the 17 standard COCO keypoints.  For the Ski-2DPose
convention (24 keypoints incl. skis and poles) you would fine-tune the model
on that dataset — see train/train_pose.py.

Keypoint index reference (COCO 17):
    0  nose            5  left shoulder    10  right wrist
    1  left eye        6  right shoulder   11  left hip
    2  right eye       7  left elbow       12  right hip
    3  left ear        8  right elbow      13  left knee
    4  right ear       9  left wrist       14  right knee
                                           15  left ankle
                                           16  right ankle
Extended Ski-2DPose (indices 17-23 after fine-tuning):
    17 left ski tip    19 right ski tip    21 left pole tip    23 right pole tip
    18 left ski tail   20 right ski tail   22 left pole mid    (helper)
"""

import numpy as np
from typing import List, Optional, Tuple

from ultralytics import YOLO

# (cx,cy,conf) tuple type alias
Keypoint = Tuple[float, float, float]

# Default model — swap for fine-tuned weights after training
DEFAULT_MODEL = "yolo26s-pose.pt"


class PoseEstimator:
    """
    Wraps a YOLO26-pose model for single-skier pose estimation.

    Parameters
    ----------
    model_path : str
        Path to .pt weights, or Ultralytics auto-download name
        (e.g. "yolo26s-pose.pt", "yolo26m-pose.pt").
    conf : float
        Minimum detection confidence threshold.
    device : str
        "cpu", "cuda", "mps", or "" (auto).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf: float = 0.35,
        device: str = "",
    ):
        self.model = YOLO(model_path)
        self.conf  = conf
        self.device = device or ("cuda" if self._cuda_available() else "cpu")

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(
        self,
        frame: np.ndarray,
        skier_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Keypoint]:
        """
        Run pose estimation on *frame*.

        If *skier_bbox* (x1,y1,x2,y2) is provided the frame is cropped first,
        which both speeds up inference and reduces false positives.

        Returns
        -------
        list of (x, y, conf) in original frame coordinates, length = num_kpts.
        Empty list if no person is detected.
        """
        if skier_bbox is not None:
            x1, y1, x2, y2 = [max(0, int(v)) for v in skier_bbox]
            crop   = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            crop   = frame
            offset = (0, 0)

        if crop.size == 0:
            return []

        results = self.model(
            crop,
            task="pose",
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        if not results or results[0].keypoints is None:
            return []

        # Pick the highest-confidence detection (= the skier)
        kp_data  = results[0].keypoints.data   # shape (N, K, 3)
        conf_arr = results[0].boxes.conf        # shape (N,)
        if len(conf_arr) == 0:
            return []

        best_idx  = int(conf_arr.argmax())
        kpts_raw  = kp_data[best_idx].cpu().numpy()  # (K, 3) → x, y, conf

        ox, oy = offset
        keypoints: List[Keypoint] = [
            (float(x + ox), float(y + oy), float(c))
            for x, y, c in kpts_raw
        ]
        return keypoints

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False