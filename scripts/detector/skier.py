"""
detector/skier.py
-----------------
Detects the skier bounding box in every frame using Ultralytics YOLO26
(detection task, person class = COCO class 0).

If only one skier is expected (typical race broadcast) we return the
highest-confidence person detection.  For multi-person scenes you can
extend this to return all boxes.
"""

import numpy as np
from typing import Optional, Tuple

from ultralytics import YOLO

BBox = Tuple[int, int, int, int]   # (x1, y1, x2, y2)

DEFAULT_MODEL = "yolo26s.pt"
PERSON_CLASS  = 0    # COCO class index for "person"


class SkierDetector:
    """
    Wraps YOLO26 detection to return the dominant skier bounding box.

    Parameters
    ----------
    model_path : str
        Ultralytics model name or path to fine-tuned .pt file.
    conf : float
        Minimum confidence threshold.
    device : str
        "cpu", "cuda", "mps", or "" (auto).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf: float = 0.40,
        device: str = "",
    ):
        self.model  = YOLO(model_path)
        self.conf   = conf
        self.device = device or ("cuda" if self._cuda_available() else "cpu")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Optional[BBox]:
        """
        Run detection on *frame* and return the best person bbox, or None.

        Returns
        -------
        (x1, y1, x2, y2) in pixel coordinates, or None if no person found.
        """
        results = self.model(
            frame,
            task="detect",
            conf=self.conf,
            classes=[PERSON_CLASS],
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return None

        boxes = results[0].boxes
        if len(boxes) == 0:
            return None

        # Pick highest-confidence detection
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

        # Add a small vertical padding to include skis
        h = y2 - y1
        pad = int(h * 0.10)
        y2 = min(frame.shape[0], y2 + pad)

        return int(x1), int(y1), int(x2), int(y2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False