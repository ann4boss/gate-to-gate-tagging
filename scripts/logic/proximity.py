"""
logic/proximity.py
------------------
For every frame, computes the minimum Euclidean distance from any visible
keypoint to each tracked gate centroid.

Output is a dict {gate_id: [dist_frame0, dist_frame1, ...]} — one value per
frame, with NaN when the gate or the skier was not visible in that frame.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple

Keypoint    = Tuple[float, float, float]          # (x, y, conf)
GateCentroids = Dict[str, Tuple[int, int]]        # {gate_id: (cx, cy)}
FrameRecord = Tuple[int, List[Keypoint], GateCentroids]


def compute_distances(
    frame_data: List[FrameRecord],
    min_kpt_conf: float = 0.20,
) -> Dict[str, List[float]]:
    """
    Build per-gate distance-over-time signals.

    Parameters
    ----------
    frame_data : list of (frame_idx, keypoints, gate_centroids)
        One entry per processed video frame.
    min_kpt_conf : float
        Keypoints with confidence below this value are ignored.

    Returns
    -------
    dict  {gate_id: np.array of distances, length == len(frame_data)}
        Values are NaN where gate or skier was not detected.
    """
    # Collect the full set of gate ids seen across all frames
    all_gate_ids: set = set()
    for _, _, gates in frame_data:
        all_gate_ids.update(gates.keys())

    n_frames = len(frame_data)
    signals: Dict[str, np.ndarray] = {
        gid: np.full(n_frames, np.nan) for gid in all_gate_ids
    }

    for fidx, (_, keypoints, gates) in enumerate(frame_data):
        # Filter to confident keypoints only
        valid_kpts = [
            (x, y) for x, y, c in keypoints
            if c >= min_kpt_conf
        ]
        if not valid_kpts:
            continue  # leave NaN for this frame

        for gate_id, (cx, cy) in gates.items():
            min_dist = min(
                math.hypot(x - cx, y - cy) for x, y in valid_kpts
            )
            signals[gate_id][fidx] = min_dist

    return signals