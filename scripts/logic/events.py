"""
logic/events.py
---------------
Converts per-gate distance-over-time signals into gate passage events.

Algorithm
---------
1. Fill short NaN gaps by linear interpolation.
2. Smooth with a Gaussian (σ ≈ 2-3 frames) to suppress single-frame noise.
3. Find local minima with a refractory distance constraint (so one physical
   gate crossing cannot produce two events).
4. Sort events by frame → sequential gate_number assignment.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal  import find_peaks
from typing import Dict, List


def detect_gate_passages(
    signals: Dict[str, np.ndarray],
    fps: float,
    refractory_s: float = 0.35,
    sigma: float = 2.5,
    min_prominence: float = 15.0,   # pixels
    max_nan_gap: int = 5,           # frames — larger gaps stay NaN
) -> List[dict]:
    """
    Detect gate-passage frames from distance signals.

    Parameters
    ----------
    signals : {gate_id: array of distances (NaN where undetected)}
    fps : video frame rate
    refractory_s : minimum seconds between two passages of the same gate
    sigma : Gaussian smoothing σ in frames
    min_prominence : minimum drop depth (pixels) to count as a real passage
    max_nan_gap : consecutive NaN frames that get linearly interpolated

    Returns
    -------
    List of passage dicts sorted by frame:
        {gate_id, frame, min_dist_px}
    """
    refractory_frames = max(1, int(refractory_s * fps))
    events: List[dict] = []

    for gate_id, raw in signals.items():
        arr = raw.copy()

        # Interpolate short NaN runs
        arr = _interpolate_nans(arr, max_gap=max_nan_gap)

        # If still mostly NaN, skip this gate_id (spurious detection)
        if np.isnan(arr).mean() > 0.5:
            continue

        # Replace remaining NaN with large value so they don't form minima
        arr = np.where(np.isnan(arr), np.nanmax(arr) * 2, arr)

        smoothed = gaussian_filter1d(arr, sigma=sigma)

        # find_peaks on inverted = find minima
        peaks, props = find_peaks(
            -smoothed,
            distance=refractory_frames,
            prominence=min_prominence,
        )

        for frame_idx in peaks:
            events.append({
                "gate_id":    gate_id,
                "frame":      int(frame_idx),
                "min_dist_px": float(raw[frame_idx])
                               if not np.isnan(raw[frame_idx])
                               else float(smoothed[frame_idx]),
            })

    events.sort(key=lambda e: e["frame"])
    return events


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interpolate_nans(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Linearly interpolate NaN runs of length ≤ max_gap.
    Longer runs are left as NaN.
    """
    out = arr.copy()
    n   = len(arr)
    i   = 0
    while i < n:
        if np.isnan(out[i]):
            # find end of NaN run
            j = i
            while j < n and np.isnan(out[j]):
                j += 1
            gap = j - i
            if gap <= max_gap and i > 0 and j < n:
                # linear interpolation between out[i-1] and out[j]
                left  = out[i - 1]
                right = out[j]
                for k in range(gap):
                    out[i + k] = left + (right - left) * (k + 1) / (gap + 1)
            i = j
        else:
            i += 1
    return out