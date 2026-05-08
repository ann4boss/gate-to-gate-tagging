"""
Cached proximity detection library for Approach 1.

06_detect_gates.py is the source of truth for all detection logic.
This library imports from it and adds a caching layer so that:
  - YOLO inference runs once and raw detections are saved to disk
  - Kalman tracking + signal processing runs on cached data (fast, tunable)
"""

from __future__ import annotations

import importlib.util
import json
import os

import cv2
import numpy as np

# ── Import from 06_detect_gates.py (source of truth) ─────────────────────────
# Filename starts with a digit so standard import doesn't work — use importlib.

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "detect_gates", os.path.join(_HERE, "..", "06_detect_gates.py")
)
_dg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dg)

# Re-export constants and functions from 06_detect_gates.py
FPS               = _dg.FPS
POSE_CONF         = _dg.POSE_CONF
GATE_CONF         = _dg.GATE_CONF
MIN_KP_CONF       = _dg.MIN_KP_CONF
MATCH_DIST_PX     = _dg.MATCH_DIST_PX
MAX_MISSING       = _dg.MAX_MISSING
PROX_SIGMA        = _dg.PROX_SIGMA
PROX_PROMINENCE   = _dg.PROX_PROMINENCE
PROX_REFRACTORY_S = _dg.PROX_REFRACTORY_S

run_pose          = _dg.run_pose
run_gate_detector = _dg.run_gate_detector
GateTracker       = _dg.GateTracker
_min_kp_dist      = _dg._min_kp_dist
_interpolate_nans = _dg._interpolate_nans


# ── Cache build (slow, once) ──────────────────────────────────────────────────

def build_proximity_cache(frame_dir, pose_model, gate_model, device, cache_path):
    """
    Run YOLO inference on all frames and save raw detections to cache.
    Returns frame_data: list of {kpts, gates} per frame.
    """
    if os.path.exists(cache_path):
        print(f"  Loading cached proximity data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    import cv2 as _cv2
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    print(f"  Building proximity cache: {len(frames)} frames ...")

    frame_data = []
    for i, fname in enumerate(frames):
        img = _cv2.imread(os.path.join(frame_dir, fname))
        if img is None:
            frame_data.append({"kpts": [], "gates": []})
            continue
        frame_data.append({
            "kpts":  [list(kp) for kp in run_pose(pose_model, img, device)],
            "gates": [list(g)  for g  in run_gate_detector(gate_model, img, device)],
        })
        if (i + 1) % 100 == 0:
            print(f"\r  {i+1}/{len(frames)}", end="", flush=True)

    print()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(frame_data, f)
    print(f"  Cache saved → {cache_path}")
    return frame_data


# ── Signal processing on cache (fast, tunable) ────────────────────────────────

def detect_from_cache(
    frame_data: list,
    match_dist_px: float = MATCH_DIST_PX,
    max_missing: int = MAX_MISSING,
    prox_sigma: float = PROX_SIGMA,
    prox_prominence: float = PROX_PROMINENCE,
    prox_refractory_s: float = PROX_REFRACTORY_S,
    fps: float = FPS,
) -> list[int]:
    """
    Kalman tracking + distance signal + peak detection on cached frame data.
    Uses GateTracker and helpers imported directly from 06_detect_gates.py.
    Returns sorted list of gate passage frame indices.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    tracker = GateTracker()
    # Temporarily patch MATCH_DIST_PX and MAX_MISSING used inside GateTracker
    _dg.MATCH_DIST_PX = match_dist_px
    _dg.MAX_MISSING   = max_missing

    centroids_per_frame = []
    for fd in frame_data:
        centroids = tracker.update(fd["gates"])
        centroids_per_frame.append(dict(centroids))

    # Restore original values
    _dg.MATCH_DIST_PX = MATCH_DIST_PX
    _dg.MAX_MISSING   = MAX_MISSING

    all_gate_ids: set = set()
    for c in centroids_per_frame:
        all_gate_ids.update(c.keys())

    n = len(frame_data)
    signals = {gid: np.full(n, np.nan) for gid in all_gate_ids}

    for fidx, (fd, centroids) in enumerate(zip(frame_data, centroids_per_frame)):
        for gid, (cx, cy) in centroids.items():
            d = _min_kp_dist(fd["kpts"], cx, cy)
            if d is not None:
                signals[gid][fidx] = d

    refractory = max(1, int(prox_refractory_s * fps))
    events = []
    for gid, raw in signals.items():
        arr = _interpolate_nans(raw, max_gap=5)
        if np.sum(~np.isnan(arr)) < 10:
            continue
        arr = np.where(np.isnan(arr), np.nanmax(arr) * 2, arr)
        smoothed = gaussian_filter1d(arr, sigma=prox_sigma)
        peaks, _ = find_peaks(-smoothed, distance=refractory, prominence=prox_prominence)
        events.extend(int(f) for f in peaks)

    return sorted(events)
