"""
Cached proximity detection library for Approach 1.

Separates the slow part (YOLO inference → cache) from the fast part
(Kalman tracking + signal processing → tunable without re-inference).

Cache format: JSON with per-frame raw pose keypoints and gate detections,
stored before Kalman tracking so all downstream params can be re-tuned.
"""

from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from ultralytics import YOLO

FPS           = 30
POSE_CONF     = 0.35
GATE_CONF     = 0.35
MIN_KP_CONF   = 0.20


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_pose(model: YOLO, frame: np.ndarray, device: str) -> list:
    res = model(frame, task="pose", conf=POSE_CONF, device=device, verbose=False)
    if not res or res[0].keypoints is None or len(res[0].boxes.conf) == 0:
        return []
    best_idx = int(res[0].boxes.conf.argmax())
    kpts = res[0].keypoints.data[best_idx].cpu().numpy()
    return [[float(x), float(y), float(c)] for x, y, c in kpts]


def _run_gate_detector(model: YOLO, frame: np.ndarray, device: str) -> list:
    res = model(frame, task="detect", conf=GATE_CONF, device=device, verbose=False)
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        return []
    names = res[0].names
    out = []
    for box in res[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_name = names[int(box.cls[0])]
        out.append([(x1 + x2) / 2, (y1 + y2) / 2, cls_name])
    return out


# ── Cache build (slow, once) ──────────────────────────────────────────────────

def build_proximity_cache(
    frame_dir: str,
    pose_model: YOLO,
    gate_model: YOLO,
    device: str,
    cache_path: str,
) -> list:
    """
    Run YOLO inference on all frames and save raw detections to cache.
    Returns the loaded frame data (list of {kpts, gates} per frame).
    """
    if os.path.exists(cache_path):
        print(f"  Loading cached proximity data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    print(f"  Building proximity cache: {len(frames)} frames ...")

    frame_data = []
    for i, fname in enumerate(frames):
        img = cv2.imread(os.path.join(frame_dir, fname))
        if img is None:
            frame_data.append({"kpts": [], "gates": []})
            continue
        kpts  = _run_pose(pose_model, img, device)
        gates = _run_gate_detector(gate_model, img, device)
        frame_data.append({"kpts": kpts, "gates": gates})
        if (i + 1) % 100 == 0:
            print(f"\r  {i+1}/{len(frames)}", end="", flush=True)

    print()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(frame_data, f)
    print(f"  Cache saved → {cache_path}")
    return frame_data


# ── Kalman tracker ────────────────────────────────────────────────────────────

def _make_kalman(cx: float, cy: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix   = np.eye(2, 4, dtype=np.float32)
    kf.transitionMatrix    = np.float32([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-1
    kf.statePre  = np.float32([[cx],[cy],[0.],[0.]])
    kf.statePost = kf.statePre.copy()
    kf.errorCovPre = np.eye(4, dtype=np.float32)
    return kf


def _track_gates(frame_data: list, match_dist_px: float, max_missing: int) -> list:
    """
    Re-run Kalman gate tracking on cached raw detections.
    Returns list of {gate_id: (cx, cy)} dicts, one per frame.
    """
    trackers: dict = {}
    next_id = 0
    centroids_per_frame = []

    for fd in frame_data:
        detections = fd["gates"]  # [[cx, cy, class], ...]

        # predict
        predictions = {gid: t["kf"].predict()[:2].flatten()
                       for gid, t in trackers.items()}

        matched_det = set()
        for gid, trk in trackers.items():
            pred = predictions[gid]
            best_d, best_i = match_dist_px, -1
            for i, (cx, cy, col) in enumerate(detections):
                if col != trk["color"] or i in matched_det:
                    continue
                d = math.hypot(cx - pred[0], cy - pred[1])
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                cx, cy, _ = detections[best_i]
                trk["kf"].correct(np.float32([[cx],[cy]]))
                trk["cx"], trk["cy"] = cx, cy
                trk["missed"] = 0
                matched_det.add(best_i)
            else:
                trk["missed"] += 1

        for i, (cx, cy, col) in enumerate(detections):
            if i not in matched_det:
                gid = f"{col}{next_id}"
                next_id += 1
                trackers[gid] = {
                    "color": col, "kf": _make_kalman(cx, cy),
                    "cx": cx, "cy": cy, "missed": 0,
                }

        trackers = {gid: t for gid, t in trackers.items()
                    if t["missed"] <= max_missing}

        centroids_per_frame.append(
            {gid: (int(t["cx"]), int(t["cy"])) for gid, t in trackers.items()}
        )

    return centroids_per_frame


def _min_kp_dist(kpts: list, gate_cx: int, gate_cy: int) -> float | None:
    dists = [
        math.hypot(x - gate_cx, y - gate_cy)
        for x, y, c in kpts
        if c >= MIN_KP_CONF
    ]
    return min(dists) if dists else None


def _interpolate_nans(arr: np.ndarray, max_gap: int = 5) -> np.ndarray:
    out = arr.copy()
    n, i = len(arr), 0
    while i < n:
        if np.isnan(out[i]):
            j = i
            while j < n and np.isnan(out[j]):
                j += 1
            if j - i <= max_gap and i > 0 and j < n:
                left, right = out[i-1], out[j]
                for k in range(j - i):
                    out[i+k] = left + (right - left) * (k+1) / (j - i + 1)
            i = j
        else:
            i += 1
    return out


# ── Signal processing (fast, tunable) ────────────────────────────────────────

def detect_from_cache(
    frame_data: list,
    match_dist_px: float,
    max_missing: int,
    prox_sigma: float,
    prox_prominence: float,
    prox_refractory_s: float,
    fps: float = FPS,
) -> list[int]:
    """
    Run Kalman tracking + distance signal + peak detection on cached frame data.
    Returns list of gate passage frame indices.
    """
    centroids_per_frame = _track_gates(frame_data, match_dist_px, max_missing)

    all_gate_ids: set = set()
    for c in centroids_per_frame:
        all_gate_ids.update(c.keys())

    n = len(frame_data)
    signals: dict[str, np.ndarray] = {gid: np.full(n, np.nan) for gid in all_gate_ids}

    for fidx, (fd, centroids) in enumerate(zip(frame_data, centroids_per_frame)):
        for gid, (cx, cy) in centroids.items():
            d = _min_kp_dist(fd["kpts"], cx, cy)
            if d is not None:
                signals[gid][fidx] = d

    refractory = max(1, int(prox_refractory_s * fps))
    events = []
    for gid, raw in signals.items():
        arr = _interpolate_nans(raw)
        if np.sum(~np.isnan(arr)) < 10:
            continue
        arr = np.where(np.isnan(arr), np.nanmax(arr) * 2, arr)
        smoothed = gaussian_filter1d(arr, sigma=prox_sigma)
        peaks, _ = find_peaks(-smoothed, distance=refractory, prominence=prox_prominence)
        for frame_idx in peaks:
            events.append(int(frame_idx))

    return sorted(events)
