"""Shared functions for the gate detection pipeline (Approach 2)."""
from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from ultralytics import YOLO

YOLO_CONF      = 0.10
FPS            = 30
SMOOTH_WINDOW  = 11
MIN_PROMINENCE = 5
MIN_GATE_SEP   = 10
START_SKIP     = 0
END_SKIP       = 0
MIN_KP_CONF    = 0.30

# Ski-2DPose 24-keypoint indices (EPFL custom skeleton)
SHOULDER_KPS = [2, 6]   # shoulder_R, shoulder_L
HIP_KPS      = [10, 13] # hip_R, hip_L


def build_cx_series(frame_dir: str, model: YOLO, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        print(f"  Loading cached cx series from {cache_path}")
        return np.load(cache_path)

    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    cx = np.full(len(frames), np.nan)

    for i, fname in enumerate(frames):
        r = model(os.path.join(frame_dir, fname), conf=YOLO_CONF, verbose=False)[0]
        if r.boxes and len(r.boxes):
            best = int(r.boxes.conf.argmax())
            cx[i] = float(r.boxes.xywh[best][0])
        print(f"\r  {i+1}/{len(frames)}", end="", flush=True)

    print()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, cx)
    return cx


def build_lean_series(frame_dir: str, model: YOLO, cache_path: str) -> np.ndarray:
    """Extract torso lean signal: shoulder_mid_x − hip_mid_x per frame."""
    if os.path.exists(cache_path):
        print(f"  Loading cached lean series from {cache_path}")
        return np.load(cache_path)

    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    lean = np.full(len(frames), np.nan)

    for i, fname in enumerate(frames):
        r = model(os.path.join(frame_dir, fname), conf=YOLO_CONF, verbose=False)[0]
        if not (r.boxes and len(r.boxes) and r.keypoints is not None and len(r.keypoints)):
            print(f"\r  {i+1}/{len(frames)}", end="", flush=True)
            continue

        best    = int(r.boxes.conf.argmax())
        kp_xy   = r.keypoints.xy[best]    # [24, 2]
        kp_conf = r.keypoints.conf[best]  # [24]

        kp_idxs = SHOULDER_KPS + HIP_KPS
        if all(float(kp_conf[j]) >= MIN_KP_CONF for j in kp_idxs):
            shoulder_x = (float(kp_xy[SHOULDER_KPS[0]][0]) + float(kp_xy[SHOULDER_KPS[1]][0])) / 2
            hip_x      = (float(kp_xy[HIP_KPS[0]][0])      + float(kp_xy[HIP_KPS[1]][0]))      / 2
            lean[i]    = shoulder_x - hip_x

        print(f"\r  {i+1}/{len(frames)}", end="", flush=True)

    print()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, lean)
    return lean


def combine_signals(cx: np.ndarray, lean: np.ndarray) -> np.ndarray:
    """Normalise cx and lean to unit IQR then average, falling back to whichever is available."""
    def normalise(s: np.ndarray) -> np.ndarray:
        finite = s[np.isfinite(s)]
        if len(finite) < 2:
            return s
        q25, q75 = np.percentile(finite, [25, 75])
        iqr = q75 - q25
        if iqr < 1e-6:
            return s
        return (s - np.nanmedian(s)) / iqr

    cx_n   = normalise(cx)
    lean_n = normalise(lean)

    both      = np.isfinite(cx_n) & np.isfinite(lean_n)
    cx_only   = np.isfinite(cx_n) & ~np.isfinite(lean_n)
    lean_only = ~np.isfinite(cx_n) & np.isfinite(lean_n)

    combined = np.full(len(cx), np.nan)
    combined[both]      = (cx_n[both] + lean_n[both]) / 2
    combined[cx_only]   = cx_n[cx_only]
    combined[lean_only] = lean_n[lean_only]
    return combined


def detect_gate_frames(signal: np.ndarray, min_gate_sep: int = MIN_GATE_SEP) -> list[int]:
    finite = np.isfinite(signal)
    if finite.sum() < 10:
        return []

    idx      = np.arange(len(signal))
    filled   = np.interp(idx, idx[finite], signal[finite])
    smoothed = uniform_filter1d(filled, size=SMOOTH_WINDOW)

    peaks_max, props_max = find_peaks( smoothed, distance=min_gate_sep, prominence=MIN_PROMINENCE)
    peaks_min, props_min = find_peaks(-smoothed, distance=min_gate_sep, prominence=MIN_PROMINENCE)

    # track peak type: +1 = local max, -1 = local min
    all_idx  = np.concatenate([peaks_max,              peaks_min])
    all_prom = np.concatenate([props_max["prominences"], props_min["prominences"]])
    all_type = np.array([1] * len(peaks_max) + [-1] * len(peaks_min))
    order    = np.argsort(all_idx)
    all_idx  = all_idx[order]
    all_prom = all_prom[order]
    all_type = all_type[order]

    mask     = (all_idx >= START_SKIP) & (all_idx < len(signal) - END_SKIP)
    all_idx  = all_idx[mask]
    all_prom = all_prom[mask]
    all_type = all_type[mask]

    # proximity deduplication: if two peaks within min_gate_sep, keep more prominent
    keep = np.ones(len(all_idx), dtype=bool)
    for i in range(1, len(all_idx)):
        if not keep[i - 1]:
            continue
        if all_idx[i] - all_idx[i - 1] < min_gate_sep:
            if all_prom[i] >= all_prom[i - 1]:
                keep[i - 1] = False
            else:
                keep[i] = False
    all_idx  = all_idx[keep]
    all_prom = all_prom[keep]
    all_type = all_type[keep]

    # alternation constraint: consecutive peaks must alternate max/min
    # if two consecutive peaks are the same type, drop the less prominent one
    result = []
    for i in range(len(all_idx)):
        if not result or all_type[i] != all_type[result[-1]]:
            result.append(i)
        elif all_prom[i] > all_prom[result[-1]]:
            result[-1] = i

    return sorted(all_idx[result].tolist())


def frames_to_json(gate_frames: list[int], run_id: str, fps: int = FPS) -> dict:
    gates = []
    for i, frame in enumerate(gate_frames):
        position_ms = round(frame / fps * 1000)
        entry = {
            "gate_number": i + 1,
            "gate_label":  f"Gate {i + 1}",
            "position_ms": position_ms,
            "position_s":  round(position_ms / 1000, 3),
            "frame":       frame,
        }
        if i + 1 < len(gate_frames):
            next_ms = round(gate_frames[i + 1] / fps * 1000)
            entry["duration_ms"] = next_ms - position_ms
            entry["duration_s"]  = round((next_ms - position_ms) / 1000, 3)
        gates.append(entry)
    return {"run_id": run_id, "fps": fps, "gates": gates}


def nearest_neighbour_errors(gt_frames: list[int], pr_frames: list[int]) -> list[int]:
    errors = []
    for gt_f in gt_frames:
        if not pr_frames:
            break
        errors.append(abs(min(pr_frames, key=lambda p: abs(p - gt_f)) - gt_f))
    return errors


def match_gates(gt_frames: list[int], pr_frames: list[int], threshold: int) -> tuple[int, int, int]:
    """Greedy one-to-one matching within threshold frames.

    Returns (TP, FP, FN):
      TP = matched pairs
      FP = predictions with no GT match
      FN = GT gates with no prediction match
    """
    if not gt_frames or not pr_frames:
        return 0, len(pr_frames), len(gt_frames)

    pairs = sorted(
        ((abs(g - p), gi, pi) for gi, g in enumerate(gt_frames) for pi, p in enumerate(pr_frames)),
        key=lambda x: x[0],
    )
    matched_gt = set()
    matched_pr = set()
    for err, gi, pi in pairs:
        if err > threshold:
            break
        if gi not in matched_gt and pi not in matched_pr:
            matched_gt.add(gi)
            matched_pr.add(pi)

    tp = len(matched_gt)
    fp = len(pr_frames) - tp
    fn = len(gt_frames)  - tp
    return tp, fp, fn
