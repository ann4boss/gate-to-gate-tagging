"""Shared functions for the gate detection pipeline (Approach 2)."""
from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from ultralytics import YOLO

YOLO_CONF      = 0.10
FPS            = 30
SMOOTH_WINDOW  = 7
MIN_PROMINENCE = 10
MIN_GATE_SEP   = 15
START_SKIP     = 30


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


def detect_gate_frames(cx: np.ndarray) -> list[int]:
    finite = np.isfinite(cx)
    if finite.sum() < 10:
        return []

    idx    = np.arange(len(cx))
    filled = np.interp(idx, idx[finite], cx[finite])
    smoothed = uniform_filter1d(filled, size=SMOOTH_WINDOW)

    peaks_max, props_max = find_peaks( smoothed, distance=MIN_GATE_SEP, prominence=MIN_PROMINENCE)
    peaks_min, props_min = find_peaks(-smoothed, distance=MIN_GATE_SEP, prominence=MIN_PROMINENCE)

    all_idx  = np.concatenate([peaks_max, peaks_min])
    all_prom = np.concatenate([props_max["prominences"], props_min["prominences"]])
    order    = np.argsort(all_idx)
    all_idx  = all_idx[order]
    all_prom = all_prom[order]

    mask    = all_idx >= START_SKIP
    all_idx  = all_idx[mask]
    all_prom = all_prom[mask]

    keep = np.ones(len(all_idx), dtype=bool)
    for i in range(1, len(all_idx)):
        if not keep[i - 1]:
            continue
        if all_idx[i] - all_idx[i - 1] < MIN_GATE_SEP:
            if all_prom[i] >= all_prom[i - 1]:
                keep[i - 1] = False
            else:
                keep[i] = False

    return sorted(all_idx[keep].tolist())


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
