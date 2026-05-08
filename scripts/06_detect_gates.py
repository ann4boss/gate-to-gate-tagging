"""
06_detect_gates.py
------------------
Run gate passage detection on a single SwissSki run using both:

  1. Proximity method — minimum keypoint-to-gate-centroid distance signal;
     gate passage = local minimum.
  2. Tilt method — horizontal oscillation of the skier bounding box centre;
     gate passage = directional reversal (local min/max).

Both methods write a JSON result file in the manually-tagged schema:
    {run_id, fps, gates: [{gate_number, gate_label, position_ms, position_s,
                           frame, duration_ms, duration_s}]}

Usage:
    python 06_detect_gates.py <run_id>
    python 06_detect_gates.py <run_id> --method proximity
    python 06_detect_gates.py <run_id> --method tilt
    python 06_detect_gates.py <run_id> --compare   # print vs GT if available

Paths (relative to project root):
    Frames      : data/frames/swsk/<run_id>/
    Pose weights: runs/pose/skier_pose/weights/best.pt
    Gate weights: runs/detect/gate_poles/weights/best.pt
    GT annot    : data/annotations/swsk/<run_id>.json
    Output      : outputs/predictions/<run_id>_{method}.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
FRAMES_BASE    = "data/frames/swsk"
ANNOT_BASE     = "data/annotations/swsk"
OUT_DIR        = "outputs/predictions"
CACHE_DIR      = "outputs/cx_cache"

POSE_WEIGHTS   = "runs/pose/skier_pose/weights/best.pt"
GATE_WEIGHTS   = "runs/detect/gate_poles/weights/best.pt"

FPS            = 30

# ── Inference constants ───────────────────────────────────────────────────────
POSE_CONF      = 0.35
GATE_CONF      = 0.6
MIN_KP_CONF    = 0.20    # keypoints below this are ignored in proximity calc

# Kalman gate tracker constants
MATCH_DIST_PX  = 90
MAX_MISSING    = 10

# Proximity method
PROX_SIGMA        = 2.5    # Gaussian smoothing σ (frames)
PROX_PROMINENCE   = 5.0   # minimum signal dip (pixels)
PROX_REFRACTORY_S = 0.25   # minimum seconds between passages of same gate


# Tilt method
TILT_SMOOTH_W     = 7      # moving-average window (frames)
TILT_PROMINENCE   = 10     # minimum peak prominence (pixels)
TILT_MIN_SEP      = 15     # minimum frames between peaks
TILT_START_SKIP   = 30     # skip first N frames (camera settling)
TILT_REFRACTORY_S = 0.35


# ═══════════════════════════════════════════════════════════════════════════════
# Gate tracker (Kalman, colour-separated)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_kalman_2d(cx: float, cy: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix  = np.eye(2, 4, dtype=np.float32)
    kf.transitionMatrix   = np.float32([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 5e-1
    kf.statePre  = np.float32([[cx],[cy],[0.],[0.]])
    kf.statePost = kf.statePre.copy()
    kf.errorCovPre = np.eye(4, dtype=np.float32)
    return kf


class GateTracker:
    """Kalman-based multi-gate tracker returned as {gate_id: (cx, cy)}."""

    def __init__(self):
        self._trackers: dict = {}
        self._next_id = 0

    def update(self, detections: list) -> dict:
        """detections: list of (cx, cy, class_name) from YOLO gate detector."""
        predictions = {gid: t["kf"].predict()[:2].flatten()
                       for gid, t in self._trackers.items()}

        matched_det = set()
        for gid, trk in self._trackers.items():
            pred = predictions[gid]
            best_d, best_i = MATCH_DIST_PX, -1
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
                gid = f"{col}{self._next_id}"
                self._next_id += 1
                self._trackers[gid] = {
                    "color": col, "kf": _make_kalman_2d(cx, cy),
                    "cx": cx, "cy": cy, "missed": 0,
                }

        self._trackers = {gid: t for gid, t in self._trackers.items()
                          if t["missed"] <= MAX_MISSING}

        return {gid: (int(t["cx"]), int(t["cy"])) for gid, t in self._trackers.items()}

    def reset(self):
        self._trackers.clear()
        self._next_id = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Per-frame inference helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_pose(model: YOLO, frame: np.ndarray, device: str) -> list:
    """Return list of (x, y, conf) keypoints for the most-confident person."""
    res = model(frame, task="pose", conf=POSE_CONF, device=device, verbose=False)
    if not res or res[0].keypoints is None or len(res[0].boxes.conf) == 0:
        return []
    best_idx = int(res[0].boxes.conf.argmax())
    kpts = res[0].keypoints.data[best_idx].cpu().numpy()
    return [(float(x), float(y), float(c)) for x, y, c in kpts]


def run_gate_detector(model: YOLO, frame: np.ndarray, device: str) -> list:
    """Return list of (cx, cy, class_name) for all detected gate poles."""
    res = model(frame, task="detect", conf=GATE_CONF, device=device, verbose=False)
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        return []
    names = res[0].names
    out = []
    for box in res[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_name = names[int(box.cls[0])]
        out.append(((x1+x2)/2, (y1+y2)/2, cls_name))
    return out


def skier_cx_from_pose(model: YOLO, frame: np.ndarray, device: str) -> float | None:
    """Extract horizontal bounding-box centre for the tilt method."""
    res = model(frame, task="pose", conf=POSE_CONF, device=device, verbose=False)
    if not res or res[0].boxes is None or len(res[0].boxes.conf) == 0:
        return None
    best_idx = int(res[0].boxes.conf.argmax())
    x1, _, x2, _ = res[0].boxes.xyxy[best_idx].cpu().numpy()
    return float((x1 + x2) / 2)


def deduplicate_events(events: list[dict], fps: float, min_sep_s: float = 0.5) -> list[dict]:
    """
    Keep only one event per temporal cluster.
    Multiple gate IDs firing near the same frame = same physical gate passage.
    """
    if not events:
        return []
    
    events = sorted(events, key=lambda e: e["frame"])
    min_sep_frames = int(min_sep_s * fps)
    
    merged = [events[0]]
    for ev in events[1:]:
        if ev["frame"] - merged[-1]["frame"] >= min_sep_frames:
            merged.append(ev)
        else:
            # Keep whichever has smaller min_dist_px (closer passage)
            if ev["min_dist_px"] < merged[-1]["min_dist_px"]:
                merged[-1] = ev
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# Proximity method
# ═══════════════════════════════════════════════════════════════════════════════

def _min_kp_dist(keypoints: list, gate_cx: int, gate_cy: int) -> float | None:
    dists = [
        math.hypot(x - gate_cx, y - gate_cy)
        for x, y, c in keypoints
        if c >= MIN_KP_CONF
    ]
    return min(dists) if dists else None


def detect_gates_proximity(
    frame_dir: str,
    pose_model: YOLO,
    gate_model: YOLO,
    device: str,
    fps: float,
    cache_path: str | None = None,
) -> list[dict]:
    """
    Proximity method: detect gate passages as local minima in the
    keypoint-to-gate-centroid distance signal.
    """
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    gate_tracker = GateTracker()

    frame_data = []
    print(f"  [Proximity] processing {len(frames)} frames ...")

    for i, fname in enumerate(frames):
        img = cv2.imread(os.path.join(frame_dir, fname))
        if img is None:
            frame_data.append((i, [], {}))
            continue

        keypoints  = run_pose(pose_model, img, device)
        detections = run_gate_detector(gate_model, img, device)
        centroids  = gate_tracker.update(detections)
        if i < 5:  # print first 5 frames
            print(f"    frame {i}: pose_kpts={len(keypoints)}  gate_dets={len(detections)}  tracked={list(centroids.keys())}")
        frame_data.append((i, keypoints, centroids))

        if (i + 1) % 100 == 0:
            print(f"  [Proximity] {i+1}/{len(frames)}", flush=True)

    print()

    # Build per-gate distance signals
    all_gate_ids: set = set()
    for _, _, gates in frame_data:
        all_gate_ids.update(gates.keys())

    n = len(frame_data)
    signals: dict[str, np.ndarray] = {gid: np.full(n, np.nan) for gid in all_gate_ids}

    for fidx, (_, kpts, gates) in enumerate(frame_data):
        for gid, (cx, cy) in gates.items():
            d = _min_kp_dist(kpts, cx, cy)
            if d is not None:
                signals[gid][fidx] = d

    # Detect minima
    refractory = max(1, int(PROX_REFRACTORY_S * fps))
    events = []
    for gid, raw in signals.items():
        arr = _interpolate_nans(raw, max_gap=5)
        
        #valid_count = np.sum(~np.isnan(arr))
        #if valid_count < 10:   # need at least 10 valid frames to detect a passage
        #    continue
        
        valid_mask = ~np.isnan(arr)

        max_run = 0
        current_run = 0

        for is_valid in valid_mask:
            if is_valid:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        if max_run < 20:
            continue
        
        arr = np.where(np.isnan(arr), np.nanmax(arr) * 2, arr)
        smoothed = gaussian_filter1d(arr, sigma=PROX_SIGMA)
        peaks, props = find_peaks(-smoothed, distance=refractory, prominence=PROX_PROMINENCE)
        for frame_idx in peaks:
            events.append({
                "gate_id": gid,
                "frame":   int(frame_idx),
                "min_dist_px": float(raw[frame_idx]) if not np.isnan(raw[frame_idx]) else float(smoothed[frame_idx]),
            })

    events.sort(key=lambda e: e["frame"])
     # Deduplicate: if multiple gate IDs fire on the same frame, keep only the
     ## one with the smallest min_dist_px (closest approach).
    seen_frames: dict[int, dict] = {}
    for ev in events:
        f = ev["frame"]
        if f not in seen_frames or ev["min_dist_px"] < seen_frames[f]["min_dist_px"]:
            seen_frames[f] = ev
            events = [seen_frames[f] for f in sorted(seen_frames)]

    # ── DEBUG ──────────────────────────────────────────────────────────────
    print(f"\n  [DEBUG] Unique gate IDs tracked: {len(all_gate_ids)}")
    gate_detection_counts = {}
    for _, _, gates in frame_data:
        for gid in gates:
            gate_detection_counts[gid] = gate_detection_counts.get(gid, 0) + 1
    for gid, cnt in sorted(gate_detection_counts.items(), key=lambda x: -x[1])[:10]:
        arr = signals[gid]
        nan_pct = np.isnan(arr).mean() * 100
        valid = arr[~np.isnan(arr)]
        print(f"    {gid:12s}  frames={cnt:4d}  nan={nan_pct:5.1f}%  "
            f"min={valid.min():.1f}  max={valid.max():.1f}  range={valid.max()-valid.min():.1f}"
            if len(valid) else f"    {gid:12s}  frames={cnt:4d}  ALL NaN")
    print(f"  [DEBUG] Total events before return: {len(events)}")
# ── END DEBUG ──────────────────────────────────────────────────────────
    #events.sort(key=lambda e: e["frame"])
    #events = deduplicate_events(events, fps, min_sep_s=0.5)
    return events


def _interpolate_nans(arr: np.ndarray, max_gap: int) -> np.ndarray:
    out = arr.copy()
    n = len(arr)
    i = 0
    while i < n:
        if np.isnan(out[i]):
            j = i
            while j < n and np.isnan(out[j]):
                j += 1
            if j - i <= max_gap and i > 0 and j < n:
                left, right = out[i-1], out[j]
                for k in range(j - i):
                    out[i+k] = left + (right-left)*(k+1)/(j-i+1)
            i = j
        else:
            i += 1
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Tilt method
# ═══════════════════════════════════════════════════════════════════════════════

def detect_gates_tilt(
    frame_dir: str,
    pose_model: YOLO,
    device: str,
    fps: float,
    cache_path: str | None = None,
) -> list[dict]:
    """
    Tilt method: detect gate passages as directional reversals of the
    skier's horizontal bounding-box centre.
    """
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))

    # Use cache if available
    if cache_path and os.path.exists(cache_path):
        cx_arr = np.load(cache_path)
        print(f"  [Tilt] Loaded cached cx from {cache_path}")
    else:
        cx_arr = np.full(len(frames), np.nan)
        print(f"  [Tilt] processing {len(frames)} frames ...")
        for i, fname in enumerate(frames):
            img = cv2.imread(os.path.join(frame_dir, fname))
            if img is None:
                continue
            cx = skier_cx_from_pose(pose_model, img, device)
            if cx is not None:
                cx_arr[i] = cx
            if (i + 1) % 100 == 0:
                print(f"\r  [Tilt] {i+1}/{len(frames)}", end="", flush=True)
        print()
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, cx_arr)

    finite = np.isfinite(cx_arr)
    if finite.sum() < 10:
        return []

    idx    = np.arange(len(cx_arr))
    filled = np.interp(idx, idx[finite], cx_arr[finite])
    smoothed = uniform_filter1d(filled, size=TILT_SMOOTH_W)

    peaks_max, props_max = find_peaks( smoothed, distance=TILT_MIN_SEP, prominence=TILT_PROMINENCE)
    peaks_min, props_min = find_peaks(-smoothed, distance=TILT_MIN_SEP, prominence=TILT_PROMINENCE)

    all_idx  = np.concatenate([peaks_max, peaks_min])
    all_prom = np.concatenate([props_max["prominences"], props_min["prominences"]])
    order    = np.argsort(all_idx)
    all_idx  = all_idx[order]
    all_prom = all_prom[order]

    # Skip first N frames
    mask = all_idx >= TILT_START_SKIP
    all_idx  = all_idx[mask]
    all_prom = all_prom[mask]

    # Resolve close peaks (keep higher prominence)
    keep = np.ones(len(all_idx), dtype=bool)
    for i in range(1, len(all_idx)):
        if not keep[i-1]:
            continue
        if all_idx[i] - all_idx[i-1] < TILT_MIN_SEP:
            if all_prom[i] >= all_prom[i-1]:
                keep[i-1] = False
            else:
                keep[i] = False

    events = [{"gate_id": f"tilt_{j}", "frame": int(f), "min_dist_px": 0.0}
              for j, f in enumerate(sorted(all_idx[keep].tolist()))]
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════════════

def events_to_output(run_id: str, fps: float, events: list[dict]) -> dict:
    gates = []
    for i, ev in enumerate(events):
        frame  = ev["frame"]
        pos_ms = round(frame / fps * 1000)
        pos_s  = round(pos_ms / 1000, 3)
        if i + 1 < len(events):
            next_ms  = round(events[i+1]["frame"] / fps * 1000)
            dur_ms   = next_ms - pos_ms
            dur_s    = round(dur_ms / 1000, 3)
        else:
            dur_ms = dur_s = None
        gates.append({
            "gate_number": i + 1,
            "gate_label":  f"Gate {i+1}",
            "position_ms": pos_ms,
            "position_s":  pos_s,
            "frame":       frame,
            "duration_ms": dur_ms,
            "duration_s":  dur_s,
        })
    return {"run_id": run_id, "fps": fps, "gates": gates}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--method",     choices=["proximity", "tilt", "both"], default="both")
    p.add_argument("--pose-model", default=POSE_WEIGHTS)
    p.add_argument("--gate-model", default=GATE_WEIGHTS)
    p.add_argument("--device",     default="")
    p.add_argument("--compare",    action="store_true")
    p.add_argument("--no-cache",   action="store_true")
    args = p.parse_args()

    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = "cpu"

    frame_dir  = os.path.join(FRAMES_BASE, args.run_id)
    annot_path = os.path.join(ANNOT_BASE,  f"{args.run_id}.json")
    cache_path = os.path.join(CACHE_DIR,   f"{args.run_id}.npy")
    os.makedirs(OUT_DIR,   exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.isdir(frame_dir):
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    if args.no_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    print(f"\n=== Gate Detection: {args.run_id}  [{args.method}] ===\n")

    pose_model = YOLO(args.pose_model)
    methods    = [args.method] if args.method != "both" else ["proximity", "tilt"]

    results = {}
    for method in methods:
        print(f"--- {method.capitalize()} method ---")
        if method == "proximity":
            gate_model = YOLO(args.gate_model)
            events = detect_gates_proximity(frame_dir, pose_model, gate_model, device, FPS)
        else:
            events = detect_gates_tilt(frame_dir, pose_model, device, FPS, cache_path)

        output = events_to_output(args.run_id, FPS, events)
        out_path = os.path.join(OUT_DIR, f"{args.run_id}_{method}.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Gates detected: {len(events)}  →  {out_path}\n")
        results[method] = output

    if args.compare and os.path.exists(annot_path):
        with open(annot_path) as f:
            gt = json.load(f)
        gt_frames = [g["frame"] for g in gt["gates"]]
        for method, output in results.items():
            pr_frames = [g["frame"] for g in output["gates"]]
            errors = [abs(min(pr_frames, key=lambda p: abs(p-g), default=0) - g) for g in gt_frames]
            mean_err = np.mean(errors) if errors else float("nan")
            w033 = sum(1 for e in errors if e <= 10)
            w05  = sum(1 for e in errors if e <= 15)
            print(f"[{method}]  GT:{len(gt_frames)}  Pred:{len(pr_frames)}  "
                  f"MAE:{mean_err:.1f}f ({mean_err/FPS*1000:.0f}ms)  "
                  f"≤0.33s:{w033}/{len(gt_frames)}  ≤0.5s:{w05}/{len(gt_frames)}")


if __name__ == "__main__":
    main()
