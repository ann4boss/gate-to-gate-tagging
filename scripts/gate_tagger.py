"""
gate_tagger.py
==============
Main CLI entry point for automatic ski gate tagging.

Pipeline per frame:
    1. SkierDetector    — YOLO26 + Kalman tracker → skier bbox
    2. PoseEstimator    — YOLO26-pose → 24 keypoints (or 17 pretrained)
    3. GateDetector     — YOLO26 fine-tuned → gate_contact / gate_outer poles
    4. Proximity        — distance from pole-grip keypoint to nearest contact pole
    5. Event detection  — Gaussian smoothed distance minima → passage timestamps

Note: TCN classifier is skipped. Gate passages are detected using the
proximity/distance-minimum heuristic from logic/events.py.  This is the
baseline approach — TCN can be added later as a drop-in replacement for
step 5.

Supports two input modes:
    1. --video PATH          Process a video file directly
    2. --frames-dir PATH     Process a folder of pre-extracted frames

Usage
-----
    # From a video file:
    python gate_tagger.py --video race.mp4 --run-id ATHLETE_01 --out out.json

    # From pre-extracted frames:
    python gate_tagger.py --frames-dir data/frames/swsk/Aerni_Lauf1 \
        --fps 25 --run-id Aerni_Lauf1 --out out.json

    # With all fine-tuned models:
    python gate_tagger.py --frames-dir data/frames/swsk/Aerni_Lauf1 \
        --fps 25 \
        --det-model   runs/detect/skier_epfl/weights/best.pt \
        --pose-model  runs/pose/skier_pose_epfl/weights/best.pt \
        --gate-model  runs/detect/gate_poles/weights/best.pt \
        --run-id      Aerni_Lauf1 \
        --out         out/Aerni_Lauf1.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

# ── Project imports ───────────────────────────────────────────────────────────
from detector.skier import SkierDetector
from detector.gates import GateDetector
from detector.pose  import PoseEstimator
from logic.proximity import compute_distances
from logic.events    import detect_gate_passages
from output.formatter import build_output


# ── Frame sources ─────────────────────────────────────────────────────────────

def _frames_from_video(video_path: str):
    """Generator: yields (frame_index, BGR ndarray, fps, total_frames)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx   = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame, fps, total
        idx += 1
    cap.release()


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def _frames_from_dir(frames_dir: str, fps: float):
    """Generator: yields (frame_index, BGR ndarray, fps, total_frames)."""
    # Try to read fps from meta.json if present
    meta_path = Path(frames_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        fps = float(meta.get("fps", fps))
        print(f"[INFO] FPS from meta.json: {fps}")

    paths = sorted(
        p for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in _IMG_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"No image files found in {frames_dir}")
    total = len(paths)
    for idx, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[WARN] Could not read {path}, skipping.", file=sys.stderr)
            continue
        yield idx, frame, fps, total


# ── Keypoint helpers ──────────────────────────────────────────────────────────
 
# Keypoint indices matching Ski-2DPose 24-kpt model
# Falls back gracefully to COCO 17-kpt wrists if ski-specific kpts absent
_KP_LEFT_WRIST      = 9
_KP_RIGHT_WRIST     = 10
_KP_LEFT_ANKLE      = 15
_KP_RIGHT_ANKLE     = 16
_KP_LEFT_POLE_GRIP  = 19
_KP_RIGHT_POLE_GRIP = 22

def _point_to_vertical_segment_distance(px: float, py: float, bbox) -> float:
    """
    Distance from a point to the visible gate-pole segment.

    The gate bbox is approximated as a vertical pole line:
        x = center of bbox
        y from bbox top to bbox bottom
    """
    x1, y1, x2, y2 = bbox

    pole_x = (x1 + x2) / 2.0
    closest_y = min(max(py, y1), y2)

    return ((px - pole_x) ** 2 + (py - closest_y) ** 2) ** 0.5


def _closest_hand_to_pole(kpts: list, gate_bbox):
    """
    Return (x, y) of the hand / pole-grip keypoint closest to the gate pole.

    Priority:
    1. Ski-specific pole grips, if using the 24-kpt model.
    2. Wrists, as fallback.
    """
    if not kpts:
        return None

    grip_candidates = []

    for idx in (_KP_LEFT_POLE_GRIP, _KP_RIGHT_POLE_GRIP):
        if idx < len(kpts):
            x, y, c = kpts[idx]
            if c > 0:
                d = _point_to_vertical_segment_distance(x, y, gate_bbox)
                grip_candidates.append((x, y, d))

    if grip_candidates:
        best = min(grip_candidates, key=lambda t: t[2])
        return best[0], best[1]

    wrist_candidates = []

    for idx in (_KP_LEFT_WRIST, _KP_RIGHT_WRIST):
        if idx < len(kpts):
            x, y, c = kpts[idx]
            if c > 0:
                d = _point_to_vertical_segment_distance(x, y, gate_bbox)
                wrist_candidates.append((x, y, d))

    if wrist_candidates:
        best = min(wrist_candidates, key=lambda t: t[2])
        return best[0], best[1]

    return None
 
 
def _ankle_midpoint(kpts: list):
    # Return midpoint of left and right ankle, or None if not both visible.
    # Ankles are large joints, consistently detected, and provide a stable
    # vertical position signal useful as a secondary distance metric.
    if len(kpts) <= _KP_RIGHT_ANKLE:
        return None
    _, _, cl = kpts[_KP_LEFT_ANKLE]
    _, _, cr = kpts[_KP_RIGHT_ANKLE]
    if cl <= 0 or cr <= 0:
        return None
    xl, yl, _ = kpts[_KP_LEFT_ANKLE]
    xr, yr, _ = kpts[_KP_RIGHT_ANKLE]
    return ((xl + xr) / 2.0, (yl + yr) / 2.0)
 
 
def _combined_distance_to_pole(kpts: list, gate_bbox):
    """
    Compute combined distance from skier keypoints to the actual pole segment.

    Uses:
    - pole grip / wrist distance as primary signal
    - ankle midpoint distance as secondary signal
    """
    hand = _closest_hand_to_pole(kpts, gate_bbox)
    ankle = _ankle_midpoint(kpts)

    def _dist(pos):
        if pos is None:
            return None

        return _point_to_vertical_segment_distance(pos[0], pos[1], gate_bbox)

    d_hand = _dist(hand)
    d_ankle = _dist(ankle)

    if d_hand is not None and d_ankle is not None:
        return 0.8 * d_hand + 0.2 * d_ankle

    if d_hand is not None:
        return d_hand

    if d_ankle is not None:
        return d_ankle

    return None
 
 
def _hip_midpoint(kpts: list):
    """Return midpoint of hips (kpts 11+12), or None."""
    if len(kpts) < 13:
        return None
    _, _, c11 = kpts[11]
    _, _, c12 = kpts[12]
    if c11 <= 0 or c12 <= 0:
        return None
    x = (kpts[11][0] + kpts[12][0]) / 2.0
    y = (kpts[11][1] + kpts[12][1]) / 2.0
    return x, y

def _filter_contact_gates_near_skier(gates: dict, skier_bbox, margin: int = 120) -> dict:
    """
    Keep only contact-gate detections that are spatially plausible for the skier.

    This removes duplicate / far-away / irrelevant gate detections before building
    the distance signal.
    """
    if skier_bbox is None:
        return {
            gid: info
            for gid, info in gates.items()
            if info["class"] == 0
        }

    sx1, sy1, sx2, sy2 = skier_bbox

    out = {}

    for gid, info in gates.items():
        if info["class"] != 0:
            continue

        x1, y1, x2, y2 = info["bbox"]
        cx = info["cx"]

        near_x = sx1 - margin <= cx <= sx2 + margin
        vertical_overlap = not (y2 < sy1 - margin or y1 > sy2 + margin)

        if near_x and vertical_overlap:
            out[gid] = info

    return out

# ── Core processing ───────────────────────────────────────────────────────────

def process(
    frame_source,
    run_id:         str,
    det_model:      str   = "yolo26s.pt",
    pose_model:     str   = "yolo26s-pose.pt",
    gate_model:     str   = "runs/detect/gate_poles/weights/best.pt",
    det_conf:       float = 0.40,
    pose_conf:      float = 0.35,
    gate_conf:      float = 0.45,
    refractory_s:   float = 0.35,
    sigma:          float = 2.0,
    min_prominence: float = 0.06,
    device:         str   = "",
    debug_dir:      Optional[str] = None,
) -> dict:
    """
    Process all frames and return a gate-passage JSON dict.

    Parameters
    ----------
    frame_source    : generator yielding (idx, frame, fps, total)
    run_id          : identifier written into output JSON
    det_model       : skier detector weights path
    pose_model      : pose estimator weights path
    gate_model      : gate pole detector weights path
    det_conf        : skier detector confidence threshold
    pose_conf       : pose estimator confidence threshold
    gate_conf       : gate detector confidence threshold
    refractory_s    : minimum seconds between two passages of the same gate
    sigma           : Gaussian smoothing sigma for distance signal (frames)
    min_prominence  : minimum distance-signal prominence for a valid passage
    device          : inference device
    debug_dir       : if set, write annotated debug frames here
    """

    # ── Initialise detectors ──────────────────────────────────────────────────
    print(f"[INFO] Loading skier detector  : {det_model}")
    skier_det = SkierDetector(det_model,  conf=det_conf,  device=device)

    print(f"[INFO] Loading pose estimator  : {pose_model}")
    pose_est  = PoseEstimator(pose_model,  conf=pose_conf, device=device)

    print(f"[INFO] Loading gate detector   : {gate_model}")
    gate_det  = GateDetector(gate_model,  conf=gate_conf, device=device,
                             contact_only=False)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

# ── Per-frame data collection ─────────────────────────────────────────────
# frame_data: list of (frame_idx, keypoints, filtered_gate_info_dict, skier_bbox)
# gate_info_dict: {gate_id: {"cx", "cy", "bbox", "class", "label", "conf"}}
    frame_data = []
    fps_value = None

    with tqdm(desc="Processing frames", unit="frame") as pbar:
        for frame_idx, frame, fps, total in frame_source:
            if fps_value is None:
                fps_value = fps
                pbar.total = total

            # Skip first/last 5 seconds after fps is known
            skip_frames = int(5 * fps_value)

            if frame_idx < skip_frames:
                continue

            if skip_frames and total and frame_idx >= total - skip_frames:
                continue

            # Step 1: skier bbox
            skier_bbox = skier_det.detect(frame)

            # Step 2: pose keypoints, cropped to skier if available
            kpts = pose_est.estimate(frame, skier_bbox)

            # Step 3: gate poles
            gates_all = gate_det.update(frame)

            # Step 4: keep only plausible contact gates near the skier
            gates = _filter_contact_gates_near_skier(
                gates_all,
                skier_bbox,
                margin=120,
            )

            frame_data.append((frame_idx, kpts, gates, skier_bbox))

            if debug_dir:
                _draw_debug(frame, skier_bbox, kpts, gates, frame_idx, fps_value, debug_dir)

            pbar.update(1)

    if fps_value is None:
        raise RuntimeError("No frames were processed.")

    n_frames       = len(frame_data)
    unique_gate_ids = sorted({gid for _, _, g, _ in frame_data for gid in g})
    print(f"\n[INFO] Processed {n_frames} frames at {fps_value:.2f} fps")
    print(f"[INFO] Unique gate IDs tracked: {unique_gate_ids}")

    # ── Distance signal: hand/grip keypoint → nearest contact pole ────────────
    # compute_distances expects: list of (frame_idx, keypoints, gate_centroids)
    # gate_centroids format expected by proximity.py: {gate_id: (cx, cy)}
    # We pass only gate_contact poles (class 0) for the distance signal
    # Build per-gate distance signals using combined hand + ankle metric.
    # Instead of passing raw keypoints to compute_distances (which uses all
    # confident keypoints), we pre-compute the combined distance per gate per
    # frame and pass synthetic single-keypoint data so proximity.py works as-is.
    from collections import defaultdict

    # Build {gate_id: [(local_idx, original_frame_idx, normalized_dist)]}
    gate_signal_raw = defaultdict(list)

    for local_idx, item in enumerate(frame_data):
        frame_idx, kpts, gates, skier_bbox = item

        for gid, info in gates.items():
            if info["class"] != 0:
                continue

            # Step 1: distance to actual pole segment, not bbox centroid
            dist_px = _combined_distance_to_pole(kpts, info["bbox"])

            if dist_px is None:
                continue

            # Step 5: normalize by skier height
            if skier_bbox is not None:
                sx1, sy1, sx2, sy2 = skier_bbox
                skier_h = float(sy2 - sy1)
                dist = dist_px / max(skier_h, 1.0)
            else:
                # Fallback if skier bbox is missing
                dist = dist_px

            gate_signal_raw[gid].append((local_idx, frame_idx, dist))

    # Convert to {gate_id: np.array}, indexed by local processed-frame index.
    # This fixes the frame-indexing bug when skipped frames are used.
    n_frames = len(frame_data)
    signals = {}
    frame_lookup = {}

    for gid, readings in gate_signal_raw.items():
        arr = np.full(n_frames, np.nan)

        for local_idx, original_frame_idx, dist in readings:
            arr[local_idx] = dist
            frame_lookup[local_idx] = original_frame_idx

        signals[gid] = arr
 
    # Fall back to proximity.py if no signals computed (safety net)
    if not signals:
        print("[WARN] No pole-segment distance signals computed.")
        print("[WARN] Falling back to proximity.py centroid distance computation.")

        from logic.proximity import compute_distances as _cd

        proximity_data = []

        for local_idx, item in enumerate(frame_data):
            frame_idx, kpts, gates, skier_bbox = item

            contact_centroids = {
                gid: (info["cx"], info["cy"])
                for gid, info in gates.items()
                if info["class"] == 0
            }

            proximity_data.append((local_idx, kpts, contact_centroids))
            frame_lookup[local_idx] = frame_idx

        signals = _cd(proximity_data)
    passages = detect_gate_passages(
        signals,
        fps           = fps_value,
        refractory_s  = refractory_s,
        sigma         = sigma,
        min_prominence= min_prominence,
    )

    # Event detection runs on local processed-frame indices.
    # Convert back to original video frame indices for the output JSON.
    for event in passages:
        local_frame = int(event["frame"])
        event["local_frame"] = local_frame
        event["frame"] = int(frame_lookup.get(local_frame, local_frame))
 
    print(f"[INFO] Gate passages detected: {len(passages)}")
    return build_output(run_id=run_id, fps=fps_value, passages=passages)
 


# ── Debug visualisation ───────────────────────────────────────────────────────

def _draw_debug(frame, skier_bbox, kpts, gates, frame_idx, fps, debug_dir):
    vis = frame.copy()

    # Skier bbox
    if skier_bbox is not None:
        x1, y1, x2, y2 = skier_bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "SKIER", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Keypoints
    for x, y, c in kpts:
        if c >= 0.20:
            cv2.circle(vis, (int(x), int(y)), 4, (255, 180, 0), -1)

    # Gate poles — colour by class
    for gid, info in gates.items():
        color = (0, 0, 220) if info["class"] == 0 else (220, 80, 0)
        cx, cy = info["cx"], info["cy"]
        x1, y1, x2, y2 = info["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis,  (cx, cy), 5, color, -1)
        cv2.putText(vis, f"{info['label']} {info['conf']:.2f}",
                    (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # Frame info
    t_ms = frame_idx / fps * 1000
    cv2.putText(vis, f"frame={frame_idx}  t={t_ms:.0f}ms",
                (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    out_path = os.path.join(debug_dir, f"{frame_idx:06d}.jpg")
    cv2.imwrite(out_path, vis)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automatic ski gate tagger — skier + pose + gate detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",      metavar="PATH", help="Input video file")
    src.add_argument("--frames-dir", metavar="PATH",
                     help="Directory of pre-extracted frames (sorted)")

    p.add_argument("--fps",       type=float, default=25.0,
                   help="Frame rate (used with --frames-dir if no meta.json)")
    p.add_argument("--run-id",    default="RUN_001",
                   help="Identifier written into the output JSON")
    p.add_argument("--out",       default="gates.json",
                   help="Output JSON file path")

    # ── Model paths ───────────────────────────────────────────────────────────
    p.add_argument("--det-model",  default="yolo26s.pt",
                   help="Skier detector weights")
    p.add_argument("--pose-model", default="yolo26s-pose.pt",
                   help="Pose estimator weights")
    p.add_argument("--gate-model",
                   default="runs/detect/gate_poles/weights/best.pt",
                   help="Gate pole detector weights")

    # ── Confidence thresholds ─────────────────────────────────────────────────
    p.add_argument("--det-conf",   type=float, default=0.40)
    p.add_argument("--pose-conf",  type=float, default=0.35)
    p.add_argument("--gate-conf",  type=float, default=0.45)

    # ── Event detection tuning ────────────────────────────────────────────────
    p.add_argument("--refractory", type=float, default=0.35,
                   help="Min seconds between two passages of the same gate")
    p.add_argument("--sigma",      type=float, default=2.0,
                   help="Gaussian smoothing σ (frames) for distance signal")
    p.add_argument("--min-prom",   type=float, default=0.06,
                   help="Min prominence for normalized distance signal; try 0.04–0.08")

    p.add_argument("--device",     default="",
                   help="Inference device: cpu | cuda | mps (empty = auto)")
    p.add_argument("--debug-dir",  metavar="PATH", default=None,
                   help="Write annotated debug frames to this folder")
    return p


def main():
    args = _build_parser().parse_args()

    if args.frames_dir:
        source = _frames_from_dir(args.frames_dir, fps=args.fps)
    else:
        source = _frames_from_video(args.video)

    result = process(
        frame_source   = source,
        run_id         = args.run_id,
        det_model      = args.det_model,
        pose_model     = args.pose_model,
        gate_model     = args.gate_model,
        det_conf       = args.det_conf,
        pose_conf      = args.pose_conf,
        gate_conf      = args.gate_conf,
        refractory_s   = args.refractory,
        sigma          = args.sigma,
        min_prominence = args.min_prom,
        device         = args.device,
        debug_dir      = args.debug_dir,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    n = len(result.get("gates", []))
    print(f"\n[DONE] {n} gate(s) written → {args.out}")


if __name__ == "__main__":
    main()