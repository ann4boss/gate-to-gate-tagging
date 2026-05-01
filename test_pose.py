"""
test_pose.py
------------
Visual test for PoseEstimator.

Reads a folder of pre-extracted frames, runs PoseEstimator on each one
(optionally using SkierDetector to crop first), draws keypoints + skeleton,
and writes an annotated MP4 you can scrub through.

Works with both:
  - yolo26s-pose.pt         (17 COCO keypoints)
  - fine-tuned best.pt      (24 Ski-2DPose keypoints)
The script auto-detects the number of keypoints from model output.

Console summary prints:
  - Detection rate (% of frames where a pose was found)
  - Per-keypoint visibility rate (helps spot which keypoints are unreliable)

Usage
-----
    # Pose only (no skier detector crop)
    python test_pose.py --frames-dir data/frames/MyRun_001 --out test_pose_out.mp4

    # With skier detector crop (more accurate, recommended)
    python test_pose.py --frames-dir data/frames/MyRun_001 --out test_pose_out.mp4 --use-skier-detector

    # With fine-tuned pose weights
    python test_pose.py --frames-dir data/frames/MyRun_001 --out test_pose_out.mp4 \
        --pose-model runs/pose/skier_pose_epfl/weights/best.pt

Optional flags
    --det-model      path to skier detector weights (default: yolo26s.pt)
    --pose-model     path to pose model weights (default: yolo26s-pose.pt)
    --conf           pose detection confidence threshold (default: 0.35)
    --max-frames     limit to first N frames (quick sanity check)
    --device         cpu | cuda | mps (default: auto)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from detector.pose import PoseEstimator


# ── Skeleton definitions ──────────────────────────────────────────────────────
# COCO 17-keypoint skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2),           # nose → eyes
    (1, 3), (2, 4),           # eyes → ears
    (5, 6),                   # shoulders
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 11), (6, 12),         # shoulders → hips
    (11, 12),                 # hips
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]

# Additional Ski-2DPose connections (only drawn when n_kpts == 24)
SKI_SKELETON = [
    (15, 17), (15, 18),       # left ankle → left ski tip/tail
    (16, 20), (16, 21),       # right ankle → right ski tip/tail
    (9,  19), (10, 22),       # wrists → pole grips
    (19, 23),                 # left pole grip → pole tip
]

COCO_KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]
SKI_KP_NAMES = [
    "l_ski_tip", "l_ski_tail", "l_pole_grip",
    "r_ski_tip",  "r_ski_tail",  "r_pole_grip",
    "pole_tip",
]

# Keypoint colours by body region
def kp_color(idx):
    if idx in (0, 1, 2, 3, 4):              return (255, 200,   0)   # head: yellow
    if idx in (5, 6, 7, 8, 9, 10):          return (  0, 255, 100)   # arms: green
    if idx in (11, 12):                      return (255, 100,   0)   # hips: orange
    if idx in (13, 14, 15, 16):              return (  0, 180, 255)   # legs: cyan
    return                                          (200,   0, 255)   # ski/pole: purple


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_pose(frame, keypoints, skier_bbox=None):
    vis = frame.copy()
    n   = len(keypoints)

    # Draw skier bbox if available
    if skier_bbox is not None:
        x1, y1, x2, y2 = skier_bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # Choose skeleton based on keypoint count
    skeleton = COCO_SKELETON + (SKI_SKELETON if n >= 24 else [])

    # Draw skeleton lines
    for a, b in skeleton:
        if a >= n or b >= n:
            continue
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca <= 0 or cb <= 0:
            continue
        cv2.line(vis,
                 (int(xa), int(ya)),
                 (int(xb), int(yb)),
                 (180, 180, 180), 1, cv2.LINE_AA)

    # Draw keypoints
    for i, (x, y, c) in enumerate(keypoints):
        if c <= 0:
            continue
        color  = kp_color(i)
        radius = 5 if i >= 17 else 4   # slightly larger for ski keypoints
        cv2.circle(vis, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

    return vis


def draw_overlay(frame, keypoints, skier_bbox, frame_idx, fps, n_detected, n_frames):
    vis = draw_pose(frame, keypoints, skier_bbox)
    h   = frame.shape[0]

    t_s  = frame_idx / fps
    rate = 100 * n_detected / max(1, frame_idx + 1)
    info = f"frame {frame_idx:05d}  t={t_s:.2f}s  detection rate {rate:.0f}%"
    cv2.putText(vis, info, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if not keypoints:
        cv2.putText(vis, "NO POSE", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Visual test for PoseEstimator")
    p.add_argument("--frames-dir",        required=True)
    p.add_argument("--out",               default="test_pose_out.mp4")
    p.add_argument("--pose-model",        default="yolo26s-pose.pt")
    p.add_argument("--det-model",         default="yolo26s.pt")
    p.add_argument("--use-skier-detector",action="store_true",
                   help="Crop to skier bbox before pose estimation (recommended)")
    p.add_argument("--conf",              type=float, default=0.35)
    p.add_argument("--max-frames",        type=int,   default=None)
    p.add_argument("--fps",               type=float, default=None)
    p.add_argument("--device",            default="")
    args = p.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        print(f"[ERROR] Not a directory: {frames_dir}", file=sys.stderr)
        sys.exit(1)

    # ── FPS ───────────────────────────────────────────────────────────────────
    fps = args.fps
    meta_path = frames_dir / "meta.json"
    if fps is None and meta_path.exists():
        with open(meta_path) as f:
            fps = float(json.load(f).get("fps", 25))
        print(f"[INFO] FPS from meta.json: {fps}")
    elif fps is None:
        fps = 25.0
        print(f"[WARN] No meta.json, defaulting to {fps} FPS")

    # ── Frames ────────────────────────────────────────────────────────────────
    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    frame_paths = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if not frame_paths:
        print(f"[ERROR] No images in {frames_dir}", file=sys.stderr)
        sys.exit(1)
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    print(f"[INFO] {len(frame_paths)} frames to process")

    # ── Models ────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading pose model: {args.pose_model}")
    estimator = PoseEstimator(
        model_path  = args.pose_model,
        conf        = args.conf,
        device      = args.device,
    )

    skier_detector = None
    if args.use_skier_detector:
        print(f"[INFO] Loading skier detector: {args.det_model}")
        from detector.skier import SkierDetector
        skier_detector = SkierDetector(
            model_path = args.det_model,
            device     = args.device,
        )

    # ── Video writer ──────────────────────────────────────────────────────────
    first = cv2.imread(str(frame_paths[0]))
    h, w  = first.shape[:2]
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_detected  = 0
    n_total     = len(frame_paths)
    # Per-keypoint visibility counts — sized after first detection
    kp_visible  = None
    n_kpts      = 0

    for idx, path in enumerate(tqdm(frame_paths, desc="Testing pose estimator")):
        frame = cv2.imread(str(path))
        if frame is None:
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
            continue

        # Optionally get skier bbox first
        skier_bbox = None
        if skier_detector is not None:
            skier_bbox = skier_detector.detect(frame)

        keypoints = estimator.estimate(frame, skier_bbox)

        if keypoints:
            n_detected += 1
            if kp_visible is None:
                n_kpts     = len(keypoints)
                kp_visible = [0] * n_kpts
            for i, (x, y, c) in enumerate(keypoints):
                if c > 0:
                    kp_visible[i] += 1

        vis = draw_overlay(frame, keypoints, skier_bbox, idx, fps,
                           n_detected, n_total)
        writer.write(vis)

    writer.release()

    # ── Summary ───────────────────────────────────────────────────────────────
    all_kp_names = COCO_KP_NAMES + (SKI_KP_NAMES if n_kpts >= 24 else [])

    print(f"\n{'='*55}")
    print(f"  Total frames    : {n_total}")
    print(f"  Pose detected   : {n_detected:4d}  ({100*n_detected/n_total:.1f}%)")
    print(f"  Keypoints/model : {n_kpts}")
    print(f"\n  Per-keypoint visibility (of {n_detected} detected frames):")

    if kp_visible and n_detected > 0:
        for i, count in enumerate(kp_visible):
            name = all_kp_names[i] if i < len(all_kp_names) else f"kp_{i}"
            bar  = "█" * int(20 * count / n_detected)
            pct  = 100 * count / n_detected
            flag = "  ⚠️ " if pct < 50 else ""
            print(f"    {i:2d}  {name:<16} {pct:5.1f}%  {bar}{flag}")

    print(f"{'='*55}")
    print(f"\n  Output video : {args.out}\n")

    # ── Guidance ──────────────────────────────────────────────────────────────
    if n_detected / n_total < 0.70:
        print("[WARN] Detection rate is low (<70%). Try:")
        print("       --conf 0.25  to lower the threshold")
        print("       --use-skier-detector  to crop before pose estimation")
    elif n_detected / n_total >= 0.85:
        print("[OK]  Pose detector looks healthy.")

    if n_kpts == 17:
        print("\n[NOTE] Running with pretrained 17-keypoint COCO model.")
        print("       Ski-specific keypoints (ski tips, pole grips) will only")
        print("       appear after fine-tuning on Ski-2DPose completes.")
        print("       Pass --pose-model runs/pose/skier_pose_epfl/weights/best.pt")
        print("       once training finishes to test the full 24-keypoint model.")


if __name__ == "__main__":
    main()
