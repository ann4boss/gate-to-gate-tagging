"""
test_skier_pose_pipeline.py
---------------------------
Chains SkierDetector → PoseEstimator and writes an annotated MP4.

The skier bbox from SkierDetector is passed directly into PoseEstimator
so pose estimation runs on a tight crop around the skier rather than the
full frame. This is the intended usage in the final pipeline.

Colour coding in output video:
  Skier bbox:
    grey outline    = Kalman coasting (no YOLO match this frame)
    white outline   = YOLO matched

  Keypoints (17 COCO or 24 Ski-2DPose):
    yellow  = head   (nose, eyes, ears)
    green   = arms   (shoulders, elbows, wrists)
    orange  = hips
    cyan    = legs   (knees, ankles)
    purple  = ski/pole specific (only with fine-tuned 24-kpt model)

  Skeleton lines:
    light grey = COCO body connections
    purple     = ski/pole connections (24-kpt model only)

Console output:
  - Skier detection rate
  - Pose detection rate
  - Per-keypoint visibility table

Usage
-----
    # Pretrained weights (17 keypoints)
    python test_skier_pose_pipeline.py \
        --frames-dir data/frames/swsk/Aerni_Lauf1 \
        --out        test_pipeline_out.mp4

    # Fine-tuned weights (24 keypoints)
    python test_skier_pose_pipeline.py \
        --frames-dir data/frames/swsk/Aerni_Lauf1 \
        --out        test_pipeline_out.mp4 \
        --pose-model runs/pose/skier_pose_epfl/weights/best.pt \
        --det-model  runs/detect/skier_epfl/weights/best.pt

    # Quick 100-frame sanity check
    python test_skier_pose_pipeline.py \
        --frames-dir data/frames/swsk/Aerni_Lauf1 \
        --out        test_pipeline_out.mp4 \
        --max-frames 100
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from detector.skier import SkierDetector
from detector.pose  import PoseEstimator


# ── Skeleton ──────────────────────────────────────────────────────────────────
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
SKI_SKELETON = [
    (15,17),(15,18),   # left ankle → ski tip/tail
    (16,20),(16,21),   # right ankle → ski tip/tail
    (9, 19),(10,22),   # wrists → pole grips
    (19,23),           # left pole grip → pole tip
]

COCO_KP_NAMES = [
    "nose","l_eye","r_eye","l_ear","r_ear",
    "l_shoulder","r_shoulder","l_elbow","r_elbow",
    "l_wrist","r_wrist","l_hip","r_hip",
    "l_knee","r_knee","l_ankle","r_ankle",
]
SKI_KP_NAMES = [
    "l_ski_tip","l_ski_tail","l_pole_grip",
    "r_ski_tip","r_ski_tail","r_pole_grip",
    "pole_tip",
]


def kp_color(idx):
    if idx <= 4:             return (255, 220,   0)   # head:  yellow
    if idx <= 10:            return (  0, 255, 100)   # arms:  green
    if idx in (11, 12):      return (255, 120,   0)   # hips:  orange
    if idx <= 16:            return (  0, 210, 255)   # legs:  cyan
    return                          (220,   0, 255)   # ski/pole: purple


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_frame(frame, skier_bbox, bbox_state, keypoints, frame_idx, fps,
               skier_detected, pose_detected, total_so_far):
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── Skier bbox ────────────────────────────────────────────────────────────
    if skier_bbox is not None:
        x1, y1, x2, y2 = skier_bbox
        color = (160, 160, 160) if bbox_state == "coasting" else (255, 255, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = "COAST" if bbox_state == "coasting" else "SKIER"
        cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ── Skeleton lines ────────────────────────────────────────────────────────
    n_kpts   = len(keypoints)
    skeleton = COCO_SKELETON + (SKI_SKELETON if n_kpts >= 24 else [])

    for a, b in skeleton:
        if a >= n_kpts or b >= n_kpts:
            continue
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca <= 0 or cb <= 0:
            continue
        line_color = (180, 0, 220) if a >= 17 else (160, 160, 160)
        cv2.line(vis, (int(xa), int(ya)), (int(xb), int(yb)),
                 line_color, 1, cv2.LINE_AA)

    # ── Keypoints ─────────────────────────────────────────────────────────────
    for i, (x, y, c) in enumerate(keypoints):
        if c <= 0:
            continue
        cv2.circle(vis, (int(x), int(y)), 5 if i >= 17 else 4,
                   kp_color(i), -1, cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────────────────────
    t_s        = frame_idx / fps
    det_rate   = 100 * skier_detected / max(1, total_so_far)
    pose_rate  = 100 * pose_detected  / max(1, total_so_far)

    cv2.putText(vis, f"frame {frame_idx:05d}  t={t_s:.2f}s",
                (8, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(vis, f"skier {det_rate:.0f}%  pose {pose_rate:.0f}%",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    if not skier_bbox:
        cv2.putText(vis, "NO SKIER", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif not keypoints:
        cv2.putText(vis, "NO POSE", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    return vis


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Chain SkierDetector → PoseEstimator and write annotated MP4"
    )
    p.add_argument("--frames-dir",  required=True,
                   help="Folder of pre-extracted frames")
    p.add_argument("--out",         default="test_pipeline_out.mp4")
    p.add_argument("--det-model",   default="yolo26s.pt",
                   help="Skier detector weights (default: yolo26s.pt)")
    p.add_argument("--pose-model",  default="yolo26s-pose.pt",
                   help="Pose estimator weights (default: yolo26s-pose.pt)")
    p.add_argument("--det-conf",    type=float, default=0.40,
                   help="Skier detector confidence (default: 0.40)")
    p.add_argument("--pose-conf",   type=float, default=0.35,
                   help="Pose estimator confidence (default: 0.35)")
    p.add_argument("--max-frames",  type=int,   default=None)
    p.add_argument("--fps",         type=float, default=None)
    p.add_argument("--device",      default="",
                   help="cpu | cuda | mps (empty = auto)")
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
        f for f in frames_dir.iterdir() if f.suffix.lower() in IMG_EXTS
    )
    if not frame_paths:
        print(f"[ERROR] No images found in {frames_dir}", file=sys.stderr)
        sys.exit(1)
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    print(f"[INFO] {len(frame_paths)} frames to process")

    # ── Models ────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading skier detector : {args.det_model}")
    skier_detector = SkierDetector(
        model_path = args.det_model,
        conf       = args.det_conf,
        device     = args.device,
    )

    print(f"[INFO] Loading pose estimator : {args.pose_model}")
    pose_estimator = PoseEstimator(
        model_path = args.pose_model,
        conf       = args.pose_conf,
        device     = args.device,
    )

    # ── Video writer ──────────────────────────────────────────────────────────
    first = cv2.imread(str(frame_paths[0]))
    h, w  = first.shape[:2]
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_total         = len(frame_paths)
    n_skier         = 0   # frames where skier bbox was returned
    n_pose          = 0   # frames where keypoints were returned
    n_pose_no_skier = 0   # frames where pose ran on full frame (no bbox)
    kp_visible      = None
    n_kpts          = 0

    for idx, path in enumerate(tqdm(frame_paths, desc="Skier + Pose pipeline")):
        frame = cv2.imread(str(path))
        if frame is None:
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
            continue

        # ── Step 1: skier detection ───────────────────────────────────────────
        coast_before = skier_detector._coast_count
        skier_bbox   = skier_detector.detect(frame)

        coast_after  = skier_detector._coast_count
        if skier_bbox is not None:
            n_skier += 1
            bbox_state = ("coasting"
                          if coast_after > coast_before or coast_after > 0
                          else "detected")
        else:
            bbox_state = "none"

        # ── Step 2: pose estimation using skier bbox ──────────────────────────
        keypoints = pose_estimator.estimate(frame, skier_bbox)

        if not keypoints and skier_bbox is None:
            # Fallback: try pose on full frame so we still get something
            keypoints = pose_estimator.estimate(frame, skier_bbox=None)
            if keypoints:
                n_pose_no_skier += 1

        if keypoints:
            n_pose += 1
            if kp_visible is None:
                n_kpts     = len(keypoints)
                kp_visible = [0] * n_kpts
            for i, (x, y, c) in enumerate(keypoints):
                if c > 0:
                    kp_visible[i] += 1

        # ── Draw ──────────────────────────────────────────────────────────────
        vis = draw_frame(
            frame, skier_bbox, bbox_state, keypoints,
            idx, fps, n_skier, n_pose, idx + 1
        )
        writer.write(vis)

    writer.release()

    # ── Summary ───────────────────────────────────────────────────────────────
    all_kp_names = COCO_KP_NAMES + (SKI_KP_NAMES if n_kpts >= 24 else [])

    print(f"\n{'='*55}")
    print(f"  Total frames      : {n_total}")
    print(f"  Skier detected    : {n_skier:4d}  ({100*n_skier/n_total:.1f}%)")
    print(f"  Pose detected     : {n_pose:4d}  ({100*n_pose/n_total:.1f}%)")
    if n_pose_no_skier:
        print(f"  Pose (full frame) : {n_pose_no_skier:4d}  (fallback, no skier bbox)")
    print(f"  Keypoints/model   : {n_kpts}")

    if kp_visible and n_pose > 0:
        print(f"\n  Per-keypoint visibility (of {n_pose} frames with pose):")
        for i, count in enumerate(kp_visible):
            name = all_kp_names[i] if i < len(all_kp_names) else f"kp_{i}"
            pct  = 100 * count / n_pose
            bar  = "█" * int(20 * pct / 100)
            warn = "  ⚠️ " if pct < 50 else ""
            print(f"    {i:2d}  {name:<16} {pct:5.1f}%  {bar}{warn}")

    print(f"{'='*55}")
    print(f"\n  Output video : {args.out}\n")

    # ── Guidance ──────────────────────────────────────────────────────────────
    if n_skier / n_total < 0.70:
        print("[WARN] Skier detection rate low — check test_skier.py guidance")
    if n_pose / n_total < 0.60:
        print("[WARN] Pose detection rate low. Possible causes:")
        print("       - Skier detector failing (pose has no crop to work with)")
        print("       - Lower --pose-conf (try 0.25)")
    if n_kpts == 17:
        print("\n[NOTE] Using pretrained 17-keypoint COCO model.")
        print("       Ski-specific keypoints (ski tips, pole grips) not available yet.")
        print("       Rerun with --pose-model runs/pose/skier_pose_epfl/weights/best.pt")
        print("       once fine-tuning completes.")
    if n_skier > 0 and n_pose / max(1, n_skier) >= 0.85:
        print("[OK]  Pipeline looks healthy — good to move to gate detection.")


if __name__ == "__main__":
    main()