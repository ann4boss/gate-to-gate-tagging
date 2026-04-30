"""
test_skier.py
-------------
Visual test for SkierDetector.

Reads a folder of pre-extracted frames (sorted alphabetically),
runs SkierDetector on each one, draws the bounding box + tracker
state, and writes an annotated MP4 you can scrub through.

Also prints a short summary to stdout:
  - detection rate (% of frames where a bbox was returned)
  - coast events (frames where Kalman prediction was used, no YOLO match)
  - track resets

Usage
-----
    python test_skier.py --frames-dir data/frames/MyRun_001 --out test_out.mp4

Optional flags
    --fps        FPS for the output video (default: read from meta.json, else 25)
    --conf       YOLO confidence threshold (default: 0.40)
    --max-frames Limit to first N frames (handy for a quick sanity check)
    --device     cpu | cuda | mps  (default: auto)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── make sure the detector package is importable ──────────────────────────────
# Adjust this if your project layout differs
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from detector.skier import SkierDetector


# ── drawing helpers ───────────────────────────────────────────────────────────

COLORS = {
    "detected": (0,   255,  0),    # green  — YOLO match
    "coasting": (0,   200, 255),   # yellow — Kalman prediction only
    "none":     (0,     0,  0),    # (not drawn)
}

def draw_overlay(frame, bbox, state, frame_idx, fps, coast_count):
    vis = frame.copy()
    h_frame = frame.shape[0]

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color = COLORS[state]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = "YOLO+KF" if state == "detected" else f"COAST({coast_count})"
        cv2.putText(vis, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Frame counter + timestamp
    t_s = frame_idx / fps
    info = f"frame {frame_idx:05d}  t={t_s:.2f}s"
    cv2.putText(vis, info, (8, h_frame - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Visual test for SkierDetector")
    p.add_argument("--frames-dir", required=True,
                   help="Folder of pre-extracted frames (jpg/png)")
    p.add_argument("--out",        default="test_skier_out.mp4",
                   help="Output annotated video path")
    p.add_argument("--fps",        type=float, default=None,
                   help="Output FPS (default: from meta.json, else 25)")
    p.add_argument("--conf",       type=float, default=0.40,
                   help="YOLO confidence threshold")
    p.add_argument("--max-frames", type=int,   default=None,
                   help="Only process the first N frames")
    p.add_argument("--device",     default="",
                   help="Inference device: cpu | cuda | mps (empty=auto)")
    args = p.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        print(f"[ERROR] Not a directory: {frames_dir}", file=sys.stderr)
        sys.exit(1)

    # ── FPS: prefer meta.json, then CLI flag, then default ───────────────────
    fps = args.fps
    meta_path = frames_dir / "meta.json"
    if fps is None and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        fps = float(meta.get("fps", 25))
        print(f"[INFO] FPS from meta.json: {fps}")
    elif fps is None:
        fps = 25.0
        print(f"[WARN] No meta.json and --fps not set, defaulting to {fps}")

    # ── Collect frames ────────────────────────────────────────────────────────
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frame_paths = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if not frame_paths:
        print(f"[ERROR] No images found in {frames_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]

    print(f"[INFO] Found {len(frame_paths)} frames to process")

    # ── Peek at first frame for video dimensions ──────────────────────────────
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        print(f"[ERROR] Could not read {frame_paths[0]}", file=sys.stderr)
        sys.exit(1)
    h, w = first.shape[:2]

    # ── Video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    # ── Detector ──────────────────────────────────────────────────────────────
    detector = SkierDetector(conf=args.conf, device=args.device)

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_detected = 0   # frames with a YOLO match
    n_coasting = 0   # frames using Kalman prediction only
    n_none     = 0   # frames returning None
    n_resets   = 0   # track resets

    prev_had_track = False

    for idx, path in enumerate(tqdm(frame_paths, desc="Testing skier detector")):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[WARN] Skipping unreadable frame: {path}", file=sys.stderr)
            n_none += 1
            continue

        coast_before = detector._coast_count
        had_track_before = detector._kf is not None

        bbox = detector.detect(frame)

        coast_after  = detector._coast_count
        has_track_now = detector._kf is not None

        # Detect a reset: had a track, now don't (and bbox is None)
        if had_track_before and not has_track_now:
            n_resets += 1

        # Classify the state for colouring
        if bbox is None:
            state = "none"
            n_none += 1
        elif coast_after > coast_before or (coast_after > 0 and coast_before > 0):
            state = "coasting"
            n_coasting += 1
        else:
            state = "detected"
            n_detected += 1

        vis = draw_overlay(frame, bbox, state, idx, fps, coast_after)
        writer.write(vis)

    writer.release()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(frame_paths)
    print(f"\n{'='*50}")
    print(f"  Total frames   : {total}")
    print(f"  YOLO matched   : {n_detected:4d}  ({100*n_detected/total:.1f}%)")
    print(f"  Coasting (KF)  : {n_coasting:4d}  ({100*n_coasting/total:.1f}%)")
    print(f"  No detection   : {n_none:4d}  ({100*n_none/total:.1f}%)")
    print(f"  Track resets   : {n_resets}")
    print(f"{'='*50}")
    print(f"\n  Output video   : {args.out}")
    print()

    # ── Guidance ──────────────────────────────────────────────────────────────
    if n_detected / total < 0.70:
        print("[WARN] Detection rate is low (<70%). Consider:")
        print("       - Lowering --conf (try 0.25-0.35)")
        print("       - Fine-tuning YOLO on the EPFL dataset")
    if n_resets > 5:
        print("[WARN] Many track resets. Consider:")
        print("       - Increasing --conf to avoid latching onto spectators")
        print("       - Increasing max_coast_frames in skier.py")
    if n_detected / total >= 0.85 and n_resets <= 2:
        print("[OK]  Tracker looks healthy. Good to move to pose estimation.")


if __name__ == "__main__":
    main()
