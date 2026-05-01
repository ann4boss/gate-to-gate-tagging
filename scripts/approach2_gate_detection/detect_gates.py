"""
Approach 2: Gate passage detection via lateral oscillation of the skier's bounding box.

Strategy
────────
YOLO detects the skier in each frame and records cx (horizontal centre of the
bounding box).  The signal is smoothed and local minima/maxima are found —
each reversal corresponds to one gate passage.

Usage
─────
    python scripts/detect_gates.py <run_id>
    python scripts/detect_gates.py <run_id> --compare   # compare vs GT
    python scripts/detect_gates.py <run_id> --no-cache  # force re-run YOLO

    <run_id> must match a folder under data/frames/swsk/
    e.g.  Aerni_SUI_Lauf1_20251116_SL_Levi

Paths (relative to project root)
─────────────────────────────────
    Frames      : data/frames/swsk/<run_id>/
    Weights     : runs/skier_epfl/skier_weights/best.pt
    Annotations : data/annotations/swsk/<run_id>.json   (GT, optional)
    Cache       : outputs/cx_cache/<run_id>.npy
    Output      : outputs/predictions/<run_id>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))

from gate_detection_lib import (
    FPS,
    build_cx_series,
    detect_gate_frames,
    frames_to_json,
    nearest_neighbour_errors,
)

WEIGHTS     = "runs/pose_epfl/pose_weights/best.pt"
FRAMES_BASE = "data/frames/swsk"
ANNOT_BASE  = "data/annotations/swsk"
OUT_DIR     = "outputs/predictions"
CACHE_DIR   = "outputs/cx_cache"


def compare_to_gt(predicted: dict, gt_path: str) -> None:
    with open(gt_path) as f:
        gt = json.load(f)

    gt_frames = [g["frame"] for g in gt["gates"]]
    pr_frames = [g["frame"] for g in predicted["gates"]]
    errors    = nearest_neighbour_errors(gt_frames, pr_frames)

    within05 = sum(1 for e in errors if e <= 15)
    within03 = sum(1 for e in errors if e <= 10)

    print(f"\n  GT gates  : {len(gt_frames)}")
    print(f"  Predicted : {len(pr_frames)}")
    print(f"\n{'Gate':>5}  {'GT frame':>9}  {'Nearest pred':>12}  {'Error':>8}  {'ms':>6}")
    print("-" * 52)
    for i, (gt_f, err) in enumerate(zip(gt_frames, errors)):
        nearest = min(pr_frames, key=lambda p: abs(p - gt_f))
        flag    = "  ← >" if err > 15 else ""
        print(f"{i+1:>5}  {gt_f:>9}  {nearest:>12}  {err:>+8}  {err/FPS*1000:>5.0f}ms{flag}")

    print()
    print(f"  Mean abs error : {np.mean(errors):.1f} frames  ({np.mean(errors)/FPS*1000:.0f} ms)")
    print(f"  Median error   : {np.median(errors):.1f} frames  ({np.median(errors)/FPS*1000:.0f} ms)")
    print(f"  Within 0.33 s  : {within03}/{len(gt_frames)}")
    print(f"  Within 0.5 s   : {within05}/{len(gt_frames)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect gate passages in a SwissSki run.")
    parser.add_argument("run_id", help="Run folder name under data/frames/swsk/")
    parser.add_argument("--compare",  action="store_true", help="Compare vs GT annotation")
    parser.add_argument("--no-cache", action="store_true", help="Force re-run YOLO inference")
    args = parser.parse_args()

    root       = os.path.join(os.path.dirname(__file__), "..", "..")
    frame_dir  = os.path.join(root, FRAMES_BASE, args.run_id)
    annot_path = os.path.join(root, ANNOT_BASE,  f"{args.run_id}.json")
    cache_path = os.path.join(root, CACHE_DIR,   f"{args.run_id}.npy")
    out_dir    = os.path.join(root, OUT_DIR)
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(os.path.join(root, CACHE_DIR), exist_ok=True)

    if not os.path.isdir(frame_dir):
        print(f"ERROR: frame directory not found: {frame_dir}")
        sys.exit(1)

    if args.no_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    print("Gate passage detection — Approach 2 (lateral oscillation)")
    print(f"  Run     : {args.run_id}")
    print(f"  Weights : {WEIGHTS}")
    print()

    model = YOLO(os.path.join(root, WEIGHTS))

    cx = build_cx_series(frame_dir, model, cache_path)
    print(f"  Frames with detection: {np.isfinite(cx).sum()}/{len(cx)}")

    gate_frames = detect_gate_frames(cx)
    print(f"  Gates detected: {len(gate_frames)}")

    result   = frames_to_json(gate_frames, args.run_id)
    out_path = os.path.join(out_dir, f"{args.run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Prediction saved: {out_path}")

    if args.compare:
        if not os.path.exists(annot_path):
            print(f"WARNING: no GT annotation found at {annot_path}")
        else:
            compare_to_gt(result, annot_path)


if __name__ == "__main__":
    main()
