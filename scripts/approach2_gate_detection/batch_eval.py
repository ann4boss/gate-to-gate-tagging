"""
Run gate detection on all available SwissSki runs and print a summary table.

Usage
─────
    python scripts/approach2_gate_detection/batch_eval.py
    python scripts/approach2_gate_detection/batch_eval.py --no-cache
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


def eval_run(run_id: str, model: YOLO, root: str, no_cache: bool) -> dict:
    frame_dir  = os.path.join(root, FRAMES_BASE, run_id)
    annot_path = os.path.join(root, ANNOT_BASE,  f"{run_id}.json")
    cache_path = os.path.join(root, CACHE_DIR,   f"{run_id}.npy")
    out_dir    = os.path.join(root, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(root, CACHE_DIR), exist_ok=True)

    if no_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    cx          = build_cx_series(frame_dir, model, cache_path)
    det_rate    = np.isfinite(cx).mean()
    gate_frames = detect_gate_frames(cx)

    result   = frames_to_json(gate_frames, run_id)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    row = {
        "run_id":    run_id,
        "det_rate":  det_rate,
        "predicted": len(gate_frames),
        "gt":        None,
        "mean_err":  None,
        "w033":      None,
        "w05":       None,
    }

    if os.path.exists(annot_path):
        with open(annot_path) as f:
            gt = json.load(f)
        gt_frames  = [g["frame"] for g in gt["gates"]]
        errors     = nearest_neighbour_errors(gt_frames, gate_frames)
        row["gt"]       = len(gt_frames)
        row["mean_err"] = np.mean(errors)
        row["w033"]     = sum(1 for e in errors if e <= 10)
        row["w05"]      = sum(1 for e in errors if e <= 15)

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    root  = os.path.join(os.path.dirname(__file__), "..", "..")
    frames_dir = os.path.join(root, FRAMES_BASE)
    runs  = sorted(r for r in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, r)))
    model = YOLO(os.path.join(root, WEIGHTS))

    results = []
    for i, run_id in enumerate(runs):
        print(f"\n[{i+1}/{len(runs)}] {run_id}")
        results.append(eval_run(run_id, model, root, args.no_cache))

    print("\n" + "=" * 100)
    print(f"{'Run':<50} {'Det%':>5} {'GT':>4} {'Pred':>5} {'Mean err':>9} {'≤0.33s':>8} {'≤0.5s':>7}")
    print("-" * 100)
    for r in results:
        disc = "SL" if "_SL_" in r["run_id"] else "GS"
        name = r["run_id"][:48]
        det  = f"{r['det_rate']*100:.0f}%"
        gt   = str(r["gt"])   if r["gt"]       is not None else "-"
        w033 = f"{r['w033']}/{r['gt']}" if r["gt"] else "-"
        w05  = f"{r['w05']}/{r['gt']}"  if r["gt"] else "-"
        merr = f"{r['mean_err']/FPS*1000:.0f}ms" if r["mean_err"] is not None else "-"
        print(f"[{disc}] {name:<48} {det:>5} {gt:>4} {r['predicted']:>5} {merr:>9} {w033:>8} {w05:>7}")

    # aggregate
    has_gt = [r for r in results if r["gt"] is not None]
    if has_gt:
        all_gt   = sum(r["gt"]   for r in has_gt)
        all_w033 = sum(r["w033"] for r in has_gt)
        all_w05  = sum(r["w05"]  for r in has_gt)
        avg_merr = np.mean([r["mean_err"] for r in has_gt]) / FPS * 1000
        print("-" * 100)
        print(f"{'TOTAL / MEAN':<55} {'':>5} {all_gt:>4} {'':>5} {avg_merr:>8.0f}ms {all_w033}/{all_gt} {all_w05}/{all_gt}")
    print()


if __name__ == "__main__":
    main()
