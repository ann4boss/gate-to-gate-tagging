"""
Unified batch evaluation for gate detection.

Methods
───────
  tilt      — Approach 2: horizontal skier oscillation (cx signal).
              Slow on first run, cached afterwards.

  proximity — Approach 1: minimum keypoint-to-gate-pole distance.
              Slow on first run (two YOLO models), cached afterwards.

Usage
─────
    python scripts/batch_eval.py --method tilt
    python scripts/batch_eval.py --method proximity
    python scripts/batch_eval.py --method both
    python scripts/batch_eval.py --method both --runs Gut_SUI_Lauf1_20251025_GS_Soelden
    python scripts/batch_eval.py --method tilt --no-cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "approach2_gate_detection"))
sys.path.insert(0, os.path.join(_HERE, "approach1_gate_detection"))

from gate_detection_lib import (     # noqa: E402  (approach 2 tilt)
    FPS,
    build_cx_series,
    detect_gate_frames,
    match_gates,
    nearest_neighbour_errors,
)
from proximity_lib import (          # noqa: E402  (approach 1 proximity)
    build_proximity_cache,
    detect_from_cache,
)

POSE_WEIGHTS = "runs/pose_epfl/pose_weights/best.pt"
GATE_WEIGHTS = "runs/gate_poles/gate_poles_weights/best.pt"
FRAMES_BASE  = "data/frames/swsk"
ANNOT_BASE   = "data/annotations/swsk"
OUT_DIR      = "outputs/predictions"
TILT_CACHE   = "outputs/cx_cache"
PROX_CACHE   = "outputs/proximity_cache"

# Proximity detection params (tune with tune_params_approach1_proximity.py)
MATCH_DIST_PX      = 90
MAX_MISSING        = 10
PROX_SIGMA         = 2.5
PROX_PROMINENCE    = 5.0
PROX_REFRACTORY_S  = 0.25


# ── Per-run evaluation ────────────────────────────────────────────────────────

def eval_tilt(run_id: str, model: YOLO, root: str, no_cache: bool) -> dict:
    frame_dir  = os.path.join(root, FRAMES_BASE, run_id)
    annot_path = os.path.join(root, ANNOT_BASE,  f"{run_id}.json")
    cache_path = os.path.join(root, TILT_CACHE,  f"{run_id}.npy")
    os.makedirs(os.path.join(root, TILT_CACHE), exist_ok=True)
    os.makedirs(os.path.join(root, OUT_DIR),    exist_ok=True)

    if no_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    cx          = build_cx_series(frame_dir, model, cache_path)
    gate_frames = detect_gate_frames(cx)
    _save_prediction(run_id, gate_frames, root, suffix="tilt")
    return _build_row(run_id, gate_frames, annot_path)


def eval_proximity(run_id: str, pose_model: YOLO, gate_model: YOLO,
                   root: str, no_cache: bool, device: str) -> dict:
    frame_dir  = os.path.join(root, FRAMES_BASE, run_id)
    annot_path = os.path.join(root, ANNOT_BASE,  f"{run_id}.json")
    cache_path = os.path.join(root, PROX_CACHE,  f"{run_id}.json")
    os.makedirs(os.path.join(root, PROX_CACHE), exist_ok=True)
    os.makedirs(os.path.join(root, OUT_DIR),    exist_ok=True)

    if no_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    frame_data  = build_proximity_cache(frame_dir, pose_model, gate_model, device, cache_path)
    gate_frames = detect_from_cache(
        frame_data, MATCH_DIST_PX, MAX_MISSING,
        PROX_SIGMA, PROX_PROMINENCE, PROX_REFRACTORY_S,
    )
    _save_prediction(run_id, gate_frames, root, suffix="proximity")
    return _build_row(run_id, gate_frames, annot_path)


def _save_prediction(run_id: str, gate_frames: list, root: str, suffix: str) -> None:
    out = {"run_id": run_id, "fps": FPS, "gates": [
        {"gate_number": i+1, "gate_label": f"Gate {i+1}",
         "frame": f, "position_ms": round(f / FPS * 1000),
         "position_s": round(f / FPS, 3)}
        for i, f in enumerate(gate_frames)
    ]}
    path = os.path.join(root, OUT_DIR, f"{run_id}_{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _build_row(run_id: str, gate_frames: list, annot_path: str) -> dict:
    row = {
        "run_id": run_id, "predicted": len(gate_frames), "gt": None,
        "mean_err": None,
        "w3": None, "fp3": None, "fn3": None,
        "w8": None, "fp8": None, "fn8": None,
        "w15": None, "fp15": None, "fn15": None,
        "w30": None, "fp30": None, "fn30": None,
    }
    if os.path.exists(annot_path):
        with open(annot_path) as f:
            gt = json.load(f)
        gt_frames       = [g["frame"] for g in gt["gates"]]
        errors          = nearest_neighbour_errors(gt_frames, gate_frames)
        row["gt"]       = len(gt_frames)
        row["mean_err"] = np.mean(errors) if errors else float("nan")
        for thresh, key in [(3, "3"), (8, "8"), (15, "15"), (30, "30")]:
            tp, fp, fn      = match_gates(gt_frames, gate_frames, thresh)
            row[f"w{key}"]  = tp
            row[f"fp{key}"] = fp
            row[f"fn{key}"] = fn
    return row


# ── Table printing ────────────────────────────────────────────────────────────

def print_table(results: list, method: str) -> None:
    W = 140
    print(f"\n{'='*W}")
    print(f"  Method: {method}")
    print(f"{'='*W}")
    print(f"{'Run':<50} {'GT':>4} {'Pred':>5} {'Merr':>7}"
          f"  {'——≤3f——':^14}  {'——≤8f——':^14}  {'——≤15f——':^14}  {'——≤30f——':^14}")
    print(f"{'':50} {'':4} {'':5} {'':7}"
          f"  {'TP   FP   FN':^16}  {'TP   FP   FN':^16}  {'TP   FP   FN':^16}  {'TP   FP   FN':^16}")
    print("-" * W)

    for r in results:
        disc = "SL" if "_SL_" in r["run_id"] else "GS"
        name = r["run_id"][:48]
        merr = f"{r['mean_err']/FPS*1000:.0f}ms" if r["mean_err"] is not None else "-"

        def fmt(tp, fp, fn):
            return f"{tp:>3} {fp:>3} {fn:>3}" if tp is not None else f"{'':>11}"

        print(f"[{disc}] {name:<48} {r['gt'] or '-':>4} {r['predicted']:>5} {merr:>7}"
              f"  {fmt(r['w3'],  r['fp3'],  r['fn3'])}"
              f"  {fmt(r['w8'],  r['fp8'],  r['fn8'])}"
              f"  {fmt(r['w15'], r['fp15'], r['fn15'])}"
              f"  {fmt(r['w30'], r['fp30'], r['fn30'])}")

    has_gt = [r for r in results if r["gt"] is not None]
    if has_gt:
        def mean_row(label: str, subset: list) -> None:
            if not subset:
                return
            avg_gt   = np.mean([r["gt"]       for r in subset])
            avg_pred = np.mean([r["predicted"] for r in subset])
            avg_merr = np.mean([r["mean_err"]  for r in subset
                                if r["mean_err"] is not None]) / FPS * 1000
            cols = ""
            for key in ["3", "8", "15", "30"]:
                tp = np.mean([r[f"w{key}"]  for r in subset])
                fp = np.mean([r[f"fp{key}"] for r in subset])
                fn = np.mean([r[f"fn{key}"] for r in subset])
                cols += f"  {tp:>5.1f} {fp:>5.1f} {fn:>5.1f}"
            print(f"{label:<50} {avg_gt:>4.0f} {avg_pred:>5.0f} {avg_merr:>6.0f}ms{cols}")

        sl = [r for r in has_gt if "_SL_" in r["run_id"]]
        gs = [r for r in has_gt if "_SL_" not in r["run_id"]]
        print("-" * W)
        mean_row(f"MEAN SL  (n={len(sl)})", sl)
        mean_row(f"MEAN GS  (n={len(gs)})", gs)
        print("-" * W)
        mean_row(f"MEAN ALL (n={len(has_gt)})", has_gt)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tilt", "proximity", "both"], default="both")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--runs", nargs="+", metavar="RUN_ID",
                        help="Specific run IDs (default: all)")
    parser.add_argument("--pose-model", default=POSE_WEIGHTS)
    parser.add_argument("--gate-model", default=GATE_WEIGHTS)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = "cpu"

    root       = os.path.join(_HERE, "..")
    frames_dir = os.path.join(root, FRAMES_BASE)

    runs = args.runs or sorted(
        r for r in os.listdir(frames_dir)
        if os.path.isdir(os.path.join(frames_dir, r))
    )

    methods = ["tilt", "proximity"] if args.method == "both" else [args.method]

    tilt_model = gate_model = None

    for method in methods:
        if method == "tilt" and tilt_model is None:
            tilt_model = YOLO(os.path.join(root, args.pose_model))
        if method == "proximity" and gate_model is None:
            if tilt_model is None:
                tilt_model = YOLO(os.path.join(root, args.pose_model))
            gate_model = YOLO(os.path.join(root, args.gate_model))

        results = []
        for i, run_id in enumerate(runs):
            print(f"\n[{i+1}/{len(runs)}] {run_id}  [{method}]")
            if method == "tilt":
                results.append(eval_tilt(run_id, tilt_model, root, args.no_cache))
            else:
                results.append(eval_proximity(run_id, tilt_model, gate_model,
                                              root, args.no_cache, device))

        print_table(results, method)


if __name__ == "__main__":
    main()
