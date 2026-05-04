"""
07_evaluate.py
--------------
Evaluate gate passage predictions against ground truth.

Modes:
  Single run:
      python 07_evaluate.py --gt  data/annotations/swsk/<run_id>.json \
                            --pred outputs/predictions/<run_id>_tilt.json

  Batch (all runs in outputs/predictions/):
      python 07_evaluate.py --batch --method tilt
      python 07_evaluate.py --batch --method proximity
      python 07_evaluate.py --batch --method both

Metrics:
  - Precision, Recall, F1  (within --tolerance-ms window)
  - MAE, RMSE in milliseconds
  - Per-discipline breakdown (SL / GS detected from run_id suffix)

A prediction matches a GT gate if |pred_ms - gt_ms| ≤ tolerance_ms.
Matching is greedy (nearest first), each GT gate matched at most once.
"""

import argparse
import json
import math
import os
from pathlib import Path


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def match_gates(gt_times: list, pred_times: list, tol: float) -> tuple:
    """Greedy nearest-neighbour matching. Returns (errors_ms, TP, FP, FN)."""
    matched_gt   = [False] * len(gt_times)
    matched_pred = [False] * len(pred_times)
    errors = []

    for pi, pt in enumerate(pred_times):
        best_err, best_gi = tol, -1
        for gi, gt_t in enumerate(gt_times):
            if matched_gt[gi]:
                continue
            err = abs(pt - gt_t)
            if err < best_err:
                best_err, best_gi = err, gi
        if best_gi >= 0:
            matched_pred[pi] = True
            matched_gt[best_gi] = True
            errors.append(best_err)

    TP = sum(matched_pred)
    FP = len(pred_times) - TP
    FN = len(gt_times)   - TP
    return errors, TP, FP, FN


def evaluate_pair(gt_path: str, pred_path: str, tolerance_ms: float = 200.0) -> dict:
    gt   = load(gt_path)
    pred = load(pred_path)

    gt_times   = [g["position_ms"] for g in gt["gates"]]
    pred_times = [g["position_ms"] for g in pred["gates"]]

    errors, TP, FP, FN = match_gates(gt_times, pred_times, tolerance_ms)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    mae       = sum(errors) / len(errors) if errors else float("nan")
    rmse      = math.sqrt(sum(e**2 for e in errors) / len(errors)) if errors else float("nan")

    return {
        "run_id":      gt.get("run_id", Path(gt_path).stem),
        "n_gt":        len(gt_times),
        "n_pred":      len(pred_times),
        "tolerance_ms": tolerance_ms,
        "TP": TP, "FP": FP, "FN": FN,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "mae_ms":    round(mae,       1),
        "rmse_ms":   round(rmse,      1),
        "errors_ms": [round(e, 1) for e in errors],
    }


def print_single(r: dict):
    print(f"\n{'='*60}")
    print(f"  Run     : {r['run_id']}")
    print(f"  Tolerance: ±{r['tolerance_ms']} ms")
    print(f"{'='*60}")
    print(f"  Gates GT / Pred : {r['n_gt']} / {r['n_pred']}")
    print(f"  TP / FP / FN    : {r['TP']} / {r['FP']} / {r['FN']}")
    print(f"  Precision       : {r['precision']:.1%}")
    print(f"  Recall          : {r['recall']:.1%}")
    print(f"  F1              : {r['f1']:.1%}")
    print(f"  MAE             : {r['mae_ms']:.1f} ms")
    print(f"  RMSE            : {r['rmse_ms']:.1f} ms")
    print(f"{'='*60}\n")


def print_batch(results: list, method: str, tol: float):
    # Separate SL / GS by run_id convention
    sl = [r for r in results if "_SL_" in r["run_id"]]
    gs = [r for r in results if "_GS_" in r["run_id"]]
    other = [r for r in results if "_SL_" not in r["run_id"] and "_GS_" not in r["run_id"]]

    def _aggregate(group: list, label: str):
        if not group:
            return
        tp  = sum(r["TP"]  for r in group)
        fp  = sum(r["FP"]  for r in group)
        fn  = sum(r["FN"]  for r in group)
        p   = tp / (tp + fp) if (tp+fp) > 0 else 0
        rec = tp / (tp + fn) if (tp+fn) > 0 else 0
        f1  = 2*p*rec/(p+rec) if (p+rec) > 0 else 0
        all_err = [e for r in group for e in r["errors_ms"]]
        mae  = sum(all_err)/len(all_err) if all_err else float("nan")
        rmse = math.sqrt(sum(e**2 for e in all_err)/len(all_err)) if all_err else float("nan")
        print(f"  {label:>6}  n={len(group):>3}  P={p:.1%}  R={rec:.1%}  F1={f1:.1%}  "
              f"MAE={mae:.0f}ms  RMSE={rmse:.0f}ms")

    print(f"\n{'='*70}")
    print(f"  Batch Evaluation — method={method}  tolerance=±{tol}ms  n={len(results)}")
    print(f"{'='*70}")
    _aggregate(results, "ALL")
    _aggregate(sl,  "SL")
    _aggregate(gs,  "GS")
    if other:
        _aggregate(other, "OTHER")

    print(f"\n  {'Run':<50} {'GT':>4} {'Pred':>5} {'F1':>6} {'MAE':>8}")
    print(f"  {'-'*74}")
    for r in sorted(results, key=lambda x: x["run_id"]):
        disc = "SL" if "_SL_" in r["run_id"] else "GS" if "_GS_" in r["run_id"] else "??"
        name = r["run_id"][:48]
        print(f"  [{disc}] {name:<48} {r['n_gt']:>4} {r['n_pred']:>5} "
              f"{r['f1']:>5.1%} {r['mae_ms']:>7.0f}ms")
    print(f"{'='*70}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt",           default=None, help="Ground truth JSON (single mode)")
    p.add_argument("--pred",         default=None, help="Prediction JSON (single mode)")
    p.add_argument("--batch",        action="store_true", help="Evaluate all runs in OUT_DIR")
    p.add_argument("--method",       default="tilt", choices=["proximity", "tilt", "both"])
    p.add_argument("--tolerance-ms", type=float, default=200.0)
    p.add_argument("--gt-dir",       default="data/annotations/swsk")
    p.add_argument("--pred-dir",     default="outputs/predictions")
    p.add_argument("--out",          default=None, help="Write results JSON")
    args = p.parse_args()

    tol = args.tolerance_ms

    if args.batch:
        methods = ["proximity", "tilt"] if args.method == "both" else [args.method]
        for method in methods:
            pred_files = [
                f for f in os.listdir(args.pred_dir)
                if f.endswith(f"_{method}.json")
            ] if os.path.isdir(args.pred_dir) else []

            results = []
            for pf in pred_files:
                run_id   = pf.replace(f"_{method}.json", "")
                gt_path  = os.path.join(args.gt_dir,   f"{run_id}.json")
                pred_path = os.path.join(args.pred_dir, pf)
                if not os.path.exists(gt_path):
                    print(f"  [SKIP] No GT for {run_id}")
                    continue
                results.append(evaluate_pair(gt_path, pred_path, tol))

            print_batch(results, method, tol)
            if args.out:
                out_path = args.out.replace(".json", f"_{method}.json")
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  Results written → {out_path}")

    else:
        if not args.gt or not args.pred:
            p.error("Provide --gt and --pred for single-run evaluation, or use --batch")
        r = evaluate_pair(args.gt, args.pred, tol)
        print_single(r)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(r, f, indent=2)
            print(f"Results written → {args.out}")


if __name__ == "__main__":
    main()