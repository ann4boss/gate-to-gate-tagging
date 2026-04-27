"""
eval/evaluate.py
================
Compare auto-tagged gate JSON against the manually tagged ground truth.

Metrics reported
----------------
  - Gate count accuracy   (predicted vs expected)
  - Per-gate timing error (|predicted_ms - gt_ms|)
  - Mean Absolute Error   (MAE) in ms
  - Root Mean Square Error (RMSE) in ms
  - Gate-level precision / recall / F1
    (a prediction is a TP if it matches a GT gate within --tolerance-ms)

Usage
-----
    python eval/evaluate.py \
        --gt   manually_tagged/run01.json \
        --pred auto_tagged/run01.json     \
        --tolerance-ms 200
"""

import argparse
import json
import math
from typing import List, Optional, Tuple


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def evaluate(
    gt_path:      str,
    pred_path:    str,
    tolerance_ms: float = 200.0,
    verbose:      bool  = True,
) -> dict:
    gt   = load(gt_path)
    pred = load(pred_path)

    gt_gates   = gt["gates"]
    pred_gates = pred["gates"]

    gt_times   = [g["position_ms"]   for g in gt_gates]
    pred_times = [g["position_ms"] for g in pred_gates]

    # ── Matching (greedy nearest within tolerance) ────────────────────────────
    matched_gt:   List[bool] = [False] * len(gt_times)
    matched_pred: List[bool] = [False] * len(pred_times)
    errors_ms:    List[float] = []

    for pi, pt in enumerate(pred_times):
        best_err, best_gi = tolerance_ms, -1
        for gi, gt_t in enumerate(gt_times):
            if matched_gt[gi]:
                continue
            err = abs(pt - gt_t)
            if err < best_err:
                best_err, best_gi = err, gi
        if best_gi >= 0:
            matched_pred[pi] = True
            matched_gt[best_gi] = True
            errors_ms.append(best_err)

    TP = sum(matched_pred)
    FP = len(pred_gates) - TP
    FN = len(gt_gates)   - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    mae  = sum(errors_ms) / len(errors_ms) if errors_ms else float("nan")
    rmse = math.sqrt(sum(e**2 for e in errors_ms) / len(errors_ms)) \
           if errors_ms else float("nan")

    results = {
        "run_id_gt":        gt.get("run_id"),
        "run_id_pred":      pred.get("run_id"),
        "n_gates_gt":       len(gt_gates),
        "n_gates_pred":     len(pred_gates),
        "tolerance_ms":     tolerance_ms,
        "TP": TP, "FP": FP, "FN": FN,
        "precision":        round(precision, 4),
        "recall":           round(recall,    4),
        "f1":               round(f1,        4),
        "mae_ms":           round(mae,  1),
        "rmse_ms":          round(rmse, 1),
        "per_gate_errors_ms": [round(e, 1) for e in errors_ms],
    }

    if verbose:
        _print_report(results, gt_gates, pred_gates)

    return results


def _print_report(r: dict, gt_gates: list, pred_gates: list):
    print("\n" + "=" * 60)
    print(f"  Run GT  : {r['run_id_gt']}")
    print(f"  Run PRED: {r['run_id_pred']}")
    print(f"  Tolerance: ±{r['tolerance_ms']} ms")
    print("=" * 60)
    print(f"  Gates GT / Pred : {r['n_gates_gt']} / {r['n_gates_pred']}")
    print(f"  TP / FP / FN    : {r['TP']} / {r['FP']} / {r['FN']}")
    print(f"  Precision       : {r['precision']:.1%}")
    print(f"  Recall          : {r['recall']:.1%}")
    print(f"  F1              : {r['f1']:.1%}")
    print(f"  MAE             : {r['mae_ms']:.1f} ms")
    print(f"  RMSE            : {r['rmse_ms']:.1f} ms")

    # Per-gate detail table
    print("\n  Gate-by-gate comparison (GT gate_number, GT ms, Pred ms, Δms):")
    pred_times = [g["position_ms"] for g in pred_gates]
    used_pred  = set()
    tol = r["tolerance_ms"]
    for g in gt_gates:
        gt_ms = g["position_ms"]
        best_err, best_pi = tol, -1
        for pi, pt in enumerate(pred_times):
            if pi in used_pred:
                continue
            err = abs(pt - gt_ms)
            if err < best_err:
                best_err, best_pi = err, pi
        if best_pi >= 0:
            used_pred.add(best_pi)
            pred_ms = pred_times[best_pi]
            tag = "✓"
        else:
            pred_ms = None
            tag = "✗ MISS"
        pred_str = f"{pred_ms:.0f}" if pred_ms is not None else "  —  "
        err_str  = f"{best_err:.0f}" if best_pi >= 0 else "  —  "
        print(f"    Gate {g['gate_number']:>3}  {gt_ms:>8.0f} ms  "
              f"→  {pred_str:>8} ms   Δ={err_str:>6} ms  {tag}")
    print("=" * 60 + "\n")


def _build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate auto-tagged gates against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gt",           required=True, help="Ground truth JSON")
    p.add_argument("--pred",         required=True, help="Predicted JSON")
    p.add_argument("--tolerance-ms", type=float, default=200.0,
                   help="Match window in milliseconds")
    p.add_argument("--out",          default=None,
                   help="Optional path to write results JSON")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    results = evaluate(args.gt, args.pred, args.tolerance_ms)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.out}")