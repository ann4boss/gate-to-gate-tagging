"""
count_gates.py
--------------
Reads all prediction JSONs and matching ground-truth annotations,
then prints per-run and aggregate gate counts.

Usage
-----
    python scripts/count_gates.py
    python scripts/count_gates.py --pred-dir outputs/predictions \
                                  --gt-dir   data/annotations/swsk
"""

import argparse
import json
from pathlib import Path


def load_gate_count(json_path: Path) -> int:
    with open(json_path) as f:
        return len(json.load(f).get("gates", []))


def main(pred_dir: str, gt_dir: str):
    pred_dir = Path(pred_dir)
    gt_dir   = Path(gt_dir)

    pred_files = sorted(pred_dir.glob("*.json"))
    if not pred_files:
        print(f"No prediction JSONs found in {pred_dir}")
        return

    col = "{:<55} {:>4}  {:>4}  {:>5}"
    header = col.format("Run", "Pred", "GT", "Diff")
    print(header)
    print("-" * len(header))

    total_pred = 0
    total_gt   = 0
    matched    = 0
    missing_gt = []

    for pred_path in pred_files:
        run_id = pred_path.stem
        gt_path = gt_dir / f"{run_id}.json"

        n_pred = load_gate_count(pred_path)
        total_pred += n_pred

        if gt_path.exists():
            n_gt = load_gate_count(gt_path)
            total_gt += n_gt
            matched  += 1
            diff = n_pred - n_gt
            print(col.format(run_id[:55], n_pred, n_gt, f"{diff:+d}"))
        else:
            missing_gt.append(run_id)
            print(col.format(run_id[:55], n_pred, "—", 0))

    print("-" * len(header))
    print(col.format(f"TOTAL  ({matched} runs with GT)", total_pred, total_gt,
                     f"{total_pred - total_gt:+d}"))

    if missing_gt:
        print(f"\nNo ground-truth found for {len(missing_gt)} run(s):")
        for r in missing_gt:
            print(f"  {r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", default="outputs/predictions")
    p.add_argument("--gt-dir",   default="data/annotations/swsk")
    args = p.parse_args()
    main(args.pred_dir, args.gt_dir)
