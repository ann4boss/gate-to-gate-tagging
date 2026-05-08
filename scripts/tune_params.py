"""
Unified parameter tuning for gate detection methods.

Build caches first (only needed once):
    python scripts/batch_eval.py --method tilt      --runs Run1 Run2 ...
    python scripts/batch_eval.py --method proximity --runs Run1 Run2 ...

Then tune:
    python scripts/tune_params.py --method tilt
    python scripts/tune_params.py --method proximity
    python scripts/tune_params.py --method tilt --sort w8 --top 20
    python scripts/tune_params.py --method proximity --runs Run1 Run2 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "approach2_gate_detection"))
sys.path.insert(0, os.path.join(_HERE, "approach1_gate_detection"))

from gate_detection_lib import FPS, nearest_neighbour_errors  # noqa: E402
from proximity_lib import detect_from_cache                   # noqa: E402

FRAMES_BASE = "data/frames/swsk"
ANNOT_BASE  = "data/annotations/swsk"
TILT_CACHE  = "outputs/cx_cache"
PROX_CACHE  = "outputs/proximity_cache"

# ── Search spaces ─────────────────────────────────────────────────────────────

TILT_SEARCH = {
    "smooth_window":  [3, 5, 7, 9, 11],
    "min_prominence": [5, 8, 10, 15, 20],
    "min_gate_sep":   [10, 12, 15, 18, 20, 25],
    "start_skip":     [0, 30, 60, 100, 150],
    "end_skip":       [0, 30, 60, 100, 150],
}

PROX_SEARCH = {
    "match_dist_px":     [50, 70, 90, 120, 150],
    "max_missing":       [5, 10, 15, 20],
    "prox_sigma":        [1.5, 2.5, 3.5, 5.0],
    "prox_prominence":   [3.0, 5.0, 8.0, 10.0, 15.0],
    "prox_refractory_s": [0.15, 0.25, 0.35, 0.5],
}


# ── Tilt detection (from cx cache) ────────────────────────────────────────────

def _detect_tilt(
    cx: np.ndarray,
    smooth_window: int,
    min_prominence: float,
    min_gate_sep: int,
    start_skip: int,
    end_skip: int,
) -> list[int]:
    finite = np.isfinite(cx)
    if finite.sum() < 10:
        return []

    idx      = np.arange(len(cx))
    filled   = np.interp(idx, idx[finite], cx[finite])
    smoothed = uniform_filter1d(filled, size=smooth_window)

    peaks_max, props_max = find_peaks( smoothed, distance=min_gate_sep, prominence=min_prominence)
    peaks_min, props_min = find_peaks(-smoothed, distance=min_gate_sep, prominence=min_prominence)

    all_idx  = np.concatenate([peaks_max,               peaks_min])
    all_prom = np.concatenate([props_max["prominences"], props_min["prominences"]])
    all_type = np.array([1] * len(peaks_max) + [-1] * len(peaks_min))
    order    = np.argsort(all_idx)
    all_idx  = all_idx[order];  all_prom = all_prom[order];  all_type = all_type[order]

    mask     = (all_idx >= start_skip) & (all_idx < len(cx) - end_skip)
    all_idx  = all_idx[mask];  all_prom = all_prom[mask];  all_type = all_type[mask]

    keep = np.ones(len(all_idx), dtype=bool)
    for i in range(1, len(all_idx)):
        if not keep[i - 1]:
            continue
        if all_idx[i] - all_idx[i - 1] < min_gate_sep:
            if all_prom[i] >= all_prom[i - 1]:
                keep[i - 1] = False
            else:
                keep[i] = False
    all_idx  = all_idx[keep];  all_prom = all_prom[keep];  all_type = all_type[keep]

    result = []
    for i in range(len(all_idx)):
        if not result or all_type[i] != all_type[result[-1]]:
            result.append(i)
        elif all_prom[i] > all_prom[result[-1]]:
            result[-1] = i

    return sorted(all_idx[result].tolist())


# ── Shared sweep + print logic ────────────────────────────────────────────────

def _sweep(data: list, search: dict, predict_fn) -> list:
    keys   = list(search.keys())
    combos = list(product(*[search[k] for k in keys]))
    n      = len(combos)
    total_gt = sum(len(d[-1]) for d in data)
    print(f"Searching {n:,} combinations on {len(data)} runs  |  {total_gt} GT gates\n")

    results  = []
    bar_width = 40
    t_start  = time.perf_counter()

    for ci, combo in enumerate(combos):
        params   = dict(zip(keys, combo))
        all_errs = []
        w3 = w8 = w15 = w30 = 0

        for *payload, gt_frames in data:
            pred   = predict_fn(*payload, **params)
            errors = nearest_neighbour_errors(gt_frames, pred) if pred else [999999] * len(gt_frames)
            all_errs.extend(errors)
            w3  += sum(1 for e in errors if e <=  3)
            w8  += sum(1 for e in errors if e <=  8)
            w15 += sum(1 for e in errors if e <= 15)
            w30 += sum(1 for e in errors if e <= 30)

        results.append({**params,
                        "mean_ms": np.mean(all_errs) / FPS * 1000,
                        "w3": w3, "w8": w8, "w15": w15, "w30": w30,
                        "total_gt": total_gt})

        done    = ci + 1
        filled  = int(bar_width * done / n)
        elapsed = time.perf_counter() - t_start
        eta     = (elapsed / done) * (n - done)
        bar     = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  [{bar}] {done:>4}/{n}  ETA {eta:4.0f}s", end="", flush=True)

    print()
    return results


def _print_results(results: list, sort: str, top: int, method: str) -> None:
    if sort == "w30":
        results.sort(key=lambda r: (-r["w30"],  r["mean_ms"]))
    elif sort == "w15":
        results.sort(key=lambda r: (-r["w15"],  r["mean_ms"]))
    elif sort == "w8":
        results.sort(key=lambda r: (-r["w8"],   r["mean_ms"]))
    elif sort == "w3":
        results.sort(key=lambda r: (-r["w3"],   r["mean_ms"]))
    else:
        results.sort(key=lambda r: ( r["mean_ms"], -r["w15"]))

    g = results[0]["total_gt"]

    if method == "tilt":
        hdr = (f"{'smth':>4}  {'prom':>4}  {'sep':>3}  {'ss':>4}  {'es':>4}"
               f"  {'Mean err':>9}  {'≤3f':>9}  {'≤8f':>9}  {'≤15f':>9}  {'≤30f':>9}")
    else:
        hdr = (f"{'mdist':>5}  {'miss':>4}  {'sig':>4}  {'prom':>5}  {'refr':>5}"
               f"  {'Mean err':>9}  {'≤3f':>9}  {'≤8f':>9}  {'≤15f':>9}  {'≤30f':>9}")

    print(f"\nTop {top} configurations  (sorted by {sort}):\n")
    print(hdr)
    print("─" * len(hdr))

    for r in results[:top]:
        if method == "tilt":
            prefix = (f"{r['smooth_window']:>4}  {r['min_prominence']:>4}  {r['min_gate_sep']:>3}"
                      f"  {r['start_skip']:>4}  {r['end_skip']:>4}")
        else:
            prefix = (f"{r['match_dist_px']:>5}  {r['max_missing']:>4}  {r['prox_sigma']:>4.1f}"
                      f"  {r['prox_prominence']:>5.1f}  {r['prox_refractory_s']:>5.2f}")
        print(f"{prefix}  {r['mean_ms']:>7.0f}ms"
              f"  {r['w3']:>3}/{g} ({r['w3']/g*100:.0f}%)"
              f"  {r['w8']:>3}/{g} ({r['w8']/g*100:.0f}%)"
              f"  {r['w15']:>3}/{g} ({r['w15']/g*100:.0f}%)"
              f"  {r['w30']:>3}/{g} ({r['w30']/g*100:.0f}%)")

    best = results[0]
    print(f"\nBest config:")
    if method == "tilt":
        print(f"  SMOOTH_WINDOW  = {best['smooth_window']}")
        print(f"  MIN_PROMINENCE = {best['min_prominence']}")
        print(f"  MIN_GATE_SEP   = {best['min_gate_sep']}")
        print(f"  START_SKIP     = {best['start_skip']}")
        print(f"  END_SKIP       = {best['end_skip']}")
    else:
        print(f"  MATCH_DIST_PX     = {best['match_dist_px']}")
        print(f"  MAX_MISSING       = {best['max_missing']}")
        print(f"  PROX_SIGMA        = {best['prox_sigma']}")
        print(f"  PROX_PROMINENCE   = {best['prox_prominence']}")
        print(f"  PROX_REFRACTORY_S = {best['prox_refractory_s']}")
    print(f"  → {best['mean_ms']:.0f}ms mean error"
          f"  |  {best['w3']}/{g} ≤3f  |  {best['w8']}/{g} ≤8f"
          f"  |  {best['w15']}/{g} ≤15f  |  {best['w30']}/{g} ≤30f")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tilt", "proximity"], required=True)
    parser.add_argument("--runs", nargs="+", metavar="RUN_ID")
    parser.add_argument("--top",  type=int, default=15)
    parser.add_argument("--sort", choices=["w30", "w15", "w8", "w3", "mean_ms"], default="w15")
    args = parser.parse_args()

    root = os.path.join(_HERE, "..")

    if args.runs:
        run_ids = args.runs
    else:
        frames_dir = os.path.join(root, FRAMES_BASE)
        run_ids = sorted(r for r in os.listdir(frames_dir)
                         if os.path.isdir(os.path.join(frames_dir, r)))

    cache_dir = TILT_CACHE if args.method == "tilt" else PROX_CACHE
    ext       = ".npy"     if args.method == "tilt" else ".json"

    data = []
    for run_id in run_ids:
        cache_path = os.path.join(root, cache_dir, f"{run_id}{ext}")
        annot_path = os.path.join(root, ANNOT_BASE, f"{run_id}.json")
        if not os.path.exists(cache_path):
            print(f"  skip (no cache): {run_id}")
            continue
        if not os.path.exists(annot_path):
            print(f"  skip (no GT):    {run_id}")
            continue
        if args.method == "tilt":
            payload = [np.load(cache_path)]
        else:
            with open(cache_path) as f:
                payload = [json.load(f)]
        gt_frames = [g["frame"] for g in json.load(open(annot_path))["gates"]]
        data.append((*payload, gt_frames))

    if not data:
        print("No runs with both cache and GT found.")
        print(f"Build the cache first:  python scripts/batch_eval.py --method {args.method} --runs <run_ids>")
        sys.exit(1)

    if args.method == "tilt":
        results = _sweep(data, TILT_SEARCH, _detect_tilt)
    else:
        results = _sweep(data, PROX_SEARCH,
                         lambda fd, **kw: detect_from_cache(fd, **kw))

    _print_results(results, args.sort, args.top, args.method)


if __name__ == "__main__":
    main()
