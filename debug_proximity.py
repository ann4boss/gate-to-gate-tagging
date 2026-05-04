"""
debug_proximity.py
------------------
Diagnostic tool for gate passage detection.

Runs the full skier + pose + gate pipeline on a run and plots the
raw distance signals per gate tracker ID so you can see exactly why
passages are or aren't being detected.

Outputs:
    - distance_signals.png  — one subplot per gate tracker ID
    - debug_stats.json      — raw numbers for inspection

Usage
-----
    python debug_proximity.py \
        --frames-dir data/frames/swsk/Aerni_Lauf1 \
        --gate-model runs/detect/gate_poles/weights/best.pt \
        --det-model  runs/detect/skier_epfl/weights/best.pt \
        --pose-model runs/pose/skier_pose_epfl/weights/best.pt \
        --out-dir    debug_proximity

    # Quick check on first 200 frames
    python debug_proximity.py \
        --frames-dir data/frames/swsk/Aerni_Lauf1 \
        --gate-model runs/detect/gate_poles/weights/best.pt \
        --max-frames 200 \
        --out-dir    debug_proximity
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from detector.skier import SkierDetector
from detector.pose  import PoseEstimator
from detector.gates import GateDetector


# ── Keypoint helpers (same as gate_tagger.py) ─────────────────────────────────
_KP_LEFT_WRIST      = 9
_KP_RIGHT_WRIST     = 10
_KP_LEFT_ANKLE      = 15
_KP_RIGHT_ANKLE     = 16
_KP_LEFT_POLE_GRIP  = 19
_KP_RIGHT_POLE_GRIP = 22
def _closest_hand_to_gate(kpts: list, gate_cx: float, gate_cy: float):
    # Return (x, y) of the hand keypoint closest to the given gate centroid.
    # Priority: pole grips (kpts 19, 22) from fine-tuned model first,
    # then wrists (kpts 9, 10) as fallback.
    # Picks left or right based on which is closer to the gate pole.
    if not kpts:
        return None

    def _dist(x, y):
        return ((x - gate_cx) ** 2 + (y - gate_cy) ** 2) ** 0.5

    # Try pole grips first (24-kpt fine-tuned model only)
    grip_candidates = []
    for idx in (_KP_LEFT_POLE_GRIP, _KP_RIGHT_POLE_GRIP):
        if idx < len(kpts):
            x, y, c = kpts[idx]
            if c > 0:
                grip_candidates.append((x, y, _dist(x, y)))
    if grip_candidates:
        best = min(grip_candidates, key=lambda t: t[2])
        return (best[0], best[1])

    # Fall back to wrists only
    wrist_candidates = []
    for idx in (_KP_LEFT_WRIST, _KP_RIGHT_WRIST):
        if idx < len(kpts):
            x, y, c = kpts[idx]
            if c > 0:
                wrist_candidates.append((x, y, _dist(x, y)))
    if wrist_candidates:
        best = min(wrist_candidates, key=lambda t: t[2])
        return (best[0], best[1])

    return None



def _best_hand_position(kpts):
    if not kpts:
        return None
    for idx in (_KP_LEFT_POLE_GRIP, _KP_RIGHT_POLE_GRIP):
        if idx < len(kpts):
            x, y, c = kpts[idx]
            if c > 0:
                return (x, y)
    for idx in (_KP_LEFT_WRIST, _KP_RIGHT_WRIST):
        if idx < len(kpts):
            x, y, c = kpts[idx]
            if c > 0:
                return (x, y)
    return None

def _ankle_midpoint(kpts: list):
    # Return midpoint of left and right ankle, or None if not both visible.
    # Ankles are large joints, consistently detected, and provide a stable
    # vertical position signal useful as a secondary distance metric.
    if len(kpts) <= _KP_RIGHT_ANKLE:
        return None
    _, _, cl = kpts[_KP_LEFT_ANKLE]
    _, _, cr = kpts[_KP_RIGHT_ANKLE]
    if cl <= 0 or cr <= 0:
        return None
    xl, yl, _ = kpts[_KP_LEFT_ANKLE]
    xr, yr, _ = kpts[_KP_RIGHT_ANKLE]
    return ((xl + xr) / 2.0, (yl + yr) / 2.0)

def _combined_distance(kpts: list, gate_cx: float, gate_cy: float):
    # Compute a combined distance signal from hand + ankle to the gate pole.
    #
    # We use a weighted combination:
    #   - Hand/grip distance (weight 0.7): primary signal — directly measures
    #     pole contact. Noisier because wrists/grips are small and sometimes
    #     occluded, but strongly indicates the passage moment.
    #   - Ankle midpoint distance (weight 0.3): secondary signal — ankles are
    #     large, stable, and consistently detected. They reach a minimum as the
    #     skier's feet pass the pole, slightly after hand contact.
    #
    # If only one signal is available, use it alone (weight = 1.0).
    # Returns None if neither signal is available.
 
    hand   = _closest_hand_to_gate(kpts, gate_cx, gate_cy)
    ankle  = _ankle_midpoint(kpts)
 
    def _dist(pos):
        if pos is None:
            return None
        return ((pos[0] - gate_cx) ** 2 + (pos[1] - gate_cy) ** 2) ** 0.5
 
    d_hand  = _dist(hand)
    d_ankle = _dist(ankle)
 
    if d_hand is not None and d_ankle is not None:
        return 0.7 * d_hand + 0.3 * d_ankle
    elif d_hand is not None:
        return d_hand
    elif d_ankle is not None:
        return d_ankle
    return None
 
 
def _hip_midpoint(kpts: list):
    """Return midpoint of hips (kpts 11+12), or None."""
    if len(kpts) < 13:
        return None
    _, _, c11 = kpts[11]
    _, _, c12 = kpts[12]
    if c11 <= 0 or c12 <= 0:
        return None
    x = (kpts[11][0] + kpts[12][0]) / 2.0
    y = (kpts[11][1] + kpts[12][1]) / 2.0
    return x, y



def main():
    p = argparse.ArgumentParser(description="Debug gate passage distance signals")
    p.add_argument("--frames-dir",  required=True)
    p.add_argument("--gate-model",  required=True)
    p.add_argument("--det-model",   default="yolo26s.pt")
    p.add_argument("--pose-model",  default="yolo26s-pose.pt")
    p.add_argument("--det-conf",    type=float, default=0.40)
    p.add_argument("--pose-conf",   type=float, default=0.35)
    p.add_argument("--gate-conf",   type=float, default=0.35)
    p.add_argument("--max-frames",  type=int,   default=None)
    p.add_argument("--fps",         type=float, default=None)
    p.add_argument("--sigma",       type=float, default=2.5)
    p.add_argument("--min-prom",    type=float, default=15.0)
    p.add_argument("--out-dir",     default="debug_proximity")
    p.add_argument("--device",      default="")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = Path(args.frames_dir)
    IMG_EXTS   = {".jpg", ".jpeg", ".png"}
    frame_paths = sorted(
        f for f in frames_dir.iterdir() if f.suffix.lower() in IMG_EXTS
    )

     # FPS
    fps = args.fps
    meta_path = frames_dir / "meta.json"
    if fps is None and meta_path.exists():
        with open(meta_path) as f:
            fps = float(json.load(f).get("fps", 25))
    elif fps is None:
        fps = 25.0
    #skip first 5 seconds and last 5 seconds
    skip_frames = int(fps * 5)
    frame_paths = frame_paths[skip_frames:-skip_frames]

    if args.max_frames:
        frame_paths = frame_paths[:args.max_frames]

   
    print(f"[INFO] FPS: {fps}  Frames: {len(frame_paths)}")

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"[INFO] Loading models...")
    skier_det = SkierDetector(args.det_model,  conf=args.det_conf,  device=args.device)
    pose_est  = PoseEstimator(args.pose_model, conf=args.pose_conf, device=args.device)
    gate_det  = GateDetector(args.gate_model,  conf=args.gate_conf, device=args.device)

    # ── Collect per-frame data ────────────────────────────────────────────────
    # {gate_id: [(frame_idx, distance_px)]}
    gate_distances = defaultdict(list)
    n_frames_with_hand   = 0
    n_frames_with_gates  = 0
    n_frames_both        = 0
    min_distances_seen   = []

    for idx, path in enumerate(tqdm(frame_paths, desc="Running pipeline")):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        skier_bbox = skier_det.detect(frame)
        kpts       = pose_est.estimate(frame, skier_bbox)
        gates      = gate_det.update(frame)

        hand_pos   = _best_hand_position(kpts)
        contact_gates = {
            gid: info for gid, info in gates.items()
            if info["class"] == 0
        }

        # Has hand: either pole grip or wrist visible
        has_hand  = _closest_hand_to_gate(kpts, 0, 0) is not None or _ankle_midpoint(kpts) is not None
        has_gates = len(contact_gates) > 0

        if has_hand:
            n_frames_with_hand += 1
        if has_gates:
            n_frames_with_gates += 1
        if has_hand and has_gates:
            n_frames_both += 1

        # Compute combined hand+ankle distance to each contact pole
        if has_gates:
            for gid, info in contact_gates.items():
                gcx, gcy = float(info["cx"]), float(info["cy"])
                dist = _combined_distance(kpts, gcx, gcy)
                if dist is not None:
                    gate_distances[gid].append((idx, dist))
                    min_distances_seen.append(dist)

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_total = len(frame_paths)
    print(f"\n{'='*60}")
    print(f"  Total frames          : {n_total}")
    print(f"  Frames with hand kpt  : {n_frames_with_hand} ({100*n_frames_with_hand/n_total:.1f}%)")
    print(f"  Frames with gates     : {n_frames_with_gates} ({100*n_frames_with_gates/n_total:.1f}%)")
    print(f"  Frames with BOTH      : {n_frames_both} ({100*n_frames_both/n_total:.1f}%)")
    print(f"\n  Gate tracker IDs seen : {sorted(gate_distances.keys())}")
    print(f"  Trackers with data    : {len(gate_distances)}")

    if min_distances_seen:
        print(f"\n  Distance stats (hand → nearest contact pole):")
        print(f"    Min  : {min(min_distances_seen):.1f} px")
        print(f"    Mean : {np.mean(min_distances_seen):.1f} px")
        print(f"    Max  : {max(min_distances_seen):.1f} px")
        print(f"\n  Frames where distance < 20px  (pole touch): "
              f"{sum(1 for d in min_distances_seen if d < 20)}")
        print(f"  Frames where distance < 50px  : "
              f"{sum(1 for d in min_distances_seen if d < 50)}")
        print(f"  Frames where distance < 100px : "
              f"{sum(1 for d in min_distances_seen if d < 100)}")
    print(f"{'='*60}")

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    print(f"\n── DIAGNOSIS ──")
    if n_frames_with_hand / n_total < 0.5:
        print(f"⚠️  Hand/grip keypoints only visible in {100*n_frames_with_hand/n_total:.0f}% of frames.")
        print(f"   → Pose model is not finding wrists/pole grips reliably.")
        print(f"   → Try: lower --pose-conf, or wait for fine-tuned 24-kpt model.")

    if n_frames_with_gates / n_total < 0.5:
        print(f"⚠️  Gate poles only visible in {100*n_frames_with_gates/n_total:.0f}% of frames.")
        print(f"   → Gate detector is not finding poles reliably.")
        print(f"   → Try: lower --gate-conf, check training data quality.")

    if len(gate_distances) > 20:
        print(f"⚠️  {len(gate_distances)} gate tracker IDs — many tracker resets.")
        print(f"   → Each reset shortens the distance signal, making minima harder to find.")
        print(f"   → This is the most likely cause of 0 passages detected.")
        print(f"   → Fix: increase max_missing in GateDetector (currently 10 frames).")

    if min_distances_seen and min(min_distances_seen) > 50:
        print(f"⚠️  Minimum distance is {min(min_distances_seen):.0f}px — hand never gets close to pole.")
        print(f"   → The proximity signal will never produce a strong minimum.")
        print(f"   → Likely cause: wrong keypoint being used (wrist vs pole grip).")
        print(f"   → Or: gate_contact poles being detected in wrong position.")

    if min_distances_seen and min(min_distances_seen) < 20:
        print(f"✅  Distance does reach <20px — pole contact signal exists.")
        print(f"   → Problem is likely in event detection tuning.")
        print(f"   → Try: --min-prom 5.0 --sigma 1.5")

    if n_frames_both / n_total < 0.3:
        print(f"⚠️  Hand AND gates only co-occur in {100*n_frames_both/n_total:.0f}% of frames.")
        print(f"   → Not enough overlap to compute a reliable distance signal.")

    # ── Plot distance signals ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        if not gate_distances:
            print("\n[WARN] No distance data to plot.")
            return

        n_plots = len(gate_distances)
        fig, axes = plt.subplots(
            n_plots, 1,
            figsize=(14, max(3, 2.5 * n_plots)),
            sharex=True
        )
        if n_plots == 1:
            axes = [axes]

        for ax, (gid, data) in zip(axes, sorted(gate_distances.items())):
            frame_idxs = [d[0] for d in data]
            distances  = [d[1] for d in data]

            # Raw signal
            ax.plot(frame_idxs, distances,
                    color="steelblue", alpha=0.4, linewidth=0.8, label="raw")

            # Smoothed signal
            if len(distances) >= 5:
                smoothed = gaussian_filter1d(distances, sigma=args.sigma)
                ax.plot(frame_idxs, smoothed,
                        color="steelblue", linewidth=1.5, label=f"smoothed (σ={args.sigma})")

                # Detected minima
                peaks, props = find_peaks(
                    [-d for d in smoothed],
                    prominence=args.min_prom,
                    distance=int(fps * 0.35),
                )
                for pk in peaks:
                    ax.axvline(frame_idxs[pk], color="red",
                               linestyle="--", alpha=0.7, linewidth=1)
                    ax.scatter([frame_idxs[pk]], [smoothed[pk]],
                               color="red", zorder=5, s=40)

            ax.axhline(20,  color="green",  linestyle=":", alpha=0.6, label="20px (touch)")
            ax.axhline(50,  color="orange", linestyle=":", alpha=0.6, label="50px")
            ax.axhline(args.min_prom, color="red", linestyle=":",
                       alpha=0.4, label=f"min_prom={args.min_prom}px")

            n_pts    = len(data)
            min_dist = min(distances) if distances else float("nan")
            n_peaks  = len(peaks) if len(distances) >= 5 else 0
            ax.set_title(
                f"{gid}  |  {n_pts} frames  |  min dist={min_dist:.0f}px  |  "
                f"passages detected={n_peaks}",
                fontsize=9
            )
            ax.set_ylabel("dist (px)", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("frame index")
        plt.suptitle(
            f"Gate distance signals — {frames_dir.name}\n"
            f"Red dashed lines = detected passages  |  "
            f"Green = 20px touch threshold",
            fontsize=10
        )
        plt.tight_layout()

        plot_path = out_dir / "distance_signals.png"
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n[INFO] Distance signal plot → {plot_path}")

    except ImportError:
        print("\n[WARN] matplotlib/scipy not installed — skipping plot.")
        print("       pip install matplotlib scipy")

    # ── Save raw stats ────────────────────────────────────────────────────────
    stats = {
        "n_total":              n_total,
        "n_frames_with_hand":   n_frames_with_hand,
        "n_frames_with_gates":  n_frames_with_gates,
        "n_frames_both":        n_frames_both,
        "gate_tracker_ids":     sorted(gate_distances.keys()),
        "n_trackers":           len(gate_distances),
        "min_distance_px":      float(min(min_distances_seen)) if min_distances_seen else None,
        "mean_distance_px":     float(np.mean(min_distances_seen)) if min_distances_seen else None,
    }
    stats_path = out_dir / "debug_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Debug stats → {stats_path}")


if __name__ == "__main__":
    main()