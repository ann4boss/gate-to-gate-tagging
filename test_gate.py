"""
test_gate_detector.py
---------------------
Visual test for GateDetector.

Reads pre-extracted frames, runs GateDetector on each frame, draws
bounding boxes coloured by class and Kalman state, and writes an
annotated MP4.

Colour coding:
    gate_contact  — red box   (the pole the skier hits)
    gate_outer    — blue box  (outer GS pole)
    Dashed/dim    — Kalman coasting (YOLO missed this frame)

Console output:
    - Detection rate per class
    - Average number of poles visible per frame
    - Track count over time

Usage
-----
    python test_gate_detector.py \
        --frames-dir data/frames/swsk/Aerni_Lauf1 \
        --model      runs/detect/gate_poles/weights/best.pt \
        --out        test_gates_out.mp4

    # Quick check
    python test_gate_detector.py \
        --frames-dir data/frames/swsk/Aerni_Lauf1 \
        --model      runs/detect/gate_poles/weights/best.pt \
        --out        test_gates_out.mp4 \
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
from detector.gates import GateDetector


# ── Colours ───────────────────────────────────────────────────────────────────
COLOR_CONTACT = (0,   0, 220)   # red   (BGR)
COLOR_OUTER   = (220, 80,  0)   # blue  (BGR)
COLOR_COAST   = (120,120,120)   # grey  (BGR) — Kalman coasting


def class_color(cls, coasting=False):
    if coasting:
        return COLOR_COAST
    return COLOR_CONTACT if cls == 0 else COLOR_OUTER


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_frame(frame, gates, prev_gate_ids, frame_idx, fps, stats):
    vis = frame.copy()
    h   = vis.shape[0]

    for gid, info in gates.items():
        coasting = gid not in prev_gate_ids  # new this frame = just spawned
        # More accurately: coasting if missed > 0 — but we don't expose that
        # directly so we use a proxy: if the gate was in prev frame, it's tracked
        color = class_color(info["class"])
        x1, y1, x2, y2 = info["bbox"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"{info['label']}  {info['conf']:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        # Centroid dot
        cv2.circle(vis, (info["cx"], info["cy"]), 4, color, -1)

    # HUD
    t_s     = frame_idx / max(fps, 1)
    n_gates = len(gates)
    n_cont  = sum(1 for g in gates.values() if g["class"] == 0)
    n_outer = sum(1 for g in gates.values() if g["class"] == 1)

    cv2.putText(vis, f"frame {frame_idx:05d}  t={t_s:.2f}s",
                (8, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(vis,
                f"contact={n_cont}  outer={n_outer}  total={n_gates}",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    if n_gates == 0:
        cv2.putText(vis, "NO GATES", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return vis


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Visual test for GateDetector")
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--model",      required=True,
                   help="Path to trained gate detector weights")
    p.add_argument("--out",        default="test_gates_out.mp4")
    p.add_argument("--conf",       type=float, default=0.35)
    p.add_argument("--max-frames", type=int,   default=None)
    p.add_argument("--fps",        type=float, default=None)
    p.add_argument("--device",     default="")
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

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading gate detector: {args.model}")
    detector = GateDetector(
        model_path = args.model,
        conf       = args.conf,
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
    n_frames_w_gate = 0
    n_contact_total = 0
    n_outer_total   = 0
    prev_gate_ids   = set()

    for idx, path in enumerate(tqdm(frame_paths, desc="Testing gate detector")):
        frame = cv2.imread(str(path))
        if frame is None:
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
            continue

        gates = detector.update(frame)

        if gates:
            n_frames_w_gate += 1
        n_contact_total += sum(1 for g in gates.values() if g["class"] == 0)
        n_outer_total   += sum(1 for g in gates.values() if g["class"] == 1)

        vis = draw_frame(frame, gates, prev_gate_ids, idx, fps, {})
        writer.write(vis)

        prev_gate_ids = set(gates.keys())

    writer.release()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Total frames          : {n_total}")
    print(f"  Frames with gates     : {n_frames_w_gate}  "
          f"({100*n_frames_w_gate/n_total:.1f}%)")
    print(f"  Avg contact poles/frame: "
          f"{n_contact_total/n_total:.2f}")
    print(f"  Avg outer poles/frame  : "
          f"{n_outer_total/n_total:.2f}")
    print(f"{'='*55}")
    print(f"\n  Output video : {args.out}\n")

    # ── Guidance ──────────────────────────────────────────────────────────────
    if n_frames_w_gate / n_total < 0.50:
        print("[WARN] Gate detection rate is low (<50%). Try:")
        print("       --conf 0.25  to lower the threshold")
        print("       Check that your weights path is correct")
    elif n_frames_w_gate / n_total >= 0.70:
        print("[OK]  Gate detector looks reasonable.")
        print("      Next step: test full pipeline with skier + pose + gates")


if __name__ == "__main__":
    main()