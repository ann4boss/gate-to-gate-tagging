"""
gate_tagger.py
==============
Main CLI entry point for automatic ski gate tagging.

Supports two input modes:
  1. --video PATH           Process a video file directly
  2. --frames-dir PATH      Process a folder of pre-extracted frames
                            (sorted alphabetically; filenames must encode
                             the original frame index or be sequential)

Usage examples
--------------
# From a video file:
    python gate_tagger.py --video race.mp4 --run-id ATHLETE_01 --out out.json

# From pre-extracted frames (FPS must be supplied explicitly):
    python gate_tagger.py --frames-dir ./frames/run01 --fps 30 \
                          --run-id ATHLETE_01 --out out.json

# Use a fine-tuned pose model and a different YOLO26 size:
    python gate_tagger.py --video race.mp4 \
        --pose-model runs/pose/exp/weights/best.pt \
        --det-model  yolo26m.pt \
        --run-id ATHLETE_01
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ── Project imports ───────────────────────────────────────────────────────────
from detector.skier    import SkierDetector
from detector.gates    import GateDetector
from detector.pose     import PoseEstimator
from logic.proximity   import compute_distances
from logic.events      import detect_gate_passages
from output.formatter  import build_output


# ── Frame sources ─────────────────────────────────────────────────────────────

def _frames_from_video(video_path: str):
    """Generator: yields (frame_index, BGR ndarray, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx   = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame, fps, total
        idx += 1
    cap.release()


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def _frames_from_dir(frames_dir: str, fps: float):
    """Generator: yields (frame_index, BGR ndarray, fps)."""
    paths = sorted(
        p for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in _IMG_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"No image files found in {frames_dir}")
    total = len(paths)
    for idx, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[WARN] Could not read {path}, skipping.", file=sys.stderr)
            continue
        yield idx, frame, fps, total


# ── Core processing ───────────────────────────────────────────────────────────

def process(
    frame_source,
    run_id: str,
    det_model:  str = "yolo26s.pt",
    pose_model: str = "yolo26s-pose.pt",
    det_conf:   float = 0.40,
    pose_conf:  float = 0.35,
    refractory_s: float = 0.35,
    sigma: float = 2.5,
    min_prominence: float = 15.0,
    device: str = "",
    debug_dir: Optional[str] = None,
) -> dict:
    skier_det = SkierDetector(det_model,  conf=det_conf,  device=device)
    gate_det  = GateDetector()
    pose_est  = PoseEstimator(pose_model, conf=pose_conf, device=device)

    frame_data = []   # [(frame_idx, keypoints, gate_centroids)]
    fps_value  = None

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    with tqdm(desc="Processing frames", unit="frame") as pbar:
        for frame_idx, frame, fps, total in frame_source:
            if fps_value is None:
                fps_value = fps
                pbar.total = total

            bbox   = skier_det.detect(frame)
            gates  = gate_det.update(frame)
            kpts   = pose_est.estimate(frame, bbox)

            frame_data.append((frame_idx, kpts, gates))

            if debug_dir:
                _draw_debug(frame, bbox, kpts, gates,
                            frame_idx, fps, debug_dir)
            pbar.update(1)

    if fps_value is None:
        raise RuntimeError("No frames were processed.")

    print(f"\n[INFO] Processed {len(frame_data)} frames at {fps_value:.2f} fps")
    print(f"[INFO] Unique gate IDs detected: "
          f"{sorted({gid for _, _, g in frame_data for gid in g})}")

    signals  = compute_distances(frame_data)
    passages = detect_gate_passages(
        signals,
        fps=fps_value,
        refractory_s=refractory_s,
        sigma=sigma,
        min_prominence=min_prominence,
    )
    print(f"[INFO] Gate passages detected: {len(passages)}")

    return build_output(run_id=run_id, fps=fps_value, passages=passages)


# ── Debug visualisation ───────────────────────────────────────────────────────

def _draw_debug(frame, bbox, kpts, gates, frame_idx, fps, debug_dir):
    vis = frame.copy()

    # Skier bbox
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Keypoints
    for x, y, c in kpts:
        if c >= 0.20:
            cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 0), -1)

    # Gate centroids
    for gid, (cx, cy) in gates.items():
        color = (0, 0, 255) if gid.startswith("R") else (255, 0, 0)
        cv2.circle(vis, (cx, cy), 8, color, -1)
        cv2.putText(vis, gid, (cx + 10, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Frame info
    t_ms = frame_idx / fps * 1000
    cv2.putText(vis, f"frame={frame_idx}  t={t_ms:.0f}ms",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = os.path.join(debug_dir, f"{frame_idx:06d}.jpg")
    cv2.imwrite(out_path, vis)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automatic ski gate tagger using YOLO26 + Kalman tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",      metavar="PATH", help="Input video file")
    src.add_argument("--frames-dir", metavar="PATH",
                     help="Directory of pre-extracted frames (sorted)")

    p.add_argument("--fps", type=float, default=30.0,
                   help="Frame rate (required when using --frames-dir)")
    p.add_argument("--run-id", default="RUN_001",
                   help="Identifier written into the output JSON")
    p.add_argument("--out", default="gates.json",
                   help="Output JSON file path")

    p.add_argument("--det-model",  default="yolo26s.pt",
                   help="YOLO26 detection weights (person bbox)")
    p.add_argument("--pose-model", default="yolo26s-pose.pt",
                   help="YOLO26 pose weights (keypoints)")
    p.add_argument("--det-conf",   type=float, default=0.40)
    p.add_argument("--pose-conf",  type=float, default=0.35)
    p.add_argument("--device",     default="",
                   help="Inference device: cpu | cuda | mps (empty = auto)")

    p.add_argument("--refractory", type=float, default=0.35,
                   help="Min seconds between two gate events (same gate)")
    p.add_argument("--sigma",      type=float, default=2.5,
                   help="Gaussian smoothing σ (frames) for distance signal")
    p.add_argument("--min-prom",   type=float, default=15.0,
                   help="Min prominence (px) for a valid gate passage minimum")

    p.add_argument("--debug-dir", metavar="PATH", default=None,
                   help="If set, write annotated debug frames to this folder")
    return p


def main():
    args = _build_parser().parse_args()

    if args.frames_dir and args.fps is None:
        print("[ERROR] --fps is required when using --frames-dir", file=sys.stderr)
        sys.exit(1)

    if args.video:
        source = _frames_from_video(args.video)
    else:
        source = _frames_from_dir(args.frames_dir, fps=args.fps)

    result = process(
        frame_source  = source,
        run_id        = args.run_id,
        det_model     = args.det_model,
        pose_model    = args.pose_model,
        det_conf      = args.det_conf,
        pose_conf     = args.pose_conf,
        refractory_s  = args.refractory,
        sigma         = args.sigma,
        min_prominence= args.min_prom,
        device        = args.device,
        debug_dir     = args.debug_dir,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    n = len(result["gates"])
    print(f"\n[DONE] {n} gate(s) written → {args.out}")


if __name__ == "__main__":
    main()