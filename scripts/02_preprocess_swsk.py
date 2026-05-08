"""
02_preprocess_swisski.py
------------------------
Extract frames from SwissSki race videos and parse gate-passage annotations
from the accompanying CSV files.

Expected input layout:
    data/raw_videos/   →  one .mp4 per run
    data/annotations/swsk/raw/  →  matching .csv files (same base name)

CSV columns (after lowercasing): position (ms), dauer (ms), gate

Output layout:
    data/frames/swsk/<run_id>/  →  extracted JPEG frames (640x640)
    data/annotations/swsk/<run_id>.json  →  gate timestamps with frame numbers

Usage:
    python 02_preprocess_swisski.py
    python 02_preprocess_swisski.py --fps 30 --size 640 640
"""

import argparse
import json
import os
import cv2
import pandas as pd

FPS         = 30
TARGET_SIZE = (640, 640)   # width x height — matches YOLO training resolution
JPEG_QUAL   = 90

VIDEOS_DIR  = "data/raw_videos"
FRAMES_DIR  = "data/frames/swsk"
CSV_DIR     = "data/annotations/swsk/raw"
ANNOT_DIR   = "data/annotations/swsk"


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, out_dir: str, size: tuple, quality: int) -> int:
    existing = [f for f in os.listdir(out_dir) if f.endswith(".jpg")] if os.path.exists(out_dir) else []
    if existing:
        print(f"    [SKIP] {len(existing)} frames already exist")
        return len(existing)

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(
            os.path.join(out_dir, f"{count:06d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        count += 1
    cap.release()

    meta = {"fps": actual_fps, "total_frames": count, "target_size": size}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"    Extracted {count} frames  (actual FPS: {actual_fps:.1f})")
    return count


# ── Annotation parsing ────────────────────────────────────────────────────────

def parse_csv(csv_path: str, fps: float) -> list:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    gates = []
    for i, row in df.iterrows():
        pos_ms  = int(row["position"])
        dur_ms  = int(row["dauer"])
        label   = str(row.get("gate", i + 1)).strip()
        pos_s   = pos_ms / 1000.0
        frame   = round(pos_s * fps)

        gates.append({
            "gate_number": i + 1,
            "gate_label":  label,
            "position_ms": pos_ms,
            "position_s":  round(pos_s, 4),
            "frame":       frame,
            "duration_ms": dur_ms,
            "duration_s":  round(dur_ms / 1000.0, 4),
        })
    return gates


# ── Run matching ──────────────────────────────────────────────────────────────

def find_runs(videos_dir: str, csv_dir: str) -> list:
    videos = {
        os.path.splitext(f)[0]: os.path.join(videos_dir, f)
        for f in os.listdir(videos_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi"))
    } if os.path.isdir(videos_dir) else {}

    csvs = {
        os.path.splitext(f)[0]: os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.lower().endswith(".csv")
    } if os.path.isdir(csv_dir) else {}

    matched   = [(rid, videos[rid], csvs[rid]) for rid in videos if rid in csvs]
    unmatched = [rid for rid in videos if rid not in csvs]
    if unmatched:
        print(f"  WARNING: No CSV for: {unmatched}")
    return sorted(matched)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(fps: float = FPS, size: tuple = TARGET_SIZE):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(ANNOT_DIR,  exist_ok=True)

    runs = find_runs(VIDEOS_DIR, CSV_DIR)
    if not runs:
        print("No matched video+CSV pairs found.")
        print(f"  Videos : {VIDEOS_DIR}")
        print(f"  CSVs   : {CSV_DIR}")
        return

    print(f"\nFound {len(runs)} run(s)  |  FPS={fps}  Size={size[0]}x{size[1]}\n{'='*60}")

    for run_id, video_path, csv_path in runs:
        print(f"\n[{run_id}]")

        frames_out = os.path.join(FRAMES_DIR, run_id)
        print(f"  Extracting frames → {frames_out}")
        extract_frames(video_path, frames_out, size, JPEG_QUAL)

        annot_out = os.path.join(ANNOT_DIR, f"{run_id}.json")
        if os.path.exists(annot_out):
            print(f"  [SKIP] Annotation already exists")
        else:
            print(f"  Parsing CSV → {csv_path}")
            gates = parse_csv(csv_path, fps)
            annotation = {"run_id": run_id, "fps": fps, "gates": gates}
            with open(annot_out, "w") as f:
                json.dump(annotation, f, indent=2)
            print(f"  Saved {len(gates)} gates → {annot_out}")

    print(f"\n{'='*60}\nPreprocessing complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps",  type=float, default=FPS)
    parser.add_argument("--size", nargs=2, type=int, default=list(TARGET_SIZE), metavar=("W", "H"))
    args = parser.parse_args()
    main(fps=args.fps, size=tuple(args.size))