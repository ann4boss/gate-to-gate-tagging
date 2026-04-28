"""
process_all_runs.py

Extracts frames and parses annotations for all runs in one go.
Run this once on your full SwissSki dataset.

Expected input layout:
    data/raw_videos/   →  one .mp4 per run, named like Aerni_SUI_Lauf1_20251116_SL_Levi.mp4
    data/annotations/raw/  →  matching .csv files with same base name

Output layout:
    data/frames/<run_id>/  →  extracted JPEG frames
    data/annotations/<run_id>.json  →  parsed gate timestamps with frame numbers

Usage:
    python process_all_runs.py
    python process_all_runs.py --fps 30 --size 384 288
"""

import os
import json
import argparse
import cv2
import pandas as pd


FPS          = 30             # your actual video frame rate
TARGET_SIZE  = (384, 288)     # width x height required by HRNet
JPEG_QUALITY = 90

VIDEOS_DIR   = "data/raw_videos"
FRAMES_DIR   = "data/frames"
RAW_CSV_DIR  = "data/annotations/raw"
ANNOT_DIR    = "data/annotations"


#*-- frame extraction
def extract_frames(video_path, output_dir, target_size=TARGET_SIZE, quality=JPEG_QUALITY):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 1:
        n = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
        print(f"    [SKIP] frames already exist ({n} frames)")
        return

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps_actual = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(
            os.path.join(output_dir, f"{frame_idx:06d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        frame_idx += 1
    cap.release()

    # Save meta so FPS is always recoverable
    meta = {"fps": fps_actual, "total_frames": frame_idx,
            "target_size": target_size, "video_path": video_path}
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"    Extracted {frame_idx} frames  (actual FPS: {fps_actual})")


#*-- annotation parsing
def parse_csv(csv_path, fps, out_path):
    if os.path.exists(out_path):
        print(f"    [SKIP] annotation already exists ({out_path})")
        with open(out_path) as f:
            return json.load(f)["gates"]
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    gates = []
    for i, row in df.iterrows():
        position_ms = int(row["position"])
        duration_ms = int(row["dauer"])
        gate_label  = str(row["gate"]).strip()

        position_s = position_ms / 1000.0
        frame      = round(position_s * fps)

        gates.append({
            "gate_number": i + 1,
            "gate_label":  gate_label,
            "position_ms": position_ms,
            "position_s":  round(position_s, 4),
            "frame":       frame,
            "duration_ms": duration_ms,
            "duration_s":  round(duration_ms / 1000.0, 4),
        })

    return gates


#*-- match videos to CSVs
def find_runs(videos_dir, csv_dir):
    """
    Match video files to CSV files by base name.
    Returns list of (run_id, video_path, csv_path) tuples.
    """
    videos = {os.path.splitext(f)[0]: os.path.join(videos_dir, f)
              for f in os.listdir(videos_dir)
              if f.lower().endswith((".mp4", ".mov", ".avi"))}

    csvs   = {os.path.splitext(f)[0]: os.path.join(csv_dir, f)
              for f in os.listdir(csv_dir)
              if f.lower().endswith(".csv")}

    matched   = [(rid, videos[rid], csvs[rid])
                 for rid in videos if rid in csvs]
    unmatched = [rid for rid in videos if rid not in csvs]

    if unmatched:
        print(f"WARNING: No CSV found for: {unmatched}\n")

    return sorted(matched)


#*-- main
def main(fps=FPS, target_size=TARGET_SIZE):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(ANNOT_DIR,  exist_ok=True)

    runs = find_runs(VIDEOS_DIR, RAW_CSV_DIR)

    if not runs:
        print(f"No matched video+CSV pairs found.")
        print(f"  Videos dir : {VIDEOS_DIR}")
        print(f"  CSV dir    : {RAW_CSV_DIR}")
        print(f"  Make sure filenames match exactly (without extension).")
        return

    print(f"Found {len(runs)} run(s) to process\n")
    print(f"Settings: FPS={fps}, size={target_size[0]}x{target_size[1]}\n")
    print("=" * 60)

    results = []

    for run_id, video_path, csv_path in runs:
        print(f"\n[{run_id}]")

        # 1. Extract frames
        frames_out = os.path.join(FRAMES_DIR, run_id)
        print(f"  Extracting frames → {frames_out}")
        extract_frames(video_path, frames_out, target_size)

        # 2. Parse CSV annotations
        annot_out = os.path.join(ANNOT_DIR, run_id + ".json")
        print(f"  Parsing annotations → {csv_path}")
        gates = parse_csv(csv_path, fps, annot_out)

        # 3. Save JSON annotation
        if not os.path.exists(annot_out):
            annotation = {
                "run_id": run_id,
                "fps":    fps,
                "gates":  gates,
            }
        annot_out = os.path.join(ANNOT_DIR, run_id + ".json")
        annotation = {
            "run_id": run_id,
            "fps":    fps,
            "gates":  gates,
        }
        with open(annot_out, "w") as f:
            json.dump(annotation, f, indent=2)

        n_gates = len(gates)
        run_duration = gates[-1]["position_s"] if gates else 0
        print(f"  Gates: {n_gates}  |  Duration: {run_duration:.2f}s  |  Saved → {annot_out}")

        results.append({
            "run_id":       run_id,
            "n_gates":      n_gates,
            "duration_s":   run_duration,
            "frames_dir":   frames_out,
            "annotation":   annot_out,
        })

    # Summary
    print("\n" + "=" * 60)
    print(f"\nAll done! Processed {len(results)} run(s):\n")
    for r in results:
        print(f"  {r['run_id']}")
        print(f"    Gates: {r['n_gates']}  |  Duration: {r['duration_s']:.1f}s")
    print()


#*-- CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps",  type=float, default=FPS)
    parser.add_argument("--size", nargs=2, type=int, default=list(TARGET_SIZE),
                        metavar=("W", "H"))
    args = parser.parse_args()
    main(fps=args.fps, target_size=tuple(args.size))