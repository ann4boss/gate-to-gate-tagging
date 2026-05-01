"""
sample_frames_for_annotation.py
--------------------------------
Samples a representative set of frames from all 12 SwissSki runs for
manual gate pole annotation in Label Studio.

We do NOT extract every frame — that would give you thousands of nearly
identical images to annotate.  Instead we use two sampling strategies:

  1. UNIFORM sampling — every N seconds throughout the run, to cover all
     gates across the full course

  2. GATE-AWARE sampling — frames close to each known gate passage
     timestamp (from your parsed annotations JSON), to ensure we capture
     frames where the pole is close to the skier and clearly visible.
     These are the most important frames for the detector.

Target: ~40 frames per run, ~480 total across all 12 runs.

Output
------
annotation_frames/
├── MyRun_001__frame_000123.jpg
├── MyRun_001__frame_000456.jpg
├── ...
└── sampling_report.json    ← what was sampled and why

The flat folder structure is what Label Studio expects for a simple
image annotation project.

Usage
-----
    python sample_frames_for_annotation.py \
        --frames-root  data/frames/swsk \
        --ann-root     data/annotations \
        --out-dir      annotation_frames

    The script expects one subfolder per run inside --frames-root, e.g.:
        data/frames/swsk/Aerni_Lauf1/
        data/frames/swsk/Aerni_Lauf2/

Optional
    --uniform-interval   seconds between uniform samples (default: 2.0)
    --gate-window        frames either side of gate timestamp to sample (default: 8)
    --max-per-run        max frames to sample per run (default: 40)
    --seed               random seed (default: 42)
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def load_meta(frames_dir: Path) -> dict:
    meta_path = frames_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"fps": 25, "total_frames": None}


def load_annotation(ann_path: Path) -> list:
    """Return list of gate frame numbers from parsed annotation JSON."""
    if not ann_path.exists():
        return []
    with open(ann_path) as f:
        data = json.load(f)
    gates = data.get("gates", [])
    return [g["frame"] for g in gates if "frame" in g]


def get_frame_paths(frames_dir: Path) -> list:
    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in frames_dir.iterdir()
                  if p.suffix.lower() in IMG_EXTS)


def sample_run(
    frames_dir:       Path,
    ann_path:         Path,
    uniform_interval: float,
    gate_window:      int,
    max_per_run:      int,
    rng:              random.Random,
) -> list:
    """Return a list of frame Paths to sample from this run, capped at max_per_run."""
    meta        = load_meta(frames_dir)
    fps         = float(meta.get("fps", 25))
    frame_paths = get_frame_paths(frames_dir)
    n_frames    = len(frame_paths)

    if n_frames == 0:
        print(f"  [WARN] No frames found in {frames_dir}")
        return []

    gate_frames = load_annotation(ann_path)

    # ── 1. Gate-aware candidates — ONE frame per gate (the passage frame itself)
    #    We take exactly the annotated frame, not a window around it.
    #    This avoids the gate_window explosion that caused 450 frames per run.
    gate_candidates = []
    for gf in gate_frames:
        if 0 <= gf < n_frames:
            gate_candidates.append(gf)

    # ── 2. Uniform candidates — evenly spaced across the run
    step = max(1, int(uniform_interval * fps))
    uniform_candidates = list(range(0, n_frames, step))

    # ── 3. Combine: gate frames first (higher priority), then uniform
    #    Deduplicate while preserving priority order, then cap at max_per_run.
    seen     = set()
    combined = []
    for idx in gate_candidates + uniform_candidates:
        if idx not in seen:
            seen.add(idx)
            combined.append(idx)

    # Trim to max_per_run — gate frames are already at the front so they
    # survive the trim; uniform frames fill the remaining slots.
    if len(combined) > max_per_run:
        # If gate frames alone already exceed the cap, subsample them evenly
        if len(gate_candidates) >= max_per_run:
            step_g   = max(1, len(gate_candidates) // max_per_run)
            combined = gate_candidates[::step_g][:max_per_run]
        else:
            n_uniform = max_per_run - len(gate_candidates)
            chosen_uniform = rng.sample(
                [i for i in uniform_candidates if i not in set(gate_candidates)],
                min(n_uniform, len(uniform_candidates))
            )
            combined = sorted(set(gate_candidates) | set(chosen_uniform))

    return [frame_paths[i] for i in sorted(combined)]


def main():
    p = argparse.ArgumentParser(
        description="Sample frames from SwissSki runs for gate pole annotation"
    )
    p.add_argument("--frames-root",       required=True,
                   help="Root folder containing one subfolder per run")
    p.add_argument("--ann-root",          required=True,
                   help="Folder containing parsed annotation JSONs")
    p.add_argument("--out-dir",           default="annotation_frames",
                   help="Output folder of sampled frames (default: annotation_frames)")
    p.add_argument("--uniform-interval",  type=float, default=2.0,
                   help="Seconds between uniform samples (default: 2.0)")
    p.add_argument("--gate-window",       type=int,   default=8,
                   help="Frames either side of gate timestamp to include (default: 8)")
    p.add_argument("--max-per-run",       type=int,   default=40,
                   help="Max frames per run (default: 40)")
    p.add_argument("--seed",              type=int,   default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    frames_root = Path(args.frames_root)
    ann_root    = Path(args.ann_root)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all run folders
    run_dirs = sorted(d for d in frames_root.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"[ERROR] No subfolders found in {frames_root}")
        return

    print(f"[INFO] Found {len(run_dirs)} run folders")

    report      = []
    total_copied = 0

    for run_dir in run_dirs:
        run_name = run_dir.name

        # Find matching annotation JSON
        ann_path = ann_root / f"{run_name}.json"
        if not ann_path.exists():
            # Try case-insensitive match
            matches = list(ann_root.glob(f"{run_name}*.json"))
            ann_path = matches[0] if matches else Path("__missing__")

        has_ann = ann_path.exists()
        if not has_ann:
            print(f"  [WARN] No annotation JSON for {run_name} — uniform only")

        sampled = sample_run(
            frames_dir       = run_dir,
            ann_path         = ann_path,
            uniform_interval = args.uniform_interval,
            gate_window      = args.gate_window,
            max_per_run      = args.max_per_run,
            rng              = rng,
        )

        print(f"  {run_name}: {len(sampled)} frames sampled")

        for src in sampled:
            # Flat filename: RunName__frame_XXXXXX.jpg
            dst_name = f"{run_name}__{src.name}"
            dst      = out_dir / dst_name
            if not dst.exists():
                shutil.copy2(src, dst)
            total_copied += 1

        report.append({
            "run":            run_name,
            "n_sampled":      len(sampled),
            "has_annotation": has_ann,
            "frames":         [f"{run_name}__{s.name}" for s in sampled],
        })

    # Write sampling report
    report_path = out_dir / "sampling_report.json"
    with open(report_path, "w") as f:
        json.dump({"total": total_copied, "runs": report}, f, indent=2)

    print(f"\n[DONE] Sampled {total_copied} frames → {out_dir}")
    print(f"       Sampling report → {report_path}")
    print(f"""
    Next steps:
    1. Install Label Studio:
            pip install label-studio
    
    2. Start Label Studio:
            label-studio start
    
    3. Create a new project in the browser (http://localhost:8080)
        - Template: Object Detection with Bounding Boxes
        - Add these labels exactly:
            gate_contact
            gate_outer
    
    4. Import your frames folder:
        - Settings → Cloud Storage → Local Files
        - Point to: {out_dir.resolve()}
    
    5. Annotate! Draw bboxes around gate poles using these rules:
    
        ── LABEL DEFINITIONS ──────────────────────────────────────────────
        gate_contact
        The pole the skier will hit or is currently hitting.
            * GS: the INNER rigid pole of the two-pole gate (the one
                on the inside of the turn, closest to the skier's path)
            * SL: the single hinged panel pole (the one the skier
                deflects with their hand or shin)
        Use this label for ALL disciplines. It is always the pole
        that is physically contacted by the skier.
    
        gate_outer
        The OUTER rigid pole of a GS gate — the second pole on the
        outside of the turn that the skier passes around but does NOT
        touch. Only exists in GS. NEVER use this label in SL frames
        because slalom gates have only one pole.
    
        ── WHICH GATES TO LABEL ───────────────────────────────────────────
        Label ALL gates that are AHEAD OF or AT the skier's current
        position in the frame. This includes:
        * Gates the skier is approaching (even if far away)
        * The gate currently being passed (even if pole is deflecting)
        Do NOT label gates that are clearly BEHIND the skier — those
        are already passed and carry no useful signal for the detector.
        In broadcast footage this usually means 1-3 gates per frame.
    
        ── PRACTICAL TIPS ─────────────────────────────────────────────────
        * Every frame should have at least one gate_contact label
            (unless all upcoming poles are fully out of frame or occluded)
        * In GS frames: also draw gate_outer on the outer pole if visible
        * In SL frames: only use gate_contact, never gate_outer
        * It is okay to skip a frame entirely if all upcoming poles
            are fully occluded or outside the frame
    
    6. Export annotations:
        - Export → JSON format
        - Save as: gate_pole_annotations.json
    
    7. Convert to YOLO format:
            python convert_labelstudio_to_yolo.py \\
                --ann-file  gate_pole_annotations.json \\
                --img-dir   {out_dir.resolve()} \\
                --out-dir   yolo_gate_poles
    """)


if __name__ == "__main__":
    main()