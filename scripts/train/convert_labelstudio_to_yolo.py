"""
convert_labelstudio_to_yolo.py
-------------------------------
Converts a Label Studio JSON export into YOLO detection format for
gate pole detector training.

Label classes → YOLO class indices:
    gate_contact → 0   (the pole the skier hits — GS inner pole or SL hinged panel)
    gate_outer   → 1   (outer pole of a GS gate only — never label this in SL)

Label Studio bbox format:
    x, y, width, height — as PERCENTAGES of image dimensions (0-100)
    origin = top-left corner of the bbox

YOLO bbox format:
    cx, cy, w, h — normalised [0,1], centre-based

Output structure
----------------
yolo_gate_poles/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml

Usage
-----
    python convert_labelstudio_to_yolo.py \
        --ann-file  gate_pole_annotations.json \
        --img-dir   annotation_frames \
        --out-dir   yolo_gate_poles

Optional
    --val-split    fraction for validation (default: 0.10)
    --test-split   fraction for test (default: 0.10)
    --seed         random seed (default: 42)
"""

import argparse
import json
import random
import shutil
from pathlib import Path


# ── Label → class index mapping ───────────────────────────────────────────────
# gate_contact (0): the pole the skier hits
#   - In GS: the rigid inner pole of the two-pole gate
#   - In SL: the single hinged panel pole
# gate_outer (1): the outer rigid pole of a GS gate only
#   - Never annotate this in SL frames (there is no outer pole in SL)
LABEL_MAP = {
    "gate_contact": 0,
    "gate_outer":   1,
}


def ls_bbox_to_yolo(x_pct, y_pct, w_pct, h_pct):
    """
    Convert Label Studio percentage bbox to normalised YOLO format.
    Label Studio gives top-left x,y + width,height as percentages (0-100).
    YOLO needs centre x,y + width,height normalised (0-1).
    """
    x  = x_pct / 100.0
    y  = y_pct / 100.0
    w  = w_pct / 100.0
    h  = h_pct / 100.0
    cx = x + w / 2.0
    cy = y + h / 2.0
    # Clamp
    cx = max(0., min(1., cx))
    cy = max(0., min(1., cy))
    w  = max(0., min(1., w))
    h  = max(0., min(1., h))
    return cx, cy, w, h


def write_yaml(out_dir: Path, splits: dict):
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        "",
        "nc: 2",
        "names: ['gate_contact', 'gate_outer']",
        "",
        f"# Split sizes  train={splits['train']}  val={splits['val']}  test={splits['test']}",
    ]
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(lines))
    print(f"[INFO] dataset.yaml written → {yaml_path}")


def main():
    p = argparse.ArgumentParser(
        description="Convert Label Studio export → YOLO format for gate pole detector"
    )
    p.add_argument("--ann-file",   required=True,
                   help="Label Studio JSON export file")
    p.add_argument("--img-dir",    required=True,
                   help="Folder containing the annotated images")
    p.add_argument("--out-dir",    default="yolo_gate_poles")
    p.add_argument("--val-split",  type=float, default=0.10)
    p.add_argument("--test-split", type=float, default=0.10)
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed)

    ann_file = Path(args.ann_file)
    img_dir  = Path(args.img_dir)
    out_dir  = Path(args.out_dir)

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    with open(ann_file) as f:
        tasks = json.load(f)

    print(f"[INFO] Loaded {len(tasks)} annotation tasks")

    # ── Parse annotations ─────────────────────────────────────────────────────
    parsed = []   # list of {filename, labels: [(class_id, cx, cy, w, h)]}
    skipped_no_ann = 0
    skipped_unknown_label = 0

    for task in tasks:
        # Label Studio stores the image filename in task["data"]["image"]
        # It may be a full path or just a filename
        raw_path = task.get("data", {}).get("image", "")
        filename = Path(raw_path).name

        annotations = task.get("annotations", [])
        if not annotations:
            skipped_no_ann += 1
            continue

        # Take the first (or only) annotation set
        results = annotations[0].get("result", [])

        label_lines = []
        for r in results:
            if r.get("type") != "rectanglelabels":
                continue
            value  = r.get("value", {})
            labels = value.get("rectanglelabels", [])
            if not labels:
                continue

            label = labels[0].lower().strip()
            if label not in LABEL_MAP:
                # Try partial match — e.g. "Inner Pole" → "inner_pole"
                label = label.replace(" ", "_")
            if label not in LABEL_MAP:
                skipped_unknown_label += 1
                print(f"  [WARN] Unknown label '{labels[0]}' in {filename} — skipping bbox")
                continue

            class_id = LABEL_MAP[label]
            cx, cy, w, h = ls_bbox_to_yolo(
                value["x"], value["y"], value["width"], value["height"]
            )
            label_lines.append((class_id, cx, cy, w, h))

        if label_lines:
            parsed.append({"filename": filename, "labels": label_lines})

    print(f"[INFO] Tasks with annotations: {len(parsed)}")
    if skipped_no_ann:
        print(f"[WARN] Skipped {skipped_no_ann} tasks with no annotations")
    if skipped_unknown_label:
        print(f"[WARN] Skipped {skipped_unknown_label} bboxes with unrecognised labels")

    if not parsed:
        print("[ERROR] No valid annotations found. Check your Label Studio export.")
        return

    # ── Train / val / test split ──────────────────────────────────────────────
    random.shuffle(parsed)
    n      = len(parsed)
    n_val  = max(1, round(n * args.val_split))
    n_test = max(1, round(n * args.test_split))
    n_train = n - n_val - n_test

    splits_data = {
        "train": parsed[:n_train],
        "val":   parsed[n_train: n_train + n_val],
        "test":  parsed[n_train + n_val:],
    }
    print(f"[INFO] Split — train: {n_train}  val: {n_val}  test: {n_test}")

    # ── Create output folders ─────────────────────────────────────────────────
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Write files ───────────────────────────────────────────────────────────
    written  = 0
    missing  = 0

    for split, items in splits_data.items():
        for item in items:
            src_img = img_dir / item["filename"]
            if not src_img.exists():
                # Try searching recursively
                matches = list(img_dir.rglob(item["filename"]))
                src_img = matches[0] if matches else src_img

            if not src_img.exists():
                print(f"  [WARN] Image not found: {item['filename']}")
                missing += 1
                continue

            stem    = Path(item["filename"]).stem
            dst_img = out_dir / "images" / split / src_img.name
            dst_lbl = out_dir / "labels" / split / f"{stem}.txt"

            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            lines = [
                f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for cls, cx, cy, w, h in item["labels"]
            ]
            dst_lbl.write_text("\n".join(lines))
            written += 1

    print(f"[INFO] Written: {written}  Missing images: {missing}")
    write_yaml(out_dir, {"train": n_train, "val": n_val, "test": n_test})

    print(f"\n[DONE] YOLO dataset ready → {out_dir}")
    print(f"       Next step:")
    print(f"       python train_gate_detector.py --data {out_dir}/dataset.yaml")


if __name__ == "__main__":
    main()