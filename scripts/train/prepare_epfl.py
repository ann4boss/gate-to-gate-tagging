"""
prepare_epfl.py
---------------
Converts the EPFL Ski-2DPose dataset (COCO-style JSON + image folder)
into the YOLO detection format expected by Ultralytics.

COCO bbox format  : [x_top_left, y_top_left, width, height]
YOLO bbox format  : [cx, cy, w, h]  — all normalised by image dimensions,
                    one .txt file per image, class index always 0 (skier)

Output folder structure
-----------------------
yolo_epfl/
├── images/
│   ├── train/   (80 %)
│   ├── val/     (10 %)
│   └── test/    (10 %)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml

Usage
-----
    python prepare_epfl.py \
        --images-dir  /path/to/epfl/images \
        --ann-file    /path/to/epfl/annotations.json \
        --out-dir     yolo_epfl

Optional
    --pad          fractional padding added to each bbox (default 0.05)
                   adds 5 % of box width/height on each side so the skier
                   is not clipped at the edge of the crop
    --seed         random seed for reproducible split (default 42)
    --no-copy      write symlinks instead of copying images (saves disk space)
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2


# ── Helpers ───────────────────────────────────────────────────────────────────

def coco_to_yolo(x, y, w, h, img_w, img_h, pad=0.05):
    """
    Convert a COCO bbox to a normalised YOLO bbox, with optional padding.

    Parameters
    ----------
    x, y, w, h : COCO bbox (top-left x, top-left y, width, height) in pixels
    img_w, img_h : image dimensions in pixels
    pad : fractional padding added on each side

    Returns
    -------
    (cx, cy, bw, bh) all in [0, 1]
    """
    # Add padding
    pad_x = w * pad
    pad_y = h * pad

    x1 = max(0.0, x - pad_x)
    y1 = max(0.0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h

    # Clamp to [0, 1] just in case
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))

    return cx, cy, bw, bh


def write_yaml(out_dir: Path, splits: dict):
    """Write the dataset.yaml that Ultralytics needs for training."""
    yaml_path = out_dir / "dataset.yaml"
    lines = [
        f"path: {out_dir.resolve()}",
        f"train: images/train",
        f"val:   images/val",
        f"test:  images/test",
        "",
        "nc: 1",
        "names: ['skier']",
        "",
        f"# Split sizes  train={splits['train']}  val={splits['val']}  test={splits['test']}",
    ]
    yaml_path.write_text("\n".join(lines))
    print(f"[INFO] dataset.yaml written → {yaml_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Convert EPFL Ski-2DPose (COCO JSON) → YOLO detection format"
    )
    p.add_argument("--images-dir", required=True,
                   help="Folder containing all EPFL images")
    p.add_argument("--ann-file",   required=True,
                   help="COCO-style JSON annotation file")
    p.add_argument("--out-dir",    default="yolo_epfl",
                   help="Output root folder (created if missing)")
    p.add_argument("--pad",        type=float, default=0.05,
                   help="Fractional bbox padding on each side (default 0.05)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--no-copy",    action="store_true",
                   help="Use symlinks instead of copying images")
    args = p.parse_args()

    random.seed(args.seed)

    images_dir = Path(args.images_dir)
    ann_file   = Path(args.ann_file)
    out_dir    = Path(args.out_dir)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    # ── Load COCO JSON ────────────────────────────────────────────────────────
    print(f"[INFO] Loading annotations from {ann_file} …")
    with open(ann_file) as f:
        coco = json.load(f)

    # Build lookup: image_id → image info
    id_to_img = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    # We keep only annotations where category is "person" / "skier"
    # (EPFL uses a single category — whatever it is we take it)
    img_to_anns: dict = {img_id: [] for img_id in id_to_img}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id in img_to_anns:
            img_to_anns[img_id].append(ann)

    # Drop images with no annotations
    valid_ids = [img_id for img_id, anns in img_to_anns.items() if anns]
    print(f"[INFO] Images with annotations: {len(valid_ids)} / {len(id_to_img)}")

    # ── Train / val / test split ──────────────────────────────────────────────
    random.shuffle(valid_ids)
    n      = len(valid_ids)
    n_val  = max(1, round(n * 0.10))
    n_test = max(1, round(n * 0.10))
    n_train = n - n_val - n_test

    split_map = {}
    for img_id in valid_ids[:n_train]:
        split_map[img_id] = "train"
    for img_id in valid_ids[n_train: n_train + n_val]:
        split_map[img_id] = "val"
    for img_id in valid_ids[n_train + n_val:]:
        split_map[img_id] = "test"

    print(f"[INFO] Split — train: {n_train}  val: {n_val}  test: {n_test}")

    # ── Create output folders ─────────────────────────────────────────────────
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Convert and write ─────────────────────────────────────────────────────
    skipped = 0
    written = 0

    for img_id, split in split_map.items():
        img_info   = id_to_img[img_id]
        file_name  = img_info["file_name"]
        src_path   = images_dir / file_name

        if not src_path.exists():
            # Try just the basename in case file_name has subdirs
            src_path = images_dir / Path(file_name).name
        if not src_path.exists():
            print(f"[WARN] Image not found, skipping: {file_name}")
            skipped += 1
            continue

        # Image dimensions — prefer from JSON, fall back to reading the file
        img_w = img_info.get("width")
        img_h = img_info.get("height")
        if not img_w or not img_h:
            tmp = cv2.imread(str(src_path))
            if tmp is None:
                print(f"[WARN] Cannot read image, skipping: {src_path}")
                skipped += 1
                continue
            img_h, img_w = tmp.shape[:2]

        # Build YOLO label lines
        label_lines = []
        for ann in img_to_anns[img_id]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            cx, cy, bw, bh = coco_to_yolo(x, y, w, h, img_w, img_h, args.pad)
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not label_lines:
            skipped += 1
            continue

        # Destination paths
        stem     = Path(file_name).stem
        dst_img  = out_dir / "images" / split / Path(file_name).name
        dst_lbl  = out_dir / "labels" / split / f"{stem}.txt"

        # Copy or symlink image
        if not dst_img.exists():
            if args.no_copy:
                dst_img.symlink_to(src_path.resolve())
            else:
                shutil.copy2(src_path, dst_img)

        # Write label file
        dst_lbl.write_text("\n".join(label_lines))
        written += 1

    print(f"[INFO] Written: {written}  Skipped: {skipped}")
    write_yaml(out_dir, {"train": n_train, "val": n_val, "test": n_test})
    print("\n[DONE] Dataset ready. Next step:")
    print(f"       python train_skier_detector.py --data {out_dir}/dataset.yaml")


if __name__ == "__main__":
    main()
