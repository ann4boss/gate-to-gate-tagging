"""
prepare_epfl_pose.py
--------------------
Converts the EPFL Ski-2DPose dataset (COCO-style JSON) into the YOLO-pose
format expected by Ultralytics for keypoint training.

COCO keypoint format in annotations:
    "keypoints": [x1, y1, v1, x2, y2, v2, ...]  (flat list, 3 values per kpt)
    visibility: 0 = not labeled, 1 = labeled but occluded, 2 = labeled & visible

YOLO-pose label format (one .txt per image, one line per person):
    class cx cy bw bh  x1 y1 v1  x2 y2 v2  ...  xK yK vK
    - class is always 0 (skier)
    - cx cy bw bh are normalised bbox [0,1]
    - xi yi are normalised keypoint coords [0,1]
    - vi is visibility: 0 = not labeled, 1 = occluded, 2 = visible

Ski-2DPose 24 keypoint order (indices 0-23):
    COCO body (0-16):
        0  nose            5  left shoulder   10  right wrist
        1  left eye        6  right shoulder  11  left hip
        2  right eye       7  left elbow      12  right hip
        3  left ear        8  right elbow     13  left knee
        4  right ear       9  left wrist      14  right knee
                                              15  left ankle
                                              16  right ankle
    Ski-specific (17-23):
        17  left ski tip       20  right ski tip
        18  left ski tail      21  right ski tail
        19  left pole grip     22  right pole grip
        23  left pole tip  (some versions have 24 — we handle both)

Usage
-----
    python prepare_epfl_pose.py \
        --images-dir  /path/to/epfl/images \
        --ann-file    /path/to/epfl/annotations.json \
        --out-dir     yolo_epfl_pose

    --pad    fractional bbox padding (default 0.05)
    --seed   random seed for split (default 42)
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2


# ── Helpers ───────────────────────────────────────────────────────────────────

def coco_bbox_to_yolo(x, y, w, h, img_w, img_h, pad=0.05):
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
    return (
        max(0., min(1., cx)),
        max(0., min(1., cy)),
        max(0., min(1., bw)),
        max(0., min(1., bh)),
    )


def coco_kpts_to_yolo(keypoints_flat, img_w, img_h, n_kpts):
    """
    Convert a flat COCO keypoints list [x,y,v, x,y,v, ...] to YOLO format.
    Returns a list of (xn, yn, v) tuples, normalised by image size.
    Missing keypoints (v==0, x==0, y==0) are kept as (0, 0, 0).
    """
    result = []
    for i in range(n_kpts):
        base = i * 3
        if base + 2 >= len(keypoints_flat):
            result.append((0.0, 0.0, 0))
            continue
        x = keypoints_flat[base]
        y = keypoints_flat[base + 1]
        v = int(keypoints_flat[base + 2])
        xn = max(0., min(1., x / img_w)) if v > 0 else 0.0
        yn = max(0., min(1., y / img_h)) if v > 0 else 0.0
        result.append((xn, yn, v))
    return result


def write_yaml(out_dir: Path, n_kpts: int, splits: dict):
    """Write dataset.yaml for Ultralytics pose training."""
    # Build flip_idx: pairs of left/right keypoints for horizontal flip augmentation
    # COCO body pairs (0-indexed): eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
    coco_flips = [
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
    ]
    # Ski-specific pairs (adjust indices to match your dataset's exact ordering)
    ski_flips  = [(17, 20), (18, 21), (19, 22)]  # ski tips, tails, pole grips

    flip_idx = list(range(n_kpts))  # identity by default
    for a, b in (coco_flips + ski_flips if n_kpts > 17 else coco_flips):
        if a < n_kpts and b < n_kpts:
            flip_idx[a] = b
            flip_idx[b] = a

    lines = [
        f"path: {out_dir.resolve()}",
        f"train: images/train",
        f"val:   images/val",
        f"test:  images/test",
        f"",
        f"nc: 1",
        f"names: ['skier']",
        f"",
        f"# Keypoint config",
        f"kpt_shape: [{n_kpts}, 3]",
        f"flip_idx: {flip_idx}",
        f"",
        f"# Split sizes  train={splits['train']}  val={splits['val']}  test={splits['test']}",
    ]
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(lines))
    print(f"[INFO] dataset.yaml written → {yaml_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Convert EPFL Ski-2DPose (COCO JSON) → YOLO-pose format"
    )
    p.add_argument("--images-dir", required=True)
    p.add_argument("--ann-file",   required=True)
    p.add_argument("--out-dir",    default="yolo_epfl_pose")
    p.add_argument("--pad",        type=float, default=0.05)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--no-copy",    action="store_true",
                   help="Symlink images instead of copying (saves disk space)")
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
    print(f"[INFO] Loading {ann_file} …")
    with open(ann_file) as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}

    # Detect number of keypoints from the first annotation that has them
    n_kpts = 17  # fallback
    for ann in coco["annotations"]:
        kpts = ann.get("keypoints", [])
        if kpts:
            n_kpts = len(kpts) // 3
            break
    print(f"[INFO] Detected {n_kpts} keypoints per annotation")
    if n_kpts not in (17, 24):
        print(f"[WARN] Expected 17 or 24 keypoints, got {n_kpts} — proceeding anyway")

    # Group annotations by image
    img_to_anns: dict = {img_id: [] for img_id in id_to_img}
    skipped_no_kpts = 0
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            continue
        if not ann.get("keypoints"):
            skipped_no_kpts += 1
            continue
        img_to_anns[img_id].append(ann)

    if skipped_no_kpts:
        print(f"[WARN] Skipped {skipped_no_kpts} annotations with no keypoints")

    valid_ids = [img_id for img_id, anns in img_to_anns.items() if anns]
    print(f"[INFO] Images with pose annotations: {len(valid_ids)} / {len(id_to_img)}")

    # ── Train / val / test split (80 / 10 / 10) ───────────────────────────────
    random.shuffle(valid_ids)
    n       = len(valid_ids)
    n_val   = max(1, round(n * 0.10))
    n_test  = max(1, round(n * 0.10))
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
    written = 0
    skipped = 0
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_id, split in split_map.items():
        img_info  = id_to_img[img_id]
        file_name = img_info["file_name"]
        src_path  = images_dir / file_name
        if not src_path.exists():
            src_path = images_dir / Path(file_name).name
        if not src_path.exists():
            print(f"[WARN] Image not found, skipping: {file_name}")
            skipped += 1
            continue

        # Image dimensions
        img_w = img_info.get("width")
        img_h = img_info.get("height")
        if not img_w or not img_h:
            tmp = cv2.imread(str(src_path))
            if tmp is None:
                print(f"[WARN] Cannot read image, skipping: {src_path}")
                skipped += 1
                continue
            img_h, img_w = tmp.shape[:2]

        # Build label lines — one per annotated person
        label_lines = []
        for ann in img_to_anns[img_id]:
            bx, by, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue

            cx, cy, nbw, nbh = coco_bbox_to_yolo(
                bx, by, bw, bh, img_w, img_h, args.pad
            )
            kpts = coco_kpts_to_yolo(
                ann["keypoints"], img_w, img_h, n_kpts
            )

            kpt_str = "  ".join(f"{x:.6f} {y:.6f} {v}" for x, y, v in kpts)
            label_lines.append(
                f"0 {cx:.6f} {cy:.6f} {nbw:.6f} {nbh:.6f}  {kpt_str}"
            )

        if not label_lines:
            skipped += 1
            continue

        # Write label file
        stem    = Path(file_name).stem
        dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
        dst_img = out_dir / "images" / split / Path(file_name).name

        dst_lbl.write_text("\n".join(label_lines))

        if not dst_img.exists():
            if args.no_copy:
                dst_img.symlink_to(src_path.resolve())
            else:
                shutil.copy2(src_path, dst_img)

        written += 1

    print(f"[INFO] Written: {written}  Skipped: {skipped}")
    write_yaml(out_dir, n_kpts, {"train": n_train, "val": n_val, "test": n_test})

    print("\n[DONE] Pose dataset ready. Next step:")
    print(f"       python train_pose.py --data {out_dir}/dataset.yaml")


if __name__ == "__main__":
    main()