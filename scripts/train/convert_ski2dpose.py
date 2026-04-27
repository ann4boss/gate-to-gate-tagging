"""
train/convert_ski2dpose.py
==========================
Converts the EPFL Ski-2DPose annotation format to YOLO-keypoint label files
so they can be used directly with YOLO26-pose training.

Ski-2DPose annotation structure (inferred from the paper):
    {
      "images": [{"id": 1, "file_name": "frame_001.jpg", "width": 1920, "height": 1080}, ...],
      "annotations": [{
          "image_id": 1,
          "bbox": [x, y, w, h],          # pixel coords, top-left + wh
          "keypoints": [x0,y0,v0, x1,y1,v1, ...],   # 24*3 values
          "num_keypoints": 24
      }, ...]
    }

Visibility flags: 0=not annotated, 1=occluded (best-guess), 2=visible.
YOLO uses the same convention so we pass them through unchanged.

Usage
-----
    python train/convert_ski2dpose.py \
        --src /data/ski2dpose_raw   \
        --dst ski2dpose             \
        --split 0.9

Directory layout of --src expected:
    <src>/
        annotations/
            train.json   (or a single annotations.json)
        images/
            *.jpg
"""

import argparse
import json
import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple


N_KEYPOINTS = 24


def convert(
    src_dir:    str,
    dst_dir:    str,
    split:      float = 0.9,
    seed:       int   = 42,
):
    src  = Path(src_dir)
    dst  = Path(dst_dir)
    rng  = random.Random(seed)

    # ── Load annotation JSON ──────────────────────────────────────────────────
    ann_candidates = [
        src / "annotations" / "train.json",
        src / "annotations" / "annotations.json",
        src / "annotations.json",
    ]
    ann_path = next((p for p in ann_candidates if p.exists()), None)
    if ann_path is None:
        raise FileNotFoundError(
            f"No annotation JSON found in {src}/annotations/. "
            "Expected train.json or annotations.json."
        )

    print(f"Loading annotations from {ann_path} ...")
    with open(ann_path) as f:
        data = json.load(f)

    images_meta = {img["id"]: img for img in data["images"]}
    annotations  = data["annotations"]

    # ── Shuffle and split ─────────────────────────────────────────────────────
    rng.shuffle(annotations)
    cut = int(len(annotations) * split)
    splits = {"train": annotations[:cut], "val": annotations[cut:]}

    for subset, anns in splits.items():
        img_out = dst / "images" / subset
        lbl_out = dst / "labels" / subset
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"  Writing {subset}: {len(anns)} annotations ...")

        for ann in anns:
            img_meta  = images_meta[ann["image_id"]]
            file_name = img_meta["file_name"]
            W, H      = img_meta["width"], img_meta["height"]

            # ── Copy image ────────────────────────────────────────────────────
            src_img = _find_image(src, file_name)
            if src_img is None:
                print(f"    [WARN] Image not found: {file_name}")
                continue
            dst_img = img_out / Path(file_name).name
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # ── Build YOLO label line ─────────────────────────────────────────
            bx, by, bw, bh = ann["bbox"]            # pixel, XYWH
            cx = (bx + bw / 2) / W
            cy = (by + bh / 2) / H
            nw = bw / W
            nh = bh / H

            kpts_raw = ann["keypoints"]              # 24*3 flat list
            kpt_parts: List[str] = []
            for i in range(N_KEYPOINTS):
                kx = kpts_raw[i * 3]     / W
                ky = kpts_raw[i * 3 + 1] / H
                kv = kpts_raw[i * 3 + 2]            # visibility flag
                kpt_parts.append(f"{kx:.6f} {ky:.6f} {int(kv)}")

            label_line = (
                f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} "
                + " ".join(kpt_parts)
            )

            lbl_path = lbl_out / (Path(file_name).stem + ".txt")
            with open(lbl_path, "a") as f:
                f.write(label_line + "\n")

    print(f"\nConversion done.  Dataset written to: {dst}")
    print(f"  Train: {len(splits['train'])}  |  Val: {len(splits['val'])}")
    print(f"\nNext step:\n"
          f"  Update data/ski2dpose.yaml  →  set  path: {dst.resolve()}\n"
          f"  Then run:  python train/train_pose.py --data data/ski2dpose.yaml")


def _find_image(src: Path, file_name: str) -> Path:
    """Search common image subdirectories."""
    candidates = [
        src / "images" / file_name,
        src / "images" / Path(file_name).name,
        src / file_name,
    ]
    return next((p for p in candidates if p.exists()), None)


def _build_parser():
    p = argparse.ArgumentParser(
        description="Convert Ski-2DPose → YOLO-keypoint label format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src",   required=True, help="Raw Ski-2DPose root dir")
    p.add_argument("--dst",   default="ski2dpose",
                   help="Output dataset dir (created if missing)")
    p.add_argument("--split", type=float, default=0.9,
                   help="Train/val split ratio")
    p.add_argument("--seed",  type=int,   default=42)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    convert(args.src, args.dst, args.split, args.seed)