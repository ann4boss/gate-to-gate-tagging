"""
03_prepare_epfl_pose.py
-----------------------
Convert the EPFL Ski-2DPose COCO-style JSON into YOLO-pose label files
for fine-tuning a 24-keypoint pose model.

Ski-2DPose keypoint order (0-indexed):
    COCO body (0-16): nose, L/R eye, L/R ear, L/R shoulder, L/R elbow,
                      L/R wrist, L/R hip, L/R knee, L/R ankle
    Ski-specific (17-23): L ski tip, L ski tail, L pole grip,
                           R ski tip, R ski tail, R pole grip, L pole tip

COCO visibility: 0=not labeled, 1=occluded, 2=visible  (passed through unchanged)

YOLO pose label format (one line per person):
    class cx cy bw bh  x1 y1 v1  x2 y2 v2  ...
    All coordinates normalised to [0, 1].

Usage:
    python 03_prepare_epfl_pose.py \
        --images-dir data/frames/epfl/Images_webp \
        --ann-file   data/annotations/epfl/ski2dpose_labels.json \
        --out-dir    data/yolo/epfl_pose
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2

N_KPTS = 24   # Ski-2DPose keypoint count

# Keypoint flip pairs for horizontal augmentation (left ↔ right)
FLIP_PAIRS = [
    (1, 2), (3, 4),           # eyes, ears
    (5, 6), (7, 8), (9, 10),  # shoulders, elbows, wrists
    (11, 12), (13, 14), (15, 16),  # hips, knees, ankles
    (17, 20), (18, 21), (19, 22),  # ski tips, tails, pole grips
]


def coco_bbox_to_yolo(x, y, w, h, img_w, img_h, pad=0.05):
    x1 = max(0.0, x - w * pad)
    y1 = max(0.0, y - h * pad)
    x2 = min(img_w, x + w + w * pad)
    y2 = min(img_h, y + h + h * pad)
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return (max(0., min(1., v)) for v in (cx, cy, bw, bh))


def coco_kpts_to_yolo(kpts_flat, img_w, img_h, n_kpts):
    result = []
    for i in range(n_kpts):
        base = i * 3
        if base + 2 >= len(kpts_flat):
            result.append((0.0, 0.0, 0))
            continue
        x, y, v = kpts_flat[base], kpts_flat[base+1], int(kpts_flat[base+2])
        xn = max(0., min(1., x / img_w)) if v > 0 else 0.0
        yn = max(0., min(1., y / img_h)) if v > 0 else 0.0
        result.append((xn, yn, v))
    return result


def build_flip_idx(n_kpts):
    flip_idx = list(range(n_kpts))
    for a, b in FLIP_PAIRS:
        if a < n_kpts and b < n_kpts:
            flip_idx[a], flip_idx[b] = b, a
    return flip_idx


def write_yaml(out_dir: Path, n_kpts: int, splits: dict):
    flip_idx = build_flip_idx(n_kpts)
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        "",
        "nc: 1",
        "names: ['skier']",
        "",
        "# Keypoint config",
        f"kpt_shape: [{n_kpts}, 3]",
        f"flip_idx: {flip_idx}",
        "",
        f"# Split  train={splits['train']}  val={splits['val']}  test={splits['test']}",
    ]
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(lines))
    print(f"  dataset.yaml → {yaml_path}")


def load_ski2dpose_native(ann_file: Path, images_dir: Path, img_w: int = 1920, img_h: int = 1080) -> dict:
    """
    Convert the native Ski-2DPose JSON format into an in-memory COCO dict.

    Native structure:
        {video_id: {cam_id: {frame_name: {"annotation": [[x, y, v], ...]}}}}
    Coordinates are normalised [0, 1]; we convert to pixels using img_w/img_h.
    Bounding box is derived as the tight box around visible keypoints.
    """
    with open(ann_file) as f:
        raw = json.load(f)

    images, annotations = [], []
    img_id = ann_id = 0

    for video_id, video_data in raw.items():
        for cam_id, cam_data in video_data.items():
            for frame_name, item in cam_data.items():
                file_name = f"{video_id}/{cam_id}/{frame_name}.jpg"

                # Accept any image extension present on disk
                src = images_dir / file_name
                if not src.exists():
                    for ext in (".webp", ".png", ".jpeg"):
                        alt = (images_dir / file_name).with_suffix(ext)
                        if alt.exists():
                            file_name = str(alt.relative_to(images_dir))
                            src = alt
                            break

                kpts_norm = item["annotation"]   # list of [x, y, v]
                kpts_px   = []
                xs, ys    = [], []

                for x, y, v in kpts_norm:
                    px, py = x * img_w, y * img_h
                    kpts_px.extend([px, py, int(v)])
                    if v > 0:
                        xs.append(px); ys.append(py)

                if not xs:
                    continue

                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

                images.append({"id": img_id, "file_name": file_name,
                                "width": img_w, "height": img_h})
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": 0,
                    "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
                    "keypoints": kpts_px,
                    "num_keypoints": sum(1 for _, _, v in kpts_norm if v > 0),
                })
                img_id  += 1
                ann_id  += 1

    return {"images": images, "annotations": annotations}


def load_annotations(ann_file: Path, images_dir: Path) -> dict:
    """Load either native Ski-2DPose JSON or standard COCO JSON."""
    with open(ann_file) as f:
        data = json.load(f)

    if "images" in data and "annotations" in data:
        print("  Detected standard COCO format.")
        return data

    print("  Detected native Ski-2DPose format — converting to COCO in memory ...")
    return load_ski2dpose_native(ann_file, images_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--ann-file",   required=True)
    p.add_argument("--out-dir",    default="data/yolo/epfl_pose")
    p.add_argument("--pad",        type=float, default=0.05)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--img-w",      type=int,   default=1920, help="Frame width for native format (default 1920)")
    p.add_argument("--img-h",      type=int,   default=1080, help="Frame height for native format (default 1080)")
    p.add_argument("--no-copy",    action="store_true", help="Symlink instead of copying images")
    args = p.parse_args()

    random.seed(args.seed)
    images_dir = Path(args.images_dir)
    ann_file   = Path(args.ann_file)
    out_dir    = Path(args.out_dir)

    print(f"\nLoading {ann_file} ...")
    coco = load_annotations(ann_file, images_dir)

    id_to_img = {img["id"]: img for img in coco["images"]}

    # Detect keypoint count
    n_kpts = N_KPTS
    for ann in coco["annotations"]:
        if ann.get("keypoints"):
            n_kpts = len(ann["keypoints"]) // 3
            break
    print(f"  Detected {n_kpts} keypoints per annotation")

    # Group annotations by image
    img_to_anns: dict = {iid: [] for iid in id_to_img}
    for ann in coco["annotations"]:
        if ann.get("keypoints") and ann["image_id"] in img_to_anns:
            img_to_anns[ann["image_id"]].append(ann)

    valid_ids = [iid for iid, anns in img_to_anns.items() if anns]
    print(f"  Images with annotations: {len(valid_ids)}")

    # 80 / 10 / 10 split
    random.shuffle(valid_ids)
    n = len(valid_ids)
    n_val   = max(1, round(n * 0.10))
    n_test  = max(1, round(n * 0.10))
    n_train = n - n_val - n_test

    split_map = {}
    for iid in valid_ids[:n_train]:             split_map[iid] = "train"
    for iid in valid_ids[n_train:n_train+n_val]: split_map[iid] = "val"
    for iid in valid_ids[n_train+n_val:]:        split_map[iid] = "test"
    print(f"  Split — train:{n_train}  val:{n_val}  test:{n_test}")

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    for img_id, split in split_map.items():
        info      = id_to_img[img_id]
        file_name = info["file_name"]
        src       = images_dir / file_name
        if not src.exists():
            src = images_dir / Path(file_name).name
        if not src.exists():
            skipped += 1
            continue

        img_w = info.get("width") or cv2.imread(str(src)).shape[1]
        img_h = info.get("height") or cv2.imread(str(src)).shape[0]

        lines = []
        for ann in img_to_anns[img_id]:
            bx, by, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            cx, cy, nbw, nbh = coco_bbox_to_yolo(bx, by, bw, bh, img_w, img_h, args.pad)
            kpts = coco_kpts_to_yolo(ann["keypoints"], img_w, img_h, n_kpts)
            kpt_str = "  ".join(f"{x:.6f} {y:.6f} {v}" for x, y, v in kpts)
            lines.append(f"0 {cx:.6f} {cy:.6f} {nbw:.6f} {nbh:.6f}  {kpt_str}")

        if not lines:
            skipped += 1
            continue

        stem = Path(file_name).stem
        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

        dst_img = out_dir / "images" / split / Path(file_name).name
        if not dst_img.exists():
            dst_img.symlink_to(src.resolve()) if args.no_copy else shutil.copy2(src, dst_img)

        written += 1

    print(f"  Written: {written}  Skipped: {skipped}")
    write_yaml(out_dir, n_kpts, {"train": n_train, "val": n_val, "test": n_test})
    print(f"\nDone. Next: python 04_train_pose.py --data {out_dir}/dataset.yaml\n")


if __name__ == "__main__":
    main()