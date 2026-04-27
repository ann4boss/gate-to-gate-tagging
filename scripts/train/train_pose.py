"""
train/train_pose.py
===================
Fine-tune YOLO26-pose on the EPFL Ski-2DPose dataset (or any COCO-keypoint
format dataset).

Ski-2DPose has 24 annotated joints.  YOLO26-pose ships pretrained on COCO-17
keypoints.  This script:
  1. Converts the Ski-2DPose annotations to COCO-keypoint YAML format.
  2. Launches YOLO26 pose training via the Ultralytics API.

Dataset structure expected
--------------------------
ski2dpose/
    images/
        train/  *.jpg|*.png
        val/    *.jpg|*.png
    labels/
        train/  *.txt   (YOLO keypoint format per image)
        val/    *.txt

YOLO keypoint label format (one line per person):
    <class> <cx> <cy> <w> <h>  [<kx> <ky> <kv>] x N_KPT
    all values normalised 0-1, kv ∈ {0=hidden, 1=occluded, 2=visible}

Usage
-----
    python train/train_pose.py \
        --data  data/ski2dpose.yaml \
        --model yolo26s-pose.pt \
        --epochs 100 \
        --imgsz 640 \
        --batch 16 \
        --project runs/pose \
        --name ski_pose_v1
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    data:    str,
    model:   str  = "yolo26s-pose.pt",
    epochs:  int  = 100,
    imgsz:   int  = 640,
    batch:   int  = 16,
    device:  str  = "",
    project: str  = "runs/pose",
    name:    str  = "ski_pose",
    patience: int = 20,
    workers: int  = 8,
    resume:  bool = False,
    freeze:  int  = 0,
):
    """
    Fine-tune YOLO26-pose.

    Parameters
    ----------
    data    : path to dataset YAML
    model   : YOLO26 pose weights to start from
    epochs  : total training epochs
    imgsz   : training image size
    batch   : batch size (-1 = auto)
    device  : "" = auto, "cpu", "0", "0,1", …
    freeze  : number of backbone layers to freeze (0 = train all)
    """
    net = YOLO(model)

    net.train(
        data      = data,
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        device    = device or ("0" if _cuda() else "cpu"),
        project   = project,
        name      = name,
        patience  = patience,
        workers   = workers,
        resume    = resume,
        freeze    = freeze if freeze > 0 else None,
        # Augmentations well-suited for outdoor snowy scenes
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 5.0,     # small rotation (skiers lean but not much)
        translate = 0.1,
        scale     = 0.5,
        flipud    = 0.0,     # never flip upside-down
        fliplr    = 0.5,
        mosaic    = 1.0,
        close_mosaic = 10,
        pose      = 12.0,    # pose loss weight (higher = more emphasis on kpts)
        kobj      = 2.0,     # keypoint objectness loss weight
    )

    print(f"\nBest weights saved to: {project}/{name}/weights/best.pt")


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _build_parser():
    p = argparse.ArgumentParser(
        description="Fine-tune YOLO26-pose on Ski-2DPose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",    required=True, help="Dataset YAML path")
    p.add_argument("--model",   default="yolo26s-pose.pt")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--device",  default="")
    p.add_argument("--project", default="runs/pose")
    p.add_argument("--name",    default="ski_pose")
    p.add_argument("--patience",type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume",  action="store_true")
    p.add_argument("--freeze",  type=int, default=0,
                   help="Freeze first N backbone layers")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(
        data     = args.data,
        model    = args.model,
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = args.device,
        project  = args.project,
        name     = args.name,
        patience = args.patience,
        workers  = args.workers,
        resume   = args.resume,
        freeze   = args.freeze,
    )