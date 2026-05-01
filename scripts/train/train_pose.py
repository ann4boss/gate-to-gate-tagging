"""
train_pose.py
-------------
Fine-tunes YOLO26s-pose on the EPFL Ski-2DPose dataset for 24-keypoint
ski pose estimation.

Starting from yolo26s-pose.pt (pretrained on COCO 17 keypoints) and
fine-tuning on the 24-keypoint Ski-2DPose annotations.

Note on keypoint count mismatch:
    COCO pretrained weights have a head for 17 keypoints.
    Ski-2DPose needs 24.  Ultralytics handles this automatically — when
    kpt_shape in dataset.yaml differs from the pretrained model it replaces
    the detection head with a randomly initialised one and keeps the backbone.
    This means the backbone's rich feature extraction is preserved while the
    new keypoints are learned from scratch.  A slightly lower LR helps here.

Usage
-----
    python train_pose.py --data yolo_epfl_pose/dataset.yaml

After training, best weights are at:
    runs/pose/skier_pose_epfl/weights/best.pt

Use in PoseEstimator:
    est = PoseEstimator(model_path="runs/pose/skier_pose_epfl/weights/best.pt")
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    p = argparse.ArgumentParser(
        description="Fine-tune YOLO26s-pose on EPFL Ski-2DPose (24 keypoints)"
    )
    p.add_argument("--data",     required=True,
                   help="Path to dataset.yaml from prepare_epfl_pose.py")
    p.add_argument("--model",    default="yolo26s-pose.pt",
                   help="Base weights (default: yolo26s-pose.pt)")
    p.add_argument("--epochs",   type=int,   default=80,
                   help="Training epochs (default: 80)")
    p.add_argument("--imgsz",    type=int,   default=384,
                   help="Input size — must match HRNet convention 384x288. "
                        "Ultralytics uses square crops so 384 is a good match.")
    p.add_argument("--batch",    type=int,   default=16,
                   help="Batch size (reduce to 8 if out of VRAM)")
    p.add_argument("--device",   default="",
                   help="cpu | cuda | mps (empty = auto)")
    p.add_argument("--project",  default="runs/pose")
    p.add_argument("--name",     default="skier_pose_epfl")
    p.add_argument("--patience", type=int,   default=20,
                   help="Early stopping patience (default: 20)")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found: {data_path}\n"
            "Run prepare_epfl_pose.py first."
        )

    print(f"[INFO] Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Starting pose fine-tuning for {args.epochs} epochs …")
    print(f"[INFO] Image size: {args.imgsz}  Batch: {args.batch}")

    results = model.train(
        data     = str(data_path),
        task     = "pose",
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = args.device or ("cuda" if _cuda_available() else "cpu"),
        project  = args.project,
        name     = args.name,
        patience = args.patience,

        # ── Optimiser ─────────────────────────────────────────────────────
        # Slightly lower LR than detection because the backbone is pretrained
        # and we don't want to overwrite its features aggressively.
        optimizer    = "AdamW",
        lr0          = 5e-4,
        lrf          = 0.01,
        weight_decay = 1e-4,
        warmup_epochs = 5,

        # ── Augmentation ──────────────────────────────────────────────────
        # Pose estimation is more sensitive to augmentation than detection —
        # too aggressive and keypoint positions become inconsistent.
        hsv_h    = 0.010,
        hsv_s    = 0.4,
        hsv_v    = 0.3,
        degrees  = 10.0,    # small rotation — skiers tilt but rarely past ~10°
        translate = 0.1,
        scale    = 0.35,
        fliplr   = 0.5,     # horizontal flip is fine (flip_idx in yaml handles kpt swap)
        flipud   = 0.0,     # never flip vertically
        mosaic   = 0.3,     # lower mosaic than detection — avoids cropping keypoints
        mixup    = 0.0,     # mixup not suitable for pose

        # ── Misc ──────────────────────────────────────────────────────────
        save_period = 10,
        val         = True,
        plots       = True,
        verbose     = True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[DONE] Pose training complete.")
    print(f"       Best weights → {best_weights}")
    print(f"\n       Update PoseEstimator to use fine-tuned weights:")
    print(f"       est = PoseEstimator(model_path='{best_weights}')")


if __name__ == "__main__":
    main()