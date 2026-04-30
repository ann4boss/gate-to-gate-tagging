"""
train_skier_detector.py
-----------------------
Fine-tunes YOLO26s on the EPFL Ski-2DPose dataset for skier detection.

The EPFL dataset has ~1982 images.  Fine-tuning from COCO pretrained weights
(yolo26s.pt) rather than training from scratch means we need far fewer epochs
and get much better results on a small dataset.

Usage
-----
    python train_skier_detector.py --data yolo_epfl/dataset.yaml

After training, the best weights are saved to:
    runs/detect/skier_epfl/weights/best.pt

Pass these to SkierDetector:
    det = SkierDetector(model_path="runs/detect/skier_epfl/weights/best.pt")
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(
        description="Fine-tune YOLO26s for skier detection on EPFL dataset"
    )
    p.add_argument("--data",       required=True,
                   help="Path to dataset.yaml produced by prepare_epfl.py")
    p.add_argument("--model",      default="yolo26s.pt",
                   help="Base weights to fine-tune from (default: yolo26s.pt)")
    p.add_argument("--epochs",     type=int,   default=50,
                   help="Number of training epochs (default: 50)")
    p.add_argument("--imgsz",      type=int,   default=640,
                   help="Input image size (default: 640)")
    p.add_argument("--batch",      type=int,   default=16,
                   help="Batch size — reduce to 8 if you run out of VRAM")
    p.add_argument("--device",     default="",
                   help="cpu | cuda | mps (empty = auto)")
    p.add_argument("--project",    default="runs/detect",
                   help="Output folder root")
    p.add_argument("--name",       default="skier_epfl",
                   help="Run name (subfolder inside --project)")
    p.add_argument("--patience",   type=int,   default=15,
                   help="Early stopping patience in epochs (default: 15)")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found: {data_path}\n"
            "Run prepare_epfl.py first."
        )

    print(f"[INFO] Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Starting fine-tuning for {args.epochs} epochs …")
    results = model.train(
        data       = str(data_path),
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        device     = args.device or ("cuda" if _cuda_available() else "cpu"),
        project    = args.project,
        name       = args.name,
        patience   = args.patience,   # early stopping

        # ── Optimiser ─────────────────────────────────────────────────────
        optimizer  = "AdamW",
        lr0        = 1e-3,            # initial LR
        lrf        = 0.01,            # final LR = lr0 * lrf  (cosine decay)
        weight_decay = 1e-4,
        warmup_epochs = 3,

        # ── Augmentation ──────────────────────────────────────────────────
        # Ski footage has specific properties: snowy backgrounds, fast motion,
        # TV broadcast framing.  We keep augmentation moderate so we don't
        # train the model to be invariant to things that actually matter
        # (e.g. colour — red/blue gate poles are real signal).
        hsv_h      = 0.010,           # small hue shift
        hsv_s      = 0.5,
        hsv_v      = 0.3,
        degrees    = 5.0,             # small rotation — camera is fairly stable
        translate  = 0.1,
        scale      = 0.4,             # skier size varies a lot in broadcast
        fliplr     = 0.5,             # horizontal flip is fine for detection
        flipud     = 0.0,             # vertical flip never happens in real footage
        mosaic     = 0.5,             # mosaic augmentation — useful on small datasets
        mixup      = 0.1,

        # ── Misc ──────────────────────────────────────────────────────────
        save_period = 10,             # save checkpoint every N epochs
        val         = True,
        plots       = True,           # save training curves
        verbose     = True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[DONE] Training complete.")
    print(f"       Best weights → {best_weights}")
    print(f"\n       Update SkierDetector to use fine-tuned weights:")
    print(f"       det = SkierDetector(model_path='{best_weights}')")
    print(f"\n       Or pass to gate_tagger.py:")
    print(f"       python gate_tagger.py --video race.mp4 --det-model {best_weights}")


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
