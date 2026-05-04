"""
train_gate_detector.py
----------------------
Fine-tunes YOLO26s on the annotated SwissSki gate pole dataset.

Classes:
    0  gate_contact  — the pole the skier hits (GS inner pole or SL panel)
    1  gate_outer    — the outer rigid pole of a GS gate (never in SL)

Usage
-----
    python train_gate_detector.py --data yolo_gate_poles/dataset.yaml

After training, best weights are at:
    runs/detect/gate_poles/weights/best.pt

Use in GateDetector:
    det = GateDetector(model_path="runs/detect/gate_poles/weights/best.pt")
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
        description="Fine-tune YOLO26s for gate pole detection"
    )
    p.add_argument("--data",      required=True,
                   help="Path to dataset.yaml from convert_labelstudio_to_yolo.py")
    p.add_argument("--model",     default="yolo26s.pt",
                   help="Base weights (default: yolo26s.pt)")
    p.add_argument("--epochs",    type=int,   default=80,
                   help="Training epochs (default: 80)")
    p.add_argument("--imgsz",     type=int,   default=640)
    p.add_argument("--batch",     type=int,   default=16,
                   help="Batch size (reduce to 8 if out of VRAM)")
    p.add_argument("--device",    default="")
    p.add_argument("--project",   default="runs/detect")
    p.add_argument("--name",      default="gate_poles")
    p.add_argument("--patience",  type=int,   default=20)
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found: {data_path}\n"
            "Run convert_labelstudio_to_yolo.py first."
        )

    print(f"[INFO] Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Starting gate pole training for {args.epochs} epochs ...")
    model.train(
        data        = str(data_path),
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = args.device or ("cuda" if _cuda_available() else "cpu"),
        project     = args.project,
        name        = args.name,
        patience    = args.patience,

        # ── Optimiser ─────────────────────────────────────────────────────
        optimizer    = "AdamW",
        lr0          = 1e-3,
        lrf          = 0.01,
        weight_decay = 1e-4,
        warmup_epochs = 3,

        # ── Augmentation ──────────────────────────────────────────────────
        # Gate poles are thin vertical objects — avoid augmentations that
        # make them horizontal or distort their aspect ratio too much.
        hsv_h     = 0.010,    # small hue shift — pole colours matter
        hsv_s     = 0.4,
        hsv_v     = 0.4,      # brightness varies a lot in broadcast footage
        degrees   = 5.0,      # very small rotation — poles are nearly vertical
        translate = 0.1,
        scale     = 0.5,      # poles vary a lot in size across the course
        fliplr    = 0.5,
        flipud    = 0.0,
        mosaic    = 0.5,
        mixup     = 0.1,

        # ── Misc ──────────────────────────────────────────────────────────
        save_period = 10,
        val         = True,
        plots       = True,
        verbose     = True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[DONE] Gate detector training complete.")
    print(f"       Best weights → {best}")
    print(f"\n       Use in GateDetector:")
    print(f"       det = GateDetector(model_path='{best}')")


if __name__ == "__main__":
    main()