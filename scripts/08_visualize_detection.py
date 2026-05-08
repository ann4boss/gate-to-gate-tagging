# visualize_detections.py
# Usage: python visualize_detections.py <run_id> [--every N] [--start F] [--end F]

import argparse, os, math, cv2
import numpy as np
from ultralytics import YOLO

FRAMES_BASE  = "data/frames/swsk"
POSE_WEIGHTS = "runs/pose/skier_pose/weights/best.pt"
GATE_WEIGHTS = "runs/detect/gate_poles/weights/best.pt"
POSE_CONF    = 0.35
GATE_CONF    = 0.25
MIN_KP_CONF  = 0.20

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),          # head
    (5,6),(5,7),(7,9),(6,8),(8,10),   # arms
    (5,11),(6,12),(11,12),            # torso
    (11,13),(13,15),(12,14),(14,16),  # legs
]

def draw_pose(img, keypoints):
    """Draw keypoints and skeleton on frame."""
    pts = {}
    for i, (x, y, c) in enumerate(keypoints):
        if c >= MIN_KP_CONF:
            pts[i] = (int(x), int(y))
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], (0, 200, 255), 2)

def draw_gates(img, detections):
    """Draw gate pole bounding boxes."""
    colors = {"red": (0,0,255), "blue": (255,100,0)}
    for (x1, y1, x2, y2, cls_name, conf) in detections:
        col = colors.get(cls_name, (255, 255, 0))
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), col, 2)
        cv2.putText(img, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--every",  type=int, default=10,  help="sample every N frames")
    p.add_argument("--start",  type=int, default=0,   help="start frame index")
    p.add_argument("--end",    type=int, default=None, help="end frame index")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out",    default="outputs/viz", help="output directory")
    p.add_argument("--show",   action="store_true",   help="display with cv2.imshow")
    args = p.parse_args()

    frame_dir = os.path.join(FRAMES_BASE, args.run_id)
    out_dir   = os.path.join(args.out, args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    frames = frames[args.start : args.end]
    frames = frames[::args.every]
    print(f"Visualizing {len(frames)} frames (every {args.every}) ...")

    pose_model = YOLO(POSE_WEIGHTS)
    gate_model = YOLO(GATE_WEIGHTS)

    for i, fname in enumerate(frames):
        img = cv2.imread(os.path.join(frame_dir, fname))
        if img is None:
            continue

        # ── Pose ──────────────────────────────────────────────────────
        res_pose = pose_model(img, task="pose", conf=POSE_CONF,
                              device=args.device, verbose=False)
        if res_pose and res_pose[0].keypoints is not None and len(res_pose[0].boxes.conf):
            best = int(res_pose[0].boxes.conf.argmax())
            kpts = res_pose[0].keypoints.data[best].cpu().numpy()
            draw_pose(img, [(float(x),float(y),float(c)) for x,y,c in kpts])
            # draw skier bbox
            x1,y1,x2,y2 = res_pose[0].boxes.xyxy[best].cpu().numpy()
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cx = int((x1+x2)/2)
            cv2.line(img,(cx,int(y1)),(cx,int(y2)),(0,255,100),1)  # vertical centre line

        # ── Gates ─────────────────────────────────────────────────────
        res_gate = gate_model(img, task="detect", conf=GATE_CONF,
                              device=args.device, verbose=False)
        raw_dets = []
        if res_gate and res_gate[0].boxes is not None and len(res_gate[0].boxes):
            names = res_gate[0].names
            for box in res_gate[0].boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                raw_dets.append((x1,y1,x2,y2,
                                 names[int(box.cls[0])],
                                 float(box.conf[0])))
        draw_gates(img, raw_dets)

        # ── Overlay info ──────────────────────────────────────────────
        frame_num = args.start + (i * args.every)
        cv2.putText(img, f"frame {frame_num}  gates={len(raw_dets)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        out_path = os.path.join(out_dir, f"viz_{frame_num:05d}.jpg")
        cv2.imwrite(out_path, img)

        if args.show:
            cv2.imshow("detections", img)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(frames)}", flush=True)

    cv2.destroyAllWindows()
    print(f"\nSaved to {out_dir}/")

if __name__ == "__main__":
    main()