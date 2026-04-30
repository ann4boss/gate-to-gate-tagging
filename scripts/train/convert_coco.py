import json
from pathlib import Path
import argparse

def convert(json_in, images_dir, json_out, width=1920, height=1080):
    with open(json_in) as f:
        data = json.load(f)

    images = []
    annotations = []
    ann_id = 0
    img_id = 0

    for video_id, video_data in data.items():
        for cam_id, cam_data in video_data.items():
            for frame_name, item in cam_data.items():

                file_name = f"{video_id}/{cam_id}/{frame_name}.jpg"
                img_path = Path(images_dir) / file_name

                if not img_path.exists():
                    print(f"[WARN] Missing image: {file_name}")
                    continue

                # Add image entry
                images.append({
                    "id": img_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                })

                kpts = item["annotation"]

                # Convert normalized → pixel coords
                kpts_px = []
                xs, ys = [], []

                for x, y, v in kpts:
                    px = x * width
                    py = y * height
                    kpts_px.extend([px, py, v])

                    if v > 0:
                        xs.append(px)
                        ys.append(py)

                if not xs:
                    continue

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                bbox = [
                    x_min,
                    y_min,
                    x_max - x_min,
                    y_max - y_min,
                ]

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "keypoints": kpts_px,
                    "num_keypoints": sum(1 for _, _, v in kpts if v > 0),
                })

                ann_id += 1
                img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 0,
                "name": "skier",
                "supercategory": "person",
                "keypoints": [f"kpt_{i}" for i in range(len(kpts))],
                "skeleton": []
            }
        ]
    }

    with open(json_out, "w") as f:
        json.dump(coco, f)

    print(f"[DONE] COCO JSON saved → {json_out}")
    print(f"Images: {len(images)}, Annotations: {len(annotations)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", default="coco_annotations.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    convert(args.input, args.images_dir, args.output, args.width, args.height)