"""
scripts/download_ski2dpose.py

Download the EPFL Ski-2DPose dataset.
https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/

Usage:
    python scripts/download_ski2dpose.py           # downloads WebP images (smallest, 86 MB)
    python scripts/download_ski2dpose.py --jpg      # JPG images (352 MB)
    python scripts/download_ski2dpose.py --png      # PNG images (1.67 GB)
    python scripts/download_ski2dpose.py --videos   # also download source videos (519 MB)

Output layout:
    data/
    ├── frames/
    │   └── epfl/
    │       └── Images_webp/   (or Images_jpg / Images_png)
    │           └── <video_id>/<split_id>/<img_id>.webp
    └── annotations/
        └── epfl/
            └── ski2dpose_labels.json
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path


# ─── Direct download URLs ─────────────────────────────────────────────────────
BASE = "https://datasets-cvlab.epfl.ch/2019-ski-2d-pose"

URLS = {
    "webp":   (f"{BASE}/ski2dpose_images_webp.zip",  "86 MB"),
    "jpg":    (f"{BASE}/ski2dpose_images_jpg.zip",   "352 MB"),
    "png":    (f"{BASE}/ski2dpose_images_png.zip",   "1.67 GB"),
    "labels": (f"{BASE}/ski2dpose_labels.json.zip", "40 KB")
}


FRAMES_DIR = Path("data/frames/epfl")
ANN_DIR    = Path("data/annotations/epfl")


# ─── Download helper ──────────────────────────────────────────────────────────

def download(url: str, dest: Path, size_hint: str) -> None:
    print(f"  → {dest.name}  ({size_hint})")

    def _progress(count, block, total):
        if total > 0:
            pct  = min(100.0, count * block * 100 / total)
            fill = int(pct / 2)
            print(f"\r     [{'█' * fill}{'░' * (50 - fill)}] {pct:5.1f}%", end="", flush=True)
        else:
            print(f"\r     {count * block / 1e6:.1f} MB downloaded", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()   # newline after progress bar


def extract(zip_path: Path, out_dir: Path) -> None:
    print(f"  Extracting {zip_path.name} …", end=" ", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    print("done")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Download EPFL Ski-2DPose dataset")
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument("--webp",   action="store_true", default=True,  help="WebP images — 86 MB  (default)")
    fmt.add_argument("--jpg",    action="store_true", default=False, help="JPG images  — 352 MB")
    fmt.add_argument("--png",    action="store_true", default=False, help="PNG images  — 1.67 GB")
    p.add_argument("--videos",   action="store_true", help="Also download source videos (519 MB)")
    args = p.parse_args()

    # Resolve image format
    if args.png:
        img_fmt = "png"
    elif args.jpg:
        img_fmt = "jpg"
    else:
        img_fmt = "webp"

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  EPFL Ski-2DPose Download")
    print(f"  Images : {img_fmt.upper()}  ({URLS[img_fmt][1]})")
    print(f"  Labels : {URLS['labels'][1]}")
    print("=" * 55)

    # 1. Labels JSON
    print("\n[1/2] Labels")
    labels_dest = ANN_DIR / "ski2dpose_labels.json"
    if labels_dest.exists():
        print(f"  Already exists — skipping ({labels_dest})")
    else:
        url, size = URLS["labels"]
        download(url, labels_dest, size)

    # 2. Images zip
    print(f"\n[2/2] Images ({img_fmt.upper()})")
    zip_dest = FRAMES_DIR / f"Images_{img_fmt}.zip"
    img_dir  = FRAMES_DIR / f"Images_{img_fmt}"
    if img_dir.exists() and any(img_dir.iterdir()):
        print(f"  Already exists — skipping ({img_dir})")
    else:
        url, size = URLS[img_fmt]
        download(url, zip_dest, size)
        extract(zip_dest, FRAMES_DIR)

    # 3. Videos (optional)
    if args.videos:
        print(f"\n[3/3] Videos")
        vid_zip  = FRAMES_DIR / "Videos.zip"
        vid_dir  = FRAMES_DIR / "Videos"
        if vid_dir.exists() and any(vid_dir.iterdir()):
            print(f"  Already exists — skipping ({vid_dir})")
        else:
            url, size = URLS["videos"]
            download(url, vid_zip, size)
            extract(vid_zip, FRAMES_DIR)

    print("\n" + "=" * 55)
    print("  Done!")
    print(f"  Frames  : {FRAMES_DIR}/Images_{img_fmt}/")
    print(f"  Labels  : {ANN_DIR}/ski2dpose_labels.json")
    print("=" * 55)


if __name__ == "__main__":
    main()