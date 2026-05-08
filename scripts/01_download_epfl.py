"""
01_download_epfl.py
-------------------
Download the EPFL Ski-2DPose dataset (images + labels).

Usage:
    python 01_download_epfl.py              # WebP images (86 MB, default)
    python 01_download_epfl.py --jpg        # JPG images (352 MB)
    python 01_download_epfl.py --png        # PNG images (1.67 GB)

Output:
    data/frames/epfl/Images_webp/<video_id>/<split_id>/<img_id>.webp
    data/annotations/epfl/ski2dpose_labels.json
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://datasets-cvlab.epfl.ch/2019-ski-2d-pose"
URLS = {
    "webp":   (f"{BASE}/ski2dpose_images_webp.zip",  "86 MB"),
    "jpg":    (f"{BASE}/ski2dpose_images_jpg.zip",   "352 MB"),
    "png":    (f"{BASE}/ski2dpose_images_png.zip",   "1.67 GB"),
    "labels": (f"{BASE}/ski2dpose_labels.json.zip",  "40 KB"),
}

FRAMES_DIR = Path("data/frames/epfl")
ANN_DIR    = Path("data/annotations/epfl")


def download(url: str, dest: Path, size_hint: str) -> None:
    print(f"  Downloading {dest.name}  ({size_hint})")
    def _progress(count, block, total):
        if total > 0:
            pct  = min(100.0, count * block * 100 / total)
            fill = int(pct / 2)
            print(f"\r  [{'█'*fill}{'░'*(50-fill)}] {pct:5.1f}%", end="", flush=True)
        else:
            print(f"\r  {count*block/1e6:.1f} MB", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def extract(zip_path: Path, out_dir: Path) -> None:
    print(f"  Extracting {zip_path.name} ...", end=" ", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    print("done")


def main():
    p = argparse.ArgumentParser()
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument("--webp", action="store_true", default=True)
    fmt.add_argument("--jpg",  action="store_true", default=False)
    fmt.add_argument("--png",  action="store_true", default=False)
    args = p.parse_args()

    img_fmt = "png" if args.png else "jpg" if args.jpg else "webp"

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== EPFL Ski-2DPose Download ({img_fmt.upper()}) ===\n")

    # Labels
    labels_dest = ANN_DIR / "ski2dpose_labels.json"
    if labels_dest.exists():
        print(f"  Labels already present — skipping")
    else:
        url, size = URLS["labels"]
        zip_dest = ANN_DIR / "labels.zip"
        download(url, zip_dest, size)
        extract(zip_dest, ANN_DIR)

    # Images
    img_dir  = FRAMES_DIR / f"Images_{img_fmt}"
    zip_dest = FRAMES_DIR / f"Images_{img_fmt}.zip"
    if img_dir.exists() and any(img_dir.iterdir()):
        print(f"  Images already present — skipping")
    else:
        url, size = URLS[img_fmt]
        download(url, zip_dest, size)
        extract(zip_dest, FRAMES_DIR)

    print(f"\nDone.\n  Frames : {FRAMES_DIR}/Images_{img_fmt}/")
    print(f"  Labels : {ANN_DIR}/ski2dpose_labels.json\n")


if __name__ == "__main__":
    main()