"""
Download the images of the datasets distributed as URLs.

    cc3m    the 12,637 CC3M validation images listed in assets/cc3m_val.csv -> <root>/val/
            (resized to shorter side 256, similar to the img2dataset download used for the paper)
    nocaps  the 4,500 NoCaps validation images from Open Images -> <root>/val/
            (expects <root>/nocaps_val_4500_captions.json)

Some CC3M URLs may no longer be available: missing images are skipped by extract_features.py.

Example:
    python scripts/download_images.py --dataset cc3m --root /path/to/cc3m
"""

import argparse
import csv
import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

CC3M_VAL = os.path.join(os.path.dirname(__file__), "..", "assets", "cc3m_val.csv")
OPEN_IMAGES_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"


def download(url, dest, shorter_side=None):
    if os.path.exists(dest):
        return True
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(request, timeout=30).read()
        if shorter_side is None:
            open(dest, "wb").write(data)
        else:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            scale = shorter_side / min(img.size)
            img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC).save(dest, quality=95)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["cc3m", "nocaps"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if args.dataset == "cc3m":
        with open(CC3M_VAL, encoding="utf-8") as f:
            jobs = [(r["url"], os.path.join(args.root, r["image_path"]), 256) for r in csv.DictReader(f)]
    else:
        with open(os.path.join(args.root, "nocaps_val_4500_captions.json")) as f:
            images = json.load(f)["images"]
        jobs = [(OPEN_IMAGES_URL.format(im["open_images_id"]), os.path.join(args.root, "val", im["file_name"]), None)
                for im in images]

    os.makedirs(os.path.join(args.root, "val"), exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        ok = sum(pool.map(lambda job: download(*job), jobs))
    print(f"{ok}/{len(jobs)} images available in {os.path.join(args.root, 'val')}")


if __name__ == "__main__":
    main()
