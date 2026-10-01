"""
Readers for the image-text datasets used in the paper.

Each reader returns a list of (image_path, [captions]) pairs, one entry per image:
    cc3m       CC3M validation split   (support set, 12,637 images / 1 caption each)
    flickr30k  Flickr30k, all Karpathy splits (31,014 images / 155,070 captions)
    coco       COCO 2014 Karpathy val + restval (35,504 images / 177,644 captions)
    nocaps     NoCaps validation split (4,500 images / 45,000 captions)
"""

import csv
import json
import os

from PIL import Image
from torch.utils.data import Dataset

# The 12,637 CC3M validation pairs used in the paper (image_path relative to the CC3M root, caption, url).
CC3M_VAL = os.path.join(os.path.dirname(__file__), "..", "assets", "cc3m_val.csv")


def read_cc3m(root):
    """Pairs listed in assets/cc3m_val.csv, images in <root>/val/."""
    with open(CC3M_VAL, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # CC3M downloads are often incomplete: keep only the images available on disk.
    items = [(os.path.join(root, r["image_path"]), [r["caption"]]) for r in rows]
    return [it for it in items if os.path.exists(it[0])]


def read_flickr30k(root):
    """<root>/dataset_flickr30k.json (Karpathy splits) and <root>/images/."""
    with open(os.path.join(root, "dataset_flickr30k.json")) as f:
        images = json.load(f)["images"]
    return [(os.path.join(root, "images", im["filename"]), [s["raw"] for s in im["sentences"]])
            for im in images]


def read_coco(root):
    """<root>/dataset_coco.json (Karpathy splits) and <root>/val2014/."""
    with open(os.path.join(root, "dataset_coco.json")) as f:
        images = json.load(f)["images"]
    return [(os.path.join(root, im["filepath"], im["filename"]), [s["raw"] for s in im["sentences"]])
            for im in images if im["split"] in ("val", "restval")]


def read_nocaps(root):
    """<root>/nocaps_val_4500_captions.json and <root>/val/."""
    with open(os.path.join(root, "nocaps_val_4500_captions.json")) as f:
        data = json.load(f)
    captions = {}
    for ann in data["annotations"]:
        captions.setdefault(ann["image_id"], []).append(ann["caption"])
    return [(os.path.join(root, "val", im["file_name"]), captions[im["id"]]) for im in data["images"]]


READERS = {"cc3m": read_cc3m, "flickr30k": read_flickr30k, "coco": read_coco, "nocaps": read_nocaps}


class ImageDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.preprocess(Image.open(self.paths[i]).convert("RGB"))
