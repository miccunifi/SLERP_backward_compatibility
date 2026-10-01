"""
Step 1: extract image and text embeddings for one dataset and one or more models.

Saves <features_dir>/<dataset>/<model>.pt with
    image:   [N_images, d]  image embeddings
    text:    [N_texts, d]   text embeddings
    txt2img: [N_texts]      index of the image each caption belongs to

Example:
    python extract_features.py --dataset coco --root /path/to/COCO2014 --models b32 l14
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from slerp_bc.data import READERS, ImageDataset
from slerp_bc.models import MODELS, load_model
from slerp_bc.utils import get_device

# Default dataset locations: edit these or pass --root.
DATA_ROOTS = {
    "cc3m":      "/path/to/cc3m",
    "flickr30k": "/path/to/Flickr30K",
    "coco":      "/path/to/COCO2014",
    "nocaps":    "/path/to/nocaps",
}


@torch.no_grad()
def extract(model_key, items, device, batch_size, num_workers):
    model, preprocess, tokenizer = load_model(model_key, device)
    paths = [path for path, _ in items]
    captions = [c for _, caps in items for c in caps]
    txt2img = torch.tensor([i for i, (_, caps) in enumerate(items) for _ in caps])

    loader = DataLoader(ImageDataset(paths, preprocess), batch_size=batch_size, num_workers=num_workers)
    image = torch.cat([model.encode_image(x.to(device)).cpu()
                       for x in tqdm(loader, desc=f"{model_key} images")])
    text = torch.cat([model.encode_text(tokenizer(captions[i:i + batch_size]).to(device)).cpu()
                      for i in tqdm(range(0, len(captions), batch_size), desc=f"{model_key} texts")])
    return {"image": image.float(), "text": text.float(), "txt2img": txt2img}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(READERS))
    parser.add_argument("--root", default=None, help="dataset directory (default: DATA_ROOTS[dataset])")
    parser.add_argument("--models", nargs="+", default=["b32", "l14"], choices=list(MODELS))
    parser.add_argument("--features_dir", default="features")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args()

    device = args.device or get_device()
    root = args.root or DATA_ROOTS[args.dataset]
    items = READERS[args.dataset](root)
    missing = [path for path, _ in items if not os.path.exists(path)]
    if not items or missing:
        raise FileNotFoundError(f"{len(missing)} images not found (e.g. {missing[0]})" if missing
                                else f"no images found in {root}: check --root / DATA_ROOTS")
    print(f"{args.dataset}: {len(items)} images, {sum(len(c) for _, c in items)} captions ({root})")

    out_dir = os.path.join(args.features_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    for model_key in args.models:
        out = os.path.join(out_dir, f"{model_key}.pt")
        if os.path.exists(out):
            print(f"{out} already exists, skipping")
            continue
        torch.save(extract(model_key, items, device, args.batch_size, args.num_workers), out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
