"""
Step 2: Procrustes alignment + SLERP for backward-compatible retrieval.

1. Fit the orthogonal map R from new- to old-model embeddings on the support set (CC3M val),
   using text (T), image (I) or both (I+T) support embeddings.
2. For every alpha, query the *unchanged* old-model gallery with
   q_alpha = SLERP(phi_old(x), phi_new(x) R, alpha) on the support set and on every test set.
3. Save all Recall@{1,5,10} to a CSV and print the paper table (alpha_hat selected on the support set).

Example (CLIP ViT-L/14 -> CLIP ViT-B/32):
    python main.py --new_model l14 --old_model b32
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F

from slerp_bc.method import pad_to, procrustes, slerp
from slerp_bc.retrieval import evaluate
from slerp_bc.table import SUPPORTS, build_table, mark_compatibility, selected_alphas
from slerp_bc.utils import get_device


def load_features(features_dir, dataset, model, device):
    path = os.path.join(features_dir, dataset, f"{model}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run `python extract_features.py --dataset {dataset} "
                                f"--models {model}` first")
    f = torch.load(path)
    return {"image": F.normalize(f["image"].to(device), dim=-1),
            "text": F.normalize(f["text"].to(device), dim=-1),
            "txt2img": f["txt2img"].to(device)}


def fit_alignment(new, old, support):
    """Procrustes map R [d_new, d] fitted on the chosen support modalities."""
    modalities = {"T": ["text"], "I": ["image"], "I+T": ["text", "image"]}[support]
    V = torch.cat([new[m] for m in modalities])
    U = torch.cat([old[m] for m in modalities])
    return procrustes(V, U)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new_model", default="l14", help="upgraded model, encodes the queries")
    parser.add_argument("--old_model", default="b32", help="deployed model, encodes the fixed gallery")
    parser.add_argument("--support", default="cc3m", help="support set for Procrustes and alpha selection")
    parser.add_argument("--datasets", nargs="+", default=["flickr30k", "coco", "nocaps"])
    parser.add_argument("--supports", nargs="+", default=SUPPORTS, choices=SUPPORTS)
    parser.add_argument("--alphas", nargs="+", type=float, default=[i / 10 for i in range(11)])
    parser.add_argument("--features_dir", default="features")
    parser.add_argument("--output", default=None, help="CSV path (default: results/<new>_to_<old>.csv)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args()
    device = args.device or get_device()

    feats = {}
    for ds in dict.fromkeys([args.support] + args.datasets):
        old = load_features(args.features_dir, ds, args.old_model, device)
        new = load_features(args.features_dir, ds, args.new_model, device)
        assert torch.equal(old["txt2img"], new["txt2img"]), f"{ds}: features of the two models do not match"
        # If d_new > d_old, the old space is zero-padded to d_new and R is a square orthogonal matrix.
        d = max(old["image"].shape[1], new["image"].shape[1])
        old["image"], old["text"] = pad_to(old["image"], d), pad_to(old["text"], d)
        feats[ds] = (old, new)

    rows = []
    for ds, (old, new) in feats.items():
        for method, f in [("old", old), ("new", new)]:
            res = evaluate(f["image"], f["text"], f["image"], f["text"], f["txt2img"])
            rows.append({"dataset": ds, "method": method, "support": "-", "alpha": None, **res})

    for sup in args.supports:
        R = fit_alignment(feats[args.support][1], feats[args.support][0], sup)
        for ds, (old, new) in feats.items():
            v_image, v_text = new["image"] @ R, new["text"] @ R
            for alpha in args.alphas:
                q_image, q_text = slerp(old["image"], v_image, alpha), slerp(old["text"], v_text, alpha)
                res = evaluate(q_image, q_text, old["image"], old["text"], old["txt2img"])
                rows.append({"dataset": ds, "method": "slerp", "support": sup, "alpha": alpha, **res})
                print(f"[{sup:>3}] {ds:<10} alpha={alpha:.2f}  "
                      f"I2T R@1 {res['i2t_r1']:.2f}  T2I R@1 {res['t2i_r1']:.2f}")

    output = args.output or os.path.join("results", f"{args.new_model}_to_{args.old_model}.csv")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"\nSaved {output}\n")

    print(f"{args.new_model} -> {args.old_model}  |  Recall@1, alpha_hat selected on {args.support}:")
    print(selected_alphas(df, args.support).to_string(), "\n")
    print(mark_compatibility(build_table(df, args.support, args.datasets)).to_string())


if __name__ == "__main__":
    main()
