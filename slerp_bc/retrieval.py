"""Image-text retrieval with Recall@K."""

import torch


@torch.no_grad()
def recall_at_k(queries, gallery, query_ids, gallery_ids, ks=(1, 5, 10), chunk=1024):
    """
    Recall@K (%) for cosine-similarity retrieval on L2-normalized embeddings.

    A query is a hit at K if any of its top-K gallery items has the same image id.

    Args:
        queries:     [Nq, d] query embeddings.
        gallery:     [Ng, d] gallery embeddings.
        query_ids:   [Nq] image id of each query.
        gallery_ids: [Ng] image id of each gallery item.
    """
    hits = []
    for i in range(0, len(queries), chunk):
        topk = (queries[i:i + chunk] @ gallery.T).topk(max(ks), dim=1).indices
        hits.append(gallery_ids[topk] == query_ids[i:i + chunk, None])
    hits = torch.cat(hits)
    return {k: hits[:, :k].any(dim=1).float().mean().item() * 100 for k in ks}


def evaluate(q_images, q_texts, g_images, g_texts, txt2img, ks=(1, 5, 10)):
    """Image-to-text (I2T) and text-to-image (T2I) Recall@K; returns {'i2t_r1': ..., 't2i_r10': ...}."""
    img_ids = torch.arange(len(g_images), device=txt2img.device)
    i2t = recall_at_k(q_images, g_texts, img_ids, txt2img, ks)
    t2i = recall_at_k(q_texts, g_images, txt2img, img_ids, ks)
    return {**{f"i2t_r{k}": v for k, v in i2t.items()}, **{f"t2i_r{k}": v for k, v in t2i.items()}}
