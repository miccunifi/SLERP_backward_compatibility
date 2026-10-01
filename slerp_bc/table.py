"""Build the paper's results table from the CSV written by main.py."""

import numpy as np
import pandas as pd

SUPPORTS = ["T", "I", "I+T"]  # support-set modality used to fit Procrustes: text, image, both
DIRECTIONS = ["i2t", "t2i"]


def _rows(df, dataset, method, support="-"):
    return df[(df.dataset == dataset) & (df.method == method) & (df.support == support)]


def select_alpha(df, dataset, support, direction, k=1):
    """Interpolation weight maximizing Recall@k on `dataset` (smallest alpha on ties)."""
    rows = _rows(df, dataset, "slerp", support).sort_values("alpha")
    return rows.alpha.iloc[int(np.argmax(rows[f"{direction}_r{k}"].values))]


def build_table(df, support_set="cc3m", datasets=("flickr30k", "coco", "nocaps"), k=1):
    """
    Rows: Old model, then for each support modality SVD (alpha=1), +SLERP with the weight
    selected on the support set (alpha_hat, per direction), +SLERP with the per-dataset
    oracle weight (alpha_star), and finally the New model. Values are Recall@k (%).
    """
    columns = pd.MultiIndex.from_product([datasets, [d.upper() for d in DIRECTIONS]])
    table = {}

    def row(fn):
        return [fn(ds, d) for ds in datasets for d in DIRECTIONS]

    table[("Old model", "-")] = row(lambda ds, d: _rows(df, ds, "old")[f"{d}_r{k}"].item())
    for sup in [s for s in SUPPORTS if s in set(df.support)]:
        a_hat = {d: select_alpha(df, support_set, sup, d) for d in DIRECTIONS}

        def at_alpha(ds, d, alpha):
            rows = _rows(df, ds, "slerp", sup)
            return rows[np.isclose(rows.alpha, alpha)][f"{d}_r{k}"].item()

        table[("SVD", sup)] = row(lambda ds, d: at_alpha(ds, d, 1.0))
        table[("+SLERP (α̂)", sup)] = row(lambda ds, d: at_alpha(ds, d, a_hat[d]))
        table[("+SLERP (α★)", sup)] = row(lambda ds, d: _rows(df, ds, "slerp", sup)[f"{d}_r{k}"].max())
    table[("New model", "-")] = row(lambda ds, d: _rows(df, ds, "new")[f"{d}_r{k}"].item())

    table = pd.DataFrame.from_dict(table, orient="index", columns=columns)
    table.index = pd.MultiIndex.from_tuples(table.index, names=["Method", "Sup."])
    return table


def selected_alphas(df, support_set="cc3m"):
    """alpha_hat selected on the support set, per support modality and retrieval direction."""
    sups = [s for s in SUPPORTS if s in set(df.support)]
    return pd.DataFrame({d.upper(): [select_alpha(df, support_set, s, d) for s in sups] for d in DIRECTIONS},
                        index=pd.Index(sups, name="Sup."))


def mark_compatibility(table):
    """Format as strings, appending ✓ / × when backward compatibility (M_new→old > M_old→old) holds / fails."""
    old = table.loc[("Old model", "-")]
    out = table.map(lambda v: f"{v:.2f}")
    for idx in table.index:
        if idx[0] not in ("Old model", "New model"):
            out.loc[idx] = [f"{v:.2f} {'✓' if v > o else '×'}" for v, o in zip(table.loc[idx], old)]
    return out
