"""Orthogonal Procrustes alignment + spherical interpolation (SLERP)."""

import torch
import torch.nn.functional as F


def pad_to(x, dim):
    """Zero-pad the last dimension of `x` up to `dim` (no-op if already `dim`-dimensional)."""
    return F.pad(x, (0, dim - x.shape[-1]))


def procrustes(V_new, U_old):
    """
    Closed-form (rectangular) orthogonal Procrustes: argmin_R ||V_new R - U_old||_F s.t. R orthogonal.

    Args:
        V_new: new-model support embeddings, [N, d_new] (L2-normalized).
        U_old: old-model support embeddings, [N, d] (L2-normalized, zero-padded to d = max(d_new, d_old)).
    Returns:
        R: alignment map, [d_new, d].
    """
    P, _, Qt = torch.linalg.svd(V_new.T @ U_old, full_matrices=False)
    return P @ Qt


def slerp(u, v, alpha, eps=1e-7):
    """
    Spherical linear interpolation between old-model queries u and aligned new-model queries v.

    Args:
        u: old-model queries, [B, d].
        v: aligned new-model queries, [B, d].
        alpha: interpolation weight (0 = old model, 1 = aligned new model).
    Returns:
        q: interpolated unit-norm queries, [B, d].
    """
    u, v = F.normalize(u, dim=-1), F.normalize(v, dim=-1)
    cos = (u * v).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cos)
    q = (torch.sin((1 - alpha) * theta) * u + torch.sin(alpha * theta) * v) / torch.sin(theta)
    return F.normalize(q, dim=-1)
