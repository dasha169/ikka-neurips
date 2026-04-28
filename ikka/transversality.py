"""Transversality term T(x): gradient non-alignment across class probabilities."""

from __future__ import annotations
import numpy as np


def transversality(grad_probs: np.ndarray, eps: float = 1e-9) -> float:
    """T(x) = prod_{i<j} (1 - |<g_i, g_j>| / (|g_i| |g_j|)).

    See Eq. (3) in the paper.

    Parameters
    ----------
    grad_probs : np.ndarray
        (K, d) array of class-probability gradients at the current point.
    """
    K = grad_probs.shape[0]
    if K < 2:
        return 1.0

    norms = np.linalg.norm(grad_probs, axis=1) + eps
    out = 1.0
    for i in range(K):
        for j in range(i + 1, K):
            cos_ij = abs(np.dot(grad_probs[i], grad_probs[j])) / (norms[i] * norms[j])
            out *= (1.0 - cos_ij)
    return float(np.clip(out, 0.0, 1.0))
