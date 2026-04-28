"""Extremality term E(x): error magnitude + tracker confidence drop."""

from __future__ import annotations
import numpy as np


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def extremality(
    error: float,
    psr: float,
    csi_var: float,
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.3,
) -> float:
    """E(x) = sigmoid(alpha * E_t + beta * (1 - PSR) + gamma * Var(CSI)).

    See Eq. (1) in the paper.
    """
    score = alpha * error + beta * (1.0 - psr) + gamma * csi_var
    return float(_sigmoid(score))
