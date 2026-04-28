"""Combined IKKA anomaly weight W(x) = E(x) * T(x) * M(x)."""

from __future__ import annotations
import numpy as np

from .extremality import extremality
from .transversality import transversality
from .persistence import persistence_M


def ikka_weight(
    error: float,
    psr: float,
    csi_var: float,
    grad_probs: np.ndarray,
    error_history: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.3,
) -> float:
    """Compute the IKKA anomaly weight at a single frame.

    Parameters
    ----------
    error : float
        Lateral error |e_x| at the current frame.
    psr : float
        Tracker peak-to-sidelobe ratio in [0, 1].
    csi_var : float
        Variance of the optional channel-state-information signal.
    grad_probs : np.ndarray
        (K, 2) array of gradients of class probabilities w.r.t. (x, y).
    error_history : np.ndarray
        Recent (t, |e_x(t)|) pairs for the persistence term M.
    alpha, beta, gamma : float
        Mixing coefficients for E.

    Returns
    -------
    float
        The combined anomaly weight W(x) in [0, 1].
    """
    E = extremality(error, psr, csi_var, alpha, beta, gamma)
    T = transversality(grad_probs)
    M = persistence_M(error_history)
    return float(E * T * M)
