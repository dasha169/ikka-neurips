"""Bounded IBVS yaw-rate command modulated by IKKA weight."""

from __future__ import annotations
import numpy as np


def deadzone(value: float, delta: float) -> float:
    """Symmetric deadzone."""
    if abs(value) <= delta:
        return 0.0
    return value - np.sign(value) * delta


def ibvs_yaw_rate(
    error_x: float,
    weight: float,
    k: float = 2.5,
    delta_x: float = 0.02,
    omega_max: float = 1.2,
) -> float:
    """tau_t = sat_{[-omega_max, omega_max]}(k * dz(e_x, delta_x) * w_t).

    See Eq. (4) in the paper.
    """
    raw = k * deadzone(error_x, delta_x) * weight
    return float(np.clip(raw, -omega_max, omega_max))
