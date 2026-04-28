"""Smoke tests for IKKA components."""

import numpy as np

from ikka import ikka_weight, extremality, transversality, persistence_M


def test_weight_in_unit_interval():
    grads = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    history = np.array([[t, 0.05 + 0.01 * np.sin(t)] for t in range(20)])
    w = ikka_weight(
        error=0.1,
        psr=0.6,
        csi_var=0.05,
        grad_probs=grads,
        error_history=history,
    )
    assert 0.0 <= w <= 1.0


def test_extremality_monotonic():
    e_low = extremality(error=0.0, psr=1.0, csi_var=0.0)
    e_high = extremality(error=1.0, psr=0.1, csi_var=0.5)
    assert e_high > e_low


def test_transversality_aligned_is_zero():
    aligned = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert transversality(aligned) < 1e-6


def test_transversality_orthogonal_is_one():
    ortho = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert abs(transversality(ortho) - 1.0) < 1e-6


def test_persistence_short_history():
    short = np.array([[0, 0.1], [1, 0.2]])
    assert persistence_M(short) == 1.0
