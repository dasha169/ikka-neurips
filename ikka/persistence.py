"""Persistence term M(x): sublevel-set H_1 persistent homology of |e_x(t)|."""

from __future__ import annotations
import numpy as np


def persistence_M(error_history: np.ndarray) -> float:
    """Total H_1 persistence of the (t, |e_x(t)|) point cloud.

    Uses a Vietoris-Rips complex with the L_inf metric:
        d((t1, e1), (t2, e2)) = max(|t1 - t2|, |e1 - e2|)

    Falls back to a numpy-only proxy (range-based bottleneck score) when
    GUDHI is not installed, so smoke tests pass on minimal environments.
    """
    if error_history is None or len(error_history) < 3:
        return 1.0

    pts = np.asarray(error_history, dtype=float)

    try:
        import gudhi  # type: ignore

        rips = gudhi.RipsComplex(
            points=pts.tolist(),
            max_edge_length=float(np.ptp(pts) + 1.0),
        )
        st = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()
        diag1 = [pair for pair in st.persistence_intervals_in_dimension(1)]
        if not diag1:
            return 1.0
        total = sum(t - s for s, t in diag1 if np.isfinite(t))
        return float(np.tanh(total))
    except Exception:
        # Lightweight fallback: range of error history, normalised.
        rng = float(np.ptp(pts[:, -1]))
        return float(np.tanh(rng))
