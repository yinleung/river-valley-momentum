"""Momentum as a temporal filter: EMA recursion, matrix polar pipelines, transfer function.

Written once and imported by every experiment driver (see ../CODING.md Pillar 2). All conventions
match `discussions/idea_v1.md`:

    EMA momentum          m_t = beta * m_{t-1} + (1 - beta) * g_t,           m_{-1} = 0
    transfer function     H_beta(z) = (1 - beta) / (1 - beta z^{-1})
    magnitude response    |H_beta(omega)| = (1 - beta) / sqrt(1 - 2 beta cos omega + beta^2)

The matrix pipelines are the three Muon-style variants compared in Experiment E5:

    pre-polar    O(m_t)            where m_t = EMA(G)_t      "filter first, then nonlinearize"
    post-polar   mtilde_t = beta * mtilde_{t-1} + (1 - beta) * O(G_t)   "nonlinearize, then filter"
    polar-only   O(G_t)

`O(.)` is the orthogonal polar factor: for G = U S V^T (thin SVD), O(G) = U V^T.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ema_momentum",
    "transfer_magnitude",
    "effective_window",
    "polar_factor",
    "pre_polar_stream",
    "post_polar_stream",
    "polar_only_stream",
]


# --- vector / scalar EMA momentum ------------------------------------------
def ema_momentum(g: np.ndarray, beta: float, m_init: np.ndarray | float = 0.0) -> np.ndarray:
    """EMA momentum buffer for a gradient stream.

    Applies m_t = beta * m_{t-1} + (1 - beta) * g_t along axis 0 with m_{-1} = `m_init`.
    This is the open-loop filter: it does not simulate any optimizer feedback. Closed-loop
    trajectories are produced by the experiment drivers, which recompute g_t at each w_t.

    Args:
        g: stream of shape (T, ...) — leading axis is time.
        beta: EMA coefficient in [0, 1).
        m_init: initial buffer m_{-1}, broadcast to g[0]'s shape (default 0).

    Returns:
        m: buffer stream of the same shape as `g`.
    """
    g = np.asarray(g, dtype=float)
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1); got {beta}")
    m = np.empty_like(g)
    prev = np.broadcast_to(np.asarray(m_init, dtype=float), g.shape[1:]).astype(float)
    for t in range(g.shape[0]):
        prev = beta * prev + (1.0 - beta) * g[t]
        m[t] = prev
    return m


def transfer_magnitude(beta: float, omega: np.ndarray | float) -> np.ndarray:
    """Theoretical EMA magnitude response |H_beta(omega)| = (1-beta)/sqrt(1-2 beta cos w + beta^2).

    At omega = 0 this is 1 (DC pass-through); at omega = pi it is (1-beta)/(1+beta).
    """
    omega = np.asarray(omega, dtype=float)
    denom = np.sqrt(1.0 - 2.0 * beta * np.cos(omega) + beta**2)
    return (1.0 - beta) / denom


def effective_window(beta: float) -> float:
    """Effective averaging length N_eff = (1+beta)/(1-beta) (variance-reduction sample size)."""
    return (1.0 + beta) / (1.0 - beta)


# --- matrix polar factor and Muon-style pipelines --------------------------
def polar_factor(G: np.ndarray) -> np.ndarray:
    """Orthogonal polar factor O(G) = U V^T from the thin SVD G = U S V^T.

    For an m x n matrix this returns the closest semi-orthogonal matrix (columns or rows
    orthonormal, depending on shape) to G in Frobenius norm — Muon's idealized orthogonalized
    update. Zero singular values map to orthonormal directions returned by the SVD.
    """
    G = np.asarray(G, dtype=float)
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    return U @ Vt


def pre_polar_stream(G: np.ndarray, beta: float) -> np.ndarray:
    """Pre-polar pipeline: filter the gradient stream, then orthogonalize.

    M_t = EMA(G)_t, output O(M_t). `G` has shape (T, m, n); returns (T, m, n).
    """
    M = ema_momentum(G, beta)
    return np.stack([polar_factor(M[t]) for t in range(M.shape[0])], axis=0)


def post_polar_stream(G: np.ndarray, beta: float) -> np.ndarray:
    """Post-polar pipeline: orthogonalize each gradient, then filter the orthogonal factors.

    mtilde_t = beta * mtilde_{t-1} + (1 - beta) * O(G_t). Returns the raw filtered buffer
    (not re-orthogonalized), matching the comparison in idea_v1.md / the Muon paper.
    """
    G = np.asarray(G, dtype=float)
    O = np.stack([polar_factor(G[t]) for t in range(G.shape[0])], axis=0)
    return ema_momentum(O, beta)


def polar_only_stream(G: np.ndarray) -> np.ndarray:
    """Polar-only pipeline: O(G_t) with no momentum."""
    G = np.asarray(G, dtype=float)
    return np.stack([polar_factor(G[t]) for t in range(G.shape[0])], axis=0)
