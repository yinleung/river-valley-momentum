"""Closed-loop EMA-momentum SGD on a river-valley landscape (shared by E1-E3, E7, E8).

This is experiment glue, not core analysis: it drives the optimizer that produces the gradient
and momentum streams; the metrics/filters it feeds live in core/. Update rule (idea_v1.md):

    g_t = grad L(w_t) + xi_t                      (xi_t = 0 in the deterministic E1/E2)
    m_t = beta_t * m_{t-1} + (1 - beta_t) * g_t,   m_{-1} = 0
    w_{t+1} = w_t - eta * m_t

beta is constant unless a schedule is given (E7 schedule arms): either a per-step array, or a
state-feedback callable (t, ctx) -> beta_t for adaptive rules that react to the trajectory.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

DIVERGE_RADIUS = 1e8  # |w| beyond this counts as divergence and stops the run


def simulate(landscape, beta, eta, w0, T, rng=None, noise_fn=None, beta_schedule=None,
             forcing_fn=None):
    """Run T closed-loop steps; return the recorded streams.

    Args:
        landscape: object exposing grad(w), river_tangent(w), hill_normal(w), hill_coordinate(w).
        beta, eta: EMA coefficient and learning rate.
        w0: initial point, shape (2,).
        T: number of steps.
        rng: numpy Generator (required iff noise_fn is given).
        noise_fn: callable(rng, shape) -> additive gradient noise xi_t, or None.
        forcing_fn: callable(t) -> deterministic additive gradient term s_t (shape (2,)),
            or None (the E13 forced-disturbance controls).
        beta_schedule: optional override of the constant `beta` — a (T,) array of per-step
            beta_t, or a callable (t, ctx) -> beta_t with ctx = dict(w, g, m, d): the current
            iterate, stochastic gradient, previous buffer m_{t-1}, and hill coordinate d_t
            (state-feedback / adaptive schedules; stateful rules close over their own state
            and must be constructed fresh per trajectory).

    Returns dict of arrays (time on axis 0; entries after a divergence are NaN):
        W (T+1, 2)       iterates w_0..w_T
        G (T, 2)         stochastic gradients g_t actually used in the update
        Gexact (T, 2)    exact gradients grad L(w_t)
        M (T, 2)         momentum buffers m_t
        R (T, 2)         river tangents r(w_t)
        N (T, 2)         hill normals n(w_t)
        d (T,)           hill coordinate (signed distance to the river floor) at w_t
        betas (T,)       the beta_t actually applied at each step
        diverged (bool)  True iff |w| exceeded DIVERGE_RADIUS (run stopped at t_div)
        t_div (int)      step of divergence, or T if none
    """
    adaptive = callable(beta_schedule)
    if beta_schedule is None:
        betas = np.full(T, beta, dtype=float)
    elif adaptive:
        betas = np.full(T, np.nan)  # recorded as the rule fires
    else:
        betas = np.asarray(beta_schedule, dtype=float)
        if betas.shape != (T,):
            raise ValueError(f"beta_schedule must have shape ({T},); got {betas.shape}")
    w = np.asarray(w0, dtype=float).copy()
    m = np.zeros(2)
    W = np.full((T + 1, 2), np.nan)
    G, Gex, M, R, N = (np.full((T, 2), np.nan) for _ in range(5))
    d = np.full(T, np.nan)
    W[0] = w
    diverged, t_div = False, T
    for t in range(T):
        gx = landscape.grad(w)
        g = gx + (noise_fn(rng, gx.shape) if noise_fn is not None else 0.0) \
            + (forcing_fn(t) if forcing_fn is not None else 0.0)
        R[t], N[t] = landscape.river_tangent(w), landscape.hill_normal(w)
        d[t] = landscape.hill_coordinate(w)
        if adaptive:
            betas[t] = beta_schedule(t, dict(w=w, g=g, m=m, d=d[t]))
        m = betas[t] * m + (1.0 - betas[t]) * g
        Gex[t], G[t], M[t] = gx, g, m
        w = w - eta * m
        if not np.all(np.isfinite(w)) or np.max(np.abs(w)) > DIVERGE_RADIUS:
            diverged, t_div = True, t + 1
            break
        W[t + 1] = w
    return {"W": W, "G": G, "Gexact": Gex, "M": M, "R": R, "N": N, "d": d, "betas": betas,
            "diverged": diverged, "t_div": t_div}
