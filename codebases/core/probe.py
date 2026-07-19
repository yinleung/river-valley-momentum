"""Gradient-collection protocols (stationary + trajectory), written once and task-agnostic.

These mirror the probe protocols in `../CODING.md` Pillar 2: identical downstream analysis for every
task. `core/` stays framework-free — adapters supply small callables (a training step that returns
the target gradient, or a fixed-model gradient sampler) and these helpers drive the loop and stack
the results into arrays that `core.metrics` consumes.

  trajectory buffer : gradients G_1..G_T along a live training run (reveals the slow/fast temporal
                      structure the momentum filter acts on);
  stationary buffer : K gradients at a frozen model (isolates the stochastic mini-batch component).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["collect_trajectory", "collect_stationary"]


def collect_trajectory(step_fn: Callable[[], np.ndarray], T: int,
                       progress_every: int = 0) -> np.ndarray:
    """Run T training steps, stacking the target gradient returned by each.

    Args:
        step_fn: callable performing ONE optimizer step and returning that step's target-matrix
            gradient `G_t` as a numpy array (same shape every call).
        T: number of steps.
        progress_every: if >0, print a progress line every this many steps.

    Returns:
        G: array of shape (T, *grad_shape) with time on axis 0.
    """
    buf = []
    for t in range(T):
        g = np.asarray(step_fn(), dtype=float)
        buf.append(g)
        if progress_every and (t + 1) % progress_every == 0:
            print(f"    trajectory probe: step {t + 1}/{T}")
    return np.stack(buf, axis=0)


def collect_stationary(grad_fn: Callable[[int], np.ndarray], K: int) -> np.ndarray:
    """Sample K gradients at a frozen model (the model is not updated between samples).

    Args:
        grad_fn: callable(k) -> target gradient for the k-th fresh mini-batch, model held fixed.
        K: number of samples.

    Returns:
        G: array of shape (K, *grad_shape).
    """
    return np.stack([np.asarray(grad_fn(k), dtype=float) for k in range(K)], axis=0)
