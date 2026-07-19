"""Forced-frequency gradient injection + lock-in readout (plan_v5 §3.1 B5; G3), written once.

The network-scale transplant of E13 (`scripts/run_e13_forced.py`): add a deterministic tone
A cos(omega t) v along a fixed direction v (the measured top Hessian eigenvector, from
core.hvp) to the gradient BEFORE the optimizer sees it,

    g~_t = g_t + A cos(omega t + phase) v,

and read out the forced response by lock-in demodulation at exactly omega. Conventions match
E13: the lock-in amplitude of a real series x_t over the scored window is

    amp = fac * |mean_t x_t exp(-i omega t)|,   fac = 2 (1 at omega = pi, coincident pair),

with t the ABSOLUTE step index, so driver and readout agree on phase.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

__all__ = ["ForcedInjector", "lockin", "lockin_amp"]


class ForcedInjector:
    """Adds A cos(omega t + phase) v to designated parameters' gradients in place.

    v: {param_name: unit-direction tensor} (jointly unit Frobenius norm across entries —
    normalized here for safety). Call inject(named_grads, t) after backward, before the
    optimizer update. Cost: one axpy per designated parameter.
    """

    def __init__(self, v: dict[str, torch.Tensor], A: float, omega: float,
                 phase: float = 0.0):
        norm = math.sqrt(sum(float((t * t).sum()) for t in v.values()))
        if norm <= 0:
            raise ValueError("ForcedInjector: zero direction")
        self.v = {k: (t / norm).detach().clone() for k, t in v.items()}
        self.A, self.omega, self.phase = float(A), float(omega), float(phase)

    @torch.no_grad()
    def inject(self, named_params: Sequence[tuple[str, torch.Tensor]], t: int) -> float:
        """Add the tone to .grad of every named parameter present in v; returns the scalar
        coefficient A cos(omega t + phase) actually applied (log it per step)."""
        c = self.A * math.cos(self.omega * t + self.phase)
        for name, p in named_params:
            d = self.v.get(name)
            if d is not None and p.grad is not None:
                p.grad.add_(d.to(p.grad.device, p.grad.dtype), alpha=c)
        return c

    @torch.no_grad()
    def project(self, named_tensors: Sequence[tuple[str, torch.Tensor]]) -> float:
        """<x, v> over the designated parameters (e.g. weights or gradients on v)."""
        s = 0.0
        for name, x in named_tensors:
            d = self.v.get(name)
            if d is not None:
                s += float((x.detach().to(d.device, torch.float32) * d).sum())
        return s


def lockin(x: np.ndarray, omega: float, t0: int = 0) -> complex:
    """Complex lock-in fac * mean(x_t e^{-i omega t}) over t >= t0 (absolute indices)."""
    x = np.asarray(x, dtype=float)
    tail = x[t0:]
    t_idx = np.arange(t0, t0 + len(tail))
    fac = 1.0 if abs(omega - np.pi) < 1e-12 else 2.0
    return complex(fac * np.mean(tail * np.exp(-1j * omega * t_idx)))


def lockin_amp(x: np.ndarray, omega: float, t0: int = 0) -> float:
    """E13's forced amplitude |lockin| — the scored readout at the forcing frequency."""
    return abs(lockin(x, omega, t0))
