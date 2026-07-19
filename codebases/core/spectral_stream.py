"""Streaming spectral instrumentation at scale (plan_v5 §3.1 B5), written once.

Three pieces, composable by any adapter/driver:

  JLSketch      fixed Gaussian Johnson-Lindenstrauss projection of per-parameter gradients
                to k dims, generated chunkwise on the fly from per-parameter seeds — the
                (d x k) matrix is never materialized. Per-parameter sketches from one pass
                add up to any grouping: group sketch = sum over its params, full model =
                sum over all (independent per-parameter projections make concatenation
                exact). Energies are preserved in expectation: E|sketch|^2 = |x|^2.
  WindowBuffer  preallocated raw-stream recorder for the designated probed matrices
                (fp16 by default), saved to results/raw/<run-id>/ on /work — the cache
                keeps only reductions.
  StreamStats   running scalar statistics of a stream (energy, mean, lag-1 autocovariance)
                for always-on cheap diagnostics outside the windows.

Determinism scope: JLSketch is reproducible given (seed, parameter names, shapes, device
*type*) — CUDA and CPU generators realize different matrices from the same seed, so sketches
are comparable across arms on one platform (all campaign runs: A100), not across platforms.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

__all__ = ["JLSketch", "WindowBuffer", "StreamStats"]


class JLSketch:
    """Fixed Gaussian sketch x -> (R^T x)/sqrt(k) of named tensors, chunk-generated.

    Usage:
        jl = JLSketch(k=128, seed=1234, device=device)
        per_param = jl.sketch_named(model.named_parameters(), lambda p: p.grad)
        vec_full  = jl.combine(per_param)                       # full-model sketch
        vec_grp   = jl.combine(per_param, names=group_names)    # any layer group

    The per-parameter projection is seeded by crc32(name) ^ seed, so it is independent of
    the run seed, identical across arms, and stable under parameter-order changes.
    """

    def __init__(self, k: int = 128, seed: int = 990731, device: torch.device | str = "cpu",
                 chunk: int = 1 << 22):
        self.k = int(k)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.chunk = int(chunk)

    def _param_seed(self, name: str) -> int:
        return (zlib.crc32(name.encode()) ^ self.seed) & 0x7FFFFFFF

    @torch.no_grad()
    def sketch_named(self, named: Iterable[tuple[str, torch.Tensor]],
                     pick=lambda p: p) -> dict[str, np.ndarray]:
        """Per-parameter k-sketches of pick(param) (default the tensor itself).

        Skips parameters where pick() returns None (e.g. frozen params without grads).
        Returns {name: float32 (k,)}.
        """
        out: dict[str, np.ndarray] = {}
        for name, p in named:
            t = pick(p)
            if t is None:
                continue
            flat = t.detach().reshape(-1).to(self.device, torch.float32)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(self._param_seed(name))
            acc = torch.zeros(self.k, device=self.device, dtype=torch.float32)
            for lo in range(0, flat.numel(), self.chunk):
                seg = flat[lo:lo + self.chunk]
                R = torch.randn((seg.numel(), self.k), generator=gen,
                                device=self.device, dtype=torch.float32)
                acc += seg @ R
            out[name] = (acc / np.sqrt(self.k)).cpu().numpy()
        return out

    @staticmethod
    def combine(per_param: dict[str, np.ndarray],
                names: Iterable[str] | None = None) -> np.ndarray:
        """Sum of per-parameter sketches = sketch of their concatenation."""
        keys = list(per_param) if names is None else [n for n in names if n in per_param]
        if not keys:
            raise ValueError("no parameter sketches selected")
        return np.sum([per_param[n] for n in keys], axis=0)


class WindowBuffer:
    """Preallocated (length, *shape) recorder for one probed matrix's raw stream."""

    def __init__(self, shape: tuple, length: int, dtype=np.float16):
        self.buf = np.zeros((length, *shape), dtype=dtype)
        self.n = 0

    def push(self, t: torch.Tensor) -> None:
        if self.n >= self.buf.shape[0]:
            raise IndexError(f"WindowBuffer full at {self.n}")
        self.buf[self.n] = t.detach().to("cpu", torch.float32).numpy()
        self.n += 1

    @property
    def full(self) -> bool:
        return self.n == self.buf.shape[0]

    def array(self) -> np.ndarray:
        return self.buf[: self.n]

    def save(self, path: str | Path, name: str = "stream") -> Path:
        """Write the (possibly partial) window to <path>.npz (raw layer, /work only)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{name: self.array()})
        return path


class StreamStats:
    """Running energy / mean / lag-1 autocovariance of a stream of tensors.

    Cheap always-on diagnostics (no storage): after the run,
        energy       sum_t |x_t|^2
        mean_sq      |mean_t x_t|^2   (DC power proxy)
        lag1         sum_t <x_t, x_{t-1}> / sqrt(energy_t * energy_{t-1})-ish, reported as
                     the raw inner-product sum alongside energy for the driver to normalize.
    """

    def __init__(self):
        self.n = 0
        self.energy = 0.0
        self.lag1_dot = 0.0
        self._sum: torch.Tensor | None = None
        self._prev: torch.Tensor | None = None

    @torch.no_grad()
    def push(self, t: torch.Tensor) -> None:
        t = t.detach().to(torch.float32)
        self.energy += float((t * t).sum())
        if self._prev is not None:
            self.lag1_dot += float((t * self._prev).sum())
        self._sum = t.clone() if self._sum is None else self._sum + t
        self._prev = t.clone()
        self.n += 1

    def summary(self) -> dict:
        out = dict(n=self.n, energy=self.energy, lag1_dot=self.lag1_dot)
        if self.n:
            m = self._sum / self.n
            out["mean_sq"] = float((m * m).sum())
            out["lag1_rho"] = self.lag1_dot / self.energy if self.energy > 0 else 0.0
        return out
