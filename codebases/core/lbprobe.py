"""Large-batch probe gradient at the visited iterate (plan_v5 §3.1 B5; G2), written once.

E12's split, generalized from `nanogpt/adapter/decomp.py`: at probe step t the raw
mini-batch gradient of a probed matrix decomposes as

    g^mb_t = gbar^LB_t + xi^res_t

with gbar^LB the mean gradient over a FIXED large probe batch evaluated at the same iterate
w_t (chunked, fp32, no autocast) and xi^res the sampling residual. `LargeBatchProbe` computes
gbar^LB while saving/restoring every parameter's .grad, so the training update sees exactly
the training gradient. `state_residual_bands` reduces a recorded window to the E12
accounting — every share metric names its denominator explicitly (§3.0 invariant 5):

    share_lb      = Eh(LB) / (Eh(LB) + Eh(xi))          state share of the high band
    cross         = (Eh(mb) - Eh(lb) - Eh(xi)) / Eh(mb)  cross-term fraction (residual check)

LB estimator bias (declared, as in E12): xi^res folds in the LB sampling error, overstating
E|xi|^2 by (1 + batch/lb_batch) and whitening the LB stream by ~batch/lb_batch of the
residual energy — both push AGAINST the state-dominance gate, so the gate is conservative.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

__all__ = ["LargeBatchProbe", "state_residual_bands", "subspace_split"]


class LargeBatchProbe:
    """Fixed-probe-batch mean gradient of designated targets at the current iterate.

    loss_fn(batch) -> scalar mean loss for that chunk, fp32 path (no autocast).
    batches: the FIXED probe chunks (identical every call — "the large-batch gradient at
    the visited iterate", not a fresh sample). targets: named tensors to read gradients of.
    """

    def __init__(self, model: torch.nn.Module, loss_fn: Callable,
                 targets: dict[str, torch.Tensor], batches: Sequence,
                 mode: str = "train"):
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval'; got {mode!r}")
        self.model = model
        self.loss_fn = loss_fn
        self.targets = targets
        self.batches = list(batches)
        self.mode = mode

    def gradient(self) -> dict[str, np.ndarray]:
        """{name: float32 array} mean gradient over the probe chunks.

        The probe is state-neutral: every parameter's .grad AND every module buffer (BN
        running stats, num_batches_tracked) are saved and restored, so the training loop
        continues exactly as if the probe never ran. mode="train" (default) evaluates the
        gradient of the training loss the dynamics actually see (BN on chunk statistics —
        declare chunk size); mode="eval" uses running stats. Stateless-norm models (LN/GN)
        are identical under both.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        saved = [None if p.grad is None else p.grad.detach().clone() for p in params]
        bufs = [(b, b.detach().clone()) for _, b in self.model.named_buffers()]
        was_training = self.model.training
        self.model.train(self.mode == "train")
        acc = {k: torch.zeros_like(t, dtype=torch.float32) for k, t in self.targets.items()}
        for batch in self.batches:
            self.model.zero_grad(set_to_none=False)
            loss = self.loss_fn(batch)
            loss.backward()
            for k, t in self.targets.items():
                acc[k] += t.grad.detach().to(torch.float32)
        out = {k: (v / len(self.batches)).cpu().numpy() for k, v in acc.items()}
        with torch.no_grad():
            for p, s in zip(params, saved):
                if s is None:
                    p.grad = None
                else:
                    p.grad.copy_(s)
            for b, s in bufs:
                b.copy_(s)
        self.model.train(was_training)
        return out


def state_residual_bands(G_mb: np.ndarray, G_lb: np.ndarray, hi_frac: float = 0.6,
                         window: str = "rect", return_specs: bool = False) -> dict:
    """E12's temporal accounting for one window: HFERs, band energies, shares, cross term.

    G_mb, G_lb: (T, ...) streams (time on axis 0). Denominators are explicit in the keys:
    share_lb over Eh(LB)+Eh(xi); cross over Eh(mb). Rectangular window primary (§3.0.3).
    Chunk-bounded via core.spectral_stream (float32 transform precision, declared): the
    float64 windowed_dft path costs ~15 GB per 2048x300k window across the three streams.
    """
    from .spectral_stream import chunked_stream_reductions

    xi = G_mb.astype(np.float32) - G_lb.astype(np.float32)
    T = G_mb.shape[0]
    omega = 2.0 * np.pi * np.fft.rfftfreq(T)
    high = omega >= hi_frac * np.pi
    nz = omega > 0
    spec_key = "spec" if window == "rect" else "spec_hann"
    eh, hfer, energy, specs = {}, {}, {}, {}
    for s, a in (("mb", G_mb), ("lb", G_lb), ("xi", xi)):
        r = chunked_stream_reductions(a, None, 0.0, 0.0, hi_fracs=(hi_frac,))
        spec = r[spec_key]
        eh[s] = float(np.sum(spec[high]))
        hfer[s] = float(np.sum(spec[high]) / (np.sum(spec[nz]) + 1e-300))
        energy[s] = r["energy"]
        specs[s] = r["spec"].astype(np.float32)  # rect spectrum for the figure panels
    out = dict(
        hfer_mb=hfer["mb"], hfer_lb=hfer["lb"], hfer_xi=hfer["xi"],
        eh_mb=eh["mb"], eh_lb=eh["lb"], eh_xi=eh["xi"],
        share_lb=eh["lb"] / (eh["lb"] + eh["xi"] + 1e-300),
        cross=(eh["mb"] - eh["lb"] - eh["xi"]) / (eh["mb"] + 1e-300),
        energy_lb=energy["lb"], energy_xi=energy["xi"],
    )
    if return_specs:
        out.update(spec_mb=specs["mb"], spec_lb=specs["lb"], spec_xi=specs["xi"])
    return out


def subspace_split(X: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a stream into (in-subspace coefficients, out-of-subspace remainder).

    X: (T, *shape); V: (k, *shape) orthonormal directions (Frobenius inner product).
    Returns (c (T, k), r (T, *shape)) with X_t = sum_i c_ti V_i + r_t. E12's geometric
    projection, shape-agnostic.
    """
    T = X.shape[0]
    k = V.shape[0]
    Xf = X.reshape(T, -1).astype(np.float32)
    Vf = V.reshape(k, -1).astype(np.float32)
    c = Xf @ Vf.T
    r = Xf - c @ Vf
    return c.astype(np.float64), r.reshape(X.shape)
