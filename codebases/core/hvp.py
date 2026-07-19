"""Hessian-vector products on CUDA/MPS/CPU (plan_v5 §3.1 B5), written once.

Generalizes the working MPS path of `nanogpt/adapter/decomp.py` from one target matrix to an
arbitrary parameter subset, in flat coordinates:

  HVPOperator       (k, D) block Hessian-vector products of the mean loss over fixed probe
                    chunks: one graph per chunk, k cheap second backwards on it.
  power_topk        block power iteration with QR reorthonormalization -> top-k eigenpairs
                    (the layer-restricted estimator; E12's method, verbatim numerics).
  lanczos_topk      Lanczos with full reorthogonalization -> top-k Ritz pairs (the
                    full-model estimator; k=16, m~=48 on a fixed probe batch).
  SharpnessTracker  warm-started power iteration for the cheap lambda_max trace every
                    ~200 steps (G1's EoS overlay data).

All probe math runs in fp32 (no autocast) on the training device; fixed probe batches are
supplied by the caller so the estimator is a deterministic function of (weights, batches).
Sharpness convention: eigenvalues of the Hessian of the MEAN loss over the supplied chunks —
the Cohen et al. EoS protocol; pair it with the paper's beta-shifted threshold 2*T_eff/eta
when overlaying (EMA protocol) or (2+2*beta)/eta_HB (conventional, stated per figure).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

__all__ = ["HVPOperator", "power_topk", "lanczos_topk", "SharpnessTracker",
           "flatten_params", "unflatten_like"]


def flatten_params(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten_like(flat: torch.Tensor, like: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    out, lo = [], 0
    for t in like:
        n = t.numel()
        out.append(flat[lo:lo + n].reshape(t.shape))
        lo += n
    return out


class HVPOperator:
    """Block HVP of the mean loss over fixed batches, restricted to `params`.

    loss_fn(batch) must run the model forward in fp32 (no autocast) and return the scalar
    mean loss for that chunk; `params` are the (leaf) tensors the Hessian is restricted to
    ("all trainable" = full-model). D = sum of numels.
    """

    def __init__(self, loss_fn: Callable, params: Sequence[torch.Tensor], batches: Sequence):
        self.loss_fn = loss_fn
        self.params = list(params)
        self.batches = list(batches)
        self.D = sum(p.numel() for p in self.params)
        self.device = self.params[0].device
        self.n_hvp = 0  # HVP count (cost accounting for B4 calibration)

    def matvec_block(self, V: torch.Tensor) -> torch.Tensor:
        """(k, D) -> (k, D): mean over batches of H v_j, one graph per batch chunk."""
        k = V.shape[0]
        out = torch.zeros_like(V)
        for batch in self.batches:
            loss = self.loss_fn(batch)
            grads = torch.autograd.grad(loss, self.params, create_graph=True)
            flat_g = flatten_params(grads)
            for j in range(k):
                hv = torch.autograd.grad((flat_g * V[j]).sum(), self.params,
                                         retain_graph=j < k - 1, materialize_grads=True)
                out[j] += flatten_params(hv).detach()
                self.n_hvp += 1
        return out / len(self.batches)


def _orthonormalize(V: torch.Tensor) -> torch.Tensor:
    """QR over the leading axis; CPU float64 for small D (decomp.py numerics), else GPU fp32."""
    if V.shape[1] <= 1 << 20:
        Q, _ = torch.linalg.qr(V.T.cpu().to(torch.float64))
        return Q.T.to(V.device, V.dtype)
    Q, _ = torch.linalg.qr(V.T)
    return Q.T.contiguous()


def power_topk(op: HVPOperator, k: int, iters: int, seed: int):
    """Top-k eigenpairs by block power iteration (E12's estimator in flat coordinates).

    Returns (eigvals (k,) desc, eigvecs (k, D) float32 numpy, resid (k,)) with
    resid_i = ||H v_i - lam_i v_i|| / max(|lam_i|, 1e-12).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    V = torch.randn((k, op.D), generator=gen).to(op.device, torch.float32)
    V = _orthonormalize(V)
    for _ in range(iters):
        V = _orthonormalize(op.matvec_block(V))
    HV = op.matvec_block(V)
    lam = (V * HV).sum(dim=1).cpu()
    resid = torch.linalg.norm(HV.cpu() - lam[:, None] * V.cpu(), dim=1) / \
        torch.clamp(lam.abs(), min=1e-12)
    order = torch.argsort(lam, descending=True)
    return (lam[order].numpy().astype(np.float64),
            V[order].detach().to("cpu", torch.float32).numpy(),
            resid[order].numpy().astype(np.float64))


def lanczos_topk(op: HVPOperator, k: int = 16, m: int = 48, seed: int = 0,
                 basis_device: str = "auto", want_vectors: bool = True):
    """Top-k Ritz pairs of the restricted Hessian by m-step Lanczos, full reorthogonalization.

    basis_device "auto" keeps the (m, D) basis on the op device below ~8 GB, else CPU.
    Returns (ritz (k,) desc, vecs (k, D) float32 numpy or None, ritz_resid (k,)) where
    ritz_resid is the standard Lanczos residual |beta_m * s_{m,i}| / max(|ritz_i|, 1e-12).
    """
    dev = op.device if (basis_device == "auto" and m * op.D * 4 < 8e9) or \
        basis_device == str(op.device) else torch.device("cpu")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(op.D, generator=gen).to(op.device, torch.float32)
    q /= torch.linalg.norm(q)
    basis = torch.zeros((m, op.D), dtype=torch.float32, device=dev)
    alphas, betas = [], []
    beta_prev, q_prev = 0.0, None
    for i in range(m):
        basis[i] = q.to(dev)
        w = op.matvec_block(q[None, :])[0]
        alpha = float(w @ q)
        w = w - alpha * q
        if q_prev is not None:
            w = w - beta_prev * q_prev
        for _ in range(2):  # full reorthogonalization against the stored basis, two passes
            coeff = basis[: i + 1] @ w.to(dev)                       # (i+1,)
            w = w - (coeff @ basis[: i + 1]).to(op.device)
        beta = float(torch.linalg.norm(w))
        alphas.append(alpha)
        if beta < 1e-10 * max(1.0, abs(alpha)):  # invariant subspace found
            m = i + 1
            basis = basis[:m]
            break
        betas.append(beta)
        q_prev, q = q, w / beta
        beta_prev = beta
    T = np.diag(np.array(alphas)) + np.diag(np.array(betas[: m - 1]), 1) + \
        np.diag(np.array(betas[: m - 1]), -1)
    evals, evecs = np.linalg.eigh(T)
    order = np.argsort(evals)[::-1][:k]
    ritz = evals[order]
    # Lanczos residual: |beta_m s_{m,i}|, beta_m = the beta linking to q_{m+1} (0 on break)
    last_beta = betas[m - 1] if len(betas) >= m else 0.0
    resid = np.abs(last_beta * evecs[-1, order]) / np.maximum(np.abs(ritz), 1e-12)
    vecs = None
    if want_vectors:
        S = torch.from_numpy(evecs[:, order].astype(np.float32)).to(dev)
        vecs = (S.T @ basis).to("cpu", torch.float32).numpy()
    return ritz.astype(np.float64), vecs, resid.astype(np.float64)


class SharpnessTracker:
    """Warm-started lambda_MAX trace: a few power steps per call on fixed probe batches.

    make_op() must rebuild the HVPOperator at the CURRENT weights with the SAME fixed
    batches each call (declared cadence, e.g. every 200 steps). Warm start reuses the last
    top vector, so iters=3 suffices after the first call (iters0 for the cold start).

    Plain power iteration converges to the largest-|lambda| eigenvalue, which early in
    training can be the NEGATIVE end of the spectrum; the EoS overlay needs the largest
    POSITIVE one. When the converged value is negative, the measurement reruns on the
    shifted operator H + c I (c = 1.5|lambda_big|), whose top eigenpair is lambda_max + c
    — reported shift-corrected. The shifted vector warm-starts the next call as usual.
    """

    def __init__(self, make_op: Callable[[], HVPOperator], iters: int = 3, iters0: int = 20,
                 seed: int = 0):
        self.make_op = make_op
        self.iters, self.iters0, self.seed = iters, iters0, seed
        self.v: torch.Tensor | None = None
        self.history: list[tuple[int, float]] = []

    def _power(self, op: HVPOperator, v: torch.Tensor, iters: int,
               shift: float) -> tuple[float, torch.Tensor]:
        lam = 0.0
        for _ in range(iters):
            w = op.matvec_block(v[None, :])[0] + shift * v
            lam = float(w @ v)
            n = float(torch.linalg.norm(w))
            if n == 0.0:
                break
            v = w / n
        return lam - shift, v

    def measure(self, step: int) -> float:
        op = self.make_op()
        if self.v is None:
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            v = torch.randn(op.D, generator=gen).to(op.device, torch.float32)
        else:
            v = self.v.to(op.device)
        v /= torch.linalg.norm(v)
        iters = self.iters0 if self.v is None else self.iters
        lam, v = self._power(op, v, iters, shift=0.0)
        if lam < 0:
            # Converged to the negative end: shift and rerun for lambda_max, from a FRESH
            # random start (the converged vector is ~orthogonal to the lambda_max one).
            gen = torch.Generator(device="cpu").manual_seed(self.seed + 1 + step)
            v = torch.randn(op.D, generator=gen).to(op.device, torch.float32)
            v /= torch.linalg.norm(v)
            lam, v = self._power(op, v, max(self.iters0, 12), shift=1.5 * abs(lam))
        self.v = v.detach()
        self.history.append((step, lam))
        return lam
