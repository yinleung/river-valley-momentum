"""Decomposition probe for the `nanogpt` task (E12): state gradient vs sampling residual.

The E11 claim under test: the raw mini-batch gradient stream of the probed matrix is
hill-dominated at every stable learning rate. HFER alone measures *high-frequency*, not
*hill*: the review's objection is that high-band energy could be sampling noise rather than
landscape-driven oscillation. This probe separates the two along both axes the theory names:

  temporal   g_t^{mb} = gbar(w_t) + xi_t, where gbar(w) is the mean-loss gradient at the
             visited iterate (estimated by a large-batch probe at the same iterate) and
             xi_t the sampling residual. A temporally independent residual has a flat
             spectrum, so any HFER excess over the white baseline must sit in gbar(w_t) --
             the state stream that the closed-loop theory (paper Thm spectrum) describes.
  geometric  top-k Hessian eigenpairs of the mean loss restricted to the probed matrix,
             estimated at the probe-window midpoint: the hill directions. Projecting the
             state stream onto this subspace tests whether its high-band energy is
             curvature-aligned (hill) rather than diffuse.

Protocol (per run: one lr, one seed, plain SGD beta = 0 so the stream is the raw one):
  - train exactly as contract.end_to_end_train's SGDM path at beta = 0 (same rngs, so the
    batch sequence matches the E11 arm at the same seed);
  - at each probe step t, record the target's mini-batch gradient AND a large-batch
    gradient at the same iterate w_t (lb_batch sequences, chunked; drawn from a separate
    rng so the training stream is untouched; all parameter grads are saved and restored
    around the probe so the update uses exactly the training-batch gradient);
  - at the window midpoint, before that step's update, run block power iteration with
    Hessian-vector products for the top-k eigenpairs of the target-restricted Hessian on a
    fixed eigen-batch (separate rng; MPS if double-backward works there, else a CPU copy).

Returned streams are float32 CPU arrays; all reductions (HFER, band energies, projections)
happen in the driver so the cached record stays small.
"""
from __future__ import annotations

import numpy as np
import torch

import contract


def _device(config: dict) -> str:
    return contract._device(config)


def _lb_gradient(model, target, get_lb, n_chunks: int) -> torch.Tensor:
    """Mean gradient of the target over n_chunks large-batch chunks at the current iterate."""
    acc = torch.zeros_like(target)
    for _ in range(n_chunks):
        x, y = get_lb("train")
        model.zero_grad(set_to_none=False)
        _, loss = model(x, y)
        loss.backward()
        acc += target.grad.detach()
    return acc / n_chunks


def _hvp_block(model, target, batches, V: torch.Tensor) -> torch.Tensor:
    """(H V_j)_j for the target-restricted Hessian of the mean loss over `batches`.

    One graph build per chunk, k cheap second backwards on it -- the block-power inner loop.
    """
    k = V.shape[0]
    out = torch.zeros_like(V)
    for x, y in batches:
        _, loss = model(x, y)
        (g,) = torch.autograd.grad(loss, target, create_graph=True)
        for j in range(k):
            (hv,) = torch.autograd.grad((g * V[j]).sum(), target,
                                        retain_graph=j < k - 1)
            out[j] += hv.detach()
    return out / len(batches)


def _orthonormalize(V: torch.Tensor) -> torch.Tensor:
    """QR orthonormalization in the Frobenius inner product over the leading axis."""
    flat = V.reshape(V.shape[0], -1)
    Q, _ = torch.linalg.qr(flat.T.cpu().to(torch.float64))
    return Q.T.to(V.device, V.dtype).reshape(V.shape)


def top_hessian_eigenpairs(model, target, batches, k: int, iters: int, seed: int):
    """Top-k eigenpairs of the target-restricted Hessian by block power iteration.

    Returns (eigvals (k,), eigvecs (k, *target.shape) float32 numpy, resid (k,)) with
    eigvecs orthonormal in the Frobenius inner product and resid_i = ||H v_i - lam_i v_i||_F
    / max(|lam_i|, 1e-12) as the convergence diagnostic. Power iteration converges to the
    top eigenvalues in |.|; for a mid-training loss these are the large positive ones.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    V = torch.randn((k, *target.shape), generator=gen).to(target.device, target.dtype)
    V = _orthonormalize(V)
    for _ in range(iters):
        V = _orthonormalize(_hvp_block(model, target, batches, V))
    HV = _hvp_block(model, target, batches, V)
    lam = torch.tensor([(V[j] * HV[j]).sum() for j in range(k)])
    resid = torch.tensor(
        [torch.linalg.norm(HV[j].cpu() - lam[j] * V[j].cpu())
         / max(abs(float(lam[j])), 1e-12) for j in range(k)])
    order = torch.argsort(lam, descending=True)
    return (lam[order].numpy().astype(np.float64),
            V[order].detach().to("cpu", torch.float32).numpy(),
            resid[order].numpy().astype(np.float64))


def _eigen_at_checkpoint(model, target, get_eig, config) -> dict:
    """Top-k eigenpairs at the current iterate, trying the training device first."""
    n_chunks = config["eig_batch"] // config["eig_chunk"]
    batches = [get_eig("train") for _ in range(n_chunks)]
    try:
        lam, vecs, resid = top_hessian_eigenpairs(
            model, target, batches, config["eig_k"], config["eig_iters"],
            seed=config["seed"] + 3000)
        dev = str(target.device)
    except (RuntimeError, NotImplementedError):
        # double backward unsupported on this device -> CPU copy of model and batches
        cpu_model = contract.build_model({**config, "device": "cpu"})
        cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
        cpu_model.train()
        cpu_target = contract.target_module(cpu_model, config)
        cpu_batches = [(x.cpu(), y.cpu()) for x, y in batches]
        lam, vecs, resid = top_hessian_eigenpairs(
            cpu_model, cpu_target, cpu_batches, config["eig_k"], config["eig_iters"],
            seed=config["seed"] + 3000)
        dev = "cpu"
    model.zero_grad(set_to_none=True)
    return dict(eigvals=lam, eigvecs=vecs, eig_resid=resid,
                eig_step=np.array([config["probe_start"] + config["probe_len"] // 2]),
                eig_device=dev)


def run_decomp(config: dict, opt_beta: float = 0.0) -> dict:
    """Plain-SGD training with the decomposition probe (E12).

    config adds to the contract config:
        lb_batch     large-batch size for the mean-gradient probe (multiple of eig_chunk)
        eig_k        number of Hessian eigenpairs
        eig_iters    power-iteration count
        eig_batch    eigen-batch size (multiple of eig_chunk)
        eig_chunk    chunk size for lb/eigen batches (sequences per forward)

    Returns dict with losses, G_mb / G_lb (probe_len, *target.shape) float32, eigvals,
    eigvecs, eig_resid, eig_step, eig_device, diverged.
    """
    lr, T = config["lr"], config["steps"]
    p0, p1 = config["probe_start"], config["probe_start"] + config["probe_len"]
    mid = p0 + config["probe_len"] // 2
    chunk = config["eig_chunk"]

    model = contract.build_model(config)
    data = contract.data_loader(config)
    target = contract.target_module(model, config)

    # separate loaders so probe draws never touch the training batch sequence
    get_lb = contract.data_loader({**config, "seed": config["seed"] + 1000,
                                   "batch_size": chunk})
    n_lb_chunks = config["lb_batch"] // chunk
    get_eig = contract.data_loader({**config, "seed": config["seed"] + 2000,
                                    "batch_size": chunk})

    bufs = {id(p): torch.zeros_like(p) for p in model.parameters()}
    losses, G_mb, G_lb = [], [], []
    eig = None
    diverged = False
    loss0 = None
    model.train()
    for t in range(T):
        if t == mid and eig is None:
            eig = _eigen_at_checkpoint(model, target, get_eig, config)
        x, y = data("train")
        _, loss = model(x, y)
        lval = loss.item()
        loss0 = lval if loss0 is None else loss0
        losses.append(lval)
        if not np.isfinite(lval) or lval > 3.0 * max(loss0, 1.0):
            diverged = True
            losses.pop()
            break
        model.zero_grad(set_to_none=False)
        loss.backward()
        if p0 <= t < p1:
            saved = {id(p): p.grad.detach().clone() for p in model.parameters()}
            G_mb.append(saved[id(target)].to("cpu", torch.float32).numpy().copy())
            G_lb.append(_lb_gradient(model, target, get_lb, n_lb_chunks)
                        .to("cpu", torch.float32).numpy().copy())
            with torch.no_grad():
                for p in model.parameters():
                    p.grad.copy_(saved[id(p)])
        with torch.no_grad():
            for p in model.parameters():
                buf = bufs[id(p)]
                buf.mul_(opt_beta).add_(p.grad, alpha=1.0 - opt_beta)
                p -= lr * buf
    return dict(losses=np.asarray(losses),
                G_mb=np.asarray(G_mb, dtype=np.float32),
                G_lb=np.asarray(G_lb, dtype=np.float32),
                diverged=diverged, **(eig or {}))
