"""Integration Contract for the first-party `mlp-diagnostic` task (CODING.md Pillar 1).

A tiny two-layer ReLU MLP trained full-batch by plain gradient descent on an ill-conditioned
synthetic regression. Run at a learning rate near the edge of stability, the sharp curvature
directions oscillate — the real-network analogue of the river-valley hill — so the target layer's
gradient stream carries high temporal frequency. There is no upstream repo (first-party), so this
adapter *is* the model; it still exposes the standard contract so `core.probe` drives it like any
dropped-in task and `core.metrics` analyses the result unchanged.

Contract surface:
    build_model(config)                 -> torch MLP on CPU
    data_loader(config)                 -> (X, y) full-batch tensors
    val_data_loader(config)             -> held-out (X, y) from the same distribution
    target_module(model, config)        -> the named weight tensor to probe (first layer)
    make_step_fn(model, data, target, config) -> callable doing one GD step, returning grad as numpy
    end_to_end_train(model, data, target, config, opt) -> closed-loop training curve + streams
        (the optional end-to-end member of the Integration Contract; used by E7)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def build_model(config: dict) -> nn.Module:
    torch.manual_seed(config["seed"])
    return TwoLayerMLP(config["d_in"], config["d_hidden"], config["d_out"]).double()


def data_loader(config: dict):
    """Full-batch ill-conditioned regression with a non-realizable high-frequency target.

    Anisotropic inputs (log-spaced std -> condition number `cond`) and target
    y = sin(freq * <w, x>) + label noise. The small student cannot fit the high-frequency target
    exactly; trained full-batch at a learning rate near the edge of stability, the sharp curvature
    directions oscillate, so the first-layer gradient stream concentrates at omega = pi.
    """
    g = torch.Generator().manual_seed(config["seed"] + 1)
    n, d = config["n"], config["d_in"]
    scales = torch.logspace(0, -0.5 * np.log10(config["cond"]), d, base=10.0, dtype=torch.double)
    X = torch.randn(n, d, generator=g, dtype=torch.double) * scales
    w = torch.randn(d, config["d_out"], generator=g, dtype=torch.double)
    y = torch.sin(config["freq"] * (X @ w)) + config["label_noise"] * torch.randn(
        n, config["d_out"], generator=g, dtype=torch.double)
    return X, y


def target_module(model: nn.Module, config: dict) -> torch.Tensor:
    """The probed weight matrix: the first-layer weight (d_hidden x d_in)."""
    return model.fc1.weight


def make_step_fn(model, data, target, config):
    """Return step_fn(): one full-batch GD step at config['eta']; returns target.grad as numpy.

    The gradient returned is the one used for the update at that step (the beta=0 stream); the
    momentum filter is applied open-loop in analysis, matching idea_v1.md Task 4.2.
    """
    X, y = data
    eta = config["eta"]
    loss_fn = nn.MSELoss()

    def step_fn():
        model.zero_grad(set_to_none=False)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        g = target.grad.detach().clone().numpy()
        with torch.no_grad():
            for p in model.parameters():
                p -= eta * p.grad
        return g

    return step_fn


def full_loss(model, data) -> float:
    X, y = data
    with torch.no_grad():
        return float(nn.MSELoss()(model(X), y))


def val_data_loader(config):
    """Held-out batch from the same input/label distribution (independent draw)."""
    cfg = dict(config)
    cfg["seed"] = config["seed"] + 101
    return data_loader(cfg)


def _polar(g: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(g, full_matrices=False)
    return U @ Vt


def end_to_end_train(model, data, target, config, opt):
    """Closed-loop training under the specified optimizer (E7; Contract's end-to-end member).

    opt = {"kind": "sgdm", "beta": b}
        every parameter follows EMA-SGDM at config["eta"]:
        m = b*m + (1-b)*grad; p -= eta*m  (b = 0 recovers make_step_fn's plain GD).
    opt = {"kind": "muon", "beta": b, "variant": "pre"|"post"|"polar", "eta_o": lr}
        the target matrix follows the Muon-style orthogonalized update
        (pre: -eta_o * O(EMA(G)); post: -eta_o * EMA(O(G)); polar: -eta_o * O(G), the
        no-momentum polar-only pipeline, for which beta is ignored), all other parameters
        plain GD at config["eta"].

    Returns dict: losses (T,) training loss before each step, G (T, *target.shape) raw
    target gradient stream, M (T, *target.shape) the buffer stream the optimizer used
    (for muon-post, the filtered-polar buffer), val_loss (float, at the end), diverged.
    """
    X, y = data
    eta = config["eta"]
    T = config["T"]
    loss_fn = nn.MSELoss()
    kind, beta = opt["kind"], opt["beta"]
    bufs = {id(p): torch.zeros_like(p) for p in model.parameters()}
    losses, Gs, Ms = [], [], []
    diverged = False
    loss0 = full_loss(model, data)
    for t in range(T):
        losses.append(full_loss(model, data))
        if not np.isfinite(losses[-1]) or losses[-1] > 1e3 * max(loss0, 1.0):
            diverged = True
            losses = losses[:-1]
            break
        model.zero_grad(set_to_none=False)
        loss_fn(model(X), y).backward()
        g_t = target.grad.detach().clone().numpy()
        Gs.append(g_t)
        with torch.no_grad():
            for p in model.parameters():
                if kind == "muon" and p is target:
                    continue
                b_eff = beta if kind == "sgdm" else 0.0
                buf = bufs[id(p)]
                buf.mul_(b_eff).add_(p.grad, alpha=1.0 - b_eff)
                p -= eta * buf
            if kind == "sgdm":
                Ms.append(bufs[id(target)].detach().clone().numpy())
            elif kind == "muon":
                buf = bufs[id(target)]
                if opt["variant"] == "pre":
                    buf.mul_(beta).add_(target.grad, alpha=1.0 - beta)
                    upd = torch.from_numpy(_polar(buf.numpy()))
                    Ms.append(buf.detach().clone().numpy())
                elif opt["variant"] == "post":
                    og = torch.from_numpy(_polar(target.grad.numpy()))
                    buf.mul_(beta).add_(og, alpha=1.0 - beta)
                    upd = buf
                    Ms.append(buf.detach().clone().numpy())
                else:  # polar-only: no buffer, orthogonalize the raw gradient
                    upd = torch.from_numpy(_polar(target.grad.numpy()))
                    Ms.append(upd.detach().clone().numpy())
                target -= opt["eta_o"] * upd
    return dict(losses=np.asarray(losses), G=np.asarray(Gs), M=np.asarray(Ms),
                val_loss=full_loss(model, val_data_loader(config)), diverged=diverged)
