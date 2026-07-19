"""Integration Contract for the `nanogpt` task (CODING.md Pillar 1).

Wraps karpathy/nanoGPT (in ../upstream, NEVER edited) as a mini-batch character-level
language-modeling task: a small GPT on shakespeare_char, trained with EMA-SGDM or Muon-style
orthogonalized updates on one probed weight matrix. This is the scale-transfer task (E11):
the same probe/metrics/optimizer pipeline as mlp-diagnostic, but a real transformer with
genuine mini-batch gradient noise on a real corpus.

Contract surface:
    build_model(config)                 -> upstream GPT on config["device"] (float32)
    data_loader(config)                 -> get_batch(split) callable yielding (x, y) batches
    val_data_loader(config)             -> same loader (split="val" selects held-out data)
    target_module(model, config)        -> the probed weight matrix (last block's mlp.c_fc)
    end_to_end_train(model, data, target, config, opt) -> losses, val loss, probe-window
        gradient/buffer streams, diverged flag (the Contract's end-to-end member; E11)

Conventions: EMA normalization m = beta*m + (1-beta)*g, w -= lr*m (the paper's momentum
convention); no gradient clipping (a divergence guard stops runs instead, so instability is
observed, not masked); batch order is driven by config["seed"], so arms sharing a seed see
identical data sequences.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch

_UPSTREAM = os.path.join(os.path.dirname(__file__), "..", "upstream")
sys.path.insert(0, os.path.abspath(_UPSTREAM))

from model import GPT, GPTConfig  # noqa: E402  (upstream nanoGPT)

_DATA_DIR = os.path.join(_UPSTREAM, "data", "shakespeare_char")


def _device(config: dict) -> str:
    want = config.get("device", "auto")
    if want != "auto":
        return want
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_model(config: dict) -> GPT:
    """Small upstream GPT (float32) on the task device, seeded."""
    torch.manual_seed(config["seed"])
    gptconf = GPTConfig(
        block_size=config["block_size"], vocab_size=config["vocab_size"],
        n_layer=config["n_layer"], n_head=config["n_head"], n_embd=config["n_embd"],
        dropout=0.0, bias=False)
    return GPT(gptconf).to(_device(config))


def data_loader(config: dict):
    """get_batch(split) -> (x, y) mini-batches from the shakespeare_char bins.

    Batch order is drawn from a generator seeded by config["seed"], so two training arms
    with the same seed consume identical batch sequences. Run
    `python upstream/data/shakespeare_char/prepare.py` once to create the bins.
    """
    device = _device(config)
    rng = np.random.default_rng(config["seed"] + 7)
    bins = {s: np.memmap(os.path.join(_DATA_DIR, f"{s}.bin"), dtype=np.uint16, mode="r")
            for s in ("train", "val")}
    B, L = config["batch_size"], config["block_size"]

    def get_batch(split: str = "train"):
        data = bins[split]
        ix = rng.integers(0, len(data) - L - 1, size=B)
        x = torch.from_numpy(np.stack([data[i:i + L] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + L] for i in ix]).astype(np.int64))
        return x.to(device), y.to(device)

    return get_batch


def val_data_loader(config: dict):
    """The same loader; call with split="val" for held-out batches."""
    return data_loader(config)


def vocab_size() -> int:
    with open(os.path.join(_DATA_DIR, "meta.pkl"), "rb") as f:
        return pickle.load(f)["vocab_size"]


def target_module(model: GPT, config: dict) -> torch.Tensor:
    """The probed weight matrix: the last block's MLP expansion (4*n_embd x n_embd)."""
    return model.transformer.h[-1].mlp.c_fc.weight


def _polar(g: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(g, full_matrices=False)
    return U @ Vt


@torch.no_grad()
def _val_loss(model: GPT, get_batch, iters: int) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch("val")
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def end_to_end_train(model: GPT, data, target: torch.Tensor, config: dict, opt: dict):
    """Closed-loop mini-batch training under the specified optimizer (E11).

    opt = {"kind": "sgdm", "beta": b}
        every parameter follows EMA-SGDM at config["lr"].
    opt = {"kind": "muon", "beta": b, "variant": "pre"|"post"|"polar", "eta_o": lr}
        the target matrix follows the Muon-style orthogonalized update (pre: O(EMA(G));
        post: EMA(O(G)); polar: O(G), beta ignored), all other parameters plain SGD
        (beta = 0) at config["lr"].

    Returns dict: losses (T,) per-step training-batch loss, val_loss (float, at the end,
    over config["eval_iters"] held-out batches), G_win / M_win (probe_len, *target.shape)
    float32 streams of the target's raw mini-batch gradient and the buffer the optimizer
    used, over steps [probe_start, probe_start + probe_len), diverged (bool).
    """
    device = _device(config)
    lr, T = config["lr"], config["steps"]
    kind, beta = opt["kind"], opt["beta"]
    p0, p1 = config["probe_start"], config["probe_start"] + config["probe_len"]
    bufs = {id(p): torch.zeros_like(p) for p in model.parameters()}
    losses, G_win, M_win = [], [], []
    diverged = False
    loss0 = None
    model.train()
    for t in range(T):
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
        in_probe = p0 <= t < p1
        if in_probe:
            G_win.append(target.grad.detach().to("cpu", torch.float32).numpy().copy())
        with torch.no_grad():
            for p in model.parameters():
                if kind == "muon" and p is target:
                    continue
                b_eff = beta if kind == "sgdm" else 0.0
                buf = bufs[id(p)]
                buf.mul_(b_eff).add_(p.grad, alpha=1.0 - b_eff)
                p -= lr * buf
            if kind == "sgdm":
                if in_probe:
                    M_win.append(bufs[id(target)].detach()
                                 .to("cpu", torch.float32).numpy().copy())
            elif kind == "muon":
                buf = bufs[id(target)]
                g_np = target.grad.detach().to("cpu", torch.float32).numpy()
                if opt["variant"] == "pre":
                    buf.mul_(beta).add_(target.grad, alpha=1.0 - beta)
                    m_np = buf.detach().to("cpu", torch.float32).numpy()
                    upd = torch.from_numpy(_polar(m_np)).to(device)
                elif opt["variant"] == "post":
                    og = torch.from_numpy(_polar(g_np)).to(device)
                    buf.mul_(beta).add_(og, alpha=1.0 - beta)
                    m_np = buf.detach().to("cpu", torch.float32).numpy()
                    upd = buf
                else:  # polar-only
                    m_np = _polar(g_np)
                    upd = torch.from_numpy(m_np).to(device)
                if in_probe:
                    M_win.append(m_np.copy())
                target -= opt["eta_o"] * upd
    val = _val_loss(model, data, config["eval_iters"]) if not diverged else float("inf")
    return dict(losses=np.asarray(losses), val_loss=val,
                G_win=np.asarray(G_win, dtype=np.float32),
                M_win=np.asarray(M_win, dtype=np.float32), diverged=diverged)
