"""Integration Contract for the `resnet-cifar` task (CODING.md Pillar 1; plan_v5 G1–G3).

Wraps kuangliu/pytorch-cifar (../upstream, NEVER edited) as the ResNet-18/CIFAR closed-loop
task: the canonical CIFAR ResNet-18 (3x3 stem, no maxpool) trained by EMA-SGDM (P-thy) or
conventional heavy-ball SGDM (P-prac). The instrumented loop itself lives in
`core.looprunner` (written once, §3.0-invariant probes); this adapter supplies the task
hooks and the CIFAR-specific conventions.

Contract surface:
    build_model(config)                  -> ResNet-18 on device (norm: "bn" | "gn")
    data_loader(config)                  -> get_batch(split) seeded mini-batches
    probe_batches(config, offset, n, bs) -> FIXED probe chunks (LB / eigen / sharpness)
    target_modules(model, config)        -> the 5 probed weight matrices across depth
    layer_groups(model)                  -> parameter-name groups for JL sketches
    end_to_end_train(model, data, targets, config, opt, injector=None, raw_sink=None)
                                         -> core.looprunner.run_closed_loop under the hooks

Conventions (declared once here, cited by every G-experiment):
  - EMA normalization m = beta*m + (1-beta)*g, w -= lr*m (opt kind "ema", protocol P-thy);
    conventional HB m = beta*m + g, w -= lr*m (kind "hb", P-prac; eta_HB = eta*(1-beta)).
  - No gradient clipping; divergence guard per core.looprunner (§3.0.2).
  - Batch order is a function of config["seed"] alone (sampling with replacement); the
    optional augmentation (config["augment"]) draws crop/flip from a dedicated seed+5000
    generator — a fixed per-seed sequence shared across arms. Probe batches come from
    their own generators (offsets: lb 1000, eig 2000, lam 3000), so probe cadence never
    shifts the training draw sequence.
  - Probes run the TRAIN-mode loss on fixed chunks (BN batch statistics; buffers saved and
    restored around every probe — the core.lbprobe pattern). The "gn" arm (GroupNorm
    surgery on the built model, upstream untouched) is the control with no BN caveat.
  - Training runs fp32 (no autocast): CIFAR ResNet-18 is small, and probe and training
    precision then match exactly.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM = os.path.abspath(os.path.join(_HERE, "..", "upstream"))
_DATA = os.path.abspath(os.path.join(_HERE, "..", "data"))
_CODEBASES = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_UPSTREAM, _CODEBASES):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.resnet import ResNet18  # noqa: E402  (upstream pytorch-cifar)

from core.device import resolve_device  # noqa: E402
from core.looprunner import TaskHooks, run_closed_loop  # noqa: E402

_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761)),
}


# --- model ------------------------------------------------------------------
def _swap_bn_to_gn(module: torch.nn.Module, groups: int = 32) -> None:
    """Replace every BatchNorm2d with GroupNorm(min(groups, C), C) in place (BN-free
    control arm, plan G1). Model surgery on the built model — upstream files untouched."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            c = child.num_features
            setattr(module, name, torch.nn.GroupNorm(min(groups, c), c))
        else:
            _swap_bn_to_gn(child, groups)


def build_model(config: dict) -> torch.nn.Module:
    """Upstream CIFAR ResNet-18 on the task device, seeded; norm 'bn' (default) or 'gn'."""
    torch.manual_seed(config["seed"])
    num_classes = 100 if config.get("dataset", "cifar10") == "cifar100" else 10
    model = ResNet18()
    if num_classes != 10:
        model.linear = torch.nn.Linear(model.linear.in_features, num_classes)
    if config.get("norm", "bn") == "gn":
        _swap_bn_to_gn(model)
    return model.to(resolve_device(config.get("device", "auto")))


# --- data -------------------------------------------------------------------
def _load_cifar(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(train_x uint8 (N,3,32,32), train_y, test_x, test_y) from the pickled batches."""
    def batches(paths):
        xs, ys = [], []
        for p in paths:
            with open(p, "rb") as f:
                d = pickle.load(f, encoding="bytes")
            xs.append(d[b"data"].reshape(-1, 3, 32, 32))
            ys.append(np.asarray(d[b"labels" if b"labels" in d else b"fine_labels"]))
        return np.concatenate(xs), np.concatenate(ys)

    if dataset == "cifar10":
        root = os.path.join(_DATA, "cifar-10-batches-py")
        tx, ty = batches([os.path.join(root, f"data_batch_{i}") for i in range(1, 6)])
        vx, vy = batches([os.path.join(root, "test_batch")])
    else:
        root = os.path.join(_DATA, "cifar-100-python")
        tx, ty = batches([os.path.join(root, "train")])
        vx, vy = batches([os.path.join(root, "test")])
    return tx, ty, vx, vy


def _normalize(x_u8: np.ndarray, y: np.ndarray, dataset: str, device):
    mean, std = _STATS[dataset]
    x = torch.from_numpy(x_u8).to(device=device, dtype=torch.float32).div_(255.0)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s, torch.from_numpy(np.ascontiguousarray(y)).to(
        device=device, dtype=torch.long)


def data_loader(config: dict):
    """get_batch(split) -> (x, y) on device; whole dataset resident, seeded draw order."""
    device = resolve_device(config.get("device", "auto"))
    dataset = config.get("dataset", "cifar10")
    tx, ty, vx, vy = _load_cifar(dataset)
    X_tr, Y_tr = _normalize(tx, ty, dataset, device)
    X_va, Y_va = _normalize(vx, vy, dataset, device)
    B = config["batch_size"]
    rng = np.random.default_rng(config["seed"] + 7)
    rng_val = np.random.default_rng(config["seed"] + 8)
    rng_aug = np.random.default_rng(config["seed"] + 5000)
    augment = bool(config.get("augment", False))

    def get_batch(split: str = "train"):
        if split == "train":
            ix = rng.integers(0, X_tr.shape[0], size=B)
            x, y = X_tr[ix], Y_tr[ix]
            if augment:
                x = _augment(x, rng_aug)
            return x, y
        ix = rng_val.integers(0, X_va.shape[0], size=B)
        return X_va[ix], Y_va[ix]

    get_batch.train_size = X_tr.shape[0]
    get_batch.val_tensors = (X_va, Y_va)
    return get_batch


def _augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Pad-4 random crop + hflip, parameters drawn per image from the dedicated rng."""
    B = x.shape[0]
    pad = F.pad(x, (4, 4, 4, 4))
    dx = rng.integers(0, 9, size=B)
    dy = rng.integers(0, 9, size=B)
    flip = rng.random(B) < 0.5
    out = torch.empty_like(x)
    for i in range(B):
        out[i] = pad[i, :, dy[i]:dy[i] + 32, dx[i]:dx[i] + 32]
    if flip.any():
        idx = torch.from_numpy(np.nonzero(flip)[0]).to(x.device)
        out[idx] = out[idx].flip(-1)
    return out


def probe_batches(config: dict, seed_offset: int, total: int, chunk: int) -> list:
    """FIXED probe chunks [(x, y), ...] drawn once from a dedicated generator.

    Never touches the training generator; identical for every call with the same args
    (the fixed-probe-batch convention of core.lbprobe / the abridged sharpness subset).
    """
    device = resolve_device(config.get("device", "auto"))
    dataset = config.get("dataset", "cifar10")
    tx, ty, _, _ = _load_cifar(dataset)
    rng = np.random.default_rng(config["seed"] + seed_offset)
    ix = rng.choice(tx.shape[0], size=total, replace=False)
    x, y = _normalize(tx[ix], ty[ix], dataset, device)
    return [(x[i:i + chunk], y[i:i + chunk]) for i in range(0, total, chunk)]


# --- probed matrices / groups ----------------------------------------------
def target_modules(model: torch.nn.Module, config: dict) -> dict[str, torch.Tensor]:
    """The designated probed weight matrices across depth (raw windows live here).

    layer4.1.conv2 (2.36M params) was dropped from the RAW-window set after the first
    scan run: it alone is ~7x the other four targets combined in capture + reduction
    cost, and its temporal band structure stays measured through the layer4 JL-sketch
    group spectra. Decided BEFORE the G1 grid predeclaration; raw depth coverage is
    conv1 (stem) / layer2 / layer3 / linear + all-block sketches.
    """
    return {
        "conv1": model.conv1.weight,
        "layer2.0.conv1": model.layer2[0].conv1.weight,
        "layer3.0.conv1": model.layer3[0].conv1.weight,
        "linear": model.linear.weight,
    }


def layer_groups(model: torch.nn.Module) -> dict[str, list[str]]:
    """Parameter-name groups for JL sketch combining (stem / layer1..4 / head)."""
    groups: dict[str, list[str]] = {
        "stem": [], "layer1": [], "layer2": [], "layer3": [], "layer4": [], "head": []}
    for name, _ in model.named_parameters():
        if name.startswith("layer"):
            groups[name.split(".")[0]].append(name)
        elif name.startswith("linear"):
            groups["head"].append(name)
        else:
            groups["stem"].append(name)
    return groups


# --- evaluation + the instrumented loop -------------------------------------
@torch.no_grad()
def _evaluate(model, get_batch, chunk: int = 1000) -> tuple[float, float]:
    """Full val-split (CIFAR test) loss and accuracy; eval mode, state-neutral."""
    X, Y = get_batch.val_tensors
    was = model.training
    model.eval()
    tot_loss, tot_correct = 0.0, 0
    for i in range(0, X.shape[0], chunk):
        logits = model(X[i:i + chunk])
        tot_loss += float(F.cross_entropy(logits, Y[i:i + chunk], reduction="sum"))
        tot_correct += int((logits.argmax(1) == Y[i:i + chunk]).sum())
    model.train(was)
    return tot_loss / X.shape[0], tot_correct / X.shape[0]


def end_to_end_train(model, data, targets: dict[str, torch.Tensor], config: dict,
                     opt: dict, injector=None, raw_sink=None, step_callback=None) -> dict:
    """The Contract's end-to-end member: core.looprunner under the CIFAR hooks.

    Returns the run_closed_loop dict (val_metric = test accuracy). fp32 throughout.
    """
    def forward_loss(batch):
        x, y = batch
        return F.cross_entropy(model(x), y)

    hooks = TaskHooks(
        forward_loss=forward_loss,
        forward_loss_fp32=forward_loss,
        next_batch=lambda: data("train"),
        evaluate=lambda: _evaluate(model, data),
        probe_batches=lambda off, total, chunk: probe_batches(config, off, total, chunk),
        layer_groups=layer_groups(model),
        probe_mode="train",
    )
    return run_closed_loop(model, hooks, targets, config, opt,
                           injector=injector, raw_sink=raw_sink,
                           step_callback=step_callback)
