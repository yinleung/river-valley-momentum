"""Integration Contract for the `nanogpt-gpu` task (plan_v5 G4–G6; §3.1 B5).

The GPU-scale config extension of the `nanogpt` task: the SAME vendored upstream
(../../nanogpt/upstream, karpathy/nanoGPT @ 3adf61e, never edited — no second copy) built
at GPT-2 scale (BPE vocab padded to 50304, seq 1024; 20M diagnostic / 124M confirmation
presets in config.yaml) on the FineWeb 10BT GPT-2 shards staged on /work (llm.c format:
256-int32 header [magic 20240520, version 1, ntok], then uint16 tokens). The instrumented
loop is `core.looprunner` — identical probes and optimizer conventions to resnet-cifar.

Contract surface: build_model, data_loader, probe_batches, target_modules, layer_groups,
end_to_end_train (+ ddp_setup for the 8-GPU full-node runs).

Conventions:
  - Training forward under bf16 autocast on CUDA (B2 policy: GPT-scale speed); probes fp32.
  - Batch order = f(seed, rank): per-rank generators seeded seed*1000+rank+7, so paired
    arms at one (seed, world_size) see identical global batch sequences; `batch_size` is
    PER-RANK (global = batch_size x world_size; compare runs only at equal world_size).
  - DDP (world > 1) is for training-throughput runs (B4 calibration, G4 confirmation,
    G5 arms): window/LB/HVP instrumentation asserts world == 1 — mechanism probes run on
    the 20M single-GPU rungs. NOTE for the G4 session: the divergence guard is rank-local;
    before any 124M run where divergence is a live outcome, synchronize the stop decision
    across ranks (allreduce the flag) or ranks will desync at the next collective.
  - LayerNorm only (no BN): probe_mode "train" == "eval"; no BN caveats anywhere.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  (parity with sibling adapters)

_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM = os.path.abspath(os.path.join(_HERE, "..", "..", "nanogpt", "upstream"))
_CODEBASES = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_UPSTREAM, _CODEBASES):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import GPT, GPTConfig  # noqa: E402  (upstream nanoGPT)

from core.device import autocast_ctx, resolve_device  # noqa: E402
from core.looprunner import TaskHooks, run_closed_loop  # noqa: E402

_DEFAULT_DATA = "/work/gs26/s26001/modded-nanogpt/data/fineweb10B"
_MAGIC = 20240520


# --- data: llm.c fineweb shards ---------------------------------------------
class _Shards:
    """Memmapped llm.c uint16 token shards with header-verified lengths."""

    def __init__(self, data_dir: str, split: str):
        pat = "fineweb_train_" if split == "train" else "fineweb_val_"
        paths = sorted(p for p in os.listdir(data_dir) if p.startswith(pat))
        if not paths:
            raise FileNotFoundError(f"no {pat}*.bin under {data_dir}")
        self.maps, self.lens = [], []
        for p in paths:
            full = os.path.join(data_dir, p)
            head = np.fromfile(full, dtype=np.int32, count=3)
            if head[0] != _MAGIC or head[1] != 1:
                raise ValueError(f"bad llm.c header in {full}: {head[:2]}")
            ntok = int(head[2])
            self.maps.append(np.memmap(full, dtype=np.uint16, mode="r",
                                       offset=1024, shape=(ntok,)))
            self.lens.append(ntok)
        self.total = int(sum(self.lens))

    def sample(self, rng: np.random.Generator, B: int, L: int):
        """(x, y) int64 CPU tensors: B random length-L windows, shard-proportional."""
        w = np.asarray(self.lens, dtype=float) - (L + 1)
        w = np.maximum(w, 0.0)
        p = w / w.sum()
        xs = np.empty((B, L), dtype=np.int64)
        ys = np.empty((B, L), dtype=np.int64)
        for b in range(B):
            si = int(rng.choice(len(self.maps), p=p))
            off = int(rng.integers(0, self.lens[si] - L - 1))
            seg = self.maps[si][off:off + L + 1].astype(np.int64)
            xs[b], ys[b] = seg[:-1], seg[1:]
        return torch.from_numpy(xs), torch.from_numpy(ys)


def _rank_world() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def build_model(config: dict) -> torch.nn.Module:
    """Upstream GPT at the configured size on the task device, seeded."""
    torch.manual_seed(config["seed"])
    gptconf = GPTConfig(
        block_size=config["block_size"], vocab_size=config.get("vocab_size", 50304),
        n_layer=config["n_layer"], n_head=config["n_head"], n_embd=config["n_embd"],
        dropout=0.0, bias=False)
    rank, world = _rank_world()
    if world > 1:
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank % 8))}")
    else:
        device = resolve_device(config.get("device", "auto"))
    return GPT(gptconf).to(device)


def ddp_setup(model: torch.nn.Module):
    """Wrap in DistributedDataParallel when torchrun set WORLD_SIZE > 1."""
    rank, world = _rank_world()
    if world == 1:
        return model
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    local = int(os.environ.get("LOCAL_RANK", rank % 8))
    torch.cuda.set_device(local)
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])


def data_loader(config: dict):
    """get_batch(split) -> (x, y) on device; per-rank seeded order (see conventions)."""
    device = next_device(config)
    rank, _ = _rank_world()
    data_dir = config.get("data_dir", _DEFAULT_DATA)
    train = _Shards(data_dir, "train")
    val = _Shards(data_dir, "val")
    B, L = config["batch_size"], config["block_size"]
    rng = np.random.default_rng(config["seed"] * 1000 + rank + 7)
    rng_val = np.random.default_rng(config["seed"] * 1000 + rank + 8)

    def get_batch(split: str = "train"):
        src = train if split == "train" else val
        r = rng if split == "train" else rng_val
        x, y = src.sample(r, B, L)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    get_batch.val_shards = val
    return get_batch


def next_device(config: dict) -> torch.device:
    rank, world = _rank_world()
    if world > 1:
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank % 8))}")
    return resolve_device(config.get("device", "auto"))


def probe_batches(config: dict, seed_offset: int, total: int, chunk: int) -> list:
    """FIXED probe chunks [(x, y), ...] of `total` sequences from the train shards,
    drawn once from a dedicated generator (rank-independent: probes are world==1)."""
    device = next_device(config)
    data_dir = config.get("data_dir", _DEFAULT_DATA)
    train = _Shards(data_dir, "train")
    rng = np.random.default_rng(config["seed"] + seed_offset)
    out = []
    for lo in range(0, total, chunk):
        x, y = train.sample(rng, min(chunk, total - lo), config["block_size"])
        out.append((x.to(device), y.to(device)))
    return out


# --- probed matrices / groups ----------------------------------------------
def target_modules(model: torch.nn.Module, config: dict) -> dict[str, torch.Tensor]:
    """Probed matrices across depth: mlp.c_fc at layers {0, mid, last} + attn.c_attn at
    mid (4 matrices; config["probe_layers"] overrides the mlp trio)."""
    gpt = model.module if hasattr(model, "module") else model
    L = len(gpt.transformer.h)
    layers = config.get("probe_layers", [0, L // 2, L - 1])
    out = {f"h{i}.mlp.c_fc": gpt.transformer.h[i].mlp.c_fc.weight for i in layers}
    out[f"h{L // 2}.attn.c_attn"] = gpt.transformer.h[L // 2].attn.c_attn.weight
    return out


def layer_groups(model: torch.nn.Module) -> dict[str, list[str]]:
    """Parameter-name groups: embed / per-block h<i> (lm_head is tied to wte)."""
    gpt = model.module if hasattr(model, "module") else model
    groups: dict[str, list[str]] = {"embed": []}
    for name, _ in gpt.named_parameters():
        if name.startswith("transformer.h."):
            groups.setdefault("h" + name.split(".")[2], []).append(name)
        else:
            groups["embed"].append(name)
    return groups


# --- evaluation + the instrumented loop -------------------------------------
@torch.no_grad()
def _evaluate(model, data, config: dict) -> tuple[float, float]:
    """Val loss over eval_iters FIXED batches (fresh fixed-seed generator per call)."""
    device = next_device(config)
    rng = np.random.default_rng(config["seed"] + 4000)
    gpt = model.module if hasattr(model, "module") else model
    was = gpt.training
    gpt.eval()
    losses = []
    B, L = config.get("eval_batch", config["batch_size"]), config["block_size"]
    for _ in range(config.get("eval_iters", 20)):
        x, y = data.val_shards.sample(rng, B, L)
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device):
            _, loss = gpt(x, y)
        losses.append(float(loss))
    gpt.train(was)
    v = float(np.mean(losses))
    return v, v


def end_to_end_train(model, data, targets: dict[str, torch.Tensor], config: dict,
                     opt: dict, injector=None, raw_sink=None, step_callback=None) -> dict:
    """The Contract's end-to-end member: core.looprunner under the GPT hooks.

    Training forward under bf16 autocast (CUDA); probe forwards fp32. val_metric = val
    loss (no accuracy notion). Instrumentation requires world_size == 1.
    """
    rank, world = _rank_world()
    if world > 1 and any(config.get(k) for k in
                         ("windows", "lb_batch", "lam_every", "eig_at_mid", "sketch_k")):
        raise AssertionError("mechanism instrumentation is single-GPU only (world==1)")
    device = next_device(config)

    def forward_train(batch):
        x, y = batch
        _, loss = model(x, y)
        return loss

    gpt = model.module if hasattr(model, "module") else model

    def forward_fp32(batch):
        # Probe path: fp32, and MATH attention backend — the fused flash SDPA kernel has
        # no double-backward, which the HVP probes need (upstream model untouched).
        x, y = batch
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            _, loss = gpt(x, y)
        return loss

    hooks = TaskHooks(
        forward_loss=forward_train,
        forward_loss_fp32=forward_fp32,
        next_batch=lambda: data("train"),
        evaluate=lambda: _evaluate(model, data, config),
        probe_batches=lambda off, total, chunk: probe_batches(config, off, total, chunk),
        layer_groups=layer_groups(model),
        autocast=lambda: autocast_ctx(device),
        probe_mode="train",
    )
    return run_closed_loop(model, hooks, targets, config, opt,
                           injector=injector, raw_sink=raw_sink,
                           step_callback=step_callback)
