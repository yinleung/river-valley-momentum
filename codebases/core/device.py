"""Device + determinism policy for GPU/MPS/CPU runs (plan_v5 §3.1 B2), written once.

Every GPU driver calls, in order:

    device = resolve_device(config.get("device", "auto"))     # cuda -> mps -> cpu
    seed_all(config["seed"])                                   # python/numpy/torch(+cuda)
    det = setup_determinism()                                  # policy dict -> run config

Precision policy (B2): training steps run under `autocast_ctx(device)` (bf16 on CUDA);
probes (gradient windows, HVP, large-batch probes, spectra) run in fp32 with TF32 matmul
disabled — `setup_determinism` turns TF32 off globally, and training re-enables nothing:
speed on A100 comes from the explicit bf16 autocast, so fp32 paths stay bit-honest.

Where exact determinism is impossible (some cuDNN backward kernels),
`torch.use_deterministic_algorithms(warn_only=True)` keeps best-effort determinism without
crashing; the standing discipline still applies — quote numbers from cached arrays.npz,
never from re-runs.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["resolve_device", "seed_all", "setup_determinism", "autocast_ctx", "device_info"]


def resolve_device(want: str = "auto") -> torch.device:
    """cuda -> mps -> cpu resolution; explicit `want` (e.g. "cuda:1") passes through."""
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    """Seed python, numpy's global RNG, and torch (all CUDA devices included)."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_determinism(strict: bool = False) -> dict:
    """Apply the B2 determinism settings; returns what was set, for the run config.

    strict=False (default) uses warn_only deterministic algorithms — cuDNN paths without
    deterministic kernels warn instead of raising. strict=True raises on any
    nondeterministic kernel (use in smoke tests to enumerate offenders).
    """
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = {
        "deterministic_algorithms": True,
        "deterministic_strict": strict,
        "cudnn_benchmark": False,
        "tf32": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }
    if torch.cuda.is_available() and not policy["cublas_workspace_config"]:
        # Reproducible cuBLAS needs this env var set BEFORE the first kernel launch; the
        # pjsub templates export :4096:8. Flag loudly if a driver forgot.
        print("WARNING core.device: CUBLAS_WORKSPACE_CONFIG unset; cuBLAS not reproducible")
    return policy


def autocast_ctx(device: torch.device, enabled: bool = True):
    """bf16 autocast on CUDA training paths; a no-op context on MPS/CPU and for probes."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib

    return contextlib.nullcontext()


def device_info(device: torch.device) -> dict:
    """Device identity for config.json: name, capability, driver-side versions."""
    info = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update(
            gpu_name=p.name,
            capability=f"{p.major}.{p.minor}",
            total_mem_gb=round(p.total_memory / 2**30, 1),
            cuda_built=torch.version.cuda,
        )
    return info
