"""Phase-0 smoke job (plan_v5 §3.1 gate): device + determinism + full-instrumentation
micro-runs on both tasks + the B3 acceptance record.

One share-debug GPU job:
  1. device identity (name/driver/capability) + determinism policy applied;
  2. resnet-cifar micro-run exercising EVERY instrumentation path (window, JL sketch,
     sharpness tracker, LB probe, midpoint Lanczos eigenpairs) on CUDA;
  3. nanogpt-gpu micro-run (bf16 autocast train + fp32 probes) likewise;
  4. determinism verification: the resnet micro-run repeated at the same seed must
     reproduce its loss trace (report max |delta|; CUBLAS_WORKSPACE_CONFIG required);
  5. one log_run per task micro-run -> full cache record (config.json/metrics.json/
     arrays.npz + runs.csv row with git SHA + stage flag) — the B3 acceptance criterion.

Run (inside a pjsub job):  cd codebases && python scripts/wisteria/smoke_gpu.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch

_CODEBASES = pathlib.Path(__file__).resolve().parents[2]  # scripts/wisteria/ -> codebases/
sys.path.insert(0, str(_CODEBASES))

from core.device import (device_info, resolve_device, seed_all,  # noqa: E402
                         setup_determinism)
from core.logging import git_sha, log_run  # noqa: E402

sys.path.insert(0, str(_CODEBASES / "resnet-cifar" / "adapter"))
import contract as resnet  # noqa: E402

# the nanogpt-gpu adapter module name collides with resnet's; load via importlib
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "contract_nanogpt_gpu", _CODEBASES / "nanogpt-gpu" / "adapter" / "contract.py")
gpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpt)


RESNET_CFG = dict(seed=0, dataset="cifar10", batch_size=128, steps=60, lr=0.1,
                  device="auto", eval_every=0, windows=[[16, 24]], sketch_k=32,
                  lam_every=25, lam_batch=256, lam_chunk=256, lb_batch=512, lb_chunk=256,
                  lb_every=8, eig_at_mid=True, eig_k=4, eig_m=12, eig_batch=512,
                  eig_chunk=256, stage="explore")
GPT_CFG = dict(seed=0, n_layer=6, n_head=6, n_embd=384, block_size=1024, vocab_size=50304,
               batch_size=8, steps=40, lr=0.05, device="auto", eval_iters=4, eval_batch=8,
               windows=[[8, 16]], sketch_k=32, lam_every=16, lam_batch=16, lam_chunk=8,
               lb_batch=32, lb_chunk=8, lb_every=8, eig_at_mid=True, eig_k=4, eig_m=12,
               eig_batch=16, eig_chunk=8, stage="explore")


def run_resnet(cfg: dict) -> dict:
    model = resnet.build_model(cfg)
    data = resnet.data_loader(cfg)
    targets = resnet.target_modules(model, cfg)
    return resnet.end_to_end_train(model, data, targets, cfg, dict(kind="ema", beta=0.9))


def summarize(tag: str, out: dict) -> dict:
    n_sk = sum(len(v) for v in out["sketches"].values())
    m = dict(
        loss0=float(out["losses"][0]), loss_end=float(out["losses"][-1]),
        val_loss=float(out["val_loss"]), diverged=bool(out["diverged"]),
        spikes=int(out["spikes"]),
        lam_last=float(out["lam_trace"][-1, 1]) if out["lam_trace"].size else float("nan"),
        chi_last=float(out["chi_trace"][-1, 1]) if out["chi_trace"].size else float("nan"),
        n_raw_windows=len(out["raw"]), n_lb_windows=len(out["lb_raw"]),
        n_sketch_steps=n_sk, n_lb_align=len(out["lb_align"]),
        n_eig=len(out["eig_rec"]),
        eig_top=float(out["eig_rec"][0]["eigvals"][0]) if out["eig_rec"] else float("nan"),
    )
    print(f"  {tag}: " + " ".join(f"{k}={v}" for k, v in m.items()))
    checks = [m["n_raw_windows"] > 0, m["n_sketch_steps"] > 0, m["n_lb_align"] > 0,
              m["n_eig"] > 0, np.isfinite(m["lam_last"]), not m["diverged"]]
    m["instrumentation_ok"] = bool(all(checks))
    return m


def main() -> None:
    dev = resolve_device("auto")
    if dev.type != "cuda":
        print(f"FATAL: expected CUDA, resolved {dev}")
        sys.exit(2)
    seed_all(0)
    det = setup_determinism()
    info = device_info(dev)
    gsha, dirty = git_sha()
    print(f"smoke: {info} | git {gsha}{'+dirty' if dirty else ''} | det {det}")

    t0 = time.time()
    out_r = run_resnet(RESNET_CFG)
    m_r = summarize("resnet-cifar", out_r)
    m_r["secs"] = round(time.time() - t0, 1)

    # determinism verification: same seed -> identical loss trace
    out_r2 = run_resnet(RESNET_CFG)
    n = min(len(out_r["losses"]), len(out_r2["losses"]))
    max_dev = float(np.max(np.abs(out_r["losses"][:n] - out_r2["losses"][:n]))) \
        if n else float("nan")
    m_r["repro_max_abs_dev"] = max_dev
    # B2 policy: bitwise where kernels allow; the gate tolerance 1e-6 flags any real
    # nondeterminism (loss scale ~2.3) while not failing on a benign atomics path.
    m_r["repro_ok"] = bool(n and max_dev <= 1e-6)
    print(f"  determinism: max|delta loss| over {n} steps = {max_dev:.3e} "
          f"({'bitwise' if max_dev == 0.0 else 'within 1e-6' if m_r['repro_ok'] else 'FAIL'})")

    t0 = time.time()
    model = gpt.build_model(GPT_CFG)
    data = gpt.data_loader(GPT_CFG)
    targets = gpt.target_modules(model, GPT_CFG)
    out_g = gpt.end_to_end_train(model, data, targets, GPT_CFG, dict(kind="ema", beta=0.9))
    m_g = summarize("nanogpt-gpu", out_g)
    m_g["secs"] = round(time.time() - t0, 1)

    ok = m_r["instrumentation_ok"] and m_g["instrumentation_ok"] and m_r["repro_ok"]
    for task, cfg, m in (("resnet-cifar", RESNET_CFG, m_r), ("nanogpt-gpu", GPT_CFG, m_g)):
        rid = log_run(task=task, probe="smoke", key=f"T{cfg['steps']}_1gpu",
                      config={**cfg, "determinism": det, **info},
                      metrics={**m, "gate_pass": ok}, arrays=dict(),
                      extra_files=[__file__])
        print(f"  logged {rid}")
    print(f"smoke gate: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
