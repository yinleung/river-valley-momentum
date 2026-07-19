"""B4 throughput calibration (plan_v5 §3.1): model x batch x GPUs -> steps/s, tokens/s,
probe overhead multiplier — the table feeding the §3.4 budget check.

Component-timing design: rather than mimicking a full G-run, measure the atoms —
  plain step time            (per model/batch, autocast policy as deployed)
  window-step overhead       (raw fp16 capture + JL sketch per step)
  LB probe cost              (one LargeBatchProbe.gradient at family config)
  sharpness-tracker cost     (warm measure at family config)
  Lanczos eigenpair cost     (one midpoint call at G2 config)
— then COMPOSE each family's projected wall-clock from its actual cadences, and compare
with the §3.4 budgeted hours. Prints the table + per-family projection; logs one
calibration record. The +30% overrun rule (pause and surface) is evaluated here.

Run: 1-GPU part (share-short):  cd codebases && python scripts/wisteria/calibrate.py
     8-GPU part (short-a):      torchrun --standalone --nproc_per_node=8 \
                                    scripts/wisteria/calibrate.py --ddp124m
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_CODEBASES = pathlib.Path(__file__).resolve().parents[2]  # scripts/wisteria/ -> codebases/
sys.path.insert(0, str(_CODEBASES))

from core.device import autocast_ctx, resolve_device, seed_all, setup_determinism  # noqa: E402
from core.hvp import HVPOperator, SharpnessTracker, lanczos_topk  # noqa: E402
from core.lbprobe import LargeBatchProbe  # noqa: E402
from core.logging import log_run  # noqa: E402
from core.spectral_stream import JLSketch  # noqa: E402

sys.path.insert(0, str(_CODEBASES / "resnet-cifar" / "adapter"))
import contract as resnet  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "contract_nanogpt_gpu", _CODEBASES / "nanogpt-gpu" / "adapter" / "contract.py")
gpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpt)

GPT20M = dict(seed=0, n_layer=6, n_head=6, n_embd=384, block_size=1024, vocab_size=50304,
              batch_size=32, device="auto")  # 64 OOMs: fp32 loss logits B x 1024 x 50304
GPT124M = dict(seed=0, n_layer=12, n_head=12, n_embd=768, block_size=1024,
               vocab_size=50304, batch_size=24, device="auto")


def time_steps(step_fn, n_warm: int = 10, n_meas: int = 50) -> float:
    """Median-of-3 blocks of n_meas steps; returns seconds/step."""
    for _ in range(n_warm):
        step_fn()
    torch.cuda.synchronize()
    blocks = []
    for _ in range(3):
        t0 = time.time()
        for _ in range(n_meas):
            step_fn()
        torch.cuda.synchronize()
        blocks.append((time.time() - t0) / n_meas)
    return float(np.median(blocks))


def make_sgd_step(model, get_batch, lr=0.05, beta=0.9):
    """Plain fp32 EMA-SGDM step closure for the resnet timing (no probes)."""
    params = [p for p in model.parameters() if p.requires_grad]
    bufs = [torch.zeros_like(p) for p in params]

    def step():
        x, y = get_batch("train")
        loss = F.cross_entropy(model(x), y)
        model.zero_grad(set_to_none=False)
        loss.backward()
        with torch.no_grad():
            for p, b in zip(params, bufs):
                b.mul_(beta).add_(p.grad, alpha=1 - beta)
                p -= lr * b
    return step


def calibrate_resnet() -> dict:
    cfg = dict(seed=0, dataset="cifar10", batch_size=128, device="auto")
    model = resnet.build_model(cfg)
    data = resnet.data_loader(cfg)
    targets = resnet.target_modules(model, cfg)
    named = list(model.named_parameters())
    dev = next(model.parameters()).device
    step = make_sgd_step(model, data)
    t_plain = time_steps(step)

    jl = JLSketch(k=128, device=dev)
    def wstep():
        step()
        for tn, t in targets.items():
            _ = t.grad.detach().to("cpu", torch.float16).numpy()
        per = jl.sketch_named(named, pick=lambda p: p.grad)
        _ = JLSketch.combine(per)
    t_win = time_steps(wstep, n_warm=5, n_meas=20)

    lb = LargeBatchProbe(model, lambda b: F.cross_entropy(model(b[0]), b[1]), targets,
                         resnet.probe_batches(cfg, 1000, 2048, 512), mode="train")
    t0 = time.time(); lb.gradient(); torch.cuda.synchronize()
    t_lb = time.time() - t0
    lb4096 = LargeBatchProbe(model, lambda b: F.cross_entropy(model(b[0]), b[1]),
                             {k: targets[k] for k in list(targets)[:3]},
                             resnet.probe_batches(cfg, 1000, 4096, 512), mode="train")
    t0 = time.time(); lb4096.gradient(); torch.cuda.synchronize()
    t_lb_g2 = time.time() - t0

    lam_chunks = resnet.probe_batches(cfg, 3000, 512, 512)
    params = [p for p in model.parameters() if p.requires_grad]
    tracker = SharpnessTracker(
        lambda: HVPOperator(lambda b: F.cross_entropy(model(b[0]), b[1]), params,
                            lam_chunks), iters=3, iters0=20, seed=11)
    t0 = time.time(); tracker.measure(0); torch.cuda.synchronize()
    t_lam_cold = time.time() - t0
    t0 = time.time(); tracker.measure(1); torch.cuda.synchronize()
    t_lam = time.time() - t0

    op = HVPOperator(lambda b: F.cross_entropy(model(b[0]), b[1]), params,
                     resnet.probe_batches(cfg, 2000, 2048, 512))
    t0 = time.time(); lanczos_topk(op, k=16, m=48, seed=13); torch.cuda.synchronize()
    t_eig = time.time() - t0
    return dict(t_plain=t_plain, t_win=t_win, t_lb=t_lb, t_lb_g2=t_lb_g2,
                t_lam=t_lam, t_lam_cold=t_lam_cold, t_eig=t_eig)


def calibrate_gpt(cfg_model: dict, tag: str) -> dict:
    cfg = {**cfg_model}
    model = gpt.build_model(cfg)
    model = gpt.ddp_setup(model)
    data = gpt.data_loader(cfg)
    dev = gpt.next_device(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    bufs = [torch.zeros_like(p) for p in params]

    def step():
        x, y = data("train")
        with autocast_ctx(dev):
            _, loss = model(x, y)
        model.zero_grad(set_to_none=False)
        loss.backward()
        with torch.no_grad():
            for p, b in zip(params, bufs):
                b.mul_(0.9).add_(p.grad, alpha=0.1)
                p -= 0.05 * b
    t_plain = time_steps(step, n_warm=10, n_meas=30)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_step = cfg["batch_size"] * cfg["block_size"] * world
    out = dict(t_plain=t_plain, tokens_per_s=tokens_per_step / t_plain, world=world)
    if world == 1 and tag == "gpt20m":
        targets = gpt.target_modules(model, cfg)
        named = list(model.named_parameters())
        jl = JLSketch(k=128, device=dev)
        def wstep():
            step()
            for tn, t in targets.items():
                _ = t.grad.detach().to("cpu", torch.float16).numpy()
            per = jl.sketch_named(named, pick=lambda p: p.grad)
            _ = JLSketch.combine(per)
        out["t_win"] = time_steps(wstep, n_warm=5, n_meas=15)
        lb = LargeBatchProbe(model, lambda b: gpt_loss(model, b), targets,
                             gpt.probe_batches(cfg, 1000, 256, 16), mode="train")
        t0 = time.time(); lb.gradient(); torch.cuda.synchronize()
        out["t_lb"] = time.time() - t0
        lam_chunks = gpt.probe_batches(cfg, 3000, 64, 16)
        tracker = SharpnessTracker(
            lambda: HVPOperator(lambda b: gpt_loss(model, b), params, lam_chunks),
            iters=3, iters0=20, seed=11)
        t0 = time.time(); tracker.measure(0); torch.cuda.synchronize()
        out["t_lam_cold"] = time.time() - t0
        t0 = time.time(); tracker.measure(1); torch.cuda.synchronize()
        out["t_lam"] = time.time() - t0
    return out


def gpt_loss(model, batch):
    # Probe-path loss: MATH SDPA — the fused flash/mem-efficient kernels have no
    # double-backward, which the HVP probes need (same wrap as the task adapter).
    x, y = batch
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        _, loss = model(x, y)
    return loss


def project_family(name: str, budget_h: float, formula: str, hours: float) -> dict:
    ratio = hours / budget_h if budget_h else float("nan")
    flag = "OVER +30% — PAUSE" if ratio > 1.3 else "ok"
    print(f"  {name:22s} budget {budget_h:6.0f} h  projected {hours:7.1f} h  "
          f"x{ratio:4.2f}  {flag}   [{formula}]")
    return dict(name=name, budget_h=budget_h, projected_h=hours, ratio=ratio, flag=flag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddp124m", action="store_true",
                    help="8-GPU DDP 124M throughput (torchrun)")
    args = ap.parse_args()
    seed_all(0)
    setup_determinism()

    if args.ddp124m:
        r = calibrate_gpt(GPT124M, "gpt124m")
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"gpt124m x{r['world']}GPU: {r['t_plain']*1000:.1f} ms/step, "
                  f"{r['tokens_per_s']/1e3:.0f}k tok/s")
            log_run(task="wisteria", probe="calibration",
                    key=f"gpt124m_ddp{r['world']}",
                    config={**GPT124M, "stage": "explore", "world": r["world"]},
                    metrics={k: v for k, v in r.items()}, arrays=dict(),
                    extra_files=[__file__])
        return

    print("== resnet-cifar (batch 128, fp32) ==")
    rn = calibrate_resnet()
    for k, v in rn.items():
        print(f"  {k}: {v*1000:.1f} ms")
    print(f"== gpt20m (batch {GPT20M['batch_size']}x1024, bf16 autocast) ==")
    g20 = calibrate_gpt(GPT20M, "gpt20m")
    for k, v in g20.items():
        print(f"  {k}: {v if k in ('world', 'tokens_per_s') else round(v*1000, 1)}"
              f"{'' if k in ('world', 'tokens_per_s') else ' ms'}")
    print("== gpt124m 1-GPU (batch 24x1024, bf16 autocast) ==")
    g124 = calibrate_gpt(GPT124M, "gpt124m")
    print(f"  t_plain: {g124['t_plain']*1000:.1f} ms  "
          f"tokens/s: {g124['tokens_per_s']/1e3:.0f}k")

    # ---- §3.4 family projections from component times ----
    print("\n== family projections vs §3.4 budgets ==")
    win_steps, lam_n, lb_n = 3 * 2048, 10000 // 200, 3 * (2048 // 256)
    g1_run_h = (10000 * rn["t_plain"] + win_steps * (rn["t_win"] - rn["t_plain"])
                + lam_n * rn["t_lam"] + rn["t_lam_cold"] + lb_n * rn["t_lb"]) / 3600
    g1_runs = 9 + 90 + 12 + 10 + 45 + 15  # scan + grid + P-prac + 5-seed topups + C100 + GN
    fams = [project_family("G1 resnet sweep", 110, f"{g1_runs} runs x {g1_run_h:.2f} h",
                           g1_runs * g1_run_h)]
    g2_run_h = (10000 * rn["t_plain"] + win_steps * (rn["t_win"] - rn["t_plain"])
                + win_steps * rn["t_lb_g2"] + 3 * rn["t_eig"]) / 3600
    fams.append(project_family("G2 decomposition", 55, f"12 runs x {g2_run_h:.2f} h",
                               12 * g2_run_h))
    g3_run_h = (500 * rn["t_win"]) / 3600
    fams.append(project_family("G3 forced", 15, f"~102 arms x {g3_run_h:.3f} h",
                               102 * g3_run_h + 2 * rn["t_eig"] / 3600))
    g4d_run_h = (6000 * g20["t_plain"] + 2 * 2048 * (g20["t_win"] - g20["t_plain"])
                 + 24 * g20["t_lam"] + 16 * g20["t_lb"]) / 3600
    g4d_runs = 5 * 4 * 3
    fams.append(project_family("G4 20M diagnostic", 75,
                               f"{g4d_runs} runs x {g4d_run_h:.2f} h",
                               g4d_runs * g4d_run_h))
    # 124M confirmation: 5 configs x 3 seeds x 2.5B tokens at the 8-GPU rate (8 GPUs
    # charged per GPU-h -> hours here are GPU-hours = 8 x wall)
    conf_wall_h = 2.5e9 / max(g124["tokens_per_s"] * 6.5, 1.0) / 3600  # ~6.5x scaling est
    fams.append(project_family("G4 124M confirmation", 90,
                               f"15 runs x {conf_wall_h:.1f} wall-h x 8 GPU (est. from "
                               "1-GPU rate x6.5; refine with --ddp124m)",
                               15 * conf_wall_h * 8))
    metrics = dict(**{f"resnet_{k}": v for k, v in rn.items()},
                   **{f"gpt20m_{k}": v for k, v in g20.items()},
                   **{f"gpt124m_{k}": v for k, v in g124.items()},
                   **{f"proj_{f['name'].split()[0]}_h": f["projected_h"] for f in fams})
    over = [f for f in fams if f["ratio"] > 1.3]
    metrics["any_family_over_30pct"] = bool(over)
    log_run(task="wisteria", probe="calibration", key="components_1gpu",
            config=dict(stage="explore", resnet_batch=128, gpt20m=GPT20M,
                        gpt124m=GPT124M),
            metrics=metrics, arrays=dict(), extra_files=[__file__])
    print("\ncalibration logged." + (" WARNING: over-budget families above!" if over
                                     else " All families within +30% of budget."))


if __name__ == "__main__":
    main()
