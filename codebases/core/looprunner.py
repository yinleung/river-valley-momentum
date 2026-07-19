"""The instrumented closed-loop runner (plan_v5 §3.1 B5), written once, task-agnostic.

Every task adapter's `end_to_end_train` delegates here: the optimizer conventions, the
divergence discipline, and the probe orchestration are defined in ONE place, so G1–G6
measure identical quantities on every task (Pillar 2: analysis written once). The adapter
supplies a `TaskHooks` with its forward/data/eval callables; `core/` never imports tasks.

Optimizer conventions (§3.0 invariant 1):
    kind "ema"   m = beta*m + (1-beta)*g;  w -= lr*m      (P-thy, the paper's normalization)
    kind "hb"    m = beta*m + g;           w -= lr*m      (P-prac conventional; eta_HB = eta*(1-beta))

Divergence discipline (§3.0 invariant 2): no clipping; run stops on non-finite loss or
loss > 3x initial; spike counter = steps with loss > 2x trailing-50 median. Divergence
counts as failure downstream.

Precision policy (B2): the training forward runs under hooks.autocast (bf16 on CUDA for
GPT-scale tasks, nullcontext for fp32 tasks); every probe path (windows read .grad as
produced by the training backward; LB/HVP probes call hooks.forward_loss_fp32) is fp32.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from .forced import ForcedInjector
from .hvp import HVPOperator, SharpnessTracker, lanczos_topk
from .lbprobe import LargeBatchProbe
from .spectral_stream import JLSketch, WindowBuffer

__all__ = ["TaskHooks", "run_closed_loop"]


@dataclass
class TaskHooks:
    """Task-side callables the runner drives (the adapter's Contract surface).

    forward_loss(batch)       scalar training loss, graph attached (train-mode forward;
                              the runner wraps it in `autocast` for the training step).
    forward_loss_fp32(batch)  same loss OUTSIDE autocast (probe paths; may be identical).
    next_batch()              next training batch (seeded order is the adapter's duty).
    evaluate()                (val_loss, val_metric) on the held-out split, state-neutral.
    probe_batches(offset, total, chunk)  FIXED probe chunks from a dedicated generator.
    layer_groups              {group: [param names]} for JL sketch combining.
    autocast()                context manager for the training forward (nullcontext ok).
    probe_mode                "train" | "eval" forward mode for LB/HVP probes (BN caveat
                              is the adapter's call; stateless-norm models: identical).
    """

    forward_loss: Callable
    forward_loss_fp32: Callable
    next_batch: Callable
    evaluate: Callable
    probe_batches: Callable
    layer_groups: dict = field(default_factory=dict)
    autocast: Callable = contextlib.nullcontext
    probe_mode: str = "train"


def _named_buffer_state(model):
    return [(b, b.detach().clone()) for _, b in model.named_buffers()]


def _restore_buffers(saved):
    with torch.no_grad():
        for b, s in saved:
            b.copy_(s)


def run_closed_loop(model: torch.nn.Module, hooks: TaskHooks,
                    targets: dict[str, torch.Tensor], config: dict, opt: dict,
                    injector: ForcedInjector | None = None,
                    raw_sink: Callable | None = None,
                    step_callback: Callable | None = None) -> dict:
    """Instrumented closed-loop training; see the resnet-cifar adapter docstring for the
    config schema (windows, lb_*, sketch_k, lam_*, eig_*, eval_every, raw_dtype).

    injector: optional ForcedInjector applied AFTER backward, BEFORE the update — window
        streams and sketches record the post-injection gradient (what the optimizer sees).
    raw_sink(wi, name, kind, array, m0): spills each completed window ("G" | "GLB") so
        long runs never hold all windows in RAM (a 2048-step window over 5 ResNet targets
        is ~11 GB fp16); m0 = the window-start buffer snapshot for "G" (None for "GLB"),
        so sinks can reduce (e.g. reconstruct the m-stream for MSR) before discarding.
        None keeps all windows in the returned dict (smoke/preflight scale only).
    step_callback(t, model): optional (e.g. checkpointing at declared steps).

    Returns: losses, gnorm/mnorm, eval_trace (t, val_loss, val_metric), lam_trace,
    chi_trace (t, eta_eff*(1-beta)*lam), raw/lb_raw windows, m0 snapshots, sketches,
    lb_align scalars, eig_rec, inj_rec, spikes, diverged, val_loss, val_metric.
    """
    lr, T = float(config["lr"]), int(config["steps"])
    kind, beta = opt["kind"], float(opt["beta"])
    if kind not in ("ema", "hb"):
        raise ValueError(f"opt kind must be 'ema' or 'hb'; got {kind!r}")
    coef_g = (1.0 - beta) if kind == "ema" else 1.0

    windows = [tuple(w) for w in config.get("windows", [])]
    lb_batch = int(config.get("lb_batch", 0))
    lb_every = int(config.get("lb_every", 8))
    sketch_k = int(config.get("sketch_k", 0))
    lam_every = int(config.get("lam_every", 0))
    eval_every = int(config.get("eval_every", 0))
    tnames = list(targets)
    lb_targets = list(config.get("lb_targets", tnames))

    params = [p for p in model.parameters() if p.requires_grad]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    bufs = {id(p): torch.zeros_like(p) for p in params}
    device = params[0].device

    # probes (all on fixed chunks from dedicated generators; offsets: lb 1000, eig 2000,
    # lam 3000 — the E12 separation)
    lbp = None
    if lb_batch:
        chunks = hooks.probe_batches(1000, lb_batch, int(config.get("lb_chunk", 512)))
        lbp = LargeBatchProbe(model, hooks.forward_loss_fp32,
                              {tn: targets[tn] for tn in lb_targets}, chunks,
                              mode=hooks.probe_mode)
    eig_batches = None
    if bool(config.get("eig_at_mid", False)):
        eig_batches = hooks.probe_batches(2000, int(config.get("eig_batch", 2048)),
                                          int(config.get("eig_chunk", 512)))
    tracker = None
    if lam_every:
        lam_chunks = hooks.probe_batches(3000, int(config.get("lam_batch", 512)),
                                         int(config.get("lam_chunk", 512)))

        def make_op():
            return HVPOperator(hooks.forward_loss_fp32, params, lam_chunks)

        tracker = SharpnessTracker(make_op, iters=int(config.get("lam_iters", 3)),
                                   iters0=int(config.get("lam_iters0", 20)),
                                   seed=config["seed"] + 11)
    jl = JLSketch(k=sketch_k, device=device) if sketch_k else None

    # window recorders (preallocated; spilled via raw_sink as each window completes)
    rdt = np.dtype(config.get("raw_dtype", "float16"))
    raw: dict[tuple, WindowBuffer | None] = {}
    lb_raw: dict[tuple, WindowBuffer | None] = {}
    m0: dict[str, np.ndarray] = {}
    for wi, (s, ln) in enumerate(windows):
        for tn in tnames:
            raw[(wi, tn)] = WindowBuffer(tuple(targets[tn].shape), ln, dtype=rdt)
        if lbp is not None:
            n_lb = (ln + lb_every - 1) // lb_every
            for tn in lb_targets:
                lb_raw[(wi, tn)] = WindowBuffer(tuple(targets[tn].shape), n_lb, dtype=rdt)
    sketches: dict[int, list] = {wi: [] for wi in range(len(windows))}
    lb_align: list[dict] = []
    eig_rec: list[dict] = []
    inj_rec: list[dict] = []
    lam_trace: list[tuple[int, float]] = []
    eval_trace: list[tuple[int, float, float]] = []

    def _finish_window(wi: int) -> None:
        if raw_sink is None:
            return
        for tn in tnames:
            raw_sink(wi, tn, "G", raw[(wi, tn)].array(), m0.get(f"{wi}:{tn}"))
            raw[(wi, tn)] = None
        for tn in lb_targets:
            if (wi, tn) in lb_raw and lb_raw[(wi, tn)] is not None:
                raw_sink(wi, tn, "GLB", lb_raw[(wi, tn)].array(), None)
                lb_raw[(wi, tn)] = None

    mids = {s + ln // 2: wi for wi, (s, ln) in enumerate(windows)}
    win_of_step = np.full(T, -1, dtype=int)
    for wi, (s, ln) in enumerate(windows):
        win_of_step[s:s + ln] = wi

    losses, gnorm, mnorm = [], [], []
    diverged = False
    loss0 = None
    spikes = 0
    model.train()
    for t in range(T):
        # window-midpoint eigenpairs (G2) — before this step's update
        if eig_batches is not None and t in mids:
            saved_bufs = _named_buffer_state(model)
            op = HVPOperator(hooks.forward_loss_fp32, params, eig_batches)
            ritz, vecs, resid = lanczos_topk(
                op, k=int(config.get("eig_k", 16)), m=int(config.get("eig_m", 48)),
                seed=config["seed"] + 13, want_vectors=True)
            _restore_buffers(saved_bufs)
            model.zero_grad(set_to_none=True)
            eig_rec.append(dict(step=t, window=mids[t], eigvals=ritz, eig_resid=resid,
                                eigvecs=vecs))
        if lam_every and t % lam_every == 0:
            saved_bufs = _named_buffer_state(model)
            lam_trace.append((t, tracker.measure(t)))
            _restore_buffers(saved_bufs)
            model.zero_grad(set_to_none=True)

        batch = hooks.next_batch()
        with hooks.autocast():
            loss = hooks.forward_loss(batch)
        lval = float(loss.detach())
        # Divergence guard value: under DDP the decision must be identical on every rank
        # or a lone breaking rank desyncs the collectives and hangs the job (Codex
        # Phase-0 review finding). MAX-reduce the loss, with non-finite mapped to +inf
        # (NaN through NCCL MAX is not defined).
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([lval if np.isfinite(lval) else float("inf")],
                             device=device, dtype=torch.float32)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            guard_val = float(t.item())
        else:
            guard_val = lval
        loss0 = guard_val if loss0 is None else loss0
        losses.append(lval)
        if not np.isfinite(guard_val) or guard_val > 3.0 * max(loss0, 1.0):
            diverged = True
            losses.pop()
            break
        if t > 50 and lval > 2.0 * float(np.median(losses[-51:-1])):
            spikes += 1
        model.zero_grad(set_to_none=False)
        loss.backward()

        if injector is not None:
            c = injector.inject(named, t)
            inj_rec.append(dict(step=t, coef=c, w_proj=injector.project(named),
                                g_proj=injector.project(
                                    [(n, p.grad) for n, p in named])))

        wi = int(win_of_step[t])
        if wi >= 0:
            s, ln = windows[wi]
            if t == s:  # buffer snapshot: m reconstructable from G + m0 (see conventions)
                for tn in tnames:
                    m0[f"{wi}:{tn}"] = bufs[id(targets[tn])].detach().to(
                        "cpu", torch.float32).numpy().copy()
            for tn in tnames:
                raw[(wi, tn)].push(targets[tn].grad)
            if jl is not None:
                per = jl.sketch_named(named, pick=lambda p: p.grad)
                rec = {"all": JLSketch.combine(per)}
                rec.update({g: JLSketch.combine(per, names=ns)
                            for g, ns in hooks.layer_groups.items() if ns})
                sketches[wi].append(rec)
            if lbp is not None and (t - s) % lb_every == 0:
                glb = lbp.gradient()
                rec = dict(step=t, window=wi)
                for tn in lb_targets:
                    g = targets[tn].grad.detach().to("cpu", torch.float32).numpy()
                    m = bufs[id(targets[tn])].detach().to("cpu", torch.float32).numpy()
                    gl = glb[tn]
                    lb_raw[(wi, tn)].push(torch.from_numpy(gl))
                    rec[f"{tn}:align_m"] = float(
                        np.sum(m * gl) / (np.linalg.norm(m) * np.linalg.norm(gl) + 1e-30))
                    rec[f"{tn}:align_g"] = float(
                        np.sum(g * gl) / (np.linalg.norm(g) * np.linalg.norm(gl) + 1e-30))
                lb_align.append(rec)
            if t == s + ln - 1:
                _finish_window(wi)

        gn2, mn2 = 0.0, 0.0
        with torch.no_grad():
            for p in params:
                buf = bufs[id(p)]
                buf.mul_(beta).add_(p.grad, alpha=coef_g)
                p -= lr * buf
                gn2 += float((p.grad * p.grad).sum())
                mn2 += float((buf * buf).sum())
        gnorm.append(float(np.sqrt(gn2)))
        mnorm.append(float(np.sqrt(mn2)))

        if step_callback is not None:
            step_callback(t, model)
        if eval_every and (t + 1) % eval_every == 0:
            eval_trace.append((t, *hooks.evaluate()))

    if not diverged:
        eval_trace.append((len(losses) - 1, *hooks.evaluate()))

    lam_arr = np.asarray(lam_trace, dtype=float).reshape(-1, 2)
    eta_eff = lr if kind == "ema" else (lr / (1.0 - beta) if beta < 1 else float("inf"))
    return dict(
        losses=np.asarray(losses),
        gnorm=np.asarray(gnorm), mnorm=np.asarray(mnorm),
        eval_trace=np.asarray(eval_trace, dtype=float).reshape(-1, 3),
        lam_trace=lam_arr,
        chi_trace=(np.column_stack([lam_arr[:, 0], eta_eff * (1.0 - beta) * lam_arr[:, 1]])
                   if lam_arr.size else lam_arr),
        raw={k: v.array() for k, v in raw.items() if v is not None},
        lb_raw={k: v.array() for k, v in lb_raw.items() if v is not None},
        m0=m0,
        sketches=sketches,
        lb_align=lb_align,
        eig_rec=eig_rec,
        inj_rec=inj_rec,
        spikes=spikes,
        diverged=diverged,
        val_loss=eval_trace[-1][1] if eval_trace else float("inf"),
        val_metric=eval_trace[-1][2] if eval_trace else 0.0,
    )
