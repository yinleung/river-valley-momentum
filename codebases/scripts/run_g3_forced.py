"""G3 — forced-frequency controls on the real network (plan_v5 §3.2; review Exp C,
package #4). E13's methodology at network scale: inject A cos(omega t) v along the
measured top Hessian eigenvector and read the response by lock-in at exactly omega.

Design (Codex plan-review fix baked in: an explicit beta x omega grid from COMMON
checkpoints, forcing frequencies fixed per checkpoint, not per arm):

  make_ckpts   train the two base runs (subcritical + near-edge LR from the G1 map,
               beta = 0.9, seed 0) to the declared branch step (mid-training, after the
               EoS-onset window), save weights to results/ckpt/, then at the frozen
               checkpoint: top-1 full-model Hessian eigenvector v (Lanczos, fixed probe
               batch), the STATIONARY rms of the v-aligned mini-batch gradient over 50
               frozen draws (sets A = amp_frac x rms), and lam_hat for the |G_beta| guides.
  arms         per checkpoint: omega in {0.02 (passband), theta_hat at beta=0.9
               (resonance, from measured lam_hat), 0.6pi, pi (Nyquist)} x
               beta in {0, 0.5, 0.9, 0.95, 0.99} x 3 seeds + one extra arm
               (theta_hat_{0.99}, beta = 0.99) per checkpoint to confirm the resonance
               peak moves with beta. Each arm: load ckpt, fresh buffer m = 0 (declared),
               constant lr (frozen schedule; P-thy exemption §3.0.8), 500 steps, injector
               active throughout; every arm at one (checkpoint, seed) shares the batch
               sequence.
  report       gates below over the arm grid.

Measures per arm: lock-in amplitude (steps >= 100 scored; absolute step index keeps the
phase convention of core.forced) of w_proj = <w_t, v> and of the raw-gradient projection
g_proj - c_t (the known injected tone is subtracted); tail train loss (last 250 steps);
parameter-free guide A|G_beta(omega)| from the checkpoint lam_hat (core.closedloop).

Predeclared gates (network-scale operationalizations of E13's; toy values cited):
  F1 passband neutrality  amp(0.02, beta) max/min <= 1.5 across beta (toy gate: 1.1);
  F2 Nyquist suppression  amp(pi, 0.99)/amp(pi, 0) <= 0.1 (toy: 0.01);
  F3 resonant harm        tail loss(theta_0.9, beta = 0.9) >= 1.25 x tail loss(theta_0.9,
                          beta = 0) with 2-SE separation, AND relief at mismatched beta
                          (loss at beta in {0, 0.99} below the matched-beta loss);
  F4 peak moves           the extra (theta_0.99, 0.99) arm's loss exceeds its
                          (theta_0.9 -> 0.99) counterpart — the resonance follows beta.

Run:  cd codebases && python scripts/run_g3_forced.py --stage make_ckpts
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_CODEBASES = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODEBASES))
sys.path.insert(0, str(_CODEBASES / "resnet-cifar" / "adapter"))

from core import closedloop as CL  # noqa: E402
from core.device import seed_all, setup_determinism  # noqa: E402
from core.forced import ForcedInjector, lockin_amp  # noqa: E402
from core.hvp import HVPOperator, lanczos_topk, unflatten_like  # noqa: E402
from core.logging import log_run  # noqa: E402
import contract  # noqa: E402

CFG_PATH = _CODEBASES / "resnet-cifar" / "config.yaml"
CKPT_DIR = _CODEBASES / "results" / "ckpt"
GATES = ("F1 passband max/min<=1.5; F2 nyquist 0.99/0 <=0.1; "
         "F3 resonance matched>=1.25x beta0 with 2-SE sep + mismatch relief; "
         "F4 peak moves with beta")
BRANCH_STEP = 5000
ARM_STEPS = 500
SCORE_FROM = 100  # absolute in-arm step where lock-in scoring starts
EIG_K, EIG_M = 4, 48          # checkpoint top-eigenvector Lanczos
EIG_BATCH, EIG_CHUNK = 2048, 512
RMS_DRAWS = 50                # stationary draws for the aligned-gradient rms


def load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _ckpt_lrs(y: dict) -> dict[str, float]:
    lrs = y["g1"].get("grid_lrs") or []
    if len(lrs) < 2:
        raise SystemExit("g1.grid_lrs missing — run the G1 scan first")
    return {"subcritical": float(lrs[0]), "nearedge": float(lrs[-2])}


def stage_make_ckpts(y: dict) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for tag, lr in _ckpt_lrs(y).items():
        cfg = dict(seed=0, device="auto", dataset=y["data"]["dataset"],
                   batch_size=y["data"]["batch_size"], augment=y["data"]["augment"],
                   norm=y["model"]["norm"], lr=lr, steps=BRANCH_STEP, windows=[],
                   eval_every=0, stage="confirm", protocol="P-thy")
        seed_all(0)
        cfg["determinism"] = setup_determinism()
        model = contract.build_model(cfg)
        data = contract.data_loader(cfg)
        targets = contract.target_modules(model, cfg)
        t0 = time.time()
        out = contract.end_to_end_train(model, data, targets, cfg,
                                        dict(kind="ema", beta=0.9))
        if out["diverged"]:
            raise SystemExit(f"checkpoint base run diverged at lr={lr} — re-pick LR")
        torch.save(model.state_dict(), CKPT_DIR / f"g3_{tag}.pt")

        # frozen-checkpoint quantities: top eigenpair, aligned-gradient rms, lam_hat
        params = [p for p in model.parameters() if p.requires_grad]
        batches = contract.probe_batches(cfg, 2000, EIG_BATCH, EIG_CHUNK)
        op = HVPOperator(lambda b: F.cross_entropy(model(b[0]), b[1]), params, batches)
        ritz, vecs, resid = lanczos_topk(op, k=EIG_K, m=EIG_M, seed=13,
                                         want_vectors=True)
        v_named = {n: t for (n, _), t in zip(
            [(n, p) for n, p in model.named_parameters() if p.requires_grad],
            unflatten_like(torch.from_numpy(vecs[0]).to(params[0].device), params))}
        rms_draws = []
        probe = contract.data_loader({**cfg, "seed": 12345})
        model.zero_grad(set_to_none=False)
        for _ in range(RMS_DRAWS):
            x, yb = probe("train")
            model.zero_grad(set_to_none=False)
            F.cross_entropy(model(x), yb).backward()
            s = sum(float((p.grad * v_named[n]).sum())
                    for n, p in model.named_parameters() if n in v_named)
            rms_draws.append(s)
        rms = float(np.sqrt(np.mean(np.square(rms_draws))))
        A = float(y["g3"]["amp_frac"]) * rms
        np.savez(CKPT_DIR / f"g3_{tag}_probe.npz",
                 eigvals=ritz, eig_resid=resid, v_flat=vecs[0].astype(np.float32),
                 aligned_rms=rms, A=A)
        lam = float(ritz[0])
        theta09 = CL.resonant_frequency(0.9, lr, lam)
        theta099 = CL.resonant_frequency(0.99, lr, lam)
        log_run(task="resnet-cifar", probe="g3ckpt", key=f"{tag}_lr{lr:g}",
                config={**cfg, "gates": GATES, "branch_step": BRANCH_STEP},
                metrics=dict(lam_top=lam, eig_resid_top=float(resid[0]),
                             aligned_rms=rms, A=A, theta09=theta09, theta099=theta099,
                             tail_loss=float(np.mean(out["losses"][-250:])),
                             secs=round(time.time() - t0, 1)),
                arrays=dict(eigvals=ritz, losses=out["losses"].astype(np.float32)),
                extra_files=[__file__])
        print(f"  ckpt {tag}: lam={lam:.1f} rms={rms:.4f} A={A:.4f} "
              f"theta09={theta09:.3f} theta099={theta099:.3f}")


def _arm_omegas(tag: str) -> dict[str, float]:
    m = json.loads((_CODEBASES / "results" / "cache" /
                    _find_ckpt_run(tag) / "metrics.json").read_text())
    return {"passband": 0.02, "resonance": m["theta09"], "high": 0.6 * np.pi,
            "nyquist": float(np.pi)}, m


def _find_ckpt_run(tag: str) -> str:
    import csv
    with open(_CODEBASES / "results" / "index" / "runs.csv") as f:
        rows = [r["run_id"] for r in csv.DictReader(f)
                if r["probe"] == "g3ckpt" and r["key"].startswith(tag)]
    if not rows:
        raise SystemExit(f"no g3ckpt record for {tag} — run make_ckpts")
    return rows[-1]


def run_arm(y: dict, tag: str, lr: float, om_name: str, omega: float, beta: float,
            seed: int, A: float, v_flat: np.ndarray, extra: str = "") -> str:
    cfg = dict(seed=seed, device="auto", dataset=y["data"]["dataset"],
               batch_size=y["data"]["batch_size"], augment=y["data"]["augment"],
               norm=y["model"]["norm"], lr=lr, steps=ARM_STEPS, windows=[],
               eval_every=0, stage="confirm", gates=GATES, protocol="P-thy",
               ckpt=tag, omega=float(omega), A=float(A), om_name=om_name)
    seed_all(seed)
    cfg["determinism"] = setup_determinism()
    model = contract.build_model(cfg)
    model.load_state_dict(torch.load(CKPT_DIR / f"g3_{tag}.pt", weights_only=True))
    data = contract.data_loader(cfg)
    targets = contract.target_modules(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    v_named = {n: t for (n, _), t in zip(
        [(n, p) for n, p in model.named_parameters() if p.requires_grad],
        unflatten_like(torch.from_numpy(v_flat).to(params[0].device), params))}
    inj = ForcedInjector(v_named, A=A, omega=omega)
    t0 = time.time()
    out = contract.end_to_end_train(model, data, targets, cfg,
                                    dict(kind="ema", beta=beta), injector=inj)
    secs = time.time() - t0
    ir = out["inj_rec"]
    w_proj = np.array([r["w_proj"] for r in ir])
    g_proj = np.array([r["g_proj"] - r["coef"] for r in ir])  # injected tone removed
    amp_w = lockin_amp(w_proj, omega, t0=SCORE_FROM) if len(w_proj) > SCORE_FROM else \
        float("nan")
    amp_g = lockin_amp(g_proj, omega, t0=SCORE_FROM) if len(g_proj) > SCORE_FROM else \
        float("nan")
    tail = float(np.mean(out["losses"][-250:])) if len(out["losses"]) >= 250 else \
        float(np.mean(out["losses"])) if len(out["losses"]) else float("nan")
    key = f"arm{extra}_{tag}_{om_name}_b{beta:g}_s{seed}"
    rid = log_run(task="resnet-cifar", probe="g3", key=key, config=cfg,
                  metrics=dict(diverged=bool(out["diverged"]), amp_w=amp_w, amp_g=amp_g,
                               tail_loss=tail, spikes=int(out["spikes"]),
                               secs=round(secs, 1)),
                  arrays=dict(losses=out["losses"].astype(np.float32),
                              w_proj=w_proj.astype(np.float32),
                              g_proj=g_proj.astype(np.float32)),
                  extra_files=[__file__])
    print(f"  {rid}: amp_w={amp_w:.4g} tail={tail:.3f} div={out['diverged']} "
          f"({secs:.0f}s)", flush=True)
    return rid


def stage_arms(y: dict, only_ckpt: str | None) -> None:
    for tag, lr in _ckpt_lrs(y).items():
        if only_ckpt and tag != only_ckpt:
            continue
        omegas, m = _arm_omegas(tag)
        probe = np.load(CKPT_DIR / f"g3_{tag}_probe.npz")
        v_flat, A = probe["v_flat"], float(probe["A"])
        betas = [float(b) for b in y["g3"]["betas"]]
        for (om_name, omega), beta, seed in itertools.product(
                omegas.items(), betas, [int(s) for s in y["g3"]["seeds"]]):
            run_arm(y, tag, lr, om_name, omega, beta, seed, A, v_flat)
        for seed in [int(s) for s in y["g3"]["seeds"]]:  # peak-moves extra arm
            run_arm(y, tag, lr, "resonance99", m["theta099"], 0.99, seed, A, v_flat,
                    extra="X")


def stage_report(y: dict) -> None:
    import csv
    with open(_CODEBASES / "results" / "index" / "runs.csv") as f:
        rows = [r for r in csv.DictReader(f)
                if r["task"] == "resnet-cifar" and r["probe"] == "g3"]
    rec: dict[tuple, list[dict]] = {}
    for r in rows:
        m = json.loads((_CODEBASES / "results" / "cache" / r["run_id"] /
                        "metrics.json").read_text())
        c = json.loads((_CODEBASES / "results" / "cache" / r["run_id"] /
                        "config.json").read_text())
        rec.setdefault((c["ckpt"], c["om_name"], _beta_of(r["key"])), []).append(m)

    def agg(tag, om, b, field):
        ms = rec.get((tag, om, b), [])
        vals = [m[field] for m in ms if np.isfinite(m[field])]
        return (float(np.mean(vals)), float(np.std(vals) / max(len(vals), 1) ** 0.5)) \
            if vals else (float("nan"), float("nan"))

    y3 = y["g3"]
    betas = [float(b) for b in y3["betas"]]
    for tag in _ckpt_lrs(y):
        if not any(k[0] == tag for k in rec):
            continue
        print(f"\ncheckpoint {tag}:")
        amps = {b: agg(tag, "passband", b, "amp_w")[0] for b in betas}
        f1 = max(amps.values()) / max(min(amps.values()), 1e-12) <= 1.5
        print(f"  F1 passband amp max/min = "
              f"{max(amps.values()) / max(min(amps.values()), 1e-12):.2f} "
              f"{'PASS' if f1 else 'FAIL'}  ({amps})")
        r_n = agg(tag, "nyquist", 0.99, "amp_w")[0] / \
            max(agg(tag, "nyquist", 0.0, "amp_w")[0], 1e-12)
        print(f"  F2 nyquist 0.99/0 = {r_n:.3f} {'PASS' if r_n <= 0.1 else 'FAIL'}")
        l09, se09 = agg(tag, "resonance", 0.9, "tail_loss")
        l00, se00 = agg(tag, "resonance", 0.0, "tail_loss")
        l99, _ = agg(tag, "resonance", 0.99, "tail_loss")
        f3 = l09 >= 1.25 * l00 and (l09 - 2 * se09) > (l00 + 2 * se00) and \
            l00 < l09 and l99 < l09
        print(f"  F3 resonance: matched(0.9)={l09:.3f}±{se09:.3f} beta0={l00:.3f}"
              f"±{se00:.3f} beta0.99={l99:.3f} {'PASS' if f3 else 'FAIL'}")
        lX, _ = agg(tag, "resonance99", 0.99, "tail_loss")
        f4 = np.isfinite(lX) and lX > l99
        print(f"  F4 peak moves: theta99@0.99={lX:.3f} vs theta09@0.99={l99:.3f} "
              f"{'PASS' if f4 else 'FAIL'}")


def _beta_of(key: str) -> float:
    for part in key.split("_"):
        if part.startswith("b") and part[1:].replace(".", "").isdigit():
            return float(part[1:])
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["make_ckpts", "arms", "report"])
    ap.add_argument("--ckpt", default=None, choices=[None, "subcritical", "nearedge"])
    args = ap.parse_args()
    y = load_cfg()
    if args.stage == "make_ckpts":
        stage_make_ckpts(y)
    elif args.stage == "arms":
        stage_arms(y, args.ckpt)
    else:
        stage_report(y)


if __name__ == "__main__":
    main()
