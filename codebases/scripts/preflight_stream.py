"""Preflight gradient-stream check (plan_v5 §3.1 B5) — the muTransfer coord-check analogue.

Run on every new task/scale BEFORE any grid launches: a few-hundred-step beta = 0 run per
ladder LR, recording the probed matrices' raw gradient streams over one short window, then
PSD / autocorrelation / HFER diagnostics against a white-noise surrogate band. Verifies the
low-pass/river-valley regime (high-frequency, landscape-driven gradient structure) is even
present before GPU budget is committed. Released with the public artifact as the paper's
practical preflight recipe.

Deliberate deviation, declared: the preflight window (256 steps) is SHORTER than the §3.0.3
diagnostic-window rule (>= 1024) — this is a cheap presence check at beta = 0 (T_eff = 1),
not a paper measurement; campaign windows obey the rule.

Predeclared PASS rules (per task; evaluated on the ladder):
  P0  wiring: every ladder run starts finite and the stream records completely at every
      LR that does not diverge; at least two ladder LRs complete un-diverged.
  P1  regime presence: at >= 1 completed LR, the PRIMARY probed matrix has
      HFER(0.6 pi, rect) above the white-noise 97.5% surrogate quantile AND rising PSD
      (Spearman rho(omega, S(omega)) >= 0.5).
  P2  structure is temporal, not amplitude: at every completed LR the stream's lag-1
      autocorrelation |rho_1| is finite and the P1 cell has rho_1 < -0.2 or > 0.2
      (a white stream has rho_1 ~ 0; the hill mode gives negative rho_1 near the edge).

Usage:
  cd codebases && python scripts/preflight_stream.py --task resnet-cifar
  cd codebases && python scripts/preflight_stream.py --task nanogpt-gpu
Logs one run record per task: task=<task>, probe="preflight", stage="explore".
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import time

import numpy as np
import yaml

_CODEBASES = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODEBASES))

from core import metrics as Me  # noqa: E402
from core.device import seed_all, setup_determinism  # noqa: E402
from core.logging import log_run  # noqa: E402

HI_FRACS = (0.5, 0.6, 0.7)


def load_task(task: str):
    """Import <task>/adapter/contract.py as a module (adapters set their own sys.path)."""
    path = _CODEBASES / task / "adapter" / "contract.py"
    spec = importlib.util.spec_from_file_location(f"contract_{task.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def task_preflight_config(task: str) -> dict:
    """Base config + ladder from the task's config.yaml."""
    with open(_CODEBASES / task / "config.yaml") as f:
        y = yaml.safe_load(f)
    if task == "resnet-cifar":
        base = dict(seed=0, dataset=y["data"]["dataset"], batch_size=y["data"]["batch_size"],
                    norm="bn", augment=False, device="auto")
        lrs = [0.0125, 0.05, 0.2, 0.8, 1.6]
        steps, window = 384, [128, 256]
    elif task == "nanogpt-gpu":
        pf = y["preflight"]
        base = dict(seed=0, device="auto", data_dir=y["data"]["data_dir"],
                    block_size=y["data"]["block_size"], vocab_size=y["data"]["vocab_size"],
                    eval_iters=y["train"]["eval_iters"], eval_batch=y["train"]["eval_batch"],
                    **y["models"][pf["model"]])
        lrs = list(pf["lrs"])
        steps, window = int(pf["steps"]), list(pf["windows"][0])
    else:
        raise SystemExit(f"unknown task {task}")
    base.update(steps=steps, windows=[window], raw_dtype="float32")
    return base, lrs


def white_band(T: int, n_surr: int = 200, hi_frac: float = 0.6,
               seed: int = 0) -> tuple[float, float]:
    """(mean, 97.5% quantile) of HFER(rect) for white (T, 8, 8) surrogate streams."""
    rng = np.random.default_rng(seed)
    vals = [Me.high_freq_energy_ratio(rng.standard_normal((T, 8, 8)), window="rect",
                                      hi_frac=hi_frac) for _ in range(n_surr)]
    return float(np.mean(vals)), float(np.quantile(vals, 0.975))


def stream_diags(G: np.ndarray) -> dict:
    """Per-stream diagnostics: HFERs (rect+hann, 3 cutoffs), PSD Spearman, lag-1 rho."""
    G = G.astype(np.float64)
    out = {}
    for wname in ("rect", "hann"):
        for hf in HI_FRACS:
            out[f"hfer_{wname}_{hf}"] = Me.high_freq_energy_ratio(G, window=wname,
                                                                  hi_frac=hf)
    omega, X = Me.windowed_dft(G, window="rect")
    axes = tuple(range(1, X.ndim))
    p = np.sum(np.abs(X) ** 2, axis=axes) if X.ndim > 1 else np.abs(X) ** 2
    nz = omega > 0
    r_om = np.argsort(np.argsort(omega[nz]))
    r_p = np.argsort(np.argsort(p[nz]))
    out["psd_spearman"] = float(np.corrcoef(r_om, r_p)[0, 1])
    flat = G.reshape(G.shape[0], -1)
    num = float(np.sum(flat[1:] * flat[:-1]))
    den = float(np.sum(flat * flat))
    out["lag1_rho"] = num / den if den > 0 else float("nan")
    out["spectrum"] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["resnet-cifar", "nanogpt-gpu"])
    ap.add_argument("--steps", type=int, default=0, help="override ladder run length")
    args = ap.parse_args()

    contract = load_task(args.task)
    base, lrs = task_preflight_config(args.task)
    if args.steps:
        base["steps"] = args.steps
    seed_all(base["seed"])
    det = setup_determinism()
    wlen = base["windows"][0][1]
    wmean, wq = white_band(wlen)
    print(f"preflight {args.task}: lrs={lrs} steps={base['steps']} window={base['windows']}"
          f" white(0.6pi) mean={wmean:.3f} q97.5={wq:.3f}")

    rows = []
    spectra = {}
    primary = None
    for lr in lrs:
        cfg = {**base, "lr": lr}
        t0 = time.time()
        model = contract.build_model(cfg)
        data = contract.data_loader(cfg)
        targets = contract.target_modules(model, cfg)
        if primary is None:
            primary = list(targets)[min(1, len(targets) - 1)]  # a mid-depth matrix
        out = contract.end_to_end_train(model, data, targets, cfg,
                                        dict(kind="ema", beta=0.0))
        row = dict(lr=lr, diverged=bool(out["diverged"]),
                   n_steps=int(len(out["losses"])),
                   loss0=float(out["losses"][0]) if len(out["losses"]) else float("nan"),
                   loss_end=float(out["losses"][-1]) if len(out["losses"]) else float("nan"),
                   val=float(out["val_loss"]), secs=round(time.time() - t0, 1))
        if not out["diverged"]:
            G = out["raw"][(0, primary)]
            complete = G.shape[0] == wlen
            row["window_complete"] = complete
            if complete:
                d = stream_diags(G)
                spectra[str(lr)] = d.pop("spectrum")
                row.update({k: round(float(v), 4) for k, v in d.items()})
        rows.append(row)
        print(f"  lr={lr}: " + " ".join(f"{k}={v}" for k, v in row.items() if k != "lr"))

    done = [r for r in rows if not r["diverged"] and r.get("window_complete")]
    p0 = len(done) >= 2 and all(np.isfinite(r["loss0"]) for r in rows)
    p1_cells = [r for r in done
                if r["hfer_rect_0.6"] > wq and r["psd_spearman"] >= 0.5]
    p1 = len(p1_cells) >= 1
    p2 = all(np.isfinite(r["lag1_rho"]) for r in done) and \
        (p1 and any(abs(r["lag1_rho"]) > 0.2 for r in p1_cells))
    ok = p0 and p1 and p2
    print(f"\n  P0 wiring ({len(done)} complete): {'PASS' if p0 else 'FAIL'}")
    print(f"  P1 regime presence ({len(p1_cells)} cells over white q97.5={wq:.3f}): "
          f"{'PASS' if p1 else 'FAIL'}")
    print(f"  P2 temporal structure: {'PASS' if p2 else 'FAIL'}")
    print(f"  preflight {args.task}: {'PASS' if ok else 'FAIL'}")

    arrays = dict(lrs=np.array(lrs), white=np.array([wmean, wq]),
                  **{f"spec_{k}": v for k, v in spectra.items()})
    metrics = dict(gate_pass=ok, p0=p0, p1=p1, p2=p2, primary_target=primary,
                   white_q975=wq,
                   **{f"lr{r['lr']}_{k}": r[k] for r in rows for k in
                      ("diverged", "loss_end", "hfer_rect_0.6", "psd_spearman", "lag1_rho")
                      if k in r})
    run_id = log_run(task=args.task, probe="preflight",
                     key=f"ladder{len(lrs)}_T{base['steps']}",
                     config={**base, "lrs": lrs, "stage": "explore",
                             "determinism": det, "primary_target": primary},
                     metrics=metrics, arrays=arrays, extra_files=[__file__])
    print(f"  logged run: {run_id}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
