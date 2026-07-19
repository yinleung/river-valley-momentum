"""G1 — ResNet-18/CIFAR closed-loop LR x beta sweep (plan_v5 §3.2; review Exp A, package #1).

Stages, in launch order (each stage = one or more pjsub jobs via scripts/wisteria/submit.sh):

  scan         beta = 0 coarse LR scan (g1.scan_lrs x 1 seed, stage: explore). Classifies
               each LR {subcritical | intermediate | near-edge | unstable} from divergence
               + chi = eta*lam_hat + stream HFER, prints the four-regime proposal, and
               WRITES g1.grid_lrs into resnet-cifar/config.yaml (predeclaration: the grid
               is fixed, committed, before any grid run launches).
  preregister  §3.0.10 exact-simulator predictions from core.closedloop for the declared
               grid: stability-boundary shape in (eta, beta), CL-3 loss ordering in beta
               at fixed eta, the beta-shifted sharpness-equilibrium lines 2*T_eff/eta, and
               theta_hat resonance table for G3. Shapes and orderings, never constants.
               Logged as one run record; commit before `grid`.
  grid         P-thy: g1.betas x g1.grid_lrs x g1.seeds at fixed eta (stage: confirm,
               gates in every run config). --rows filters betas for job packing.
  pprac        P-prac companion: per-beta 3-point conventional-HB LR tune around
               eta_HB = eta*(1-beta) at the two interesting regimes (near-edge +
               beta0-unstable), 3 seeds.
  c100         CIFAR-100 thinned complete sweep: 3 LRs x 5 betas x 3 seeds.
  gn           GroupNorm (BN-free) control at the headline cells, 3 seeds.
  topup        5-seed extension of the headline cells (g1.headline_seeds).
  report       aggregate the four predeclared gates from results/cache/, print PASS/FAIL.

Predeclared gates (G1 card; verbatim in every grid-run config):
  (a) momentum benefit (best-beta minus beta=0, divergence-counted) increases with
      eta*lam_hat_max and with state-HFER;
  (b) some LR exists where beta = 0 diverges on all seeds and beta >= 0.9 trains;
  (c) at subcritical LR the beta-effect is small (within the seed noise band);
  (d) in EoS-regime cells the measured sharpness equilibrates near 2*T_eff/eta.

Raw-window policy (storage: §3.4): G1 cells are reduced IN-DRIVER (per-window spectra,
HFER 3 cutoffs rect + hann-0.6, MSR from the reconstructed m-stream, sketch-group spectra
-> arrays.npz, small); full fp16 raw G windows go to results/raw/<run_id>/ ONLY for
seed 0 at the EoS-onset window (index 1) on layer3.0.conv1 + linear (~11 GB across the
grid). G2 re-runs its ~12 selected cells with the full heavy decomposition probes.

Run:  cd codebases && python scripts/run_g1_resnet.py --stage scan
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import yaml

_CODEBASES = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODEBASES))
sys.path.insert(0, str(_CODEBASES / "resnet-cifar" / "adapter"))

from core import closedloop as CL  # noqa: E402
from core import metrics as Me  # noqa: E402
from core.device import seed_all, setup_determinism  # noqa: E402
from core.logging import log_run  # noqa: E402
from core.spectral_stream import chunked_stream_reductions  # noqa: E402
import contract  # noqa: E402

CFG_PATH = _CODEBASES / "resnet-cifar" / "config.yaml"
RAW_DIR = _CODEBASES / "results" / "raw"
HI_FRACS = (0.5, 0.6, 0.7)
GATES = ("a: benefit increases with eta*lam_hat and state-HFER; "
         "b: exists LR with beta=0 all-seed divergence and beta>=0.9 training; "
         "c: subcritical beta-effect within noise; "
         "d: EoS cells equilibrate sharpness near 2*T_eff/eta (EMA protocol)")


def load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def base_config(y: dict, **over) -> dict:
    base = dict(seed=0, device="auto", dataset=y["data"]["dataset"],
                batch_size=y["data"]["batch_size"], augment=y["data"]["augment"],
                norm=y["model"]["norm"])
    base.update(y["train"])
    base.update(over)
    return base


def teff(beta: float) -> float:
    return (1 + beta) / (1 - beta)


# --- per-window reductions ---------------------------------------------------
def reduce_G_window(G: np.ndarray, m0: np.ndarray, beta: float, coef_g: float,
                    key: str) -> dict:
    """Spectral reductions for one (window, target) raw stream: HFERs, MSR, spectrum.

    Chunk-bounded (core.spectral_stream): the naive float64 path needed ~40 GB for the
    layer4 conv and OOM-killed the first G1 scan job (9224311, silent SIGKILL).
    """
    r = chunked_stream_reductions(G, m0, beta, coef_g, hi_fracs=HI_FRACS)
    red: dict[str, np.ndarray | float] = {
        f"{key}_hfer{hf}": r[f"hfer{hf}"] for hf in HI_FRACS}
    red[f"{key}_hfer0.6_hann"] = r["hfer0.6_hann"]
    red[f"{key}_msr"] = r["msr"]
    red[f"spec_{key}"] = r["spec"].astype(np.float32)
    red[f"gnorm_{key}"] = r["gnorm_rms"]
    return red


def sketch_reductions(out: dict) -> dict:
    """Per-group sketch spectra + HFERs (sketch streams are small; reduced post-run)."""
    red: dict[str, np.ndarray | float] = {}
    for wi, recs in out["sketches"].items():
        if not recs:
            continue
        for grp in recs[0]:
            S = np.stack([r[grp] for r in recs]).astype(np.float64)
            _, X = Me.windowed_dft(S, window="rect")
            p = np.sum(np.abs(X) ** 2, axis=tuple(range(1, X.ndim)))
            red[f"sspec_w{wi}_{grp}"] = p.astype(np.float32)
            red[f"shfer_w{wi}_{grp}"] = Me.high_freq_energy_ratio(S, window="rect",
                                                                  hi_frac=0.6)
    return red


def run_cell(y: dict, stage: str, lr: float, beta: float, seed: int, kind: str = "ema",
             norm: str = "bn", dataset: str = "cifar10", keep_raw: bool = False,
             over: dict | None = None, resume: bool = True) -> str:
    """One training run + in-sink reductions + cache record; returns run_id.

    Raw windows are reduced AS THEY COMPLETE in the raw_sink (holding all windows in RAM
    would be ~33 GB fp16 per run); the fp16 stream itself is persisted only per the
    module-docstring raw policy (seed-0 EoS-onset window, two mid targets).
    resume=True skips a cell whose key already has a cached g1 record (crash/elapse
    recovery for multi-run jobs); `over` = stage-specific config overrides.
    """
    key = f"{stage}_{dataset}_{norm}_{kind}_lr{lr:g}_b{beta:g}_s{seed}"
    if resume:
        import csv
        idx = _CODEBASES / "results" / "index" / "runs.csv"
        if idx.exists():
            with open(idx) as f:
                hit = [r["run_id"] for r in csv.DictReader(f)
                       if r["task"] == "resnet-cifar" and r["probe"] == "g1"
                       and r["key"] == key]
            if hit:
                print(f"  [resume] {key} cached as {hit[-1]} — skipping")
                return hit[-1]
    cfg = base_config(y, lr=lr, seed=seed, norm=norm, dataset=dataset)
    if over:
        cfg.update(over)
    cfg["stage"] = "explore" if stage == "scan" else "confirm"
    cfg["gates"] = GATES
    cfg["protocol"] = "P-thy" if kind == "ema" else "P-prac"
    seed_all(seed)
    det = setup_determinism()
    cfg["determinism"] = det
    model = contract.build_model(cfg)
    data = contract.data_loader(cfg)
    targets = contract.target_modules(model, cfg)

    coef_g = (1.0 - beta) if kind == "ema" else 1.0
    red: dict[str, np.ndarray | float] = {}
    raw_paths: list[str] = []

    def sink(wi, tn, k, arr, m0):
        if k != "G":
            return  # G1 keeps only the alignment scalars from the LB probe
        red.update(reduce_G_window(arr, m0, beta, coef_g, f"w{wi}_{tn}"))
        if keep_raw and wi == 1 and tn in ("layer3.0.conv1", "linear"):
            p = RAW_DIR / key / f"w{wi}_{tn}_G.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, stream=arr, m0=m0)
            raw_paths.append(str(p))

    t0 = time.time()
    out = contract.end_to_end_train(model, data, targets, cfg,
                                    dict(kind=kind, beta=beta), raw_sink=sink)
    secs = time.time() - t0
    red.update(sketch_reductions(out))
    for (wi, tn), G in out["raw"].items():  # windows cut short by divergence
        red.update(reduce_G_window(G, out["m0"].get(f"{wi}:{tn}", np.zeros(G.shape[1:])),
                                   beta, coef_g, f"w{wi}_{tn}")
                   if G.shape[0] >= 64 else {})

    thr = 2 * teff(beta) / (lr if kind == "ema" else lr / (1 - beta))  # sharpness line
    lam = out["lam_trace"]
    lam_tail = float(np.median(lam[-5:, 1])) if lam.size else float("nan")
    metrics = dict(
        diverged=bool(out["diverged"]), n_steps=int(len(out["losses"])),
        spikes=int(out["spikes"]),
        train_loss_tail=float(np.mean(out["losses"][-500:])) if len(out["losses"]) else
        float("nan"),
        val_loss=float(out["val_loss"]), val_acc=float(out["val_metric"]),
        lam_tail=lam_tail, chi_tail=float((lr if kind == "ema" else lr / (1 - beta))
                                          * (1 - beta) * lam_tail) if np.isfinite(lam_tail)
        else float("nan"),
        sharpness_threshold=thr, secs=round(secs, 1),
        hfer_w1_mid=red.get("w1_layer3.0.conv1_hfer0.6",
                            red.get("w0_layer3.0.conv1_hfer0.6", float("nan"))),
        raw_paths=";".join(raw_paths),
    )
    arrays = dict(
        losses=out["losses"].astype(np.float32), gnorm=out["gnorm"].astype(np.float32),
        mnorm=out["mnorm"].astype(np.float32), eval_trace=out["eval_trace"],
        lam_trace=out["lam_trace"], chi_trace=out["chi_trace"],
        lb_align=np.array([[r["step"]] + [r[k] for k in sorted(r) if ":align" in k]
                           for r in out["lb_align"]], dtype=np.float32)
        if out["lb_align"] else np.zeros((0,)),
        **{k: v for k, v in red.items() if isinstance(v, np.ndarray)},
    )
    metrics.update({k: float(v) for k, v in red.items() if isinstance(v, float)})
    rid = log_run(task="resnet-cifar", probe="g1", key=key, config=cfg,
                  metrics=metrics, arrays=arrays, extra_files=[__file__])
    print(f"  {rid}: div={metrics['diverged']} tail={metrics['train_loss_tail']:.3f} "
          f"acc={metrics['val_acc']:.3f} lam={lam_tail:.1f} thr={thr:.1f} "
          f"({secs:.0f}s)", flush=True)
    return rid


# --- stages ------------------------------------------------------------------
def stage_scan(y: dict) -> None:
    """Coarse beta = 0 scan, slimmed to its classification purpose (divergence, chi,
    one-window HFER): one mid window, no sketches, no LB, lam cadence 400. The full
    3-window instrumentation is the GRID's job — at ~51 min/run (first scan attempt) the
    full config would blow both the job elapse and the family budget."""
    lrs = y["g1"]["scan_lrs"]
    slim = dict(windows=[[4000, 2048]], sketch_k=0, lb_batch=0, lam_every=400)
    rows = []
    for lr in lrs:
        rid = run_cell(y, "scan", lr, 0.0, seed=0, over=slim)
        cache = _CODEBASES / "results" / "cache" / rid / "metrics.json"
        m = json.loads(cache.read_text())
        rows.append((lr, m))
    print("\nscan classification (beta = 0):")
    classes = {}
    for lr, m in rows:
        chi = m.get("chi_tail", float("nan"))
        if m["diverged"]:
            c = "unstable"
        elif np.isfinite(chi) and chi > 1.5:
            c = "near-edge"
        elif np.isfinite(chi) and chi > 0.5:
            c = "intermediate"
        else:
            c = "subcritical"
        classes[lr] = c
        print(f"  lr={lr:g}: {c} (chi={chi if np.isfinite(chi) else float('nan'):.2f}, "
              f"div={m['diverged']}, acc={m.get('val_acc', 0):.3f})")
    # propose 6 grid LRs: 1 subcritical, 2 intermediate, 2 near-edge, 1 unstable
    def pick(cls, n):
        c = [lr for lr, k in classes.items() if k == cls]
        return c[-n:] if cls != "unstable" else c[:n]
    prop = sorted(set(pick("subcritical", 1) + pick("intermediate", 2)
                      + pick("near-edge", 2) + pick("unstable", 1)))
    print(f"  proposed grid_lrs: {prop}")
    y["g1"]["grid_lrs"] = [float(x) for x in prop]
    with open(CFG_PATH, "w") as f:
        yaml.safe_dump(y, f, sort_keys=False)
    print(f"  grid_lrs written to {CFG_PATH} — COMMIT before launching `grid`.")


def stage_preregister(y: dict) -> None:
    lrs = y["g1"].get("grid_lrs")
    if not lrs:
        raise SystemExit("run `scan` first: g1.grid_lrs missing")
    betas = y["g1"]["betas"]
    pred = {}
    for b in betas:
        pred[f"threshold_2Teff_b{b:g}"] = 2 * teff(b)   # eta*lam stability boundary
        for lr in lrs:
            pred[f"sharpline_lr{lr:g}_b{b:g}"] = 2 * teff(b) / lr
    # CL-3 ordering at fixed eta: loss decreasing in beta while stable (shape claim)
    pred["cl3_ordering"] = "loss(beta) strictly decreasing to saturation at fixed eta*lam"
    pred["divergence_shape"] = "unstable iff eta*lam_max > 2*T_eff: boundary moves UP with beta"
    theta = {}
    for b in (0.5, 0.9, 0.95, 0.99):
        try:
            theta[f"theta_b{b:g}"] = CL.resonant_frequency(b, 0.1, 20.0)
        except ValueError:
            pass
    arrays = dict(betas=np.array(betas), grid_lrs=np.array(lrs),
                  guide_eta_lam=np.array([[2 * teff(b) for b in betas]]),
                  theta_examples=np.array(sorted(theta.values())))
    log_run(task="resnet-cifar", probe="preregister", key=f"g1_grid{len(lrs)}x{len(betas)}",
            config=dict(seed=0, stage="confirm", grid_lrs=lrs, betas=betas,
                        note="shapes and orderings, never constants (§3.0.10)"),
            metrics={k: v for k, v in pred.items()}, arrays=arrays,
            extra_files=[__file__])
    print("preregistered predictions logged — commit results/ before `grid`.")


def stage_grid(y: dict, rows: list[float] | None) -> None:
    lrs = y["g1"].get("grid_lrs") or []
    if not lrs:
        raise SystemExit("run `scan` + commit first: g1.grid_lrs missing")
    betas = [b for b in y["g1"]["betas"] if rows is None or b in rows]
    for beta, lr, seed in itertools.product(betas, lrs, y["g1"]["seeds"]):
        run_cell(y, "grid", lr, beta, seed, keep_raw=(seed == 0))


def stage_pprac(y: dict, rows: list[float] | None) -> None:
    lrs = y["g1"].get("grid_lrs") or []
    two = lrs[-2:]  # near-edge + unstable rows (practitioner-relevant regimes)
    betas = [b for b in y["g1"]["betas"] if b > 0 and (rows is None or b in rows)]
    for beta, lr0, seed in itertools.product(betas, two, y["g1"]["seeds"]):
        for f in (0.5, 1.0, 2.0):  # 3-point tune around the eta_HB conversion
            run_cell(y, "pprac", lr0 * (1 - beta) * f, beta, seed, kind="hb")


def stage_c100(y: dict) -> None:
    lrs = (y["g1"].get("grid_lrs") or [])[1:4]  # 3 LRs spanning the mid regimes
    for beta, lr, seed in itertools.product(y["g1"]["betas"], lrs, y["g1"]["seeds"]):
        run_cell(y, "c100", lr, beta, seed, dataset="cifar100")


def stage_gn(y: dict) -> None:
    for lr, beta, seed in _headline_cells(y):
        run_cell(y, "gn", lr, beta, seed, norm="gn")


def stage_topup(y: dict) -> None:
    extra = [s for s in y["g1"]["headline_seeds"] if s not in y["g1"]["seeds"]]
    for lr, beta, _ in _headline_cells(y):
        for seed in extra:
            run_cell(y, "grid", lr, beta, seed, keep_raw=False)


def _headline_cells(y: dict):
    lrs = y["g1"].get("grid_lrs") or []
    picks = [(lr, b) for lr in lrs[-2:] for b in (0.0, 0.9)]  # the headline pairs
    return [(lr, b, s) for lr, b in picks for s in y["g1"]["seeds"]]


def stage_report(y: dict) -> None:
    import csv
    idx = _CODEBASES / "results" / "index" / "runs.csv"
    with open(idx) as f:
        rows = [r for r in csv.DictReader(f)
                if r["task"] == "resnet-cifar" and r["probe"] == "g1"
                and r["key"].startswith("grid_")]
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        m = json.loads((_CODEBASES / "results" / "cache" / r["run_id"] /
                        "metrics.json").read_text())
        c = json.loads((_CODEBASES / "results" / "cache" / r["run_id"] /
                        "config.json").read_text())
        cells.setdefault((c["lr"], _beta_of(r["key"])), []).append(m)
    if not cells:
        raise SystemExit("no grid cells cached yet")
    print("G1 grid summary (divergence-counted tail loss; mean over seeds):")
    grid_lrs = sorted({k[0] for k in cells})
    betas = sorted({k[1] for k in cells})
    for lr in grid_lrs:
        parts = []
        for b in betas:
            ms = cells.get((lr, b), [])
            div = sum(m["diverged"] for m in ms)
            # divergence counts as failure: diverged seeds enter as +inf loss
            vals = [float("inf") if m["diverged"] else m["train_loss_tail"] for m in ms]
            v = np.mean(vals) if vals else float("nan")
            parts.append(f"b{b:g}: {'DIV' if not np.isfinite(v) else f'{v:.3f}'}"
                         f"({div}/{len(ms)}d)")
        print(f"  lr={lr:g}: " + "  ".join(parts))
    # gate b: an LR with beta=0 all-seed divergence and beta>=0.9 all-seed training
    gb = any(all(m["diverged"] for m in cells.get((lr, 0.0), [dict(diverged=False)]))
             and any(all(not m["diverged"] for m in cells.get((lr, b), [dict(diverged=True)]))
                     for b in betas if b >= 0.9) for lr in grid_lrs)
    print(f"\n  gate (b) stability extension: {'PASS' if gb else 'FAIL'}")
    print("  gates (a)/(c)/(d) need the assembled benefit-vs-chi and sharpness overlays "
          "(rendered by Leon from cache); their numeric summaries:")
    for lr in grid_lrs:
        ms0 = cells.get((lr, 0.0), [])
        if not ms0:
            continue
        base = np.mean([float("inf") if m["diverged"] else m["train_loss_tail"]
                        for m in ms0])
        best = min((np.mean([float("inf") if m["diverged"] else m["train_loss_tail"]
                             for m in cells.get((lr, b), [])]) for b in betas if b > 0),
                   default=float("nan"))
        chi = np.nanmean([m.get("chi_tail", np.nan) for m in ms0])
        print(f"    lr={lr:g}: benefit={base - best if np.isfinite(base) else np.inf:.4f} "
              f"chi0={chi:.2f}")


def _beta_of(key: str) -> float:
    for part in key.split("_"):
        if part.startswith("b") and part[1:].replace(".", "").isdigit():
            return float(part[1:])
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["scan", "preregister", "grid", "pprac", "c100", "gn",
                             "topup", "report"])
    ap.add_argument("--rows", type=str, default=None,
                    help="comma-separated beta filter for job packing, e.g. '0.0,0.5'")
    args = ap.parse_args()
    y = load_cfg()
    rows = [float(x) for x in args.rows.split(",")] if args.rows else None
    dict(scan=stage_scan, preregister=stage_preregister,
         grid=lambda y: stage_grid(y, rows), pprac=lambda y: stage_pprac(y, rows),
         c100=stage_c100, gn=stage_gn, topup=stage_topup,
         report=stage_report)[args.stage](y)


if __name__ == "__main__":
    main()
