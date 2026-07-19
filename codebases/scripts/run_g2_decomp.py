"""G2 — state vs. residual decomposition + Hessian subspace (plan_v5 §3.2; review Exp B,
packages #2–3). THE CAMPAIGN KILL-SWITCH: if the four support conditions fail, the
river-valley reading does not transfer to CIFAR — stop, write the negative result.

Paired decompositions on ACTUAL beta trajectories (Codex plan-review fix): the raw stream
is the pre-EMA mini-batch gradient g^mb_t, recorded along beta = 0 stable cells AND along
beta >= 0.9 momentum-stabilized cells — including LRs where beta = 0 diverges, which is
where the mechanism claim lives. Per selected cell: 3 diagnostic windows (early /
EoS-onset / late, 2048 steps), per-step gbar^LB on a fixed 4096-image probe batch
(lb_every = 1), and top-16 TARGET-RESTRICTED Lanczos eigenpairs at each window midpoint.

Stages:
  select   pick ~12 cells from the G1 grid cache (4 LR regimes x {beta0-or-0.9, 0.9,
           0.99}, seed 0), write g2.cells into resnet-cifar/config.yaml — predeclaration;
           commit before `run`.
  run      execute the decomposition runs (--cells i,j to shard across jobs). Raw G and
           GLB fp16 streams go to results/raw/<key>/ (transient tier, ~50–100 GB);
           reductions are computed in-sink and cached small.
  report   evaluate the four predeclared gates (below) over all cells.

Predeclared gates (the review's four support conditions, operationalized as E12 did —
written into every run config at launch):
  (1) state dominance      share_lb = Eh(LB)/(Eh(LB)+Eh(xi)) >= 0.5 per cell-window
                           (window 1, EoS-onset, is the scored one; others reported);
  (2) sharp concentration  HFER(LB in P_top) >= HFER(LB out) + 0.10 AND the top-16
                           subspace captures >= 0.25 of the LB high band;
  (3) low-freq outside     low-band share of the OUT-of-subspace LB component exceeds
                           the in-subspace low-band share by >= 0.10;
  (4) benefit coupling     across LR regimes, G1's divergence-counted momentum benefit
                           increases with the high-band state share (Spearman rho > 0
                           over >= 4 regime points; exact values reported).
Whiteness of xi (supporting, E12 gate-B style): |HFER(xi) - white| <= 0.05 at 0.6pi rect,
reported at all 3 cutoffs + Hann, plus Ljung-Box rejection fraction over top-eigen and
fixed random projections (chi2, L = 20 lags, 5% level).

Run:  cd codebases && python scripts/run_g2_decomp.py --stage select
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
from scipy import stats as sstats

_CODEBASES = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODEBASES))
sys.path.insert(0, str(_CODEBASES / "resnet-cifar" / "adapter"))

from core import metrics as Me  # noqa: E402
from core.device import seed_all, setup_determinism  # noqa: E402
from core.lbprobe import state_residual_bands, subspace_split  # noqa: E402
from core.logging import log_run  # noqa: E402
from core.spectral_stream import chunked_stream_reductions  # noqa: E402
import contract  # noqa: E402

CFG_PATH = _CODEBASES / "resnet-cifar" / "config.yaml"
RAW_DIR = _CODEBASES / "results" / "raw"
CACHE = _CODEBASES / "results" / "cache"
HI_FRACS = (0.5, 0.6, 0.7)
GATES = ("1: share_lb>=0.5 (w1); 2: HFER_in>=HFER_out+0.10 and share_high_in>=0.25; "
         "3: low-band share out-of-subspace >= in + 0.10; "
         "4: Spearman(benefit, state share)>0 across regimes")


def load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def ljung_box_p(x: np.ndarray, L: int = 20) -> float:
    """Ljung-Box white-noise test p-value on a scalar series (chi2, L lags)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom <= 0 or n <= L + 1:
        return float("nan")
    r = np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, L + 1)])
    q = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, L + 1)))
    return float(1.0 - sstats.chi2.cdf(q, L))


def whiteness_stats(xi: np.ndarray, eigvecs: np.ndarray | None, seed: int = 0) -> dict:
    """HFER-vs-white at 3 cutoffs (+Hann) and Ljung-Box rejection fraction of xi."""
    out = {}
    r = chunked_stream_reductions(xi, None, 0.0, 0.0, hi_fracs=HI_FRACS)
    for hf in HI_FRACS:
        out[f"hfer_xi_{hf}"] = r[f"hfer{hf}"]
    out["hfer_xi_0.6_hann"] = r["hfer0.6_hann"]
    flat = xi.reshape(xi.shape[0], -1)
    rng = np.random.default_rng(seed + 71)
    dirs = [rng.standard_normal(flat.shape[1]) for _ in range(8)]
    dirs = [d / np.linalg.norm(d) for d in dirs]
    if eigvecs is not None:
        dirs += [v.reshape(-1) for v in eigvecs[:4]]
    ps = [ljung_box_p(flat @ d) for d in dirs]
    ps = [p for p in ps if np.isfinite(p)]
    out["lb_reject_frac"] = float(np.mean([p < 0.05 for p in ps])) if ps else float("nan")
    out["lb_min_p"] = float(np.min(ps)) if ps else float("nan")
    return out


def low_share_from_spec(spec: np.ndarray, T: int, lo_frac: float = 0.15) -> float:
    """Low-band energy share (omega <= lo_frac*pi, DC excluded) from a rect spectrum."""
    omega = 2.0 * np.pi * np.fft.rfftfreq(T)
    low = (omega <= lo_frac * np.pi) & (omega > 0)
    nz = omega > 0
    return float(np.sum(spec[low]) / (np.sum(spec[nz]) + 1e-300))


def run_cell(y: dict, lr: float, beta: float, seed: int) -> str:
    g2 = y["g2"]
    cfg = dict(seed=seed, device="auto", dataset=y["data"]["dataset"],
               batch_size=y["data"]["batch_size"], augment=y["data"]["augment"],
               norm=y["model"]["norm"], lr=lr)
    cfg.update(y["train"])
    cfg.update(lb_every=g2["lb_every"], lb_batch=g2["lb_batch"], lb_chunk=g2["lb_chunk"],
               lb_targets=list(g2["lb_targets"]), eig_at_mid=True, eig_k=g2["eig_k"],
               eig_m=g2["eig_m"], eig_batch=g2["eig_batch"], eig_chunk=g2["eig_chunk"],
               eig_targets=list(g2["lb_targets"]), sketch_k=0,
               stage="confirm", gates=GATES, protocol="P-thy")
    seed_all(seed)
    cfg["determinism"] = setup_determinism()
    model = contract.build_model(cfg)
    data = contract.data_loader(cfg)
    targets = contract.target_modules(model, cfg)
    key = f"decomp_lr{lr:g}_b{beta:g}_s{seed}"
    (RAW_DIR / key).mkdir(parents=True, exist_ok=True)

    coef_g = 1.0 - beta  # G2 runs are P-thy EMA
    red: dict[str, float | np.ndarray] = {}
    pending_G: dict[tuple, np.ndarray] = {}
    raw_paths: list[str] = []

    def sink(wi, tn, kind, arr, m0):
        # Raw persistence policy (storage §3.4): full fp16 G+GLB for the lb_targets and
        # conv1; layer4.1.conv2 (10 GB/window) is reduced-only. All reductions are
        # chunk-bounded (core.spectral_stream) — no float64 copies of big streams.
        p = RAW_DIR / key / f"w{wi}_{tn}_{kind}.npz"
        if kind == "G":
            if tn in cfg["lb_targets"] or tn == "conv1":
                np.savez_compressed(p, stream=arr, m0=m0)
                raw_paths.append(str(p))
            if tn in cfg["lb_targets"]:
                pending_G[(wi, tn)] = arr
            else:
                r = chunked_stream_reductions(arr, m0, beta, coef_g, hi_fracs=HI_FRACS)
                red.update({f"w{wi}_{tn}_hfer{hf}": r[f"hfer{hf}"] for hf in HI_FRACS})
                red[f"w{wi}_{tn}_msr"] = r["msr"]
            return
        # kind == "GLB": paired with the stashed G stream (lb_every = 1 -> aligned)
        np.savez_compressed(p, stream=arr)
        raw_paths.append(str(p))
        G = pending_G.pop((wi, tn), None)
        if G is None or G.shape[0] != arr.shape[0]:
            red[f"w{wi}_{tn}_pair_error"] = 1.0
            return
        bands = state_residual_bands(G, arr, hi_frac=0.6, window="rect",
                                     return_specs=True)
        red.update({f"spec_w{wi}_{tn}_{s}": bands.pop(f"spec_{s}")
                    for s in ("mb", "lb", "xi")})
        red.update({f"w{wi}_{tn}_{k}": v for k, v in bands.items()})
        for hf in (0.5, 0.7):
            b2 = state_residual_bands(G, arr, hi_frac=hf, window="rect")
            red[f"w{wi}_{tn}_share_lb_{hf}"] = b2["share_lb"]
        bh = state_residual_bands(G, arr, hi_frac=0.6, window="hann")
        red[f"w{wi}_{tn}_share_lb_hann"] = bh["share_lb"]

    t0 = time.time()
    out = contract.end_to_end_train(model, data, targets, cfg,
                                    dict(kind="ema", beta=beta), raw_sink=sink)
    secs = time.time() - t0

    # geometric analyses: reload the saved LB streams, project on the restricted eigvecs
    for er in out["eig_rec"]:
        wi, tn = er["window"], er["target"]
        f = RAW_DIR / key / f"w{wi}_{tn}_GLB.npz"
        fg = RAW_DIR / key / f"w{wi}_{tn}_G.npz"
        if not (f.exists() and fg.exists() and er["eigvecs"] is not None):
            continue
        L32 = np.load(f)["stream"].astype(np.float32)
        G32 = np.load(fg)["stream"].astype(np.float32)
        V = er["eigvecs"]
        c, r = subspace_split(L32, V)
        T = L32.shape[0]
        omega = 2.0 * np.pi * np.fft.rfftfreq(T)
        high = omega >= 0.6 * np.pi
        nz = omega > 0
        rc = chunked_stream_reductions(c, None, 0.0, 0.0, hi_fracs=(0.6,))
        rr = chunked_stream_reductions(r, None, 0.0, 0.0, hi_fracs=(0.6,))
        red[f"w{wi}_{tn}_hfer_in"] = rc["hfer0.6"]
        red[f"w{wi}_{tn}_hfer_out"] = rr["hfer0.6"]
        eh_in = float(np.sum(rc["spec"][high]))
        eh_out = float(np.sum(rr["spec"][high]))
        red[f"w{wi}_{tn}_share_high_in"] = eh_in / (eh_in + eh_out + 1e-300)
        red[f"w{wi}_{tn}_low_in"] = low_share_from_spec(rc["spec"], T)
        red[f"w{wi}_{tn}_low_out"] = low_share_from_spec(rr["spec"], T)
        red[f"w{wi}_{tn}_hfer_dir"] = np.array(
            [Me.high_freq_energy_ratio(c[:, i], window="rect", hi_frac=0.6)
             for i in range(V.shape[0])], dtype=np.float32)
        red[f"w{wi}_{tn}_eigvals"] = er["eigvals"].astype(np.float32)
        red[f"w{wi}_{tn}_eig_resid"] = er["eig_resid"].astype(np.float32)
        red[f"w{wi}_{tn}_eta_lam"] = (lr * er["eigvals"]).astype(np.float32)
        red.update({f"w{wi}_{tn}_{k}": v for k, v in
                    whiteness_stats(G32 - L32, er["eigvecs"], seed).items()})

    wlen = cfg["windows"][0][1]
    rng = np.random.default_rng(0)
    white = Me.high_freq_energy_ratio(rng.standard_normal((wlen, 8, 8)), window="rect",
                                      hi_frac=0.6)
    metrics = dict(diverged=bool(out["diverged"]), n_steps=int(len(out["losses"])),
                   spikes=int(out["spikes"]), val_loss=float(out["val_loss"]),
                   val_acc=float(out["val_metric"]), secs=round(secs, 1),
                   white_baseline=float(white),
                   lam_tail=float(np.median(out["lam_trace"][-5:, 1]))
                   if out["lam_trace"].size else float("nan"),
                   raw_paths=";".join(raw_paths))
    metrics.update({k: float(v) for k, v in red.items() if isinstance(v, float)})
    arrays = dict(losses=out["losses"].astype(np.float32), eval_trace=out["eval_trace"],
                  lam_trace=out["lam_trace"], chi_trace=out["chi_trace"],
                  **{k: v for k, v in red.items() if isinstance(v, np.ndarray)})
    rid = log_run(task="resnet-cifar", probe="g2", key=key, config=cfg,
                  metrics=metrics, arrays=arrays, extra_files=[__file__])
    s1 = metrics.get("w1_layer3.0.conv1_share_lb", float("nan"))
    print(f"  {rid}: div={metrics['diverged']} share_lb(w1,l3)={s1:.3f} "
          f"({secs:.0f}s)", flush=True)
    return rid


def stage_select(y: dict) -> None:
    """Pick the ~12 decomposition cells from the G1 grid cache; write g2.cells."""
    import csv
    with open(_CODEBASES / "results" / "index" / "runs.csv") as f:
        rows = [r for r in csv.DictReader(f)
                if r["task"] == "resnet-cifar" and r["probe"] == "g1"
                and r["key"].startswith("grid_")]
    stat: dict[tuple, dict] = {}
    for r in rows:
        m = json.loads((CACHE / r["run_id"] / "metrics.json").read_text())
        c = json.loads((CACHE / r["run_id"] / "config.json").read_text())
        b = [p for p in r["key"].split("_") if p.startswith("b")][0]
        cell = (float(c["lr"]), float(b[1:]))
        stat.setdefault(cell, []).append(m["diverged"])
    lrs = sorted({c[0] for c in stat})
    cells = []
    for lr in lrs:
        b0_ok = (lr, 0.0) in stat and not any(stat[(lr, 0.0)])
        picks = ([0.0] if b0_ok else []) + [0.9, 0.99]
        cells += [[lr, b, 0] for b in picks[:3]]
    print(f"selected {len(cells)} cells: {cells}")
    y["g2"]["cells"] = cells
    with open(CFG_PATH, "w") as f:
        yaml.safe_dump(y, f, sort_keys=False)
    print(f"written to {CFG_PATH} — COMMIT before `run`.")


def stage_run(y: dict, which: list[int] | None) -> None:
    cells = y["g2"].get("cells") or []
    if not cells:
        raise SystemExit("run `select` + commit first: g2.cells missing")
    for i, (lr, beta, seed) in enumerate(cells):
        if which is None or i in which:
            run_cell(y, float(lr), float(beta), int(seed))


def stage_report(y: dict) -> None:
    import csv
    with open(_CODEBASES / "results" / "index" / "runs.csv") as f:
        rows = [r for r in csv.DictReader(f)
                if r["task"] == "resnet-cifar" and r["probe"] == "g2"]
    if not rows:
        raise SystemExit("no g2 runs cached")
    per_cell = {}
    for r in rows:
        m = json.loads((CACHE / r["run_id"] / "metrics.json").read_text())
        c = json.loads((CACHE / r["run_id"] / "config.json").read_text())
        per_cell[(c["lr"], _beta_of(r["key"]))] = m
    tn_scored = y["g2"]["lb_targets"][1]  # mid-depth matrix is the scored one
    w = 1  # EoS-onset window is scored; others reported

    def col(m, name, default=float("nan")):
        return m.get(f"w{w}_{tn_scored}_{name}", default)

    print("G2 report (scored: window 1, target " + tn_scored + "):")
    g1v, g2v, g3v = [], [], []
    for (lr, b), m in sorted(per_cell.items()):
        s, hi, ho = col(m, "share_lb"), col(m, "hfer_in"), col(m, "hfer_out")
        shi, li, lo = col(m, "share_high_in"), col(m, "low_in"), col(m, "low_out")
        hx, wb = col(m, "hfer_xi_0.6"), m.get("white_baseline", float("nan"))
        print(f"  lr={lr:g} b={b:g}: share_lb={s:.3f} hfer_in/out={hi:.3f}/{ho:.3f} "
              f"share_high_in={shi:.3f} low_in/out={li:.3f}/{lo:.3f} "
              f"|hfer_xi-white|={abs(hx - wb):.3f} "
              f"lb_reject={col(m, 'lb_reject_frac'):.2f}")
        g1v.append(s >= 0.5)
        g2v.append(hi >= ho + 0.10 and shi >= 0.25)
        g3v.append(lo >= li + 0.10)
    n = len(per_cell)
    g1 = np.mean(g1v) >= 0.75 if n else False   # >= 3/4 of cells (seed-0 cells)
    g2 = np.mean(g2v) >= 0.75 if n else False
    g3 = np.mean(g3v) >= 0.75 if n else False
    print(f"\n  gate 1 state dominance: {sum(g1v)}/{n} {'PASS' if g1 else 'FAIL'}")
    print(f"  gate 2 sharp concentration: {sum(g2v)}/{n} {'PASS' if g2 else 'FAIL'}")
    print(f"  gate 3 low-freq outside: {sum(g3v)}/{n} {'PASS' if g3 else 'FAIL'}")
    print("  gate 4 (benefit coupling) needs the G1 benefit table: run "
          "run_g1_resnet.py --stage report and correlate (documented in summary.md).")
    verdict = ("clear so far" if (g1 and g2) else
               "TRIPPED — stop, write the negative-result summary, report to Leon")
    print(f"\n  KILL-SWITCH: {verdict}")


def _beta_of(key: str) -> float:
    for part in key.split("_"):
        if part.startswith("b") and part[1:].replace(".", "").isdigit():
            return float(part[1:])
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "run", "report"])
    ap.add_argument("--cells", type=str, default=None,
                    help="comma-separated cell indices for job sharding")
    args = ap.parse_args()
    y = load_cfg()
    which = [int(x) for x in args.cells.split(",")] if args.cells else None
    if args.stage == "select":
        stage_select(y)
    elif args.stage == "run":
        stage_run(y, which)
    else:
        stage_report(y)


if __name__ == "__main__":
    main()
