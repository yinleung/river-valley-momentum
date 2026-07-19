"""Experiment E3 extension — CL-2 heatmap, escape frequency, and the beta-window sweep
(fable_dgs_v1.md §8.2 item 2).

Three parts, one logged run (task `cl2_grid`):

  part 1  straight-valley grid over (eta*lam, beta): measured stationary tail rms of the hill
          coordinate vs the CL-2 guide (core.closedloop.stationary_hill_std). The linear model
          is exact here, so the ratio should be 1 within seed error wherever the loop is
          stable; cells beyond the CL-1 threshold (eta*lam >= 2*T_eff) must diverge.
  part 2  curved-valley escape frequency vs beta at eta*lam in {1.8, 2.5} (the E8 landscape,
          finer beta grid): the tube-restoration curve — escape frequency drops where the
          CL-2 rms is small against the tube radius, rises again only via divergence at
          beta = 0 beyond threshold.
  part 3  beta-window sweep: fixed eta*lam = 1.8 and hill lam = 10, conditioning
          lam/mu in {10, 100, 1000} on the curved valley. Each conditioning is run and scored
          over its own traveling horizon K = 3/(eta*mu) — the phase in which the river is
          actively traversed (the quadratic river is exponential, so at small lam/mu it is
          over within ~30 steps and any tail metric measures a static problem where maximal
          filtering trivially wins). Good-beta window = {beta : tube-escape frequency <= 0.25
          AND mean river-following lag 1 - align(m, r) <= 1.25 x its own minimum over beta}
          — the two mechanism metrics whose edges are the T6' edges (escape excludes small
          beta, lag excludes large beta). The lag tolerance is relative because the absolute
          lag level under noise is set by the buffer's noise floor (~sigma/sqrt(T_eff)
          against the river gradient), not by tracking error; the *shape* over beta is the
          mechanism signal. River progress and travel loss are logged as reference; at
          lam/mu = 1000 and sigma = 2 progress is diffusion-limited (the hill-coupled x
          random walk exceeds the river distance), so performance there is not a lag
          readout — the mechanism window is the falsifiable object, and mechanism-vs-
          performance is E7's question. T6' predicts the upper edge 1/T_eff >~ eta*mu:
          the window widens with lam/mu.

Decision gates:
  1. part 1: |ratio - 1| <= 2 x seed-SE + 0.02 in >= 90% of comfortably-stable cells
     (eta*lam <= 0.9 x 2*T_eff), and every cell beyond the CL-1 threshold diverges;
  2. part 3: the window's upper beta edge is nondecreasing in lam/mu and spans at least
     two grid steps from smallest to largest conditioning.

Run:  cd codebases && python scripts/run_e3_heatmap.py
"""
from __future__ import annotations

import pathlib
import sys
from functools import partial

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import closedloop as Cl  # noqa: E402
from core import metrics as Me  # noqa: E402
from core.landscapes import CurvedValley, StraightValley, gaussian_isotropic  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
import rivervalley_sim  # noqa: E402
from rivervalley_sim import simulate  # noqa: E402

SHA = code_sha([__file__, rivervalley_sim.__file__])

CFG = dict(lam=10.0, mu=0.1, x_star=5.0, sigma=2.0, n_seeds=8, n_seeds_window=16,
           T_grid=6000, T_curved=300, tube_radius=2.0, a=2.0, k=0.9,
           etalams=[0.3, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0],
           betas_grid=[0.0, 0.5, 0.9, 0.95, 0.99],
           betas_fine=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999],
           conds=[10.0, 100.0, 1000.0], lag_rel_tol=1.25)


def part1_straight():
    """(n_etalam, n_beta) arrays: measured rms, predicted rms, ratio, ratio seed-SE, div."""
    land = StraightValley(mu=CFG["mu"], lam=CFG["lam"], x_star=CFG["x_star"])
    noise_fn = partial(gaussian_isotropic, sigma=CFG["sigma"])
    nE, nB = len(CFG["etalams"]), len(CFG["betas_grid"])
    rms = np.full((nE, nB), np.nan)
    ratio = np.full((nE, nB), np.nan)
    ratio_se = np.full((nE, nB), np.nan)
    pred = np.full((nE, nB), np.nan)
    div = np.zeros((nE, nB))
    T = CFG["T_grid"]
    for i, el in enumerate(CFG["etalams"]):
        eta = el / CFG["lam"]
        for j, b in enumerate(CFG["betas_grid"]):
            pred[i, j] = Cl.stationary_hill_std(eta, CFG["lam"], CFG["sigma"], b)
            rs = []
            for sd in range(CFG["n_seeds"]):
                rng = np.random.default_rng(50_000 + 1000 * sd)
                s = simulate(land, b, eta, np.array([-3.0, 1.0]), T,
                             rng=rng, noise_fn=noise_fn)
                if s["diverged"]:
                    div[i, j] += 1
                    continue
                tail = s["d"][T // 2:]
                rs.append(float(np.sqrt(np.mean(tail**2))))
            if rs:
                rms[i, j] = np.mean(rs)
                if np.isfinite(pred[i, j]):
                    r = np.array(rs) / pred[i, j]
                    ratio[i, j] = r.mean()
                    ratio_se[i, j] = r.std(ddof=1) / np.sqrt(len(r))
    return dict(rms=rms, ratio=ratio, ratio_se=ratio_se, pred=pred, div=div)


def part2_escape():
    """Escape frequency + rms vs beta on the curved valley at eta*lam in {1.8, 2.5}."""
    land = CurvedValley(mu=CFG["mu"], lam=CFG["lam"], x_star=CFG["x_star"],
                        a=CFG["a"], k=CFG["k"])
    w0 = np.array([-3.0, land.f(-3.0) + 1.0])
    noise_fn = partial(gaussian_isotropic, sigma=CFG["sigma"])
    out = {}
    for el in (1.8, 2.5):
        eta = el / CFG["lam"]
        esc, rms, pred, ndiv = [], [], [], []
        for b in CFG["betas_fine"]:
            e_flags, r_vals = [], []
            nd = 0
            for sd in range(CFG["n_seeds"]):
                rng = np.random.default_rng(70_000 + 1000 * sd)
                s = simulate(land, b, eta, w0, CFG["T_curved"], rng=rng, noise_fn=noise_fn)
                if s["diverged"]:
                    nd += 1
                    e_flags.append(True)
                    continue
                off = land.hill_coordinate(s["W"][1:])
                e_flags.append(Me.escape_fraction(off, CFG["tube_radius"]) > 0)
                r_vals.append(np.sqrt(np.mean(off[CFG["T_curved"] // 2:] ** 2)))
            esc.append(np.mean(e_flags))
            rms.append(np.mean(r_vals) if r_vals else np.nan)
            pred.append(Cl.stationary_hill_std(eta, CFG["lam"], CFG["sigma"], b))
            ndiv.append(nd)
        out[el] = dict(esc=np.array(esc), rms=np.array(rms), pred=np.array(pred),
                       ndiv=np.array(ndiv))
    return out


def part3_window():
    """River progress + escape over the traveling window per conditioning; good-beta windows."""
    eta = 0.18
    noise_fn = partial(gaussian_isotropic, sigma=CFG["sigma"])
    shape = (len(CFG["conds"]), len(CFG["betas_fine"]))
    progress, esc_freq, loss_travel, lag_mean = (np.full(shape, np.nan) for _ in range(4))
    travel_T = []
    for ci, cond in enumerate(CFG["conds"]):
        mu = CFG["lam"] / cond
        K = int(np.ceil(3.0 / (eta * mu)))  # traveling window: 3 river time constants
        travel_T.append(K)
        land = CurvedValley(mu=mu, lam=CFG["lam"], x_star=CFG["x_star"],
                            a=CFG["a"], k=CFG["k"])
        w0 = np.array([-3.0, land.f(-3.0) + 1.0])
        for j, b in enumerate(CFG["betas_fine"]):
            pr, ef, lt, lg = [], [], [], []
            for sd in range(CFG["n_seeds_window"]):
                rng = np.random.default_rng(90_000 + 1000 * sd)
                s = simulate(land, b, eta, w0, K, rng=rng, noise_fn=noise_fn)
                if s["diverged"]:
                    pr.append(-np.inf)
                    ef.append(True)
                    lt.append(np.inf)
                    lg.append(1.0)
                    continue
                W = s["W"][1:]
                off = land.hill_coordinate(W)
                pr.append((W[K - 1, 0] - w0[0]) / (CFG["x_star"] - w0[0]))
                ef.append(Me.escape_fraction(off[:K], CFG["tube_radius"], tail_frac=1.0) > 0)
                lt.append(np.mean([land.loss(W[t]) for t in range(K)]))
                lg.append(float(np.mean(Me.lag(s["M"], s["R"]))))
            progress[ci, j] = np.mean(pr)
            esc_freq[ci, j] = np.mean(ef)
            loss_travel[ci, j] = np.mean(lt)
            lag_mean[ci, j] = np.mean(lg)
    windows = {}
    for ci, cond in enumerate(CFG["conds"]):
        windows[cond] = ((esc_freq[ci] <= 0.25)
                         & (lag_mean[ci] <= CFG["lag_rel_tol"] * np.nanmin(lag_mean[ci])))
    return progress, esc_freq, loss_travel, lag_mean, windows, travel_T


def main() -> str:
    betas_g, etalams, betas_f = CFG["betas_grid"], CFG["etalams"], CFG["betas_fine"]

    p1 = part1_straight()
    print(f"\nE3-heatmap part 1: straight-valley rms/CL-2 ratio (seeds={CFG['n_seeds']}, "
          f"T={CFG['T_grid']})")
    print("  eta*lam \\ beta " + " ".join(f"{b:>12}" for b in betas_g))
    gate_cells, gate_bad, div_ok = 0, 0, True
    for i, el in enumerate(etalams):
        row = []
        for j, b in enumerate(betas_g):
            teff = (1 + b) / (1 - b)
            if el >= 2 * teff:  # beyond CL-1: must diverge
                all_div = p1["div"][i, j] == CFG["n_seeds"]
                div_ok = div_ok and all_div
                row.append("   diverged " if all_div else " NOT-DIV(!) ")
                continue
            r, se = p1["ratio"][i, j], p1["ratio_se"][i, j]
            row.append(f"{r:6.3f}±{se:.3f}")
            if el <= 0.9 * 2 * teff:
                gate_cells += 1
                if abs(r - 1.0) > 2 * se + 0.02:
                    gate_bad += 1
        print(f"  {el:>8}       " + " ".join(f"{c:>12}" for c in row))
    g1 = (gate_bad <= 0.1 * gate_cells) and div_ok
    print(f"  gate 1: {gate_cells - gate_bad}/{gate_cells} stable cells within 2·SE+0.02, "
          f"unstable cells all diverge: {div_ok}  -> {'PASS' if g1 else 'FAIL'}")

    p2 = part2_escape()
    print(f"\nE3-heatmap part 2: curved-valley escape frequency (tube R={CFG['tube_radius']})")
    for el, d in p2.items():
        print(f"  eta*lam={el}: beta " + " ".join(f"{b:>7}" for b in betas_f))
        print(f"           esc_freq " + " ".join(f"{v:>7.2f}" for v in d["esc"]))
        print(f"           rms/pred " + " ".join(
            f"{v / p:>7.2f}" if np.isfinite(v) and np.isfinite(p) else f"{'--':>7}"
            for v, p in zip(d["rms"], d["pred"])))

    progress, esc_w, loss_travel, lag_mean, windows, travel_T = part3_window()
    print(f"\nE3-heatmap part 3: good-beta window vs conditioning "
          f"(window = escape freq <= 0.25 & mean lag <= {CFG['lag_rel_tol']} x min; "
          f"traveling windows T={travel_T})")
    print("  lam/mu \\ beta " + " ".join(f"{b:>8}" for b in betas_f))
    edges = []
    for ci, cond in enumerate(CFG["conds"]):
        marks = ["   in   " if w else "   .    " for w in windows[cond]]
        print(f"  {cond:>8}     " + " ".join(f"{m:>8}" for m in marks))
        inw = np.where(windows[cond])[0]
        edges.append(betas_f[inw.max()] if len(inw) else np.nan)
    for name, arr in (("lag", lag_mean), ("esc_freq", esc_w), ("progress", progress)):
        print(f"  {name} rows: " + "; ".join(
            f"lam/mu={int(c)}: " + " ".join(f"{v:.2f}" for v in arr[ci])
            for ci, c in enumerate(CFG["conds"])))
    g2 = all(edges[i + 1] >= edges[i] for i in range(len(edges) - 1)) \
        and betas_f.index(edges[-1]) - betas_f.index(edges[0]) >= 2
    print(f"  window upper edges (beta): {edges}  -> gate 2 {'PASS' if g2 else 'FAIL'}")

    gate = g1 and g2
    print(f"\n  decision gate: {'PASS' if gate else 'FAIL'}")

    arrays = dict(etalams=np.array(etalams), betas_grid=np.array(betas_g),
                  betas_fine=np.array(betas_f), conds=np.array(CFG["conds"]),
                  grid_rms=p1["rms"], grid_ratio=p1["ratio"], grid_ratio_se=p1["ratio_se"],
                  grid_pred=p1["pred"], grid_div=p1["div"], window_progress=progress,
                  window_esc=esc_w, window_lag=lag_mean, window_loss=loss_travel,
                  travel_T=np.array(travel_T))
    for el, d in p2.items():
        tag = str(el).replace(".", "p")
        for k2 in ("esc", "rms", "pred", "ndiv"):
            arrays[f"esc{tag}_{k2}"] = d[k2]
    for ci, cond in enumerate(CFG["conds"]):
        arrays[f"window_in_{int(cond)}"] = windows[cond].astype(float)
    metrics = dict(gate_pass=gate, gate1_grid=g1, gate2_window=g2,
                   grid_cells=gate_cells, grid_cells_bad=gate_bad,
                   window_edges={str(int(c)): float(e) for c, e in zip(CFG["conds"], edges)})
    run_id = log_run(task="cl2_grid", probe="sweep",
                     key=f"el{len(etalams)}_b{len(betas_g)}_seeds{CFG['n_seeds']}",
                     config=CFG, metrics=metrics, arrays=arrays, sha=SHA)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
