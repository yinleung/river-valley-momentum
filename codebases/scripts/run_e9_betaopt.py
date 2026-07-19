"""Experiment E9 — does larger beta improve river-valley optimization? (idea_v2.md sections
1 and 10: the direct optimization-quality experiment, straight noisy valley.)

The mechanism experiments (E1-E6) show *filtering*; the closed-loop grid shows CL-2
*exactness*. E9 asks the reviewer's question directly: in the noisy straight valley, does
increasing beta improve the actual optimization metrics — and exactly as CL-3 predicts,
including where it barely helps and where the beta = 0 baseline cannot run at all?

One logged run (task `beta_opt`), four parts:

  part 1 (main grid, sigma = 2)  betas x eta*lam x conditioning lam/mu, 16 seeds, T = 6000,
      tail = second half (>= 7.5 T_eff even at beta = 0.995, so the CL-3' burn-in is cleared
      in every cell; T/T_eff is logged per beta). Per cell: mean tail hill loss
      (lam/2) d^2 and total loss, stationary rms + ratio to the CL-2 guide, tube-escape
      frequency (R = 2), diverged seeds, tail-stream HFER and buffer MSR.
  part 2 (sigma scaling)  sigma in {0.5, 1.0} at eta*lam = 1.8, cond = 100: CL-2 predicts
      hill loss proportional to sigma^2 exactly.
  part 3 (normalization ablation, idea_v2.md section 10)  the same beta sweep at FIXED
      heavy-ball step eta_HB = 0.6/lam: the HB update w -= eta_HB (g + beta v) is exactly the
      EMA update at eta = eta_HB/(1-beta), so the run reuses `simulate` with that eta and the
      CL-3 guide composed the same way. Sharing the beta = 0 cell with the EMA eta*lam = 0.6
      row, the two conventions then move in OPPOSITE directions: EMA fixed-eta decreases in
      beta, HB fixed-eta_HB increases (the numerator eta grows like 1/(1-beta)).
  part 4 (report)  measured maximal relative hill-loss reduction per eta*lam row vs the CL-3
      prediction eta*lam/2 (for eta*lam < 2) and the stabilized regime beyond.

Decision gates (predeclared):
  1  stability boundary exact: every cell beyond CL-1 (eta*lam >= 2 T_eff) diverges on all
     seeds; every comfortably-stable cell (eta*lam <= 0.9 x 2 T_eff) diverges on none.
  2  CL-2/CL-3 exactness: |rms/guide - 1| <= 2 x seed-SE + 0.02 in >= 90% of comfortably-
     stable cells of the main grid (all conds).
  3  regime-scoped improvement: per (eta*lam >= 1.6, cond) row, mean tail hill loss is
     near-monotone decreasing in beta (no consecutive increase beyond 2 x SE(diff) + 2%),
     and at eta*lam in {1.6, 1.8} the measured maximal relative reduction is within 0.10 of
     the CL-3 value eta*lam/2; at eta*lam in {2.2, 2.5}, beta = 0 diverges while beta = 0.9
     runs (the stabilized regime).
  4  the improvement is regime-scoped, quantitatively: at eta*lam in {0.6, 1.2} the measured
     maximal relative hill-loss reduction is within 0.10 of eta*lam/2 (30% / 60%) — modest
     exactly where the theory says it is modest, NOT zero and NOT large.
  5  sigma scaling: per beta, hill loss ratios across sigma in {0.5, 1, 2} are within 15%
     of the predicted (sigma_i/sigma_j)^2.
  6  normalization: the HB row's hill loss is increasing in beta (Spearman >= +0.9 over
     stable cells) and matches its composed guide in >= 90% of cells (gate-2 criterion);
     the matching EMA row (eta*lam = 0.6, cond = 100) is near-monotone decreasing.

Run:  cd codebases && python scripts/run_e9_betaopt.py
"""
from __future__ import annotations

import pathlib
import sys
from functools import partial

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import closedloop as Cl  # noqa: E402
from core import metrics as Me  # noqa: E402
from core.landscapes import StraightValley, gaussian_isotropic  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
from core.momentum import effective_window  # noqa: E402
import rivervalley_sim  # noqa: E402
from rivervalley_sim import simulate  # noqa: E402

SHA = code_sha([__file__, rivervalley_sim.__file__])

CFG = dict(lam=10.0, x_star=5.0, sigma=2.0, n_seeds=16, T=6000, tube_radius=2.0,
           betas=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995],
           etalams=[0.6, 1.2, 1.6, 1.8, 2.2, 2.5],
           conds=[10.0, 100.0, 1000.0],
           sigmas_scaling=[0.5, 1.0],  # part 2, at eta*lam = 1.8, cond = 100
           etalam_scaling=1.8, cond_scaling=100.0,
           eta_hb_lam=0.6, cond_hb=100.0,  # part 3, fixed heavy-ball step
           mono_tol=0.02, ratio_tol=0.02, reduction_tol=0.10, sigma_tol=0.15)


def run_cell(eta, lam, cond, sigma, beta, T, n_seeds):
    """Seed-aggregated metrics for one closed-loop straight-valley cell."""
    land = StraightValley(mu=lam / cond, lam=lam, x_star=CFG["x_star"])
    noise_fn = partial(gaussian_isotropic, sigma=sigma)
    w0 = np.array([-3.0, 1.0])
    pred_std = Cl.stationary_hill_std(eta, lam, sigma, beta)
    hill, total, ratio, esc, hfer, msr = [], [], [], [], [], []
    ndiv = 0
    for sd in range(n_seeds):
        rng = np.random.default_rng(110_000 + 1000 * sd)
        s = simulate(land, beta, eta, w0, T, rng=rng, noise_fn=noise_fn)
        if s["diverged"]:
            ndiv += 1
            esc.append(1.0)
            continue
        d_tail = s["d"][T // 2:]
        W = s["W"][1:]
        hill.append(float(np.mean(0.5 * lam * d_tail**2)))
        total.append(float(np.mean(land.loss(W[T // 2:]))))
        if np.isfinite(pred_std) and pred_std > 0:
            ratio.append(float(np.sqrt(np.mean(d_tail**2))) / pred_std)
        esc.append(float(Me.escape_fraction(s["d"], CFG["tube_radius"]) > 0))
        g_hill_tail = s["G"][T // 2:, 1]
        m_hill_tail = s["M"][T // 2:, 1]
        hfer.append(Me.high_freq_energy_ratio(g_hill_tail))
        msr.append(Me.momentum_suppression_ratio(g_hill_tail, m_hill_tail))

    def mstat(x):
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return np.nan, np.nan
        if len(x) == 1:
            return float(x[0]), np.nan
        return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))

    hill_m, hill_se = mstat(hill)
    ratio_m, ratio_se = mstat(ratio)
    return dict(hill=hill_m, hill_se=hill_se,
                total=mstat(total)[0], ratio=ratio_m, ratio_se=ratio_se,
                esc=float(np.mean(esc)) if esc else 1.0, ndiv=ndiv,
                hfer=mstat(hfer)[0], msr=mstat(msr)[0],
                hill_seeds=np.asarray(hill, dtype=float))


def near_monotone_decreasing(means, ses, tol):
    """No consecutive increase beyond 2 x SE(diff) + tol (relative); ignores NaN cells."""
    ok = True
    prev_i = None
    for i in range(len(means)):
        if not np.isfinite(means[i]):
            continue
        if prev_i is not None:
            se_diff = np.hypot(ses[prev_i], ses[i]) if np.isfinite(ses[i]) else 0.0
            if means[i] > means[prev_i] + 2 * se_diff + tol * means[prev_i]:
                ok = False
        prev_i = i
    return ok


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(np.sum(rx * ry) / np.sqrt(np.sum(rx**2) * np.sum(ry**2)))


def main() -> str:
    betas, etalams, conds = CFG["betas"], CFG["etalams"], CFG["conds"]
    lam, T, n_seeds = CFG["lam"], CFG["T"], CFG["n_seeds"]
    nB, nE, nC = len(betas), len(etalams), len(conds)

    # ---------------- part 1: main grid (sigma = 2) -------------------------
    print(f"\nE9 part 1: straight noisy valley, betas x eta*lam x cond "
          f"(sigma={CFG['sigma']}, seeds={n_seeds}, T={T})")
    shape = (nC, nE, nB)
    hill, hill_se, total, ratio, ratio_se, esc, ndiv, hfer, msr = \
        (np.full(shape, np.nan) for _ in range(9))
    hill_seeds = np.full(shape + (n_seeds,), np.nan)
    for ci, cond in enumerate(conds):
        for ei, el in enumerate(etalams):
            for bi, b in enumerate(betas):
                c = run_cell(el / lam, lam, cond, CFG["sigma"], b, T, n_seeds)
                hill[ci, ei, bi], hill_se[ci, ei, bi] = c["hill"], c["hill_se"]
                total[ci, ei, bi] = c["total"]
                ratio[ci, ei, bi], ratio_se[ci, ei, bi] = c["ratio"], c["ratio_se"]
                esc[ci, ei, bi], ndiv[ci, ei, bi] = c["esc"], c["ndiv"]
                hfer[ci, ei, bi], msr[ci, ei, bi] = c["hfer"], c["msr"]
                hill_seeds[ci, ei, bi, :len(c["hill_seeds"])] = c["hill_seeds"]

    pred_hill = np.array([[Cl.stationary_hill_loss(el / lam, lam, CFG["sigma"], b)
                           for b in betas] for el in etalams])  # (nE, nB); cond-free
    teffs = np.array([effective_window(b) for b in betas])

    # gate 1: stability boundary (CL-1)
    unstable = np.array([[el >= Cl.stability_threshold(b) for b in betas]
                         for el in etalams])
    comfortable = np.array([[el <= 0.9 * Cl.stability_threshold(b) for b in betas]
                            for el in etalams])
    g1 = True
    for ci in range(nC):
        g1 &= bool(np.all(ndiv[ci][unstable] == n_seeds))
        g1 &= bool(np.all(ndiv[ci][comfortable] == 0))
    print(f"  gate 1 (CL-1 boundary exact on all {nC} conds): {'PASS' if g1 else 'FAIL'}")

    # gate 2: CL-2 exactness
    cells = bad = 0
    for ci in range(nC):
        for ei in range(nE):
            for bi in range(nB):
                if comfortable[ei, bi]:
                    cells += 1
                    r, se = ratio[ci, ei, bi], ratio_se[ci, ei, bi]
                    if not np.isfinite(r) or abs(r - 1) > 2 * se + CFG["ratio_tol"]:
                        bad += 1
    g2 = bad <= 0.1 * cells
    print(f"  gate 2 (rms/guide within 2SE+{CFG['ratio_tol']}): "
          f"{cells - bad}/{cells} cells -> {'PASS' if g2 else 'FAIL'}")

    # gates 3+4: regime-scoped improvement, against the CL-3 reduction eta*lam/2
    reduction = np.full((nC, nE), np.nan)
    g3 = g4 = True
    for ci in range(nC):
        for ei, el in enumerate(etalams):
            row, row_se = hill[ci, ei], hill_se[ci, ei]
            stable = np.isfinite(row)
            if el >= 1.6:
                g3 &= near_monotone_decreasing(row, row_se, CFG["mono_tol"])
            if stable[0]:
                reduction[ci, ei] = (row[0] - np.nanmin(row)) / row[0]
        # stabilized regime: beta = 0 diverges, beta = 0.9 runs
        for ei, el in enumerate(etalams):
            if el >= 2.0:
                g3 &= ndiv[ci, ei, 0] == n_seeds and ndiv[ci, ei, betas.index(0.9)] == 0
    for ci in range(nC):
        for ei, el in enumerate(etalams):
            pred_red = Cl.cl3_relative_reduction(el)
            if el in (1.6, 1.8):
                g3 &= abs(reduction[ci, ei] - pred_red) <= CFG["reduction_tol"]
            if el in (0.6, 1.2):
                g4 &= abs(reduction[ci, ei] - pred_red) <= CFG["reduction_tol"]
    ci_ref = conds.index(100.0) if 100.0 in conds else 0
    print(f"  measured max hill-loss reduction vs CL-3 (cond={conds[ci_ref]:g}): "
          + " ".join(f"{el}:{reduction[ci_ref, ei]:.2f}/{Cl.cl3_relative_reduction(el):.2f}"
                     for ei, el in enumerate(etalams)))
    print(f"  gate 3 (near-monotone + reduction match at eta*lam>=1.6 + stabilized regime):"
          f" {'PASS' if g3 else 'FAIL'}")
    print(f"  gate 4 (modest-but-nonzero reduction ~ eta*lam/2 at eta*lam in {{0.6,1.2}}): "
          f"{'PASS' if g4 else 'FAIL'}")

    # ---------------- part 2: sigma scaling ---------------------------------
    el_s, cond_s = CFG["etalam_scaling"], CFG["cond_scaling"]
    print(f"\nE9 part 2: sigma scaling at eta*lam={el_s}, cond={cond_s}")
    sig_hill = {CFG["sigma"]: hill[conds.index(cond_s), etalams.index(el_s)]}
    for sg in CFG["sigmas_scaling"]:
        sig_hill[sg] = np.array([run_cell(el_s / lam, lam, cond_s, sg, b, T, n_seeds)["hill"]
                                 for b in betas])
    g5 = True
    sigs_all = sorted(CFG["sigmas_scaling"] + [CFG["sigma"]])
    sig_pairs = [(hi, lo) for i, lo in enumerate(sigs_all)
                 for hi in sigs_all[i + 1:]]  # every pair, not only adjacent ones
    for s_hi, s_lo in sig_pairs:
        r = sig_hill[s_hi] / sig_hill[s_lo]
        pred = (s_hi / s_lo) ** 2
        ok = np.all(np.abs(r / pred - 1) <= CFG["sigma_tol"])
        g5 &= bool(ok)
        print(f"  sigma {s_hi}/{s_lo}: loss ratio {np.nanmean(r):.2f} vs {pred} "
              f"({'ok' if ok else 'BAD'})")
    print(f"  gate 5 (hill loss ~ sigma^2 per beta, within {CFG['sigma_tol']:.0%}): "
          f"{'PASS' if g5 else 'FAIL'}")

    # ---------------- part 3: normalization ablation ------------------------
    ehb = CFG["eta_hb_lam"] / lam
    print(f"\nE9 part 3: heavy-ball normalization at fixed eta_HB*lam={CFG['eta_hb_lam']} "
          f"(EMA counterpart row: eta*lam=0.6, cond={CFG['cond_hb']})")
    hb_hill, hb_se, hb_ratio, hb_ratio_se = (np.full(nB, np.nan) for _ in range(4))
    for bi, b in enumerate(betas):
        eta_eff = ehb / (1.0 - b)
        c = run_cell(eta_eff, lam, CFG["cond_hb"], CFG["sigma"], b, T, n_seeds)
        hb_hill[bi], hb_se[bi] = c["hill"], c["hill_se"]
        hb_ratio[bi], hb_ratio_se[bi] = c["ratio"], c["ratio_se"]
    hb_pred = np.array([Cl.stationary_hill_loss(ehb / (1 - b), lam, CFG["sigma"], b)
                        for b in betas])
    stable_hb = np.isfinite(hb_hill)
    rho_hb = spearman(hb_hill[stable_hb], np.array(betas)[stable_hb])
    hb_cells = int(np.sum(stable_hb))
    hb_bad = int(np.sum(np.abs(hb_ratio[stable_hb] - 1)
                        > 2 * hb_ratio_se[stable_hb] + CFG["ratio_tol"]))
    ema_row = hill[conds.index(CFG["cond_hb"]), etalams.index(0.6)]
    ema_row_se = hill_se[conds.index(CFG["cond_hb"]), etalams.index(0.6)]
    g6 = (rho_hb >= 0.9 and hb_bad <= max(1, int(0.1 * hb_cells))
          and near_monotone_decreasing(ema_row, ema_row_se, CFG["mono_tol"]))
    print("  HB hill loss: " + " ".join(f"{v:.3g}" for v in hb_hill)
          + f"  (Spearman vs beta {rho_hb:+.2f}; guide match {hb_cells - hb_bad}/{hb_cells})")
    print("  EMA hill loss (eta*lam=0.6): " + " ".join(f"{v:.3g}" for v in ema_row))
    print(f"  gate 6 (HB increases, EMA decreases, both on-guide): "
          f"{'PASS' if g6 else 'FAIL'}")

    gate = g1 and g2 and g3 and g4 and g5 and g6
    print(f"\n  E9 decision gate: {'PASS' if gate else 'FAIL'}")

    arrays = dict(betas=np.array(betas), etalams=np.array(etalams),
                  conds=np.array(conds), teffs=teffs,
                  hill=hill, hill_se=hill_se, total=total, ratio=ratio,
                  ratio_se=ratio_se, esc=esc, ndiv=ndiv, hfer=hfer, msr=msr,
                  hill_seeds=hill_seeds, pred_hill=pred_hill, reduction=reduction,
                  hb_hill=hb_hill, hb_se=hb_se, hb_pred=hb_pred,
                  ema_row=ema_row, ema_row_se=ema_row_se,
                  ema_row_pred=pred_hill[etalams.index(0.6)])
    for sg in CFG["sigmas_scaling"]:
        arrays[f"sig_hill_{str(sg).replace('.', 'p')}"] = sig_hill[sg]
    metrics = dict(gate_pass=gate, gate1_boundary=g1, gate2_ratio=g2,
                   gate3_improvement=g3, gate4_regime_scoped=g4, gate5_sigma=g5,
                   gate6_normalization=g6, grid_cells=cells, grid_cells_bad=bad,
                   rho_hb=rho_hb,
                   reduction_ref={str(el): float(reduction[ci_ref, ei])
                                  for ei, el in enumerate(etalams)},
                   reduction_ref_cond=float(conds[ci_ref]),
                   t_over_teff_min=float(T / 2 / teffs.max()))
    run_id = log_run(task="beta_opt", probe="sweep",
                     key=f"b{nB}_el{nE}_c{nC}_seeds{n_seeds}",
                     config=CFG, metrics=metrics, arrays=arrays, sha=SHA)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
