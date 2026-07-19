"""Experiment E10 — the beta window vs bend frequency: too-large beta hurts on curved rivers
(idea_v2.md section 2).

E2/E8 show the lag reversal at one bend frequency; the closed-loop grid shows the window
widening with conditioning. E10 makes the *curvature* axis systematic: on curved valleys
y = a sin(kx) with the amplitude a = 2 held fixed (the idea_v2 design; k = 0.9 is the E2/E8
landscape), raising the bend wavenumber k raises the river's curvature a*k^2 and slope range
a*k — the "harder river" axis.

What the pilots taught (two designs rejected, one prediction corrected):
  - At FIXED max slope a*k, the amplitude shrinks with k and corner-cutting becomes free
    (the straight path through small wiggles never leaves the tube; the lag saturates), so
    the window *widens* with k — that variant does not test the claim.
  - At fixed amplitude, idea_v2's expected "optimal beta decreases with curvature" is NOT
    what happens, for a closed-loop reason worth reporting: the river SPEED v collapses with
    k (measured; the iterate crawls through sharp bends), so the temporal bend frequency
    k*v self-regulates below the filter's passband edge and the window's top stays soft.
    What is exogenous and monotone instead: the CONFINEMENT FLOOR — the smallest beta that
    keeps the trajectory in the tube — rises with k (the local hill sharpness
    lam*(1+f'^2) grows as a*k does, shifting the stability/variance demand), and the window
    NARROWS from below while the traversal slows. The U-shape in travel loss at each k is
    unaffected: too-small beta escapes, too-large beta lags/stalls.

One logged run (task `bend_window`):
  sweep  beta x k x eta*lam, 16 seeds, sigma = 2, cond lam/mu = 100, scored over the
      traveling horizon K = 3/(eta*mu) (the closed-loop-grid convention: the phase in which
      the river is actively traversed). Per cell: tube-escape frequency, mean river-following
      lag 1 - Align_R(m), mean travel loss, river progress, buffer hill-normal energy/step,
      and the mean river speed v (per-step x displacement of confined seeds).
  window  good-beta window per (eta*lam, k): the contiguous beta range around the lag
      minimum where escape <= 0.25 AND lag <= 1.25 x the row minimum — the closed-loop-grid
      criterion plus contiguity (stray off-window cells from seed noise do not extend an
      edge). Two E10-specific choices, both declared: the tube radius is 2.5 = 1.25 x the
      floor amplitude (at R = a exactly, a corner-cutting trajectory *touches* R and noise
      decides the escape flag; with the margin, escape flags dynamical blow-up and the
      corner-cutting cost is carried by the lag and loss metrics), and contiguity replaces
      the raw set. Note the local hill sharpness lam*(1+f'^2) grows with k (part of the
      harder-river axis): it squeezes the window from the small-beta side, while the lag
      squeezes it from the large-beta side.
  guide  the bend enters the filter as a passband tone at omega_bend = k * v. The predicted
      upper edge is the beta at which T1's low-band distortion at omega_bend reaches a
      threshold eps*: beta_edge(k) = lag_edge_beta(k * v, eps*), with eps* calibrated ONCE on
      the anchor cell (eta*lam = 1.8, k = 0.9) and the other nine (eta*lam, k) cells
      predicted (core.closedloop.lag_edge_beta; a calibrated guide, not a theorem).

Decision gates (predeclared):
  1  the confinement floor (smallest beta with escape <= 0.25) is nondecreasing in k for
     both eta*lam and rises by >= 2 grid steps from k = 0.3 to k = 1.8;
  2  the U-shape is real where the bend is fast (eta*lam = 1.8, k >= 0.9): the best-beta by
     travel loss is interior (neither 0 nor 0.999), beta = 0.999 costs >= 1.5x the best, and
     beta = 0 escapes the tube (insufficient filtering);
  3  the window narrows: width (grid steps between the floor and the lag-criterion top) at
     k = 1.8 is at least 2 steps smaller than at k = 0.3, for both eta*lam.
Reported, not gated: the river-speed collapse v(k); the lag-criterion top edge per cell; the
calibrated lag_edge_beta guide (eps* anchored at eta*lam = 1.8, k = 0.9) against the top
edges — with v endogenous this is a consistency overlay, not a prediction.

Run:  cd codebases && python scripts/run_e10_bendwindow.py
"""
from __future__ import annotations

import pathlib
import sys
from functools import partial

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import closedloop as Cl  # noqa: E402
from core import metrics as Me  # noqa: E402
from core.landscapes import CurvedValley, gaussian_isotropic  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
import rivervalley_sim  # noqa: E402
from rivervalley_sim import simulate  # noqa: E402

SHA = code_sha([__file__, rivervalley_sim.__file__])

CFG = dict(lam=10.0, mu=0.1, x_star=5.0, sigma=2.0, n_seeds=16,
           tube_radius=2.5,  # 1.25 x the floor amplitude; see docstring
           a=2.0,
           ks=[0.3, 0.6, 0.9, 1.2, 1.8],
           etalams=[1.8, 2.5],
           betas=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999],
           lag_rel_tol=1.25, esc_tol=0.25,
           anchor=dict(etalam=1.8, k=0.9))


def main() -> str:
    betas, ks, etalams = CFG["betas"], CFG["ks"], CFG["etalams"]
    lam, mu, sig = CFG["lam"], CFG["mu"], CFG["sigma"]
    nE, nK, nB = len(etalams), len(ks), len(betas)
    noise_fn = partial(gaussian_isotropic, sigma=sig)

    shape = (nE, nK, nB)
    esc, lag_m, loss_m, progress, hillE, vbar = (np.full(shape, np.nan) for _ in range(6))
    horizons = np.zeros(nE, dtype=int)
    print(f"\nE10: curved-valley beta window vs bend frequency "
          f"(amplitude a={CFG['a']}, cond={lam / mu:.0f}, seeds={CFG['n_seeds']})")
    for ei, el in enumerate(etalams):
        eta = el / lam
        K = int(np.ceil(3.0 / (eta * mu)))
        horizons[ei] = K
        for ki, k in enumerate(ks):
            land = CurvedValley(mu=mu, lam=lam, x_star=CFG["x_star"],
                                a=CFG["a"], k=k)
            w0 = np.array([-3.0, land.f(-3.0) + 1.0])
            for bi, b in enumerate(betas):
                ef, lg, lt, pr, he, vb = [], [], [], [], [], []
                for sd in range(CFG["n_seeds"]):
                    rng = np.random.default_rng(130_000 + 1000 * sd)
                    s = simulate(land, b, eta, w0, K, rng=rng, noise_fn=noise_fn)
                    if s["diverged"]:
                        ef.append(True)
                        lg.append(1.0)
                        lt.append(np.inf)
                        pr.append(-np.inf)
                        continue
                    W = s["W"][1:]
                    off = land.hill_coordinate(W)
                    ef.append(Me.escape_fraction(off, CFG["tube_radius"], tail_frac=1.0) > 0)
                    lg.append(float(np.mean(Me.lag(s["M"], s["R"]))))
                    lt.append(float(np.mean(land.loss(W))))
                    pr.append((W[-1, 0] - w0[0]) / (CFG["x_star"] - w0[0]))
                    he.append(Me.hill_energy(s["M"], s["N"]) / K)
                    if not ef[-1]:
                        vb.append(float(np.mean(np.diff(s["W"][:, 0]))))
                esc[ei, ki, bi] = np.mean(ef)
                lag_m[ei, ki, bi] = np.mean(lg)
                loss_m[ei, ki, bi] = np.mean(lt)
                progress[ei, ki, bi] = np.mean(pr)
                hillE[ei, ki, bi] = np.mean(he) if he else np.inf
                vbar[ei, ki, bi] = np.mean(vb) if vb else np.nan

    # windows: confinement floor (escape), lag-criterion top (contiguous), best beta
    in_window = np.zeros(shape, dtype=bool)
    floor_edge = np.full((nE, nK), np.nan)
    top_edge = np.full((nE, nK), np.nan)
    width = np.full((nE, nK), np.nan)
    best_beta = np.full((nE, nK), np.nan)
    for ei in range(nE):
        for ki in range(nK):
            ok_esc = esc[ei, ki] <= CFG["esc_tol"]
            if ok_esc.any():
                floor_edge[ei, ki] = betas[int(np.argmax(ok_esc))]
            row_lag = lag_m[ei, ki]
            win = ok_esc & (row_lag <= CFG["lag_rel_tol"] * np.nanmin(row_lag))
            # contiguous window around the lag minimum
            i0 = int(np.nanargmin(row_lag))
            if win[i0]:
                lo = i0
                while lo > 0 and win[lo - 1]:
                    lo -= 1
                hi = i0
                while hi < nB - 1 and win[hi + 1]:
                    hi += 1
                win = np.zeros(nB, dtype=bool)
                win[lo:hi + 1] = True
            else:
                win = np.zeros(nB, dtype=bool)
            in_window[ei, ki] = win
            idx = np.where(win)[0]
            if len(idx):
                top_edge[ei, ki] = betas[idx.max()]
                width[ei, ki] = idx.max() - betas.index(floor_edge[ei, ki])
            finite = np.isfinite(loss_m[ei, ki])
            if finite.any():
                best_beta[ei, ki] = betas[int(np.nanargmin(
                    np.where(finite, loss_m[ei, ki], np.nan)))]

    for ei, el in enumerate(etalams):
        print(f"  eta*lam={el} (horizon K={horizons[ei]}):")
        print("    k \\ beta   " + " ".join(f"{b:>7}" for b in betas))
        for ki, k in enumerate(ks):
            marks = ["   in  " if w else "   .   " for w in in_window[ei, ki]]
            print(f"    {k:>4}       " + " ".join(f"{m:>7}" for m in marks)
                  + f"   floor={floor_edge[ei, ki]}, top={top_edge[ei, ki]}, "
                  f"best={best_beta[ei, ki]}, v={np.nanmean(np.where(in_window[ei, ki], vbar[ei, ki], np.nan)):.3f}")

    # gate 1: confinement floor nondecreasing in k, rises >= 2 grid steps end to end
    g1 = True
    for ei in range(nE):
        f_row = floor_edge[ei]
        g1 &= bool(np.all(np.diff(f_row) >= -1e-12))
        g1 &= betas.index(f_row[-1]) - betas.index(f_row[0]) >= 2
    print(f"  gate 1 (confinement floor rises with bend frequency): "
          f"{'PASS' if g1 else 'FAIL'}")

    # gate 2: U-shape at eta*lam = 1.8, k >= 0.9
    ei18 = etalams.index(1.8)
    g2 = True
    for ki, k in enumerate(ks):
        if k < 0.9:
            continue
        bb = best_beta[ei18, ki]
        row = loss_m[ei18, ki]
        g2 &= bool(np.isfinite(bb)) and bb not in (betas[0], betas[-1])
        g2 &= bool(np.any(np.isfinite(row)) and np.isfinite(row[-1])
                   and row[-1] >= 1.5 * np.nanmin(row))
        g2 &= bool(esc[ei18, ki, 0] >= 0.75)
    print(f"  gate 2 (interior best beta, 0.999 costs >=1.5x, beta=0 escapes): "
          f"{'PASS' if g2 else 'FAIL'}")

    # gate 3: the window narrows end to end
    g3 = True
    for ei in range(nE):
        g3 &= bool(np.isfinite(width[ei, 0]) and np.isfinite(width[ei, -1])
                   and width[ei, 0] - width[ei, -1] >= 2)
    print(f"  window widths (grid steps): "
          + "; ".join(f"eta*lam={el}: {[f'{w:.0f}' for w in width[ei]]}"
                      for ei, el in enumerate(etalams)))
    print(f"  gate 3 (window narrows by >= 2 steps from k=0.3 to k=1.8): "
          f"{'PASS' if g3 else 'FAIL'}")

    # reported (not gated): river-speed collapse and the calibrated lag guide
    aei = etalams.index(CFG["anchor"]["etalam"])
    aki = ks.index(CFG["anchor"]["k"])
    v_row = np.array([[np.nanmean(np.where(in_window[ei, ki], vbar[ei, ki], np.nan))
                       for ki in range(nK)] for ei in range(nE)])
    omega_anchor = CFG["anchor"]["k"] * v_row[aei, aki]
    eps_star = float(Cl.low_band_distortion(top_edge[aei, aki], omega_anchor))
    pred_edge = np.array([[Cl.lag_edge_beta(ks[ki] * v_row[ei, ki], eps_star)
                           for ki in range(nK)] for ei in range(nE)])
    print(f"  river speed v(k): "
          + "; ".join(f"eta*lam={el}: " + " ".join(f"{v:.3f}" for v in v_row[ei])
                      for ei, el in enumerate(etalams)))
    print(f"  lag guide (report only; eps*={eps_star:.3f} at anchor): predicted/observed "
          "top edges:")
    for ei, el in enumerate(etalams):
        print(f"    eta*lam={el}: " + " ".join(
            f"k={k}: {pred_edge[ei, ki]:.3f}/{top_edge[ei, ki]}"
            for ki, k in enumerate(ks)))

    gate = g1 and g2 and g3
    print(f"\n  E10 decision gate: {'PASS' if gate else 'FAIL'}")

    arrays = dict(betas=np.array(betas), ks=np.array(ks), etalams=np.array(etalams),
                  esc=esc, lag=lag_m, loss=loss_m, progress=progress, hillE=hillE,
                  vbar=vbar, in_window=in_window.astype(float), floor_edge=floor_edge,
                  top_edge=top_edge, width=width, v_row=v_row,
                  best_beta=best_beta, pred_edge=pred_edge, horizons=horizons)
    metrics = dict(gate_pass=gate, gate1_floor=g1, gate2_ushape=g2, gate3_narrows=g3,
                   eps_star=eps_star,
                   floor_edge={f"el{el}_k{k}": float(floor_edge[ei, ki])
                               for ei, el in enumerate(etalams)
                               for ki, k in enumerate(ks)},
                   top_edge={f"el{el}_k{k}": float(top_edge[ei, ki])
                             for ei, el in enumerate(etalams)
                             for ki, k in enumerate(ks)},
                   best_beta={f"el{el}_k{k}": float(best_beta[ei, ki])
                              for ei, el in enumerate(etalams)
                              for ki, k in enumerate(ks)},
                   v_row={f"el{el}_k{k}": float(v_row[ei, ki])
                          for ei, el in enumerate(etalams)
                          for ki, k in enumerate(ks)})
    run_id = log_run(task="bend_window", probe="sweep",
                     key=f"k{nK}_el{nE}_b{nB}_seeds{CFG['n_seeds']}",
                     config=CFG, metrics=metrics, arrays=arrays, sha=SHA)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
