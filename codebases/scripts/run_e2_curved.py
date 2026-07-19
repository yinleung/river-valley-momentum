"""Experiment E2 — curved river-valley filtering demo (idea_v1.md, Task 1.2).

Landscape L(x,y) = (mu/2)(x-x*)^2 + (lam/2)(y - a sin(k x))^2. The river floor bends, so we test
whether momentum still filters the hill while following the bending river, and whether overly
large beta lags. River tangent r(x) and hill normal n(x) are evaluated at the live iterate.

Decision gate (idea_v1.md): continue if moderate beta improves river alignment but beta = 0.99
shows lag or diminished progress.

Run:  cd codebases && python scripts/run_e2_curved.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import metrics as Me  # noqa: E402
from core.landscapes import CurvedValley  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
import rivervalley_sim  # noqa: E402
from rivervalley_sim import simulate  # noqa: E402

SHA = code_sha([__file__, rivervalley_sim.__file__])  # tie runs to driver + shared simulator

BETAS = [0.0, 0.5, 0.9, 0.95, 0.99]
CFG = dict(mu=0.1, lam=10.0, x_star=6.0, a=2.0, k=0.9, x0=-4.0, y0=1.0, eta=0.18, T=300)


def main() -> str:
    land = CurvedValley(mu=CFG["mu"], lam=CFG["lam"], x_star=CFG["x_star"], a=CFG["a"], k=CFG["k"])
    w0 = np.array([CFG["x0"], CFG["y0"]])
    runs = {b: simulate(land, b, CFG["eta"], w0, CFG["T"]) for b in BETAS}

    align_m, align_g, hill_e, lag_m, progress = ({} for _ in range(5))
    for b in BETAS:
        s = runs[b]
        align_g[b] = float(np.mean(Me.river_alignment(s["Gexact"], s["R"])))
        align_m[b] = float(np.mean(Me.river_alignment(s["M"], s["R"])))
        hill_e[b] = Me.hill_energy(s["M"], s["N"])
        lag_m[b] = float(np.mean(Me.lag(s["M"], s["R"])))
        # river progress toward x*: fraction of the initial gap closed
        progress[b] = float((s["W"][-1, 0] - CFG["x0"]) / (CFG["x_star"] - CFG["x0"]))

    print(f"\nE2 curved valley  f(x)={CFG['a']}*sin({CFG['k']}x),  eta*lam={CFG['eta']*CFG['lam']:.2f}")
    print(f"  {'beta':>5} {'align(g)':>9} {'align(m)':>9} {'hillE(m)':>10} {'lag(m)':>8} {'progress':>9}")
    for b in BETAS:
        print(f"  {b:>5} {align_g[b]:>9.3f} {align_m[b]:>9.3f} {hill_e[b]:>10.2e} "
              f"{lag_m[b]:>8.3f} {progress[b]:>9.3f}")

    best_align_beta = max(BETAS, key=lambda b: align_m[b])
    moderate_best = best_align_beta in (0.9, 0.95)
    lag_increases_top = lag_m[0.99] > lag_m[best_align_beta]
    progress_drops_top = progress[0.99] < max(progress[b] for b in BETAS) - 1e-9
    gate_pass = moderate_best and (lag_increases_top or progress_drops_top)

    print(f"\n  best river alignment at beta={best_align_beta}  (moderate-best: {moderate_best})")
    print(f"  lag(0.99)={lag_m[0.99]:.3f} > lag(best)={lag_m[best_align_beta]:.3f}: {lag_increases_top}")
    print(f"  progress(0.99)={progress[0.99]:.3f} below max progress: {progress_drops_top}")
    print(f"  decision gate (filtering-lag tradeoff): {'PASS' if gate_pass else 'FAIL'}")

    arrays = {"betas": np.array(BETAS)}
    for k in ("W", "G", "Gexact", "M", "R", "N", "d"):
        arrays[k] = np.stack([runs[b][k] for b in BETAS], axis=0)
    metrics = {
        "gate_pass": gate_pass,
        "best_align_beta": best_align_beta,
        "moderate_best": moderate_best,
        "align_g": {str(b): align_g[b] for b in BETAS},
        "align_m": {str(b): align_m[b] for b in BETAS},
        "hill_energy": {str(b): hill_e[b] for b in BETAS},
        "lag": {str(b): lag_m[b] for b in BETAS},
        "progress": {str(b): progress[b] for b in BETAS},
    }
    key = f"a{int(CFG['a'])}_k{str(CFG['k']).replace('.', 'p')}_eta{str(CFG['eta']).replace('.', 'p')}_T{CFG['T']}"
    run_id = log_run(task="curved_valley", probe="trajectory", key=key,
                     config={**CFG, "betas": BETAS}, metrics=metrics, arrays=arrays, sha=SHA)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
