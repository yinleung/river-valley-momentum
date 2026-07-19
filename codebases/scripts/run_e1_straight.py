"""Experiment E1 — straight river-valley filtering demo (idea_v1.md, Task 1.1).

Deterministic landscape L(x,y) = (mu/2)(x-x*)^2 + (lam/2) y^2 with eta in the oscillatory regime
1 < eta*lam < 2, so the hill coordinate alternates sign and the hill gradient is high-frequency.
For beta in {0, 0.5, 0.9, 0.95, 0.99} we check that momentum suppresses the hill component while
preserving river progress.

Decision gate (idea_v1.md): continue only if hill oscillation is visible for beta = 0.

Run:  cd codebases && python scripts/run_e1_straight.py
Logs one sweep run to results/cache/ and prints the decision-gate report.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import metrics as Me  # noqa: E402
from core.landscapes import StraightValley  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
import rivervalley_sim  # noqa: E402
from rivervalley_sim import simulate  # noqa: E402

SHA = code_sha([__file__, rivervalley_sim.__file__])  # tie runs to driver + shared simulator

BETAS = [0.0, 0.5, 0.9, 0.95, 0.99]
CFG = dict(mu=0.1, lam=10.0, x_star=5.0, x0=-3.0, y0=1.0, eta=0.18, T=200)


def main() -> str:
    land = StraightValley(mu=CFG["mu"], lam=CFG["lam"], x_star=CFG["x_star"])
    w0 = np.array([CFG["x0"], CFG["y0"]])
    assert 1.0 < CFG["eta"] * CFG["lam"] < 2.0, "eta*lam must be in (1, 2) for hill oscillation"

    runs = {b: simulate(land, b, CFG["eta"], w0, CFG["T"]) for b in BETAS}

    # --- metrics per beta ---------------------------------------------------
    hsr, align_g, align_m, dist, hfer_g, hfer_m, msr = ({} for _ in range(7))
    r_hat = np.array([1.0, 0.0])
    for b in BETAS:
        s = runs[b]
        g_y, m_y = s["Gexact"][:, 1], s["M"][:, 1]
        hsr[b] = Me.hill_suppression_ratio(m_y, g_y)
        align_g[b] = float(np.mean(Me.river_alignment(s["Gexact"], r_hat)))
        align_m[b] = float(np.mean(Me.river_alignment(s["M"], r_hat)))
        dist[b] = Me.distance_rms(s["d"])
        hfer_g[b] = Me.high_freq_energy_ratio(g_y, window="rect")
        hfer_m[b] = Me.high_freq_energy_ratio(m_y, window="rect")
        msr[b] = Me.momentum_suppression_ratio(g_y, m_y, window="rect")

    # --- decision gate: hill oscillation visible at beta = 0 ----------------
    g_y0 = runs[0.0]["Gexact"][:, 1]
    sign_changes = int(np.sum(np.diff(np.sign(g_y0[np.abs(g_y0) > 1e-6])) != 0))
    gate_pass = sign_changes >= 5

    # --- report -------------------------------------------------------------
    print(f"\nE1 straight valley  (eta*lam={CFG['eta']*CFG['lam']:.2f},  1-eta*lam="
          f"{1-CFG['eta']*CFG['lam']:+.2f})")
    print(f"  decision gate: hill-gradient sign changes at beta=0 = {sign_changes}  "
          f"-> {'PASS' if gate_pass else 'FAIL'} (need >=5)\n")
    hdr = f"  {'beta':>5} {'HSR':>10} {'align(g)':>9} {'align(m)':>9} {'dist_rms':>9} {'MSR':>9} {'|H(pi)|^2':>10}"
    print(hdr)
    for b in BETAS:
        hpi2 = ((1 - b) / (1 + b)) ** 2
        print(f"  {b:>5} {hsr[b]:>10.2e} {align_g[b]:>9.3f} {align_m[b]:>9.3f} "
              f"{dist[b]:>9.3f} {msr[b]:>9.2e} {hpi2:>10.2e}")

    hsr_ordered = all(hsr[BETAS[i]] >= hsr[BETAS[i + 1]] for i in range(len(BETAS) - 1))
    align_improved = all(align_m[b] >= align_g[b] - 1e-9 for b in BETAS)
    print(f"\n  HSR monotone decreasing in beta: {hsr_ordered}")
    print(f"  river alignment improved (m vs g) for all beta: {align_improved}")
    print(f"  best dist_rms at beta={min(BETAS, key=lambda b: dist[b])}")

    # --- log one sweep run --------------------------------------------------
    arrays = {"betas": np.array(BETAS)}
    for k in ("W", "G", "Gexact", "M", "R", "N", "d"):
        arrays[k] = np.stack([runs[b][k] for b in BETAS], axis=0)
    metrics = {
        "sign_changes_b0": sign_changes,
        "gate_pass": gate_pass,
        "hsr_monotone": hsr_ordered,
        "align_improved": align_improved,
        "HSR": {str(b): hsr[b] for b in BETAS},
        "align_g": {str(b): align_g[b] for b in BETAS},
        "align_m": {str(b): align_m[b] for b in BETAS},
        "dist_rms": {str(b): dist[b] for b in BETAS},
        "HFER_g": {str(b): hfer_g[b] for b in BETAS},
        "HFER_m": {str(b): hfer_m[b] for b in BETAS},
        "MSR": {str(b): msr[b] for b in BETAS},
    }
    key = f"eta{str(CFG['eta']).replace('.', 'p')}_lam{int(CFG['lam'])}_T{CFG['T']}"
    run_id = log_run(task="straight_valley", probe="trajectory", key=key,
                     config={**CFG, "betas": BETAS}, metrics=metrics, arrays=arrays, sha=SHA)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
