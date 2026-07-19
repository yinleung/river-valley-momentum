"""Experiment E3 — river-valley with stochastic noise (idea_v1.md, Task 1.3).

Adds three gradient-noise models to the straight valley and checks that momentum filters both the
deterministic hill oscillation and the stochastic perturbation, over multiple seeds:

    Gaussian isotropic   xi ~ N(0, sigma^2 I)
    anisotropic hill     hill coordinate noise scaled up
    heavy-tailed         Student-t (df>2, finite variance), rescaled to sigma

Headline metric: the noise-only residual NSR(beta) = stochastic_residual_ratio (m vs the filtered
clean signal EMA(grad L)), which decreases toward the white-noise floor (1-beta)/(1+beta) = 1/N_eff.
We also report NSR_inst = noise_suppression_ratio, the idea_v1.md literal formula (m vs the
instantaneous grad L); it is non-monotone because it also carries the deterministic lag bias (T6).

Decision gate (idea_v1.md): continue if the filtering effect survives noise (NSR -> 1/N_eff and
river alignment of m exceeds that of g, across all three noise models).

Run:  cd codebases && python scripts/run_e3_noise.py
"""
from __future__ import annotations

import pathlib
import sys
from functools import partial

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import metrics as Me  # noqa: E402
from core import landscapes as L  # noqa: E402
from core.landscapes import StraightValley  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
import rivervalley_sim  # noqa: E402
from rivervalley_sim import simulate  # noqa: E402

SHA = code_sha([__file__, rivervalley_sim.__file__])  # tie runs to driver + shared simulator

BETAS = [0.0, 0.5, 0.9, 0.95, 0.99]
CFG = dict(mu=0.1, lam=10.0, x_star=5.0, x0=-3.0, y0=1.0, eta=0.18, T=200,
           sigma=2.0, hill_scale=4.0, df=3.0, n_seeds=12)
NOISE = {
    "gaussian": lambda s: partial(L.gaussian_isotropic, sigma=s["sigma"]),
    "anisotropic": lambda s: partial(L.anisotropic_hill, sigma=s["sigma"], hill_scale=s["hill_scale"]),
    "heavy_tailed": lambda s: partial(L.heavy_tailed, sigma=s["sigma"], df=s["df"]),
}


def main() -> str:
    land = StraightValley(mu=CFG["mu"], lam=CFG["lam"], x_star=CFG["x_star"])
    w0 = np.array([CFG["x0"], CFG["y0"]])
    r_hat = np.array([1.0, 0.0])

    # per (noise, metric) -> array (n_beta, n_seeds). NSR isolates noise (vs filtered clean
    # signal); NSR_inst uses the instantaneous exact gradient and exposes the lag tradeoff.
    results = {nt: {k: np.zeros((len(BETAS), CFG["n_seeds"]))
                    for k in ("NSR", "NSR_inst", "HSR", "align_g", "align_m", "dist_rms")}
               for nt in NOISE}
    sample_streams = {}  # one representative seed per (noise, beta) for figures

    for nt, make in NOISE.items():
        noise_fn = make(CFG)
        for bi, b in enumerate(BETAS):
            for sd in range(CFG["n_seeds"]):
                rng = np.random.default_rng(1000 * bi + sd)  # independent across beta and seed
                s = simulate(land, b, CFG["eta"], w0, CFG["T"], rng=rng, noise_fn=noise_fn)
                R = results[nt]
                R["NSR"][bi, sd] = Me.stochastic_residual_ratio(s["M"], s["G"], s["Gexact"], b)
                R["NSR_inst"][bi, sd] = Me.noise_suppression_ratio(s["M"], s["G"], s["Gexact"])
                R["HSR"][bi, sd] = Me.hill_suppression_ratio(s["M"][:, 1], s["G"][:, 1])
                R["align_g"][bi, sd] = np.mean(Me.river_alignment(s["G"], r_hat))
                R["align_m"][bi, sd] = np.mean(Me.river_alignment(s["M"], r_hat))
                R["dist_rms"][bi, sd] = Me.distance_rms(s["d"])
                if sd == 0:
                    sample_streams[f"{nt}_b{bi}_G"] = s["G"]
                    sample_streams[f"{nt}_b{bi}_M"] = s["M"]
                    sample_streams[f"{nt}_b{bi}_Gexact"] = s["Gexact"]

    # --- report + gate ------------------------------------------------------
    print(f"\nE3 noisy valley  (sigma={CFG['sigma']}, seeds={CFG['n_seeds']}, eta*lam="
          f"{CFG['eta']*CFG['lam']:.1f})   NSR = mean +/- std over seeds")
    gate = True
    for nt in NOISE:
        nsr = results[nt]["NSR"]
        nsr_mean = nsr.mean(1)
        nsr_inst_mean = results[nt]["NSR_inst"].mean(1)
        print(f"\n  [{nt}]  {'beta':>5} {'NSR(noise)':>16} {'1/Neff':>8} {'NSR_inst':>9} "
              f"{'HSR':>10} {'align(g)':>9} {'align(m)':>9}")
        for bi, b in enumerate(BETAS):
            floor = (1 - b) / (1 + b)
            print(f"        {b:>5} {nsr_mean[bi]:>7.3f}+/-{nsr[bi].std():<6.3f} {floor:>8.3f} "
                  f"{nsr_inst_mean[bi]:>9.3f} "
                  f"{results[nt]['HSR'][bi].mean():>10.2e} "
                  f"{results[nt]['align_g'][bi].mean():>9.3f} "
                  f"{results[nt]['align_m'][bi].mean():>9.3f}")
        nsr_decreasing = bool(np.all(np.diff(nsr_mean) <= 1e-6))
        align_improved = bool(np.all(results[nt]["align_m"].mean(1)
                                     >= results[nt]["align_g"].mean(1) - 1e-9))
        inst_argmin = BETAS[int(np.argmin(nsr_inst_mean))]
        print(f"        NSR(noise) decreasing in beta: {nsr_decreasing} | "
              f"align(m)>=align(g): {align_improved} | NSR_inst min at beta={inst_argmin}")
        gate = gate and nsr_decreasing and align_improved

    print(f"\n  decision gate (filtering survives noise): {'PASS' if gate else 'FAIL'}")

    # --- log ----------------------------------------------------------------
    arrays = {"betas": np.array(BETAS)}
    for nt in NOISE:
        for k, v in results[nt].items():
            arrays[f"{nt}_{k}"] = v
    arrays.update(sample_streams)
    metrics = {"gate_pass": gate}
    for nt in NOISE:
        metrics[f"{nt}_NSR_mean"] = {str(b): float(results[nt]["NSR"][bi].mean())
                                     for bi, b in enumerate(BETAS)}
        metrics[f"{nt}_NSR_std"] = {str(b): float(results[nt]["NSR"][bi].std())
                                    for bi, b in enumerate(BETAS)}
        metrics[f"{nt}_NSR_inst_mean"] = {str(b): float(results[nt]["NSR_inst"][bi].mean())
                                          for bi, b in enumerate(BETAS)}
        metrics[f"{nt}_align_g_mean"] = {str(b): float(results[nt]["align_g"][bi].mean())
                                         for bi, b in enumerate(BETAS)}
        metrics[f"{nt}_align_m_mean"] = {str(b): float(results[nt]["align_m"][bi].mean())
                                         for bi, b in enumerate(BETAS)}
    key = (f"sig{str(CFG['sigma']).replace('.', 'p')}_seeds{CFG['n_seeds']}"
           f"_eta{str(CFG['eta']).replace('.', 'p')}")
    run_id = log_run(task="noisy_valley", probe="trajectory", key=key,
                     config={**CFG, "betas": BETAS, "noise_types": list(NOISE)},
                     metrics=metrics, arrays=arrays, sha=SHA)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
