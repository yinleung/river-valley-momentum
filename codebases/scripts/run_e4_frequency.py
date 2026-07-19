"""Experiment E4 — frequency-domain validation on toy landscapes (idea_v1.md, Phase 2).

Directly checks the EMA magnitude-response prediction. For fixed gradient streams g_1..g_T we form
m = EMA_beta(g) open-loop and compare the empirical ratio R(omega) = |m_hat(omega)|/|g_hat(omega)|
with the theory |H_beta(omega)| = (1-beta)/sqrt(1 - 2 beta cos omega + beta^2).

Three fixed input streams:
  synthetic   slow tone + fast (near-pi) tone + white noise, stationary over the window;
  straight    the hill-gradient stream g_{t,y} of a beta=0 straight-valley trajectory;
  curved      the hill-gradient stream of a beta=0 curved-valley trajectory.

Success (idea_v1.md): low-frequency R ~ 1, high-frequency R ~ (1-beta)/(1+beta), attenuation
increases with beta, for both straight and curved landscapes.

Run:  cd codebases && python scripts/run_e4_frequency.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import metrics as Me  # noqa: E402
from core import momentum as Mo  # noqa: E402
from core.landscapes import CurvedValley, StraightValley  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
import rivervalley_sim  # noqa: E402
from rivervalley_sim import simulate  # noqa: E402

SHA = code_sha([__file__, rivervalley_sim.__file__])  # tie runs to driver + shared simulator

BETAS = [0.5, 0.9, 0.95, 0.99]
CFG = dict(
    T=512, omega_slow=0.05, omega_fast=0.85, amp_slow=1.0, amp_fast=0.6,
    sigma=0.3, seed=0, window="hann", eta=0.18,
    straight=dict(mu=0.1, lam=10.0, x_star=5.0, x0=-3.0, y0=1.0),
    curved=dict(mu=0.1, lam=10.0, x_star=6.0, a=2.0, k=0.9, x0=-4.0, y0=1.0),
)


def synthetic_stream() -> np.ndarray:
    rng = np.random.default_rng(CFG["seed"])
    t = np.arange(CFG["T"])
    return (CFG["amp_slow"] * np.cos(CFG["omega_slow"] * np.pi * t)
            + CFG["amp_fast"] * np.cos(CFG["omega_fast"] * np.pi * t)
            + CFG["sigma"] * rng.standard_normal(CFG["T"]))


def recorded_hill_stream(curved: bool) -> np.ndarray:
    """Hill-gradient stream g_{t,y} of a deterministic beta=0 trajectory (a fixed input).

    All landscape and trajectory parameters are read from CFG (and therefore logged).
    """
    p = CFG["curved"] if curved else CFG["straight"]
    land = (CurvedValley(mu=p["mu"], lam=p["lam"], x_star=p["x_star"], a=p["a"], k=p["k"])
            if curved else StraightValley(mu=p["mu"], lam=p["lam"], x_star=p["x_star"]))
    s = simulate(land, beta=0.0, eta=CFG["eta"], w0=np.array([p["x0"], p["y0"]]), T=CFG["T"])
    return s["Gexact"][:, 1]


def agreement(g: np.ndarray, beta: float):
    """Empirical ratio R(omega), theory |H|, and weighted relative error on significant bins."""
    m = Mo.ema_momentum(g, beta)
    omega, R = Me.empirical_transfer(g, m, window=CFG["window"])
    H = Mo.transfer_magnitude(beta, omega)
    _, G = Me.windowed_dft(g, CFG["window"])
    power = np.abs(G) ** 2
    sig = power > 0.01 * power.max()  # bins with enough input energy for a meaningful ratio
    rel_err = float(np.median(np.abs(R[sig] - H[sig]) / H[sig]))
    return omega, R, H, rel_err


def main() -> str:
    streams = {
        "synthetic": synthetic_stream(),
        "straight": recorded_hill_stream(curved=False),
        "curved": recorded_hill_stream(curved=True),
    }
    omega_ref = Me.windowed_dft(streams["synthetic"], CFG["window"])[0]
    low, high = Me.band_masks(omega_ref)

    arrays = {"omega": omega_ref}
    metrics = {}
    print(f"\nE4 frequency validation  (T={CFG['T']}, window={CFG['window']})")
    all_ok = True
    for name, g in streams.items():
        print(f"\n  [{name}]  {'beta':>5} {'R_low(med)':>11} {'R_high(med)':>12} "
              f"{'(1-b)/(1+b)':>12} {'rel_err':>9}")
        for b in BETAS:
            omega, R, H, rel_err = agreement(g, b)
            hi_theory = (1 - b) / (1 + b)
            r_low, r_high = float(np.median(R[low])), float(np.median(R[high]))
            ok = rel_err < 0.15
            all_ok = all_ok and ok
            print(f"        {b:>5} {r_low:>11.3f} {r_high:>12.3f} {hi_theory:>12.3f} "
                  f"{rel_err:>9.3f} {'' if ok else '  <-- rel_err>0.15'}")
            arrays[f"{name}_R_b{str(b).replace('.', 'p')}"] = R
            arrays[f"{name}_H_b{str(b).replace('.', 'p')}"] = H
            metrics[f"{name}_relerr_b{b}"] = rel_err
            metrics[f"{name}_Rhigh_b{b}"] = r_high
        arrays[f"{name}_stream"] = g

    metrics["all_relerr_below_0p15"] = all_ok
    print(f"\n  all streams/betas track |H_beta(omega)| within 15% (significant bins): {all_ok}")

    run_id = log_run(task="frequency_validation", probe="spectral",
                     key=f"T{CFG['T']}_{CFG['window']}",
                     config={**CFG, "betas": BETAS, "streams": list(streams)},
                     metrics=metrics, arrays=arrays, sha=SHA)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
