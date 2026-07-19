"""Experiment E13 — forced-disturbance controls (review_v3 "ablations that can falsify the
mechanism"): performance must follow the disturbance's frequency content, not beta.

Straight noisy valley with a deterministic tone A cos(omega t) added to the HILL gradient
(scripts/rivervalley_sim.py `forcing_fn`), swept over four frequencies at fixed amplitude:

    omega = 0.02          passband: FR predicts gain 1/lam at every beta -- momentum
                          CANNOT remove it, the tail hill loss is beta-flat;
    omega = theta_{0.9}   the beta = 0.9 hill-mode frequency: FR predicts RESONANT
                          amplification -- momentum HURTS (the anti-benefit);
    omega = 0.6 pi        high band: attenuated with beta;
    omega = pi            Nyquist: removed at the |G(pi)| ~ (eta/2)/T_eff rate.

Every cell carries the parameter-free guide `forced_hill_loss` (FR + CL-2 superposition)
and the lock-in forced amplitude against A * `forced_gain` -- the same formulas at every
frequency; only the tone moves. Landscape/eta match E1/E9 (lam = 10, mu = 0.1,
eta*lam = 1.8), sigma = 1, A = 3, T = 6000, tail half scored, 12 seeds,
beta in {0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99} (T/2 = 15 T_eff at beta = 0.99: burn-in clear).

Decision gates (predeclared):
  1  guide match: tail hill loss within 2 x seed SE + 5% of `forced_hill_loss` in >= 90%
     of the 28 (omega, beta) cells.
  2  Nyquist removal: lock-in amplitude ratio amp(pi, 0.99)/amp(pi, 0) <= 0.01
     (FR predicts 5e-4; the residual is the lock-in noise floor).
  3  passband neutrality: max_beta/min_beta of amp(0.02, beta) <= 1.1 (FR predicts 1.02
     spread): no beta removes a passband disturbance.
  4  resonance harm: tail hill loss(theta_{0.9}, beta = 0.9) >= 3 x loss(theta_{0.9}, 0)
     (FR predicts 6.1x): momentum amplifies its own mode's frequency.

Run:  cd codebases && python scripts/run_e13_forced.py          (~2 min)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import closedloop as CL  # noqa: E402
from core.landscapes import StraightValley, gaussian_isotropic  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
import rivervalley_sim as sim  # noqa: E402

SHA = code_sha([__file__, sim.__file__])

ETA, LAM, MU, SIGMA, A_F, T, SEEDS = 0.18, 10.0, 0.1, 1.0, 3.0, 6000, 12
BETAS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]


def lockin_amp(d: np.ndarray, omega: float, t0: int) -> float:
    """Forced amplitude of the scored tail of d_t at frequency omega (pi: coincident pair)."""
    tail = d[t0:]
    t_idx = np.arange(t0, t0 + len(tail))
    fac = 1.0 if abs(omega - np.pi) < 1e-12 else 2.0
    return fac * abs(np.mean(tail * np.exp(-1j * omega * t_idx)))


def main() -> str:
    theta09 = CL.resonant_frequency(0.9, ETA, LAM)
    omegas = [0.02, theta09, 0.6 * np.pi, np.pi]
    om_names = ["0.02 (passband)", f"{theta09:.3f} (theta_0.9)", "0.6pi (high)",
                "pi (Nyquist)"]
    land = StraightValley(mu=MU, lam=LAM, x_star=5.0)
    noise = lambda rng, shape: gaussian_isotropic(rng, shape, SIGMA)  # noqa: E731
    t0 = T // 2

    nO, nB = len(omegas), len(BETAS)
    loss = np.full((nO, nB, SEEDS), np.nan)
    amp = np.full((nO, nB, SEEDS), np.nan)
    print(f"E13 forced controls (eta*lam={ETA * LAM}, sigma={SIGMA}, A={A_F}, T={T}, "
          f"seeds={SEEDS}, theta_0.9={theta09:.4f})")
    for oi, om in enumerate(omegas):
        force = lambda t, om=om: np.array([0.0, A_F * np.cos(om * t)])  # noqa: E731
        for bi, b in enumerate(BETAS):
            for sd in range(SEEDS):
                out = sim.simulate(land, b, ETA, np.array([0.0, 0.0]), T,
                                   rng=np.random.default_rng(sd), noise_fn=noise,
                                   forcing_fn=force)
                if out["diverged"]:
                    continue
                loss[oi, bi, sd] = np.mean(0.5 * LAM * out["d"][t0:] ** 2)
                amp[oi, bi, sd] = lockin_amp(out["d"], om, t0)
        row = np.nanmean(loss[oi], axis=-1)
        print(f"  omega={om_names[oi]:>18}: tail hill loss "
              + " ".join(f"{v:7.3f}" for v in row))

    guide = np.array([[CL.forced_hill_loss(ETA, LAM, SIGMA, b, A_F, om)
                       for b in BETAS] for om in omegas])
    amp_guide = np.array([[A_F * float(CL.forced_gain(om, b, ETA, LAM))
                           for b in BETAS] for om in omegas])
    mean_loss = np.nanmean(loss, axis=-1)
    se = np.nanstd(loss, axis=-1) / np.sqrt(SEEDS)
    mean_amp = np.nanmean(amp, axis=-1)

    on_guide = np.abs(mean_loss - guide) <= 2 * se + 0.05 * guide
    g1 = on_guide.mean() >= 0.90
    print(f"\n  gate 1 (loss on FR guide in >= 90% of cells): "
          f"{int(on_guide.sum())}/{on_guide.size} {'PASS' if g1 else 'FAIL'}")
    r_nyq = mean_amp[3, BETAS.index(0.99)] / mean_amp[3, BETAS.index(0.0)]
    g2 = r_nyq <= 0.01
    print(f"  gate 2 (Nyquist amp ratio 0.99/0 <= 0.01): {r_nyq:.2e} "
          f"{'PASS' if g2 else 'FAIL'}")
    r_flat = np.max(mean_amp[0]) / np.min(mean_amp[0])
    g3 = r_flat <= 1.1
    print(f"  gate 3 (passband amp max/min <= 1.1): {r_flat:.3f} "
          f"{'PASS' if g3 else 'FAIL'}")
    r_res = mean_loss[1, BETAS.index(0.9)] / mean_loss[1, BETAS.index(0.0)]
    g4 = r_res >= 3.0
    print(f"  gate 4 (resonance loss ratio 0.9/0 >= 3): {r_res:.2f} "
          f"(guide {guide[1, BETAS.index(0.9)] / guide[1, BETAS.index(0.0)]:.2f}) "
          f"{'PASS' if g4 else 'FAIL'}")
    gate = g1 and g2 and g3 and g4
    print(f"\n  E13 decision gate: {'PASS' if gate else 'FAIL'}")

    metrics = dict(gate_pass=gate, gate_1=g1, gate_2=g2, gate_3=g3, gate_4=g4,
                   on_guide_cells=int(on_guide.sum()), cells=int(on_guide.size),
                   nyquist_amp_ratio=float(r_nyq), passband_amp_spread=float(r_flat),
                   resonance_loss_ratio=float(r_res),
                   resonance_loss_ratio_guide=float(guide[1, BETAS.index(0.9)]
                                                    / guide[1, BETAS.index(0.0)]),
                   theta09=float(theta09),
                   loss_passband={str(b): float(v) for b, v in zip(BETAS, mean_loss[0])},
                   loss_resonance={str(b): float(v) for b, v in zip(BETAS, mean_loss[1])},
                   loss_nyquist={str(b): float(v) for b, v in zip(BETAS, mean_loss[3])})
    arrays = dict(omegas=np.array(omegas), betas=np.array(BETAS), loss=loss, amp=amp,
                  guide=guide, amp_guide=amp_guide)
    run_id = log_run(task="forced_controls", probe="sweep",
                     key=f"om{nO}_b{nB}_seeds{SEEDS}",
                     config=dict(eta=ETA, lam=LAM, mu=MU, sigma=SIGMA, A=A_F, T=T,
                                 tail_from=t0, betas=BETAS,
                                 omegas=[float(o) for o in omegas]),
                     metrics=metrics, arrays=arrays, sha=SHA)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
