"""Experiment E8 — headline regime figure on the curved noisy valley (fable_dgs_v1.md §7).

Shen-et-al.-Figure-2-style trajectory families on the E2 curved landscape f(x) = 2 sin(0.9 x)
with E3 Gaussian gradient noise, over the beta grid {0, 0.5, 0.9, 0.99, 0.999}:

    panel a   eta = 0.18 (eta*lam = 1.8), sigma = 2     noisy large-LR sweep
    panel b   eta = 0.25 (eta*lam = 2.5 > 2), sigma = 2  beyond the beta=0 stability threshold
    panel a0  eta = 0.18, sigma = 0                      deterministic curved control (logged,
                                                         not a headline panel)

The figure's caveat panel (c) is E1's existing straight-clean run — read from the cache by
fig_e8_headline.py, not re-run here.

Guides come from core.closedloop (CL-1 threshold, CL-2 stationary rms). The CL-2 guide uses
the straight-valley lam: linearizing the offset dynamics gives contraction 1 - eta*lam(1+f'^2)
with noise variance eta^2 sigma^2 (1+f'^2), and the (1+f'^2) factors cancel in the stationary
variance up to the (2 - eta*lam_eff/T_eff) denominator — so the plain-lam guide is accurate at
large T_eff (beta = 0.9 cells) while at beta = 0 the local threshold eta*lam(1+f'^2) < 2
already fails wherever |f'| > sqrt(2/(eta*lam) - 1), which is why the iterate escapes the tube.

Decision gates (fable_dgs_v1.md §8.2):
    1. beta = 0 diverges on every seed in panel b;
    2. in the designated in-tube cells (beta = 0.9) the measured tail rms matches the CL-2
       guide within ~10%, with the applicable guide predeclared per panel in CFG
       (`gate_guide`): panel a uses the plain-lam guide (eta*lam/T_eff = 0.09, bend
       correction below the tolerance); panel b uses the bend-corrected guide
       (core.closedloop.stationary_hill_std with fp2 = realized tail E[f'^2]; at
       eta*lam/T_eff = 0.13 the derived correction is ~6% and cannot be neglected);
    3. lag reversal by beta = 0.999 in panel a: worse tail rms, worse tail loss, and less
       river distance than beta = 0.9.

Run:  cd codebases && python scripts/run_e8_headline.py
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

BETAS = [0.0, 0.5, 0.9, 0.99, 0.999]
CFG = dict(mu=0.1, lam=10.0, x_star=5.0, a=2.0, k=0.9, sigma=2.0, T=300, n_seeds=12,
           tube_radius=2.0, gate_rms_tol=0.10, gate_guide={"a": "plain", "b": "bend"})
PANELS = {"a": dict(eta=0.18, sigma=2.0), "b": dict(eta=0.25, sigma=2.0),
          "a0": dict(eta=0.18, sigma=0.0)}


def main() -> str:
    land = CurvedValley(mu=CFG["mu"], lam=CFG["lam"], x_star=CFG["x_star"],
                        a=CFG["a"], k=CFG["k"])
    w0 = np.array([-3.0, land.f(-3.0) + 1.0])
    T = CFG["T"]

    metrics, arrays = {}, {"betas": np.array(BETAS)}
    summary = {}  # (panel, beta) -> aggregates for gates
    for pname, p in PANELS.items():
        n_seeds = CFG["n_seeds"] if p["sigma"] > 0 else 1
        noise_fn = partial(gaussian_isotropic, sigma=p["sigma"]) if p["sigma"] > 0 else None
        for b in BETAS:
            per = dict(x_final=[], rms=[], loss=[], esc=[], div=[], fp2=[])
            loss_curves = np.full((n_seeds, T), np.nan)
            for sd in range(n_seeds):
                rng = np.random.default_rng(10_000 + 100 * sd)  # common across beta/panel
                s = simulate(land, b, p["eta"], w0, T, rng=rng, noise_fn=noise_fn)
                per["div"].append(s["diverged"])
                W = s["W"][1:]  # iterates w_1..w_T, aligned with steps
                off = land.hill_coordinate(W)
                Lc = np.array([land.loss(W[t]) for t in range(T)])
                loss_curves[sd] = Lc
                if sd == 0:
                    arrays[f"{pname}_b{b}_W"] = s["W"]
                    arrays[f"{pname}_b{b}_off"] = off
                if s["diverged"]:
                    per["x_final"].append(np.nan)
                    per["rms"].append(np.nan)
                    per["loss"].append(np.nan)
                    per["esc"].append(1.0)
                    per["fp2"].append(np.nan)
                    continue
                tail = slice(T // 2, None)
                per["x_final"].append(W[-1, 0])
                per["rms"].append(float(np.sqrt(np.mean(off[tail] ** 2))))
                per["loss"].append(float(np.mean(Lc[tail])))
                per["esc"].append(Me.escape_fraction(off, CFG["tube_radius"]))
                per["fp2"].append(float(np.mean(land.fp(W[tail, 0]) ** 2)))
            arrays[f"{pname}_b{b}_losscurves"] = loss_curves
            n_div = int(np.sum(per["div"]))
            fp2 = float(np.nanmean(per["fp2"])) if n_div < n_seeds else float("nan")
            agg = dict(
                n_div=n_div, n_seeds=n_seeds,
                x_final=float(np.nanmean(per["x_final"])) if n_div < n_seeds else float("nan"),
                rms=float(np.nanmean(per["rms"])) if n_div < n_seeds else float("nan"),
                loss=float(np.nanmean(per["loss"])) if n_div < n_seeds else float("nan"),
                esc_freq=float(np.mean([e > 0 for e in per["esc"]])),
                pred_rms=Cl.stationary_hill_std(p["eta"], CFG["lam"], p["sigma"], b)
                if p["sigma"] > 0 else float("nan"),
                pred_rms_bend=Cl.stationary_hill_std(p["eta"], CFG["lam"], p["sigma"], b,
                                                     fp2=fp2)
                if p["sigma"] > 0 and np.isfinite(fp2) else float("nan"),
            )
            summary[(pname, b)] = agg
            for k2, v2 in agg.items():
                metrics[f"{pname}_b{b}_{k2}"] = v2

    # --- report ---------------------------------------------------------------
    for pname, p in PANELS.items():
        etalam = p["eta"] * CFG["lam"]
        print(f"\nE8 panel {pname}  (eta*lam={etalam}, sigma={p['sigma']}, "
              f"seeds={summary[(pname, BETAS[0])]['n_seeds']})")
        print(f"  {'beta':>6} {'div':>6} {'x_final':>8} {'rms(tail)':>10} {'pred':>7} "
              f"{'pred_bd':>7} {'loss(tail)':>10} {'esc_freq':>8}")
        for b in BETAS:
            a = summary[(pname, b)]
            print(f"  {b:>6} {a['n_div']:>4}/{a['n_seeds']:<2} {a['x_final']:>8.3f} "
                  f"{a['rms']:>10.4f} {a['pred_rms']:>7.3f} {a['pred_rms_bend']:>7.3f} "
                  f"{a['loss']:>10.3f} {a['esc_freq']:>8.2f}")

    # --- decision gates ---------------------------------------------------------
    g1 = summary[("b", 0.0)]["n_div"] == summary[("b", 0.0)]["n_seeds"]
    ratios, ratios_bend, gate_ratios = {}, {}, {}
    for pname in ("a", "b"):
        a = summary[(pname, 0.9)]
        ratios[pname] = a["rms"] / a["pred_rms"]
        ratios_bend[pname] = a["rms"] / a["pred_rms_bend"]
        gate_ratios[pname] = ratios[pname] if CFG["gate_guide"][pname] == "plain" \
            else ratios_bend[pname]
    g2 = all(abs(r - 1.0) <= CFG["gate_rms_tol"] for r in gate_ratios.values())
    a9, a999 = summary[("a", 0.9)], summary[("a", 0.999)]
    g3 = (a999["rms"] > a9["rms"] and a999["loss"] > a9["loss"]
          and a999["x_final"] < a9["x_final"])
    print(f"\n  gate 1 (beta=0 diverges {summary[('b',0.0)]['n_seeds']}/"
          f"{summary[('b',0.0)]['n_seeds']} in panel b): {'PASS' if g1 else 'FAIL'}")
    print(f"  gate 2 (in-tube rms vs CL-2 at beta=0.9, predeclared guides "
          f"{CFG['gate_guide']}: gated a={gate_ratios['a']:.3f} b={gate_ratios['b']:.3f}; "
          f"plain {ratios['a']:.3f}/{ratios['b']:.3f}, bend "
          f"{ratios_bend['a']:.3f}/{ratios_bend['b']:.3f}; tol {CFG['gate_rms_tol']}): "
          f"{'PASS' if g2 else 'FAIL'}")
    print(f"  gate 3 (lag reversal at beta=0.999 in panel a): {'PASS' if g3 else 'FAIL'}")
    gate = g1 and g2 and g3
    print(f"  decision gate: {'PASS' if gate else 'FAIL'}")

    metrics.update(gate_pass=gate, gate1_divergence=g1, gate2_cl2=g2, gate3_lag=g3,
                   rms_ratio_a=ratios["a"], rms_ratio_b=ratios["b"],
                   rms_ratio_bend_a=ratios_bend["a"], rms_ratio_bend_b=ratios_bend["b"])
    key = f"eta0p18_0p25_sig2_seeds{CFG['n_seeds']}_T{T}"
    run_id = log_run(task="valley_regimes", probe="trajectory", key=key,
                     config={**CFG, "betas": BETAS, "panels": PANELS}, metrics=metrics,
                     arrays=arrays, sha=SHA)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
