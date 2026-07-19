"""Experiment E5 — matrix-valued river-valley / Muon-style toy demo (idea_v1.md, Phase 3).

Builds matrix gradients G_t = S_t + H_t + Xi_t with

    S_t = U diag(Lambda_t) V^T     slowly varying rank-r signal (low temporal frequency)
    H_t = a_H * (-1)^t * A          deterministic high-frequency (omega = pi) disturbance
    Xi_t                            i.i.d. Gaussian perturbation

and compares the three Muon-style pipelines (see core.momentum):

    pre-polar    O(EMA_beta(G)_t)            filter first, then orthogonalize
    post-polar   EMA_beta(O(G))_t            orthogonalize first, then filter (the raw buffer
                                             M-tilde_t used directly, as in idea_v1.md)
    polar-only   O(G_t)

Key test (idea_v1.md): pre-polar should dominate when the disturbance H_t is high-frequency,
*even though H_t is deterministic, not zero-mean stochastic noise* — generalizing the Muon
"denoise first" result from stochastic to temporal high-frequency filtering. The stochastic-only
scenario (H_t = 0) is included as the control that recovers the original Muon setting.

Metrics: subspace error ||sin Theta(U_pipeline, U_S)||_2 (magnitude-free headline), signal
alignment <A, O(S_t)>_F/min(m,n), and the spectral gap sigma_r - sigma_{r+1} of the buffer.

T5 overlay extension (fable_dgs_v1.md §8.2 item 3): a third scenario `highfreq_clean`
(H_t only, Xi = 0 — the exact setting of Theorem T5 in discussions/theory_cl_t2_t5.md) logs
per-step traces of the pre-polar buffer tail sigma_{r+1}(M_t) against the two-sided guide
|eps_t| sigma_{2r+1}(A) <= sigma_{r+1}(M_t) <= |eps_t| ||A||_2 (eps_t from
core.closedloop.nyquist_ema_coeff, transient included), and of the measured subspace error
against the Wedin guide |eps_t| max(||A V_S||, ||A^T U_S||) / (sigma_r(EMA(S)_t) - |eps_t| ||A||_2)
(signal shrink tracked exactly through the drifting EMA(S)). The noisy `highfreq` scenario
logs the same traces plus the stochastic floor guide (core.closedloop.ema_noise_opnorm_guide)
for reference — the T5 gates apply to the clean scenario only:

    gate T5-a  the two-sided tail bound holds at every step (exact inequality);
    gate T5-b  the tail upper guide is tight: median over the second half of
               |eps_t| ||A||_2 / sigma_{r+1}(M_t) <= 2;
    gate T5-c  sinTheta <= Wedin guide wherever the gap is positive, and the guide is tight
               at the end: bound/measured <= 3 at t = T.

Run:  cd codebases && python scripts/run_e5_matrix.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import closedloop as Cl  # noqa: E402
from core import metrics as Me  # noqa: E402
from core import momentum as Mo  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402

BETAS = [0.5, 0.9, 0.95, 0.99]
OVERLAY_BETAS = [0.9, 0.99]
CFG = dict(m=24, n=18, r=3, T=400, f_slow=0.005, drift=0.3,
           base_sv=[4.0, 3.0, 2.0], a_H=0.24, sigma_xi_hf=0.2, sigma_xi_stoch=0.30,
           burn_in=100, seed=0)


def build_streams(scenario: str):
    """Return (G stream (T,m,n), U_signal, V_signal, S stream, H amplitude a_H*A) per scenario."""
    rng = np.random.default_rng(CFG["seed"])
    m, n, r, T = CFG["m"], CFG["n"], CFG["r"], CFG["T"]
    U = np.linalg.qr(rng.standard_normal((m, r)))[0]
    V = np.linalg.qr(rng.standard_normal((n, r)))[0]
    A = rng.standard_normal((m, n))
    A = A / np.linalg.norm(A) * np.sqrt(sum(np.square(CFG["base_sv"])))  # ||H_t|| ~ ||S_t||
    t = np.arange(T)
    phase = np.array([0.0, 1.0, 2.0])[:r]
    Lam = np.stack([CFG["base_sv"][i] * (1 + CFG["drift"] *
                    np.sin(2 * np.pi * CFG["f_slow"] * t + phase[i])) for i in range(r)], axis=1)
    S = np.einsum("mr,tr,nr->tmn", U, Lam, V)
    if scenario == "highfreq":
        H = CFG["a_H"] * ((-1.0) ** t)[:, None, None] * A
        Xi = CFG["sigma_xi_hf"] * rng.standard_normal((T, m, n))
    elif scenario == "highfreq_clean":
        H = CFG["a_H"] * ((-1.0) ** t)[:, None, None] * A
        Xi = np.zeros((T, m, n))
    elif scenario == "stochastic":
        H = np.zeros((T, m, n))
        Xi = CFG["sigma_xi_stoch"] * rng.standard_normal((T, m, n))
    else:
        raise ValueError(scenario)
    return S + H + Xi, U, V, S, CFG["a_H"] * A


def pipeline_outputs(G: np.ndarray, beta: float):
    """(direction, buffer) per pipeline, from the core pipelines (idea_v1.md definitions).

    Direction outputs are the core pipeline streams: pre-polar O(EMA(G)), post-polar EMA(O(G))
    (the raw buffer M-tilde used directly, as in idea_v1.md), polar-only O(G). The subspace
    metric needs the signal-carrying *buffer*, not the orthogonal output: O(.) of a matrix has
    degenerate singular values (all ones), so its left singular vectors do not order by signal.
    Pre-polar's buffer is EMA(G); post-polar's buffer is the same EMA(O(G)) it outputs; polar-
    only's buffer is the raw gradient G.
    """
    directions = {
        "pre_polar": Mo.pre_polar_stream(G, beta),
        "post_polar": Mo.post_polar_stream(G, beta),
        "polar_only": Mo.polar_only_stream(G),
    }
    buffers = {
        "pre_polar": Mo.ema_momentum(G, beta),
        "post_polar": directions["post_polar"],
        "polar_only": G,
    }
    return {name: (directions[name], buffers[name]) for name in directions}


def _subspace_err(buffer_t: np.ndarray, U_sig: np.ndarray) -> float:
    Ub = np.linalg.svd(buffer_t, full_matrices=False)[0][:, : U_sig.shape[1]]
    return Me.subspace_sin_theta(Ub, U_sig)


def evaluate(scenario: str):
    G, U_sig, _, S, _ = build_streams(scenario)
    T, b0, r = CFG["T"], CFG["burn_in"], CFG["r"]
    rows = {}
    for beta in BETAS:
        outs = pipeline_outputs(G, beta)
        rows[beta] = {}
        for name, (direction, buffer_) in outs.items():
            align = np.mean([Me.signal_alignment(direction[t], S[t], rank=r) for t in range(b0, T)])
            sinth = np.mean([_subspace_err(buffer_[t], U_sig) for t in range(b0, T)])
            gap = np.mean([Me.spectral_gap(buffer_[t], r) for t in range(b0, T)])
            rows[beta][name] = dict(align=float(align), sin_theta=float(sinth), gap=float(gap))
    return rows


def t5_overlay(scenario: str, beta: float):
    """Per-step T5 traces: buffer tail + subspace error vs the theorem guides."""
    G, U_sig, V_sig, S, A = build_streams(scenario)
    T, r = CFG["T"], CFG["r"]
    M = Mo.ema_momentum(G, beta)
    Sm = Mo.ema_momentum(S, beta)  # EMA of the (drifting) signal stream, tracked exactly
    ts = np.arange(1, T + 1)
    eps = np.abs(Cl.nyquist_ema_coeff(ts, beta))
    normA = np.linalg.norm(A, 2)
    sv_A = np.linalg.svd(A, compute_uv=False)
    resid = max(np.linalg.norm(A @ V_sig, 2), np.linalg.norm(A.T @ U_sig, 2))
    tail_meas = np.empty(T)
    sin_meas = np.empty(T)
    sig_r_shrunk = np.empty(T)
    for t in range(T):
        sv = np.linalg.svd(M[t], compute_uv=False)
        tail_meas[t] = sv[r]
        Ub = np.linalg.svd(M[t], full_matrices=False)[0][:, :r]
        sin_meas[t] = Me.subspace_sin_theta(Ub, U_sig)
        sig_r_shrunk[t] = np.linalg.svd(Sm[t], compute_uv=False)[r - 1]
    wedin = Cl.wedin_sin_theta_bound(eps, normA, 1.0, shrink=sig_r_shrunk, resid_op=resid)
    return dict(t=ts, eps=eps, tail_meas=tail_meas, tail_hi=eps * normA,
                tail_lo=eps * sv_A[2 * r], sin_meas=sin_meas, wedin=wedin,
                noise_floor=Cl.ema_noise_opnorm_guide(
                    CFG["sigma_xi_hf"] if scenario == "highfreq" else 0.0,
                    CFG["m"], CFG["n"], beta, t=ts))


def main() -> str:
    metrics, arrays = {}, {}
    overall_pass = True
    for scenario in ("highfreq", "highfreq_clean", "stochastic"):
        rows = evaluate(scenario)
        print(f"\nE5 [{scenario}]  signal alignment (higher=better) / subspace sinTheta (lower=better)")
        print(f"  {'beta':>5} | {'pre-polar':>21} | {'post-polar':>21} | {'polar-only':>21}")
        print(f"  {'':>5} | {'align':>9} {'sinTheta':>11} | {'align':>9} {'sinTheta':>11} | "
              f"{'align':>9} {'sinTheta':>11}")
        for beta in BETAS:
            r_ = rows[beta]
            print(f"  {beta:>5} | {r_['pre_polar']['align']:>9.3f} {r_['pre_polar']['sin_theta']:>11.3f}"
                  f" | {r_['post_polar']['align']:>9.3f} {r_['post_polar']['sin_theta']:>11.3f}"
                  f" | {r_['polar_only']['align']:>9.3f} {r_['polar_only']['sin_theta']:>11.3f}")
            # pre-polar should have best (highest) alignment and lowest subspace error
            best_align = (r_["pre_polar"]["align"] >= r_["post_polar"]["align"] - 1e-6
                          and r_["pre_polar"]["align"] >= r_["polar_only"]["align"] - 1e-6)
            best_sub = (r_["pre_polar"]["sin_theta"] <= r_["post_polar"]["sin_theta"] + 1e-6
                        and r_["pre_polar"]["sin_theta"] <= r_["polar_only"]["sin_theta"] + 1e-6)
            metrics[f"{scenario}_b{beta}_prepolar_best"] = bool(best_align and best_sub)
            for name in ("pre_polar", "post_polar", "polar_only"):
                for mk in ("align", "sin_theta", "gap"):
                    metrics[f"{scenario}_b{beta}_{name}_{mk}"] = r_[name][mk]
            arrays[f"{scenario}_b{str(beta).replace('.', 'p')}_align"] = np.array(
                [rows[beta][p]["align"] for p in ("pre_polar", "post_polar", "polar_only")])
            arrays[f"{scenario}_b{str(beta).replace('.', 'p')}_sinTheta"] = np.array(
                [rows[beta][p]["sin_theta"] for p in ("pre_polar", "post_polar", "polar_only")])

        # headline check at beta=0.9
        r09 = rows[0.9]
        scen_pass = (r09["pre_polar"]["align"] > r09["post_polar"]["align"]
                     and r09["pre_polar"]["align"] > r09["polar_only"]["align"]
                     and r09["pre_polar"]["sin_theta"] < r09["post_polar"]["sin_theta"]
                     and r09["pre_polar"]["sin_theta"] < r09["polar_only"]["sin_theta"])
        print(f"  -> pre-polar dominates at beta=0.9 (align & subspace): {scen_pass}")
        overall_pass = overall_pass and scen_pass

    print(f"\n  E5 key test (pre-polar wins for deterministic high-freq AND stochastic): {overall_pass}")

    # --- T5 overlay: per-step traces vs theorem guides ------------------------
    t5_pass = True
    print("\n  T5 overlay (clean scenario gates; noisy logged for reference)")
    for scenario in ("highfreq_clean", "highfreq"):
        for beta in OVERLAY_BETAS:
            ov = t5_overlay(scenario, beta)
            tag = f"{scenario}_b{str(beta).replace('.', 'p')}"
            for k2, v2 in ov.items():
                arrays[f"ov_{tag}_{k2}"] = v2
            half = CFG["T"] // 2
            two_sided = bool(np.all(ov["tail_meas"] >= ov["tail_lo"] - 1e-9)
                             and np.all(ov["tail_meas"] <= ov["tail_hi"] + 1e-9))
            tight = float(np.median(ov["tail_hi"][half:] / ov["tail_meas"][half:]))
            ok_gap = np.isfinite(ov["wedin"])
            wedin_holds = bool(np.all(ov["sin_meas"][ok_gap] <= ov["wedin"][ok_gap] + 1e-9))
            wedin_end = float(ov["wedin"][-1] / ov["sin_meas"][-1])
            print(f"    [{tag}] two-sided tail: {two_sided}; upper-guide tightness "
                  f"(median hi/meas, tail half): {tight:.2f}; sinTheta<=Wedin: {wedin_holds}; "
                  f"Wedin/measured at T: {wedin_end:.2f}")
            metrics[f"t5_{tag}"] = dict(two_sided=two_sided, tail_tightness=tight,
                                        wedin_holds=wedin_holds, wedin_end_ratio=wedin_end)
            if scenario == "highfreq_clean":
                t5_pass = t5_pass and two_sided and tight <= 2.0 and wedin_holds \
                    and wedin_end <= 3.0
    print(f"  T5 overlay gates (clean): {'PASS' if t5_pass else 'FAIL'}")

    arrays["betas"] = np.array(BETAS)
    metrics["overall_pass"] = overall_pass
    metrics["t5_overlay_pass"] = t5_pass
    run_id = log_run(task="matrix_muon", probe="pipeline",
                     key=f"m{CFG['m']}n{CFG['n']}r{CFG['r']}_T{CFG['T']}",
                     config={**CFG, "betas": BETAS}, metrics=metrics, arrays=arrays,
                     sha=code_sha([__file__]))
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
