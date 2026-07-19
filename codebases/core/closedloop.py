"""Closed-loop and filter-first theory guides (CL-1, CL-2, CL-3, T2, T5), one definition each.

Reference formulas from `discussions/theory_cl_t2_t5.md` and `discussions/theory_cl3_t5b.md`,
written once here and imported by the
experiment drivers and figure modules (../CODING.md Pillar 2) so every dashed guide in a figure
traces to a single audited definition. Conventions match `discussions/idea_v1.md`:

    EMA momentum      m_t = beta * m_{t-1} + (1 - beta) * g_t,   m_0 = 0
    closed hill loop  g_t = lam * y_t + xi_t,  y_{t+1} = y_t - eta * m_t,  xi white, var sigma^2
    T_eff             (1 + beta) / (1 - beta)

All functions are elementary and vectorized over their first argument where noted.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "stability_threshold",
    "stationary_hill_var",
    "stationary_hill_std",
    "stationary_hill_loss",
    "cl3_relative_reduction",
    "hill_gradient_spectrum",
    "filtered_power_ratio",
    "nyquist_ema_coeff",
    "wedin_sin_theta_bound",
    "ema_noise_opnorm_guide",
    "low_band_distortion",
    "lag_edge_beta",
    "forced_gain",
    "resonant_frequency",
    "forced_hill_loss",
]


def _teff(beta: float) -> float:
    return (1.0 + beta) / (1.0 - beta)


def stability_threshold(beta: float) -> float:
    """CL-1: the closed hill loop is stable iff eta*lam < 2*(1+beta)/(1-beta) = 2*T_eff."""
    return 2.0 * _teff(beta)


def stationary_hill_var(eta: float, lam: float, sigma: float, beta: float,
                        fp2: float = 0.0) -> float:
    """CL-2: stationary hill variance Var(y) = eta*sigma^2 / (lam*(2 - eta*lam(1+fp2)/T_eff)).

    Exact for the linear (straight-valley) hill under white gradient noise (`fp2 = 0`), for
    any eta*lam < 2*T_eff. `fp2` = mean squared river-floor slope E[f'(x)^2] applies the
    curved-valley correction: linearizing the offset dynamics, the local hill sharpness is
    lam*(1+f'^2) and the hill-normal noise variance sigma^2*(1+f'^2); the (1+f'^2) factors
    cancel except inside the (2 - .) denominator. Returns NaN outside the stability region.
    """
    if eta * lam * (1.0 + fp2) >= stability_threshold(beta):
        return float("nan")
    return eta * sigma**2 / (lam * (2.0 - eta * lam * (1.0 + fp2) / _teff(beta)))


def stationary_hill_std(eta: float, lam: float, sigma: float, beta: float,
                        fp2: float = 0.0) -> float:
    """sqrt of `stationary_hill_var` — the rms guide plotted against measured tail rms."""
    return float(np.sqrt(stationary_hill_var(eta, lam, sigma, beta, fp2)))


def stationary_hill_loss(eta: float, lam: float, sigma: float, beta: float,
                         fp2: float = 0.0) -> float:
    """CL-3: stationary expected hill loss E[(lam/2) y^2] = eta*sigma^2 / (2(2 - eta*lam/T_eff)).

    (lam/2) x `stationary_hill_var`; strictly decreasing in beta on the stable range, from
    the SGD value at beta = 0 to the saturation eta*sigma^2/4 as beta -> 1. NaN outside
    stability. Heavy-ball normalization at fixed eta_HB is obtained by calling with
    eta = eta_HB/(1-beta) (the exact EMA rewriting of the HB update), under which the
    small-eta*lam monotonicity inverts.
    """
    return 0.5 * lam * stationary_hill_var(eta, lam, sigma, beta, fp2)


def cl3_relative_reduction(etalam: float) -> float:
    """CL-3: maximal relative stationary-hill-loss reduction over beta at fixed eta*lam.

    1 - lim_{beta->1} loss(beta)/loss(0) = eta*lam/2 for eta*lam < 2; for eta*lam >= 2 the
    beta = 0 loop diverges and the reduction is 1 (capped).
    """
    return min(etalam, 2.0) / 2.0


def hill_gradient_spectrum(omega, etalam: float, sigma: float = 1.0):
    """T2: stationary spectral density of the beta=0 hill-gradient stream.

    S_g(w) = sigma^2 * |1-e^{-iw}|^2 / |1-a e^{-iw}|^2 with a = 1 - eta*lam. Zero at DC,
    strictly increasing on [0, pi], peak S_g(pi) = 4 sigma^2/(2-eta*lam)^2. Vectorized in omega.
    """
    omega = np.asarray(omega, dtype=float)
    a = 1.0 - etalam
    e = np.exp(-1j * omega)
    return sigma**2 * np.abs(1.0 - e) ** 2 / np.abs(1.0 - a * e) ** 2


def filtered_power_ratio(beta: float, etalam: float) -> float:
    """T2 corollary: exact open-loop ratio Var(EMA_beta(g))/Var(g) for the T2 stream.

    Partial-fraction closed form; satisfies 1/T_eff^2 <~ ratio <= 1/T_eff (Chebyshev), the
    upper end reached as etalam/(1-beta) -> 0, the lower as etalam -> 2. Near the
    repeated-pole point a = beta (etalam = 1-beta) the partial fractions cancel
    catastrophically in floating point, so within 1e-3 of it the value is taken as the
    symmetric average at +-1e-3 (relative error ~1e-6; the ratio is smooth there).
    """
    a = 1.0 - etalam
    if abs(a - beta) < 1e-3:  # repeated pole: evaluate the smooth limit from outside
        return 0.5 * (filtered_power_ratio(beta, etalam + 1e-3)
                      + filtered_power_ratio(beta, etalam - 1e-3))
    A = (1.0 - beta) / (a - beta)
    B = -etalam / (a - beta)
    var_m = (1.0 - beta) ** 2 * (A**2 / (1.0 - beta**2) + 2.0 * A * B / (1.0 - a * beta)
                                 + B**2 / (1.0 - a**2))
    return float(var_m / (2.0 / (2.0 - etalam)))


def nyquist_ema_coeff(t, beta: float):
    """T5(a): exact EMA coefficient of a Nyquist tone, m_t = eps_t * A for g_t = (-1)^t A.

    eps_t = (-1)^t (1-beta)(1-(-beta)^t)/(1+beta); |eps_t| -> 1/T_eff. Vectorized in t
    (t = 1, 2, ... — the first filtered sample is t = 1).
    """
    t = np.asarray(t, dtype=float)
    return (-1.0) ** t * (1.0 - beta) * (1.0 - (-beta) ** t) / (1.0 + beta)


def wedin_sin_theta_bound(eps_abs, normA_op: float, sigma_r: float, shrink=1.0,
                          resid_op: float | None = None):
    """T5(c): pre-polar subspace-error guide  |eps_t| resid / (shrink*sigma_r - |eps_t| ||A||_2).

    `eps_abs` = |eps_t| (scalar or array); `shrink*sigma_r` = the r-th singular value of the
    filtered signal (shrink = 1 - beta^t for a constant signal, or pass the measured
    sigma_r(EMA(S)_t) as `shrink` with sigma_r = 1 for a drifting one); `resid_op` = the
    projected numerator max(||A V_S||_2, ||A^T U_S||_2) (defaults to the crude ||A||_2).
    Returns NaN where the gap condition shrink*sigma_r > |eps_t| ||A||_2 fails.
    """
    eps_abs = np.asarray(eps_abs, dtype=float)
    shrink = np.asarray(shrink, dtype=float)
    num = eps_abs * (normA_op if resid_op is None else resid_op)
    gap = shrink * sigma_r - eps_abs * normA_op
    out = np.where(gap > 0, num / np.where(gap > 0, gap, 1.0), np.nan)
    return out


def low_band_distortion(beta: float, omega) -> np.ndarray:
    """T1's low-band distortion eps_low at frequency omega:  |H_beta(omega) - 1|.

    Closed form 2 beta sin(omega/2) / sqrt(1 - 2 beta cos omega + beta^2); increasing in both
    beta and omega on [0, pi], with eps_low(beta, omega) -> 1 as beta -> 1 for omega > 0.
    Vectorized in omega.
    """
    omega = np.asarray(omega, dtype=float)
    return (2.0 * beta * np.sin(omega / 2.0)
            / np.sqrt(1.0 - 2.0 * beta * np.cos(omega) + beta**2))


def lag_edge_beta(omega_bend: float, eps_target: float) -> float:
    """E10 guide: the beta at which the low-band distortion at the bend frequency reaches
    `eps_target` — the predicted upper edge of the good-beta window on a curved river.

    A river floor bending at temporal frequency omega_bend = k * v (bend wavenumber x mean
    river speed per step) sits in the filter's passband only while
    `low_band_distortion(beta, omega_bend) < eps_target`; beyond it the buffer lags the
    tangent. eps_target is calibrated once on an anchor cell and the *shape* over k is the
    prediction (a guide in the sense of `ema_noise_opnorm_guide`: derived scale, calibrated
    constant). Solved by bisection (the distortion is increasing in beta); returns 1.0 if
    even beta -> 1 stays below the target.
    """
    if not 0.0 < eps_target < 1.0:
        raise ValueError(f"eps_target must be in (0, 1); got {eps_target}")
    if low_band_distortion(1.0 - 1e-9, omega_bend) <= eps_target:
        return 1.0
    lo, hi = 0.0, 1.0 - 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if low_band_distortion(mid, omega_bend) < eps_target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def ema_noise_opnorm_guide(sigma: float, m: int, n: int, beta: float, t=None):
    """Approximate operator-norm level of EMA_beta(Xi)_t for iid N(0, sigma^2) entries.

    Entrywise std is sigma*sqrt((1-beta)(1-beta^{2t})/(1+beta)); for an iid Gaussian m x n
    matrix E||G||_2 ~= entry_std * (sqrt(m)+sqrt(n)). A guide (concentration heuristic, not a
    theorem) for the stochastic floor in E5-style overlays; the deterministic T5 statements do
    not use it.
    """
    fac = 1.0 if t is None else (1.0 - beta ** (2.0 * np.asarray(t, dtype=float)))
    entry_std = sigma * np.sqrt((1.0 - beta) * fac / (1.0 + beta))
    return entry_std * (np.sqrt(m) + np.sqrt(n))


def forced_gain(omega, beta: float, eta: float, lam: float):
    """FR: closed-loop forced-response gain |G_beta(omega)| = eta(1-beta)/|p(e^{i omega})|.

    Stationary amplitude of the hill coordinate per unit tone amplitude when the tone
    A cos(omega t) is added to the hill gradient (discussions/theory_fr_cv_be.md, Prop FR).
    Special values: |G(0)| = 1/lam for every beta (passband bias is never removed);
    |G(pi)| = eta(1-beta)/(2(1+beta)-e), decreasing in beta; at the hill-mode frequency
    `resonant_frequency` the gain grows like sqrt(eta/(lam(1-beta))) as beta -> 1.
    Vectorized over omega.
    """
    z = np.exp(1j * np.asarray(omega, dtype=float))
    e = eta * (1.0 - beta) * lam
    p = z * z - (1.0 + beta - e) * z + beta
    return eta * (1.0 - beta) / np.abs(p)


def resonant_frequency(beta: float, eta: float, lam: float) -> float:
    """FR: underdamped hill-mode angle theta_beta = arccos((1+beta-e)/(2 sqrt(beta))).

    Defined in the complex-pole regime (1+beta-e)^2 < 4 beta; the closed-loop poles are
    sqrt(beta) e^{+/- i theta_beta} and `forced_gain` peaks near theta_beta.
    """
    e = eta * (1.0 - beta) * lam
    c = (1.0 + beta - e) / (2.0 * np.sqrt(beta)) if beta > 0 else np.inf
    if not (0.0 < beta < 1.0) or abs(c) >= 1.0:
        raise ValueError("complex-pole regime requires 0 < beta < 1 and (1+beta-e)^2 < 4 beta")
    return float(np.arccos(c))


def forced_hill_loss(eta: float, lam: float, sigma: float, beta: float,
                     A: float, omega: float) -> float:
    """FR + CL-3: stationary E[(lam/2) y^2] under white noise plus the tone A cos(omega t).

    Superposition of the CL-2 noise variance and the tone's mean square
    (A |G|)^2/2 -- with the 1/2 replaced by 1 at omega = pi, where the +/- tone pair
    coincides. NaN outside stability.
    """
    var = stationary_hill_var(eta, lam, sigma, beta)
    fac = 1.0 if abs(omega - np.pi) < 1e-12 else 0.5
    return 0.5 * lam * (var + fac * (A * float(forced_gain(omega, beta, eta, lam))) ** 2)
