"""Analysis metrics for the momentum-filtering experiments, one definition each.

Every metric in `discussions/idea_v1.md` is implemented here exactly once and imported by the
experiment drivers and figure modules (see ../CODING.md Pillar 2). Time is always axis 0.

Grouped as:
  - temporal spectra        windowed_dft, band_masks, empirical_transfer
  - river-valley scalars     hill_suppression_ratio, river_alignment, hill_energy, lag,
                             noise_suppression_ratio, distance_rms
  - frequency-energy ratios  high_freq_energy_ratio, momentum_suppression_ratio
  - matrix (Muon) metrics    signal_alignment, subspace_sin_theta, spectral_gap
"""
from __future__ import annotations

import numpy as np

from .momentum import ema_momentum, polar_factor

__all__ = [
    "windowed_dft",
    "band_masks",
    "empirical_transfer",
    "hill_suppression_ratio",
    "river_alignment",
    "hill_energy",
    "lag",
    "noise_suppression_ratio",
    "stochastic_residual_ratio",
    "distance_rms",
    "escape_fraction",
    "high_freq_energy_ratio",
    "momentum_suppression_ratio",
    "signal_alignment",
    "subspace_sin_theta",
    "spectral_gap",
]

_EPS = 1e-12


# --- temporal spectra ------------------------------------------------------
def windowed_dft(x: np.ndarray, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """Finite-window one-sided DFT of a real time series along axis 0.

    Returns (omega, X) where omega in [0, pi] are the non-negative DFT frequencies
    omega_l = 2 pi l / T and X = rfft(window * x) has matching leading length floor(T/2)+1.
    A Hann window suppresses spectral leakage from the finite window / EMA transient; pass
    window="rect" for the raw rectangular window.

    For multi-dimensional `x` (e.g. matrix streams of shape (T, m, n)) the transform is taken
    along time and the other axes are preserved.
    """
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    if window == "hann":
        w = np.hanning(T)
    elif window == "rect":
        w = np.ones(T)
    else:
        raise ValueError(f"unknown window {window!r}")
    w_shape = (T,) + (1,) * (x.ndim - 1)
    X = np.fft.rfft(x * w.reshape(w_shape), axis=0)
    omega = 2.0 * np.pi * np.fft.rfftfreq(T)  # rfftfreq returns l/T in [0, 0.5]
    return omega, X


def band_masks(
    omega: np.ndarray, lo_frac: float = 0.15, hi_frac: float = 0.6
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (low, high) frequency-band masks over omega in [0, pi].

    low band  = {omega <= lo_frac * pi},  high band = {omega >= hi_frac * pi}.
    """
    low = omega <= lo_frac * np.pi
    high = omega >= hi_frac * np.pi
    return low, high


def empirical_transfer(
    g: np.ndarray, m: np.ndarray, window: str = "hann"
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical magnitude ratio R(omega) = |m_hat(omega)| / |g_hat(omega)| (Experiment E4).

    Both streams are scalar series of shape (T,). Returns (omega, R).
    """
    omega, G = windowed_dft(g, window)
    _, M = windowed_dft(m, window)
    return omega, np.abs(M) / (np.abs(G) + _EPS)


# --- river-valley scalar metrics -------------------------------------------
def hill_suppression_ratio(m_hill: np.ndarray, g_hill: np.ndarray) -> float:
    """HSR = sum_t |m_hill|^2 / sum_t |g_hill|^2 (energy in the hill component, momentum/raw)."""
    m_hill = np.asarray(m_hill, dtype=float)
    g_hill = np.asarray(g_hill, dtype=float)
    return float(np.sum(m_hill**2) / (np.sum(g_hill**2) + _EPS))


def river_alignment(v: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Per-step river alignment |<v_t, r_t>| / |v_t|.

    `v` has shape (T, d). `r` is either a single unit vector (d,) or a per-step stream (T, d).
    Returns an array of shape (T,).
    """
    v = np.asarray(v, dtype=float)
    r = np.asarray(r, dtype=float)
    if r.ndim == 1:
        r = np.broadcast_to(r, v.shape)
    dots = np.abs(np.sum(v * r, axis=-1))
    return dots / (np.linalg.norm(v, axis=-1) + _EPS)


def hill_energy(m: np.ndarray, n: np.ndarray) -> float:
    """HillEnergy = sum_t |<m_t, n_t>|^2 (energy of the buffer along the hill normal)."""
    m = np.asarray(m, dtype=float)
    n = np.asarray(n, dtype=float)
    if n.ndim == 1:
        n = np.broadcast_to(n, m.shape)
    return float(np.sum(np.sum(m * n, axis=-1) ** 2))


def lag(m: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Per-step river-following lag 1 - |<m_t, r_t>| / |m_t| (Experiment E2)."""
    return 1.0 - river_alignment(m, r)


def noise_suppression_ratio(m: np.ndarray, g: np.ndarray, grad_exact: np.ndarray) -> float:
    """NSR = sum_t |m_t - grad_exact_t|^2 / sum_t |g_t - grad_exact_t|^2 (idea_v1.md, E3).

    The plan's literal definition, comparing the buffer to the *instantaneous* exact gradient.
    In a moving closed-loop trajectory it folds the deterministic lag bias EMA(grad_exact) -
    grad_exact together with the filtered noise, so it is non-monotone in beta (the T6 filtering-
    lag tradeoff). For the noise component alone, use `stochastic_residual_ratio`.
    """
    m = np.asarray(m, dtype=float)
    g = np.asarray(g, dtype=float)
    grad_exact = np.asarray(grad_exact, dtype=float)
    num = np.sum((m - grad_exact) ** 2)
    den = np.sum((g - grad_exact) ** 2)
    return float(num / (den + _EPS))


def stochastic_residual_ratio(
    m: np.ndarray, g: np.ndarray, grad_exact: np.ndarray, beta: float
) -> float:
    """Noise-only residual sum_t |m_t - EMA_beta(grad_exact)_t|^2 / sum_t |g_t - grad_exact_t|^2.

    Isolates the *stochastic* component that survives filtering: because the EMA is linear,
    m_t - EMA_beta(grad_exact)_t = EMA_beta(xi)_t with xi_t = g_t - grad_exact_t, so this ratio is
    the filter's attenuation of the realized noise and tends to (1-beta)/(1+beta) = 1/N_eff for
    white noise. It removes the deterministic lag bias that `noise_suppression_ratio` includes,
    and matches the quantity bounded by Theory T2.
    """
    m = np.asarray(m, dtype=float)
    g = np.asarray(g, dtype=float)
    grad_exact = np.asarray(grad_exact, dtype=float)
    m_clean = ema_momentum(grad_exact, beta)
    num = np.sum((m - m_clean) ** 2)
    den = np.sum((g - grad_exact) ** 2)
    return float(num / (den + _EPS))


def distance_rms(d: np.ndarray) -> float:
    """RMS distance to the river floor sqrt(mean_t d_t^2) (d_t = hill coordinate)."""
    d = np.asarray(d, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def escape_fraction(d: np.ndarray, radius: float, tail_frac: float = 0.5) -> float:
    """Fraction of tail steps with |d_t| > radius — the tube-escape measure (E8, E3 heatmap).

    `d` is the hill-coordinate series (T,) (NaN entries, e.g. after divergence, count as
    escaped). The tail window is the last `tail_frac` of the run. A trajectory is called
    escaped when this fraction is positive; aggregating the indicator over seeds gives the
    escape frequency.
    """
    d = np.asarray(d, dtype=float)
    tail = d[int(len(d) * (1.0 - tail_frac)):]
    out = np.abs(tail) > radius
    return float(np.mean(out | ~np.isfinite(tail)))


# --- frequency-energy ratios -----------------------------------------------
def high_freq_energy_ratio(
    x: np.ndarray, window: str = "hann", hi_frac: float = 0.6
) -> float:
    """HFER = (energy in high band) / (total energy), Frobenius-summed over non-time axes.

    `x` has shape (T, ...). High band = {omega >= hi_frac * pi}; DC (omega = 0) is excluded
    from the denominator so a slowly drifting mean does not dominate.
    """
    omega, X = windowed_dft(x, window)
    _, high = band_masks(omega, hi_frac=hi_frac)
    p = np.abs(X) ** 2
    axes_tail = tuple(range(1, X.ndim))
    p_freq = np.sum(p, axis=axes_tail) if X.ndim > 1 else p  # energy per frequency
    nonzero = omega > 0
    total = np.sum(p_freq[nonzero])
    return float(np.sum(p_freq[high]) / (total + _EPS))


def momentum_suppression_ratio(
    g: np.ndarray, m: np.ndarray, window: str = "hann", hi_frac: float = 0.6
) -> float:
    """MSR = (high-band energy of m) / (high-band energy of g), Frobenius-summed (Experiment E6)."""
    omega, G = windowed_dft(g, window)
    _, M = windowed_dft(m, window)
    _, high = band_masks(omega, hi_frac=hi_frac)
    axes_tail = tuple(range(1, G.ndim))
    eg = np.sum(np.abs(G) ** 2, axis=axes_tail) if G.ndim > 1 else np.abs(G) ** 2
    em = np.sum(np.abs(M) ** 2, axis=axes_tail) if M.ndim > 1 else np.abs(M) ** 2
    return float(np.sum(em[high]) / (np.sum(eg[high]) + _EPS))


# --- matrix (Muon) metrics -------------------------------------------------
def signal_alignment(A: np.ndarray, S: np.ndarray, rank: int | None = None) -> float:
    """Align(A, S) = <A, O_rank(S)>_F / d (Experiment E5).

    A is a candidate update (typically orthogonal, e.g. O(M_t) or a filtered buffer); S is the
    slow signal matrix whose orthogonal factor defines the ideal direction. With `rank=None`,
    O(S) is the full polar factor and d = min(m, n) (the idea_v1.md definition). With `rank=r`,
    the ideal direction is the rank-r factor U_r V_r^T and d = r, which is the faithful target
    when S is low rank (the full polar factor completes the null directions arbitrarily). Equals
    1 when A matches the (rank-r) orthogonal factor of S on the signal directions.
    """
    A = np.asarray(A, dtype=float)
    S = np.asarray(S, dtype=float)
    if rank is None:
        return float(np.sum(A * polar_factor(S)) / min(A.shape))
    U, _, Vt = np.linalg.svd(S, full_matrices=False)
    ideal = U[:, :rank] @ Vt[:rank]
    return float(np.sum(A * ideal) / rank)


def subspace_sin_theta(Ua: np.ndarray, Us: np.ndarray) -> float:
    """Largest principal-angle sine ||sin Theta(Ua, Us)||_2 between two column subspaces.

    Ua, Us have orthonormal columns (n x r). Returns sqrt(max(0, 1 - sigma_min(Ua^T Us)^2)).
    """
    Ua = np.asarray(Ua, dtype=float)
    Us = np.asarray(Us, dtype=float)
    s = np.linalg.svd(Ua.T @ Us, compute_uv=False)
    cos_min = np.clip(s.min(), 0.0, 1.0)
    return float(np.sqrt(max(0.0, 1.0 - cos_min**2)))


def spectral_gap(M: np.ndarray, r: int) -> float:
    """Spectral gap sigma_r(M) - sigma_{r+1}(M) (1-indexed r) of a matrix M."""
    s = np.linalg.svd(np.asarray(M, dtype=float), compute_uv=False)
    if r < 1 or r >= len(s):
        raise ValueError(f"r must satisfy 1 <= r < {len(s)}; got {r}")
    return float(s[r - 1] - s[r])
