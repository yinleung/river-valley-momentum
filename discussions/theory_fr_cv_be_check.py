"""Numerical checks for theory_fr_cv_be.md: FR (closed-loop forced response),
CV (curved-valley reduction), BE (band-energy filter-first).

Standalone numpy; Check B3 reads the E10 run record from codebases/results/cache/.
Run:  python discussions/theory_fr_cv_be_check.py   (from the repo root; ~20 s)
Every [check] line must end PASS.
"""
from __future__ import annotations

import pathlib

import numpy as np

rng = np.random.default_rng(0)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def ok(name: str, cond: bool, detail: str = "") -> None:
    print(f"[check] {name}: {'PASS' if cond else 'FAIL'}  {detail}")
    assert cond, name


# ---------------------------------------------------------------- FR: forced response
def p_poly(z: complex, beta: float, e: float) -> complex:
    return z * z - (1.0 + beta - e) * z + beta


def gain(omega: float, beta: float, eta: float, lam: float) -> float:
    e = eta * (1.0 - beta) * lam
    return eta * (1.0 - beta) / abs(p_poly(np.exp(1j * omega), beta, e))


def simulate_hill(beta, eta, lam, T, s=None, sigma=0.0, seed=0):
    """Closed-loop EMA hill coordinate: m_t = b m + (1-b)(lam y + xi + s), y -= eta m."""
    r = np.random.default_rng(seed)
    y = np.zeros(T + 1)
    y[0] = 0.0
    m = 0.0
    for t in range(T):
        g = lam * y[t] + (s[t] if s is not None else 0.0) \
            + (sigma * r.standard_normal() if sigma else 0.0)
        m = beta * m + (1.0 - beta) * g
        y[t + 1] = y[t] - eta * m
    return y[1:]


def check_A() -> None:
    eta, lam = 0.18, 10.0  # eta*lam = 1.8, the paper's running example
    # A1: lock-in amplitude of the stationary response vs A |G_beta(omega)|
    T, A = 40000, 0.7
    worst = 0.0
    for beta in (0.0, 0.5, 0.9, 0.99):
        e = eta * (1 - beta) * lam
        th = float(np.arccos((1 + beta - e) / (2 * np.sqrt(beta)))) if beta > 0 else 0.9
        for omega in (0.05, th, 0.6 * np.pi, np.pi):
            s = A * np.cos(omega * np.arange(T))
            y = simulate_hill(beta, eta, lam, T, s=s)
            tail = y[T // 2:]
            t_idx = np.arange(T // 2, T)
            # at omega = pi the +/- tone pair coincides: no factor 2
            fac = 1.0 if abs(omega - np.pi) < 1e-12 else 2.0
            amp = fac * abs(np.mean(tail * np.exp(-1j * omega * t_idx)))
            pred = A * gain(omega, beta, eta, lam)
            worst = max(worst, abs(amp - pred) / pred)
    ok("A1 lock-in amplitude = A|G| (all beta x omega)", worst < 2e-2,
       f"worst rel err {worst:.2e}")

    # A2: special values -- DC gain 1/lam for every beta; Nyquist closed form;
    # resonance asymptotic sqrt(eta/(lam(1-beta)))
    dc = [gain(0.0, b, eta, lam) for b in (0.0, 0.5, 0.9, 0.99, 0.999)]
    ok("A2 DC gain = 1/lambda, beta-free", np.allclose(dc, 1 / lam, rtol=1e-12),
       f"values {np.round(dc, 6)}")
    for beta in (0.5, 0.9, 0.99):
        e = eta * (1 - beta) * lam
        ny = eta * (1 - beta) / (2 * (1 + beta) - e)
        ok(f"A2 Nyquist gain closed form (beta={beta})",
           abs(gain(np.pi, beta, eta, lam) - ny) < 1e-14, f"{ny:.5f}")
    print("  resonance: beta, theta_b, |G(theta)|, asymptote sqrt(eta/(lam(1-b))), "
          "amplification vs beta=0")
    for beta in (0.9, 0.99, 0.999):
        e = eta * (1 - beta) * lam
        th = float(np.arccos((1 + beta - e) / (2 * np.sqrt(beta))))
        g_res = gain(th, beta, eta, lam)
        asym = np.sqrt(eta / (lam * (1 - beta)))
        amp0 = g_res / gain(th, 0.0, eta, lam)
        print(f"    {beta:6.3f}  {th:.4f}  {g_res:8.4f}  {asym:8.4f}  {amp0:8.1f}x")
    b = 0.999
    e = eta * (1 - b) * lam
    th = float(np.arccos((1 + b - e) / (2 * np.sqrt(b))))
    ok("A2 resonance ~ sqrt(eta/(lam(1-beta))) (beta=0.999 within 10%)",
       abs(gain(th, b, eta, lam) / np.sqrt(eta / (lam * (1 - b))) - 1) < 0.10)

    # A3: tone + noise additivity of the stationary hill loss
    beta, omega, A, sigma, T = 0.9, np.pi, 1.5, 2.0, 60000
    e = eta * (1 - beta) * lam
    var_cl = eta * sigma**2 / (lam * (2 - eta * lam / ((1 + beta) / (1 - beta))))
    pred = 0.5 * lam * (var_cl + 0.5 * A**2 * gain(omega, beta, eta, lam) ** 2)
    s = A * np.cos(omega * np.arange(T))
    losses = [np.mean(0.5 * lam * simulate_hill(beta, eta, lam, T, s=s, sigma=sigma,
                                                seed=sd)[T // 2:] ** 2)
              for sd in range(8)]
    meas = float(np.mean(losses))
    ok("A3 stationary hill loss = noise part + A^2|G|^2/2 part",
       abs(meas - pred) / pred < 0.05, f"measured {meas:.4f} vs {pred:.4f}")


# ------------------------------------------------- CV: curved-valley reduction
def check_B() -> None:
    # B1: the exact GD one-step identity on f(x) = a sin(kx):
    # delta_{t+1} = (1 - eta lam (1+f'^2)) delta_t + eta f' phi' + R,  |R| <= max|f''|/2 dx^2
    a, k, lam, mu, eta = 2.0, 0.9, 10.0, 0.1, 0.18
    xs, ys = 0.3, 2.1  # start off the floor
    x, y = xs, ys
    worst_excess = -np.inf
    for _ in range(400):
        fp = a * k * np.cos(k * x)
        delta = y - a * np.sin(k * x)
        gx = mu * x - lam * delta * fp
        gy = lam * delta
        x1, y1 = x - eta * gx, y - eta * gy
        d1 = y1 - a * np.sin(k * x1)
        main = (1 - eta * lam * (1 + fp**2)) * delta + eta * fp * (mu * x)
        R = d1 - main
        bound = 0.5 * a * k**2 * (eta * gx) ** 2
        worst_excess = max(worst_excess, abs(R) - bound * (1 + 1e-9))
        x, y = x1, y1
        if not np.isfinite(x + y):
            break
    ok("B1 exact curved GD recursion, remainder <= max|f''|/2 (eta g_x)^2",
       worst_excess <= 1e-12, f"worst |R|-bound {worst_excess:.2e}")

    # B2: tilted straight floor f(x) = c x, linear river potential phi' = phi0:
    # frozen-coefficient reduction is exact (f''=0, f' constant); prediction:
    # mean offset  = c phi0 / (lam (1+c^2))          (beta-free: the DC gain)
    # Var(delta)   = eta sigma^2 / (lam (2 - eta lam_loc / Teff))   with lam_loc = lam(1+c^2)
    c, phi0, lam, eta, sigma, T = 1.5, 0.8, 10.0, 0.05, 1.0, 200000
    lam_loc = lam * (1 + c**2)
    pred_mean = c * phi0 / lam_loc
    print("  tilted floor: beta, mean offset (pred %.5f), var ratio meas/pred"
          % pred_mean)
    for beta in (0.0, 0.5, 0.9, 0.95):
        teff = (1 + beta) / (1 - beta)
        pred_var = eta * sigma**2 / (lam * (2 - eta * lam_loc / teff))
        r = np.random.default_rng(3)
        xv, yv, m = 0.0, 0.0, np.zeros(2)
        ds = np.empty(T)
        for t in range(T):
            d = yv - c * xv
            g = np.array([phi0 - lam * d * c + sigma * r.standard_normal(),
                          lam * d + sigma * r.standard_normal()])
            m = beta * m + (1 - beta) * g
            xv, yv = xv - eta * m[0], yv - eta * m[1]
            ds[t] = yv - c * xv
        tail = ds[T // 2:]
        mmean, mvar = float(np.mean(tail)), float(np.var(tail))
        print(f"    {beta:4.2f}  {mmean:+.5f}  {mvar / pred_var:.3f}")
        ok(f"B2 mean offset beta-free = c phi'/lam_loc (beta={beta})",
           abs(mmean - pred_mean) < 0.005, f"{mmean:+.5f} vs {pred_mean:+.5f}")
        ok(f"B2 Var(delta) = CL-2 with lam_loc (beta={beta})",
           abs(mvar / pred_var - 1) < 0.06, f"ratio {mvar / pred_var:.3f}")

    # B3: E10 confinement floors vs the frozen-coefficient worst-case prediction
    cache = ROOT / "codebases" / "results" / "cache" / \
        "bend_window_sweep_k5_el2_b10_seeds16_c54d90a6" / "arrays.npz"
    d = np.load(cache)
    betas, ks, etalams = d["betas"], d["ks"], d["etalams"]
    a_amp = 2.0  # E10 landscape amplitude f = 2 sin(kx)
    print("  E10 floors: etalam, k, measured floor, predicted floor "
          "(min grid beta with etalam(1+(ak)^2) < 2 Teff)")
    hits = steps = 0
    for i, el in enumerate(etalams):
        for j, kk in enumerate(ks):
            need = el * (1 + (a_amp * kk) ** 2) / 2.0
            pred = next((b for b in betas if (1 + b) / (1 - b) > need), None)
            meas = d["floor_edge"][i, j]
            gap = abs(int(np.argmin(np.abs(betas - meas)))
                      - int(np.argmin(np.abs(betas - pred))))
            hits += gap == 0
            steps += gap <= 1
            print(f"    {el:.1f}  {kk:.1f}  {meas:.2f}  {pred:.2f}  "
                  f"({'exact' if gap == 0 else f'{gap} grid step'})")
    ok("B3 measured E10 floor within one grid step of prediction (10/10)",
       steps == 10, f"exact {hits}/10, within one step {steps}/10")


# ------------------------------------------------- BE: band-energy filter-first
def ema(x: np.ndarray, beta: float) -> np.ndarray:
    m = np.zeros_like(x[0], dtype=float)
    out = np.empty_like(x, dtype=float)
    for t in range(len(x)):
        m = beta * m + (1 - beta) * x[t]
        out[t] = m
    return out


def synth_stream(T, shape, gamma, om_c, rr):
    """Real stream with an exact low-band (folded < om_c, incl. DC) energy fraction gamma."""
    grid = 2 * np.pi * np.arange(T) / T
    folded = np.minimum(grid, 2 * np.pi - grid)
    low = folded < om_c
    Z = rr.standard_normal((T, *shape)) + 1j * rr.standard_normal((T, *shape))
    Z[0] = Z[0].real  # DC real
    if T % 2 == 0:
        Z[T // 2] = Z[T // 2].real
    for l in range(1, (T + 1) // 2):  # conjugate symmetry -> real stream
        Z[T - l] = np.conj(Z[l])
    e_low = np.sum(np.abs(Z[low]) ** 2)
    e_high = np.sum(np.abs(Z[~low]) ** 2)
    if gamma == 0.0:
        Z[low] = 0.0
    else:
        Z[low] *= np.sqrt(gamma / (1 - gamma) * e_high / e_low)
    A = np.fft.ifft(Z, axis=0).real * np.sqrt(T)  # scale to O(1) entries
    return A


def check_C() -> None:
    T, shape, om_c = 512, (12, 8), 0.6 * np.pi
    grid = 2 * np.pi * np.arange(T) / T
    folded = np.minimum(grid, 2 * np.pi - grid)
    low = folded < om_c

    # C1: exact response identity EMA(A)_t = (1/T) sum_w H(w)(e^{iwt} - beta^t) Ahat(w)
    A = rng.standard_normal((T, *shape))
    beta = 0.9
    H = (1 - beta) / (1 - beta * np.exp(-1j * grid))
    ts = np.arange(1, T + 1)
    Ah1 = np.tensordot(np.exp(-1j * np.outer(grid, ts)), A, axes=(1, 0))  # u=1..T DFT
    kernel = (np.exp(1j * np.outer(grid, ts)) - beta ** ts[None, :]) * H[:, None]
    resp = np.tensordot(kernel, Ah1, axes=(0, 0)) / T  # (t, m, n)
    direct = ema(A, beta)
    err = np.max(np.abs(resp.real - direct)) / np.max(np.abs(direct))
    ok("C1 exact per-tone response identity", err < 1e-9, f"max rel err {err:.1e}")

    # C2: corollary bounds on synthetic band-limited-with-leakage streams + rank-r signal
    r, sig_r = 2, 3.0
    U, _ = np.linalg.qr(rng.standard_normal((shape[0], r)))
    V, _ = np.linalg.qr(rng.standard_normal((shape[1], r)))
    S = sig_r * (U @ V.T)
    T0, theta = T // 2, 0.1
    for beta in (0.5, 0.9, 0.99):
        rho_h = (1 - beta) / np.sqrt(1 - 2 * beta * np.cos(om_c) + beta**2)
        for gamma in (0.0, 0.05, 0.3):
            A = synth_stream(T, shape, gamma, om_c, np.random.default_rng(7))
            A *= 0.5 / np.sqrt(np.mean(np.sum(A**2, axis=(1, 2))))  # alpha_rms = 0.5
            E_A = float(np.sum(A**2))
            At = ema(A, beta)  # buffer response to the disturbance alone
            rho_bar = np.sqrt(rho_h**2 + gamma) \
                + beta ** (T0 + 1) * (rho_h + np.sqrt(gamma)) / np.sqrt(1 - beta**2)
            lhs = np.sqrt(np.sum(At[T0:] ** 2))
            ok(f"C2 energy bound (beta={beta}, gamma={gamma})",
               lhs <= rho_bar * np.sqrt(E_A) * (1 + 1e-12),
               f"ratio {lhs / (rho_bar * np.sqrt(E_A)):.3f}")
            per = np.sum(At[T0:] ** 2, axis=(1, 2))
            thr = rho_bar**2 * E_A / (theta * (T - T0))
            ok(f"C2 exceptional fraction <= theta (beta={beta}, gamma={gamma})",
               np.mean(per > thr) <= theta, f"frac {np.mean(per > thr):.3f}")
            # subspace bound on non-exceptional steps
            worst = 0.0
            for t in np.nonzero(per <= thr)[0][::16]:
                tt = T0 + t + 1  # 1-indexed step
                Mt = (1 - beta**tt) * S + At[T0 + t]
                bnd = np.sqrt(thr)
                gap = (1 - beta**tt) * sig_r - bnd
                if gap <= 0:
                    continue
                Um = np.linalg.svd(Mt)[0][:, :r]
                sin_t = np.sqrt(max(0.0, 1 - np.linalg.svd(Um.T @ U)[1].min() ** 2))
                worst = max(worst, sin_t - bnd / gap)
            ok(f"C2 sin-theta bound on kept steps (beta={beta}, gamma={gamma})",
               worst <= 1e-12, f"worst excess {worst:.1e}")


if __name__ == "__main__":
    print("== FR: closed-loop forced response ==")
    check_A()
    print("== CV: curved-valley reduction ==")
    check_B()
    print("== BE: band-energy filter-first ==")
    check_C()
    print("all checks passed")
