"""Numerical verification for theory_cl3_t5b.md (Checks A-D).

Check A  CL-3: stationary hill loss formula vs simulation; strict monotonicity in beta;
         relative-reduction limit eta*lam/2; heavy-ball-normalization inversion.
Check B  CL-3': second-moment convergence at ratio beta (complex-root regime); burn-in.
Check C  T5b: per-tone lemma (machine precision); exact buffer decomposition; bounds
         (a)/(b)/(c); Wedin tightness; SDR-gain corollary; Nyquist reduction to T5's eps_t.
Check D  B1: boundary-to-window weight — constant stream exact ratio, stationary-stream
         1/T decay, and the T ~ T_eff failure mode.

Run:  python discussions/theory_cl3_t5b_check.py     (exit 0 iff all checks pass)
"""
from __future__ import annotations

import sys

import numpy as np

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def teff(beta):
    return (1.0 + beta) / (1.0 - beta)


def hill_loss(eta, lam, sigma, beta):
    """CL-3 display: E[(lam/2) y_inf^2] = eta sigma^2 / (2 (2 - eta*lam/T_eff))."""
    den = 2.0 - eta * lam / teff(beta)
    return eta * sigma**2 / (2.0 * den) if den > 0 else np.inf


def simulate_hill(eta, lam, sigma, beta, T, n_ens=32, rng0=0):
    """Closed hill loop: g = lam*y + xi; m = b m + (1-b) g; y -= eta m.

    Vectorized ensemble; returns the tail-half mean of (lam/2) y^2 over time and members
    (inf on divergence)."""
    rng = np.random.default_rng(rng0)
    y = np.full(n_ens, 1.0)
    m = np.zeros(n_ens)
    acc, n = 0.0, 0
    for t in range(T):
        g = lam * y + sigma * rng.standard_normal(n_ens)
        m = beta * m + (1 - beta) * g
        y = y - eta * m
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e8:
            return np.inf
        if t >= T // 2:
            acc += 0.5 * lam * np.mean(y * y)
            n += 1
    return float(acc / n)


# --------------------------------------------------------------------------- Check A
def check_A():
    print("Check A — CL-3 stationary hill loss")
    betas = np.linspace(0, 0.995, 400)
    for el in (0.3, 1.8, 2.5):
        losses = np.array([hill_loss(el / 10.0, 10.0, 2.0, b) for b in betas])
        stable = np.isfinite(losses)
        d = np.diff(losses[stable])
        check(f"strictly decreasing on stable range (eta*lam={el})", np.all(d < 0),
              f"max diff {d.max():.2e}")
    # simulation match (T long, tail half; T scaled with the beta=0.99 correlation time)
    for el, b in [(1.8, 0.0), (1.8, 0.5), (1.8, 0.9), (1.8, 0.99), (2.5, 0.9), (0.3, 0.9)]:
        eta, lam, sig = el / 10.0, 10.0, 2.0
        T = 200_000 if b >= 0.99 else 60_000
        sim = simulate_hill(eta, lam, sig, b, T=T)
        pred = hill_loss(eta, lam, sig, b)
        check(f"simulation matches formula (eta*lam={el}, beta={b})",
              abs(sim / pred - 1) < 0.05, f"sim {sim:.4f} vs pred {pred:.4f}")
    # beta=0 diverges beyond threshold while beta=0.9 is finite
    check("eta*lam=2.5: beta=0 diverges, beta=0.9 finite",
          np.isinf(simulate_hill(0.25, 10.0, 2.0, 0.0, 2000))
          and np.isfinite(simulate_hill(0.25, 10.0, 2.0, 0.9, 2000)))
    # relative-reduction limit eta*lam/2
    for el in (0.3, 1.0, 1.8):
        red = 1 - hill_loss(el / 10, 10, 2.0, 0.9999) / hill_loss(el / 10, 10, 2.0, 0.0)
        check(f"relative reduction -> eta*lam/2 (eta*lam={el})",
              abs(red - el / 2) < 5e-3, f"{red:.4f} vs {el / 2:.4f}")
    # heavy-ball normalization inverts monotonicity at small eta_HB*lam
    ehb, lam, sig = 0.03, 10.0, 2.0  # eta_HB*lam = 0.3
    hb = [hill_loss(ehb / (1 - b), lam, sig, b) for b in (0.0, 0.5, 0.9, 0.95)]
    check("HB normalization: loss increasing in beta at eta_HB*lam=0.3",
          all(np.diff(hb) > 0), " ".join(f"{v:.4f}" for v in hb))
    sim_hb = [simulate_hill(ehb / (1 - b), lam, sig, b, 60_000) for b in (0.0, 0.9)]
    check("HB inversion confirmed by simulation",
          sim_hb[1] > 1.5 * sim_hb[0], f"beta=0: {sim_hb[0]:.4f}, beta=0.9: {sim_hb[1]:.4f}")


# --------------------------------------------------------------------------- Check B
def check_B():
    print("Check B — CL-3' burn-in rate")
    lam, sig = 10.0, 2.0
    for el, b in [(1.8, 0.5), (1.8, 0.9)]:
        eta = el / lam
        e = eta * (1 - b) * lam
        disc = (1 + b - e) ** 2 - 4 * b
        assert disc < 0, "test points must sit in the complex-root regime"
        # the claim is about the (exact, deterministic) moment recursion: iterate it
        A = np.array([[1 + b - e, -b], [1.0, 0.0]])
        Q = np.zeros((2, 2))
        Q[0, 0] = (eta * (1 - b) * sig) ** 2
        Sig = np.full((2, 2), 9.0)  # y_0 = y_{-1} = 3 deterministic
        Sig_inf = Sig.copy()
        for _ in range(20_000):  # fixed point by iteration
            Sig_inf = A @ Sig_inf @ A.T + Q
        T = 200
        dev = np.empty(T)
        for t in range(T):
            dev[t] = abs(Sig[0, 0] - Sig_inf[0, 0])
            Sig = A @ Sig @ A.T + Q
        # complex poles make dev oscillate; rate from the rolling-max envelope
        w = 6
        env = np.array([dev[t:t + w].max() for t in range(T - w)])
        t1, t2 = 10, 60
        ratio = float((env[t2] / env[t1]) ** (1.0 / (t2 - t1)))
        check(f"second-moment decay ratio ~ beta (eta*lam={el}, beta={b})",
              abs(ratio - b) < 0.03, f"envelope rate {ratio:.3f} vs beta {b}")
        var_inf = eta * sig**2 / (lam * (2 - eta * lam / teff(b)))
        check(f"moment recursion fixed point = CL-2 variance (beta={b})",
              abs(Sig_inf[0, 0] / var_inf - 1) < 1e-9)
    # burn-in illustration at beta=0.99
    eta, b, T = 0.18, 0.99, 6000
    rng = np.random.default_rng(2)
    n_ens = 4000
    y = np.full(n_ens, 3.0)
    m = np.zeros(n_ens)
    ey2 = np.empty(T)
    for t in range(T):
        g = lam * y + sig * rng.standard_normal(n_ens)
        m = b * m + (1 - b) * g
        y = y - eta * m
        ey2[t] = np.mean(y * y)
    var_inf = eta * sig**2 / (lam * (2 - eta * lam / teff(b)))
    k = int(teff(b))
    check("beta=0.99: tail window within 5% of Var_inf, first-T_eff window >= 2x",
          abs(np.mean(ey2[T // 2:]) / var_inf - 1) < 0.05
          and np.mean(ey2[:k]) > 2 * var_inf,
          f"tail/pred {np.mean(ey2[T // 2:]) / var_inf:.3f}, "
          f"head/pred {np.mean(ey2[:k]) / var_inf:.1f}")


# --------------------------------------------------------------------------- Check C
def check_C():
    print("Check C — T5b band-limited filter-first")
    rng = np.random.default_rng(11)
    m_, n_, r = 24, 18, 3

    def rand_orth(d, k):
        q, _ = np.linalg.qr(rng.standard_normal((d, k)))
        return q

    def H(beta, w):
        return (1 - beta) / (1 - beta * np.exp(-1j * w))

    U = rand_orth(m_, r)
    V = rand_orth(n_, r)
    svals = np.array([3.0, 2.0, 1.0])
    S = U @ np.diag(svals) @ V.T
    sr = svals[-1]
    w0, wc = 0.10 * np.pi, 0.60 * np.pi
    # slow in-subspace drift: 2 conjugate pairs, amplitude a ~ 0.2*sigma_r
    slow = []
    for w in (0.03 * np.pi, 0.08 * np.pi):
        B = (rng.standard_normal((r, r)) + 1j * rng.standard_normal((r, r)))
        B *= 0.05 / np.linalg.norm(U @ B @ V.T, 2)
        slow += [(w, U @ B @ V.T), (-w, np.conj(U @ B @ V.T))]
    # high-frequency disturbance: 2 pairs + a Nyquist tone, amplitude alpha ~ 2*sigma_r
    high = []
    for w, s in ((0.7 * np.pi, 0.8), (0.9 * np.pi, 0.6)):
        D = (rng.standard_normal((m_, n_)) + 1j * rng.standard_normal((m_, n_)))
        D *= s / np.linalg.norm(D, 2)
        high += [(w, D), (-w, np.conj(D))]
    Dpi = rng.standard_normal((m_, n_))
    Dpi *= 0.5 / np.linalg.norm(Dpi, 2)
    high.append((np.pi, Dpi))
    a = sum(np.linalg.norm(C, 2) for _, C in slow)
    alpha = sum(np.linalg.norm(D, 2) for _, D in high)
    beta, T = 0.9, 400
    rho_high = abs(H(beta, wc))
    eps_low = abs(H(beta, w0) - 1)
    print(f"    a={a:.3f}, alpha={alpha:.3f}, sigma_r={sr}, rho_high={rho_high:.4f}, "
          f"eps_low={eps_low:.4f}")

    # per-tone lemma at a single tone (machine precision)
    w_test, Vmat = 0.7 * np.pi, rng.standard_normal((4, 3))
    mbuf = np.zeros((4, 3), dtype=complex)
    worst = 0.0
    for t in range(1, 60):
        mbuf = beta * mbuf + (1 - beta) * np.exp(1j * w_test * t) * Vmat
        ref = H(beta, w_test) * (np.exp(1j * w_test * t) - beta**t) * Vmat
        worst = max(worst, np.abs(mbuf - ref).max())
    check("per-tone lemma exact", worst < 1e-12, f"max dev {worst:.1e}")
    # Nyquist reduction: H(pi)((-1)^t - beta^t) == eps_t of T5
    t = np.arange(1, 50)
    lhs = H(beta, np.pi).real * ((-1.0) ** t - beta**t)
    eps_t = (-1.0) ** t * (1 - beta) * (1 - (-beta) ** t) / (1 + beta)
    check("Nyquist tone reduces to T5's eps_t", np.abs(lhs - eps_t).max() < 1e-14)

    # run the EMA on the full real stream; verify decomposition + bounds each step
    Mbuf = np.zeros((m_, n_))
    ok_dec, ok_a, ok_b, ok_tail, ok_wed, wed_tight, ok_sdr, sdr_tight = \
        True, True, True, True, True, [], True, []
    for t in range(1, T + 1):
        G = S + sum((np.exp(1j * w * t) * C for w, C in slow + high)).real
        Mbuf = beta * Mbuf + (1 - beta) * G
        Xt = (1 - beta**t) * S + sum(
            (H(beta, w) * (np.exp(1j * w * t) - beta**t) * C for w, C in slow)).real
        At = sum((H(beta, w) * (np.exp(1j * w * t) - beta**t) * D for w, D in high)).real
        ok_dec &= np.abs(Mbuf - (Xt + At)).max() < 1e-10
        bd_a = rho_high * (1 + beta**t) * alpha
        ok_a &= np.linalg.norm(At, 2) <= bd_a + 1e-12
        St = S + sum((np.exp(1j * w * t) * C for w, C in slow)).real
        bd_b = beta**t * np.linalg.norm(S, 2) + (eps_low + beta**t) * a
        ok_b &= np.linalg.norm(Xt - St, 2) <= bd_b + 1e-12
        sv = np.linalg.svd(Mbuf, compute_uv=False)
        ok_tail &= sv[r] <= bd_a + 1e-12
        delta = (1 - beta**t) * sr - (1 + beta**t) * a - bd_a
        if delta > 0:
            Usvd, _, Vtsvd = np.linalg.svd(Mbuf)
            Um, Vm = Usvd[:, :r], Vtsvd[:r].T
            sin_meas = max(
                np.sqrt(max(0.0, 1 - np.linalg.svd(Um.T @ U)[1].min() ** 2)),
                np.sqrt(max(0.0, 1 - np.linalg.svd(Vm.T @ V)[1].min() ** 2)))
            bound = bd_a / delta
            ok_wed &= sin_meas <= bound + 1e-12
            if t > T // 2:
                wed_tight.append(bound / max(sin_meas, 1e-300))
            gain = (np.linalg.svd(Xt, compute_uv=False)[r - 1]
                    / np.linalg.norm(At, 2)) / (sr / alpha)
            gain_bd = ((1 - beta**t) - (1 + beta**t) * a / sr) / (rho_high * (1 + beta**t))
            ok_sdr &= gain >= gain_bd - 1e-12
            if t > T // 2:
                sdr_tight.append(gain / gain_bd)
    check("exact buffer decomposition M = X + A~", ok_dec)
    check("(a) disturbance bound holds each step", ok_a)
    check("(b) fidelity bound holds each step", ok_b)
    check("tail bound sigma_{r+1}(M) <= rho_high(1+b^t) alpha", ok_tail)
    check("(c) Wedin bound holds for BOTH subspaces where delta_t > 0", ok_wed,
          f"median late-half tightness x{np.median(wed_tight):.2f}")
    check("SDR gain >= corollary bound", ok_sdr,
          f"median late-half gain/bound x{np.median(sdr_tight):.2f}")
    check("filtering is material: rho_high*alpha < sigma_r < alpha",
          rho_high * alpha < sr < alpha)


# --------------------------------------------------------------------------- Check D
def check_D():
    print("Check D — B1 boundary weight")
    beta = 0.9

    def boundary(g, beta):
        m = np.zeros(g.shape[1:])
        for t in range(len(g)):
            m = beta * m + (1 - beta) * g[t]
        return beta / (1 - beta) * m

    # (i) constant stream: exact ratio beta(1-beta^T)/((1-beta) T)
    T = 500
    g = np.ones((T, 1))
    B = boundary(g, beta)
    ratio = np.linalg.norm(B) / np.linalg.norm(g.sum(axis=0))
    exact = beta * (1 - beta**T) / ((1 - beta) * T)
    check("(i) constant stream exact ratio", abs(ratio - exact) < 1e-12,
          f"{ratio:.5f} = beta(1-b^T)/((1-b)T)")
    # (ii) stationary stream (T2's hill-gradient stream): E||B||^2 / E|g^(w)|^2 ~ 1/T
    etalam, sig = 1.8, 1.0
    a_ar = 1 - etalam
    w_bin = np.pi / 2
    Ts = [250, 500, 1000, 2000, 4000]
    ratios = []
    rng = np.random.default_rng(5)
    for T in Ts:
        b2s, g2s = [], []
        for _ in range(200):
            xi = sig * rng.standard_normal(T + 200)
            y = np.zeros(T + 201)
            for t in range(T + 200):
                y[t + 1] = a_ar * y[t] - 0.18 * xi[t]  # eta=0.18, lam=10 -> g = 10 y + xi
            gs = 10.0 * y[200:-1] + xi[200:]
            ghat = np.sum(gs * np.exp(-1j * w_bin * np.arange(1, T + 1)))
            b2s.append(np.abs(boundary(gs[:, None], beta)) ** 2)
            g2s.append(np.abs(ghat) ** 2)
        ratios.append(float(np.mean(b2s) / np.mean(g2s)))
    slope = np.polyfit(np.log(Ts), np.log(ratios), 1)[0]
    check("(ii) stationary boundary weight decays ~ 1/T",
          abs(slope + 1.0) < 0.25, f"log-log slope {slope:.2f}; ratios "
          + " ".join(f"{r:.2e}" for r in ratios))
    # (ii') the denominator asymptotic itself: E|g^(w)|^2 / T -> S(w) (T2 closed form).
    # Vectorized over 4000 realizations (single-bin periodograms are ~exponential, so the
    # mean needs many draws); ghat accumulated on the fly.
    n_r, T = 4000, 4000
    rng = np.random.default_rng(6)
    y = np.zeros(n_r)
    for _ in range(200):  # burn-in to stationarity
        y = a_ar * y - 0.18 * sig * rng.standard_normal(n_r)
    ghat = np.zeros(n_r, dtype=complex)
    for t in range(1, T + 1):
        xi = sig * rng.standard_normal(n_r)
        g = 10.0 * y + xi
        ghat += g * np.exp(-1j * w_bin * t)
        y = a_ar * y - 0.18 * xi
    g2_per_T = float(np.mean(np.abs(ghat) ** 2) / T)
    S_w = sig**2 * abs(1 - np.exp(-1j * w_bin)) ** 2 / abs(1 - a_ar * np.exp(-1j * w_bin)) ** 2
    check("(ii') E|g^(w)|^2 / T -> S(w)", abs(g2_per_T / S_w - 1) < 0.05,
          f"measured {g2_per_T:.4f} vs S(pi/2)={S_w:.4f}, T={T}, {n_r} realizations")
    # (iii) failure mode: window ~ T_eff has O(1) DC-relative weight
    Tshort = int((1 + beta) / (1 - beta))  # = T_eff
    g = np.ones((Tshort, 1))
    r_short = np.linalg.norm(boundary(g, beta)) / np.linalg.norm(g.sum(axis=0))
    check("(iii) window ~ T_eff: weight is O(1)", r_short > 0.2, f"ratio {r_short:.2f}")


if __name__ == "__main__":
    check_A()
    check_B()
    check_C()
    check_D()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("All checks pass.")
