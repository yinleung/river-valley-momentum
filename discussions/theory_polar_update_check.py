"""Checks for theory_polar_update.md: T3 update-level theorems for the polar step.

A-checks: T3a (Li polar-perturbation bound; applied Nyquist-model bound; tightness).
B-checks: T3b (scalar NS response h and L_h; matrix Lipschitz via dilation; normalization
lemma; composed deployed-operator theorem; the reviewer's rank-one construction as the
necessity example).

Run:  .venv/bin/python discussions/theory_polar_update_check.py   (repo root; ~1-2 min)
Every [check] line must end PASS.
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)

A_NS, B_NS, C_NS, K_NS, EPS_NS = 3.4445, -4.7750, 2.0315, 5, 1e-7


def ok(name: str, cond: bool, detail: str = "") -> None:
    print(f"[check] {name}: {'PASS' if cond else 'FAIL'}  {detail}")
    assert cond, name


# ------------------------------------------------------------------ primitives
def polar(M: np.ndarray) -> np.ndarray:
    """Unitary polar factor (full column rank assumed), d1 >= d2."""
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def polar_partial(M: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Partial polar: sum of u_i v_i^T over singular values > tol (rank-deficient case)."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > tol
    return (U[:, keep]) @ (Vt[keep, :])


def ns_step(X: np.ndarray) -> np.ndarray:
    XtX = X.T @ X
    return A_NS * X + B_NS * X @ XtX + C_NS * X @ (XtX @ XtX)


def ns_k(X: np.ndarray, k: int = K_NS) -> np.ndarray:
    for _ in range(k):
        X = ns_step(X)
    return X


def deployed(M: np.ndarray) -> np.ndarray:
    return ns_k(M / (np.linalg.norm(M) + EPS_NS))


def h_scalar(s, k: int = K_NS):
    s = np.asarray(s, dtype=float)
    for _ in range(k):
        s = A_NS * s + B_NS * s**3 + C_NS * s**5
    return s


def h_prime(s, k: int = K_NS):
    """h'(s) by the chain rule: prod_j q'(s_j)."""
    s = np.asarray(s, dtype=float)
    out = np.ones_like(s)
    for _ in range(k):
        out = out * (A_NS + 3 * B_NS * s**2 + 5 * C_NS * s**4)
        s = A_NS * s + B_NS * s**3 + C_NS * s**5
    return out


def eps_t(beta: float, t: int) -> float:
    return (-1) ** t * (1 - beta) * (1 - (-beta) ** t) / (1 + beta)


def rand_mat(d1, d2, smin=None):
    M = rng.standard_normal((d1, d2))
    if smin is not None:  # rescale spectrum so sigma_min = smin, sigma_max ~ 1..2
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        s = np.linspace(1.7, smin, len(s))
        M = U @ np.diag(s) @ Vt
    return M


# ------------------------------------------------------- A1: Li perturbation bound
def check_a1() -> None:
    worst = 0.0
    for _ in range(300):
        d1, d2 = (8, 8) if rng.random() < 0.5 else (9, 5)
        A = rand_mat(d1, d2, smin=10 ** rng.uniform(-3, 0))
        B = A + rng.uniform(0.01, 2.0) * rng.standard_normal((d1, d2)) * 0.3
        if np.linalg.svd(B, compute_uv=False)[-1] < 1e-8:
            continue
        lhs = np.linalg.norm(polar(A) - polar(B))
        rhs = 2 / (np.linalg.svd(A, compute_uv=False)[-1]
                   + np.linalg.svd(B, compute_uv=False)[-1]) * np.linalg.norm(A - B)
        worst = max(worst, lhs / rhs)
    ok("A1a Li bound holds (300 random square+rectangular, incl. near-singular)",
       worst <= 1 + 1e-9, f"worst lhs/rhs {worst:.4f}")
    sig = 0.7  # tight family: A=(s,0)^T, B=(0,s)^T -> both sides sqrt(2)
    A = np.array([[sig], [0.0]])
    B = np.array([[0.0], [sig]])
    ratio = np.linalg.norm(polar(A) - polar(B)) / (2 / (2 * sig) * np.linalg.norm(A - B))
    ok("A1b Li constant attained on the rectangular tight family", ratio > 0.999,
       f"ratio {ratio:.6f}")


# ------------------------------------- A2: update-level bound on the Nyquist model
def check_a2() -> None:
    d1, d2 = 12, 7
    S = rand_mat(d1, d2, smin=1.0)
    A = rng.standard_normal((d1, d2))
    A *= 0.6 / np.linalg.svd(A, compute_uv=False)[0]  # ||A||_2 = 0.6 < sigma_min(S)
    smin = np.linalg.svd(S, compute_uv=False)[-1]
    worst, pre_end, ponly_min = 0.0, {}, np.inf
    for beta in (0.5, 0.9, 0.99):
        for t in range(1, 400):
            e = eps_t(beta, t)
            M = (1 - beta**t) * S + e * A
            lhs = np.linalg.norm(polar(M) - polar(S))
            den = (np.linalg.svd(M, compute_uv=False)[-1] + (1 - beta**t) * smin)
            rhs = 2 * abs(e) * np.linalg.norm(A) / den
            worst = max(worst, lhs / rhs if rhs > 0 else 0)
            G = S + (-1) ** t * A
            ponly_min = min(ponly_min, np.linalg.norm(polar(G) - polar(S)))
        teff = (1 + beta) / (1 - beta)
        pre_end[beta] = lhs * teff  # ~ ||A||_F/smin modulo O(1/T_eff)
    ok("A2a update-level bound holds on the Nyquist model (all beta, t)",
       worst <= 1 + 1e-9, f"worst lhs/bound {worst:.4f}")
    ok("A2b pre-polar error dies at 1/T_eff (T_eff * error bounded), polar-only does not",
       max(pre_end.values()) <= 2 * np.linalg.norm(A) / smin and ponly_min > 0.05,
       f"T_eff*err: {max(pre_end.values()):.3f} <= {2 * np.linalg.norm(A) / smin:.3f}; "
       f"polar-only floor {ponly_min:.3f}")


# -------------------------------------------------- A3: tightness of the T3a bound
def check_a3() -> None:
    ratios = []
    for _ in range(200):
        S = rand_mat(10, 6, smin=1.0)
        A = rng.standard_normal((10, 6))
        A *= 0.5 / np.linalg.svd(A, compute_uv=False)[0]
        beta, t = 0.9, 100
        e = eps_t(beta, t)
        M = (1 - beta**t) * S + e * A
        lhs = np.linalg.norm(polar(M) - polar(S))
        den = np.linalg.svd(M, compute_uv=False)[-1] + (1 - beta**t) * np.linalg.svd(
            S, compute_uv=False)[-1]
        ratios.append((2 * abs(e) * np.linalg.norm(A) / den) / lhs)
    med = float(np.median(ratios))
    ok("A3a bound within an order of magnitude of measured (random instances)",
       1 <= min(ratios) and med <= 10, f"median bound/measured {med:.2f}, "
       f"range [{min(ratios):.2f}, {max(ratios):.2f}]")
    # rotational worst case: A generates a rotation of the polar factor
    d = 6
    S = np.eye(d)
    A = np.zeros((d, d))
    A[0, 1], A[1, 0] = 0.5, -0.5
    beta, t = 0.9, 200
    e = eps_t(beta, t)
    M = (1 - beta**t) * S + e * A
    lhs = np.linalg.norm(polar(M) - polar(S))
    den = np.linalg.svd(M, compute_uv=False)[-1] + (1 - beta**t)
    ratio = (2 * abs(e) * np.linalg.norm(A) / den) / lhs
    ok("A3b bound tight to a small constant on the rotational family", ratio < 1.2,
       f"bound/measured {ratio:.4f}")


# ----------------------------------------- B1: scalar response h and its derivative
def check_b1() -> None:
    ok("B1a h(0)=0 and h'(0) = a^k", abs(h_scalar(0.0)) == 0
       and abs(h_prime(0.0) - A_NS**K_NS) < 1e-9 * A_NS**K_NS,
       f"a^k = {A_NS**K_NS:.2f}")
    ok("B1b q is not monotone on [0,1]: q'(1) = a+3b+5c < 0",
       A_NS + 3 * B_NS + 5 * C_NS < 0, f"q'(1) = {A_NS + 3 * B_NS + 5 * C_NS:.4f}")
    s = np.linspace(0, 1, 2_000_001)
    hp = h_prime(s)
    L_h = float(np.abs(hp).max())
    ok("B1c L_h = sup|h'| on [0,1] attained at 0 and equals a^k",
       abs(L_h - A_NS**K_NS) < 1e-6 * A_NS**K_NS, f"L_h = {L_h:.4f}")
    ok("B1d the composed response h itself is nonmonotone (h' < 0 somewhere on [0,1])",
       float(hp.min()) < 0, f"min h' = {float(hp.min()):.2f} at s = "
       f"{float(s[int(np.argmin(hp))]):.4f}")
    mid = float(np.abs(h_prime(np.linspace(0.3, 1, 700_001))).max())
    print(f"        (sup|h'| on [0.3,1] = {mid:.2f} — the constant lives at sigma ~ 0)")


# ------------------------------------------- B2: matrix Lipschitz (dilation transfer)
def check_b2() -> None:
    L_h = A_NS**K_NS
    worst = 0.0
    for _ in range(200):
        d1, d2 = (7, 7) if rng.random() < 0.5 else (8, 5)
        X = rng.standard_normal((d1, d2))
        X *= rng.uniform(0.05, 1.0) / np.linalg.svd(X, compute_uv=False)[0]
        Y = X + rng.uniform(1e-4, 0.5) * rng.standard_normal((d1, d2))
        s0 = np.linalg.svd(Y, compute_uv=False)[0]
        if s0 > 1:
            Y /= s0 * (1 + 1e-12)
        lhs = np.linalg.norm(ns_k(X) - ns_k(Y))
        worst = max(worst, lhs / (L_h * np.linalg.norm(X - Y)))
    ok("B2a NS_k is L_h-Lipschitz in Frobenius on the unit operator-norm ball",
       worst <= 1 + 1e-9, f"worst ratio {worst:.4f}")
    # adversarial: rank-one at tiny sigma approaches the constant
    d = 5
    u, v = np.zeros((d, 1)), np.zeros((d, 1))
    u[0], v[0] = 1, 1
    E = u @ v.T
    lo = np.linalg.norm(ns_k(1e-4 * E) - ns_k(2e-4 * E)) / (L_h * 1e-4)
    ok("B2b constant approached by small-singular-value pairs", lo > 0.95,
       f"achieved fraction {lo:.4f}")


# ------------------------------------------------------- B3: normalization lemma
def check_b3() -> None:
    worst, tight = 0.0, 0.0
    for _ in range(500):
        d1, d2 = 6, 4
        M = rng.standard_normal((d1, d2)) * rng.uniform(0.1, 3)
        X = rng.standard_normal((d1, d2)) * rng.uniform(0.1, 3)
        r_low = min(np.linalg.norm(M), np.linalg.norm(X))
        nu = lambda Z: Z / (np.linalg.norm(Z) + EPS_NS)
        ratio = np.linalg.norm(nu(M) - nu(X)) / (
            2 / (r_low + EPS_NS) * np.linalg.norm(M - X))
        worst = max(worst, ratio)
        tight = max(tight, ratio)
    E = rng.standard_normal((6, 4))
    E /= np.linalg.norm(E)
    r = 0.3
    nu = lambda Z: Z / (np.linalg.norm(Z) + EPS_NS)
    anti = np.linalg.norm(nu(r * E) - nu(-r * E)) / (2 / (r + EPS_NS) * 2 * r)
    ok("B3 normalization bound holds; antiparallel pair attains half the constant",
       worst <= 1 + 1e-9 and abs(anti - 0.5) < 1e-3,
       f"worst ratio {worst:.4f}; antiparallel {anti:.4f}")


# ------------------------------ B4: composed deployed-operator theorem (T3b proper)
def check_b4() -> None:
    L_h = A_NS**K_NS
    worst = 0.0
    for full_rank in (True, False):
        for trial in range(30):
            d1, d2 = 9, 6
            if full_rank:
                S = rand_mat(d1, d2, smin=rng.uniform(0.3, 1.0))
            else:  # rank-one signal: T3b still applies (r_low from ||.||_F)
                S = np.outer(rng.standard_normal(d1), rng.standard_normal(d2))
                S /= np.linalg.norm(S)
            A = rng.standard_normal((d1, d2))
            A *= rng.uniform(0.1, 0.5) * np.linalg.norm(S) / np.linalg.norm(A)
            for beta in (0.5, 0.9, 0.99):
                for t in (1, 3, 10, 50, 200):
                    e = eps_t(beta, t)
                    M = (1 - beta**t) * S + e * A
                    X = (1 - beta**t) * S
                    r_low = (1 - beta**t) * np.linalg.norm(S) - abs(e) * np.linalg.norm(A)
                    # the bound is analytic in r_low; tiny floors only inflate it, so we
                    # test moderate annuli, not the r_low -> 0 boundary
                    if r_low <= 0.02:
                        continue
                    lhs = np.linalg.norm(deployed(M) - deployed(X))
                    rhs = 2 * L_h / (r_low + EPS_NS) * abs(e) * np.linalg.norm(A)
                    worst = max(worst, lhs / rhs)
    ok("B4 deployed-operator bound ||D(M_t)-D(X_t)|| <= L_NS |eps_t| ||A||_F "
       "(full-rank and rank-one signals)", worst <= 1 + 1e-9, f"worst ratio {worst:.4f}")


# ------------------------- B5: the reviewer's construction = necessity, not a bug
def check_b5() -> None:
    d = 6
    u, up = np.zeros(d), np.zeros(d)
    v, vp = np.zeros(d), np.zeros(d)
    u[0] = v[1] = up[2] = vp[3] = 1.0
    sig_s, alpha = 1.0, 2.0
    S = sig_s * np.outer(u, v)
    A = alpha * np.outer(up, vp)
    t = 400
    exact_err, depl_err = [], []
    for beta in (0.5, 0.9, 0.99):
        e = eps_t(beta, t)
        M = (1 - beta**t) * S + e * A
        exact_err.append(np.linalg.norm(polar_partial(M) - polar_partial(S)))
        depl_err.append(np.linalg.norm(deployed(M) - deployed((1 - beta**t) * S)))
    ok("B5a exact polar: error is exactly 1 at every beta (no amplitude transfer)",
       max(abs(x - 1) for x in exact_err) < 1e-9,
       f"errors {[f'{x:.4f}' for x in exact_err]}")
    ok("B5b deployed operator: error decreases in beta (graceful, on the T3b bound)",
       depl_err[0] > depl_err[1] > depl_err[2],
       f"errors {[f'{x:.4f}' for x in depl_err]}")
    # B5c: the sub-knee linear zone — below amplitude ~1/L_NS the bound is near-tight
    L_h = A_NS**K_NS
    beta, t = 0.9, 200
    e = abs(eps_t(beta, t))
    rows = []
    for alpha in (0.01, 0.002):
        M = (1 - beta**t) * S + eps_t(beta, t) * (alpha / 2.0) * A  # ||A||_F = alpha
        X = (1 - beta**t) * S
        err = np.linalg.norm(deployed(M) - deployed(X))
        r_low = (1 - beta**t) * np.linalg.norm(S) - e * alpha
        bound = 2 * L_h / (r_low + EPS_NS) * e * alpha
        rows.append((alpha, err, bound, err / bound))
    ok("B5c sub-knee amplitudes: deployed error decays ~linearly, bound tight to <4x",
       all(0.25 < r[3] <= 1 for r in rows) and rows[1][1] < rows[0][1] < 0.5,
       "; ".join(f"alpha={r[0]}: err {r[1]:.4f} vs bound {r[2]:.4f}" for r in rows))


# ---------------------------------------- B6: singular-frame commutation of NS_k
def check_b6() -> None:
    worst = 0.0
    for _ in range(50):
        X = rng.standard_normal((8, 5))
        X /= np.linalg.svd(X, compute_uv=False)[0] * rng.uniform(1.0, 3.0)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        direct = ns_k(X)
        via_h = U @ np.diag(h_scalar(s)) @ Vt
        worst = max(worst, np.linalg.norm(direct - via_h) / max(np.linalg.norm(direct), 1e-12))
    ok("B6 NS_k(X) == U h(Sigma) V^T in the input's own frame", worst < 1e-10,
       f"worst rel err {worst:.1e}")


if __name__ == "__main__":
    print("== A: T3a exact-polar update theorems ==")
    check_a1()
    check_a2()
    check_a3()
    print("== B: T3b deployed Newton-Schulz operator ==")
    check_b1()
    check_b2()
    check_b3()
    check_b4()
    check_b5()
    check_b6()
    print("all checks passed")
