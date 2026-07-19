"""Checks for theory_cv2_coupled.md: T1 repair of Prop 5(b) — the linear-floor curved
valley under EMA momentum is exactly two straight-valley closed loops over the Hessian
eigenframe.

A-checks are symbolic (sympy): modal char-poly identity, exact-rational Lyapunov vs the
modal closed form, beta=0 agreement with the 2-D pre-check, the mu->0 limit and its
first-order coefficient, eigen-identities, and the exact stability boundary.
B-checks are Monte-Carlo (numpy) on the real (x,y) loop, including the reviewer's beta=0
audit cell, a discrimination cell where the frozen formula is off by >2x, and the
tilted-profile tier.

Run:  .venv/bin/python discussions/theory_cv2_coupled_check.py   (repo root; ~2-4 min)
Every [check] line must end PASS.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

rng = np.random.default_rng(0)


def ok(name: str, cond: bool, detail: str = "") -> None:
    print(f"[check] {name}: {'PASS' if cond else 'FAIL'}  {detail}")
    assert cond, name


# ------------------------------------------------------------------ shared constructions
ETA, BET, MU, LAM, C, S2 = sp.symbols("eta beta mu lambda c sigma2", positive=True)
LLOC = LAM * (1 + C**2)


def a4_q4(eta, bet, mu, lam, c, s2):
    """4-D transition matrix and noise injection for state (x, d, m^x, m^d)."""
    lloc = lam * (1 + c**2)
    ob = 1 - bet
    A = sp.Matrix(
        [
            [1 - eta * ob * mu, eta * ob * c * lam, -eta * bet, 0],
            [eta * ob * c * mu, 1 - eta * ob * lloc, 0, -eta * bet],
            [ob * mu, -ob * c * lam, bet, 0],
            [-ob * c * mu, ob * lloc, 0, bet],
        ]
    )
    N = sp.Matrix([[-eta * ob, 0], [0, -eta * ob], [ob, 0], [0, ob]])
    Cn = s2 * sp.Matrix([[1, -c], [-c, 1 + c**2]])  # Cov(xi^x, xi^d), xi^d = xi^y - c xi^x
    return A, N * Cn * N.T


def nu12(mu, lam, c, sqrt=sp.sqrt):
    lloc = lam * (1 + c**2)
    disc = (lloc - mu) ** 2 + 4 * c**2 * lam * mu
    n1 = ((mu + lloc) + sqrt(disc)) / 2
    n2 = ((mu + lloc) - sqrt(disc)) / 2
    return n1, n2


def modal_var(eta, bet, mu, lam, c, s2, sqrt=sp.sqrt):
    """Closed form: Var(d_inf) = sum_i b_i^2 * eta*s2 / (nu_i (2 - eta nu_i / T_eff)).

    Weight quotients are defined for c != 0 (at c = 0, b2^2 is 0/0 with limit 0;
    see check A5e for the continuous extension)."""
    lloc = lam * (1 + c**2)
    n1, n2 = nu12(mu, lam, c, sqrt)
    teff = (1 + bet) / (1 - bet)
    b1s = (lloc - n2) ** 2 / ((lam - n2) * (n1 - n2))
    b2s = (n1 - lloc) ** 2 / ((n1 - lam) * (n1 - n2))
    v = lambda n: eta * s2 / (n * (2 - eta * n / teff))
    return b1s * v(n1) + b2s * v(n2)


def frozen_var(eta, bet, mu, lam, c, s2):
    lloc = lam * (1 + c**2)
    teff = (1 + bet) / (1 - bet)
    return eta * s2 / (lam * (2 - eta * lloc / teff))


# ---------------------------------------------------------------- A1: modal char poly
def check_a1() -> None:
    z = sp.Symbol("z")
    A, _ = a4_q4(ETA, BET, MU, LAM, C, S2)
    cp = A.charpoly(z).as_expr()
    # product of the two modal AR(2) polynomials, written via the symmetric functions
    # e1 = nu1+nu2 = mu+lloc, e2 = nu1*nu2 = mu*lam (radical-free)
    k = ETA * (1 - BET)
    D = z**2 - (1 + BET) * z + BET
    prod = sp.expand(D**2 + D * z * k * (MU + LLOC) + z**2 * k**2 * MU * LAM)
    ok("A1 charpoly(A4) = p_nu1 * p_nu2", sp.simplify(sp.expand(cp) - prod) == 0)


# ------------------------------------- A2: exact-rational Lyapunov vs modal closed form
def lyap_var_d(eta, bet, mu, lam, c, s2):
    """Exact-rational stationary Var(d) by solving P = A P A^T + Q (10 unknowns)."""
    A, Q = a4_q4(eta, bet, mu, lam, c, s2)
    ps = sp.symbols("p0:10")
    idx = [(i, j) for i in range(4) for j in range(i, 4)]
    P = sp.zeros(4, 4)
    for k_, (i, j) in enumerate(idx):
        P[i, j] = P[j, i] = ps[k_]
    E = A * P * A.T + Q - P
    eqs = [E[i, j] for (i, j) in idx]
    sol = sp.solve(eqs, list(ps), dict=True)[0]
    return P.subs(sol)[1, 1]


def check_a2() -> None:
    cells = []
    r = lambda a, b: sp.Rational(a, b)
    # hand-picked: reviewer's cell (beta=0), momentum cells, extreme-coupling cell
    cells += [
        (r(1, 10), sp.Integer(0), r(1, 10), 10, r(1, 2), 4),
        (r(1, 10), r(9, 10), r(1, 10), 10, r(1, 2), 4),
        (r(7, 100), r(3, 10), 4, 10, r(6, 5), 1),
        (r(3, 20), r(19, 20), r(3, 2), 10, 1, 2),
    ]
    for _ in range(8):  # random stable rational cells
        while True:
            eta = sp.Rational(int(rng.integers(1, 12)), 100)
            bet = sp.Rational(int(rng.integers(0, 10)), 10)
            mu = sp.Rational(int(rng.integers(1, 40)), 10)
            lam = sp.Rational(int(rng.integers(5, 15)), 1)
            c = sp.Rational(int(rng.integers(1, 15)), 10)
            n1, _ = nu12(mu, lam, c)
            if mu < lam and eta * n1 < 2 * (1 + bet) / (1 - bet):
                cells.append((eta, bet, mu, lam, c, 1))
                break
    worst = 0.0
    for cell in cells:
        lv = sp.N(lyap_var_d(*cell), 80)  # exact rational, evaluated once
        mv = sp.N(modal_var(*cell), 80)  # two positive terms: no cancellation inside
        d = abs(float((lv - mv) / lv))
        worst = max(worst, d)
    ok("A2 modal closed form == exact 4-D Lyapunov", worst < 1e-60,
       f"{len(cells)} rational cells, worst rel diff {worst:.1e}")


# ------------------------- A3: beta=0 agrees with the 2-D Lyapunov AND the reviewer
def check_a3() -> None:
    # State (d, x). Correct noise: d gets -eta*xi^d, x gets -eta*xi^x, and
    # Cov(xi^d, xi^x) = -c sigma^2 — the off-diagonal is NEGATIVE. The 2026-07-19
    # pre-check (plan_v5_t1_precheck.py) used +c here; its conclusion that the
    # reviewer's beta=0 formula matches no natural noise convention is thereby
    # superseded: under per-coordinate iid noise the reviewer's formula is exact.
    lloc = LLOC
    A2d = sp.Matrix([[1 - ETA * lloc, C * ETA * MU], [ETA * C * LAM, 1 - ETA * MU]])
    Q2d = ETA**2 * S2 * sp.Matrix([[1 + C**2, -C], [-C, 1]])
    p11, p12, p22 = sp.symbols("P11 P12 P22")
    P = sp.Matrix([[p11, p12], [p12, p22]])
    E = A2d * P * A2d.T + Q2d - P
    sol = sp.solve([E[0, 0], E[0, 1], E[1, 1]], [p11, p12, p22], dict=True)[0]
    pre = sp.simplify(sol[p11])
    mv0 = modal_var(ETA, sp.Integer(0), MU, LAM, C, S2)
    ok("A3a beta=0 modal form == 2-D Lyapunov (symbolic)",
       sp.simplify(sp.radsimp(mv0 - pre)) == 0)
    reviewer = ETA * S2 * (2 - ETA * MU) / (
        LAM * (4 - 2 * ETA * LAM * (1 + C**2) - 2 * ETA * MU + ETA**2 * LAM * MU))
    ok("A3b beta=0 exact == the reviewer's displayed formula (their audit vindicated)",
       sp.simplify(pre - reviewer) == 0)


# ----------------------------------- A4: mu->0 limit is the paper formula; O(mu) coeff
def check_a4() -> None:
    mv = modal_var(ETA, BET, MU, LAM, C, S2)
    lim = sp.simplify(sp.limit(mv, MU, 0))
    paper = sp.simplify(frozen_var(ETA, BET, 0, LAM, C, S2))
    ok("A4a mu->0 limit == frozen-coefficient formula", sp.simplify(lim - paper) == 0)
    K = sp.simplify(sp.series(mv, MU, 0, 2).removeO().coeff(MU, 1))
    teff = (1 + BET) / (1 - BET)
    K_compact = C**2 * ETA**3 * S2 / (2 * (2 * teff - ETA * LLOC) ** 2)
    ok("A4b first-order-in-mu coefficient K = c^2 eta^3 s2 / (2 (2 T_eff - eta lloc)^2)",
       sp.simplify(K - K_compact) == 0)


# ------------------------------------------------------------- A5: eigen-identities
def check_a5() -> None:
    n1, n2 = nu12(MU, LAM, C)
    lloc = LLOC
    ok("A5a (nu1-lam)(lam-nu2) = c^2 lam^2",
       sp.simplify((n1 - LAM) * (LAM - n2) - C**2 * LAM**2) == 0)
    ok("A5b (nu1-lloc)(lloc-nu2) = c^2 lam mu",
       sp.simplify((n1 - lloc) * (lloc - n2) - C**2 * LAM * MU) == 0)
    b1s = (lloc - n2) ** 2 / ((LAM - n2) * (n1 - n2))
    b2s = (n1 - lloc) ** 2 / ((n1 - LAM) * (n1 - n2))
    ok("A5c b1^2 + b2^2 = 1 + c^2", sp.simplify(b1s + b2s - 1 - C**2) == 0)
    gap = sp.simplify(n1 - lloc)
    val = float(gap.subs({MU: sp.Rational(1, 7), LAM: 9, C: sp.Rational(2, 3)}))
    lim = sp.limit(gap / MU, MU, 0)
    ok("A5d nu1 >= lloc (frozen threshold necessary, not sufficient)",
       val > 0 and sp.simplify(lim - C**2 / (1 + C**2)) == 0,
       f"nu1-lloc at a cell: {val:.4f}; leading order c^2/(1+c^2) mu")
    dl = sp.Symbol("delta", positive=True)  # lam = mu + delta encodes mu < lam
    sub = {LAM: MU + dl}
    b1s = ((lloc - n2) ** 2 / ((LAM - n2) * (n1 - n2))).subs(sub)
    b2s = ((n1 - lloc) ** 2 / ((n1 - LAM) * (n1 - n2))).subs(sub)
    lim_modal = sp.limit(modal_var(ETA, BET, MU, LAM, C, S2).subs(sub), C, 0)
    tgt = frozen_var(ETA, BET, MU, LAM, 0, S2).subs(sub)
    ok("A5e c->0 continuous extension: b1^2 -> 1, b2^2 -> 0, modal -> straight formula",
       sp.limit(b1s, C, 0) == 1 and sp.limit(b2s, C, 0) == 0
       and sp.simplify(lim_modal - tgt) == 0)


# ------------------------------------------- A6/B2: the exact stability boundary
def check_boundary() -> None:
    mu, lam, c, bet = 4.0, 10.0, 1.2, 0.5
    n1, _ = nu12(mu, lam, c, sqrt=np.sqrt)
    teff = (1 + bet) / (1 - bet)
    eta_star = 2 * teff / n1
    eta_froz = 2 * teff / (lam * (1 + c**2))

    def rho(eta):
        A, _ = a4_q4(eta, bet, mu, lam, c, 1)
        return max(abs(np.linalg.eigvals(np.array(A, dtype=float))))

    ok("B2a spectral radius crosses 1 exactly at eta = 2 T_eff / nu1",
       rho(eta_star * (1 - 1e-6)) < 1 < rho(eta_star * (1 + 1e-6)),
       f"eta* = {eta_star:.6f}")
    ok("B2b frozen threshold is not sufficient: unstable strictly below 2 T_eff / lloc",
       eta_star < eta_froz and rho(0.5 * (eta_star + eta_froz)) > 1,
       f"eta*_frozen/eta* = {eta_froz / eta_star:.3f}")


# ------------------------------------------------------------------ B1/B3: Monte Carlo
def simulate(eta, bet, mu, lam, c, sigma, T, seeds, tilt=None, seed0=0):
    """Full (x,y) EMA-momentum loop; returns tail mean and Var of d = y - c x.

    tilt=None: quadratic profile phi = mu/2 x^2. tilt=G: phi' == G (tilted profile)."""
    r = np.random.default_rng(seed0)
    x = np.zeros(seeds)
    y = np.zeros(seeds)
    mx = np.zeros(seeds)
    my = np.zeros(seeds)
    t0 = T // 2
    acc_d, acc_d2, n = 0.0, 0.0, 0
    for t in range(T):
        d = y - c * x
        gx = (tilt if tilt is not None else mu * x) - c * lam * d + sigma * r.standard_normal(seeds)
        gy = lam * d + sigma * r.standard_normal(seeds)
        mx = bet * mx + (1 - bet) * gx
        my = bet * my + (1 - bet) * gy
        x = x - eta * mx
        y = y - eta * my
        if t >= t0:
            d = y - c * x
            acc_d += d.sum()
            acc_d2 += (d * d).sum()
            n += d.size
    mean = acc_d / n
    return mean, acc_d2 / n - mean**2


def check_b1() -> None:
    f = lambda expr: float(expr)
    # (i) the reviewer's beta=0 audit cell (eta=.1, lam=10, mu=.1, c=.5, sigma^2=4)
    for name, (eta, bet, mu, lam, c, s2), tol in [
        ("reviewer cell beta=0", (0.1, 0.0, 0.1, 10, 0.5, 4), 0.03),
        ("same cell beta=0.9", (0.1, 0.9, 0.1, 10, 0.5, 4), 0.03),
        ("extreme-coupling cell beta=0", (0.07, 0.0, 4, 10, 1.2, 1), 0.05),
    ]:
        closed = f(modal_var(eta, bet, mu, lam, c, s2, sqrt=np.sqrt))
        _, mc = simulate(eta, bet, mu, lam, c, np.sqrt(s2), T=300_000, seeds=24)
        ok(f"B1 MC == closed form ({name})", abs(mc / closed - 1) < tol,
           f"MC {mc:.5g} closed {closed:.5g}")
        if name.startswith("extreme"):
            froz = f(frozen_var(eta, bet, mu, lam, c, s2))
            ok("B1d frozen formula off by >2x where coupling is strong; modal exact",
               abs(mc / closed - 1) < tol and closed / froz > 2,
               f"frozen {froz:.5g} vs exact {closed:.5g} (ratio {closed / froz:.2f})")


def check_b3() -> None:
    eta, bet, mu, lam, c, sigma, G = 0.15, 0.9, 0.0, 10.0, 0.8, 1.0, 3.0
    lloc = lam * (1 + c**2)
    teff = (1 + bet) / (1 - bet)
    mean, var = simulate(eta, bet, mu, lam, c, sigma, T=300_000, seeds=24, tilt=G)
    ok("B3a tilted profile: E[d] = c phi' / lloc",
       abs(mean / (c * G / lloc) - 1) < 0.02, f"MC {mean:.5g} vs {c * G / lloc:.5g}")
    tgt = eta * sigma**2 / (lam * (2 - eta * lloc / teff))
    ok("B3b tilted profile: Var(d) on the tier-(i) formula",
       abs(var / tgt - 1) < 0.03, f"MC {var:.5g} vs {tgt:.5g}")


if __name__ == "__main__":
    print("== A: symbolic (modal reduction, Lyapunov identity, limits) ==")
    check_a1()
    check_a2()
    check_a3()
    check_a4()
    check_a5()
    print("== B: stability boundary + Monte Carlo ==")
    check_boundary()
    check_b1()
    check_b3()
    print("all checks passed")
