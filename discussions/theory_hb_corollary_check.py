"""Checks for the T5 heavy-ball-normalization corollary (plan_v5 T5).

Claims: at fixed eta_HB with eta = eta_HB/(1-beta), the stationary hill loss is
    l(beta) = eta_HB sigma^2 / (2 (1-beta) (2 - eta_HB lambda/(1+beta))),
stable iff eta_HB lambda < 2(1+beta); strictly increasing in beta for eta_HB lambda <= 1;
U-shaped with interior minimum beta* = sqrt(eta_HB lambda) - 1 for 1 < eta_HB lambda < 2;
for eta_HB lambda >= 2, beta = 0 unstable and stability requires beta > eta_HB lambda/2 - 1.

Run:  .venv/bin/python discussions/theory_hb_corollary_check.py   (repo root; ~30 s)
Every [check] line must end PASS.
"""
from __future__ import annotations

import numpy as np
import sympy as sp


def ok(name: str, cond: bool, detail: str = "") -> None:
    print(f"[check] {name}: {'PASS' if cond else 'FAIL'}  {detail}")
    assert cond, name


BET, EHB, LAM, S2 = sp.symbols("beta etaHB lambda sigma2", positive=True)


def check_symbolic() -> None:
    eta = EHB / (1 - BET)
    teff = (1 + BET) / (1 - BET)
    ema_loss = eta * S2 / (2 * (2 - eta * LAM / teff))
    hb_loss = EHB * S2 / (2 * (1 - BET) * (2 - EHB * LAM / (1 + BET)))
    ok("S1 substitution: EMA loss at eta = eta_HB/(1-beta) equals the displayed form",
       sp.simplify(ema_loss - hb_loss) == 0)
    d = sp.simplify(sp.diff(hb_loss, BET))
    # sign of d equals sign of 2 - 2*eta_HB*lambda/(1+beta)^2 ... times -1: recompute
    crit = sp.solve(sp.Eq(d, 0), BET)
    crit = [sp.simplify(c) for c in crit if sp.simplify(c).is_real is not False]
    tgt = sp.sqrt(EHB * LAM) - 1
    ok("S2 interior critical point is beta* = sqrt(eta_HB lambda) - 1",
       any(sp.simplify(c - tgt) == 0 for c in crit), f"roots: {crit}")
    ok("S3 derivative positive at beta=0 iff eta_HB lambda < 1 fails, i.e. loss falls "
       "first when eta_HB lambda > 1",
       sp.simplify(d.subs(BET, sp.Rational(1, 10**6)).subs(
           {EHB: sp.Rational(3, 2), LAM: 1, S2: 1})) < 0
       and sp.simplify(d.subs(BET, sp.Rational(1, 10**6)).subs(
           {EHB: sp.Rational(1, 2), LAM: 1, S2: 1})) > 0)


def simulate_hb(beta, eta_hb, lam, sigma, T=400_000, seeds=24, seed0=0):
    r = np.random.default_rng(seed0)
    eta = eta_hb / (1 - beta)
    y = np.zeros(seeds)
    m = np.zeros(seeds)
    acc, n = 0.0, 0
    for t in range(T):
        g = lam * y + sigma * r.standard_normal(seeds)
        m = beta * m + (1 - beta) * g
        y = y - eta * m
        if t >= T // 2:
            acc += (y * y).sum()
            n += y.size
    return 0.5 * lam * acc / n


def check_numeric() -> None:
    lam, sigma = 1.0, 1.0
    f = lambda b, ehl: float(
        (EHB * S2 / (2 * (1 - BET) * (2 - EHB * LAM / (1 + BET)))).subs(
            {BET: b, EHB: ehl, LAM: 1, S2: 1}) * sp.Rational(1, 2) * 2)
    # regime eta_HB lambda = 0.6: increasing
    ls = [simulate_hb(b, 0.6, lam, sigma) for b in (0.0, 0.3, 0.6, 0.9)]
    ok("N1 eta_HB lambda = 0.6: loss increasing in beta (MC)",
       ls[0] < ls[1] < ls[2] < ls[3], f"{[f'{x:.4f}' for x in ls]}")
    guides = [lam * f(b, 0.6) for b in (0.0, 0.3, 0.6, 0.9)]
    worst = max(abs(a / g - 1) for a, g in zip(ls, guides))
    ok("N2 MC on the closed-form guide", worst < 0.04, f"worst rel err {worst:.3f}")
    # regime eta_HB lambda = 1.5: U-shape, minimum near sqrt(1.5)-1 = 0.2247
    bstar = np.sqrt(1.5) - 1
    l_lo = simulate_hb(0.02, 1.5, lam, sigma)
    l_st = simulate_hb(bstar, 1.5, lam, sigma)
    l_hi = simulate_hb(0.7, 1.5, lam, sigma)
    ok("N3 eta_HB lambda = 1.5: interior minimum near beta* = 0.2247 (MC)",
       l_st < l_lo and l_st < l_hi,
       f"loss at (0.02, beta*, 0.7) = ({l_lo:.4f}, {l_st:.4f}, {l_hi:.4f})")
    # regime eta_HB lambda = 2.4: beta=0 diverges, beta > 0.2 stable
    y = simulate_hb(0.05, 2.4, lam, sigma, T=4000, seeds=4)
    ok("N4 eta_HB lambda = 2.4: unstable below beta = eta_HB lambda/2 - 1 = 0.2, "
       "stable above",
       not np.isfinite(y) or y > 1e6, f"sub-threshold tail loss {y:.2e}")
    y2 = simulate_hb(0.35, 2.4, lam, sigma)
    ok("N5 ... and finite above it", np.isfinite(y2) and y2 < 1e3, f"loss {y2:.4f}")


if __name__ == "__main__":
    print("== S: symbolic ==")
    check_symbolic()
    print("== N: Monte Carlo ==")
    check_numeric()
    print("all checks passed")
