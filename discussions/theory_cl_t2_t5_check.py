"""Numerical checks for every claim in discussions/theory_cl_t2_t5.md.

Extends discussions/fable_dgs_v1_check.py (Claims 1-5 there verified CL-1/CL-2/T2-spectrum/
EMA-of-Nyquist and the E8 pilot). Here each check is tied to a statement of the theory note:

  Check A (CL-1)   Jury conditions on z^2-(1+b-e)z+b, e=eta(1-b)lam: stability iff
                   eta*lam < 2(1+b)/(1-b) = 2*T_eff; root -> -1 at threshold; |roots|=sqrt(b)
                   in the complex regime; slow mode contracts at 1-eta*mu + O((eta*mu)^2),
                   b-independent to first order.
  Check B (CL-2)   AR(2) stationary variance Var(y) = eta*sigma^2/(lam*(2 - eta*lam/T_eff));
                   exact vs long simulation; monotone decreasing in b; reduction factor
                   (2-eta*lam)/(2-eta*lam/T_eff) vs SGD; d-dim loss-gap corollary.
  Check C (T2)     hill-gradient spectrum S_g(w) = sigma^2*|1-e^{-iw}|^2/|1-a e^{-iw}|^2,
                   a=1-eta*lam: zero DC, MONOTONE increasing on [0,pi] for every a in (-1,1)
                   (so "Nyquist-peaked iff eta*lam>1" must be read as concentration, not argmax),
                   low-frequency bound S(w) <= sigma^2 w^2 / min(eta*lam, 2-eta*lam)^2,
                   Var(g) = 2 sigma^2/(2-eta*lam), and the exact open-loop filtered-power ratio
                   rho(b) = Var(EMA_b(g))/Var(g) interpolating 1/T_eff (white) .. 1/T_eff^2 (tone).
  Check D (T3')    exact DC-telescoping identity g_hat(0) = (w_1 - w_{T+1})/eta + b/(1-b) m_T
                   on a closed-loop curved-valley run (machine precision), and the resulting
                   confined-hill vs travelled-river DC-mass gap growing with T.
  Check E (T5)     EMA of G_t = S + (-1)^t A: M_t = (1-b^t) S + eps_t A with
                   eps_t = (-1)^t (1-b)(1-(-b)^t)/(1+b); Weyl head/tail bounds; two-sided tail
                   |eps_t| sigma_{2r+1}(A) <= sigma_{r+1}(M_t) <= |eps_t| sigma_1(A);
                   Wedin bound sinTheta <= |eps_t| ||A V_S|| / ((1-b^t)sigma_r - |eps_t|||A||)
                   vs measured; polar-only stuck at sinTheta = 1 when ||A|| > sigma_r(S)
                   (orthogonal worked example) and pre-polar's recovery there;
                   post-polar exact period-2 limit m~_+/- = (P_+/- + b P_-/+)/(1+b).

Run:  python3 discussions/theory_cl_t2_t5_check.py     (numpy only, ~1 min)
"""
import numpy as np

rng = np.random.default_rng(0)
line = lambda s="": print(s)


def teff(b):
    return (1 + b) / (1 - b)


# ======================= Check A: CL-1 stability =============================
line("== Check A (CL-1): Jury conditions, threshold mode, slow-mode rate ==")
for b in [0.0, 0.5, 0.9, 0.99]:
    thr = 2 * teff(b)
    for frac, tag in [(0.999, "below"), (1.001, "above")]:
        etalam = thr * frac
        e = etalam * (1 - b)
        roots = np.roots([1, -(1 + b - e), b])
        mod = np.max(np.abs(roots))
        worst = roots[np.argmax(np.abs(roots))]
        ok = (mod < 1) == (frac < 1)
        line(f"  b={b:<5} etalam={etalam:9.3f} (thr {thr:8.3f} x{frac}): |root|max={mod:.4f} "
             f"worst-root={np.round(worst, 3)} jury{'==sim OK' if ok else ' MISMATCH'}")
# complex-regime modulus + slow-mode contraction rate
for b in [0.5, 0.9]:
    e = 1.8 * (1 - b)
    roots = np.roots([1, -(1 + b - e), b])
    line(f"  b={b}: complex regime |roots|={np.abs(roots)[0]:.6f} vs sqrt(b)={np.sqrt(b):.6f}")
for b in [0.0, 0.5, 0.9, 0.99]:
    etamu = 1e-3
    e = etamu * (1 - b)
    roots = np.roots([1, -(1 + b - e), b])
    slow = roots[np.argmax(roots.real)].real  # slow mode ~ 1 - eta*mu
    line(f"  b={b:<5} slow-mode root at eta*mu=1e-3: {slow:.8f}  (1-eta*mu={1-etamu:.8f}, "
         f"dev={abs(slow-(1-etamu)):.2e})")

# ======================= Check B: CL-2 stationary variance ===================
line("\n== Check B (CL-2): stationary Var(y) closed form vs simulation ==")


def simulate_var(b, etalam, eta=0.1, sigma=1.0, T=2_000_000, seed=1):
    r = np.random.default_rng(seed)
    lam = etalam / eta
    y, m = 0.0, 0.0
    acc, acc2, n = 0.0, 0.0, 0
    xi = r.normal(0, sigma, T)
    for t in range(T):
        g = lam * y + xi[t]
        m = b * m + (1 - b) * g
        y = y - eta * m
        if t >= T // 4:
            acc += y
            acc2 += y * y
            n += 1
    mean = acc / n
    return acc2 / n - mean * mean


def var_cl2(b, etalam, eta=0.1, sigma=1.0):
    lam = etalam / eta
    return eta * sigma**2 / (lam * (2 - etalam / teff(b)))


for etalam in [0.3, 1.0, 1.8, 2.5]:
    row = []
    for b in [0.0, 0.5, 0.9, 0.95]:
        if etalam >= 2 * teff(b):
            row.append((np.nan, np.nan))
            continue
        v_sim = simulate_var(b, etalam)
        v_th = var_cl2(b, etalam)
        row.append((v_sim, v_th))
    txt = "  ".join("unstable " if np.isnan(s) else f"{s:.5f}/{t:.5f}" for s, t in row)
    line(f"  etalam={etalam:<4}: sim/theory at b=0,0.5,0.9,0.95 -> {txt}")
line("  reduction factor vs SGD (2-etalam)/(2-etalam/T_eff), etalam=1.8:")
for b in [0.5, 0.9, 0.99]:
    line(f"    b={b}: {(2-1.8)/(2-1.8/teff(b)):.4f}")
# d-dim corollary: loss gap E[sum_i lam_i y_i^2/2] = sum_i eta sigma^2/(2(2-eta lam_i/T_eff))
lams, eta, sig, b = np.array([2.0, 6.0, 18.0]), 0.1, 1.0, 0.9
pred = np.sum(eta * sig**2 / (2 * (2 - eta * lams / teff(b))))
sim = sum(0.5 * l * simulate_var(b, eta * l, eta=eta, seed=3 + i) for i, l in enumerate(lams))
line(f"  d-dim loss gap (lams={lams}, b={b}): sim={sim:.5f} theory={pred:.5f}")

# ======================= Check C: T2 spectrum ================================
line("\n== Check C (T2): spectrum shape, monotonicity, low-freq bound, filtered power ==")


def S_g(w, etalam, sigma=1.0):
    a = 1 - etalam
    return sigma**2 * np.abs(1 - np.exp(-1j * w)) ** 2 / np.abs(1 - a * np.exp(-1j * w)) ** 2


w = np.linspace(0, np.pi, 20001)
for etalam in [0.3, 1.0, 1.8]:
    S = S_g(w, etalam)
    mono = bool(np.all(np.diff(S) >= -1e-12))
    a = 1 - etalam
    lowbound = w**2 / min(etalam, 2 - etalam) ** 2
    bound_ok = bool(np.all(S <= lowbound + 1e-9))
    conc = S[-1] / S[len(w) // 2]  # S(pi)/S(pi/2): concentration, not argmax
    line(f"  etalam={etalam}: S(0)={S[0]:.1e}, monotone increasing on [0,pi]: {mono}, "
         f"S<=sigma^2 w^2/min(el,2-el)^2: {bound_ok}, S(pi)/S(pi/2)={conc:8.3f}, "
         f"S(pi)={S[-1]:.3f} vs 4/(2-el)^2={4/(2-etalam)**2:.3f}")
# Var(g) = 2 sigma^2/(2-etalam)  (numerical integral of S/2pi over (-pi,pi])
for etalam in [0.3, 1.0, 1.8]:
    var_num = getattr(np, 'trapezoid', np.trapz)(S_g(w, etalam), w) / np.pi  # even: 2x integral over [0,pi] /2pi
    line(f"  etalam={etalam}: Var(g) integral={var_num:.4f} vs 2/(2-el)={2/(2-etalam):.4f}")


# exact open-loop filtered power ratio rho(b) = Var(EMA(g))/Var(g), partial fractions
def rho_filtered(b, etalam):
    a = 1 - etalam
    A = (1 - b) / (a - b)
    B = -etalam / (a - b)
    var_m = (1 - b) ** 2 * (A**2 / (1 - b**2) + 2 * A * B / (1 - a * b) + B**2 / (1 - a**2))
    return var_m / (2 / (2 - etalam))


for etalam in [0.3, 1.0, 1.8]:
    a = 1 - etalam
    for b in [0.5, 0.9]:
        # numerical check: integrate |H_b|^2 S_g / 2pi
        H2 = (1 - b) ** 2 / np.abs(1 - b * np.exp(-1j * w)) ** 2
        num = getattr(np, 'trapezoid', np.trapz)(H2 * S_g(w, etalam), w) / np.pi / (2 / (2 - etalam))
        th = rho_filtered(b, etalam)
        line(f"  etalam={etalam} b={b}: filtered/raw power  integral={num:.5f} "
             f"closed-form={th:.5f}  [1/T_eff^2={1/teff(b)**2:.5f}, 1/T_eff={1/teff(b):.5f}]")
# Chebyshev claim: rho(b; etalam) <= 1/T_eff for every etalam in (0,2), sweeping fine grids
worst = -np.inf
for b in [0.3, 0.5, 0.9, 0.99]:
    for etalam in np.linspace(0.01, 1.99, 397):
        if abs((1 - etalam) - b) < 1e-9:
            continue  # repeated-pole point, take by continuity
        worst = max(worst, rho_filtered(b, etalam) - 1 / teff(b))
line(f"  Chebyshev claim rho <= 1/T_eff: max(rho - 1/T_eff) over grid = {worst:.2e} (<= 0 ok)")

# ======================= Check D: T3' telescoping ============================
line("\n== Check D (T3'): DC telescoping identity on a closed-loop curved-valley run ==")
mu, lam, amp, kk, xstar = 0.1, 10.0, 2.0, 0.9, 5.0
f = lambda x: amp * np.sin(kk * x)
fp = lambda x: amp * kk * np.cos(kk * x)


def grad(wv):
    x, y = wv
    d = y - f(x)
    return np.array([mu * (x - xstar) - lam * d * fp(x), lam * d])


for b, sig, T in [(0.9, 2.0, 300), (0.5, 0.0, 200), (0.99, 2.0, 800)]:
    r = np.random.default_rng(7)
    wv = np.array([-3.0, f(-3.0) + 1.0])
    w1 = wv.copy()
    m = np.zeros(2)
    gsum = np.zeros(2)
    G = []
    for t in range(T):
        g = grad(wv) + (sig * r.standard_normal(2) if sig else 0.0)
        G.append(g)
        m = b * m + (1 - b) * g
        wv = wv - 0.18 * m
    G = np.asarray(G)
    ghat0 = G.sum(0)  # g_hat(0) = sum_t g_t
    ident = (w1 - wv) / 0.18 + b / (1 - b) * m
    err = np.max(np.abs(ghat0 - ident))
    line(f"  b={b} sigma={sig} T={T}: max|g_hat(0) - [(w_1-w_(T+1))/eta + b/(1-b) m_T]| = {err:.2e}")
    # hill vs river DC mass on this run (project g on live hill normal / river tangent at each t
    # is NOT the DC statement; T3' is per fixed coordinate -> use x and the hill offset coordinate)
line("  fixed-coordinate DC mass growth (straight valley, so coordinates are exact):")
land_lam, land_mu = 10.0, 0.1
for T in [200, 800, 3200]:
    r = np.random.default_rng(11)
    wv = np.array([-3.0, 1.0])
    w1 = wv.copy()
    m = np.zeros(2)
    G, Ys = [], []
    for t in range(T):
        g = np.array([land_mu * (wv[0] - xstar), land_lam * wv[1]]) + 2.0 * r.standard_normal(2)
        G.append(g)
        m = 0.9 * m + 0.1 * g
        wv = wv - 0.18 * m
        Ys.append(wv[1])
    G = np.asarray(G)
    R = np.max(np.abs(Ys))  # realized tube radius of the hill coordinate
    bound = 2 * R / 0.18 + 0.9 / 0.1 * abs(m[1])
    line(f"    T={T:>5}: |g_hat_river(0)|={abs(G[:,0].sum()):9.2f}  "
         f"|g_hat_hill(0)|={abs(G[:,1].sum()):7.2f}  "
         f"T3' hill bound 2R/eta+(b/(1-b))|m_T,hill|={bound:7.2f} (R={R:.2f})")

# ======================= Check E: T5 =========================================
line("\n== Check E (T5): Weyl/Wedin bounds, polar-only failure, post-polar limit ==")
m_, n_, r_ = 24, 18, 3
U0 = np.linalg.qr(rng.standard_normal((m_, r_)))[0]
V0 = np.linalg.qr(rng.standard_normal((n_, r_)))[0]
sv = np.array([4.0, 3.0, 2.0])
S = U0 @ np.diag(sv) @ V0.T
A = rng.standard_normal((m_, n_))
A = A / np.linalg.norm(A, 2) * 1.0  # ||A||_op = 1, generic
sigr = sv[-1]


def eps_t(b, t):
    return (-1.0) ** t * (1 - b) * (1 - (-b) ** t) / (1 + b)


def sin_theta(Ua, Ub):
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - np.clip(s.min(), 0, 1) ** 2)))


for b in [0.9, 0.99]:
    Mbuf = np.zeros((m_, n_))
    worst_dev, worst_wedin, worst_tail = 0.0, 0.0, 0.0
    for t in range(1, 401):
        Mbuf = b * Mbuf + (1 - b) * (S + (-1.0) ** t * A)
        Mth = (1 - b**t) * S + eps_t(b, t) * A
        worst_dev = max(worst_dev, np.max(np.abs(Mbuf - Mth)))
        e = abs(eps_t(b, t))
        s_all = np.linalg.svd(Mbuf, compute_uv=False)
        # two-sided tail bound
        lo, hi = e * np.linalg.svd(A, compute_uv=False)[2 * r_], e * 1.0
        if not (lo - 1e-12 <= s_all[r_] <= hi + 1e-12):
            worst_tail = max(worst_tail, max(lo - s_all[r_], s_all[r_] - hi))
        # Wedin
        delta = (1 - b**t) * sigr - e * 1.0
        if delta > 0:
            Umt = np.linalg.svd(Mbuf, full_matrices=False)[0][:, :r_]
            measured = sin_theta(Umt, U0)
            bound = e * max(np.linalg.norm(A @ V0, 2), np.linalg.norm(A.T @ U0, 2)) / delta
            if measured > bound + 1e-12:
                worst_wedin = max(worst_wedin, measured - bound)
    line(f"  b={b}: max|EMA - closed form|={worst_dev:.2e}; tail-bound violations={worst_tail:.1e}; "
         f"Wedin violations={worst_wedin:.1e}")
    t = 400
    e = abs(eps_t(b, t))
    Umt = np.linalg.svd(Mbuf, full_matrices=False)[0][:, :r_]
    line(f"        t=400: sinTheta measured={sin_theta(Umt, U0):.5f}  "
         f"Wedin bound={e*max(np.linalg.norm(A@V0,2),np.linalg.norm(A.T@U0,2))/((1-b**t)*sigr-e):.5f}  "
         f"crude ||A||/T_eff/(sigma_r-||A||/T_eff)={1/teff(b)/(sigr-1/teff(b)):.5f}")

# polar-only stuck: orthogonal example with alpha > sigma_r
line("  polar-only failure example (S=sigma uv^T, A=alpha u'v'^T, alpha=2 sigma):")
u1, u2 = np.eye(6)[:, :1], np.eye(6)[:, 1:2]
v1, v2 = np.eye(5)[:, :1], np.eye(5)[:, 1:2]
sig0, alp = 1.0, 2.0
S1 = sig0 * u1 @ v1.T
A1 = alp * u2 @ v2.T
Mb = np.zeros((6, 5))
for t in range(1, 201):
    Gt = S1 + (-1.0) ** t * A1
    Mb = 0.9 * Mb + 0.1 * Gt
    if t in (1, 10, 200):
        Upol = np.linalg.svd(Gt, full_matrices=False)[0][:, :1]
        Upre = np.linalg.svd(Mb, full_matrices=False)[0][:, :1]
        line(f"    t={t:>3}: polar-only sinTheta={sin_theta(Upol, u1):.3f}   "
             f"pre-polar sinTheta={sin_theta(Upre, u1):.3f}")

# post-polar exact period-2 limit
line("  post-polar stationary period-2 limit m~ = (P_s + b P_-s)/(1+b):")


def polar(X):
    Uu, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Uu @ Vt


Pp, Pm = polar(S + A), polar(S - A)
for b in [0.5, 0.9]:
    mt = np.zeros((m_, n_))
    for t in range(1, 3001):
        mt = b * mt + (1 - b) * (Pp if t % 2 == 0 else Pm)
    # t=3000 even -> limit (Pp + b Pm)/(1+b)
    lim = (Pp + b * Pm) / (1 + b)
    line(f"    b={b}: max|m~_T - limit| = {np.max(np.abs(mt - lim)):.2e};  "
         f"midpoint dev from orthogonality ||(P++P-)/2 sv - 1||: "
         f"{np.max(np.abs(np.linalg.svd((Pp+Pm)/2, compute_uv=False) - 1)):.3f}")
# alignment comparison in the generic setup at large t
for b in [0.9, 0.99]:
    Mb = np.zeros((m_, n_))
    mt = np.zeros((m_, n_))
    for t in range(1, 401):
        Gt = S + (-1.0) ** t * A
        Mb = b * Mb + (1 - b) * Gt
        mt = b * mt + (1 - b) * polar(Gt)
    ideal = U0 @ V0.T
    a_pre = np.sum(polar(Mb) * ideal) / r_
    a_post = np.sum(mt * ideal) / r_
    a_only = np.sum(polar(S + A) * ideal) / r_
    line(f"    generic alignment at t=400, b={b}: pre={a_pre:.4f} post={a_post:.4f} "
         f"polar-only={a_only:.4f}")
line("\ndone.")
