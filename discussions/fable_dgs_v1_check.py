"""Sanity checks for analytical claims to be made in discussions/fable_dgs_v1.md.

Claim 1 (closed-loop hill map with EMA momentum):
  y_{t+1} = (1+b - e)y_t - b y_{t-1},  e = eta*(1-b)*lam
  - stability iff eta*lam < 2(1+b)/(1-b) = 2*T_eff
  - complex-root regime has |roots| = sqrt(b)
  - at the threshold the mode is a period-2 flip (root -> -1)

Claim 2 (stationary hill-gradient spectrum, noise-driven SGD, a = 1-eta*lam):
  g_t = lam*y_t + xi_t, y_{t+1} = a*y_t - eta*xi_t
  => spectral density S_g(w) = sigma^2 * |1-e^{-iw}|^2 / |1-a e^{-iw}|^2
  (zero DC mass; high-pass with cutoff ~ eta*lam; peak at pi iff eta*lam > 1)

Claim 3 (EMA of Nyquist tone h_t = (-1)^t A):
  m_t = (-1)^t * (1-b)(1-(-b)^t)/(1+b) * A   (=> |m| -> |A|/T_eff)

Claim 4 (SGDM vs SGD stationary hill variance at fixed eta, small/large eta*lam).
"""
import numpy as np

rng = np.random.default_rng(0)

# ---- Claim 1: roots of the closed-loop hill map ----
print("== Claim 1: closed-loop hill map ==")
for b in [0.0, 0.5, 0.9]:
    Teff = (1 + b) / (1 - b)
    for etalam in [1.8, 2.5, 2 * Teff * 0.999, 2 * Teff * 1.001]:
        e = etalam * (1 - b)
        roots = np.roots([1, -(1 + b - e), b])
        mod = np.max(np.abs(roots))
        tag = "stable" if mod < 1 else "UNSTABLE"
        cplx = np.iscomplex(roots).any()
        print(f" b={b:4} etalam={etalam:8.3f} 2Teff={2*Teff:7.3f} |root|max={mod:.4f} "
              f"complex={cplx} sqrt(b)={np.sqrt(b):.4f} {tag} roots={np.round(roots,3)}")

# empirically: simulate and confirm stability boundary
def simulate_hill(b, etalam, T=4000, y0=1.0):
    eta, lam = 0.1, etalam / 0.1
    y, m = y0, 0.0
    ys = []
    for t in range(T):
        g = lam * y
        m = b * m + (1 - b) * g
        y = y - eta * m
        ys.append(y)
        if abs(y) > 1e12:
            return np.inf
    return np.max(np.abs(ys[-100:]))

for b in [0.5, 0.9]:
    Teff = (1 + b) / (1 - b)
    lo, hi = simulate_hill(b, 2 * Teff * 0.99), simulate_hill(b, 2 * Teff * 1.01)
    print(f" b={b}: sim tail-amp at 0.99*2Teff={lo:.3e}, at 1.01*2Teff={hi}")

# ---- Claim 2: stationary hill-gradient spectrum ----
print("\n== Claim 2: stationary spectrum of hill gradient stream ==")
def hill_gradient_stream(etalam, T=2**16, sigma=1.0, eta=0.1):
    lam = etalam / eta
    a = 1 - etalam
    y = 0.0
    gs = np.empty(T)
    xis = rng.normal(0, sigma, T)
    for t in range(T):
        g = lam * y + xis[t]
        gs[t] = g
        y = a * y - eta * xis[t]
    return gs

for etalam in [0.3, 1.0, 1.8]:
    gs = hill_gradient_stream(etalam)
    gs = gs[2000:]  # burn-in
    T = len(gs)
    f = np.fft.rfftfreq(T) * 2 * np.pi
    emp = np.abs(np.fft.rfft(gs)) ** 2 / T
    a = 1 - etalam
    theo = np.abs(1 - np.exp(-1j * f)) ** 2 / np.abs(1 - a * np.exp(-1j * f)) ** 2
    # compare band-averaged ratios (skip DC bin)
    bands = np.array_split(np.arange(1, len(f)), 8)
    ratios = [emp[idx].mean() / theo[idx].mean() for idx in bands]
    print(f" etalam={etalam}: band-averaged emp/theory ratios = "
          + " ".join(f"{r:.2f}" for r in ratios)
          + f" | DC bin emp={emp[0]:.3e} vs mean={emp.mean():.3e}"
          + f" | peak at w={f[np.argmax(emp)]:.3f} (pi={np.pi:.3f})")

# ---- Claim 3: EMA of Nyquist tone ----
print("\n== Claim 3: EMA of (-1)^t A ==")
for b in [0.5, 0.9, 0.99]:
    T = 200
    h = np.array([(-1) ** t for t in range(1, T + 1)], dtype=float)
    m = np.zeros(T)
    acc = 0.0
    for t in range(T):
        acc = b * acc + (1 - b) * h[t]
        m[t] = acc
    t_idx = np.arange(1, T + 1)
    pred = (-1.0) ** t_idx * (1 - b) * (1 - (-b) ** t_idx) / (1 + b)
    err = np.max(np.abs(m - pred))
    print(f" b={b}: max|sim - closed form| = {err:.2e}; "
          f"tail amp={abs(m[-1]):.5f} vs (1-b)/(1+b)={(1-b)/(1+b):.5f}")

# ---- Claim 4: closed-loop stationary hill variance, SGDM vs SGD ----
print("\n== Claim 4: stationary Var(y), SGDM vs SGD (fixed eta) ==")
def stationary_var(b, etalam, T=400_000, eta=0.1, sigma=1.0, seed=1):
    r = np.random.default_rng(seed)
    lam = etalam / eta
    y, m = 0.0, 0.0
    xs = np.empty(T)
    xi = r.normal(0, sigma, T)
    for t in range(T):
        g = lam * y + xi[t]
        m = b * m + (1 - b) * g
        y = y - eta * m
        xs[t] = y
    return np.var(xs[T // 4:])

for etalam in [0.3, 1.0, 1.8]:
    row = []
    for b in [0.0, 0.5, 0.9, 0.95]:
        row.append(stationary_var(b, etalam))
    print(f" etalam={etalam}: Var(y) at b=0,0.5,0.9,0.95 -> "
          + " ".join(f"{v:.4f}" for v in row))


# ===== Claim 5 (E8 headline-figure design check) =====
"""Claim 5 (E8 design check): does a Figure-2-style trajectory figure actually show
'stronger beta wins' in the *noisy, large-LR* curved river valley — and fail to show it
in the deterministic one, as the closed-loop analysis predicts?

Landscape (E2/E3 params): L = (mu/2)(x-xs)^2 + (lam/2)(y - a*sin(k x))^2,
mu=0.1, lam=10, a=2, k=0.9, xs=5, start (-3, f(-3)+1). EMA-SGDM, additive Gaussian
gradient noise sigma (E3 used sigma=2; we sweep panel configs).

Panel A: eta=0.18 (eta*lam=1.8), sigma=2, beta in {0,0.5,0.9,0.99}.
  Expect: hill-offset rms decreases monotonically in beta (after settling),
  x-progress roughly equal, loss tail decreases in beta.
Panel B: eta=0.25 (eta*lam=2.5 > 2), sigma=2.
  Expect: beta=0 diverges; beta>=0.5 tracks the river (2*Teff(0.5)=6 > 2.5).
Panel A': same as A but sigma=0 (deterministic).
  Expect: NO monotone benefit — moderate beta best (underdamped ringing at 0.99).
"""
import numpy as np

mu, lam, amp, k, xs = 0.1, 10.0, 2.0, 0.9, 5.0

def f(x):  return amp * np.sin(k * x)
def fp(x): return amp * k * np.cos(k * x)

def loss(x, y):
    return 0.5 * mu * (x - xs) ** 2 + 0.5 * lam * (y - f(x)) ** 2

def grad(x, y):
    d = y - f(x)
    return np.array([mu * (x - xs) - lam * d * fp(x), lam * d])

def run(beta, eta, sigma, T=300, seed=0):
    rng = np.random.default_rng(seed)
    w = np.array([-3.0, f(-3.0) + 1.0])
    m = np.zeros(2)
    xsr, offs, losses = [], [], []
    for t in range(T):
        g = grad(*w) + rng.normal(0, sigma, 2)
        m = beta * m + (1 - beta) * g
        w = w - eta * m
        if not np.all(np.isfinite(w)) or np.abs(w).max() > 1e8:
            return dict(diverged=True, t=t)
        xsr.append(w[0]); offs.append(w[1] - f(w[0])); losses.append(loss(*w))
    xsr, offs, losses = map(np.asarray, (xsr, offs, losses))
    half = T // 2
    return dict(diverged=False,
                x_final=xsr[-1],
                off_rms_tail=float(np.sqrt(np.mean(offs[half:] ** 2))),
                loss_tail=float(np.mean(losses[half:])))

def table(title, eta, sigma, betas=(0.0, 0.5, 0.9, 0.99), seeds=range(8)):
    print(f"-- {title} (eta={eta}, eta*lam={eta*lam}, sigma={sigma}) --")
    print("   beta   div   x_final    off_rms(tail)   loss(tail)   pred_rms_ratio")
    base = None
    for b in betas:
        rs = [run(b, eta, sigma, seed=s) for s in seeds]
        ndiv = sum(r["diverged"] for r in rs)
        ok = [r for r in rs if not r["diverged"]]
        if not ok:
            print(f"   {b:5}  {ndiv}/{len(rs)}   (all diverged)")
            continue
        xf = np.mean([r["x_final"] for r in ok])
        rms = np.mean([r["off_rms_tail"] for r in ok])
        lt = np.mean([r["loss_tail"] for r in ok])
        # linear-model prediction of rms ratio vs beta=0 (straight-valley formula)
        Teff = (1 + b) / (1 - b)
        v = 1.0 / (2 - eta * lam / Teff) if eta * lam / Teff < 2 else np.inf
        if base is None: base = v
        pred = np.sqrt(v / base) if np.isfinite(v) and np.isfinite(base) else float("nan")
        print(f"   {b:5}  {ndiv}/{len(rs)}   {xf:7.3f}    {rms:11.4f}    {lt:9.4f}    {pred:8.3f}")
    print()

table("Panel A: noisy, large LR", eta=0.18, sigma=2.0)
table("Panel B: beyond GD stability", eta=0.25, sigma=2.0)
table("Panel A': deterministic control", eta=0.18, sigma=0.0, seeds=range(1))
table("Lag edge: very large beta", eta=0.18, sigma=2.0, betas=(0.9, 0.99, 0.999))

# ---------------------------------------------------------------------------
# Captured output (python3, 2026-07-03) — quoted in fable_dgs_v1.md:
# ---------------------------------------------------------------------------
# == Claim 1: closed-loop hill map ==
#  b= 0.0 etalam=   1.800 2Teff=  2.000 |root|max=0.8000 complex=False sqrt(b)=0.0000 stable roots=[-0.8  0. ]
#  b= 0.0 etalam=   2.500 2Teff=  2.000 |root|max=1.5000 complex=False sqrt(b)=0.0000 UNSTABLE roots=[-1.5  0. ]
#  b= 0.0 etalam=   1.998 2Teff=  2.000 |root|max=0.9980 complex=False sqrt(b)=0.0000 stable roots=[-0.998  0.   ]
#  b= 0.0 etalam=   2.002 2Teff=  2.000 |root|max=1.0020 complex=False sqrt(b)=0.0000 UNSTABLE roots=[-1.002  0.   ]
#  b= 0.5 etalam=   1.800 2Teff=  6.000 |root|max=0.7071 complex=True sqrt(b)=0.7071 stable roots=[0.3+0.64j 0.3-0.64j]
#  b= 0.5 etalam=   2.500 2Teff=  6.000 |root|max=0.7071 complex=True sqrt(b)=0.7071 stable roots=[0.125+0.696j 0.125-0.696j]
#  b= 0.5 etalam=   5.994 2Teff=  6.000 |root|max=0.9940 complex=False sqrt(b)=0.7071 stable roots=[-0.994 -0.503]
#  b= 0.5 etalam=   6.006 2Teff=  6.000 |root|max=1.0060 complex=False sqrt(b)=0.7071 UNSTABLE roots=[-1.006 -0.497]
#  b= 0.9 etalam=   1.800 2Teff= 38.000 |root|max=0.9487 complex=True sqrt(b)=0.9487 stable roots=[0.86+0.4j 0.86-0.4j]
#  b= 0.9 etalam=   2.500 2Teff= 38.000 |root|max=0.9487 complex=True sqrt(b)=0.9487 stable roots=[0.825+0.468j 0.825-0.468j]
#  b= 0.9 etalam=  37.962 2Teff= 38.000 |root|max=0.9487 complex=True sqrt(b)=0.9487 stable roots=[-0.948+0.033j -0.948-0.033j]
#  b= 0.9 etalam=  38.038 2Teff= 38.000 |root|max=1.0301 complex=False sqrt(b)=0.9487 UNSTABLE roots=[-1.03  -0.874]
#  b=0.5: sim tail-amp at 0.99*2Teff=5.097e-113, at 1.01*2Teff=inf
#  b=0.9: sim tail-amp at 0.99*2Teff=4.389e-89, at 1.01*2Teff=inf
# 
# == Claim 2: stationary spectrum of hill gradient stream ==
#  etalam=0.3: band-averaged emp/theory ratios = 0.97 1.02 0.99 0.97 1.00 1.00 1.01 1.02 | DC bin emp=1.313e-06 vs mean=1.177e+00 | peak at w=2.663 (pi=3.142)
#  etalam=1.0: band-averaged emp/theory ratios = 1.01 1.01 1.02 0.99 0.99 1.01 1.02 1.01 | DC bin emp=1.921e-06 vs mean=2.017e+00 | peak at w=3.012 (pi=3.142)
#  etalam=1.8: band-averaged emp/theory ratios = 1.04 0.98 1.02 0.99 1.01 1.02 0.99 1.01 | DC bin emp=2.907e-04 vs mean=1.006e+01 | peak at w=3.085 (pi=3.142)
# 
# == Claim 3: EMA of (-1)^t A ==
#  b=0.5: max|sim - closed form| = 5.55e-17; tail amp=0.33333 vs (1-b)/(1+b)=0.33333
#  b=0.9: max|sim - closed form| = 1.39e-17; tail amp=0.05263 vs (1-b)/(1+b)=0.05263
#  b=0.99: max|sim - closed form| = 6.94e-18; tail amp=0.00435 vs (1-b)/(1+b)=0.00503
# 
# == Claim 4: stationary Var(y), SGDM vs SGD (fixed eta) ==
#  etalam=0.3: Var(y) at b=0,0.5,0.9,0.95 -> 0.0195 0.0175 0.0167 0.0166
#  etalam=1.0: Var(y) at b=0,0.5,0.9,0.95 -> 0.0100 0.0060 0.0051 0.0050
#  etalam=1.8: Var(y) at b=0,0.5,0.9,0.95 -> 0.0277 0.0040 0.0029 0.0028
# -- Panel A: noisy, large LR (eta=0.18, eta*lam=1.7999999999999998, sigma=2.0) --
#    beta   div   x_final    off_rms(tail)   loss(tail)   pred_rms_ratio
#      0.0  0/8     7.521         4.3970     285.7950       1.000
#      0.5  0/8     0.228         0.5177       3.2487       0.378
#      0.9  0/8     3.496         0.1886       0.4969       0.324
#     0.99  0/8     0.882         0.3528       2.1416       0.317
# 
# -- Panel B: beyond GD stability (eta=0.25, eta*lam=2.5, sigma=2.0) --
#    beta   div   x_final    off_rms(tail)   loss(tail)   pred_rms_ratio
#      0.0  8/8   (all diverged)
#      0.5  0/8     5.410         1.9922      64.8117       1.000
#      0.9  0/8     4.901         0.2456       0.6285       0.790
#     0.99  0/8     1.870         0.3068       1.9360       0.766
# 
# -- Panel A': deterministic control (eta=0.18, eta*lam=1.7999999999999998, sigma=0.0) --
#    beta   div   x_final    off_rms(tail)   loss(tail)   pred_rms_ratio
#      0.0  0/1    15.360         3.6628      92.1044       1.000
#      0.5  0/1     1.769         0.3032       0.7492       0.378
#      0.9  0/1     3.837         0.0082       0.1500       0.324
#     0.99  0/1     3.988         0.2834       0.5895       0.317
# 
# -- Lag edge: very large beta (eta=0.18, eta*lam=1.7999999999999998, sigma=2.0) --
#    beta   div   x_final    off_rms(tail)   loss(tail)   pred_rms_ratio
#      0.9  0/8     3.496         0.1886       0.4969       1.000
#     0.99  0/8     0.882         0.3528       2.1416       0.978
#    0.999  0/8    -0.816         0.5947       3.9247       0.976
