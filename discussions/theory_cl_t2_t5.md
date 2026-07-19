# Theory note: CL-1/CL-2 (closed loop), T2 (stochastic spectrum), T3′ (confinement), T5 (filter-first)

Staging note for the theorem program of `fable_dgs_v1.md` §8.1, companion to `theory_t1_t3.md`
(whose Proposition 1 / Theorem T1 / Theorem T3 are used freely below). Rigorous; to be ported into
`latex/` under `WRITING.md` once stable. Notation follows `idea_v1.md` and the existing note: EMA
momentum `m_t = β m_{t-1} + (1-β) g_t`, `m_0 = 0`, `T_eff = (1+β)/(1-β)`, step `η`, hill curvature
`λ`, river curvature `μ`, noise scale `σ`. Every claim is flagged [✓ verified] with the checking
code in `theory_cl_t2_t5_check.py` (Checks A–E; key numbers quoted inline) — plus Claims 1–5 of
`fable_dgs_v1_check.py` where noted.

One correction to `fable_dgs_v1.md` §3.2(a)/T2 discovered while verifying: the sampled
hill-gradient spectrum is maximized at `ω = π` for **every** `ηλ ∈ (0,2)` — "peaks at Nyquist iff
`ηλ > 1`" is false as an argmax statement. The correct dichotomy is *concentration*: the closed
form below gives `S(π)/S(π/2) = 2(1+a²)/(1+a)² ≥ 2` iff `ηλ ≥ 1` (with `a = 1-ηλ`), diverging as
`ηλ → 2`, versus `≈ 1` (white above the cutoff) for `ηλ ≪ 1`. T2 is stated in the corrected form.

---

## 0. Setup: the closed-loop hill model

The straight-valley hill coordinate under EMA-SGDM with additive gradient noise:

    g_t = λ y_t + ξ_t,      m_t = β m_{t-1} + (1-β) g_t,      y_{t+1} = y_t - η m_t,

with `ξ_t` white, `E[ξ_t] = 0`, `E[ξ_t²] = σ²` (deterministic case: `σ = 0`). Eliminating `m`
via `m_t = (y_t - y_{t+1})/η` gives the scalar closed-loop recursion [algebra; used throughout]

    y_{t+1} = (1 + β - e) y_t - β y_{t-1} - η(1-β) ξ_t,        e := η(1-β)λ,        (CL)

an AR(2) driven by `-η(1-β)ξ`. Its characteristic polynomial is `p(z) = z² - (1+β-e) z + β`.
The river coordinate obeys (CL) with `λ → μ` (and its own noise component). Everything in
§§1–2 is exact for this linear model; §3 reads off the `β = 0` gradient stream; §4 needs no
model at all; §5 is the matrix/Muon case.

---

## 1. Proposition CL-1 — closed-loop stability threshold  [✓ Check A; fable Claim 1]

**Statement.** For `β ∈ [0,1)` and `η, λ > 0`, the recursion (CL) is stable (both roots of `p`
inside the unit disk) **iff**

    ηλ < 2 (1+β)/(1-β) = 2 T_eff .

At the threshold the exiting root is `z = -1` (a period-2 flip: the Nyquist mode survives
momentum, at the shifted threshold). In the complex-root regime the root modulus is exactly
`√β`. The slow (river) mode is `z = 1 - ημ + O(β (ημ)²/(1-β))`, independent of `β` to first
order in `ημ`.

**Proof.** Jury conditions for `z² + a₁ z + a₀` with `a₁ = -(1+β-e)`, `a₀ = β`:
(i) `p(1) = e > 0` — always; (ii) `p(-1) = 2(1+β) - e > 0` ⟺ `η(1-β)λ < 2(1+β)` ⟺
`ηλ < 2T_eff`; (iii) `|a₀| = β < 1` — always. So (ii) is the binding condition, and at equality
`p(-1) = 0`, i.e. the unstable mode is the alternating one. Complex roots have product `a₀ = β`,
hence modulus `√β`. For the slow mode write the root as `1 - δ`: the product of roots is `β`, so
the second root is `β + O(δ)`, and the sum condition `(1-δ) + (β+βδ+O(δ²)) = 1 + β - e` gives
`δ = e/(1-β) + O(βδ²/(1-β)) = ημ + O(β(ημ)²/(1-β))`. ∎

[✓ Check A: root modulus crosses 1 within ±0.1% of `2T_eff` for `β ∈ {0, 0.5, 0.9, 0.99}`;
exiting root real and `≈ -1`; complex modulus `= √β` to 6 digits; slow-mode deviation from
`1-ημ` at `ημ = 10⁻³` is `0, 10⁻⁶, 9·10⁻⁶, 1.3·10⁻⁴` for `β = 0, 0.5, 0.9, 0.99` — the
`β(ημ)²/(1-β)` scaling. Simulated blow-up brackets the threshold to ±1% (fable Claim 1).]

**Remarks.**
- At `β = 0` this recovers `ηλ < 2`. In sharpness language the threshold is the known
  heavy-ball edge-of-stability form `2(1+β)/η_HB` (Cohen et al. 2021; arXiv:2604.14108 for the
  stochastic-stability version): with the EMA normalization `η_HB = η(1-β)`, the two forms are
  the same inequality. The contribution here is deriving it inside the paper's own hill map and
  the reading: at fixed `η`, momentum buys a factor `T_eff` of stability headroom against the
  steepest hills.
- The `√β` modulus (the underdamped hill mode of the draft's closed-loop remark) and the
  threshold are two faces of the same polynomial: momentum makes the transient ring longer
  *and* the loop harder to destabilize.
- The slow-mode expansion quantifies "river drift preserved": the `β`-dependence of the river
  contraction rate is `O(β(ημ)²/(1-β))`, negligible exactly when `ημ ≪ (1-β)/β ≈ 2/T_eff` —
  the same condition as the T6 window's lower edge (§6).

---

## 2. Proposition CL-2 — stationary hill variance  [✓ Check B; fable Claim 4]

**Statement.** Under (CL) with `ηλ < 2T_eff`, the stationary variance of the hill coordinate is

    Var(y_∞) = η σ² / ( λ (2 - ηλ/T_eff) ),

monotone decreasing in `β`, with these regime readings:

- `β = 0`: `Var = ησ²/(λ(2-ηλ))` — the classical SGD value, diverging as `ηλ → 2`.
- fixed `ηλ < 2`: the momentum-to-SGD variance ratio is `(2-ηλ)/(2-ηλ/T_eff)`, which is
  `≈ (2-ηλ)/2` for large `β` — a bounded, modest gain at small `ηλ` (`≈ 0.85` at `ηλ = 0.3`)
  and an unbounded one as `ηλ → 2` (`→ 0.10` at `ηλ = 1.8`).
- `2 ≤ ηλ < 2T_eff`: SGD diverges; EMA-SGDM has finite variance `≈ ησ²/(2λ)` for large `T_eff`.
  Combined with CL-1: momentum both extends the usable learning-rate range by `T_eff` and
  suppresses the stationary hill energy inside it, at first-order-preserved river drift (CL-1).

**Proof.** (CL) is an AR(2) `y_{t+1} = φ₁ y_t + φ₂ y_{t-1} + w_t` with `φ₁ = 1+β-e`,
`φ₂ = -β`, `Var(w) = η²(1-β)²σ²`. The stationary variance of an AR(2) is
`Var(y) = Var(w)(1-φ₂) / [(1+φ₂)(1-φ₁-φ₂)(1+φ₁-φ₂)]`. Here `1-φ₂ = 1+β`, `1+φ₂ = 1-β`,
`1-φ₁-φ₂ = e`, `1+φ₁-φ₂ = 2(1+β)-e`, so

    Var(y) = η²(1-β)²σ²(1+β) / [ (1-β) · η(1-β)λ · (2(1+β) - η(1-β)λ) ]
           = η σ² / ( λ (2(1+β) - η(1-β)λ)/(1+β) )
           = η σ² / ( λ (2 - ηλ/T_eff) ).

Monotonicity: `T_eff` increases in `β`, so the denominator increases. ∎

[✓ Check B: closed form vs 2·10⁶-step simulation agrees to 3 digits at every
`(ηλ, β) ∈ {0.3, 1.0, 1.8, 2.5} × {0, 0.5, 0.9, 0.95}` (e.g. `ηλ = 1.8, β = 0.9`:
0.00292/0.00292); at `ηλ = 2.5` the `β = 0` cell diverges while `β ≥ 0.5` matches the formula.]

**Corollary (d-dimensional loss gap).** For a quadratic valley with hill eigenvalues
`{λ_i}` and isotropic per-coordinate gradient noise `σ²`, the stationary excess loss is

    E[ Σ_i (λ_i/2) y_i² ] = Σ_i η σ² / ( 2 (2 - ηλ_i/T_eff) ),

each term requiring `ηλ_i < 2T_eff`. [✓ Check B: `λ = (2,6,18)`, `β = 0.9`: simulation
0.07678 vs formula 0.07678.] At `β = 0` and small `ηλ_i` each term is `ησ²/4` — the same
`O(dησ²)` scaling as the stable-phase loss gap of Wen et al. (their constant differs with the
noise convention); the formula extends it across `β` and up to `ηλ_i → 2T_eff`.

**Remark (convention).** All of this is the EMA normalization. In heavy-ball normalization
(`m_t = βm_{t-1} + g_t`, step `η_HB` fixed), the effective step is `η_HB/(1-β)·(1-β)⁻¹`-scaled
differently along slow directions and the small-`ηλ` conclusion inverts; the two conventions
coincide for scale-invariant (polar/normalized) updates but not for SGDM trajectories.

**Remark (what CL-2 does for the paper).** Remark (closedloop) in `latex/main.tex` currently
concedes that momentum does not damp its own deterministic transient. CL-1+CL-2 supply the
positive half: under *sustained* excitation (gradient noise — or a bending river, which
re-forces the hill mode every step, cf. E8), larger `β` strictly lowers the stationary hill
energy and extends stability, specifically at large `ηλ`. The deterministic-transient caveat
survives as the straight-and-clean special case.

---

## 3. Theorem T2 — the sampled hill-gradient spectrum  [✓ Check C; fable Claim 2]

The `β = 0` closed loop (`y_{t+1} = a y_t - ηξ_t`, `a = 1-ηλ`, `|a| < 1`) emits the gradient
stream `g_t = λ y_t + ξ_t` that a momentum filter would see. This is the stochastic
generalization of Theorem T3 (which handles the deterministic transient at `1 < ηλ < 2`).

**Statement.** In stationarity, `g` has spectral density

    S_g(ω) = σ² · |1 - e^{-iω}|² / |1 - a e^{-iω}|²
           = σ² · (2 - 2cos ω) / (1 + a² - 2a cos ω),

and consequently:

1. **(exact differencing / zero DC)** `S_g(0) = 0`. Structurally: `g_t = (y_t - y_{t+1})/η`
   exactly, so the stream is the increment process of a bounded stationary coordinate — the
   stationary instance of Lemma T3′.
2. **(monotone, always Nyquist-argmax)** `S_g` is strictly increasing on `[0, π]` for every
   `a ∈ (-1,1)`: `dS_g/d(cos ω) = -2σ²(1-a)²/(1+a²-2a cos ω)² < 0`. The peak value is
   `S_g(π) = 4σ²/(2-ηλ)²`, and the total power is `Var(g) = 2σ²/(2-ηλ)`.
3. **(high-pass with cutoff `ηλ`)** `S_g(ω) ≤ σ² ω² / min(ηλ, 2-ηλ)²`, and the half-of-peak
   point solves `cos ω_c = 2a/(1+a²)`, giving `ω_c ≈ ηλ` for small `ηλ`.
4. **(concentration dichotomy)** `S_g(π)/S_g(π/2) = 2(1+a²)/(1+a)²`, which is `≥ 2` iff
   `ηλ ≥ 1` and diverges as `ηλ → 2`: the spectrum is Nyquist-*concentrated* in the
   edge-of-stability regime and `≈` flat above the cutoff ("white above `ηλ`") for `ηλ ≪ 1`.

**Proof.** `y` is the AR(1) `y_{t+1} = a y_t - ηξ_t`, so the transfer from `ξ` to
`g = λy + ξ` is `G(z) = 1 - ηλ z⁻¹/(1 - a z⁻¹) = (1 - (a+ηλ) z⁻¹)/(1 - a z⁻¹)`, and
`a + ηλ = 1` collapses the numerator to `1 - z⁻¹`. `S_g = σ²|G(e^{iω})|²` gives the display.
(1): direct; the identity `g_t = (y_t - y_{t+1})/η` is the update rule. (2): the derivative
in `c = cos ω` shown; `S_g(π) = 4σ²/(1+a)²`; `Var(g) = λ²Var(y) + σ²` with
`Var(y) = η²σ²/(1-a²)` and `Cov(y_t, ξ_t) = 0`, giving `σ²(1 + ηλ/(2-ηλ)) = 2σ²/(2-ηλ)`.
(3): `2-2cos ω ≤ ω²` and `1+a²-2a cos ω ≥ (1-|a|)² = min(ηλ, 2-ηλ)²`; the half-peak equation
`(2-2c)/(1+a²-2ac) = 2/(1+a)²` solves linearly for `c`. (4): evaluate at `c = -1, 0`;
`2(1+a²) ≥ 2(1+a)²` ⟺ `a ≤ 0`. ∎

[✓ Check C: monotonicity and the bound (3) hold on a 20001-point grid for
`ηλ ∈ {0.3, 1.0, 1.8}`; `S(π)` and `Var(g)` match the closed forms to 4+ digits;
concentration ratios 1.03 / 2.00 / 82.0 at `ηλ = 0.3 / 1.0 / 1.8`. Band-averaged empirical
periodogram/theory ratios `1.00 ± 0.04`, DC bin `~10⁻⁶` of mean energy (fable Claim 2).]

**Corollary (exact open-loop filtered power).** The EMA-filtered stream `m = EMA_β(g)` has
stationary power ratio, by partial fractions of `H_β(z)(1-z⁻¹)/(1-az⁻¹)`,

    ρ(β; ηλ) := Var(m)/Var(g)
             = (1-β)² [ A²/(1-β²) + 2AB/(1-aβ) + B²/(1-a²) ] · (2-ηλ)/2,
    A = (1-β)/(a-β),   B = -ηλ/(a-β)        (for a ≠ β; a = β, i.e. ηλ = 1-β, is the
                                              repeated-pole case, obtained as the limit),

with limits `ρ → 1/T_eff` as `ηλ/(1-β) → 0` (hill inside the passband — no separation to
exploit) and `ρ → 1/T_eff²`-order as `ηλ → 2` (Nyquist tone). [✓ Check C: closed form matches
numerical integration of `|H_β|² S_g` to 5 digits; e.g. `β = 0.9`: `ρ = 0.0142 / 0.0053 /
0.0031` at `ηλ = 0.3 / 1.0 / 1.8`, against `1/T_eff = 0.0526`, `1/T_eff² = 0.0028`.]
Moreover `ρ ≤ 1/T_eff` for **every** `ηλ ∈ (0,2)`: `|H_β(ω)|²` is decreasing and `S_g(ω)`
increasing on `[0,π]` (T2(2)), so Chebyshev's integral inequality gives
`(1/π)∫|H_β|²S_g ≤ [(1/π)∫|H_β|²][(1/π)∫S_g] = Var(g)/T_eff`. Any hill stream with a
monotone-increasing spectrum is filtered at least as well as white noise; the white floor
`1/T_eff` of the stochastic bound is the boundary case, approached as `ηλ/(1-β) → 0`. (A
zero-DC stream with mass concentrated just above DC would filter *worse* than white — zero
DC alone buys nothing; monotonicity is the operative property.)

---

## 4. Lemma T3′ — confinement bounds DC mass  [✓ Check D]

Nearly assumption-free landscape statement: no quadratic model, no noise model, no
edge-of-stability condition. Uses only the update rule and Proposition 1 (exact windowed
transfer) at `ω = 0`.

**Statement.** Run EMA-SGDM `w_{t+1} = w_t - η m_t` (any loss, any dimension, any gradient
stream `g_t`, `m_0 = 0`) for `t = 1..T`. Then, exactly,

    ĝ(0) = Σ_{t=1}^T g_t = (w_1 - w_{T+1})/η + (β/(1-β)) m_T .                    (DC)

Project on any fixed unit vector `u`:

- **(confined ⇒ bounded DC)** if `|⟨w_t - w_1, u⟩| ≤ 2R` for all `t` (the iterate stays in a
  tube of radius `R` about a center in direction `u`), then
  `|⟨ĝ(0), u⟩| ≤ 2R/η + (β/(1-β)) |⟨m_T, u⟩|`, bounded in `T` whenever the terminal buffer
  is (for a bounded stream, `|⟨m_T, u⟩| ≤ sup_t |⟨g_t, u⟩|`).
- **(travelled ⇒ growing DC)** if `⟨w_{T+1} - w_1, v⟩ = D_T` (net displacement in direction
  `v`), then `|⟨ĝ(0), v⟩| ≥ D_T/η - (β/(1-β)) |⟨m_T, v⟩|`.

"The river is where the net displacement happens; hills are where you bounce": along confined
(hill) directions the DC bin of the gradient stream is `O(R/η)` regardless of how much energy
the stream carries, while along the travelled (river) direction it grows with the distance
covered.

**Proof.** Telescoping the update, `Σ_{t=1}^T m_t = (w_1 - w_{T+1})/η`. Proposition 1 at
`ω = 0` reads `m̂(0) = H_β(0)(ĝ(0) - B) = ĝ(0) - (β/(1-β)) m_T`, and `m̂(0) = Σ_t m_t`.
Combine and project. ∎

[✓ Check D: identity (DC) holds to `≤ 4·10⁻¹⁴` on closed-loop curved-valley runs
(noisy and clean, `β ∈ {0.5, 0.9, 0.99}`). Straight noisy valley, `β = 0.9, σ = 2`:
`|ĝ_hill(0)| = 11.0 / 8.4 / 4.2` at `T = 200 / 800 / 3200` against the tube bound
`13.9 / 11.6 / 10.8` (realized `R = 0.77`), while `|ĝ_river(0)| = 34 / 37 / 44` approaches
`D_∞/η = 8/0.18 = 44.4` as the river converges.]

**Corollary (band version, one mixing assumption).** If the hill-projected stream is weakly
stationary with `S(ω) ≤ κ ω²` on `[0, ω₀]` (a spectral autocorrelation-time assumption:
`κ ~ σ²τ²` for correlation time `τ`), the expected band mass below `ω₀` is
`(1/π)∫₀^{ω₀} S ≤ κω₀³/(3π)` against total power `Var(g)` — small whenever `ω₀ ≪ 1/τ`.
Theorem T2's stream is the concrete instance, with `κ = σ²/min(ηλ, 2-ηλ)²` exactly (T2(3)),
i.e. `τ ≍ 1/min(ηλ, 2-ηλ)`: away from the flip edge this is the hill relaxation time
`1/(ηλ)`; near `ηλ = 2` the flip mode's slow amplitude decay `|a|→1` takes over.

**Scope.** (DC) is a finite-window statement in the same frame as T1; the band corollary is
asymptotic (stationarity). The lemma says nothing about *which* directions are confined —
that is an input (from confinement of the dynamics, e.g. CL-2's variance being small against
the tube radius, or empirically from E8's tube-restoration reading).

---

## 5. Theorem T5 — deterministic filter-first for the polar map  [✓ Check E; fable Claim 3]

Matrix stream `G_t = S + (-1)^t A` in `R^{m×n}`: a fixed (or slowly varying — see the remark)
rank-`r` signal `S` with `σ_r := σ_r(S) > 0`, plus a deterministic Nyquist hill `A ≠ 0`.
`O(·)` is the polar factor; `U_S ∈ R^{m×r}`, `V_S ∈ R^{n×r}` are `S`'s top-`r` singular
subspaces. The three pipelines are pre-polar `O(M_t)` with `M_t = EMA_β(G)_t`, post-polar
`m̃_t = EMA_β(O(G))_t`, polar-only `O(G_t)`.

**(a) Exact buffer.** `M_t = (1-β^t) S + ε_t A` with

    ε_t = (-1)^t (1-β)(1-(-β)^t)/(1+β),      |ε_t| ≤ (1-β)(1+β^t)/(1+β)  →  1/T_eff .

[✓ machine precision, Check E and fable Claim 3.] The disturbance dies at the *deterministic*
rate `1/T_eff` in any unitarily invariant norm — no probability, against the `T_eff^{-1/4}`
operator-norm rate for the stochastic (BVMZOS) perturbation of li2026denoise Theorem 1.

**(b) Head, tail, gap (Weyl + interlacing).** For all `t ≥ 1`:

    σ_r(M_t)   ≥ (1-β^t) σ_r - |ε_t| ‖A‖₂ ,
    |ε_t| σ_{2r+1}(A)  ≤  σ_{r+1}(M_t)  ≤  |ε_t| ‖A‖₂ ,
    gap_r(M_t) ≥ (1-β^t) σ_r - 2 |ε_t| ‖A‖₂ .

(The tail lower bound uses `σ_{i+j-1}(X+Y) ≤ σ_i(X) + σ_j(Y)` with the rank-`r` signal as
one summand.) So the buffer's tail is pinned to the `|ε_t| → 1/T_eff` envelope from both
sides — the quantity E5's overlay plots.

**(c) Pre-polar subspace error (Wedin).** Whenever `δ_t := (1-β^t)σ_r - |ε_t|‖A‖₂ > 0`,

    max( ‖sinΘ(U_{M_t}, U_S)‖₂ , ‖sinΘ(V_{M_t}, V_S)‖₂ )
        ≤ |ε_t| · max( ‖A V_S‖₂ , ‖Aᵀ U_S‖₂ ) / δ_t
        ≤ |ε_t| ‖A‖₂ / δ_t   →   (‖A‖₂/T_eff) / (σ_r - ‖A‖₂/T_eff)   →  0  as β → 1.

Since `O(M_t)` shares `M_t`'s singular subspaces (and, by (b), `M_t`'s singular-value
ordering identifies which `r` of them carry the signal), the pre-polar update's signal
subspace inherits the bound.

**Proof of (b), (c).** Write `M_t = X + E` with `X = (1-β^t)S`, `E = ε_t A`. Weyl gives the
head bound and the tail upper bound. For the tail lower bound apply the singular-value sum
inequality `σ_{i+j-1}(P+Q) ≤ σ_i(P) + σ_j(Q)` with `P = M_t`, `Q = -X`, `i = j = r+1`:
since `σ_{r+1}(X) = 0` (rank `r`), `σ_{2r+1}(ε_t A) = σ_{2r+1}(M_t - X) ≤ σ_{r+1}(M_t)`. For
(c), apply Wedin's sinΘ theorem taking `M_t` as the base matrix and `X = M_t - E` as its
perturbation: the residuals, formed with `X`'s exact SVD (`X V_S = U_S Σ_X`,
`Xᵀ U_S = V_S Σ_X`), are `M_t V_S - U_S Σ_X = E V_S` and `M_tᵀ U_S - V_S Σ_X = Eᵀ U_S`; the
separation condition compares `X`'s signal singular values against `M_t`'s tail,
`σ_r(X) - σ_{r+1}(M_t) ≥ (1-β^t)σ_r - |ε_t|‖A‖₂ = δ_t > 0` by (b). ∎

[✓ Check E: zero violations of (b) and (c) over `t = 1..400`, `β ∈ {0.9, 0.99}`, generic
`24×18` instance; at `t = 400, β = 0.9` the bound is tight to a factor 1.2 (measured sinΘ
0.0149 vs Wedin 0.0178; the crude `‖A‖₂`-numerator form gives 0.0270).]

**(d) Polar-only misidentifies the signal subspace at exactly the disturbances pre-polar
tolerates.** The polar output `O(·)` has all singular values equal to 1, so the *only*
signal-subspace estimate available to each pipeline is the singular-value ordering of the
matrix it feeds to the map: `M_t` for pre-polar, `G_t` for polar-only (this is precisely the
buffer convention of E5's subspace metric). Take the orthogonal rank-1 instance
`S = σ u v^T`, `A = α u' v'^T` with `u ⊥ u'`, `v ⊥ v'` and `α > σ`. Then for every `t` the
top-1 singular subspace of `G_t = S + (-1)^t A` is `span(u')`: `sinΘ = 1` identically — with
no filtering stage, the signal subspace is never identified. Pre-polar identifies it
geometrically fast: the top-1 subspace of `M_t` is `span(u)` for all `t` with

    β^t < (σ - α/T̃) / (σ + α/T̃),      T̃ := (1+β)/(1-β) = T_eff,

which is non-empty iff `T_eff > α/σ`. So on this instance the pre-polar subspace error drops
from 1 to 0 by step `t* = ⌈ log((σ+α/T_eff)/(σ-α/T_eff)) / log(1/β) ⌉` (finite for
`T_eff > α/σ`, though large near that margin) while polar-only's stays 1 forever. [✓ Check E:
`α = 2σ`, `β = 0.9`: polar-only sinΘ = 1.000 at `t = 1, 10, 200`; pre-polar = 0.000 from
`t ≤ 10` (threshold predicts `t ≥ 3`).]

**Proof.** `G_t` has singular pairs `(σ, u, v)` and `(α, u', v')` by orthogonality; `α > σ`
orders them. `M_t = (1-β^t)σ uv^T + ε_t α u'v'^T` similarly; its top pair is `(u, v)` iff
`(1-β^t)σ > |ε_t| α`, and `|ε_t| ≤ (1-β)(1+β^t)/(1+β)` reduces this to the display. ∎

**Scope of (d).** In the exactly-orthogonal instance the polar *outputs* themselves are
benign — `O(G_t) = uv^T + (-1)^t u'v'^T` still carries the signal direction at coefficient 1,
so an output-level rank-`r` alignment would not detect the failure; the failure is one of
signal-subspace *identification*, which is what Muon-style low-rank reasoning (and E5's
headline metric) requires. For generic `A` the misordering contaminates the output too:
E5 measures polar-only subspace error 0.508 vs pre-polar 0.114 at `β = 0.9`, and Check E's
generic-instance alignment is 0.975 (polar-only) vs 0.9999 (pre-polar). The stochastic
analogue (polar-only alignment vanishing at low SNR as `m → ∞`) is li2026denoise's Theorem 3;
(d) is its deterministic counterpart at the subspace level.

**(e) Post-polar: exact stationary limit.** The post-polar buffer has period-2 stationary
points; with `P_± := O(S ± A)` (well-defined when `S ± A` has full rank `min(m,n)`),

    m̃_even → (P_+ + β P_-)/(1+β),      m̃_odd → (P_- + β P_+)/(1+β),      (β → 1: → (P_+ + P_-)/2),

approached geometrically. [✓ Check E: recursion meets the formula to `≤ 2·10⁻¹⁶`.] The limit
is generically **not** semi-orthogonal (in the orthogonal instance of (d), `(P_+ + P_-)/2` has
a zero singular value: the alternating component's polar factor cancels, the signal's
persists). Two consequences, matching E5: (i) the post-polar *subspace* can be nearly as good
as pre-polar's — the sign of the hill flips and averages out of the buffer; (ii) the
post-polar *output* is not orthogonal and its alignment degrades as `β → 1` because it
averages two rotated orthogonal factors whose disagreement does not vanish with `T_eff`
(`P_±` are fixed matrices independent of `β`). Pre-polar has no such floor: its input error
vanishes at `1/T_eff` *before* the nonlinearity. [✓ Check E, generic instance at `t = 400`:
alignment pre / post / polar-only = 0.9999 / 0.9744 / 0.9752 at `β = 0.9` and
1.0000 / 0.9569 / 0.9752 at `β = 0.99` — post-polar *decreasing* in `β`, pre-polar → 1.]

**Remarks.**
- **Slowly varying signal.** For `S_t` drifting in a low band, replace `(1-β^t)S` by
  `EMA_β(S)_t` and carry Theorem T1(b)'s distortion `ε_low(β)` into `δ_t`; the E5 overlay
  (drift `f_slow = 0.005`) absorbs this within its factor-2 gate.
- **Added noise.** With `G_t = S + (-1)^t A + Ξ_t`, `Ξ` zero-mean, the buffer gains
  `EMA_β(Ξ)_t`, controlled in operator norm at the `T_eff^{-1/4}`-to-`T_eff^{-1/2}` stochastic
  rates of li2026denoise Theorem 1; the deterministic part of the perturbation still dies at
  `1/T_eff`. At large `β` the *noise floor dominates* the Nyquist term (e.g. E5's
  `σ_ξ = 0.2` vs `‖ε_t A‖₂ ≈ 0.003` at `β = 0.99`), so empirical overlays must include both
  terms or use the clean scenario.
- **Scope.** This is a statement about the buffer and the direction map input — the honest
  scope for Muon-class optimizers, where the update is a function of the buffer. It does not
  claim a full nonlinear trajectory theorem (out of scope by design, fable_dgs_v1 §8.1).

---

## 6. Corollary T6′ — the principled β window

Collecting the pieces: momentum separates river from hill exactly when the EMA memory sits
between the hill relaxation time and the river traversal time,

    ημ  ≪  (1-β)/(1+β) = 1/T_eff  ≲  ηλ .

- **Lower edge** (`ημ ≪ 1/T_eff`): CL-1's slow-mode expansion (river rate `β`-independent up
  to `O(β(ημ)²/(1-β))`) *and* T1(b)'s lag distortion `ε_low ≈ βω₀/(1-β)` small at the river's
  frequencies — both fail as `T_eff` approaches the river timescale `1/(ημ)` (E2/E8's lag edge
  at `β = 0.999`).
- **Upper edge** (`1/T_eff ≲ ηλ`): T2's corollary — the filtered-power gain over white noise
  exists only when the hill band `[ω_c ≈ ηλ, π]` sits in the EMA stopband; for
  `ηλ ≪ 1/T_eff` the ratio saturates at the white floor `1/T_eff` (and CL-2's closed-loop
  ratio at `(2-ηλ)/2 ≈ 1`).
- **Prediction**: the window — hence the range of well-performing `β` — widens with the valley
  conditioning `λ/μ`. Testable on the E3 grid (E3 heatmap, λ/μ sweep).

---

## 7. Status: where each statement lands

| statement | destination in `latex/main.tex` | witness |
|---|---|---|
| CL-1 | Sec. River-Valley Dynamics, new closed-loop proposition (upgrades Remark closedloop) | E8(b) divergence split; fable Claim 1 |
| CL-2 (+ corollary) | same subsection | E8(a,b) rms guides; E3 heatmap ratio ≈ 1 |
| T2 (+ filtered-power corollary) | Sec. River-Valley Dynamics, stochastic subsection | fable Claim 2 band ratios; E3 heatmap |
| T3′ (+ band corollary) | anchor lemma, Sec. River-Valley Dynamics | Check D; E6 onset (confinement timing) |
| T5 (a)–(e) | Sec. Filter-First, argued → theorem | E5 + E5 overlay (Wedin/tail guides) |
| T6′ window | Discussion (or corollary after T2) | E3 heatmap λ/μ sweep; E2/E8 lag edge |

Convention note for the port: all constants above are in the EMA normalization; `2T-1` of
li2026denoise equals `T_eff` here (one line at first cite).
