# Theory note: FR (closed-loop forced response), CV (curved-valley reduction), BE (band-energy filter-first)

Staging note for the `review_v3.md` theory items — the falsification-control machinery
(review "ablations that can falsify the mechanism"), the curved-valley local theorem (review
weakness 3), and the band-energy form of the band-limited theorem (review weakness 4 /
"make Theorem 5 more realistic"). Companion to `theory_t1_t3.md`, `theory_cl_t2_t5.md`,
`theory_cl3_t5b.md`, whose results are used freely. Notation as before: EMA momentum
`m_t = β m_{t-1} + (1-β) g_t`, `m_0 = 0`, `T_eff = (1+β)/(1-β)`,
`H_β(ω) = (1-β)/(1-β e^{-iω})`, `ρ_high(β) = |H_β(ω_c)|`, closed hill loop (CL)
`y_{t+1} = (1+β-e) y_t - β y_{t-1} - η(1-β) ξ_t`, `e = η(1-β)λ`, characteristic polynomial
`p(z) = z² - (1+β-e) z + β`. Every claim is flagged [✓ verified] with the checking code in
`theory_fr_cv_be_check.py` (Checks A–C; key numbers quoted inline).

---

## 1. Proposition FR — closed-loop forced response  [✓ Check A]

**Statement.** Add a deterministic forcing `s_t` to the hill-gradient noise in (CL), i.e.
the observed hill gradient is `λ y_t + ξ_t + s_t`. For the tone `s_t = A cos(ω t)` with
`ηλ < 2 T_eff`, the stationary response is the tone `A |G_β(ω)| cos(ω t + ψ)` with

    |G_β(ω)| = η(1-β) / |p(e^{iω})| ,

and (superposition) the stationary hill loss under tone plus white noise is
`E[(λ/2) y²] = (λ/2)( Var(y_∞) + A² |G_β(ω)|² / 2 )` (Var from CL-2; at `ω = π` the ± tone
pair coincides and the `/2` becomes `1`). Three frequencies organize the response:

1. **DC.** `|G_β(0)| = η(1-β)/p(1) = η(1-β)/e = 1/λ` for every β: a constant (or passband)
   hill-gradient bias displaces the stationary point by `bias/λ` regardless of momentum.
   Filtering cannot remove what sits in the passband — the closed loop converges to the
   *shifted* optimum at the same offset for every β.
2. **Nyquist.** `|G_β(π)| = η(1-β)/(2(1+β) - e)`, strictly decreasing in β; for `e ≪ 1`
   this is `(η/2)|H_β(π)|(1+O(e))` — the closed loop inherits the open-loop filter
   attenuation `1/T_eff`, so the forced hill *loss* at ω = π falls like `1/T_eff²`.
3. **Resonance.** In the complex-pole regime (`(1+β-e)² < 4β`) the poles are
   `√β e^{±iθ_β}`, `cos θ_β = (1+β-e)/(2√β)`, and at the hill-mode frequency

       |G_β(θ_β)| = η(1+√β) / sqrt( (1-√β)² cos²θ_β + (1+√β)² sin²θ_β )
                  = sqrt( η/(λ(1-β)) ) · (1+o(1))    (β → 1, θ_β ≈ sqrt(ηλ(1-β)) → 0),

   which **diverges** as β → 1: momentum amplifies disturbances at the frequency of its own
   underdamped hill mode, and the forced hill loss there grows like `η A² T_eff/8`. Measured
   amplification over β = 0 at ηλ = 1.8: 4.2× (β = 0.9), 13.4× (0.99), 42.4× (0.999)
   [Check A2]; the asymptote matches to four digits at β = 0.99 (1.3420 vs 1.3416).

**Proof.** z-transform of (CL) with input `s`: `Y(z) = -η(1-β) z S(z) / p(z)`, so the gain
at `z = e^{iω}` is the display (|z| = 1); stability makes the transient decay. DC:
`p(1) = e`. Nyquist: `p(-1) = 2(1+β) - e`. Resonance: factor
`|p(e^{iθ})| = |e^{iθ} - √β e^{iθ}| · |e^{iθ} - √β e^{-iθ}|
= (1-√β) sqrt((1-√β)² cos²θ + (1+√β)² sin²θ) / (1-√β)`… computing directly,
`|e^{iθ} - √β e^{-iθ}|² = (1-√β)² cos²θ + (1+√β)² sin²θ`, and `1-β = (1-√β)(1+√β)` cancels
the near-pole distance. Asymptotics: `e → 0` gives `cos θ_β = 1 - ηλ(1-β)/2 + O((1-β)²)`,
so `θ_β = sqrt(ηλ(1-β))(1+o(1))` and the display reduces to `η·2/(2 sin θ_β)(1+o(1))`. ∎

Verified: lock-in amplitude vs `A|G_β(ω)|` over β ∈ {0, 0.5, 0.9, 0.99} × four frequencies,
worst rel. err 8×10⁻⁴ [Check A1]; loss additivity, measured 0.1876 vs predicted 0.1891 over
8 seeds [Check A3].

**Reading.** FR is the quantitative form of the open-loop/closed-loop remark: the closed
loop attenuates *high-frequency* exogenous content (Nyquist gain ↓ in β), is *neutral* to
passband content (DC gain 1/λ, β-free), and *amplifies* content at its own underdamped
mode frequency θ_β (unboundedly as β → 1). It also fixes the correct negative controls for
experiments: momentum should (i) remove a Nyquist tone at rate ~1/T_eff², (ii) fail to
remove a passband tone at any β, (iii) *hurt* at ω = θ_β. E13 runs exactly these arms with
the parameter-free guides.

---

## 2. Proposition CV — curved-valley reduction  [✓ Check B]

Curved valley `L(x,y) = φ(x) + (λ/2)(y - f(x))²`, hill offset `δ_t = y_t - f(x_t)`,
`λ_loc(x) = λ (1 + f'(x)²)` (the local hill sharpness).

**(a) Exact GD one-step identity.** For gradient descent, with
`Δx_t = -η(φ'(x_t) - λ δ_t f'(x_t))` the river step,

    δ_{t+1} = (1 - η λ_loc(x_t)) δ_t + η f'(x_t) φ'(x_t) - (1/2) f''(ζ_t) Δx_t² ,

for some `ζ_t` between `x_t` and `x_{t+1}` (exact; Lagrange form). The remainder is
`≤ (1/2) max|f''| Δx²` — second order in the river step. Verified along a 400-step
trajectory on `f = 2 sin(0.9 x)` [Check B1].

**(b) Momentum, frozen coefficients.** Under EMA momentum the same algebra gives
`δ_{t+1} = δ_t - η [m_t^y - f'(x_t) m_t^x] - (1/2) f''(ζ_t) (η m_t^x)²`, and if `f'` is
frozen at `f'_0` over the filter memory, `m^y - f'_0 m^x = EMA_u[ λ_loc δ_u - f'_0 φ'(x_u)
+ (ξ^y_u - f'_0 ξ^x_u) ]`: the hill offset obeys **exactly the straight-valley closed loop
(CL) with `λ → λ_loc = λ(1+f'_0²)`, exogenous forcing `s_u = -f'_0 φ'(x_u)`, and noise
variance `σ²(1+f'_0²)`**. For a linear floor `f = c x` (where `f'' = 0` and `f'` is
constant) the reduction is exact, and CL-2/FR predict, for every β,

    E[δ_∞] = c φ' / λ_loc          (the tracking offset: FR's DC gain, β-free)
    Var(δ_∞) = η σ² / ( λ (2 - η λ_loc / T_eff) ) .

Verified on the tilted floor (c = 1.5, φ' ≡ 0.8): measured mean +0.03702 vs predicted
+0.03692 at every β ∈ {0, 0.5, 0.9, 0.95}; variance ratio to prediction 0.95–1.00
[Check B2].

**(c) Consequences (the two window edges, now with constants).**
1. *Confinement floor.* Frozen-coefficient stability requires `η λ_loc < 2 T_eff`
   pointwise; the worst case on `f = a sin(kx)` is `η λ (1 + (ak)²) < 2 T_eff`, so the
   smallest admissible β **rises with the bend frequency** through the slope range `ak`.
   Against E10's measured floors (run `bend_window_sweep_k5_el2_b10_seeds16`): the
   prediction is exact in 7 of 10 (ηλ, k) cells and one grid step off in 3, with
   deviations in both directions — the prediction is a deterministic stability threshold,
   the measurement a noise-driven escape criterion [Check B3].
2. *Tracking offset.* The geometric forcing `f' φ'` enters at the river's temporal
   frequencies (≈ `kv` and harmonics for `f = a sin(kx)` at river speed v) — passband
   content. By FR(1), its response `f'φ'/λ_loc` is β-free: **momentum removes the
   self-generated Nyquist bounce but not the geometric tracking offset**; curvature is paid
   at every β.
3. *Validity / top edge.* Freezing `f'` over the memory needs the coefficient drift per
   `T_eff` steps to be small: `T_eff · v · max|f''| ≪ max(1, |f'|)` — for the sinusoid,
   `k v T_eff ≪ 1` up to the slope scale. When `k v T_eff ≳ 1` the reduction (and the
   buffer's tracking of the bend) fails: the lag edge of the β window, with E10's measured
   self-regulation of v closing the loop.

**Scope.** (b) is a frozen-coefficient reduction, not a global theorem: on a genuinely
curved floor it is first order in δ and in the coefficient drift, with the f''-remainder of
(a). The straight-valley formulas it imports carry their own scope (white state-independent
noise, stationarity, EMA normalization at fixed η).

---

## 3. Corollary BE — band-energy filter-first  [✓ Check C]

Replaces T5b's finite tone sums by windowed band *energies* — the hypothesis the HFER
diagnostic actually estimates (review: "finite-window band-energy theorem").

**Setup.** `G_t = S + A_t` (t = 1..T), `S` constant of rank r with `σ_r = σ_r(S) > 0`;
`A_t` an arbitrary real matrix stream with windowed DFT `Âₗ = Σ_t A_t e^{-iω_ℓ t}`,
`ω_ℓ = 2πℓ/T`. Fix `ω_c` and split bins by folded frequency: `E_high = (1/T) Σ_{f_ℓ ≥ ω_c}
‖Âₗ‖_F²`, `E_low = (1/T) Σ_{f_ℓ < ω_c} ‖Âₗ‖_F²` (DC included in low). Parseval:
`E_high + E_low = Σ_t ‖A_t‖_F² =: E_A`. Leakage fraction `γ = E_low / E_A` (so `1-γ` is the
high-band energy fraction; HFER estimates it, up to HFER's DC-exclusion, which only raises
HFER). Score the window `t ∈ (T₀, T]`.

**Exact response.** By the per-tone identity and the inverse DFT (`A_u = (1/T) Σ_ℓ Âₗ
e^{iω_ℓ u}`, u = 1..T),

    Ã_t := EMA_β(A)_t = Φ_t - β^t Ψ ,   Φ_t = (1/T) Σ_ℓ H_β(ω_ℓ) e^{iω_ℓ t} Âₗ ,
                                        Ψ  = (1/T) Σ_ℓ H_β(ω_ℓ) Âₗ ,

verified to 8×10⁻¹⁴ relative error [Check C1]. Orthogonality of `{e^{iω_ℓ t}}_t` gives
`Σ_{t=1}^T ‖Φ_t‖_F² = (1/T) Σ_ℓ |H(ω_ℓ)|² ‖Âₗ‖_F² ≤ ρ_high² E_high + E_low`, and
Cauchy–Schwarz per band gives `‖Ψ‖_F ≤ ρ_high √E_high + √E_low`.

**Statement.** Let

    ρ̄(β, γ, T₀) = sqrt(ρ_high² + γ) + β^{T₀+1} (ρ_high + √γ) / sqrt(1-β²) .

(i) *Energy contraction.* `sqrt( Σ_{t>T₀} ‖Ã_t‖_F² ) ≤ ρ̄ √E_A`, against raw disturbance
energy `√E_A`.
(ii) *Most steps.* For any θ ∈ (0,1), on at least `(1-θ)(T-T₀)` scored steps,
`‖Ã_t‖_F ≤ ρ̄ sqrt( E_A / (θ (T-T₀)) )` (Markov over the scored window).
(iii) *Subspace identification.* On those steps, with `b = ρ̄ sqrt(E_A/(θ(T-T₀)))`:
`σ_{r+1}(M_t) ≤ b`, `σ_r(M_t) ≥ (1-β^t)σ_r - b`, and whenever the gap
`δ_t = (1-β^t)σ_r - b > 0`, Wedin gives
`max(‖sinΘ(U_{M_t},U_S)‖₂, ‖sinΘ(V_{M_t},V_S)‖₂) ≤ b / δ_t`.

**Proof.** (i): triangle inequality in ℓ²(t; Frobenius) over the two response parts;
`Σ_{t>T₀} β^{2t} ≤ β^{2(T₀+1)}/(1-β²)`; the two displays above. (ii): Markov. (iii): Weyl
and Wedin exactly as in T4/T5b, with `X_t = (1-β^t)S` (`σ_{r+1}(X_t) = 0`). ∎

Verified over β ∈ {0.5, 0.9, 0.99} × γ ∈ {0, 0.05, 0.3}: all bounds hold, energy-bound
tightness ratio 0.04–0.63, exceptional fraction 0 at θ = 0.1 [Check C2].

**Reading and price.** The hypotheses are now *measured* quantities: `1-γ` is the
disturbance stream's high-band energy fraction and `√(E_A/T)` its rms amplitude. Two prices
against T5b's tone form, both structural: (a) per-step control degrades to most-steps
control (a lone spike carries vanishing energy — no energy hypothesis can control every
step); (b) the crest factor `1/√θ`. Stochastic disturbances now enter through their
*realized* spectra: a white stream has `1-γ ≈` (high-band bin fraction), correctly
recovering only the `1/T_eff`-type power attenuation, not band elimination. T5b remains the
sharp statement when the disturbance really is a few tones (uniform in t, no crest factor);
BE is the statement whose hypothesis HFER can certify.

---

## Porting map (latex/main.tex)

- FR → new Proposition (forced response) in the River-Valley Dynamics section after the
  burn-in remark; E13 section runs the three arms with exact guides; feeds the E13 rows of
  the theory-to-experiment table.
- CV → new Proposition (curved-valley reduction) replacing the heuristic δ-recursion
  paragraph (which drops the (1+f'²) factor that E8/E10 already use); E10 text gains the
  floor-prediction comparison (7/10 exact, 10/10 within one grid step); E8's bend
  correction now cites the proposition instead of "derived".
- BE → new Corollary (band-energy filter-first) after the SDR remark in the Filter-First
  section; the remark's "stochastic disturbances are not covered" is rescoped (covered
  through realized band energies, with the crest-factor price); pairs with E12's measured
  HFER of the state stream.
