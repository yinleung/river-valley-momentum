# Theory note: CL-3 (stationary hill loss), T5b (band-limited filter-first), B1 (boundary weight)

Staging note for the `idea_v2.md` theory items (§7 optimization-improvement proposition, §8
Theorem-4 generalization, §9 boundary term), companion to `theory_t1_t3.md` and
`theory_cl_t2_t5.md`, whose results (Proposition 1 / T1 / T3; CL-1 / CL-2 / T2 / T3′ / T5) are
used freely. Rigorous; to be ported into `latex/` under `WRITING.md` once stable. Notation as
before: EMA momentum `m_t = β m_{t-1} + (1-β) g_t`, `m_0 = 0`, `T_eff = (1+β)/(1-β)`,
`H_β(ω) = (1-β)/(1-β e^{-iω})`, high-band contraction `ρ_high(β) = |H_β(ω_c)|`, low-band
distortion `ε_low(β) = sup_{0 ≤ ω ≤ ω_0} |H_β(ω) - 1| = |H_β(ω_0) - 1|` (folded frequency;
`|H_β|` and `|H_β - 1|` are even). Every claim is flagged
[✓ verified] with the checking code in `theory_cl3_t5b_check.py` (Checks A–D; key numbers
quoted inline).

---

## 1. Proposition CL-3 — stationary hill loss decreases in β  [✓ Check A]

The translation of CL-2 from a variance statement into an optimization statement, with its
scope conditions made explicit (idea_v2 §7).

**Statement.** Consider the closed hill loop (CL) of `theory_cl_t2_t5.md` §0 — the straight
valley's hill coordinate under EMA-SGDM with white gradient noise of variance `σ²` — and fix
`η, λ > 0`. On the stable range `B(ηλ) = {β ∈ [0,1) : ηλ < 2 T_eff(β)}` (all of `[0,1)` when
`ηλ < 2`; the *open* interval `(β_c, 1)` with `T_eff(β_c) = ηλ/2` when `ηλ ≥ 2` — stability
is strict, the variance diverges as `β ↓ β_c` — with `β_c = 0` at `ηλ = 2`), the stationary
expected hill loss

    E[ (λ/2) y_∞² ] = (λ/2) Var(y_∞) = η σ² / ( 2 (2 - ηλ/T_eff) )

is **strictly decreasing in β**, from the SGD value `ησ²/(2(2-ηλ))` at `β = 0` (finite only for
`ηλ < 2`) to the saturation value `ησ²/4` as `β → 1`. The maximal relative reduction at fixed
`ηλ < 2` is

    1 - lim_{β→1} loss(β)/loss(0) = ηλ/2 ,

vanishing as `ηλ → 0` and approaching `1` as `ηλ → 2`; for `ηλ ≥ 2` the `β = 0` loop diverges
while every `β ∈ B(ηλ)` has finite loss.

**Proof.** `E[y_∞] = 0` (the mean obeys the noise-free (CL), which contracts under CL-1), so
the expected loss is `(λ/2) Var(y_∞)` with `Var(y_∞) = ησ²/(λ(2 - ηλ/T_eff))` by CL-2.
`T_eff(β) = (1+β)/(1-β)` has `dT_eff/dβ = 2/(1-β)² > 0`, so `2 - ηλ/T_eff` is strictly
increasing in β and positive exactly on `B(ηλ)`; the loss is therefore strictly decreasing
there. Endpoints: `T_eff(0) = 1` and `T_eff → ∞` give the displayed values, and
`loss(β)/loss(0) = (2-ηλ)/(2-ηλ/T_eff) → (2-ηλ)/2`. ∎

**Scope conditions (all five load-bearing).**
1. *Straight valley* (linear hill map). On a curved valley the same formula holds with the
   derived bend correction `λ → λ(1+E[f'²])` in the denominator only (E8's guide), as a local
   approximation, not a theorem.
2. *Sustained white gradient noise* entering `g_t`. Without excitation (`σ = 0`) the
   deterministic transient rings *longer* as `β → 1` (underdamped mode, modulus `√β`): the
   monotone claim is about the noise-driven stationary state, not the transient.
3. *EMA normalization at fixed η.* Under heavy-ball normalization at fixed
   `η_HB = η(1-β)`, the effective EMA step is `η = η_HB/(1-β)`, so
   `loss = η_HB σ²/(2(1-β)(2 - η_HB λ/((1-β)T_eff)))` — the numerator grows like `1/(1-β)`
   and in the small-`ηλ` regime the monotonicity *inverts* (loss increases in β). Which
   convention is held fixed is part of the claim.
4. *Stationarity.* Finite-time runs resolve the comparison only past the burn-in of CL-3′.
5. *River drift preserved to first order* (CL-1 slow mode `1 - ημ + O(β(ημ)²/(1-β))`): the
   hill comparison is at matched river progress, so lower hill loss is not purchased by
   stalling the river.

**Remark (total loss and the β-independent floor).** The full stationary excess loss of the
d-dimensional quadratic valley (CL-2 corollary) is `Σ_i ησ²/(2(2 - ηλ_i/T_eff))` over all
curvatures, *including the river's own* `λ_i = μ`: that term is `ησ²/(2(2-ημ/T_eff))
= ησ²/4 (1 + O(ημ))`, β-independent to first order. So the measurable tail loss decreases
strictly in β **onto a β-independent noise floor** — `≈ ησ²/2` in the two-coordinate valley
(one river, one hill: the E9 setup; in d dimensions, the sum above with the saturated hill
terms) — and the improvement lives entirely in the hill term. This is the quantity E9
measures.

---

## 2. Remark CL-3′ — finite-time burn-in of the stationary comparison  [✓ Check B]

**Statement.** Write (CL) in state-space form `z_t = (y_t, y_{t-1})ᵀ`,
`z_{t+1} = A z_t - η(1-β) ξ_t e₁` with companion matrix `A = [[1+β-e, -β], [1, 0]]`,
`e = η(1-β)λ`. The second moment `Σ_t = E[z_t z_tᵀ]` obeys the *linear* recursion
`Σ_{t+1} = A Σ_t Aᵀ + η²(1-β)²σ² e₁e₁ᵀ`, so its deviation from the stationary `Σ_∞`
propagates as `Δ_{t+1} = A Δ_t Aᵀ` and

    |E[y_t²] - Var(y_∞)| ≤ ‖A^t‖² ‖Σ_0 - Σ_∞‖ ≤ κ² r(A)^{2t} ‖Σ_0 - Σ_∞‖ ,

with `r(A)` the spectral radius and `κ` the eigenvector condition number (finite whenever the
roots are distinct). In the complex-root (underdamped) regime `r(A) = √β` exactly (CL-1), so
the second moment converges geometrically with ratio `r(A)² = β`: the ε-burn-in horizon is

    t_ε = log(κ²‖Σ_0-Σ_∞‖/ε) / log(1/β) ≈ (T_eff/2) · log(κ²‖Σ_0-Σ_∞‖/ε)   (β → 1).

**Consequences.** (i) A run of length `T` resolves the stationary comparison of CL-3 only for
those β with `T_eff ≪ T`; at fixed `T` the measured tail loss turns back *up* as `β → 1` once
the burn-in eats the window — the finite-horizon caveat to the monotone claim, and the reason
E9 reports `T/T_eff` per cell. (ii) The same `β^t` rate appears in the deterministic
initial-condition transient (CL-1's `√β` per step, `β` per step in energy): burn-in and
underdamped ringing are the same clock. [✓ Check B: exact-moment-recursion envelope decay
rate 0.505 / 0.899 at β = 0.5 / 0.9 (vs β itself), fixed point equal to the CL-2 variance to
1e-9; ensemble at β = 0.99, T = 6000: tail-half mean within 0.5% of Var(y_∞) while the first
T_eff steps average 54× the stationary value.]

---

## 3. Theorem T5b — band-limited filter-first  [✓ Check C]

Generalizes T5 (`theory_cl_t2_t5.md` §5) from the exact Nyquist disturbance `(-1)^t A` to any
band-limited disturbance, and adds a slowly varying signal (idea_v2 §8). The tool is an exact
per-tone identity replacing the T5 closed form.

**Lemma (per-tone EMA response).** For the complex tone `g_t = e^{iωt} V` (`t ≥ 1`, any fixed
matrix `V`, `m_0 = 0`),

    m_t = H_β(ω) ( e^{iωt} - β^t ) V        for every t ≥ 1.

*Proof.* `m_t = (1-β) Σ_{u=1}^t β^{t-u} e^{iωu} V = (1-β) e^{iωt} Σ_{j=0}^{t-1}(β e^{-iω})^j V
= (1-β) e^{iωt} (1-(βe^{-iω})^t)/(1-βe^{-iω}) V`, and
`e^{iωt}(βe^{-iω})^t = β^t`. ∎ [✓ machine precision]

At `ω = π` this recovers T5's `ε_t`: `H_β(π)((-1)^t - β^t) = (-1)^t (1-β)(1-(-β)^t)/(1+β)`.
At `ω = 0` it recovers the constant-signal shrinkage `(1-β^t)`.

**Model.** Let `Ω ⊂ (-π, π]` be a finite signed frequency set closed under negation, and

    G_t = S + P_t + A_t,
    P_t = Σ_{ω ∈ Ω_low}  e^{iωt} C_ω   (slow drift,   Ω_low ⊆ ±(0, ω_0], C_{-ω} = conj(C_ω)),
    A_t = Σ_{ω ∈ Ω_high} e^{iωt} D_ω   (disturbance,  Ω_high ⊆ ±[ω_c, π], D_{-ω} = conj(D_ω)),

with `S` constant of rank `r`, `σ_r = σ_r(S) > 0`, `0 < ω_0 < ω_c ≤ π`, and amplitudes
`a = Σ_{Ω_low} ‖C_ω‖₂`, `α = Σ_{Ω_high} ‖D_ω‖₂` (sums over the signed sets; all streams real
by the pairing, negation taken modulo `2π` so that `ω = π` is self-conjugate with a real
coefficient). **In-subspace drift assumption** for part (c): `col(C_ω) ⊆ col(S)` and
`row(C_ω) ⊆ row(S)` for every ω — the slow signal wanders inside a fixed rank-r subspace
pair, the object Muon-style low-rank readings track.

**Statement.** Write `S_t = S + P_t` (the instantaneous slow signal). The buffer decomposes
exactly, by linearity and the Lemma, as

    M_t = X_t + Ã_t,    X_t = (1-β^t) S + Σ_{Ω_low} H_β(ω)(e^{iωt}-β^t) C_ω,
                         Ã_t =            Σ_{Ω_high} H_β(ω)(e^{iωt}-β^t) D_ω,

and for every `t ≥ 1`:

  (a) *(disturbance contraction)*  `‖Ã_t‖₂ ≤ ρ_high(β)(1+β^t) α → ρ_high(β) α`, against the
      raw level `‖A_t‖₂ ≤ α` (for the single Nyquist tone `‖A_t‖₂ = α` at every `t`; for
      generic multi-tone disturbances both amplitude sums are conservative, by the same
      triangle inequality on each side). The contraction is by the transfer-function factor
      `ρ_high`, not the Nyquist floor `1/T_eff`: T5 is the instance `Ω_high = {π}`, where
      `ρ_high = 1/T_eff`.

  (b) *(slow-signal fidelity)*  `‖X_t - S_t‖₂ ≤ β^t ‖S‖₂ + (ε_low(β) + β^t) a`, so
      `‖M_t - S_t‖₂ ≤ β^t(‖S‖₂ + a) + ε_low(β) a + ρ_high(β)(1+β^t) α`: the buffer tracks the
      instantaneous slow signal with error `ε_low·a + ρ_high·α` past burn-in.

  (c) *(subspace identification, in-subspace drift)*  `X_t` has `col(X_t) ⊆ col(S)`,
      `row(X_t) ⊆ row(S)`, `rank(X_t) ≤ r`, and `σ_r(X_t) ≥ (1-β^t)σ_r - (1+β^t)a`. Whenever

          δ_t := (1-β^t) σ_r - (1+β^t) a - ρ_high(1+β^t) α > 0,

      `rank(X_t) = r` (so its subspaces *equal* the signal's),
      `σ_r(M_t) ≥ σ_r(X_t) - ‖Ã_t‖₂ ≥ δ_t`, while
      `σ_{r+1}(M_t) ≤ σ_{r+1}(X_t) + ‖Ã_t‖₂ = ‖Ã_t‖₂ ≤ ρ_high(1+β^t)α`; the singular-value
      ordering of `M_t` identifies the signal subspace pair as soon as `δ_t > ρ_high(1+β^t)α`,
      and Wedin's sinΘ theorem (applied as in T5, base `M_t`, perturbation `-Ã_t`) gives

          max( ‖sinΘ(U_{M_t}, U_S)‖₂, ‖sinΘ(V_{M_t}, V_S)‖₂ ) ≤ ρ_high(1+β^t) α / δ_t .

      At fixed β, as `t → ∞` the bound tends to `ρ_high α / (σ_r - a - ρ_high α)` whenever
      that denominator is positive. If `σ_r > a`, taking `β → 1` *after* `t → ∞` (concretely
      `t ≫ T_eff · log`, so the `β^t` factors clear first) sends `ρ_high(β) → 0` at fixed
      `ω_c > 0` and the bound to `0`: the T5(b) conclusion at rate `ρ_high(β)` in place of
      `1/T_eff`. For fixed `t` the bound instead degrades as `β → 1` (`δ_t < 0` once `β^t`
      is order one) — the burn-in ordering is part of the statement, exactly as in T5.

**Proof.** The decomposition is the Lemma applied per tone. (a): triangle inequality,
`|H_β(ω)| ≤ ρ_high` on `±[ω_c, π]` (T1's monotonicity, `|H_β|` even), `|e^{iωt} - β^t| ≤
1 + β^t`. (b): `X_t - S_t = -β^t S + Σ_low [H_β(ω)(e^{iωt}-β^t) - e^{iωt}] C_ω`, and
`|H(e^{iωt}-β^t) - e^{iωt}| ≤ |H-1| + |H|β^t ≤ ε_low + β^t` on `±(0, ω_0]`. (c): the range
inclusions are immediate from the assumption; `σ_r(X_t) ≥ σ_r((1-β^t)S) - ‖X_t - (1-β^t)S‖₂`
with `‖Σ_low H(e^{iωt}-β^t)C_ω‖ ≤ (1+β^t)a` (`|H| ≤ 1`); rank-r equality because a rank-<r
matrix with range in an r-dimensional pair would have `σ_r = 0`. Weyl gives the head/tail
displays (`σ_{r+1}(X_t) = 0` since `rank X_t ≤ r`). Wedin as in T5(b): with `X_t = M_t - Ã_t`
as the comparison matrix, the residuals are `M_t V_X - U_X Σ_X = Ã_t V_X` and
`M_tᵀ U_X - V_X Σ_X = Ã_tᵀ U_X`, both `≤ ‖Ã_t‖₂` in norm, and the separation is
`σ_r(X_t) - σ_{r+1}(M_t) ≥ δ_t`. ∎

**Corollary (signal-to-disturbance gain; the filter-first constant).** Assume `α > 0` (the
zero-disturbance case is trivial). Define the amplitude signal-to-disturbance ratios
`SDR_in = σ_r(S)/α` (the raw stream's, worst case over the period) and
`SDR_out(t) = σ_r(X_t)/‖Ã_t‖₂`. Then

    SDR_out(t) / SDR_in ≥ ( (1-β^t) - (1+β^t) a/σ_r ) / ( ρ_high (1+β^t) )
                        →  (1 - a/σ_r) / ρ_high        (t → ∞),

positive in the limit iff `σ_r > a`. This is the proved form of the gain: the disturbance
side is contracted by the T1 constant `ρ_high`, the signal side is retained up to the drift
amplitude, and the `β^t` factors are the burn-in terms. The distortion constant `ε_low`
enters through (b) — it controls how far the retained slow part is from the *instantaneous*
signal — not through this amplitude bound; an `(1-ε_low)/ρ_high` form would describe the
drift-dominated reading (signal scale carried by `P_t` rather than `S`), which we do not
prove. The paper quotes the amplitude form.
[✓ Check C: exact decomposition and all bounds hold at every step on a random 24×18, r=3
instance (a=0.2, α=3.3, β=0.9), the Wedin bound checked for both left and right subspaces;
measured gain exceeds the bound by a median ×3.1 and the Wedin bound is loose by a median
×8.0 late in the run — the triangle-inequality amplitude sums Σ‖D_ω‖ are conservative for
multi-tone disturbances whose phases rarely align, against ×1.17 in T5's single-tone case.]

**Remarks.**
- *Relation to T5.* T5 is `Ω_low = ∅`, `Ω_high = {π}` (single self-conjugate tone): (a)
  reduces to the `|ε_t|` envelope (with the exact two-sided tail T5 adds via per-tone
  exactness), (c) to T5(b). The polar-only misidentification instance (T5's Proposition) is a
  single-tone instance of this model, so the pre-polar vs polar-only separation carries over
  verbatim: for `α > σ_r` the raw top subspace is the disturbance's at every t, while `M_t`'s
  is the signal's once `β^t` clears the threshold.
- *What is genuinely assumed.* Band-limitedness with a spectral gap `(ω_0, ω_c)` and, for the
  subspace claim, drift confined to the signal subspaces. Without the in-subspace assumption,
  (a)/(b) still hold and the subspace statement degrades gracefully: `P_t` contributes to the
  perturbation instead, replacing `ρ_high(1+β^t)α` by `ρ_high(1+β^t)α + (1+β^t)a` in the
  numerator — filtering then cannot remove the drift, correctly, since it sits in the passband.
- *Not covered:* stochastic disturbances (T5b is deterministic; the stochastic rate is
  li2026denoise's `(2T-1)^{-1/4}`), and streams whose "slow" part moves its subspace on the
  fast time scale.

---

## 4. Remark B1 — the boundary term's relative weight  [✓ Check D]

Proposition 1 (theory_t1_t3.md) is exact: `m̂(ω_ℓ) = H_β(ω_ℓ)(ĝ(ω_ℓ) - e^{-iω_ℓ(T+1)} B)`
with `B = β m_T/(1-β)`. The paper says B is "small once the filter has settled"; this remark
makes the claim quantitative and states when it fails (idea_v2 §9, Option A).

**Statement.**
(i) *(persistent stream, DC bin)* For the constant stream `g_t ≡ ḡ ≠ 0`:
`ĝ(0) = T ḡ` while `‖B‖ = β(1-β^T)/(1-β) ‖ḡ‖ ≤ (T_eff/2)‖ḡ‖·(1+O(1/T_eff))`, so the relative
boundary weight at DC is

    ‖B‖/‖ĝ(0)‖ ≤ β/((1-β) T) ≈ T_eff/(2T) .

(ii) *(stationary stream, any bin)* For a zero-mean wide-sense-stationary *scalar* stream
(one coordinate of the gradient; the DFT identity is per-coordinate) with absolutely
summable autocovariance and spectral density `S(ω)`:
`E|B|² = (β/(1-β))² E|m_T|² ≤ (β/(1-β))² Var(g)` (the EMA is a power contraction, `|H_β| ≤ 1`),
a constant in `T`, while `E|ĝ(ω_ℓ)|² = T·S(ω_ℓ)(1+o(1))` grows linearly. Hence on every bin
with `S(ω_ℓ)` bounded away from zero,

    E|B|² / E|ĝ(ω_ℓ)|² = O( (β/(1-β))² Var(g) / (T S(ω_ℓ)) ) = O(T_eff²/T) → 0 .

(iii) *(failure mode)* Both ratios are controlled by `T/T_eff`: for windows shorter than a few
`T_eff` the boundary term is *not* negligible — this, not persistence per se, is the regime
where the finite-window identity must be used in full. Experimental discipline: declare the
window type, use a burn-in of at least a few `T_eff` (equivalently report `T/T_eff`), and for
spectra use the exact identity or a taper.

**Proof.** (i) `m_T = (1-β)Σ β^{T-u} ḡ = (1-β^T)ḡ`, and `β(1-β^T)/(1-β) ≤ β/(1-β) ≤ T_eff/2`.
(ii) The first equality is the definition of B; `Var(m) = (2π)^{-1}∫|H_β|²S ≤ Var(g)`;
`E|ĝ(ω_ℓ)|² = Σ_{t,s} γ(t-s) e^{-iω_ℓ(t-s)} = T Σ_{|h|<T}(1-|h|/T)γ(h)e^{-iω_ℓ h} → T·S(ω_ℓ)`
under summability (Cesàro). (iii) is (i)+(ii) read at `T ≍ T_eff`. ∎

Note the sharpening over the qualitative claim: nothing needs the stream to be small or the
filter "settled" in a trajectory sense — persistence is harmless (i); only short windows
relative to the memory are not.
