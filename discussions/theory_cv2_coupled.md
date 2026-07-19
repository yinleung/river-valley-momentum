# Theory note: T1 — the coupled curved valley with linear floor is exactly two straight loops

Repair of Prop 5(b) per `review_v5.md` §2.2 and `plan_v5.md` T1. The reviewer is right that for
the quadratic river profile the forcing `−c φ'(x_t)` is stochastic through `x_t`, so the current
"nothing is frozen or truncated: the reduction is exact" is false. This note replaces it with a
three-tier statement whose exact tier is *stronger* than what the paper claimed: the coupled
system diagonalizes exactly over the loss Hessian, for every β. Paired check:
`theory_cv2_coupled_check.py` (sympy + numpy). Status flags [✓] refer to that script.

**Correction to the pre-check (2026-07-19).** `plan_v5_t1_precheck.py` used the wrong sign on
the off-diagonal of its β=0 noise covariance (`+c` where `Cov(ξ^d, ξ^x) = −cσ²`), and its
conclusion that the reviewer's displayed β=0 formula "matches none of three natural noise
placements" is therefore wrong. With the correct sign, **the reviewer's β=0 formula is exactly
right under the paper's per-coordinate iid convention** [✓ A3b], and the modal formula below
reproduces it at β=0 [✓ A3a] while extending it to every β. The plan_v5 §2 T1 paragraph and
its response-letter framing are superseded accordingly.

---

## 0. Setup

Linear-floor valley
`L(x,y) = φ(x) + (λ/2)(y − cx)²`, hill offset `d = y − cx`, local sharpness
`λ_loc = λ(1+c²)`, throughout `0 < μ < λ` and `β ∈ [0,1)`. EMA momentum per coordinate with
per-coordinate white noise, exactly the closed-loop conventions of the paper
(`m_t = β m_{t−1} + (1−β)(∇L(w_t) + ξ_t)`, `w_{t+1} = w_t − η m_t`, `ξ_t` iid,
`Cov(ξ) = σ² I₂` in `(x,y)`).

Two river profiles are distinguished:

- **tilted profile**: `φ' ≡ G` constant (constant-slope river potential);
- **quadratic profile**: `φ(x) = (μ/2)(x − x*)²` (the paper's standing choice), WLOG `x* = 0`.

## 1. Tier (i) — tilted profile: the current formulas are exact verbatim  [✓ B3]

With `φ' ≡ G` the `d`-combination of the gradient stream is
`g^d := g^y − c g^x = λ_loc d − cG + ξ^d`, `ξ^d = ξ^y − c ξ^x`, `Var(ξ^d) = (1+c²)σ²` —
no `x` enters. So `d` obeys the paper's closed loop (eq. closedloop) at curvature `λ_loc`,
noise variance `(1+c²)σ²`, and constant gradient bias `−cG`:

    E[d_∞]  = cG/λ_loc,
    Var(d_∞) = η·σ²(1+c²) / (λ_loc (2 − η λ_loc/T_eff)) = η σ² / (λ (2 − η λ_loc/T_eff)),

exactly the displayed formulas of the current Prop 5(b). Their honest scope is this tier.

## 2. Tier (ii) — quadratic profile: exact modal solution  [✓ A1–A6, B1–B2]

`L` is a quadratic form with Hessian (in `(x,y)`, constant everywhere)

    H = [[μ + λc², −λc], [−λc, λ]],    tr H = μ + λ_loc,   det H = μλ.

Let `ν₁ ≥ ν₂ > 0` be its eigenvalues and `ê₁, ê₂` the orthonormal eigenvectors:

    ν_{1,2} = ( (μ+λ_loc) ± √( (λ_loc−μ)² + 4c²λμ ) ) / 2 ,

real and distinct whenever `c ≠ 0` or `λ ≠ μ` (the discriminant is `(λ_loc−μ)² + 4c²λμ ≥ 0`),
with the exact product/sum `ν₁ν₂ = μλ`, `ν₁+ν₂ = μ+λ_loc`, and `ν₁ ≥ max(λ_loc, μ)`.

**Modal reduction (exact, every β).** In the coordinates `p = Qᵀ(w − w*)` (`H = Q N Qᵀ`, `Q`
orthogonal) the EMA-momentum loop separates into two independent copies of the paper's scalar
closed loop, mode `i` at curvature `ν_i`, driven by the rotated noise `Qᵀξ` which is again iid
with `Cov = σ² I₂` (isotropy is rotation-invariant). Nothing is frozen or truncated *here*: for
a quadratic loss the loop is linear and commutes with the Hessian's eigenprojections.

Consequences, all exact:

- **Stability** iff `η ν₁ < 2 T_eff` [✓ A6, B2]. Since `ν₁ ≥ λ_loc` with
  `ν₁ = λ_loc + μ c²/(1+c²) + O(μ²/λ)`, the pointwise frozen threshold `η λ_loc < 2T_eff` is
  *necessary but not sufficient*: the exact boundary is stricter by an explicit `O(ημ)` margin.
- **Mean**: `E[d_∞] = 0` and `E[x_t] → x*` (the noise-free mean recursion contracts to the
  minimum). The nonzero tracking offset `c φ'/λ_loc` belongs to tiers (i) and (iii), where the
  river pull is persistent — at stationarity under the quadratic profile it has decayed.
- **Variance**: `d = y − cx = ⟨(−c,1), w⟩` mixes the two modes with weights
  `b_i = ⟨(−c,1), ê_i⟩`:

      Var(d_∞) = b₁² · η σ²/(ν₁ (2 − η ν₁/T_eff))  +  b₂² · η σ²/(ν₂ (2 − η ν₂/T_eff)),

  with **no cross term** (independent modal noises), and

      b₁² = (λ_loc − ν₂)² / ((λ − ν₂)(ν₁ − ν₂)),
      b₂² = (ν₁ − λ_loc)² / ((ν₁ − λ)(ν₁ − ν₂)),
      b₁² + b₂² = 1 + c²   [✓ A5],

  using the identities `(ν₁−λ)(λ−ν₂) = c²λ²` and `(ν₁−λ_loc)(λ_loc−ν₂) = c²λμ` [✓ A5].
  The displayed weight quotients require `c ≠ 0` (at `c = 0` the second is `0/0`); they
  extend continuously with `b₁² → 1`, `b₂² = O(c²) → 0` [✓ A5e], recovering the paper's
  decoupled straight valley: `Var(d_∞) = ησ²/(λ(2−ηλ/T_eff))`, stability
  `η·max(λ,μ) < 2T_eff`.

Each mode's variance factor is exactly the paper's Prop 3 at curvature `ν_i` — the curved
valley with linear floor **is** the straight-valley theory, applied twice at the Hessian
curvatures and mixed by the geometry of the hill-normal.

**Relation to the frozen formula.** As `μ → 0` at fixed `c`: `ν₁ → λ_loc`, `ν₂ → 0`,
`b₁² → 1+c²`, `b₂² → 0` at rate `μ²`, and mode 2's contribution `b₂²·V(ν₂) → 0` linearly in
`μ`; the sum converges to `η σ²/(λ(2 − η λ_loc/T_eff))` — the current paper formula is exactly
the `μ→0` limit of the true variance, with first-order correction

    Var(d_∞) = η σ²/(λ (2 − η λ_loc/T_eff)) + μ · c² η³ σ² / (2 (2T_eff − η λ_loc)²) + O(μ²)

(the coefficient simplifies to this compact form; verified symbolically) [✓ A4]. At `β = 0`
the modal formula agrees symbolically with the corrected 2-D Lyapunov solution *and with the
reviewer's own displayed formula* — their audit cell gives `Var(d_∞) = 199/3725 ≈ 0.0534228`
at `η=.1, λ=10, μ=.1, c=.5, σ²=4`, which simulation confirms [✓ A3, B1].

**Proof sketch.** (1) Modal decoupling: the joint update on `(w, m)` is block-linear with all
blocks polynomials in `H`; conjugating by `diag(Q,Q)ᵀ` reduces to two scalar `(p_i, m_i)`
loops, each eliminating `m` to the paper's AR(2) at `ν_i`. (2) Rotated noise stays `σ²I`.
(3) AR(2) stationary variance per mode is Prop 3. (4) The weights: eigenvector algebra above;
`Σ b_i² = |(−c,1)|²`. (5) Stability: Jury per mode; the binding mode is `ν₁`. ∎

## 3. Tier (iii) — general floor `f`: quasi-static reduction (approximation, stated as one)

For general twice-differentiable `f`, freeze `f'` at `f'₀` over the filter memory and treat
the river path `{x_j}` as an *exogenous deterministic* slowly varying input (the two
hypotheses now stated, not implied): then `d` obeys the closed loop at `λ_loc = λ(1+f'₀²)`
with forcing `u_j = −f'₀ φ'(x_j)` and noise variance `σ²(1+f'₀²)`, up to the second-order
remainder `−½ f''(ζ_t)(η m^x_t)²` of the exact one-step identity (Prop 5(a), unchanged).
Validity margin (now a hypothesis): coefficient drift `T_eff · v_riv · |f''| ≲ max(1,|f'|)`,
i.e. `k v_riv T_eff ≲ 1` on the sinusoidal floor. Tier (ii) calibrates what the exogeneity
step costs where both apply: `O(μ)` in the river curvature, per the expansion above. E10's
confinement-floor agreement (7/10 exact, 3/10 one grid step) is the empirical calibration on
the sinusoid, not a proof.

## 4. What changes in `main.tex`

- Prop 5(b) restated as the three tiers above; "nothing is frozen or truncated: the reduction
  is exact" deleted; the exact tier carries the modal theorem, with the tilted-profile
  sentence for the old formulas.
- Every downstream use of `η λ_loc < 2T_eff` is flagged as the frozen (necessary) threshold,
  with the exact linear-floor boundary `η ν₁ < 2T_eff` stated once.
- Abstract / §6 / Discussion sweep for over-definite curved-valley phrasing (plan T1 last ¶).
- E10/E8 captions keep their frozen-threshold guides (they are (iii)-tier predictions on a
  sinusoidal floor, correctly labeled); no re-runs required.

## 5. Response-letter points

- Reviewer §2.2 sustained in full: the exactness claim was false, and their displayed β=0
  Lyapunov variance is **exactly correct** under the paper's per-coordinate iid noise
  convention [✓ A3b] (an earlier internal pre-check that suggested otherwise had a sign
  error in its noise covariance and is superseded by this note's check).
- Repair delivered: their option 3 (solve the full coupled system, all β) in closed modal
  form — reproducing their formula at β=0 and strengthening the statement to an exact
  spectral decomposition with the explicit stability boundary `η ν₁ < 2T_eff` — plus their
  option 1 (constant-slope tier) and option 2 (quasi-static tier, now with stated
  hypotheses).
