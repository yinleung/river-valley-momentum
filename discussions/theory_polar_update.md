# Theory note: T3 — update-level theorems for the polar step

Repair of the Thm 4 → Muon transfer per `review_v5.md` §2.3 and `plan_v5.md` T3. The
reviewer is right: Thm 4/5/Cor 3 control the buffer `M_t`; exact `polar(·)` flattens all
singular values to 1, so buffer-subspace recovery does not transfer to the update, and the
closing sentence of Thm 4 ("the pre-polar update inherits the bound") plus Prop 6's rank-r
readout do not describe Muon as run. Two genuine update-level theorems replace that sentence:
T3a for the exact polar under a full-rank signal, T3b for the operator practical Muon actually
applies (finite Newton–Schulz on a Frobenius-normalized buffer). The reviewer's rank-one
construction becomes the necessity example for both hypotheses. Paired check:
`theory_polar_update_check.py`. Status flags [✓] refer to that script.

Throughout: `G_t = S + (−1)^t A` (the Thm 4 model), buffer `M_t = (1−β^t)S + ε_t A` with
`|ε_t| ≤ (1−β)(1+β^t)/(1+β) → 1/T_eff`, and `X_t := (1−β^t)S` the clean-signal buffer.
Matrices in `R^{d₁×d₂}`, `d₁ ≥ d₂` WLOG.

---

## 1. T3a — full-rank polar perturbation (exact polar, update level)

**Fact (Li).** For full-column-rank `A, B ∈ R^{d₁×d₂}` with unitary polar factors
`U_A = A(AᵀA)^{−1/2}`:

    ‖U_A − U_B‖_F ≤ 2/(σ_min(A) + σ_min(B)) · ‖A − B‖_F .

R.-C. Li, SIMAX 16(1):327–332 (1995); also Higham, *Functions of Matrices* (2008), §8.2.
The constant is attained (e.g. `A = (σ,0)ᵀ`, `B = (0,σ)ᵀ`: both sides `√2`) [✓ A1: random +
rectangular + near-singular instances; tight family ratio → 1].

**Theorem T3a.** Assume `σ_min(S) > 0` (full column rank; in the Thm 4 notation `r = d₂`).
Since `polar(cM) = polar(M)` for `c > 0`, `polar(X_t) = polar(S)`, and Li's bound with
Weyl's `σ_min(M_t) ≥ (1−β^t)σ_min(S) − |ε_t|‖A‖₂` gives, for every `t` with the right side
positive,

    ‖polar(M_t) − polar(S)‖_F ≤ 2 |ε_t| ‖A‖_F / ( σ_min(M_t) + (1−β^t) σ_min(S) )
                              ≤ 2 |ε_t| ‖A‖_F / ( 2(1−β^t) σ_min(S) − |ε_t| ‖A‖₂ ),

an update-level error that dies at the buffer rate: as `t → ∞`,
`‖polar(M_t) − polar(S)‖_F ≤ (‖A‖_F/σ_min(S)) · (1/T_eff) · (1 + O(1/T_eff))` [✓ A2].
Polar-only (`polar(G_t)`) has the same bound with `ε_t` replaced by `1`: no decay — the
factor the filter buys is exactly `1/T_eff` [✓ A2].

Bound quality: on random instances the bound overshoots the measured error by a factor in
`[1.4, 2.1]` (median 1.6); on the rotational worst-case family it is tight to `1.0003`
[✓ A3, measured].

## 2. T3b — the deployed Newton–Schulz operator is Lipschitz; the rate transfers

Practical Muon never runs exact polar. It runs

    D(M) = NS_k( M / (‖M‖_F + ε_ns) ),   NS step  q(X) = a X + b X(XᵀX) + c X(XᵀX)²,

with `(a,b,c) = (3.4445, −4.7750, 2.0315)`, `k = 5`, `ε_ns = 10⁻⁷` (Jordan's Muon; bf16 in
production, fp64 here — precision is not part of the claim). Two structural facts:

1. `q` is an *odd matrix polynomial*: `q(UΣVᵀ) = U q(Σ) Vᵀ` in the input's own singular
   frame, so `NS_k(X) = U h(Σ) Vᵀ` with the scalar response `h = q^{∘k}` applied to the
   singular values [✓ B6, exact to fp roundoff]. `h(0) = 0`, `h'(0) = a^k ≈ 484.9`.
   Neither `q` nor the composed response `h` is monotone on the operating range
   (`q'(1) = a+3b+5c ≈ −0.723 < 0`; `h'` itself turns negative near `s ≈ 0.0064`), so the
   monotone-map remark (Rem 12 / mapscope) does **not** cover the deployed map — the
   ordering route is closed; the Lipschitz route below replaces it [✓ B1].
2. Exact polar is the singular-map `σ ↦ sign(σ)`, whose slope at `0` is infinite; `h` is a
   fixed odd polynomial with finite slope everywhere. This is the entire difference between
   the reviewer's counterexample and the deployed operator.

**Lemma 1 (normalization).** `ν(M) = M/(‖M‖_F + ε_ns)` satisfies, whenever
`min(‖M‖_F, ‖X‖_F) ≥ r_low`,

    ‖ν(M) − ν(X)‖_F ≤ 2 ‖M − X‖_F / (r_low + ε_ns),

by `ν(M)−ν(X) = (M−X)/(‖M‖_F+ε) + X·(‖X‖_F−‖M‖_F)/((‖M‖_F+ε)(‖X‖_F+ε))` and the reverse
triangle inequality. The constant is valid and loose by at most a factor 2: the
antiparallel equal-norm pair attains exactly half of it [✓ B3].

**Lemma 2 (odd matrix polynomial, Frobenius-Lipschitz).** For `‖X‖₂, ‖Y‖₂ ≤ 1`,

    ‖NS_k(X) − NS_k(Y)‖_F ≤ L_h ‖X − Y‖_F,   L_h = sup_{s∈[−1,1]} |h'(s)| = sup_{[0,1]} |h'|,

via the Hermitian dilation `H(X) = [[0,X],[Xᵀ,0]]`: odd polynomials satisfy
`q(H(X)) = H(q(X))`, hence `h(H(X)) = H(NS_k(X))`; spectra of the dilations lie in
`[−1,1]`; and a scalar function that is `L`-Lipschitz on an interval containing both
spectra is `L`-Lipschitz on Hermitian matrices in the Frobenius norm (Daleckii–Krein /
Birman–Solomyak divided-difference bound — true in the HS norm with the same constant,
unlike the operator norm). `‖H(X)−H(Y)‖_F = √2 ‖X−Y‖_F` on both sides cancels.
Numerically `L_h = a^k = 484.88…`, attained at `s = 0` [✓ B1, B2 — including adversarial
pairs at small singular values approaching the constant].

**Theorem T3b (deployed operator; annulus hypothesis from the signal).** Let
`r_low ≤ min(‖M_t‖_F, ‖X_t‖_F)`; in the Nyquist model
`r_low = (1−β^t)‖S‖_F − |ε_t|‖A‖_F > 0` serves. Then

    ‖D(M_t) − D(X_t)‖_F ≤ L_NS · |ε_t| · ‖A‖_F,    L_NS = 2 L_h / (r_low + ε_ns),

so the buffer's `1/T_eff` recovery transfers to the *actual deployed update* with the
explicit constant `L_NS`; asymptotically the update error is at most
`(2 L_h ‖A‖_F / ‖S‖_F) · (1/T_eff) · (1+o(1))` [✓ B4]. What separates this from the
exact polar is the finite slope: exact polar is the singular map `sign(σ)`, which is not
Lipschitz on any interval containing 0 — so no analogue of Lemma 2 exists for it unless
the occupied singular interval is bounded away from zero, which is exactly T3a's
full-rank hypothesis. T3b needs no rank hypothesis at all: the Frobenius floor `r_low`
is positive even for a rank-one signal, and the finiteness of `L_h` is a property of the
deployed polynomial, not of the signal.

Interpretation of the constant — the knee. `L_h ≈ a^k` is the worst case, attained only
near `σ = 0`; away from zero the map is far tamer (`sup_{[0.3,1]}|h'| ≈ 10`) [✓ B1 reports
both]. The constant is not decorative: NS_k *by design* re-amplifies any normalized
disturbance component above `~1/L_h` back to order one, so the deployed error decays
slowly until the disturbance amplitude crosses the knee `~1/L_NS`, and **below the knee the
map is in its linear zone `h(s) ≈ a^k s` and the T3b bound is near-tight** [✓ B5c: at
`|ε_t|‖A‖_F/‖S‖_F ≈ 5·10⁻⁴` the measured error is within a small factor of the bound and
decays linearly with amplitude]. Equivalently: the buffer's `1/T_eff` recovery becomes an
update-level recovery once `T_eff ≳ L_h · ‖A‖_F/‖S‖_F`. `1/L_NS` is the
disturbance-amplitude knee the G5 experiment will probe.

## 3. The reviewer's construction, repositioned as necessity  [✓ B5]

`S = σ_S uvᵀ` rank-one, `A = α u'v'ᵀ` with `u ⊥ u'`, `v ⊥ v'`, `α > σ_S`. Exact polar —
read in the canonical *partial-polar* sense `Σ_{σᵢ>0} uᵢvᵢᵀ`, since the unitary polar
factor is not unique at rank deficiency: `polar(M_t) = uvᵀ + sign(ε_t) u'v'ᵀ` for every
`t` — the error to `polar(S)` is exactly 1 forever; no amplitude-based bound can exist
because `σ_min(S) = 0` (T3a's hypothesis is necessary) and `sign(σ)` has no finite slope
at the occupied `σ = 0` (no Lipschitz route either). The construction does *not* violate
T3b's hypotheses — `r_low = ‖S‖_F(1−β^t) − |ε_t|α > 0` here — and indeed T3b's bound
holds on it [✓ B4 includes rank-one signals]. Deployed operator: the disturbance enters
`ν(M_t)` at scale `|ε_t|α/‖M_t‖_F`
and `h` maps it to at most `L_h · |ε_t| α / ‖M_t‖_F → (L_h α/σ_S)/T_eff`: the deployed
update degrades gracefully — decreasing in `β`, on the T3b bound, entering the near-tight
linear zone once the amplitude is below the knee — while exact polar jumps [✓ B5: measured
exact-polar error exactly 1 at every β; deployed error decreasing in β; sub-knee decay
linear]. So Prop 6 stops being a transfer device ("ties read from the input ordering") and
becomes the sharpness example showing both T3a's and T3b's hypotheses are the right ones.

## 4. What changes in `main.tex`

- Thm 4's closing sentence ("…the pre-polar update inherits the bound") deleted; Thm 4/5/
  Cor 3 stay as *buffer* theorems, explicitly so named.
- New: T3a and T3b as displayed update-level theorems (statements above; Li cited, Lemma 2
  proof in appendix — dilation + divided differences, self-contained).
- Prop 6 restated as the necessity example (§3 above); the "top-r readout" framing and the
  tie-reading sentence removed.
- Rem 12 (mapscope) scoped: monotone family only (raw buffer, powers `σ^{1−q}`, exact
  polar); one sentence noting the deployed NS map is non-monotone and is covered by T3b's
  Lipschitz route instead.
- Register sweep: "filter-first theorems for the polar map" → buffer theorems (Thm 4/5/
  Cor 3) + update theorems (T3a/T3b) + necessity example (Prop 6), in abstract,
  contributions, §7 intro, Discussion.
- E5's overlay keeps its buffer-level guides (correct as stated); its caption gains no new
  claims.

## 5. Response-letter points

- Reviewer §2.3 sustained in full; repaired along their "full-rank signal" route (T3a) and
  extended to the operator Muon actually runs (T3b), which their review asked for in the
  form "a genuine theorem about the polar output". The counterexample they supplied is now
  the paper's necessity example, cited as such.
- The `1/T_eff` rate survives at the update level in both regimes; what does not survive
  (and is no longer claimed) is any statement for the exact polar of a rank-deficient
  signal.
