# Fable working log v3: executing review_v3

*2026-07-04. Task (Leon): read `discussions/review_v3.md` (the review of `latex/main.pdf`),
think independently about whether it makes sense, improve the paper accordingly. Rules:
`latex/WRITING.md` + `CRITERION.md` for prose, `codebases/CODING.md` for code, Claude↔Codex
protocol throughout. Companion to `fable_wlog_v1.md` / `fable_wlog_v2.md`; every number below
is in a run record under `codebases/results/cache/` or in `discussions/theory_fr_cv_be_check.py`
output.*

---

## 0. TL;DR — and the assessment of review_v3

**The review is sound in its priorities and was implemented with two corrections and one
addition it did not ask for.** Its top recommendation — the large-batch decomposition ("the
single most important diagnostic improvement") — is exactly right and is now E12; its
curved-valley and band-energy theory asks are now Propositions/Corollary with numerical
checks; its four "technical tightening" items and both positioning asks are in. Everything
landed through one Codex FAIL→PASS cycle (three real blockers, §5).

**Correction 1 (to the review's framing of weakness 2).** The review worries that E11's
high-frequency content "can simply be batch noise." As stated, that is impossible: a
temporally independent residual has a flat spectrum, so it cannot push HFER above the white
baseline (0.402) — and E11 measured 0.82–0.84. The *defensible* version of the worry is that
the high band could be the state-mediated response to noise kicks rather than curvature-locked
oscillation, and that HFER says nothing about curvature alignment. That version is what E12
tests, and the review's noise-vs-hill dichotomy maps exactly onto the measured
residual-vs-state split. (This design logic is now one sentence in §E12 of the paper.)

**Correction 2 (review's Theorem-5 suggestion).** "Why not random-phase/orthogonality
assumptions for sharper constants" — declined: that would reintroduce the stochastic
assumptions the deterministic theorem exists to avoid (li2026denoise already owns the
stochastic side). The energy form (Corollary BE) instead replaces ℓ¹ tone sums with measured
band energies; its two prices (most-steps control, crest factor 1/√θ) are structural — a lone
spike carries vanishing energy, so *no* energy hypothesis can control every step. This is
stated in the paper as the honest cost, not hidden.

**Addition beyond the review.** The review asked for negative controls; the closed-loop
*forced response* (new Prop FR) gives them parameter-free guides and produced a prediction the
review did not contain: momentum **amplifies** disturbances at its own underdamped hill-mode
frequency θ_β, unboundedly as β→1 (gain ~ √(η/(λ(1−β)))). E13 measured the harm at 6.0× vs
the 6.01× guide. The same proposition's DC gain 1/λ (β-free) is what makes the curved-valley
tracking offset momentum-immune — the two new propositions interlock.

**Deferred, declared (unchanged from v2):** CIFAR-10/ResNet, modded-nanogpt-scale runs,
practical Muon training (Newton–Schulz, all matrix layers), Adam preconditioning — this
machine is a laptop; the limitations paragraph now names practical-Muon explicitly as out of
scope. These remain the review's path from borderline to strong accept (§6).

## 1. Independent assessment, point by point

| review item | verdict | disposition |
|---|---|---|
| §weakness 1: real-scale evidence small | agree | not addressable on this hardware; limitation sharpened; decomposition makes the existing scale evidence load-bearing |
| §weakness 2: HFER may conflate hill with noise | agree with Correction 1 | **E12** (new experiment, 3 gates predeclared, all PASS) |
| §weakness 3: curved valley local/heuristic | agree | **Prop CV** (frozen-coefficient reduction; exact for linear floors); also fixed a real internal inconsistency — the old body display dropped the (1+f′²) factor that E8/E10 already used |
| §weakness 4: Thm 5 stylized (ℓ¹ tone sums) | agree with Correction 2 | **Cor BE** (band-energy filter-first; hypothesis = the quantity HFER estimates) |
| §weakness 5: Muon not practical-scale | agree | declared out of scope in Limitations (named: Newton–Schulz, all layers, tuned baselines) |
| tightening 1: boundary remark energy vs amplitude | agree | rem:boundary now states energy-level O(T_eff²/T), amplitude-level O(T_eff/√T), and the S(ω)→0 caveat with the DC-bin example |
| tightening 2: Thm 5 constants | see Correction 2 | Cor BE |
| tightening 3: white-noise assumption scope | agree | "additive and state-independent" added to cor:hillloss scope; Limitations states E12 verifies only temporal whiteness of the residual, not state-independence/isotropy |
| tightening 4: EMA vs heavy-ball in abstract/intro | agree | qualifier now in the abstract, the closed-loop contribution bullet, and (already) E9 |
| positioning: don't market as "momentum-as-filter" | agree | abstract reframed around closing the loop; intro/related-work now credit the filter lineage (incl. new yao2023signal citation, verified from the arXiv page) and state what was missing: the emitted spectrum + the closed-loop consequences |
| ablations that can falsify | agree | **E13** with exact FR guides (4 gates PASS, 28/28 cells on guide) |
| curvature-alignment diagnostics | agree | folded into E12 (top-16 restricted-Hessian eigenpairs, projection split) |
| Hessian at the edge / progressive sharpening | partially | E12 measured restricted ηλ_top = 0.21–0.42 ≪ 2: the oscillating mode is a network-level object (restriction only lower-bounds sharpness by interlacing). E11's progressive-sharpening speculation replaced by the measurement |

## 2. Theory built (all in `discussions/theory_fr_cv_be.md`, checks in `theory_fr_cv_be_check.py`, all PASS)

- **Prop FR (closed-loop forced response).** Gain |G_β(ω)| = η(1−β)/|p(e^{iω})|; DC gain
  exactly 1/λ at every β; Nyquist gain η(1−β)/(2(1+β)−e) ≈ (η/2)|H_β(π)|; resonance at θ_β
  with |G_β(θ_β)| → √(η/(λ(1−β))) (asymptote matches simulation to 4 digits at β=0.99;
  amplification over β=0: 4.2×/13.4×/42.4× at β=0.9/0.99/0.999). Ported as Prop 4.
- **Prop CV (curved-valley reduction).** Exact GD one-step identity with λ_loc = λ(1+f′²),
  forcing ηf′φ′, Lagrange remainder ≤ ½max|f″|Δx²; frozen-coefficient EMA reduction onto the
  closed loop (exact for linear floors — mean offset cφ′/λ_loc β-free, variance = CL-2 with
  λ_loc; verified at every β, var ratio 0.95–1.00). Consequence: worst-case confinement floor
  ηλ(1+(ak)²) < 2T_eff **predicts E10's measured floors — 7/10 exact, 10/10 within one grid
  step** (Check B3, from the cached bend_window arrays). Ported as Prop 5.
- **Cor BE (band-energy filter-first).** From the exact response identity
  EMA(A)_t = Φ_t − β^tΨ (verified to 8e-14): energy contraction ρ̄ = √(ρ_high²+γ) + burn-in
  term; most-steps per-step bound (Markov); Wedin subspace identification on kept steps. All
  bounds verified over β×γ grid (tightness 0.04–0.63, exceptional fraction 0 at θ=0.1).
  Ported as Cor 3 + scope remark.

## 3. Experiments built and run

**E12 — decomposition** (`nanogpt/adapter/decomp.py` + `scripts/run_e12_decomp.py`; task
`nanogpt`, probe `decomp`; run `nanogpt_decomp_lr3_seeds3_LB1024_k16_4f72b186`; ~2 min/run ×
9 on MPS; `contract.py` untouched so E11's cached identity is intact). Gates predeclared
before the pilot, unchanged by it; all PASS (3-seed means, lr = 0.05/0.1/0.2):
- residual whiteness: HFER(ξ̂) = 0.400/0.404/0.409 vs white 0.402;
- high-band state share = 0.829/0.842/0.927; direct accounting of the raw high band: state
  0.722/0.734/0.860, cross 0.128/0.129/0.074, residual 0.149/0.137/0.066 (derived from cached
  spectra; recipe in `results/figures/fig_e12_decomp/summary.md`);
- curvature alignment: top-16 of 65,536 dims carry 0.917/0.885/0.824 of the state high band;
  HFER in/out = 0.880/0.887/0.942 vs 0.561/0.591/0.814;
- report: restricted ηλ_top = 0.27/0.42/0.21 (per-run 0.06–0.47) — far below the edge.
Consistency: seed-0 HFER(g^mb) = 0.821/0.823/0.835 vs E11's published 0.821/0.823/0.838 (MPS
third-digit nondeterminism). Infra note: **double backward works on MPS in torch 2.10** — the
HVP eigensolve runs on-device (CPU fallback kept); power-iteration residuals ≤ 0.012 (top 4).

**E13 — forced-disturbance controls** (`scripts/run_e13_forced.py` + `forcing_fn` param on
`rivervalley_sim.simulate` + `forced_gain`/`resonant_frequency`/`forced_hill_loss` in
`core/closedloop.py`; run `forced_controls_sweep_om4_b7_seeds12_da578cc6`, ~2 min). Four
gates predeclared, all PASS: 28/28 (ω,β) cells on the parameter-free guide (0.044–36.6);
Nyquist amplitude ratio 5.03e-4 (guide 5.05e-4); passband amplitude spread 1.022 (guide
1.02) — no β removes a passband tone; resonance harm 5.99× (guide 6.01×), frequency-matched
(the same tone costs β=0.99 only 0.047).

Figures `fig_e12_decomp`, `fig_e13_forced` (+ summaries); `figures/_data.py` gained a `probe`
filter because task `nanogpt` now has two probes — `fig_e11_scale.py` pins
`probe="closedloop"`.

## 4. Paper delta (`latex/main.tex`, 29 pp, tectonic clean, 0 unresolved refs)

Abstract reframed (loop-closing, EMA-normalization qualifier, FR/controls/decomposition);
intro: filter-lineage sentence + two contribution bullets reworked, one added; related work
"Momentum as a filter" repositioned (+`yao2023signal` in bib and `references/README.md`);
rem:boundary tightened; **Prop 4 (forced response)**, **Prop 5 (curved-valley reduction)** +
consequences ¶ replacing the (wrong-coefficient) heuristic ¶; rem:window floor formula;
**Cor 3 (band-energy filter-first)** + remark, rem:sdr rescoped; tab:map +3 rows (and col 2
→ p{6.4cm} to kill a 157pt overfull); Measurements +3 entries; **new §E13**; E10 floor
prediction sentence; E8 prop-citations; sec:scale split into §E11/§E12 with the
progressive-sharpening speculation replaced by E12's measurement; Discussion + Limitations
updated; CRITERION.md +12 canonical terms, E1–E13.

## 5. Claude↔Codex protocol

Round 1: **FAIL** with three blockers, all legitimate: (i) prop:curved(b) stated the frozen
reduction without the retained f″ remainder (overclaim vs the note) — statement now carries
the remainder, exactness reserved for linear floors; (ii) tab:map wrote "1−γ **=** measured
HFER" though the corollary counts DC low while HFER excludes it — now "HFER estimates 1−γ";
(iii) the E12 body quoted the state share (state/(state+residual)) as a fraction *of the raw
stream's high band* — Codex recomputed the direct fractions from the cache (0.72/0.73/0.86)
and I verified independently before recutting the sentence with both metrics explicit. Nits
applied: editorial phrases removed, "local hill sharpness" canonicalized, driver-docstring
dimension fraction corrected. Nits declined with reasons (Codex accepted both in round 2):
caption run-ids keep the paper-wide no-SHA convention (full ids in summary.md/config.json);
"honest scope"/"makes cheap" are pre-existing register. Round 2: **PASS**.

## 6. What remains (the review's strong-accept paths, in order of value per effort)

1. One recognized non-toy setting (CIFAR-10 ResNet SGDM/Muon, or modded-nanogpt track) with
   the same β×lr sweep + HFER + decomposition columns — the Integration Contract makes the
   wiring cheap; the compute does not fit this laptop.
2. Practical Muon (Newton–Schulz, all matrix layers, Nesterov arm) vs pre/post/polar.
3. Adam-preconditioned streams (does the preconditioner whiten the hill band?).
4. Optional: E12's subspace-drift check across the probe window (eigensolve at window start
   and end), noted as a caveat in the figure summary.

## 7. Bookkeeping

New cached runs: `nanogpt_decomp_lr3_seeds3_LB1024_k16_4f72b186`,
`forced_controls_sweep_om4_b7_seeds12_da578cc6` (indexed in `results/index/runs.csv`). No
re-runs required: `contract.py` untouched (E11 record identity intact); `core/closedloop.py`
additions are pure-additive (existing guides byte-identical); cached records remain the
artifacts behind all previously published numbers. `CODEBASE_INDEX.md` and
`nanogpt/AGENT_NOTES.md` updated. Theory note + checks:
`discussions/theory_fr_cv_be.md` / `theory_fr_cv_be_check.py` (FR/CV/BE, all checks PASS).
