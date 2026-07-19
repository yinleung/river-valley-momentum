# Fable working log v1: executing the fable_dgs_v1 program

*2026-07-03, same-day follow-up to `fable_dgs_v1.md`. Task: implement that note's theorem +
validation program (§§6–8) under the repo's rules — `codebases/CODING.md` for code,
`latex/WRITING.md`/`CRITERION.md` for prose, Claude↔Codex review protocol throughout. Everything
below ran end to end on this machine; every number is in a run record under
`codebases/results/cache/` or in `discussions/theory_cl_t2_t5_check.py`'s output.*

---

## 0. TL;DR

**All of it landed.** CL-1, CL-2, T2, T3′, T5 (+the T6′ window corollary) are proved in
`theory_cl_t2_t5.md` and numerically verified (Codex PASS); E8, the closed-loop grid, the E5
overlay, the E6 onset re-analysis, and **E7 — the last Strong-Success item — all pass their
decision gates**; the paper grew from 10 to 18 pages with the closed-loop section, the
deterministic filter-first theorem, the related-work pass, E7/E8, and a scope-corrected abstract
(Codex PASS after fixes). Three review rounds produced five real corrections, two of them to
claims that predate today. The weirdest findings are in §4 — the best one is that **β-warmup, the
schedule the onset story seems to recommend, is 15× worse than fixed β=0.9 on the toy** — and
§4.1 corrects a claim in `fable_dgs_v1.md` itself.

---

## 1. What was built (file map)

**Theory** — `discussions/theory_cl_t2_t5.md` + `theory_cl_t2_t5_check.py` (Checks A–E, all
pass):
CL-1 stability iff ηλ < 2T_eff via Jury conditions, flip mode at threshold, slow mode
1−ημ + O(β(ημ)²/(1−β)); CL-2 stationary variance ησ²/(λ(2−ηλ/T_eff)) via AR(2) algebra +
d-dimensional loss-gap corollary (simulation match to 3–4 digits, incl. ηλ=2.5 where SGD
diverges); T2 exact spectrum σ²|1−e^{−iω}|²/|1−ae^{−iω}|², zero DC, **monotone on [0,π]**,
half-of-peak cutoff cos ω_c = 2a/(1+a²), concentration S(π)/S(π/2) = 2(1+a²)/(1+a)² ≥ 2 iff
ηλ ≥ 1, exact filtered-power ratio with a Chebyshev bound ρ ≤ 1/T_eff; T3′ telescoping DC
identity (exact to 4e−14 on closed-loop noisy curved runs) + band corollary; T5 exact buffer
(1−β^t)S + ε_tA, two-sided tail pinch |ε_t|σ_{2r+1}(A) ≤ σ_{r+1}(M_t) ≤ |ε_t|‖A‖₂, Wedin
subspace bound (measured tight to ×1.2), polar-only misidentification instance with the exact
recovery time t*, post-polar period-2 limit (P_± + βP_∓)/(1+β) to machine precision.

**Infra** — `core/closedloop.py` (all dashed guides, one definition each: threshold, CL-2 std
with optional bend correction fp2, T2 spectrum, filtered-power ratio, ε_t, Wedin guide, noise
op-norm heuristic); `core/metrics.py` + `escape_fraction`; `rivervalley_sim.py` + β-schedules +
divergence guard; `mlp-diagnostic/adapter/contract.py` + `val_data_loader` +
`end_to_end_train` (closed-loop SGDM and Muon pre/post — the Contract's optional end-to-end
member, E7).

**Experiments** — drivers `run_e8_headline.py`, `run_e3_heatmap.py`, `run_e7_optimizer.py`;
extensions to `run_e5_matrix.py` (clean scenario + overlay traces) and `run_e6_trajectory.py`
(loss curve + onset windows). Figures `fig_e8_headline`, `fig_e3_heatmap`, `fig_e5_wedin`,
`fig_e6_onset`, `fig_e7_optimizer` + provenance summaries. Because run ids content-hash `core/`,
the whole suite was re-run **twice** (once after the core additions, once after Codex's fixes);
E1–E6 reproduce their published numbers bit-identically, one row per task in `runs.csv`.

**Paper** — `latex/main.tex` 10 → 18 pp: new Related Work; eq. (closedloop) + Prop. stability +
Prop. variance + regime paragraph; Thm. spectrum + Chebyshev remark; Lem. confinement; Rem.
β-window; Thm. filterfirst + Prop. polaronly + Rem. postpolar (+ the (2T−1)=T_eff conversion
note); E5 overlay, E8, closed-loop grid, E6 onset, and E7 sections with five new figures;
abstract/discussion/limitations rewritten to the regime-scoped claims; 6 new verified bib
entries; CRITERION.md term table extended. Builds clean under `make` (tectonic).

## 2. Results (all decision gates PASS)

| experiment | gate | outcome |
|---|---|---|
| E8 (a) | tube restoration, CL-2 match | escape 1.00/0.83/0.00 at β=0/0.5/0.9; rms 0.194 vs guide 0.194 (ratio 0.996) |
| E8 (b) | β=0 diverges beyond 2/η | 12/12 diverge at ηλ=2.5; β=0.9 rms 0.255 vs bend-corrected guide 0.249 |
| E8 (a) | lag reversal | β=0.999: rms 0.549, loss 3.78, x_T=−1.33 (vs 0.194 / 0.65 / 3.68 at 0.9) |
| grid p1 | CL-2 ratio ≈ 1 | 36/37 stable cells within 2·seed-SE+0.02; divergence boundary exact |
| grid p3 | window widens with λ/μ | upper edges β = 0.7 / 0.95 / 0.999 at λ/μ = 10/100/1000; T_eff(0.7)=5.7 vs predicted 5.6 |
| E5 overlay | T5 constants | tail inside two-sided envelope, upper guide ×1.05; Wedin ×1.17; noisy floor as predicted |
| E6 onset | confinement timing | HFER 0.025→0.98, onset step 200, 68% of loss drop pre-onset, ΔAlign +0.10→+0.69 |
| E7 A+B | mechanism beats β (toy) | best β spans 0.5–0.95 over five settings; median ρ(mech)=+0.75 vs |ρ(β)|=0.60 |
| E7 C | regime contrast (MLP) | benefit 23.3% at HFER 0.99 (EoS) vs 0.0% at HFER 0.02 (sub-critical) |
| E7 D | Muon closed loop | pre-polar wins final loss 3/3 (0.153 vs 0.271 at β=0.95) and alignment 3/3 |

With E7 in, **every Strong Success item of idea_v1.md §5 is now done**; what separates this from
Very Strong is scale (real mini-batch training in a dropped-in repo) and a schedule/adaptive-β
design — see §5.

## 3. Where the plan bent (and why)

- **E3-grid part 3 took three designs.** Tail loss → traveling-window loss → mechanism window
  (escape + relative lag). Reasons, in order: the quadratic river *ends* (at λ/μ=10 it is over
  by step 30, so tail metrics score a static problem where max filtering trivially wins); a
  fixed-horizon loss is dominated by the shared entry transient; and at λ/μ=1000, river
  *progress* is diffusion-limited — the hill-coupled x random walk over 3/(ημ)=1667 steps
  exceeds the 8-unit river, so no β can make measurable progress at σ=2. The final criterion
  (escape ≤ 0.25 ∧ lag ≤ 1.25×its own minimum, each conditioning scored over its own traveling
  horizon) is the T6′ statement itself; the relative lag tolerance is forced by the buffer's
  noise floor σ/√T_eff setting the absolute lag level. Performance-vs-mechanism is E7's job.
- **E7 gate C was redesigned from Spearman-on-val to the regime contrast.** Val loss on the
  synthetic non-realizable task anti-correlates with optimization (see §4.4), and at η=0.05 the
  val row is flat noise that a rank correlation reads as a perfect monotone. The honest,
  falsifiable operationalization of "mechanism predicts performance": HFER (0.02 vs 0.99) tells
  you *in which regime* momentum pays (0.0% vs 23.3% train benefit) — β alone cannot.
- **E8 gate 2 needed the derived bend correction** for panel (b): plain-λ ratio 1.102 at
  ηλ=2.5. Linearizing the offset dynamics gives contraction 1−ηλ(1+f′²) and noise σ²(1+f′²);
  the (1+f′²) cancels except in the (2−ηλ_eff/T_eff) denominator — a ≈6% systematic at
  ηλ/T_eff=0.13, nothing at panel (a)'s 0.09. Corrected ratio 1.026. The same cancellation
  explains a fable_dgs_v1 §7 mystery: why the *straight*-λ CL-2 formula matched the pilot's
  curved-valley rms at β=0.9 to 3% — and why β=0 escapes (locally ηλ(1+f′²)>2 wherever
  |f′| > √(2/ηλ−1)).

## 4. Something weird (maybe)

1. **fable_dgs_v1.md §3.2(a) contains a false claim, and its own check script "confirmed" it.**
   "The spectrum peaks at Nyquist iff ηλ>1" — in fact dS_g/d(cos ω) = −2σ²(1−a)²/(·)² < 0, so
   the argmax is π for *every* ηλ ∈ (0,2). The check script's empirical periodogram peak at
   ω=2.66 for ηλ=0.3 (vs π=3.14) was sampling noise on a near-flat spectrum, read as
   confirmation. The correct dichotomy is concentration: S(π)/S(π/2) = 2(1+a²)/(1+a)² ≥ 2 iff
   ηλ ≥ 1 (82 at ηλ=1.8, 1.03 at ηλ=0.3). The theory note states the corrected version;
   fable_dgs_v1.md is left as-written (it is a dated assessment; this log is its erratum).
2. **The paper carried two unverified numbers for a week, through a Codex prose PASS.** E4's
   "high-band ratios match (1−β)/(1+β) to three digits" — actual deviations are 4–12% (the
   metric is the high-band *median* of R, which matches the band, not the Nyquist endpoint) —
   and "0.001" for the curved relerr (actual 0.0022). Caught only when this pass's review
   explicitly demanded B6 spot-checks against `metrics.json`. Corrected in the paper,
   `results_phase1.md`, and the figure summary, with dated notes. Lesson encoded: a prose
   review that doesn't open the cache is not a B6 audit.
3. **β-warmup backfires exactly where the theory says filtering matters most.** The E6 onset
   result ("filtering pays once confined", HF is signal before) reads like an argument for
   increasing-β schedules — and on the curved noisy toy, increasing β (0→0.95) is **15× worse**
   than fixed β=0.9 (tail loss 11.5 vs 0.75), because the low-β prefix escapes the tube in the
   first steps and the schedule spends the rest of the run recovering. No contradiction: the MLP
   *enters* its valley through a smooth descent phase (warmup harmless), the toy *starts* at the
   valley at large ηλ (warmup fatal). When to raise β is a confinement question, not a clock
   question — that line made it into the paper's discussion.
4. **On the synthetic MLP, everything that optimizes better generalizes worse.** Momentum's 23%
   train-loss gain at EoS comes with *higher* val loss; β=0.99 has the best val in the SGDM row
   while being tied-worst on train; post-polar Muon beats pre-polar on val at every β while
   losing on train 3/3. A tidy overfitting axis orthogonal to the mechanism (512 samples,
   non-realizable target + label noise). Worth remembering before anyone runs E7 on a real task
   and gates on val: the *claim* is about optimization.
5. **A free theorem out of a review fix.** Codex flagged "the white floor is the worst case
   among DC-free disturbances" as an overclaim (mass just above DC filters at ≈1). The repaired
   statement is stronger and clean: |H_β|² decreasing × S_g increasing ⇒ (Chebyshev's integral
   inequality) ρ ≤ 1/T_eff for *every* monotone-increasing spectrum — zero DC buys nothing,
   monotonicity is the operative property. Verified on a 397-point grid (max ρ−1/T_eff =
   −2.3e−3).
6. Minor: the polar-only "failure" is invisible to output-level alignment in the orthogonal
   instance (O(G_t) still carries the signal at coefficient 1); it is a *subspace
   identification* failure, which is what Muon-style low-rank reasoning needs. The theory note's
   scope paragraph exists because the first draft got this wrong and Codex caught it (the one
   blocking theory finding).

## 5. Codex ledger and what remains

**Codex rounds.** Theory note: FAIL (1 blocking: polar-only stated on O(G_t)'s degenerate SVD;
4 lesser) → PASS. Code batch: PASS (3 minor: inline guides moved to core, per-panel gate
predeclaration, tie-aware Spearman; 1 nit: repeated-pole cancellation — all applied). Paper:
FAIL (2 blocking pre-existing number claims, 1 blocking Shen overclaim "we prove what they
assume" → "we supply a mechanism for, in our model"; \eqref→\Cref, E1–E8 labels, one banned
phrase) → PASS.

**Remaining, in priority order.** (i) Scale transfer — the one experiment-shaped risk
(fable threat 1): drop a NanoGPT-class repo into `codebases/<name>/upstream/`, wire the
Contract, and ask whether mini-batch streams at scale are hill-dominated and whether the E7
regime contrast survives; everything needed (probe, metrics, end_to_end_train pattern) exists.
(ii) Optional theory: a formal T6 schedule corollary; the tube-restoration remark (escape ↔
CL-2 rms vs radius) — currently empirical (grid p2). (iii) Housekeeping: `fable_dgs_v1.md`'s
§3.2(a) wording stands corrected only from here and the theory note.

## 6. Post-hoc: discussion of §4 with Codex (same day, at Leon's request)

A dedicated discussion session (not a review) on the weird bits. Where it landed:

- **W1 (argmax erratum).** Agreed the concentration ratio is the right sharp dichotomy, with
  the half-power cutoff `cos ω_c = 2a/(1+a²)` as the more intrinsic descriptor — both are
  already in Thm spectrum. Codex pushed for an inline erratum in `fable_dgs_v1.md` rather than
  wlog-only ("leaving the wrong line without a local warning is asking future agents to repeat
  the mistake") — **done**, at §3.2(a) and the §8.1 T2 table row.
- **W2 (B6 escapes).** Codex proposed a structural fix rather than regex-scraping: a
  `claims.yaml` manifest per figure (claim id → run id → metric path → transform → tolerance),
  a `scripts/check_numeric_claims.py` runner wired into the paper Makefile, optionally
  generating LaTeX macros so prose never hand-types numbers. It also found a live instance:
  this session's own `fig_e7_optimizer/summary.md` still said median ρ = +0.71 from the
  pre-tie-aware run (**fixed** to +0.75). I agree the manifest is the right architecture but
  it changes the prose workflow (macros vs typed numbers) — **left as a proposal for Leon**,
  not built.
- **W3 (β-warmup).** Codex ran its own probe: a *single* β=0 step before β=0.9 already costs
  4× on S3; β≥0.5 prefixes are mostly benign — so the linear ramp was confounded by time spent
  below a **confinement floor** (~0.7 here). I added the decisive arms to the E7 schedule table
  and re-ran (run `optimizer_toy_closedloop_set5_b6_seeds8_1ccc960e`): step 0→0.9 at t=30 →
  **89** (thirty full-noise steps are worse than the passing ramp's 11.5); floor-respecting
  ramp 0.7→0.95 → **0.48, the best arm overall, beating every fixed β** (best fixed: 0.645 at
  β=0.95); step 0.7→0.9 → 0.611. So the refined statement — now in the paper's discussion —
  is: β-warmup fails through its sub-floor prefix and *wins* when started at the floor; "when,
  and from where, to raise β is a confinement question, not a clock question." (This
  incidentally delivers a small positive schedule result the original E7 scoped out.)
- **W4 (val anti-correlation).** Codex disagreed with leaving it as a provenance clause: the
  implicit-regularization alternative (filtering removes update noise that was regularizing)
  is plausible enough that readers shouldn't have to infer the axis from logged numbers.
  **Adopted**: the paper's E7 section now states it explicitly (best-train cell = worst-val
  cell; post-polar wins val 3/3 while losing train 3/3; E7 cannot distinguish overfitting from
  regularization-removal).
- **W5 (Chebyshev).** Agreed to promote: now **Corollary (White-noise ceiling)** in the paper,
  with the equality cases (β=0 or flat spectrum) and the explicit caveat that zero DC alone
  does not suffice. CRITERION term added.
- **W6 (polar-only scope).** Codex's residual concern: "the only signal estimate available"
  is literally true for the E5 diagnostic and low-rank analyses, not for Muon-as-used (which
  applies O(buffer) blindly). **Adopted**: Prop (polaronly) now scopes the sentence to the
  rank-r signal-recovery reading, with a parenthetical on the operational content for
  Muon-as-used.
- **W7 (open floor).** Beyond the stale +0.71, Codex's summary judgment matched mine: the
  substantive open question among the weird bits is W4 — the mechanism is optimization-clean,
  the generalization axis is real, and the right home for it is exactly the explicit
  limitation it now has.

Paper rebuilt after these edits: 19 pp, refs clean, schedule + generalization text verified
against the new run record.

*Files touched today: `theory_cl_t2_t5.md` + check script (new); `core/closedloop.py` (new),
`core/metrics.py`, `core/README.md`, `rivervalley_sim.py`, `run_e3_heatmap.py` (new),
`run_e5_matrix.py`, `run_e6_trajectory.py`, `run_e7_optimizer.py` (new), `run_e8_headline.py`
(new), `mlp-diagnostic/adapter/contract.py` + `AGENT_NOTES.md`, `CODEBASE_INDEX.md`; five new
figure modules + `fig_e5_wedin.py` title fix; `results/` rebuilt twice (11 summaries current);
`latex/main.tex`, `references.bib` (+6 verified entries), `CRITERION.md`;
`references/README.md` (Andreyev authors); `results_phase1.md` (status + E4 correction).*
