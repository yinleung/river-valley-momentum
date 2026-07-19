# revision_v5 changelog — theory repairs + writing workstream (session 2026-07-19)

Executed from `plan_v5.md` on Wisteria (login node), paper-only scope per Leon's in-session
decision; the GPU campaign (G1–G6) and public-repo release remain the open front. Codex
gates per the house protocol; verdicts recorded inline.

## Decisions taken in-session (Leon, 2026-07-19)
- **W4 / plan §8 item 1 resolved: branch (a) remove.** The p.32 E7 schedule-arm numbers
  (factors 15/118, detector steps, warmup arms) are replaced by a qualitative
  confinement-floor design constraint + future-work pointer; the material stays with the
  BETASCHEDULER paper.
- **Session scope: paper-only.** Venv bring-up done as a side effect (`.venv`, python3.8,
  numpy/sympy/matplotlib); no pjsub jobs, no data staging.

## Theory (all check-gated, Codex-challenged, then ported)
- **T1 — Prop 5(b) repair** (`theory_cv2_coupled.md` + `_check.py`, all PASS).
  Linear-floor + quadratic-profile valley under EMA momentum decouples exactly over the
  loss-Hessian eigenframe (eigenvalues ν₁ ≥ ν₂; real, distinct for c ≠ 0): two independent
  straight-valley loops, iid modal noise, no cross term. Exact stability boundary
  ην₁ < 2T_eff (frozen ηλ_loc < 2T_eff is necessary-not-sufficient, gap μc²/(1+c²) + O(μ²));
  Var(d_∞) = Σᵢ bᵢ² ησ²/(νᵢ(2−ηνᵢ/T_eff)), b₁²+b₂² = 1+c²; frozen formula = exact μ→0
  limit with first-order correction μc²η³σ²/(2(2T_eff−ηλ_loc)²). Tilted-profile tier keeps
  the old formulas exactly. **Correction to the pre-check:** `plan_v5_t1_precheck.py` had a
  sign error in its noise covariance (+c for −c); with the right sign the **reviewer's β=0
  formula is exactly correct** (their audit cell: Var = 199/3725 ≈ 0.0534228) and the modal
  formula reproduces it. plan_v5's "matches none of three noise placements" narrative is
  superseded; the response letter should concede and extend, not dispute.
- **T3 — update-level Muon theorems** (`theory_polar_update.md` + `_check.py`, all PASS).
  T3a: full-rank polar update via Li (1995) (‖polar(M_t)−polar(S)‖_F ≤ 2|ε_t|‖A‖_F/(σ_min(M_t)+(1−β^t)σ_r)
  → ‖A‖_F/(σ_r T_eff)); bound measured tight to [1.4, 2.1] on random instances, 1.0003 on the
  rotational family. T3b: the deployed operator D_NS = NS₅ ∘ Frobenius-normalize is
  Lipschitz with constant 2L_h/(κ+ε_NS), L_h = sup|h_NS'| = ψ₁⁵ ≈ 485, via the Hermitian
  dilation + divided-difference bound (odd matrix polynomial ⇒ scalar Lipschitz constant
  transfers in Frobenius norm); the buffer 1/T_eff rate transfers to the deployed update
  with the knee ~(κ+ε_NS)/(2L_h) below which the bound is near-tight (sub-knee linear
  zone verified; NS re-amplifies disturbances above the knee by design). Prop 6 becomes
  the necessity example (exact polar error exactly 1 forever; partial-polar sense); h_NS
  is nonmonotone (h' < 0 near σ ≈ 0.006), so Rem mapscope's monotone class excludes it.
- **T5 — heavy-ball corollary** (`theory_hb_corollary_check.py`, all PASS). New Cor: at
  fixed η_HB, stability iff η_HBλ < 2(1+β); loss = η_HBσ²/(2(1−β)(2−η_HBλ/(1+β)));
  **minimum at β* = max(0, √(η_HBλ)−1)** — increasing for η_HBλ < 1, U-shaped for
  1 < η_HBλ < 4. E9(f)'s measured increasing sweep sits in the monotone regime.
- **T4 — acceleration complementarity.** "Three views coincide" → two views meet, the
  third (acceleration) is complementary; DC gain exactly 1 under EMA; AR(2) = heavy-ball
  hardware answering a different question; Lessard–Recht–Packard cited (loop + related
  work + discussion).
- **T2 (global curved-valley stability): not attempted this session** — stretch item;
  remains open and is stated as open in the paper.

## Paper (latex/main.tex; 34pp → 39pp total, body ≈ 32pp + appendices)
- Prop 5 restated in three labeled strengths (exact identity / exact linear-floor modal
  solution / quasi-static with stated hypotheses) + Rem frozenexact; downstream frozen
  threshold everywhere labeled "frozen diagnostic".
- sec:filterfirst split into "Buffer theorems" and "From the buffer to the update":
  Thm 4's false transfer sentence deleted; Prop 6 rewritten as necessity; new
  thm:polarupdate, lem:oddlip (proof in new appendix A), thm:nsupdate, Rem knee; Rem
  mapscope rescoped. Register "buffer theorems + update theorems + necessity example"
  in abstract/contributions/discussion/limitations.
- cor:hb displayed; E9 text tied to it.
- Abstract rewritten (no "confirm each prediction"; scope sentences; negative results
  named); statement-class paragraph + class column in tab:map.
- E1–E4 moved to appendix B as implementation checks with a body summary; proofs of
  Thm 4/5/Cor 3 moved to appendix A.
- E11 reporting made divergence-counted: benefit series 0.8/3.5/11.9% on the three clean
  learning rates (cache-verified); lr=0.4 β=0 counted as failed (ndiv=1/3), no percentage
  quoted or drawn; fig_e11_scale panel (c) re-rendered from the same cache with the
  failed-baseline annotation (figure-spec change only; provenance summary.md updated).
- Statistics-and-conventions + Reproducibility (future-tense) paragraphs in Measurements;
  whiteness/cutoff caveat added to Limitations.
- New refs: li1995polar, higham2008functions, bhatia1997matrix, lessard2016analysis,
  zhang2019noisy. CRITERION.md: new canonical rows, banned constructions
  ("experiments confirm every prediction"; percentages against diverged baselines;
  "the reduction is exact" outside Prop 5(b)), new fixed symbols (ν, ψ, K, h_NS, L_h, κ).
- Build on Wisteria: pdflatex (authblk.sty + cleveref.sty vendored; unused missing
  packages commented out of preamble.tex).

## Codex protocol log
- T1 note: round 1 FAIL (c=0 edge in modal weights) → fixed (continuous extension +
  check A5e) → math otherwise confirmed (noise sign, modal reduction, no cross term).
- T3 note: round 1 FAIL (overclaim that the reviewer construction breaks T3b's annulus
  hypothesis) + 5 nits (normalization constant "attained" → half; measured tightness
  numbers; h vs q monotonicity; partial-polar wording; B4 boundary comment) → all fixed;
  Li citation, dilation argument, DK Frobenius route, spectra containment all confirmed.
- Full-paper judge round 1 FAIL: (1) reproducibility tense, (2) benefit bar vs failed
  baseline in fig E11(c), (3) statement-class overclaim, (4) tab:map update-row test
  column, (5) "Integration Contract" internal term → all fixed.
- Full-paper judge round 2 FAIL: intro "each prediction" vs the marked-untested knee row
  (blocking) + 3 nits (summary.md 21.8% drift marker, E10 floor-to-top-span wording,
  numpy trapezoid shim in an old check) → all fixed. Codex round 2 also audited
  E9/E10/E13/grid/E6/E7/E11/E12 numbers against their cached records: all match.
- **Full-paper judge round 3: PASS — no blocking issues, no nits** (2026-07-19).
  Verified in the pass: buffer/update split, statement-class discipline, divergence-counted
  E11, unmeasured-knee scoping, no banned register, no missing cross-references.
- Final state: `latex/main.pdf` 39 pp (body ≈ 32 + appendices A/B), builds clean on
  Wisteria pdflatex; all seven `discussions/*_check.py` suites pass under `.venv`.
  Working tree deliberately left uncommitted (house rule: commit only when Leon asks).

## Open front (unchanged from plan_v5)
- G1–G6 GPU campaign on Wisteria (§3), Phase-0 bring-up, B1–B5 instrumentation.
- Public repo + tagged release at resubmission (§5); the paper's Reproducibility
  paragraph is future-tense until then.
- T2 global stability statement; T6 noise-scope lemma (both optional/stretch).
- Response letter: assemble after the campaign; concede-and-extend on review §2.2's β=0
  formula (see T1 above).
