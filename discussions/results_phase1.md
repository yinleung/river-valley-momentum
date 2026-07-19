# Phase 1–4 Results Brief: the momentum-filtering mechanism is empirically visible

Status of the `idea_v1.md` execution plan. The guiding question for E1–E5 was *not* to prove the
theory but to check whether the proposed mechanism — momentum as a temporal filter that preserves
the slow river signal and suppresses the fast hill component — is visible; E6 then asks whether it
appears in a real network. **It is: all six decision gates pass**, and the finite-window filtering
theorem (T1) and the river-valley frequency-separation theorem (T3) are proved (Codex PASS).

Code: `codebases/core/` (written-once library: momentum, metrics, landscapes, probe, logging),
`codebases/scripts/run_e[1-6]_*.py` (drivers), `codebases/figures/fig_e[1-6]_*.py` (figures), and
the first-party `codebases/mlp-diagnostic/` task (Integration Contract) for E6. Records under
`codebases/results/cache/`, indexed in `results/index/runs.csv`, per-figure provenance in
`results/figures/<label>/summary.md`. Every code/theory change was reviewed by Codex (PASS).

---

## E1 — straight river-valley (Task 1.1) — **PASS**
L(x,y)=(μ/2)(x−x*)²+(λ/2)y², η·λ=1.8 (oscillatory regime). Decision gate (hill oscillation at
β=0): **72 sign changes** in g_{t,y}. HSR monotone decreasing in β: 1.00 → 0.294 → 0.055 → 0.028 →
0.0056; MSR tracks |H_β(π)|²; align(m) ≥ align(g) for every β. *Finding beyond the plan:* rms
distance to the floor is minimised at β=0.5 and **rises** for β≥0.9 — with fixed η the hill
subsystem is underdamped (closed-loop eigenvalue magnitude √β) and rings longer. This is the T6
filtering–lag tradeoff already showing up in the simplest landscape.

## E2 — curved river-valley (Task 1.2) — **PASS**
f(x)=2·sin(0.9x). River alignment to the live tangent: 0.136 (β=0) → **0.991 (β=0.9, best)** →
0.935 (β=0.99); hill-normal energy falls ~6 orders of magnitude; lag minimised at β=0.9 and rises
at β=0.99 (the bend outruns the EMA memory ~1/(1−β)). This is the **filtering–lag tradeoff** the
plan hoped for: larger β ⇒ less hill energy, but too-large β ⇒ more river-following lag.

## E3 — stochastic noise (Task 1.3) — **PASS** (12 seeds × 3 noise models)
The principled noise-suppression ratio NSR=‖m−EMA(∇L)‖²/‖ξ‖² tracks the white-noise floor
1/N_eff=(1−β)/(1+β) almost exactly (Gaussian: 1.00, 0.330, 0.053, 0.026, 0.003 vs 1.00, 0.333,
0.053, 0.026, 0.005), across Gaussian / anisotropic-hill / heavy-tailed noise. *Refinement of the
plan:* the literal NSR with the **instantaneous** target ∇L(w_t) is non-monotone (dips then rises),
because it conflates noise filtering with the deterministic lag bias; by linearity of the EMA the
clean stochastic residual is m−EMA(∇L)=EMA(ξ). Both are reported; the instantaneous version is a
second view of the T6 tradeoff.

## E4 — frequency-domain validation (Phase 2) — **PASS**
Empirical R(ω)=|m̂|/|ĝ| vs theory |H_β(ω)|=(1−β)/√(1−2β cos ω+β²) on fixed streams. Weighted-median
relative error: synthetic <0.001, curved <0.003, straight ≤0.115 (boundary/transient). High-band
*median* ratios sit within 6% of (1−β)/(1+β) (synthetic/curved) and 12% (straight,
boundary-affected). *(Correction 2026-07-03: this brief previously claimed the high-band ratios
"match to three digits" and understated the curved error — neither was supported by
`metrics.json`; caught in a WRITING.md B6 audit during the fable_dgs_v1 execution pass.)*
The straight-valley hill stream has **no low-frequency
energy** (R ill-defined there) — direct empirical support for T3's claim that hill gradients
concentrate near ω=π.

## E5 — matrix / Muon toy (Phase 3) — **PASS (key test)**
G_t=S_t+H_t+Ξ_t with slow low-rank S_t, deterministic high-frequency H_t=(−1)ᵗA, and stochastic
Ξ_t. Pre-polar O(EMA(G)) beats post-polar O(EMA(O(G))) and polar-only O(G) on subspace error and
rank-r signal alignment, in **both** the deterministic-high-frequency scenario and the stochastic
control (high-freq β=0.9: subspace error 0.114 < 0.133 < 0.508). This is the intended
generalisation of Muon's "denoise first": filter-first wins even when the disturbance is *not*
zero-mean noise but a deterministic high-frequency mode.

## E6 — real-network gradient diagnostic (Phase 4) — **PASS**
First-party `mlp-diagnostic` task: a 2-layer ReLU MLP trained full-batch by GD at the edge of
stability on a non-realizable target. The first-layer gradient stream concentrates at the Nyquist
frequency (temporal HFER=0.994 vs white-stream baseline 0.406; spectral peak ω/π=1.000) — the
river-valley hill oscillation in a real net — while the loss still descends (0.51→0.31, river
progress). Open-loop EMA filtering suppresses it per `|H_β(π)|²` (MSR match to 3–4 digits) and lifts
alignment with the slow (mean) gradient from 0.26 (raw) to 0.82 (β=0.99). The structure is
regime-dependent: sub-critical η is smooth (HFER≈0), η≈0.3 is the EoS oscillation, η≳1 diverges.

---

## Mapping to the plan's success criteria
Minimal Success (idea_v1.md §5): (1) toy demos show clear hill-oscillation filtering — **done**
(E1, E2); (2) frequency plots match the EMA magnitude response — **done** (E4); (3) finite-window
filtering theorem — **done** (`theory_t1_t3.md`, Codex PASS); (4) bridge old frequency paper → Muon
— **mechanism shown** (E4 → E5), theorem pending (T5). **Minimal Success is met.**

Strong Success (idea_v1.md §5): (1) real gradients show slow/fast decomposition — **done** (E6);
(2) momentum improves slow-gradient alignment — **done** (E6, 0.26→0.82); (3) Muon Pre-polar
advantage extends to deterministic high-frequency — **done** (E5); (4) river-alignment improvement
proved — **done** (T1 remark / T3); (5) filtering–lag tradeoff explains scheduling — **shown
empirically** (E2, E3 NSR_inst) and **with explicit constants** (T1: `ρ_high↓`, `ε_low↑`). Remaining
for full Strong Success: E7 optimizer-level tests that the mechanism metrics *predict* performance.

## Theory delivered (`discussions/theory_t1_t3.md`, Codex PASS)
- **Proposition 1** (exact windowed-EMA transfer): `m̂(ω)=H_β(ω)(ĝ(ω) − e^{-iω(T+1)} B)`, boundary
  vector `B = β m_T/(1-β)` = the terminal filter state [✓ verified to 1e-14].
- **Theorem T1** (finite-window band filtering): high-band contraction `ρ_high(β)=|H_β(ω_c)|`,
  low-band fidelity `ε_low(β)`, explicit closed forms; `ε_low↑`/`ρ_high↓` in β is the T6 tradeoff
  and yields the T4 condition `ρ_high<1−ε_low`.
- **Theorem T3** (frequency separation): triangle-inequality proof that the hill gradient peaks
  exactly at `ω=π` and the river at `ω=0`, for every window.

## 2026-07-03 execution pass (fable_dgs_v1.md program) — all gates PASS
The theorem + validation program of `fable_dgs_v1.md` §8 is executed; working log in
`fable_wlog_v1.md`, theory in `theory_cl_t2_t5.md` (Codex PASS), guides in
`codebases/core/closedloop.py`.

- **Theory**: CL-1 (closed-loop stability iff ηλ<2T_eff, flip mode at threshold, river rate
  β-free to first order), CL-2 (Var(y)=ησ²/(λ(2−ηλ/T_eff)) + d-dim loss-gap corollary), T2
  (exact hill-gradient spectrum, zero DC, monotone, concentration ≥2 iff ηλ≥1 — argmax is π for
  *every* ηλ, correcting fable_dgs_v1's "peaks iff ηλ>1"; plus the exact filtered-power ratio
  and a Chebyshev proof that monotone spectra filter at least as well as white noise), T3′
  (DC-telescoping confinement lemma, machine-precision identity), T5 (deterministic filter-first:
  exact buffer, two-sided tail, Wedin subspace bound, polar-only misidentification instance,
  post-polar period-2 limit). All numerically verified (`theory_cl_t2_t5_check.py`).
- **E8 headline** (`run_e8_headline.py`, task `valley_regimes`): β=0 diverges 12/12 at ηλ=2.5,
  β=0.9 tracks (rms 0.255 vs bend-corrected guide 0.249); tube restoration at ηλ=1.8 (escape
  1.00/0.83/0.00 at β=0/0.5/0.9; rms 0.194 vs guide 0.194); lag reversal at β=0.999.
- **Closed-loop grid** (`run_e3_heatmap.py`, task `cl2_grid`): CL-2 ratio ≈1 in 36/37 stable
  straight-valley cells (2·SE gate), exact divergence boundary; escape-vs-β curves; good-β window
  widens with λ/μ (upper edges 0.7/0.95/0.999 at λ/μ=10/100/1000, T6′ prediction).
- **E5 overlay**: clean-scenario tail inside the two-sided |ε_t| envelope (upper guide tight to
  ×1.05), Wedin bound tight to ×1.17; noisy scenario shows the stochastic floor take over.
- **E6 onset**: windowed HFER 0.025→0.98 with onset at step 200; 68% of the loss drop
  pre-onset; EMA alignment gain +0.10 pre vs +0.69 post — "filtering pays at confinement".
- **E7 optimizer-level (the last Strong-Success item) — done, PASS**: best β varies across five
  toy settings (0.5–0.95); mechanism score (buffer hill energy + lag ranks) predicts tail-loss
  rank at median ρ=+0.75 vs |ρ(β)|=0.60; MLP regime contrast — momentum's train benefit 23.3%
  at EoS (HFER 0.99) vs 0.0% sub-critical (HFER 0.02); closed-loop Muon pre-polar beats
  post-polar 3/3 on loss (0.153 vs 0.271 at β=0.95) and alignment. Schedule note: increasing-β
  is ~15× *worse* than fixed β=0.9 on the toy (no benign early phase — low-β prefix escapes);
  β-warmup is a confinement question, not a clock question.
- **Paper**: related work (Shen/Song/Cohen central flows/Andreyev/Defazio), closed-loop
  propositions, spectrum theorem, confinement lemma, β-window remark, deterministic filter-first
  theorem + polar-only proposition + post-polar remark, E8/grid/onset/E7 sections, scope-fixed
  abstract, limitations rewrite; builds to 18 pp; Codex PASS after fixes.

## 2026-07-04 execution pass (idea_v2.md program) — working log in `fable_wlog_v2.md`
The submission-hardening program of `discussions/idea_v2.md` is executed; theory in
`theory_cl3_t5b.md` (Codex PASS), new guides in `core/closedloop.py`.

- **Theory**: CL-3 (stationary hill loss ησ²/(2(2−ηλ/T_eff)) strictly decreasing in β, maximal
  relative reduction ηλ/2, five scope conditions explicit), CL-3′ (burn-in horizon
  (T_eff/2)·log(1/ε) via the exact second-moment recursion, ratio β per step), T5b
  (band-limited filter-first: per-tone identity EMA(e^{iωt})_t = H_β(ω)(e^{iωt}−β^t), exact
  buffer decomposition, disturbance contraction ρ_high(1+β^t)α, Wedin subspace bound under
  in-subspace drift, SDR gain → (1−a/σ_r)/ρ_high), B1 (boundary weight: β(1−β^T)/((1−β)T) for
  persistent streams, O(T_eff²/T) per stationary bin; failure only at T ≲ T_eff). All verified
  (`theory_cl3_t5b_check.py`).
- **E9 β-optimization sweep (`run_e9_betaopt.py`, task `beta_opt`) — all 6 gates PASS**: CL-1
  boundary exact on 3 conditionings (16/16 divergences beyond, 0/16 inside); rms/CL-2 within
  2·SE+0.02 in 174/174 comfortably-stable cells; measured maximal hill-loss reduction
  0.30/0.60/0.82/0.90 at ηλ=0.6/1.2/1.6/1.8 vs the CL-3 law ηλ/2; σ² scaling exact
  (4.00/16.00/4.00); normalization ablation — heavy-ball at fixed η_HB is monotone *increasing*
  (Spearman +1.00, on-guide 10/10) while EMA at fixed η decreases, from a shared β=0 cell.
- **E10 bend-frequency window (`run_e10_bendwindow.py`, task `bend_window`) — all 3 gates
  PASS**: confinement floor rises with k (0.3→0.9 at ηλ=1.8; 0.3→0.95 at 2.5); window narrows
  (widths 6→4 and 6→3 grid steps); U-shape at every k (best β interior, β=0.999 ≥1.5× best).
  Corrected prediction: the river speed collapses with k (0.047→0.013), so the temporal bend
  frequency k·v self-regulates and the window's top edge stays soft — curvature is paid in
  river speed; at k=1.8 the tracking window collapses entirely.
- **E7 extensions — all 5 gates PASS**: adaptive schedule arms — tube- and flip-triggered
  0.7→0.95 raises fire at step ≈11 and land with fixed-β arms (0.83/0.74 vs 0.75); the
  sub-floor adaptive control triggers only at step ≈201 and loses at 244; the floor ramp
  remains best (0.48). Muon regime contrast (gate E, recalibrated to the relative form — see
  wlog): median pre/post gap +0.220 at EoS vs +0.068 (sign-inconsistent) sub-critical;
  polar-only worst at EoS (0.298), mid-pack sub-critical.
- **E11 scale transfer (`run_e11_scale.py`, task `nanogpt`) — gates A/C/E PASS, B/D FAIL as
  operationalized (kept, reported)**: karpathy/nanoGPT dropped in pristine (sha 3adf61e),
  0.40M-param char-GPT on shakespeare_char, mini-batch, MPS, 3 seeds. Raw streams
  hill-dominated at *every* stable lr (HFER 0.82–0.84 vs white 0.40 — no smooth sub-critical
  regime, unlike the full-batch MLP); best-β benefit rises 0.8%→3.5%→11.9%→21.8% with lr
  (the CL-3 dose-response); sweep-best cell (lr=0.4, β=0.95) at 1.946 where β=0 diverges on
  1/3 seeds and survives at 2.488 (gate B wanted 3/3 — boundary is seed-marginal at batch
  32); mechanism-score ranking does NOT transfer (median within-row ρ −0.09; gate D FAIL —
  regime identification transfers, within-regime β ranking does not); Muon pre-polar wins
  3/3 and 2/3 (gaps +0.02/+0.03); val TRACKS train (ρ=0.90, best-train = best-val cell) —
  the E7 val anti-correlation was the synthetic task's axis.
- **Paper**: CL-3 corollary + burn-in remark + boundary-weight remark + band-limited theorem
  (+SDR remark); theory-to-experiment table; E9/E10 lead the experiments section; E11 section;
  normalization ablation cited; abstract/intro sharpened to the β-window claim; adaptive-arm
  discussion.

## Remaining
- E11 full-sweep numbers (run in progress) + fig_e11 + paper section fill.
- Scale beyond 0.4M params / character data; Adam-preconditioned variants.
- Optional theory: T6 formal schedule corollary; tube-restoration remark (escape ↔ CL-2 rms).
- Parameter note: E2's η·λ=1.8 pushes β=0 near instability (vivid but extreme); a gentler η would
  give a tamer baseline if a cleaner trajectory panel is wanted.
