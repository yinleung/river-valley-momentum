# plan_v5 — JMLR resubmission: theory repairs, Wisteria GPU campaign, restructure

**Status:** plan, not yet executed. Drafted 2026-07-19 from `discussions/review_v5.md` (reject-and-resubmit, 5/10) against `latex/main.pdf` (2026-07-18 build, 34 pp).
**Inputs:** `discussions/review_v5.md`, `latex/main.pdf`, `discussions/revision_v4_changelog.md`, `codebases/CODEBASE_INDEX.md`.
**Governing standards at execution time:** `latex/WRITING.md` + `latex/CRITERION.md` for prose; `codebases/CODING.md` for code; Claude executes, Codex reviews (root `CLAUDE.md`).
**Compute target:** Wisteria/BDEC-01 **Aquarius** (U. Tokyo ITC; 45 nodes × 8× NVIDIA A100 40GB SXM4, Fujitsu TCS `pjsub` scheduler). This repo will be cloned to Wisteria via git.

> **EXECUTION STATUS (2026-07-19, after revision_v5 — read before acting).** The paper-side workstreams are **DONE and Codex-judge PASSED** (3 rounds): T1, T3, T4, T5 theory repairs; the W-pass restructure (paper now 39 pp: body ≈32 + App A proofs / App B toy checks); W4 branch (a) executed (E7 schedule numbers removed — §8 item 1 resolved). `discussions/revision_v5_changelog.md` is the authoritative record and **supersedes** this plan's §2 T1 pre-check narrative (sign error; see the correction in §2 T1) and the §8-1/W4 status text. **REMAINING (the live scope of this plan):** §3.1 Phase 0 bring-up (B1–B5) → G1–G6 campaign (§3.2, order and budgets §3.4) → paper's experimental sections from the results; T2/T6 stretch theory; §5 public release at resubmission; response letter. Executing agents: do NOT redo paper work and do NOT edit `latex/` during campaign sessions; start at Phase 0.

---

## 0. GOAL (paste-ready for the executing agent)

> Convert the JMLR reject-and-resubmit verdict in `discussions/review_v5.md` into a resubmission-ready revision. Three fronts, in dependency order. **(1) Theory:** repair the two broken claims — Proposition 5(b)'s curved-valley exactness (couple the river into the hill recursion or restrict to constant slope) and the Theorem 4→Muon transfer (replace the "pre-polar update inherits the bound" sentence with genuine update-level theorems: full-rank polar perturbation, and a Lipschitz theorem for the finite Newton–Schulz map that practical Muon actually runs) — plus the smaller reframings (acceleration is complementary, not identical; heavy-ball-normalization corollary; E7 schedule claims documented or removed). Every new statement gets a numeric check script in `discussions/` and a Codex review before it enters `latex/main.tex`. **(2) Experiments:** execute the G1–G6 GPU campaign of §3 on Wisteria — ResNet-18/CIFAR LR×β mechanism sweeps, state-vs-residual decomposition with Hessian-subspace analysis, real-network forced-frequency controls, 20M→124M GPT transfer, practical all-layer Newton–Schulz Muon pre/post/polar comparison, and the finite-β window — under the two-protocol normalization rule, divergence-counted statistics, and predeclared gates. **(3) Paper:** restructure per §4 (exact/approximation/empirics separation, GPU results promoted to the main experimental sections, toy checks demoted to appendix, abstract rewritten without "confirm every prediction", public-artifact reproducibility statement) and assemble the resubmission package with a point-by-point response letter. Do not start a later front's paper edits before the earlier front's gates pass; surface every gate failure to Leon rather than silently weakening a claim.

---

## 1. What the review requires (distilled)

Verdict: straight-valley core (Prop 1, Thm 1–3, Cor 1, Lem 1, Props 2–4, Cor 2) is *"the strongest part of the paper"* and *"largely correct and potentially publishable"*. The rejection hangs on six numbered conditions (review §Recommendation):

| # | Condition | Workstream here |
|---|-----------|-----------------|
| R1 | Correct Prop 5; separate exact / frozen-coefficient / empirical statements | T1, T2, W2 |
| R2 | Replace the polar-subspace argument with a genuine theorem about the polar **output**, or drop the Muon theorem claim | T3 |
| R3 | Reframe normalization-dependent results so practical scope is unmistakable | T5, W1 |
| R4 | Document or remove the E7 schedule-arm claims (factors 15/118, detectors, warmup arms on p. 32) | W4 |
| R5 | Public code + substantially stronger neural-network and Muon experiments | §3 (G1–G6), §5 |
| R6 | Shorten; moderate "experiments confirm every prediction" and "the three views coincide" | T4, W1, W3 |

Reviewer's **minimum viable experimental package** (review §10) = our campaign's acceptance checklist:

1. ResNet-18/CIFAR LR×β sweep showing momentum benefit grows in the high-curvature/high-HFER regime → **G1**
2. Large-batch state-vs-residual decomposition: high band is trajectory-generated → **G2**
3. Hessian-subspace analysis: high-frequency state gradients concentrate in sharp directions → **G2**
4. Real-network forced-frequency intervention: passband neutrality, Nyquist suppression, resonant harm → **G3**
5. Transformer EMA-SGDM training confirming regime transfer → **G4**
6. Practical all-layer Muon (Newton–Schulz): pre-polar vs post-polar vs polar-only → **G5**
7. A finite-β window result, both edges, at scale → **G6**

Recommended order (review §Recommended order, adopted): G1 → G2 → G3 → G4 → G5 → confirmations (G6 + 124M runs) — *"avoid spending substantial GPU compute on a scaling experiment before knowing whether the key measurable predictions survive in an ordinary neural network."*

---

## 2. Theory workstream (T1–T6)

House rule for every item: derivation drafted in a `discussions/theory_*_v5.md` note **with a paired `*_check.py`** (sympy/numpy adversarial check, same convention as `theory_fr_cv_be{,_check.py}`), Codex-reviewed, only then ported into `main.tex`. The proof-writer skill drafts; the check script is the gate.

### T1 — Repair Proposition 5(b) (curved valley) — BLOCKING

**Problem (review §2.2).** For the linear floor `f(x)=cx` with quadratic river profile `φ(x)=μ/2·(x−x⋆)²`, the forcing `−c·φ′(x_t)` is stochastic through `x_t`; `(x_t, d_t)` is a coupled linear stochastic system, so applying Prop 3 to `d_t` alone is not exact. The reviewer exhibits the β=0 full-Lyapunov variance, which differs from the paper's `ησ²/(λ(2−ηλ_loc/T_eff))` and converges to it only as river curvature → 0. The claim *"nothing is frozen or truncated: the reduction is exact"* (p. 14) is false as stated.

**Fix — a three-tier replacement, exactly the exact/approx separation R1 demands:**

- **Prop 5(b-i), exact, constant slope:** for `φ′ ≡ const` (constant-slope river — the "tilted plane" floor) the current formulas are exact verbatim. Trivial re-scoping; the current proof already goes through since the forcing is genuinely constant.
- **Prop 5(b-ii), exact, quadratic river:** solve the full coupled system under EMA momentum — state `(x_t, m^x_t, d_t, m^d_t)` (4-dim linear recursion driven by `(ξ^x, ξ^y)`), stationary covariance by discrete Lyapunov equation, **closed-form `Var(d_∞)` and `E[d_∞]` via sympy**. **Pre-verified 2026-07-19 (`discussions/plan_v5_t1_precheck.py`), β=0 case:** the exact 2D Lyapunov solution (per-coordinate iid gradient noise, so the d-equation noise is `ξ^y − cξ^x` with variance `σ²(1+c²)`) confirms (i) the paper's Prop 5(b) formula is **exactly the μ→0 limit** of the true variance — the reviewer's substantive point stands, the "exact" claim is false with O(c²μ) corrections; but (ii) the reviewer's own displayed formula `ησ²(2−ημ)/(λ[4−2ηλ(1+c²)−2ημ+η²λμ])` matches **none** of three natural noise placements (per-coordinate iid, y-only, hill-scalar chain-rule) — it differs from the exact solution by its own O(c²μ) term (numeric cell η=.1, λ=10, μ=.1, c=.5: exact 0.05337, reviewer 0.05342, paper 0.05333). So the T1 deliverable states OUR general-covariance closed form, and the response letter notes the convention-dependence politely, with the derivation attached. Remaining work: extend the pre-check's β=0 Lyapunov to the 4-dim EMA-momentum state and take the μ→0 and β→0 limits as gates.
  **CORRECTION (2026-07-19, revision_v5 — supersedes clause (ii) above):** the pre-check had a sign error in the noise cross-covariance (+c where `Cov(ξ^y − cξ^x, ξ^x) = −cσ²`). With the correct sign the exact 2D solution **equals the reviewer's displayed formula exactly** (audit cell 199/3725 ≈ 0.0534228; independently re-verified). Clause (i) stands — the paper's old formula is the μ→0 limit and the "exact" claim was false. T1 has since been executed as the exact modal solution (`theory_cv2_coupled.md` + check, Codex PASS, ported to the paper). The response letter **concedes-and-extends**; it does not dispute. Do not act on clause (ii).
- **Prop 5(b-iii), quasi-static, general `f`:** the current frozen-coefficient statement relabeled as an approximation with the remainder stated (the `kv_riv T_eff ≲ 1` validity margin already in the paper becomes a hypothesis, not an apology), and E10's floor agreement cited as its empirical calibration, not as proof.

**Also under T1 (review §2.2 last ¶):** the pointwise frozen threshold `ηλ(1+f′²) < 2T_eff` must nowhere read as a global stability theorem. Sweep abstract + §6 + Discussion for over-definite curved-valley phrasing; the honest global statement is T2's.

**Artifacts:** `discussions/theory_cv2_coupled.md` + `_check.py`; main.tex Prop 5 rewrite; E10/E8 figure captions updated if the new exact formula is overlaid (toy CPU re-runs are cheap and use the existing drivers).
**Gate:** check script PASS on symbolic identities + 10⁶-step simulation agreement within Monte-Carlo error on a (η, λ, μ, c, β) grid including the reviewer's β=0 cell. If the 4-dim closed form is intractably large, the primary exact statement may be the matrix-Lyapunov form (`P = APA^⊤ + Q` with the system matrices displayed) plus verified special limits (β=0, μ→0) and numeric evaluation — a huge printed rational is not required (Codex suggestion).

### T2 — A real global curved-valley stability statement (stretch, high reward)

**Target:** sufficient condition converting "frozen stability pointwise" into trajectory stability. Strategy: slowly-time-varying linear systems (Desoer-style): if the frozen AR(2) at every visited `x` has margin `ηλ_loc(x) ≤ (1−ρ)·2T_eff` and the per-step coefficient drift is bounded (`|λ_loc(x_{t+1})−λ_loc(x_t)|` small against a function of ρ and β — which the paper's own `kv_riv T_eff` margin supplies), then the product of frozen matrices is exponentially stable, with explicit constants from a quadratic Lyapunov metric of the frozen companion matrix. **Fallback** if the general bound's constants come out vacuous: piecewise-constant `λ_loc` with dwell-time hypothesis (switching-system dwell-time lemma) — still a theorem, honestly scoped.
**Gate:** proof verified by proof-writer + adversarial numeric search for counterexamples (random slowly-varying `λ_loc` sequences violating no hypothesis, checked for escape). If both versions stall in a week of effort, downgrade: keep "open" status but add the theorem-shaped conjecture with the numeric evidence — and say so in the response letter.

### T3 — Genuine update-level Muon theorems — BLOCKING

**Problem (review §2.3).** Thm 4/5 control the buffer `M_t`; exact `polar(·)` flattens singular values to 1, so subspace recovery does not transfer to the update; the closing sentence of Thm 4 (*"the pre-polar update inherits the bound"*) and Prop 6's rank-r readout do not describe Muon as run. The reviewer's own orthogonal rank-one construction kills any amplitude-based transfer for exact polar.

**Fix — three results, and a repositioning that turns the reviewer's counterexample into part of our story:**

- **T3a (full-rank polar perturbation).** Assume `σ_min(S) > 0` (square or full-rank rectangular). With `M_t = (1−β^t)S + ε_t A` and `|ε_t| ≤ (1−β)(1+β^t)/(1+β) → 1/T_eff`, apply a *bona fide* polar-factor perturbation bound (R.-C. Li's `‖U_A − U_B‖_F ≤ 2/(σ_min(A)+σ_min(B))·‖A−B‖_F` for the unitary polar factor; verify the exact rectangular statement and cite Li 1995 SIMAX / Higham *Functions of Matrices* ch. 8):
  `‖polar(M_t) − polar(S)‖_F ≤ 2‖A‖_F·|ε_t| / (σ_min(M_t)+σ_min((1−β^t)S))` → an **update-level** error dying at rate `1/T_eff`. This is the theorem the reviewer asked for under repair option "full-rank signal".
- **T3b (finite Newton–Schulz map — the practical Muon theorem).** Practical Muon never runs exact polar: it runs `k` (=5) iterations of the odd polynomial `p(X) = aX + b X(X^⊤X) + c X(X^⊤X)²` (Jordan's a=3.4445, b=−4.7750, c=2.0315, bf16) on the Frobenius-normalized buffer `M/(‖M‖_F + 10⁻⁷)`. The composed scalar map `h_NS = p^{∘k}` has `h_NS(0)=0` with slope-at-0 `a^k ≈ 485` (k=5) — unlike the exact polar's jump to 1, small inputs stay proportionally small. **Codex-corrected scope (plan review, 2026-07-19): `h_NS` is NOT monotone on [0,1] — `p′(1) = a+3b+5c ≈ −0.72 < 0`, the NS polynomial deliberately overshoots — so this does NOT sit inside Remark 12's monotone-map class, and the theorem must not lean on singular-value ordering.** The correct statement is a **bounded-annulus Lipschitz theorem for the exact deployed operator** `D(M) = NS_k(M/(‖M‖_F+ε))`: on the operating annulus `0 < r_low ≤ ‖M‖_F ≤ r_high` — the lower bound supplied by the signal, `r_low ≈ (1−β^t)σ_r(S) − |ε_t|‖A‖ > 0`, stated as a hypothesis — the fixed odd matrix polynomial composed with the normalization is Lipschitz with a numerically computed worst-case constant `L_NS(a,b,c,k,r_low)`, hence `‖D(M_t) − D(X_t)‖_F ≤ L_NS·|ε_t|·‖A‖_F/r-scale` — the buffer's `1/T_eff` recovery **transfers to the actual update** up to `L_NS`, with the exact-polar pathology exposed as the `r_low → 0` (equivalently `L → ∞`) limit. The check script computes `L_NS` on the annulus and verifies the bound on random instances, including near the annulus floor.
- **T3c (optional, unifying):** regularized polar `D_ε(M) = M(M^⊤M+ε²I)^{−1/2}` as the continuum object between T3a and T3b (NS_k ≈ D_ε with an effective ε_k); include only if it shortens rather than lengthens §7.
- **Reposition Prop 6** as the *necessity* example: under exact polar, no amplitude-based update bound is possible — which is exactly why T3a needs `σ_min(S)>0` and T3b needs finite NS. The reviewer's construction becomes the paper's own sharpness example (it nearly is already; the framing changes from "Muon inherits via ordering" to "here is why the hypotheses are necessary").
- **Rewrite the transfer sentences:** Thm 4's closing sentence, the "filter-first theorems for the polar map" phrasing (abstract, contributions, §7 intro, Discussion), Rem 12's tie-reading sentence. New register: buffer theorems (Thm 4/5/Cor 3) + update theorems (new T3a/T3b) + necessity example (Prop 6). **Remark 12 itself needs a scope correction:** its monotone class does NOT contain finite-NS (nonmonotone overshoot, see T3b) — the remark stays for the monotone family (raw buffer, powers σ^{1−q}, exact polar), and a sentence notes that the deployed NS map sits outside it, which is exactly why T3b replaces the ordering route with the Lipschitz route.

**Experimental hook (feeds G5):** measure the empirical singular-value transfer `h` of the deployed NS implementation once, plot against `h_NS`; quote `1/L_NS` as the predicted disturbance-amplitude knee in the G5 high-HFER arm.
**Artifacts:** `discussions/theory_polar_update.md` + `_check.py` (numeric: random S, A over ranks/gaps; verify T3a/T3b bounds hold and are within ~an order of magnitude of tight; reproduce the reviewer's counterexample and show T3b's bound degrades gracefully there while exact polar jumps).
**Gate:** check PASS; Codex confirms no remaining sentence claims exact-polar update transfer.

### T4 — Acceleration is complementary, not identical (review §2.4)

Rewrite the Discussion's "three views coincide" paragraph and the abstract's framing: EMA DC gain is exactly 1 (no acceleration); heavy-ball's `1/(1−β)` DC gain is a normalization fact; condition-number acceleration (Polyak/Nesterov √κ) arises from the coupled second-order dynamics and is **not** delivered by open-loop filtering. State what filtering does and does not explain; cite Polyak (already [10]), Sutskever et al. (already [13]), and add Lessard–Recht–Packard (IQC analysis) as the control-theoretic treatment of acceleration proper. One new short remark in §6 can note that our AR(2) *is* the heavy-ball second-order system in the hill coordinate — the two stories share hardware but answer different questions (rate vs. stationary error/stability).

### T5 — Heavy-ball-normalization corollary (review §8, R3)

Promote the conversion `η_HB = η(1−β)` from remark-level to a displayed corollary: restate Prop 2 (threshold), Prop 3/Cor 2 (variance/loss — direction of the β-sweep reverses at fixed η_HB), Prop 4 (gains) under fixed `η_HB`. E9(f) already measures the reversal; the corollary makes the scope theorem-level. Abstract sentence: benefits are claims about the EMA convention at fixed η; under fixed-η_HB the sweep direction inverts (already half-present; make it unmissable).

### T6 — Noise-scope extension (optional, low priority)

Additive/state-independent noise is a declared limitation; a cheap robustness lemma (stability threshold as inequality under bounded state-dependent variance `σ²(y) ≤ σ̄²(1+κy²)`, small κ) would blunt the "assumptions very close to proofs" criticism. Only if time permits after T1–T3; otherwise leave in Limitations.

---

## 3. GPU experimental campaign (Wisteria)

### 3.0 Protocol invariants (apply to every G-experiment)

1. **Two normalization protocols, never mixed in one curve** (review §8): **P-thy** = normalized EMA at fixed η across β (tests the propositions); **P-prac** = conventional SGDM/Muon with LR tuned per β (or converted via `η_HB = η(1−β)`), measuring practitioner-relevant benefit. Every figure states its protocol in the caption.
2. **Statistics (review §9):** ≥3 seeds everywhere, 5 for headline claims; **divergence counts as failure** and enters the criterion (no surviving-seed means — fixes the lr=0.4 E11 issue); identical batch order + init across paired arms (fix DDP sharding seed too); confidence intervals on every headline number; full (η, β) grid reported including failed cells; exploratory sweeps and confirmatory runs labeled as such in the run config (`stage: explore|confirm`) at launch time, not post hoc.
3. **Spectral hygiene (review §3.4, §9):** diagnostic windows ≫ T_eff (β=0.99 ⇒ T_eff=199 ⇒ window ≥ 2048; otherwise ≥ max(1024, 10·T_eff)); rectangular windows primary (they match the finite-window theorems), Hann-tapered secondary for robustness; HFER reported at cutoffs {0.5π, 0.6π, 0.7π}; whiteness claims require autocorrelation (Ljung–Box), cross-step covariance, and surrogate white-noise confidence bands — HFER alone earns nothing (review weakness #4).
4. **Gates predeclared** per experiment before launch (house style, keep); PASS/FAIL reported either way — the paper's negative-results habit is a credited strength.
5. **Ratio denominators stated explicitly** in every share metric (Codex B6 lesson: state-share vs direct-of-raw + cross terms, as E12 already does).
6. **Assumptions are instrumented, not assumed** (Schedule-Free's condition-(15) practice; Everett et al.'s alignment measurements): every run logs the theory's operating quantities — `χ = η(1−β)λ̂`, window/T_eff ratio, Wedin separation `δ_t` in Muon arms, `k·v_riv·T_eff` on curved toys — so each figure can show where the hypotheses hold or break. Assumption-violation curves are results, not embarrassments.
7. **Fit hygiene** (Chinchilla / data-constrained-scaling convention): wherever a curve or law is fitted (spectra, benefit-vs-`ηλ̂` trends, mechanism-score calibration), state how many runs feed the fit, hold out a validation subset, and quote held-out error.
8. **Schedule-horizon matching — scoped to P-prac performance comparisons** (Chinchilla App B, the confound that inflated Kaplan's exponents; No-Train-No-Gain; Wen et al.'s curve-crossing pitfall): in any P-prac recipe/performance comparison, every arm's LR decay completes within its budget and is re-tuned per budget, and comparisons are made **only at final checkpoints of fully-decayed schedules** — never mid-schedule, where curves cross. Explicitly EXEMPT (Codex re-review fix): P-thy fixed-η mechanism sweeps (constant η is the point), G2's mid-training diagnostic windows, and G3's frozen-schedule forced branches — those follow review §8's theory-faithful protocol instead.
9. **Per-arm tuning, never shared HPs across optimizers — same P-prac scope** (Wilson et al. / NQM / Fantastic-Pretraining-Optimizers): in tuned comparisons, each arm's LR grid is log-spaced and extended until the optimum is interior (grids published in the appendix), each arm gets its own decay scheme, and framework-default configs appear only as separately-labeled arms. P-thy sweeps hold η fixed across β by design and are labeled mechanism tests, not tuned comparisons.
10. **Exact-simulator pre-registration** (NQM discipline): before each G-experiment launches, the AR(2)/DFT exact simulator (`core/closedloop.py` guides) generates the predicted curve shapes and orderings (divergence-boundary shape in the (η,β) plane, resonance-peak location, DC/Nyquist gain ratios), committed to the repo as predictions-as-code; GPU results then confirm or falsify **shape and ordering, never constants**.

### 3.1 Wisteria bring-up (Phase 0)

Facts to plan around (verify each on first login — marked ✓ = public spec, ? = confirm):

- ✓ Aquarius: 45 nodes × (2× Xeon 8360Y + 8× A100-40GB SXM4, NVSwitch); Lustre-FEFS `/work/<group>/<user>` for data + runs, small `/home`.
- ✓ Scheduler: Fujitsu TCS — `pjsub`/`pjstat`/`pjdel`, `#PJM` directives; Aquarius resource groups include partial-node **share** (1/2/4 GPUs) and full-node **regular-a** / **short-a**. ? exact group names, elapse ceilings (typically share/regular 24–48 h, short ≤ 2 h), node-hour budget of Leon's group, concurrent-job caps.
- ✓ **Compute nodes have no general outbound internet** → all pip installs, dataset downloads, tokenizer assets staged from the login node (or rsync'd from the laptop). Our logger is already offline-first (`core/logging.py` writes `results/cache/` locally — the no-wandb adaptation becomes exactly right here); figures continue to render locally from a synced cache.
- Software: python ≥3.11 venv (or miniforge) on `/work`; torch ≥2.4 cu12x wheels staged; NO source builds unless forced. `nvidia-smi`-verified smoke job on `share` (1 GPU) is the first pjsub submission.

Bring-up deliverables (all under existing standards — new probes in **new** modules, additive to `core/`):

- **B1 Environment:** `codebases/scripts/wisteria/` — `setup_env.sh`, pjsub templates (`tpl_share_1gpu.sh`, `tpl_node_8gpu.sh` with `#PJM -g <group> -L rscgrp=…,elapse=…`, torchrun for DDP), `stage_data.sh` (CIFAR-10/100 tarballs; FineWeb-Edu sample-10BT **or** OpenWebText pre-tokenized to uint16 shards nanoGPT-style — tokenize on login node or laptop, ship shards), `sync_results.sh` (rsync `results/cache/` + `results/index/` back).
- **B2 CUDA port:** device policy (cuda→mps→cpu), bf16 autocast for training / fp32 for probes, determinism settings (`torch.use_deterministic_algorithms` where kernel support allows, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seeded generators, fixed data order). Where exact determinism is impossible (some cuDNN paths), keep the standing discipline: **quote numbers from cached `arrays.npz`, never from re-runs** (infra memory lesson).
- **B3 code_sha → git SHA:** once this repo is git-synced (§5), `core/logging.py` records `git rev-parse --short HEAD` (+ dirty flag) alongside the content hash (content hash kept as the cache key for dirty trees). **Batch all `core/` edits (B2+B3) in one commit** before the first GPU run — the code_sha churn rule says core edits invalidate future run ids, so do them once; existing cached toy runs stay valid (their configs pin the sha that produced them) and are NOT re-run unless their quoted numbers change. **Acceptance (Codex):** Wisteria runs preserve the full CODING.md Pillar-3 run identity — name `<task>_<probe>_<key-params>_<gitsha8>`, group/job_type/tags, full config incl. git SHA and `stage: explore|confirm` — and the complete artifact mirror (`config.json`/`metrics.json`/`arrays.npz` + `runs.csv` row), at parity with local runs.
- **B4 Throughput calibration:** one table (model × batch × GPUs → steps/s, tokens/s, probe overhead multiplier) feeding the §3.4 budget; calibrate before committing to grids.
- **B5 New instrumentation (new modules; adapters wrap, never edit upstream):**
  - `core/spectral_stream.py` — streaming windowed-DFT/Goertzel band-energy accumulators so full-window raw storage is optional; per-step JL sketches (fixed Gaussian projection, k=64–256) of full-model and per-layer-group gradients for temporal spectra at scale; full raw streams stored only for designated probed matrices (E11 pattern, extended to ~3–6 matrices across depth).
  - `core/hvp.py` — CUDA HVP: layer-restricted block power iteration (port of the working MPS path) + full-model Lanczos top-k (k=16) on a fixed probe batch; λ̂_max tracker cheap enough to run every ~200 steps.
  - `core/forced.py` — gradient-injection hook `g̃_t = g_t + A cos(ωt)·v` with v refreshed from `hvp.py`, plus Goertzel lock-in readout at the forcing frequency (E13's measurement transplanted).
  - `core/lbprobe.py` — fixed large-batch probe gradient `ḡ^LB_t` at the visited iterate (E12's split) with the state/residual/cross accounting built in.
  - New tasks: `codebases/resnet-cifar/` (torchvision ResNet-18 upstream-free reference implementation or pytorch-cifar dropped into `upstream/`), `codebases/nanogpt-gpu/` config extension of the existing `nanogpt` task (GPT-2 BPE, 20M/124M configs, DDP). Integration per the Contract; `AGENT_NOTES.md` + smoke test + `CODEBASE_INDEX.md` row each.
  - `scripts/preflight_stream.py` — a few-hundred-step gradient-stream PSD/autocorrelation/HFER check (the μTransfer coord-check analogue) run on every new task/scale BEFORE grid launches, verifying the low-pass/river-valley regime is even present; released with the public artifact as the paper's practical preflight recipe.

### 3.2 Experiment cards G1–G6

Each card: design → measures → predeclared gates (the falsifiable predictions from review §1's hypothesis table) → budget (A100-hours, refined by B4).

#### G1 — ResNet-18 / CIFAR closed-loop LR×β sweep (review Exp A; package #1)

- **Setup:** ResNet-18, CIFAR-10 primary; CIFAR-100 as a **thinned complete sweep** (3 LRs × 5 β × 3 seeds, +~15 h) so the mechanism claim is not single-dataset — the response letter states its thinned-control status (Codex suggestion). Primary mechanism runs without augmentation (or fixed augmentation sequence shared across arms). Coarse LR scan first (β=0, 8–10 points, 1 seed) to place four regimes: subcritical / intermediate / near-edge / β=0-unstable-but-β>0-stable. Then the grid: β ∈ {0, 0.5, 0.9, 0.95, 0.99} × 6 LRs spanning the regimes × 3 seeds (5 at headline cells), protocol P-thy; a P-prac companion (per-β tuned LR, 3-point tune each) on the two interesting regimes. BatchNorm caveat: one BN-free control arm (GroupNorm ResNet-18) at the headline cells, since Hessian/sharpness claims under BN draw fire.
- **Measures:** train/val loss + accuracy; divergence/spike flags; grad & momentum norms; probed-layer temporal spectra + sketch spectra (3 windows: early / EoS-onset / late); HFER (3 cutoffs); MSR; alignment(m, ḡ^LB); λ̂_max trace measured EoS-protocol-style (Lanczos-HVP on a fixed abridged ~5k-example subset, declared cadence; between-iterate sharpness on BN nets) **with the predicted β-shifted equilibrium line `2T_eff/η = 2(1+β)/((1−β)η)` overlaid per (η,β) cell** — the conversion to Cohen/Goh's heavy-ball MSS `(2+2β)/η_HB` stated once, and captions kept protocol-specific (EMA plots use `2T_eff/η`, conventional-HB plots use `(2+2β)/η_HB`, never mixed — Codex caveat); `ηλ̂_max`; projection of ḡ^LB onto top-Hessian subspace.
- **Gates:** (a) momentum benefit (best-β − β=0, divergence-counted) increases with `ηλ̂_max` and with state-HFER (the review's "strongest plot": benefit vs `ηλ̂_max`, not vs β); (b) some LR exists where β=0 diverges on all seeds and β ≥ 0.9 trains (stability extension); (c) at subcritical LR the β-effect is small (≤ noise band); (d) in EoS-regime cells, measured sharpness equilibrates near the β-shifted threshold line — the momentum edge-of-stability overlay (the Cohen-template headline figure). Sign-flip of any gate = report as negative result, revisit theory scope before G4.
- **Budget:** ~110 h (grid 90 runs × ~0.6 h + scan + P-prac + 5-seed top-ups + thinned CIFAR-100 sweep).

#### G2 — State vs. residual decomposition + Hessian subspace (review Exp B; package #2–3)

- **Setup:** on ~12 selected G1 runs spanning regimes × β — **paired decompositions on the actual β trajectories** (Codex plan-review fix: not β=0-only): the raw stream is defined as the pre-EMA mini-batch gradient `g^mb_t`, which exists on every arm, so the split `g^mb = ḡ^LB + ξ^res` is recorded along β=0 stable cells AND along β ≥ 0.9 momentum-stabilized cells — including regimes where β=0 diverges, which is precisely where the mechanism claim lives. 3 diagnostic windows each (early / EoS-onset / late), window length per §3.0.3; per-step `ḡ^LB` on a fixed 4096-image probe batch at every window step; top-16 eigenpairs at window midpoints (Lanczos, probe batch).
- **Analyses:** spectra of `g^mb`, `ḡ^LB`, `ξ^res`; high-band state share **with explicit denominators + cross terms**; concentration of state high band in P_top vs complement; per-direction HFER vs restricted `ηλ_i`; whiteness of `ξ^res` via Ljung–Box + surrogate bands + cutoff sensitivity + Hann robustness.
- **Gates (the review's four support conditions verbatim):** (1) most high-band energy in `ḡ^LB`, not `ξ^res`; (2) high-band state component disproportionately in P_top; (3) low-frequency content relatively stronger outside the sharp subspace; (4) G1's measured momentum benefit grows with this separation. Fail ⇒ the river-valley reading does not transfer to CIFAR — stop, write the negative result, rescope the paper's claims (this is the campaign's main kill-switch, per the review's ordering rationale).
- **Budget:** ~55 h (probe overhead ≈ probe-batch/train-batch per step over ~2k-step windows).

#### G3 — Forced-frequency controls on a real network (review Exp C; package #4)

- **Setup:** branch from common checkpoints at a mid-training regime (one subcritical, one near-edge): estimate v = top Hessian eigenvector; inject `A cos(ωt)·v` with A calibrated to ~10–30% of the rms aligned gradient (diagnostic, not dominating; A-sensitivity checked at one cell); forcing frequencies fixed per checkpoint, not per arm (Codex plan-review fix — an explicit β×ω grid): ω ∈ {0.02 (passband), θ̂ at β=0.9 (resonance computed from λ̂), 0.6π, π}, each run across ALL β arms β ∈ {0, 0.5, 0.9, 0.95, 0.99}, so matched-vs-mismatched resonance is read within one ω row (harm peaks at the β whose θ_β matches the forcing; relief elsewhere); plus one extra arm at θ̂ for β=0.99 to confirm the peak moves with β. ~500 diagnostic steps per arm; 3 seeds; frozen LR schedule.
- **Measures:** lock-in amplitude of ⟨w_t, v⟩ and of the v-projected gradient at ω (Goertzel); loss degradation vs β; overlay parameter-free `|G_β(ω)|` guides computed from measured λ̂ (E13 methodology at network scale).
- **Gates:** passband arm ≈ β-flat; Nyquist arm strongly attenuated with β; resonance arm *hurt* at the matched β and relieved at mismatched β. This is the signature mechanism experiment — its PASS carries the "frequency-specific, not variance-generic" claim at network scale.
- **Budget:** ~15 h.

#### G4 — Language-model transfer: 20M diagnostic → 124M confirmation (review Exp D; package #5)

- **Setup (diagnostic):** ~20M-param decoder-only GPT (≈6 layers, d≈384, GPT-2 BPE, seq 1024) on FineWeb-Edu sample / OpenWebText shards; EMA-SGDM **first** (isolates temporal filtering from adaptivity; AdamW-preconditioned streams stay out of headline claims — scope per Limitations); (η, β) sweep under P-thy + P-prac, β grid as G1, 4 LRs, 3 seeds; probes: 3–6 matrices across depth, sketch spectra, `ḡ^LB` windows (batch-level probe), layerwise HFER, λ̂ restricted + full-model.
- **Setup (ladder + confirmation):** the scale axis is a ladder, not one blob (a 0.4M char-GPT sits far below the field's credibility floor — Schedule-Free's "small" LM benchmark is nanoGPT-124M): 20M diagnostic → optional ~60M mid-rung (thinned grid; include if the 20M trends are clean, to show the trend is monotone in scale) → 124M confirmation, ~2.5–5B tokens, 8-GPU DDP single node, **5 configurations only** (low-HFER regime, high-HFER regime, predicted-best moderate β, excessive β, β=0-unstable-stabilized), 3 seeds. **Pre-registered prediction (Chinchilla discipline):** the predicted-best (η, β) cell at 124M is computed from the 20M(/60M) fits and written into the run config BEFORE the confirmation launches; the confirmation tests that prediction and is reported as such either way. No exploration at this scale.
- **Measures:** loss/perplexity, val loss, tokens-to-target, divergence rate, spectra/HFER by layer & regime, sharpness proxies, benefit-vs-LR curve. **Deliverable figure (μTransfer-style invariance):** the (η, β) optimum across the 20M→124M ladder plotted in naive coordinates (η_HB, β) vs filter-corrected coordinates (effective step η, T_eff) — optimum stable in corrected coordinates and drifting in naive ones is the most legible "theory predicted this" artifact; `ηλ̂_max` is logged alongside so scale-dependent curvature drift is not mistaken for a coordinate-transfer failure (Codex caveat).
- **Gates (Codex-corrected to the review's actual hypothesis):** (a) in high-HFER / near-edge regimes the state stream carries high-band content and the best-β benefit is large; (b) in low-HFER / subcritical regimes the stream is smooth and the β-effect small — this is the **null control**, not a failure; (c) the benefit grows toward the stability edge as a regime-level trend (strict monotonicity NOT required; layer heterogeneity reported as findings, per the review's own warning). E11's "hill-dominated at every stable LR" is a 0.4M-scale observation to be re-measured and reported, not a pass/fail gate.
- **Budget:** ~75 h diagnostic + ~90 h confirmation.

#### G5 — Practical Muon filter-first (review Exp E; package #6)

- **Setup — two clearly separated blocks (Codex re-review fix: momentum conventions must not be conflated):**
  - **(i) Mechanism arms — the review's pipelines under one explicit EMA recursion, β swept.** 124M GPT; **four pipelines** sharing the SAME NS implementation (k=5, coefficients (3.4445, −4.7750, 2.0315), bf16, `‖·‖_F`-normalized input): pre-polar `NS(EMA_β(G))`, post-polar `EMA_β(NS(G))`, polar-only `NS(G)`, raw-momentum control `EMA_β(G)`; β ∈ {0.5, 0.9, 0.95}; layer split held fixed across arms (NS on hidden ≥2D matrices, AdamW on embeddings/lm_head/1D params); shape-scaling convention stated (default Kimi RMS-matching `0.2·√max(rows,cols)` so AdamW-tuned LR/WD transfer; repo `max(1,rows/cols)^0.5` as the speedrun alternative); two regimes (high-HFER: large LR; low-HFER: small LR) from G4's map; identical batch sequences; compute-equalized (NS cost identical across arms; step budgets matched); LR per pipeline coarse-tuned at 20M (3-point), transferred to 124M.
  - **(ii) External baselines — as-shipped conventions, tuned, NOT rows of the β-sweep.** Community Muon exactly as deployed (Nesterov buffer, `orthogonalize(μM_t + G_t)`, μ=0.95, weight decay on long runs, NS wall-clock overhead reported) and a tuned AdamW reference; the modded-nanogpt speedrun configuration is the externally-tuned reference, so at least one comparison is independent of our own tuning (AlgoPerf-style fairness). These anchor practical relevance for the mechanism result.
  - 3 seeds (5 on the headline pair). Per-arm hyperparameters in an appendix table (Schedule-Free App-G convention).
- **Measures (update-level, per review):** alignment of the actual update with `polar(Ḡ^LB)` and with a large-batch update proxy; virtual-step descent on a held eval batch; update-stream HFER; singular-subspace stability of the update; train/val loss; tokens-to-target. Plus the **T3b bridge**: measured NS singular-value transfer curve vs `h_NS`; disturbance-amplitude knee vs predicted `1/L_NS`.
- **Gates (review's falsifiable regime interaction):** high-HFER regime: pre-polar > post-polar ≈ polar-only on update quality and optimization; low-HFER regime: the gap shrinks/vanishes. The interaction, not a uniform pre-polar win, is the claim. Theory and experiment reported as separate evidence lines (per review caveat) — T3a/T3b are the theorems; G5 is the practical test. Note polar-only ≡ momentum-off Muon ≡ "instantaneous Shampoo" — the natural no-filter control, citable as such. **Positioning:** we measure mechanism (what filtering does to the polar step's input and output), not the headline speedup dispute (Kimi's ~2× vs Wen et al.'s ~1.1–1.4× under fully-tuned AdamW) — cite both sides, report our compute-matched numbers and the trend across our two scales, adjudicate nothing.
- **Budget:** ~30 h (20M tunes) + ~145 h (124M arms) ≈ 175 h.

#### G6 — The finite-β window at scale (review Exp F; package #7)

- **Setup:** dense β grid {0, .3, .5, .7, .8, .9, .95, .97, .99, .995} at two LRs (mid + near-edge), ResNet-18 and 20M GPT, 3 seeds, P-thy (P-prac at the near-edge LR); plus a **horizon-sensitivity arm** (the 20M sweep repeated at 2× tokens) so the best-β claim is not an artifact of one training horizon (Schedule-Free §5 convention).
- **Measures:** U-shape of loss in β with CIs; lower-edge diagnostics (insufficient suppression: high-band residual, instability) and upper-edge diagnostics (lag proxy `1−cos(m_t, ḡ^LB_t)`, burn-in, ringing); **prospective mechanism score** = high-band residual + c·lag with c **frozen from G1 data before G6 launches** (review: no post-hoc per-setting tuning), evaluated on G6 runs only.
- **Gates:** both edges demonstrated at scale; best β interior; frozen score's argmin within one grid step of the loss argmin on ≥ half the settings (the E11 mechanism-score failure honestly re-tested with the prospective protocol).
- **Budget:** ~55 h (incl. the 2×-token horizon arm).

### 3.3 What runs where

- share (1 GPU): G1/G2/G3/G6-ResNet, 20M GPT singles — the bulk of jobs, cheap queue.
- regular-a (1 node, 8 GPU DDP): 124M runs (G4 confirmation, G5).
- Laptop/local (MPS/CPU): all toy re-runs (T1 overlays), figure rendering, paper builds. Figures never render on Wisteria; `sync_results.sh` brings `results/cache/` home.

### 3.4 Budget, storage, order

| Phase | A100-h |
|---|---|
| Phase 0 bring-up + calibration | 20 |
| G1 ResNet sweep (incl. thinned CIFAR-100) | 110 |
| G2 decomposition | 55 |
| G3 forced controls | 15 |
| G4 GPT 20M + 124M | 165 |
| G5 practical Muon | 175 |
| G6 β window (incl. horizon arm) | 55 |
| **Subtotal** | **595** |
| Contingency +30% | 180 |
| **Total** | **≈ 775 A100-h ≈ 97 Aquarius node-h** |

Storage: raw diagnostic windows (probed matrices fp16 + sketches) ~50–100 GB per windowed run, transient on `/work`; distilled per-run `arrays.npz` (band energies, spectra, eigendata, scalars) synced into `results/cache/` (target < 200 MB per experiment family). Raw windows kept on `/work` until the paper's numbers freeze, then archived or deleted.

Order + kill-switches: Phase 0 → G1 → G2 (KILL: river-valley reading fails at CIFAR → stop, rescope) → G3 → G4-diagnostic → G5-20M-tune → {G4-confirmation, G5-124M, G6} → 5-seed top-ups of headline cells. Theory workstream T1–T5 runs in parallel from day 1 (no GPU dependency); W-workstream consumes both.

---

## 4. Writing workstream (W1–W6)

- **W1 Abstract + contributions rewrite.** Delete "Experiments confirm each prediction" register (R6); scope sentence for EMA-vs-HB normalization (T5); curved valley labeled as reduction + validity margin; Muon bullets restated as buffer theorems + update theorems (T3) + necessity example; add one sentence of GPU-scale evidence once G-gates pass. The "three readings" sentence becomes complementarity (T4).
- **W2 Structural separation (R1).** §6 gets an explicit statement-class register: **exact** (straight valley: Props 1–4, Thms 1–3, Cors 1–2), **exact-coupled** (new Prop 5 b-ii), **quasi-static approximation** (5 b-iii, flagged in every downstream use), matching the theorem/approximation/heuristic separation the review demands. Table 2 (theory→experiment map) gains a "statement class" column and rows for G1–G6. Sweep-list addition (Codex): the `((1+β)/(1−β))²` river-to-hill ratio is scoped as an **open-loop, near-pure-tone limit** (review §2.1) everywhere it appears — Thm 2's statement, the contribution bullet, §5 prose. Scope paragraphs also name theorem-vs-deployed-update gaps explicitly (the Adam→Reddi cautionary lesson): every statement says which update rule it covers.
- **W3 Shorten (R6).** Targets: E1–E4 compress to an appendix "implementation checks" block (review: they are self-confirmation, useful but not evidence) with one summary paragraph in the body; E5 toy Muon merges into the G5 section as motivation; redundant remark prose trimmed; proofs of Thm 4/5/Cor 3 to appendix; net target: **body ≈ 30–33 pp, total ≤ ~45 pp** with complete proofs + per-experiment configs appendixed (Soudry-style JMLR split). JMLR's author guidance flags >35 pp (incl. appendices) for slower review and treats >50 pp as desk-risk needing cover-letter justification; the 57-pp Soudry precedent shows justified length passes — but the GPU sections must displace toy material, not extend it.
- **W4 E7 schedule claims (R4) — DONE (revision_v5, branch (a) remove).** The p. 32 quantitative claims (factors 15/118, detector steps) are removed, replaced by a qualitative sentence + future-work pointer; the material is deferred to the BETASCHEDULER paper. Recorded in `revision_v5_changelog.md`.
- **W5 Statistics + reproducibility text.** §8.1 gains the divergence-counted criterion, CI convention, cutoff-sensitivity convention, and window-length rule; a **Reproducibility** paragraph points to the public repo (§5), the runs index, and per-figure `summary.md` provenance; every new number B6-audited against `metrics.json` (Codex instructed to open the cache — standing lesson).
- **W6 Related work + citations.** Add: Lessard–Recht–Packard (IQC, for T4); noisy quadratic model (Zhang et al. 2019) as the closest theory-validated-at-scale precedent; polar perturbation sources (R.-C. Li; Higham); Muon-at-scale (Moonshot "Muon is Scalable", for the G5 protocol); optimizer-benchmark protocol reference (AlgoPerf/MLCommons) grounding the P-prac tuning rules. CRITERION.md rows for every new canonical term (`L_NS`, statement classes, `λ̂_max`, protocol names P-thy/P-prac). Add a short **reconciliation passage** tracing any disagreement with neighboring accounts (Schedule-Free averaging-equivalence, μP/alignment width-scaling, Shen's river-valley Muon) to an identified methodological cause rather than adjudicating by rhetoric (Chinchilla App-D.4 discipline).

---

## 5. Reproducibility + repo workstream (R-git)

1. **git init now** (this session): `.gitignore` = references PDFs (catalogued in `references/README.md`, per repo policy), latex intermediates (`*.log/*.blg/*.synctex.gz/*.aux/*.bbl/*.out`), `__pycache__`, `.DS_Store`. **Committed on purpose:** `latex/main.pdf` (the submission-of-record the review judged — archival), `codebases/figures/out/` (paper-truth renders that `main.tex` includes from `../codebases/figures/out/`; MPS re-renders wiggle at 1e-3, so the committed PDFs are the citable artifacts), `results/cache/` + `results/index/` (1.3 MB — the provenance that makes figures pure), `nanogpt/upstream/` (vendored, needed for a self-contained Wisteria clone).
2. Branch discipline from here on: `main` = paper-truth; feature branches per workstream (`t1-curved-valley`, `g1-resnet`, …); Wisteria pulls `main`, pushes run-record commits back via bundle/rsync if no remote is reachable from login nodes (verify; else push from laptop after `sync_results.sh`).
3. **Public artifact for resubmission:** GitHub repo (JMLR is single-blind; public is fine) = this repo minus `references/` + minus `discussions/` (internal notes), with README (exact reproduce commands per figure, environment lockfile, Wisteria-agnostic fallbacks), LICENSE, tagged release at submission. The paper's Reproducibility paragraph cites the tag.
4. `core/logging.py` git-SHA field lands with the B2/B3 batched core edit (§3.1).

---

## 6. Exemplar-paper lessons (layout / theory / validation / practice)

Raw surveys archived as `discussions/plan_v5_exemplars_venues.md` (verified award lists + 9 exemplars + JMLR expectations) and `discussions/plan_v5_exemplars_optimizers.md` (optimizer-paper deep-dive). This section keeps only the decisions.

### 6.1 Venue exemplars and JMLR's own expectations

The theory+experiment award papers closest to ours, and the single pattern each contributes:

| Exemplar | Pattern we adopt |
|---|---|
| **Chinchilla** (NeurIPS'22 Outstanding) | triangulate the central law by ≥2 independent routes; ONE pre-registered confirmatory run; reconcile with rival accounts causally (their App D.4 vs Kaplan) |
| **EDM** (NeurIPS'22 Outstanding) | master-equation table as the paper's spine (prior optimizers as parameter rows of one closed-loop template); cumulative single-change ablation for any recipe |
| **High-dim SGD limit theorems** (NeurIPS'22 Outstanding) | overlay predicted curves on multi-seed measured trajectories at ≥2 scales; explicit phase-diagram figure — the award-level version of our guide-overlay habit |
| **Scaling Data-Constrained LMs** (NeurIPS'23 runner-up) | per-claim designed slices of run-space; fit/validation run separation with counts; fitted laws → numeric decision thresholds |
| **D-Adaptation** (ICML'23 Outstanding) | exact scope statement per theorem; the deep-learning variant carries no theorem *and says so*; parity-vs-tuned-baseline as the success criterion |
| **Learning Dynamics of LLM Finetuning** (ICLR'25 Outstanding) | prediction-first experiment sections; a counterintuitive prediction as the flagship |
| **Geometry-adaptive harmonics** (ICLR'24 Outstanding) | negative controls where the theory predicts failure; analytic-optimum yardsticks on the toy class; controls substitute for scale |
| **The Road Less Scheduled** (NeurIPS'24 **Oral**, AlgoPerf'24 winner — *not* a best paper; award-status verified) | theorem assumptions monitored as runtime diagnostics (their condition (15)); external benchmark for recipe claims; per-workload hyperparameter appendix; sensitivity section |
| **Scaling Exponents** (ICML'24 poster) | instrument every assumption as a measured quantity; scale ladder with a sweep at each rung as the standard "holds at scale" evidence |
| **Soudry et al. JMLR 2018** (implicit bias, 57 pp) | the JMLR-native shape: body = theorems + rate predictions + small experiments measuring exactly the predicted quantities; complete proofs appendixed; beyond-theory sections labeled as such |

JMLR specifics (verified from jmlr.org author/reviewer pages): claims must be supported by analysis or experiments with **enough detail to replicate** (Pineau checklist recommended — adopt it as an appendix); clarity for a general ML reader; >35 pp total flags slower review, >50 pp needs cover-letter justification (our W3 targets body 30–33, total ≤ ~45); "reject and encourage resubmission" — our exact status — requires the action editor's explicit resubmission permission, so the cover letter must map changes to the review point-by-point (§7's response letter). The Soudry moral governs the whole revision: **JMLR does not demand SOTA scale; it demands exhaustive scope-labeling, complete proofs, and experiments that measure exactly the quantities the theorems predict** — the G-campaign adds the credibility ladder on top of that native core, not in place of it.

### 6.2 Optimizer-paper protocol lessons

| Exemplar | Protocol we adopt |
|---|---|
| **AdaGrad** (Duchi et al., JMLR 2011) | the JMLR precedent: ~15% experiment pages suffice when every experiment instantiates the theory's predicted regime and measures the bounds' own observables; label the practical variant's gap from the analyzed one |
| **Adam** (ICLR'15) + **Reddi et al.** (ICLR'18) | the canonical theorem-vs-shipped-algorithm cautionary tale (Adam's convergence proof used decaying α_t/β_1t never used in practice, later refuted) — our scope paragraphs name the deployed update each statement covers; copy Adam §6.4's targeted ablation-grid template aimed at the most-stressed assumption |
| **Edge of Stability** (Cohen et al., ICLR'21) | our G1 headline template: per-(η,β) dashed threshold lines overlaid on measured sharpness; Lanczos-HVP on a fixed abridged subset at declared cadence; exact-quadratic proofs in one appendix, numbered empirical caveats in another; their momentum MSS (2+2β)/η_HB is our threshold in heavy-ball form — state the conversion, cite Goh |
| **μTransfer** (NeurIPS'21) | invariance plot (optimum stable in theory-corrected coordinates, drifting in naive ones — G4's deliverable); proxy→flagship ladder with disclosed tuning fraction and baseline-breakage honesty; Table-1-style proved-for vs empirical-for asterisks; preflight coord-check analogue |
| **Chinchilla** (NeurIPS'22) | triangulation; schedule-horizon confound control (§3.0.8); the Epoch-AI replication episode: distrust tight parametric CIs, triangulation is what saves conclusions |
| **Schedule-Free** (NeurIPS'24 **Oral**; AlgoPerf'24 winner) | fairness template for tuned comparisons (baseline AND method grid-swept, per-horizon-tuned schedules, 10 seeds, losses reported with explanations); exact-convex/heuristic-DL/implementation-concerns three-block structure; cite award status precisely |
| **Muon blog + Kimi "Muon is Scalable"** | the pinned community baseline of G5 (NS5 coefficients, bf16, μ=0.95 nesterov, hidden-2D split, RMS-matched scaling, WD at scale, ≤1% overhead); the theory gap we fill is real: existing formal content is spectral-norm steepest descent + an update-RMS lemma — nothing about what the buffer feeds the polar step |
| **Wilson et al.** (NeurIPS'17) | tuning protocol verbatim (§3.0.9); one exact toy theorem paired with each empirical claim; concede near-ties in print |
| **Lessard–Recht–Packard** (SIAM'16) | quadratics-first presentation bridge; proved-vs-numerically-certified labeling; constructed counterexample/intervention (their heavy-ball limit cycle ↔ our resonance forcing G3); heavy-ball fragility outside quadratics = why our closed-loop results are quadratic-regime predictions, never NN theorems |
| **NQM** (Zhang et al., NeurIPS'19) | exact-simulator pre-registration (§3.0.10); qualitative transfer of curve shapes/orderings, never constants; the load-bearing noise assumption gets its own labeled empirical appendix (their App E.2 ↔ our G2) |

**2024–26 benchmark standards** (AlgoPerf; No-Train-No-Gain; Wen et al. "Fantastic Pretraining Optimizers"; Essential-AI Muon study): equal disclosed per-arm tuning budgets, never-shared HPs, seriously-tuned AdamW, compute matching with optimizer overhead charged, tokens-to-target with pre-registered targets, fully-decayed re-tuned schedules compared at final checkpoints only, ≥3 scales where feasible, divergences as first-class results, released configs. All folded into §3.0. The **Kimi ~2× vs Wen ~1.1–1.4×** Muon-speedup dispute is acknowledged in W6 and left unadjudicated — G5 is mechanism work.

### 6.3 Adopted conventions (checklist, each with its implementing item)

1. Three-tier claim taxonomy (Theorem / Approximation / Empirical) with a claim→tier→statement→figure table → **W2** (Table 2 gains the tier column; §1 gets the claim table).
2. Scope paragraph after every theorem, D-Adaptation-style → **W2**.
3. Assumptions instrumented as runtime diagnostics in every run → **§3.0 invariant 6**.
4. Master-equation table (SGD / heavy-ball / EMA / Muon as parameter rows of Eq. 7 + Eq. 8) → **W2**, one table, cheap.
5. Prediction-first experiment sections ("Proposition N predicts X; we test it by…") → **W5**, applied to every G-section.
6. Counterintuitive prediction + negative control as headliners: resonant harm (momentum *hurts* at θ_β — G3) and the low-HFER null (filter-first advantage *vanishes* — G5) are exactly these; frame them so → **W1/W5**.
7. Analytic-optimum yardstick on the toy class (measured best β vs closed-form optimum, % gap) → **G1/G6 analysis**.
8. Overlay plots + phase diagram at scale (benefit vs `ηλ̂_max` is the phase axis) → **G1/G4**.
9. Scale ladder, sweep per rung → **G4** (20M → 60M optional → 124M).
10. Pre-registered confirmatory run → **G4 confirmation**.
11. Triangulate the central law (transfer-function fit / AR(2) closed form / direct grid) → **G1/G6**.
12. Cumulative single-change ablation if a recipe is recommended → **G5**, config rows.
13. Parity-vs-tuned-baseline + published grids + one externally-tuned comparison → **G5** (modded-nanogpt speedrun baseline; appendix hyperparameter tables).
14. Sensitivity section (horizon, LR range) → **G6** horizon arm + **W5**.
15. Body/appendix split, total ≤ ~45 pp → **W3**.
16. Causal reconciliation with neighboring accounts → **W6**.
17. Fit hygiene (counts, hold-out error) → **§3.0 invariant 7**.
18. Worst-case-envelope framing for perturbation bounds; never imply certification beyond measured regimes (the ICML'22 "Privacy for Free" refutation is the cautionary tale) → **T3/W1**.
19. EoS-style sharpness overlay with β-shifted threshold lines as the headline G1 figure; Cohen measurement protocol verbatim → **G1**.
20. Exact-simulator pre-registration of curve shapes/orderings before every GPU launch → **§3.0.10 / B5**.
21. Schedule-horizon matching; final-checkpoint-only comparisons → **§3.0.8**.
22. Per-arm tuning never shared across optimizers; interior-optimum grids; defaults as labeled arms → **§3.0.9**.
23. Muon baseline pinned to the community protocol; polar-only = instantaneous-Shampoo control; Kimi-vs-Wen dispute cited, not adjudicated → **G5**.
24. μTransfer-style invariance plot in filter-corrected coordinates across the ladder → **G4**.
25. The load-bearing noise assumption gets its own labeled empirical appendix (NQM C=H discipline) → **G2 → W5**.
26. Theorem-vs-deployed-update gaps named per claim (Adam→Reddi lesson) → **W2**.
27. Preflight gradient-stream diagnostic shipped as a released artifact → **B5 / §5**.

---

## 7. Execution order, gates, and review protocol

```
Week 1–2   T1, T3 (blocking theory) + T4/T5; Phase 0 bring-up on Wisteria; R-git done day 1.
Week 2–4   G1 → G2 (KILL-SWITCH) → G3;  T2 attempt in parallel.
Week 4–6   G4 diagnostic → G5 tunes → G4/G5 at 124M.
Week 6–7   G6 + 5-seed headline top-ups;  W2/W3 restructure begins on frozen theory.
Week 8–9   Full W-pass, figures from cache, CRITERION sync.
Week 10    Codex full-paper review (with metrics.json audit), response letter, resubmission tag.
```

- Every theory item: check-script gate → Codex review → only then into `main.tex`.
- Every experiment: predeclared gates in the run config; failures reported, never silently dropped; paper text written only from cached records.
- Final package: revised `main.pdf` + public-repo tag + **response letter** mapping each review point (R1–R6, package 1–7, plus the itemized §2.2/§2.3 math points) to the change that answers it, with the honest residue (T2 fallback, Adam scope, anything failed) stated plainly.

**Plan-review log (Codex, 2026-07-19).** This plan was itself Codex-reviewed pre-handoff (read-only workspace, judged against `review_v5.md` + `main.tex` + the T1 pre-check). Round 1: FAIL — 3 blocking findings (T3b's monotone-class overclaim → rescoped to the bounded-annulus Lipschitz theorem; G2's β=0-only decomposition → paired actual-β trajectories; G4's gate stronger than the review's own hypothesis → regime-level with low-HFER null control) + 5 suggestions, all applied. Round 2: 2 blocking (fairness invariants 8–9 over-globalized → scoped to P-prac with P-thy/G2/G3 exemptions; G5 momentum-convention conflation → mechanism arms split from as-shipped external baselines) + 2 scope caveats, applied. **Final verdict: PASS.**

## 8. Decisions for Leon (resolved 2026-07-19: items 2–5 at defaults; item 1 deferred)

1. **E7 schedule claims — RESOLVED and EXECUTED (Leon, 2026-07-19, revision_v5 session): branch (a) remove.** The p. 32 quantitative claims are out of the Discussion (a qualitative confinement-floor design constraint + future-work pointer remain); the material stays with the BETASCHEDULER paper. See `revision_v5_changelog.md`.
2. **Adam scope — CONFIRMED default:** Adam-preconditioned streams stay out of headline claims; at most one exploratory subsection. No full Adam arm in G4.
3. **CIFAR-100 + BN-free control — CONFIRMED default:** CIFAR-100 as G1's thinned complete sweep (the Codex-upgraded default; 3 LR × 5 β × 3 seeds), BN-free GroupNorm control at headline cells only; full grids declined.
4. **Public repo timing — CONFIRMED default:** public at resubmission, tagged release.
5. **124M confirmation token budget — CONFIRMED default:** 2.5B tokens (~Chinchilla-20×).
