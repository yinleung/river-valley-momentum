# plan_v5 companion — optimizer-paper deep-dive (raw agent report, 2026-07-19)

Compiled by a literature agent for `plan_v5.md` §6.2. Kept near-verbatim as reference; distilled decisions live in `plan_v5.md`.

---

### A: AdaGrad (Duchi, Hazan, Singer — JMLR 12:2121–2159, 2011)
- skeleton: §1 intro (algorithm, results, related); §2 adaptive proximal functions; §3 diagonal (Thm 5); §4 full-matrix (Thm 7); §5 derived algorithms; §6 experiments (4 convex tasks + sparsity tradeoffs); §7 conclusions; App A–G proofs/derivations.
- scoping: online convex composite; data-dependent regret bounds vs best fixed diagonal preconditioner; full-matrix flagged impractical → diagonal justified by cost + constructed sparse-adversary example; Hessian intuition explicitly "informal".
- experiments: ~6 of 39 pages (~15%; theory:experiment ≈ 4.7:1), all convex/linear, each instantiating the theory's predicted sparse-feature regime; baselines cross-validated per arm; tables + variances.
- lessons: JMLR precedent — modest experiments pass when each measures the bound's own observables; label the practical variant's gap from the analyzed one; per-arm cross-validation statements.
- urls: jmlr.org/papers/v12/duchi11a.html

### B: Adam (Kingma & Ba, ICLR 2015)
- skeleton: §2 algorithm; §3 bias correction; §4 convergence (Thm 4.1 — assumptions diverge from shipped algorithm; proof later shown flawed by Reddi et al. ICLR 2018, the canonical cautionary tale); §6 experiments incl. §6.4 the targeted bias-correction ablation grid (β1×β2×α, with/without correction, stressing exactly the assumption's worst regime).
- lessons: state per-claim which (β, η) regime is exact — reviewers post-Reddi check theorem-vs-shipped gaps; copy the §6.4 template (one ablation grid aimed at the most-stressed assumption — for us finite-window/stationarity); defaults presented as scoped empirical recommendations, never consequences of theory.
- urls: arxiv.org/abs/1412.6980; arxiv.org/abs/1904.09237

### C: Edge of Stability (Cohen et al., ICLR 2021)
- skeleton: §2 quadratic stability background; §3 progressive sharpening + EoS on NNs; App A numbered caveats; App B exact quadratic proofs incl. momentum MSS (Polyak (2+2β)/η, credits Goh 2017); App G–H SGD; App I details (Lanczos-HVP, fixed abridged ~5k-example measurement subset, declared cadence; between-iterate sharpness for BN nets).
- validation: per-η dashed threshold lines overlaid on sweeps; LR-drop intervention (sharpening resumes); momentum runs plateau at the momentum MSS across 4 architectures.
- lessons: OUR HEADLINE TEMPLATE — overlay per-(η,β) dashed lines at our threshold 2(1+β)/((1−β)η), state the conversion to Goh/Cohen's heavy-ball MSS; adopt their measurement protocol verbatim; exact-quadratic theorems in one appendix, numbered empirical caveats in another.
- urls: arxiv.org/abs/2103.00065; github.com/locuslab/edge-of-stability

### D: μTransfer / Tensor Programs V (Yang, Hu et al., NeurIPS 2021)
- skeleton: parametrization primer → HPs don't transfer conventionally → μP unlocks transfer → which HPs transfer (Table with proved-for vs empirically-validated-only asterisks) → efficiency ladder (IWSLT 4M→40M; BERT 13M→350M; GPT-3 40M proxy→6.7B, 467 proxy runs, tuning ≈7% of pretraining, non-diverged counts and baseline breakages disclosed).
- lessons: Fig-1-style invariance plot (optimum stable in corrected coordinates, drifting in naive ones) is the single most legible "theory predicted X" artifact — recast our LR×β optimum in filter-corrected coordinates across scale; proxy→flagship with one compute-audited confirmatory run; Table-1 asterisk discipline; ship a cheap preflight "coord-check analogue".
- urls: arxiv.org/abs/2203.03466; github.com/microsoft/mup

### E: Chinchilla (Hoffmann et al., NeurIPS 2022)
- three independent estimation routes converge (a≈b≈0.5); >400 runs; bootstrap percentile intervals; App B: cosine cycle must match token budget (the confound that inflated Kaplan's exponents, App D.4); confirmatory 70B compute-matched vs Gopher with itemized delta list. Epoch AI replication (arXiv:2404.10102): the printed Approach-3 fit was inconsistent with the paper's own data — triangulation saved the conclusion; distrust tight parametric CIs.
- lessons: ≥2–3 independent routes for any fitted quantity; schedule-horizon matching in every sweep; itemized deltas for the flagship comparison.
- urls: arxiv.org/abs/2203.15556; arxiv.org/abs/2404.10102

### F: Schedule-Free (Defazio et al., NeurIPS 2024 Oral — NOT best paper; won MLCommons AlgoPerf 2024 self-tuning track)
- exact convex theorems (any-β worst-case optimal; online-to-batch unification recovering linear decay); §2.2 large-LR + all DL results labeled heuristic; dedicated implementation-concerns section (BatchNorm); 28 problems; both baseline AND method grid-swept over LR×WD, cosine tuned per horizon, 10 seeds on AlgoPerf, losses reported and explained (DeepSpeech).
- lessons: cite precisely ("NeurIPS 2024 Oral" + "AlgoPerf winner"); their fairness template for our tuned comparisons; exact/heuristic/implementation-concerns structure is JMLR-safe; citable as the iterate-averaging sibling of our gradient-EMA filter.
- urls: arxiv.org/abs/2405.15682; mlcommons.org/2024/08/mlc-algoperf-benchmark-competition/

### G: Muon (Jordan et al. blog 2024) + "Muon is Scalable" (Moonshot/Kimi, arXiv:2502.16982)
- THE BASELINE PROTOCOL OUR G5 MUST PIN: orthogonalize μM_t+G_t with μ=0.95, nesterov=True; Newton–Schulz 5 iterations, quintic (a,b,c)=(3.4445, −4.7750, 2.0315), bf16, input normalized by ‖X‖_F+1e-7, transpose if rows>cols; Muon ONLY on hidden ≥2D matrices (embeddings/lm_head/biases/LN → AdamW, e.g. lr 3e-4, betas (0.9,0.95)); shape scaling: repo `max(1, rows/cols)^0.5` (speedrun standard) vs Kimi RMS-matching `0.2·√max(A,B)` + weight decay (AdamW-tuned LR/WD transfer directly — cleanest fairness story); NS FLOP overhead ≤1%; weight decay necessary at scale (vanilla Muon fastest early, loses in over-train regime).
- evidence style: community-tuned speedrun record as the falsifiable baseline (persisted through 12 subsequent records); Kimi: grid-searched compute-optimal AdamW scaling baseline (399M–1.5B, Chinchilla-optimal budgets) → "~52% of AdamW FLOPs"; Moonlight 16B-A3B/5.7T with an AdamW-twin control.
- theory gap our paper fills: existing formal content is only spectral-norm steepest descent + an update-RMS lemma — nothing about what the momentum buffer feeds the polar step; momentum-off Muon = "instantaneous Shampoo" is the natural no-filter control arm.
- urls: kellerjordan.github.io/posts/muon/; github.com/KellerJordan/Muon; arxiv.org/abs/2502.16982

### H: Marginal Value of Adaptive Methods (Wilson et al., NeurIPS 2017)
- one exact toy theorem (adaptive methods provably mislabel; min-norm exact) paired with tuned deep comparisons. THE TUNING PROTOCOL: per-optimizer log-spaced 5-point LR grid extended until the optimum is interior (all grids in appendix); per-arm decay scheme; fixed budgets; 5 seeds ±1 std; dev-set selection; defaults as a separately-labeled arm; near-ties conceded in print.
- urls: arxiv.org/abs/1705.08292

### I: Lessard–Recht–Packard IQC (SIAM J. Optim. 2016)
- optimizers as LTI feedback + gradient nonlinearity; dimension-free LMI certificates; scrupulous proved-vs-numerically-certified labeling; the heavy-ball counterexample (κ=25 piecewise-quadratic with attractive period-3 limit cycle) shows quadratic-optimal tunings can provably fail one function class out; noise-robustness sector analysis (fast-but-fragile α=2/(L+m) vs robust α=1/L).
- lessons: quadratics-first presentation bridge; pair the stability boundary with a constructed intervention, not just sweeps; frame closed-loop results as quadratic-regime predictions to be tested at NNs, never NN theorems.
- urls: arxiv.org/abs/1408.3595

### J: Noisy Quadratic Model (Zhang et al., NeurIPS 2019)
- exact per-dimension risk recursions (seconds to simulate) predict batch-size-dependent effects (momentum extends perfect scaling; EMA helps small-B; critical batch size optimizer-dependent), then confirmed on real nets with per-(workload, optimizer, B) independent quasirandom searches, 100–200 non-divergent-trial budgets, interior-optimum checks, steps-to-target. The C=H codiagonalizability assumption gets its own empirical appendix (E.2), labeled "nontrivial".
- lessons: package our AR(2)/forced-gain theory as an exact simulator generating pre-registered curve-shape/ordering predictions; claim qualitative transfer, never constants; give the noise-spectrum assumption its own empirical appendix (our G2 is exactly that).
- urls: arxiv.org/abs/1907.04164

## Benchmark-protocol standards (2024–2026)

- **AlgoPerf** (Dahl et al. arXiv:2306.07179; v0.6 rules 2025; no JMLR version as of 2026-07): time-to-result on fixed hardware; validation targets from ~200-trial tuning of four standard algorithms; external-tuning = workload-agnostic space, 5 studies × 5 quasirandom trials, median; self-tuning = everything on the clock at 3× budget; held-out workload variants; performance-profile scoring. 2024 competition (Kasimbeg et al. ICLR 2025): Distributed Shampoo won external (~28% over tuned NAdamW); Schedule-Free won self-tuning; divergences scored as first-class outcomes.
- **No Train No Gain** (Kaddour et al. NeurIPS 2023): reference-system-time budgets; LR schedule fully decayed and re-tuned per budget; at 24h no efficiency method beat the budget-tuned baseline; "speedup claims without an explicit budget are misleading."
- **Fantastic Pretraining Optimizers** (Wen et al. arXiv:2509.02046): 11 optimizers × 130M–1.2B × 1–8× Chinchilla; per-optimizer-per-scale coordinate-descent over ALL HPs; matrix methods ≤1.4× at small scale shrinking to ~1.1× at 1.2B; Muon best at 1–4× Chinchilla, overtaken by Soap/Kron at ≥8×. Two pitfalls: sharing HPs across arms; comparing mid-schedule (curves cross — final checkpoints only).
- **Practical Efficiency of Muon** (Essential AI arXiv:2505.02222): Muon expands the compute-time Pareto frontier; retains data efficiency past AdamW's critical batch size; muP + telescoping sweeps to 3.7B/160B.
- Notables: Choi et al. 2019 (rankings are properties of the tuning protocol; tune ε); Schmidt et al. Crowded Valley (no dominant optimizer); Semenov et al. 2025 (regime-dependent rankings; AdEMAMix/MARS can beat Muon in some batch regimes); Kimi K2 (MuonClip, 1T-param/15.5T-token zero-spike existence proof); Volkova et al. 2026 (per-optimizer independent Chinchilla fits ill-conditioned; use shared exponents). **Headline dispute to acknowledge: Kimi ~2× vs Wen ~1.1–1.4× under fully-tuned AdamW.**
- Reviewer-expected checklist: (1) equal disclosed tuning budgets + search spaces per arm; (2) never share HPs across optimizers; (3) seriously-tuned AdamW baseline; (4) compute/token matching with optimizer overhead charged; (5) tokens-to-target with pre-registered targets; (6) compare only at end of fully-decayed schedules re-tuned per budget; (7) ≥3 scales, ≥2 data multiples, trend stated; (8) seeds with spread; (9) divergences reported as results; (10) batch-size regime disclosed; (11) explicit HP-transfer story; (12) held-out task + released code/configs.

## Synthesis: 15 campaign lessons (distilled/adopted versions in plan_v5.md §6.3)

1. AR(2) threshold as the headline falsifiable prediction, EoS-overlay-style (per-(η,β) dashed sharpness lines).
2. Per-claim exact/approx/empirical labeling as a structural device (μTransfer asterisks; Cohen's two appendices; the Adam→Reddi episode citable).
3. Never claim the NN theorem (Lessard's counterexample: quadratic-optimal tunings provably fail one function class out).
4. Run the exact simulator before every GPU run; pre-register curve shapes/orderings; qualitative transfer, never constants (NQM).
5. Every load-bearing assumption gets its own empirical appendix (NQM's C=H ↔ our G2 decomposition).
6. Wilson/NQM tuning protocol verbatim (interior-optimum grids, per-arm decay, defaults as labeled arms, divergence-counted budgets).
7. Schedule-horizon matching or the sweep is corrupted (Chinchilla App B; NTNG; Wen's curve-crossing).
8. Triangulate fitted quantities by ≥2 routes with bootstrap intervals (Chinchilla + Epoch replication).
9. Proxy→flagship with one pre-registered confirmatory run + itemized deltas + disclosed tuning fraction (μTransfer).
10. μTransfer-style invariance plot in filter-corrected coordinates across scale.
11. Pin the Muon baseline to the community protocol exactly (NS5 coefficients, bf16, μ=0.95 nesterov, hidden-2D split, stated shape-scaling, WD on long runs, overhead reported); momentum-off = instantaneous-Shampoo control.
12. Position inside the Kimi-vs-Wen speedup dispute as mechanism work, citing both.
13. One constructed intervention beats ten sweeps (Lessard's limit cycle; Cohen's LR-drop; our resonance forcing).
14. Report adverse results and near-ties in the main text.
15. Ship the preflight gradient-stream diagnostic as a released artifact; AdaGrad precedent says mechanism-instantiating experiments at ~15% of pages suffice for JMLR.
