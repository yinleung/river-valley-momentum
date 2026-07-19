# plan_v5 companion — venue award-paper survey (raw agent report, 2026-07-19)

Compiled by a literature agent for `plan_v5.md` §6; award lists verified against official venue blogs. Kept verbatim as reference material; the distilled decisions live in `plan_v5.md` §6.

---

## Award lists (verified)

Legend: **[T+E]** = substantive theory/analysis AND experiments; [T] = essentially pure theory; [E] = essentially empirical/systems; [D/P] = dataset/benchmark/position.

**ICLR 2024** (https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/) — Outstanding (5): Generalization in diffusion models arises from geometry-adaptive harmonic representations **[T+E]**; Learning Interactive Real-World Simulators [E]; Never Train from Scratch [E, methodology]; Protein Discovery with Discrete Walk-Jump Sampling **[T+E]**; Vision Transformers Need Registers [E]. Honorable mentions (11) incl.: Towards a statistical theory of data selection under weak supervision [T+E]; The mechanistic basis of data dependence and abrupt learning in an in-context classification task [T+E]; Beyond Weisfeiler-Lehman [T]; Robust agents learn causal world models [T]; Proving Test Set Contamination [T+E]; Approximating Nash Equilibria via Stochastic Optimization [T+E]; Flow Matching on General Geometries [T+E]; Amortizing intractable inference in LLMs; Is ImageNet worth 1 video?; Meta Continual Learning Revisited; Model Tells You What to Discard [E].

**ICLR 2025** (https://blog.iclr.cc/2025/04/22/announcing-the-outstanding-paper-awards-at-iclr-2025/) — Outstanding (3): Safety Alignment Should be Made More Than Just a Few Tokens Deep [E+analysis]; **Learning Dynamics of LLM Finetuning** (Ren & Sutherland) **[T+E]**; AlphaEdit [T+E]. HMs (3): Data Shapley in One Training Run [T+E]; SAM 2 [E]; Faster Cascades via Speculative Decoding [T+E].

**NeurIPS 2022** (https://blog.neurips.cc/2022/11/21/announcing-the-neurips-2022-awards/) — Outstanding (13) incl.: **Chinchilla** (compute-optimal LLM training) **[T+E]**; **EDM** (Elucidating the Design Space of Diffusion Models) **[T+E]**; High-dimensional limit theorems for SGD **[T+E]**; Beyond neural scaling laws (data pruning) **[T+E]**; Is OOD Detection Learnable? [T]; On-Demand Sampling [T]; Gradient Descent: The Ultimate Optimizer [T+E]; Gradient Estimation with Discrete Stein Operators [T+E]; Riemannian Score-Based Generative Modelling [T+E]; Imagen, ProcTHOR, Neural Corpus Indexer, NL/program abstractions [E]. D&B: LAION-5B, MineDojo. ToT: AlexNet.

**NeurIPS 2023** (https://blog.neurips.cc/2023/12/11/announcing-the-neurips-2023-paper-awards/) — Outstanding (2): Privacy Auditing with One (1) Training Run **[T+E]**; Are Emergent Abilities of LLMs a Mirage? **[T+E]**. Runner-ups (2): **Scaling Data-Constrained Language Models** **[T+E]**; Direct Preference Optimization **[T+E]**. D&B: ClimSim; DecodingTrust.

**NeurIPS 2024** (https://blog.neurips.cc/2024/12/10/announcing-the-neurips-2024-best-paper-awards/) — Best (main): Visual Autoregressive Modeling (VAR) [E]; Stochastic Taylor Derivative Estimator [T+E]. Runner-ups: Not All Tokens Are What You Need [E]; Guiding a Diffusion Model with a Bad Version of Itself **[T+E]**. D&B: PRISM. **"The Road Less Scheduled": NOT an award winner — NeurIPS 2024 Oral** (https://neurips.cc/virtual/2024/oral/98003); its recipe won the independent **MLCommons AlgoPerf 2024 self-tuning track**.

**ICML 2022** (https://aihub.org/2022/07/21/congratulations-to-the-icml2022-outstanding-paper-award-winners/) — Outstanding (10) incl.: Do differentiable simulators give better policy gradients? **[T+E]**; Learning Mixtures of Linear Dynamical Systems [T]; Privacy for Free [T+E — **caution: privacy claim later publicly refuted** (Carlini et al.)]; Understanding Dataset Difficulty with V-Usable Information [T+E]; Bayesian Model Selection, the Marginal Likelihood, and Generalization **[T+E]**; others [T].

**ICML 2023** (https://icml.cc/Conferences/2023/Awards) — Outstanding (6): **Learning-Rate-Free Learning by D-Adaptation** **[T+E, optimization]**; A Watermark for LLMs **[T+E]**; Generalization on the Unseen [T+E]; Adapting to game trees [T]; Self-Repellent Random Walks [T]; Bayesian Design Principles for Frequentist Sequential Learning [T].

**ICML 2024** (https://aihub.org/2024/07/25/congratulations-to-the-icml2024-award-winners/) — Best (10) incl.: SEDD discrete diffusion **[T+E]**; Scaling Rectified Flow Transformers [E]; Twisted SMC **[T+E]**; Stealing part of a production LM **[T+E]**; Information Complexity of Stochastic Convex Optimization [T]; Genie, VideoPoet, Debating with More Persuasive LLMs [E]; Position ×2. ToT: DeCAF. **"Scaling Exponents Across Parameterizations and Optimizers": NOT an award winner — ICML 2024 poster** (icml.cc/virtual/2024/poster/35186); included for direct relevance.

**ICML 2025** (https://joltml.com/icml-2025/awards/; corroborated ifml.institute, kempnerinstitute.harvard.edu, mcml.ai) — Outstanding (6): Train for the Worst, Plan for the Best **[T+E]**; Roll the dice & look before you leap **[T+E]**; CollabLLM [E]; Conformal Prediction as Bayesian Quadrature [T]; Score Matching with Missing Data [T+E]; The Value of Prediction in Identifying the Worst-Off [T+E]. ToT: BatchNorm.

## Selected exemplars

### 1. Chinchilla (NeurIPS 2022, Outstanding) — arxiv.org/abs/2203.15556
- layout: main ~21 pp: §3 three estimation approaches (fixed-N sweeps / IsoFLOP / parametric fit L(N,D)=E+A/N^α+B/D^β); §4 the confirmatory 70B run + evals; fitting minutiae in App A–H incl. D.4 reconciliation with Kaplan (cause: LR-schedule mismatch).
- pattern: no theorems — a parametric law with theorem-grade bookkeeping; **three independent estimation routes triangulate the same exponents**; one falsifiable prediction (70B/1.4T optimal at Gopher's budget); validation = single compute-matched head-to-head.
- lessons: triangulate the central fitted law by ≥2 routes; concentrate GPU budget in one pre-registered confirmatory run; reconcile with competing prior laws causally.

### 2. EDM (NeurIPS 2022, Outstanding) — arxiv.org/abs/2206.00364
- layout: §2 common framework (a table mapping all prior diffusion variants to choices of σ(t), s(t), preconditioning in one ODE); §3–5 per-choice analyses each immediately tested; cumulative single-change ablation ladder (config A→F, FID after each isolated change); modularity proven on checkpoints the authors did not train (FID 2.07→1.55).
- lessons: master-equation table as the paper's spine with prior methods as rows; isolated-change cumulative ablation; test the recommendation on a setup you didn't tune.

### 3. High-dimensional limit theorems for SGD (NeurIPS 2022, Outstanding) — arxiv.org/abs/2206.04030
- 43 pp; general limit theorems (ballistic ODE vs diffusive SDE, critical step-size boundary) with numbered assumptions up front; per-application corollaries; every figure overlays simulated trajectories on predicted curves across dimensions; proofs in back half.
- lessons: general theorem once, per-setting corollaries; overlay plots at ≥2 scales; explicit phase-diagram figure for step-size/momentum regimes.

### 4. Scaling Data-Constrained LMs (NeurIPS 2023, runner-up) — arxiv.org/abs/2305.16264
- 400+ runs, 182 used for fits (fit/validation separation); two orthogonal protocol slices; 5 seeds for evals; claims as numeric thresholds (repeat ≤4 epochs fine; >~16 no return); App A full derivation + LBFGS fitting.
- lessons: each claim gets its own designed slice of run-space; separate fit from validation runs and say the counts; convert fitted laws into decision thresholds.

### 5. D-Adaptation (ICML 2023, Outstanding) — arxiv.org/abs/2301.07733
- theory first (convex Lipschitz, asymptotically optimal, no extra log factors); **the Adam variant carries no theorem and says so**; experiments: ~12 convex problems with full LR-sweep baselines + ~10 deep workloads; the theory's internal quantity (D estimate) plotted during real training.
- lessons: exact scope statement per guarantee; label DL variants as principled heuristics in so many words; parity-with-tuned-baseline as the success criterion; plot the theory's internal quantities in real runs.

### 6. Learning Dynamics of LLM Finetuning (ICLR 2025, Outstanding) — arxiv.org/abs/2407.10490
- per-step decomposition as a lemma under named approximations; every experiment subsection opens "the framework predicts X"; flagship = counterintuitive prediction (preferred-response likelihood falls under prolonged DPO) confirmed; theory-derived mitigation validated.
- lessons: prediction-first experiment sections; at least one counterintuitive prediction; approximation assumptions in one labeled block.

### 7. Geometry-adaptive harmonic representations (ICLR 2024, Outstanding) — arxiv.org/abs/2310.02557
- three hypotheses up front; theory supplies analytically optimal reference solutions as quantitative yardsticks; disjoint-data intervention control; **negative controls** (shuffled pixels, off-manifold) where the theory predicts failure — and gets it.
- lessons: negative-control experiment; analytic-optimum yardsticks on restricted classes; intervention controls to kill confounds. Rigor of controls substitutes for scale.

### 8. The Road Less Scheduled (NeurIPS 2024 Oral; AlgoPerf 2024 winner) — arxiv.org/abs/2405.15682
- verified skeleton: §2 method + Thm 1–3 (convex only; Thm 3's assumption becomes **condition (15), monitored during every real run**); §4 experiments: nine DL workloads incl. nanoGPT-124M/OpenWebText, AlgoPerf (7 workloads, 10 seeds, one fixed hyperparameter set), twelve convex problems with full LR sweeps; §5 sensitivity (β across horizons; LR robustness); App G per-problem hyperparameter tables.
- lessons: monitor theorem assumptions as runtime diagnostics; external standardized benchmark for recipe claims; appendix hyperparameter tables; say plainly which experiments the theorems do not cover.

### 9. Scaling Exponents Across Parameterizations and Optimizers (ICML 2024 poster) — arxiv.org/abs/2407.05872
- exposes the implicit "full alignment" assumption in muP-style derivations, promotes it to a measured variable; 3 optimizers × 4 parameterizations × >12 LRs × 14 widths (to 26.8B); theory-diagnosed pathology → Adam-atan2 fix.
- lessons: instrument every assumption as a measured quantity with its own figure; a scale ladder with a sweep at each rung is the standard evidence a predicted trend "holds at scale"; theory-predicted numerical pathologies yield the most convincing practical payoffs.

## JMLR structural expectations
(verified: jmlr.org/author-info.html, jmlr.org/reviewer-guide.html)

- Scope: "new principled algorithms with sound empirical validation"; report insights gained, not what was done.
- Review criteria: clear goals; claims supported by experiments or analysis; significant correct contribution; related-work coverage incl. strengths/limitations/generality; clarity for a general ML reader; **enough detail to replicate**.
- Length: >35 pp (incl. appendices) ⇒ slower review, elevated rejection risk; >50 pp ⇒ justify in cover letter, desk-rejection possible; in practice 40–60 pp theory+experiment papers appear (Soudry et al. 57 pp) but every page must earn its place.
- Reproducibility: Pineau checklist recommended; online code/data appendices strongly encouraged; "all experiments must be reproducible".
- Mechanics: JMLR style file; 200-word abstract; 5 keywords; ≤50-char running title.
- Structural exemplar: **Soudry et al., "The Implicit Bias of Gradient Descent on Separable Data", JMLR v19(70):1–57, 2018** — body: setup, main theorems with **rate predictions**, proof sketches, small rate-targeted experiments measuring exactly the predicted quantities (+ a beyond-theory CIFAR section explicitly framed as such); complete proofs and extensions appendixed. Moral: JMLR does not demand SOTA scale; it demands exhaustive scope-labeling, complete proofs, and experiments that measure the exact quantities the theorems predict.

## Synthesis (18 lessons — the distilled/adopted versions live in plan_v5.md §6)

1. Three-tier claim taxonomy (Theorem / Approximation / Empirical), enforced typographically, with a claim-tier-statement-figure table.
2. Scope paragraph after every theorem ("what this does and does not cover").
3. Instrument every assumption as a measured runtime diagnostic; assumption-violation curves are results.
4. Master-equation table as the paper's spine (prior optimizers as parameter rows).
5. Prediction-first experiment sections (no experiment without a stated antecedent claim).
6. ≥1 counterintuitive prediction and ≥1 negative control.
7. Theory as analytic yardstick on the toy class (report % of analytic optimum).
8. Overlay plots for every dynamics theorem; explicit phase-diagram figure.
9. Meet scale demands with a ladder (10M→124M+), sweep at each rung; 0.4M char-GPT is far below the field's credibility floor (Schedule-Free's "small" benchmark is nanoGPT-124M).
10. One pre-registered confirmatory run (prediction stated as prediction before launch).
11. Triangulate the central quantitative law by ≥2 independent routes.
12. Cumulative single-change ablation table for any recommended recipe.
13. Parity-vs-tuned-baseline with published tuning grids; at least one comparison independent of the authors' tuning.
14. Dedicated sensitivity section (horizons, LR ranges).
15. JMLR-native body/appendix split; keep total ≤ ~40–45 pp; cut weaker material to fund the ladder.
16. Reconcile with neighboring accounts causally (Chinchilla-vs-Kaplan discipline).
17. Fit hygiene: state run counts feeding each fit; hold-out error.
18. Cautionary: overclaimed theory-practice links get publicly reversed (ICML 2022 "Privacy for Free"); tier labels + worst-case-envelope framing are the systematic defense.
