# References

The PDFs in this folder are git-ignored (copyrighted; keeps the repo lean). This manifest
catalogues them so collaborators can fetch their own copies. Search links are exact; arXiv IDs
marked *likely* are unverified — confirm before citing.

## momentum-filtering/ — the two bridged papers (Li et al.)
- **On the Performance Analysis of Momentum Method: A Frequency Domain Perspective** —
  Li, Luo, Zheng, Wang, Luo, Wen, Wu, Xu. ICLR 2025. arXiv:2411.19671.
  Momentum as a time-variant filter (Z-transform, quasi-stationary stages); orthodox vs
  unorthodox systems; empirical finding that high-frequency gradient components help early and
  hurt late; FSGDM optimizer. Empirical/heuristic framework — no landscape theory.
- **Denoise First, Orthogonalize Later: Understanding Momentum in Muon via Spectral Filtering** —
  Li, Zhang, Liu, Bao. arXiv:2606.03899. The target paper this project loosens: time-invariant
  coherent signal + BVMZOS perturbation (bounded-variance, mean-zero, temporally orthogonal);
  spectral-gap Theorem 1 (tail `O((2T-1)^{-1/4})`), Wedin Corollary 1, Pre-polar dominance
  Theorems 2–3; stationary/trajectory probes, NanoGPT + LLaMA 350M.

- **Signal Processing Meets SGD: From Momentum to Filter** — Yao, Yu, Chang, Li, Zhang, Li.
  arXiv:2311.02818 (verified from the arXiv abstract page 2026-07-04; PDF not stored here).
  Earlier signal-processing lineage: Wiener-filter first-moment estimation (SGDF optimizer);
  cited as prior filter-view work, no landscape or closed-loop theory.

## river-valley/ — landscape model
- **Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape
  Perspective** — Wen, Li, Wang, Hall, Liang, Ma. arXiv:2410.05192. Canonical river-valley
  formalization (river = flattest-Hessian direction); GD tracks the river; SGD bounces on hills
  with loss gap `(d-1)ησ²/2` (noise-driven, small-LR regime); decay descends the wall; bigram
  mechanism. No momentum analysis.
- **Towards Understanding the Power and Limits of the Muon Optimizer: A River-Valley
  Perspective** — Shen, Yang, Shi, Ma, Ma, Teng (CityUHK). arXiv:2606.21514, under review.
  Mixed-spiked matrix sensing; momentum-free Muon vs GD on the spectral river (Muon linear
  early progress, overshoots near the bottom; "Muon → GD/AdamW" two-stage advice). Momentum
  enters only as an *assumption* ("stored momentum points mostly in the river direction",
  alignment ρ in Thm 4.1) — the mechanism this project proves.

## schedule-free/ — river-adjacent optimizer line
- **The Road Less Scheduled** — Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky.
  NeurIPS 2024. arXiv:2405.15682. Schedule-Free method (interpolates Polyak–Ruppert and primal
  averaging); worst-case optimal for any momentum in convex Lipschitz problems.
- **Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model
  Training** — Song, Baek, Ahn, Yun. NeurIPS 2025. arXiv:2507.09846. SF-AdamW tracks the river
  without decay/averaging; y-iterates at edge of stability (threshold `2/((1-β₁)γ)`); central-flow
  σ² shrinks as β₁→1 — the closest existing "momentum suppresses hill oscillation" claim, via
  effective-step shrinkage in SF's y-update rather than spectral filtering; refined SF decouples
  momentum from averaging. Must-cite neighbor.
- **ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to LLMs** — Defazio.
  arXiv:2605.19095. Tech report scaling SF to LLMs; inner momentum fixes large-batch divergence.

## edge-of-stability/ — oscillation mechanism (added 2026-07-03)
- **Understanding Optimization in Deep Learning with Central Flows** — Cohen, Damian, Talwalkar,
  Kolter, Lee. ICLR 2025. arXiv:2410.24206. Central flow = time-averaged EoS trajectory; the
  formal tool behind Song et al.'s oscillation analysis; relevant to any closed-loop claim about
  the driven hill mode.
- **Momentum Further Constrains Sharpness at the Edge of Stochastic Stability** —
  Andreyev, Ananthkumar, Walden, Poggio, Beneventano. arXiv:2604.14108 (authors verified from
  the PDF, 2026-07-03). Batch-sharpness plateaus for SGDM: `2(1-β)/η` (small batch) vs `2(1+β)/η`
  (large batch, heavy-ball form ⇔ `ηλ < 2(1+β)/(1-β) = 2T_eff` in this project's EMA form).
  Cite when stating the momentum-shifted stability threshold — the threshold itself is known.

> When a paper is cited in the manuscript, add its verified BibTeX entry to
> `latex/references.bib`.
