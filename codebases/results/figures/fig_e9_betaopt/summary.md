# fig_e9_betaopt — does larger β improve river-valley optimization? (E9)

**Feeding run** (latest for task `beta_opt`): `beta_opt_sweep_b10_el6_c3_seeds16_60596297`.

**Panels.** Straight noisy valley, σ=2, 16 seeds, T=6000, tail = second half (β grid
{0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995}; ηλ ∈ {0.6, 1.2, 1.6, 1.8, 2.2, 2.5};
λ/μ ∈ {10, 100, 1000}). (a) tail hill loss (λ/2)d² vs β per ηλ at λ/μ=100, faint per-seed
points, dashed CL-3 guides (`core.closedloop.stationary_hill_loss`). (b) measured stationary
rms / CL-2 guide per (ηλ, β) cell at λ/μ=100; cells beyond the CL-1 threshold marked div.
(c) measured maximal relative hill-loss reduction per (ηλ, λ/μ) against the CL-3 prediction
ηλ/2 (dashed); shaded region ηλ≥2 is where β=0 diverges. (d) tube-escape frequency vs β per
ηλ (R=2). (e) diverged-seed fraction per cell with the CL-1 boundary ηλ=2T_eff overlaid.
(f) normalization ablation: tail hill loss vs β under EMA at fixed η (ηλ=0.6) and heavy-ball
at fixed η_HB (η_HBλ=0.6, exactly the EMA run at η=η_HB/(1−β)), seed-SE bars, each with its
composed CL-3 guide dashed; the β=0 cells coincide by construction.

**Reproduce.**
```
cd codebases && python scripts/run_e9_betaopt.py   (~15 min)
cd figures   && python fig_e9_betaopt.py
```

**Headline numbers.** All six E9 gates PASS. Gate 1: the CL-1 boundary is exact on all three
conditionings (every predicted-unstable cell diverges 16/16, every comfortably-stable cell
0/16). Gate 2: rms/guide within 2·seed-SE+0.02 in 174/174 comfortably-stable cells. Gate 3+4:
measured maximal relative hill-loss reduction vs the CL-3 value ηλ/2, at λ/μ=100 —
0.30/0.30, 0.60/0.60, 0.82/0.80, 0.90/0.90 at ηλ=0.6/1.2/1.6/1.8; at ηλ ∈ {2.2, 2.5} β=0
diverges while β=0.9 runs (the stabilized regime). Gate 5: hill loss ∝ σ² exactly (measured
ratios 4.00/16.00/4.00 vs 4/16/4 across σ ∈ {0.5, 1, 2}). Gate 6: heavy-ball at fixed η_HB is
monotone INCREASING in β (Spearman +1.00, on its guide in 10/10 cells, 0.0857 → 13.2) while
EMA at fixed η decreases (0.0857 → 0.060) — the sweep direction is a normalization
convention, and both directions are the same formula.

**Caveats.** The tail-loss floor is β-independent (the river coordinate's own stationary
term, CL-2 per-coordinate sum), so total loss flattens where hill loss saturates. T/2 = 3000
≥ 7.5·T_eff even at β=0.995, so every cell clears the CL-3′ burn-in; the finite-horizon
turn-up predicted by CL-3′ is deliberately absent from this design (E7's β=0.99 arm shows
it). Divergence declared at |w| > 1e8; diverged seeds are reported per cell and never hidden.
