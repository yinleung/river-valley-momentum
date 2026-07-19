# fig_e3_noise — momentum filters stochastic noise (E3)

**Feeding run** (latest for task `noisy_valley`, auto-resolved from `results/index/runs.csv`):
`noisy_valley_trajectory_sig2p0_seeds12_eta0p18_78215c16` (12 seeds)

**Panels.** (a) NSR(noise)=‖m−EMA(∇L)‖²/‖ξ‖² vs β for Gaussian / anisotropic-hill / heavy-tailed
noise, with the 1/N_eff=(1−β)/(1+β) floor (log y, error bars = seed std); (b) instantaneous-target
NSR=‖m−∇L(w_t)‖²/‖ξ‖² vs β (noise–lag tradeoff, minimum at moderate β); (c) river-alignment
improvement align(m)−align(g) vs β.

**Reproduce.**
```
cd codebases && python scripts/run_e3_noise.py
cd figures   && python fig_e3_noise.py
```

**Headline numbers.** decision gate PASS. NSR(noise) tracks 1/N_eff almost exactly (Gaussian:
1.00, 0.330, 0.053, 0.026, 0.003 vs floor 1.00, 0.333, 0.053, 0.026, 0.005); align(m) ≥ align(g)
for all β and noise types; NSR_inst minimised at β≈0.5 (Gaussian/heavy) or 0.95 (anisotropic).

**Caveats.** NSR isolates the *stochastic* residual via linearity of the EMA (m−EMA(∇L)=EMA(ξ));
the literal idea_v1 formula compares m to the *instantaneous* ∇L(w_t), which folds in the
deterministic lag bias — shown separately as NSR_inst in panel (b). Both are reported.
