# fig_e8_headline — regimes of momentum in the river valley (E8)

**Feeding runs** (latest per task, auto-resolved from `results/index/runs.csv`):
`valley_regimes_trajectory_eta0p18_0p25_sig2_seeds12_T300_56464b7d` (panels a, b) and
`straight_valley_trajectory_eta0p18_lam10_T200_04b98895` (panel c, the E1 record).

**Panels.** Curved valley f(x)=2 sin(0.9x), start (−3, f(−3)+1), Gaussian gradient noise σ=2,
T=300, 12 seeds; trajectories are seed 0, insets are seed-mean loss (log). (a) η·λ=1.8,
β∈{0, 0.5, 0.9, 0.999} on the loss contour with the river floor dotted; annotation gives the
β=0.9 seed-mean tail rms of the hill offset against the CL-2 guide
√(ησ²/(λ(2−ηλ/T_eff))). (b) η·λ=2.5 > 2, β∈{0, 0.5, 0.9}: β=0 diverges 12/12 (divergence
marked ×), CL-2 annotation as in (a). (c) E1's clean straight valley: rms floor distance
d_rms vs β (values from E1 `metrics.json`).

**Reproduce.**
```
cd codebases && python scripts/run_e8_headline.py
cd figures   && python fig_e8_headline.py
```

**Headline numbers.** Decision gates PASS: (1) β=0 diverges 12/12 at ηλ=2.5 while β≥0.5 is
stable (CL-1 threshold 2T_eff); (2) in-tube rms vs CL-2 at β=0.9: panel a ratio 0.996, panel
b 1.102 plain / 1.026 bend-corrected; (3) lag reversal at β=0.999 in panel a (rms 0.549 vs
0.194 at β=0.9, x_final −1.33 vs 3.68, loss 3.78 vs 0.65). Escape frequency: 1.00 / 0.83 /
0.00 at β=0 / 0.5 / ≥0.9 in panel a — momentum restores the tube-confined regime.

**Caveats.** The CL-2 guide uses the straight-valley λ: linearizing the offset dynamics gives
contraction 1−ηλ(1+f′²) with noise variance η²σ²(1+f′²), and the (1+f′²) cancels except in
the (2−ηλ_eff/T_eff) denominator — plain-λ is accurate at large T_eff (β=0.9 cells), while
at β=0 the local threshold ηλ(1+f′²)<2 fails wherever |f′|>√(2/ηλ−1), which is why the β=0
trajectories escape rather than settle. Panel b's +10% plain-ratio is this bend correction
(1.026 after applying it, computed from the realized tail E[f′²] — derived, not fitted).
β=0.99 cells sit in-tube but above the CL-2 guide (rms 0.33 vs 0.19 in panel a): the extra
is deterministic bend-lag bias, not stationary variance — the T6 lag edge approaching.
