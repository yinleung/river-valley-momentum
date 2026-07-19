# fig_e1_straight — straight river-valley filtering demo (E1)

**Feeding run** (latest for task `straight_valley`, auto-resolved from `results/index/runs.csv`):
`straight_valley_trajectory_eta0p18_lam10_T200_04b98895`

**Panels.** (a) iterate trajectories over β on the L(x,y)=(μ/2)(x−x*)²+(λ/2)y² contour;
(b) hill component vs step — raw g_{t,y} (β=0) vs momentum buffer m_{t,y} (β=0.9, 0.99);
(c) temporal spectrum of the hill component (rectangular window), high band ω≥0.6π shaded;
(d) HSR and MSR vs β with the |H_β(π)|²=((1−β)/(1+β))² guide; (e) per-step river alignment of
g vs m, averaged over t; (f) rms distance to the river floor.

**Reproduce.**
```
cd codebases && python scripts/run_e1_straight.py     # -> results/cache/, results/index/runs.csv
cd figures   && python fig_e1_straight.py             # -> out/fig_e1_straight.{pdf,png}
```

**Headline numbers (from metrics.json).** decision gate PASS (72 hill-gradient sign changes at
β=0); HSR monotone decreasing 1.00 → 0.294 → 0.055 → 0.028 → 0.0056; MSR tracks |H_β(π)|²;
align(m) ≥ align(g) for every β.

**Caveats.** Spectra use a rectangular window so the early oscillation transient is counted (a Hann
window would zero it at the edges). `dist_rms` is minimised at β=0.5 and rises for β≥0.9: with fixed
η the hill subsystem becomes underdamped (closed-loop eigenvalue magnitude √β≈0.995) and rings
longer — a genuine preview of the T6 filtering–lag tradeoff, not a defect. Config (μ,λ,x*) is read
from the run record; the loss contour is evaluated on a grid for display only.
