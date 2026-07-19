# fig_e2_curved — curved river-valley filtering–lag tradeoff (E2)

**Feeding run** (latest for task `curved_valley`, auto-resolved from `results/index/runs.csv`):
`curved_valley_trajectory_a2_k0p9_eta0p18_T300_aa4bcad3`

**Panels.** (a) filtered trajectories (β=0.9, 0.95, 0.99) on the bending floor y=f(x)=a·sin(kx),
log-loss contour background; (b) per-step river alignment to the live tangent r(x_t), g vs m;
(c) hill-normal energy of m vs β (log); (d) river-following lag 1−align(m,r) vs β.

**Reproduce.**
```
cd codebases && python scripts/run_e2_curved.py
cd figures   && python fig_e2_curved.py
```

**Headline numbers.** decision gate PASS; river alignment 0.136 (β=0) → **0.991 (β=0.9, best)** →
0.935 (β=0.99); hill energy falls ~6 orders of magnitude from β=0 to β=0.9; lag minimised at
β=0.9 and rises again at β=0.99 (the bend outruns the EMA memory).

**Caveats.** Panel (a) omits β=0 (and β=0.5): the near-unstable β=0 baseline oscillates off-scale
(x spans ≈[−100, 150]); its behaviour is quantified in (b)–(d). The aggressive regime (η·λ=1.8)
makes β=0.5 wander (river progress negative); the clean tradeoff lives in the β∈{0.9,0.95,0.99}
range plus the β=0 oscillation baseline.
