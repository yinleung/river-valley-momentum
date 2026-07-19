# fig_e4_frequency — empirical EMA transfer ratio vs |H_β(ω)| (E4)

**Feeding run** (latest for task `frequency_validation`, auto-resolved from `results/index/runs.csv`):
`frequency_validation_spectral_T512_hann_2e8a73ae`

**Panels.** (a) synthetic stream (slow tone + near-π tone + white noise); (b) straight-valley
hill-gradient stream; (c) curved-valley hill-gradient stream. In each, markers are the empirical
ratio R(ω)=|m̂(ω)|/|ĝ(ω)| on significant-energy bins; solid lines are the theory
|H_β(ω)|=(1−β)/√(1−2β cos ω+β²), for β∈{0.5,0.9,0.95,0.99} (log y).

**Reproduce.**
```
cd codebases && python scripts/run_e4_frequency.py
cd figures   && python fig_e4_frequency.py
```

**Headline numbers.** weighted-median relative error R vs |H_β|: synthetic <0.001, curved <0.003,
straight ≤0.115 (boundary/transient). High-band *median* ratios sit within 6% of the Nyquist
floor (1−β)/(1+β) for the synthetic/curved streams and within 12% for the boundary-affected
straight stream (the earlier "match to three digits" phrasing was unsupported by `metrics.json`;
corrected 2026-07-03). The
straight-stream markers cluster at ω≳0.5π (no low-frequency energy) — direct evidence that the
hill gradient is high-frequency (Theory T3).

**Caveats.** Streams are *fixed* inputs; m=EMA_β(g) is formed open-loop so the ratio is a clean
filter test. The significant-bin mask (input power > 1% of peak) drops near-zero-energy bins where
R is 0/0; it is recomputed from the cached stream for display.
