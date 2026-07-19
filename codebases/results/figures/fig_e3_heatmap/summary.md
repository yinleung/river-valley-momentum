# fig_e3_heatmap — CL-2 exactness, tube restoration, widening β window (E3 extension)

**Feeding run** (latest for task `cl2_grid`): `cl2_grid_sweep_el8_b5_seeds8_ff290968`.

**Panels.** (a) straight valley, σ=2, 8 seeds, T=6000 (tail half scored): measured stationary
rms of the hill coordinate / CL-2 guide √(ησ²/(λ(2−ηλ/T_eff))), over ηλ∈{0.3…3.0} ×
β∈{0, 0.5, 0.9, 0.95, 0.99}; cells beyond the CL-1 threshold marked "div". (b) curved valley
(E8 landscape), T=300: escape frequency (fraction of seeds leaving the tube R=2) vs β at
ηλ∈{1.8, 2.5}. (c) good-β window vs conditioning λ/μ∈{10, 100, 1000} at ηλ=1.8: each row's
window (green, arrow-spanned) = {β: escape freq ≤ 0.25 AND mean river-following lag ≤ 1.25 ×
its own minimum} over that conditioning's traveling horizon K = 3/(ημ) (16 seeds).

**Reproduce.**
```
cd codebases && python scripts/run_e3_heatmap.py
cd figures   && python fig_e3_heatmap.py
```

**Headline numbers.** Gate 1 PASS: 36/37 comfortably-stable cells have ratio 1 within
2·seed-SE+0.02 (worst rows at β=0.99 where the autocorrelation time ~1/(1−√β) makes the
estimator noisy); all cells with ηλ ≥ 2T_eff diverge, none below. Gate 2 PASS: window upper
edges β = 0.7 / 0.95 / 0.999 for λ/μ = 10 / 100 / 1000 — the window widens with conditioning
as T6′ predicts (upper edge where T_eff reaches the river traversal scale 1/(ημ):
T_eff(0.7)=5.7 vs 1/(ημ)=5.6 at λ/μ=10). Escape frequency in (b) drops to 0 exactly where
the CL-2 rms is small against R and where rms/pred → ≈1 (0.00 for β≥0.7 at ηλ=1.8; β=0
diverges at ηλ=2.5).

**Caveats.** The lag tolerance in (c) is relative because the absolute lag level under σ=2
noise is set by the buffer's noise floor (~σ/√T_eff against the river gradient), not by
tracking error; the β-shape is the mechanism signal. River progress at λ/μ=1000 is
diffusion-limited (hill-coupled x random walk exceeds the river length), so (c) reports the
mechanism window, not a performance window — mechanism-vs-performance is E7's question.
