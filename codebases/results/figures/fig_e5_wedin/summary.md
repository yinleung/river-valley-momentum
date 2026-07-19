# fig_e5_wedin — filter-first theorem constants on the matrix/Muon toy (E5 overlay)

**Feeding run** (latest for task `matrix_muon`): `matrix_muon_pipeline_m24n18r3_T400_e2cdb461`
(overlay arrays `ov_*`, scenarios `highfreq_clean` and `highfreq`, β∈{0.9, 0.99}).

**Panels.** (a) clean scenario (G_t = S_t + (−1)^t a_H A, no noise — Theorem T5's setting):
measured buffer tail σ_{r+1}(M_t) against the two-sided guide |ε_t|σ_{2r+1}(A) ≤ σ_{r+1}(M_t)
≤ |ε_t|‖A‖₂ with ε_t the exact EMA-of-Nyquist coefficient (transient included; dashed =
upper, dotted = lower). (b) clean scenario: measured subspace error ‖sinΘ‖₂ of the buffer's
top-r left subspace against the Wedin guide |ε_t|·max(‖A V_S‖,‖AᵀU_S‖)/(σ_r(EMA(S)_t) −
|ε_t|‖A‖₂) (signal shrink tracked through the drifting EMA(S)). (c) noisy scenario
(σ_ξ = 0.2): measured tail against the deterministic guide (dotted) and the guide plus the
iid-Gaussian EMA operator-norm floor (dashed).

**Reproduce.**
```
cd codebases && python scripts/run_e5_matrix.py
cd figures   && python fig_e5_wedin.py
```

**Headline numbers.** Clean-scenario gates PASS at both β: the two-sided tail bound holds at
every step; the upper guide is tight to a median factor 1.05 over the tail half; sinΘ ≤ Wedin
guide wherever the gap condition holds, with end-of-run ratio 1.17 (β=0.9) and 1.16 (β=0.99).
In the noisy scenario the deterministic guide alone under-predicts the tail by ~12–50×
at large β — the stochastic floor dominates there, as T5's noise remark states; the summed
guide tracks the measured level.

**Caveats.** The noise-floor curve is a concentration heuristic (E‖G‖₂ ≈ entry-std·(√m+√n)),
not a theorem — panel (c) is reference, the T5 gates are the clean panels. The Wedin guide
uses the projected numerator max(‖A V_S‖,‖AᵀU_S‖); the crude ‖A‖₂ form is looser by ~1.5×.
