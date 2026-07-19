# fig_e5_matrix — filter-first (pre-polar) wins (E5)

**Feeding run** (latest for task `matrix_muon`, auto-resolved from `results/index/runs.csv`):
`matrix_muon_pipeline_m24n18r3_T400_e2cdb461`

**Panels.** Subspace error ‖sinΘ(U_pipeline, U_S)‖₂ (log, lower better) vs β for (a) the
deterministic high-frequency disturbance H_t=(−1)ᵗA and (b) the stochastic disturbance Ξ_t;
rank-r signal alignment ⟨A,U_rV_rᵀ⟩_F/r (higher better) vs β for (c) high-freq and (d) stochastic.
Series: pre-polar O(EMA(G)), post-polar O(EMA(O(G))), polar-only O(G).

**Reproduce.**
```
cd codebases && python scripts/run_e5_matrix.py
cd figures   && python fig_e5_matrix.py
```

**Headline numbers.** key test PASS in both scenarios. High-freq, β=0.9: subspace error
pre-polar 0.114 < post-polar 0.133 < polar-only 0.508; alignment 0.994 > 0.992 > 0.909. Pre-polar
advantage grows with β; polar-only is β-independent (no momentum).

**Caveats.** Signal alignment uses the rank-r factor U_rV_rᵀ (the faithful low-rank target); the
full polar factor O(S) of idea_v1 completes the 15 null directions of the rank-3 S arbitrarily and
compresses the metric. The subspace metric is magnitude-free and is the headline. Post-polar is the
raw buffer EMA(O(G)) used directly (idea_v1 definition); its alignment is partly lowered by
averaging rotated orthogonal factors, so the magnitude-free subspace error is the fair comparison.
