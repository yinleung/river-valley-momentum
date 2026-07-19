# fig_e6_trajectory — real-network gradient frequency diagnostic (E6)

**Feeding run** (latest for task `mlp_diagnostic`, auto-resolved from `results/index/runs.csv`):
`mlp_diagnostic_trajectory_cond120_eta0p3_T600_de970d69`

**Panels.** (a) mean magnitude of the first-layer gradient over training (edge-of-stability
oscillation); (b) temporal Frobenius spectrum `|Ĝ(ω)|_F`, high band ω≥0.6π shaded, dominant peak
marked; (c) slow-gradient alignment of the EMA buffer vs β (left, orange) against the raw-gradient
baseline (dashed), with high-band MSR and the `|H_β(π)|²` guide on the right log axis.

**Reproduce.**
```
cd codebases && python scripts/run_e6_trajectory.py    # trains the MLP, logs the run
cd figures   && python fig_e6_trajectory.py
```

**Headline numbers (metrics.json).** decision gate PASS. Temporal HFER(G)=0.994 vs a white-stream
baseline 0.406; Frobenius spectral peak at ω/π=1.000 (Nyquist). MSR(β) matches `|H_β(π)|²`
(0.111, 0.00278, 0.00066, 2.5e-5). Slow-(mean-)gradient alignment rises from 0.261 (raw g) to
0.362/0.660/0.732/0.823 (β=0.5/0.9/0.95/0.99); the loss descends 0.51→0.31 (net river progress).

**Setup / caveats.** First-party `mlp-diagnostic` task: a 2-layer ReLU MLP trained full-batch by
plain GD on an ill-conditioned non-realizable target `y=sin(3·⟨w,x⟩)`. The ω=π structure is the
edge-of-stability phenomenon and is regime-dependent: at sub-critical η (≲0.2) the gradient stream
is smooth (HFER≈0), at η≈0.3 it is dominated by the Nyquist oscillation (HFER≈0.99), and η≳1
diverges. The slow (river) component is small here but useful — the loss still decreases — so the
slow/fast decomposition holds with a dominant hill. The `|H_β|`/MSR match is exact by linearity of
the EMA; the non-trivial findings are the high HFER and the 3× alignment gain from filtering.
