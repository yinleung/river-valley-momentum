# fig_e6_onset — confinement onset in the real-network stream (E6 extension)

**Feeding run** (latest for task `mlp_diagnostic`):
`mlp_diagnostic_trajectory_cond120_eta0p3_T600_de970d69` (windowed arrays `win_*`,
`loss_curve`).

**Panels.** Windows of length 100, stride 50, over the full 600-step first-layer gradient
stream (burn-in included). (a) windowed HFER per window start, with the training-loss curve
on the right axis and the onset (first window with HFER ≥ 0.9) marked. (b) per-window
open-loop EMA alignment gain align(m) − align(g) at β=0.9 against the window-mean gradient
(first 20 in-window steps dropped as EMA warm-up).

**Reproduce.**
```
cd codebases && python scripts/run_e6_trajectory.py
cd figures   && python fig_e6_onset.py
```

**Headline numbers.** Onset gates PASS: first-window HFER 0.025 (vs 0.998 at the end); onset
at window 4 (step 200); 68% of the total loss drop happens before onset; post-onset windows
all ≥ 0.98; the EMA alignment gain is +0.04 (mean pre-onset) vs +0.70 (mean post-onset).

**Caveats.** This is the reconciliation of li2025frequency's "high-frequency helps early"
with the filtering story: before confinement the steep-direction gradient is the descent
signal itself (low temporal frequency — filtering has nothing to remove and costs little);
the Nyquist hill appears only once the iterate is confined, and that is when filtering pays.
One run, one layer, full-batch — a regime demonstration, not a scaling claim.
