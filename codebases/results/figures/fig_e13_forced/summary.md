# fig_e13_forced — forced-disturbance controls on the FR guides (E13)

**Feeding run** (latest for task `forced_controls`):
`forced_controls_sweep_om4_b7_seeds12_da578cc6`.

**Panels.** Straight noisy valley (λ=10, μ=0.1, ηλ=1.8, σ=1), tone A cos(ωt) with A=3 added
to the hill gradient, T=6000, tail half scored, 12 seeds, β ∈ {0, 0.3, 0.5, 0.7, 0.9, 0.95,
0.99}, ω ∈ {0.02, θ_{0.9}=0.436, 0.6π, π}. (a) tail hill loss vs β per arm (log), dashed
the forced_hill_loss guides (FR + CL-2 superposition), seed-SE error bars. (b) lock-in
forced amplitude vs β per arm (log), dashed A·|G_β(ω)|. (c) |G_β(ω)| vs ω per β
(core.closedloop.forced_gain), the four arm frequencies dotted.

**Reproduce.**
```
cd codebases && python scripts/run_e13_forced.py       # ~2 min
cd figures   && python fig_e13_forced.py
```

**Headline numbers** (from `metrics.json`). All four E13 gates PASS. Gate 1: loss on the
parameter-free guide in 28/28 (ω, β) cells (within 2×SE + 5%), spanning 0.044–36.6. Gate 2
(Nyquist removal): amplitude ratio amp(π, β=0.99)/amp(π, β=0) = 5.03e-4 (guide 5.05e-4).
Gate 3 (passband neutrality): amp(0.02, ·) max/min = 1.022 over the β grid (guide 1.02) —
no β removes a passband disturbance. Gate 4 (resonance harm): loss(θ_{0.9}, β=0.9) /
loss(θ_{0.9}, β=0) = 5.99 (guide 6.01) — momentum amplifies its own hill-mode frequency;
the same tone costs β=0.99 only 0.047 (its resonance sits elsewhere): the harm is
frequency-matched.
