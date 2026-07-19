# fig_e10_bendwindow — the good-β window vs bend frequency (E10)

**Feeding run** (latest for task `bend_window`): `bend_window_sweep_k5_el2_b10_seeds16_c54d90a6`.

**Panels.** Curved valley y = 2 sin(kx) (amplitude fixed at a=2; k ∈ {0.3, 0.6, 0.9, 1.2,
1.8}; k=0.9 is the E2/E8 landscape), λ/μ=100, σ=2, 16 seeds, scored over the traveling
horizon 3/(ημ); ηλ ∈ {1.8, 2.5}; tube radius 2.5 = 1.25·a (margin so escape flags blow-up,
not marginal corner contact — declared deviation from the R=2 convention). (a) mean travel
loss vs β per k at ηλ=1.8 (log), stars at the best β. (b) window map at ηλ=1.8: green =
in-window (escape ≤ 0.25 ∧ lag ≤ 1.25× row minimum, contiguous around the lag minimum),
markers at the confinement floor (▷) and the lag-criterion top (◁). (c) floor and top edges
vs k for both ηλ. (d) measured river speed v (mean per-step x displacement of in-window
confined seeds) vs k.

**Reproduce.**
```
cd codebases && python scripts/run_e10_bendwindow.py
cd figures   && python fig_e10_bendwindow.py
```

**Headline numbers.** All three E10 gates PASS. Gate 1: the confinement floor rises
monotonically with k — β_floor = 0.3/0.5/0.7/0.7/0.9 at ηλ=1.8 and 0.3/0.5/0.7/0.8/0.95 at
ηλ=2.5. Gate 2: the loss U-shape holds at every k ≥ 0.9 (best β interior — 0.8/0.8/0.95 at
ηλ=1.8 — β=0.999 costs ≥1.5× the best, β=0 escapes). Gate 3: the window narrows — widths in
grid steps 6/6/5/4/4 (ηλ=1.8) and 6/6/3/2/3 (ηλ=2.5) from k=0.3 to k=1.8. Reported, not
gated: the river speed collapses with k (0.047→0.013 at ηλ=1.8; 0.064→0.015 at ηλ=2.5), so
the temporal bend frequency k·v self-regulates and the lag-criterion top edge stays soft
(0.95–0.999 across cells); the calibrated lag_edge_beta guide (eps*=0.976 at the k=0.9
anchor) is therefore nearly flat and uninformative about the top-edge ordering.

**Caveats.** Two pilot designs were rejected and one prediction corrected (recorded in the
driver docstring): at fixed max slope a·k the window *widens* with k (corner-cutting through
small wiggles is free), and the naive expectation "optimal β decreases with curvature" is
replaced by the two-sided picture — the floor rises (local sharpness λ(1+f′²) grows with
a·k), the window narrows, and the closed loop protects its top edge by slowing the river
rather than lagging it. At k=1.8 the lag window collapses to {0.999}: no β both tracks and
filters — the tube-confined regime is not restorable by β alone at this curvature.
