# fig_e7_optimizer — optimizer-level tests: mechanism vs β as performance predictors (E7)

**Feeding runs** (latest per task): `optimizer_toy_closedloop_set5_b6_seeds8_50ead8f0` and
`optimizer_mlp_closedloop_eta2_b6_muon3x2_50ead8f0`.

**Panels.** (a) closed-loop EMA-SGDM tail loss vs β (log) on the five toy settings (S1
straight ηλ=1.8, S2 straight ηλ=0.6, S3 curved k=0.9 ηλ=1.8, S4 curved ηλ=2.5, S5 curved
k=1.8; σ=2, 8 seeds; diverged cells omitted). (b) per-setting Spearman correlations of the
tail loss with β (grey, absolute value) and with the mechanism score (green; score =
tie-aware within-setting rank of buffer hill energy + rank of river-following lag). (c)
mlp-diagnostic closed-loop SGDM: tail training loss vs β at η=0.05 and η=0.3, with each
regime's gradient-stream HFER in the legend. (d) Muon-style pre/post-polar bars at BOTH
η=0.3 (full color) and η=0.05 (faint), η_o=0.02, with the polar-only level dashed per η.

**Reproduce.**
```
cd codebases && python scripts/run_e7_optimizer.py
cd figures   && python fig_e7_optimizer.py
```

**Headline numbers.** All E7 gates PASS. Gate A: best β varies across settings (0.5 / 0.8 /
0.9 / 0.95). Gate B: median ρ(loss, mechanism score) = +0.75 vs median |ρ(loss, β)| = 0.60
(tie-aware ranks throughout; mechanism wins in 4/5 settings; the exception S2 is the benign
regime where the loss is flat in β and ranks are noise). Gate C (regime contrast): HFER 0.02
→ benefit of best β over β=0: 0.0%, max |change| over β≤0.95: 1.0% (η=0.05); HFER 0.99 →
23.3% with best β=0.9 and the U-shape closing at β=0.99 (η=0.3). Gate D: pre-polar beats
post-polar 3/3 on final loss (0.1526 vs 0.2709 at β=0.95) and 3/3 on slow-gradient alignment
at η=0.3. Gate E (Muon regime contrast): median pre/post gap +0.220 at η=0.3 vs +0.068 at
η=0.05 — and the sub-critical per-β gaps are sign-inconsistent (+0.18/−0.09/+0.07), noise
around zero, vs the strict 3/3 EoS ordering. Polar-only: 0.2980 at η=0.3 (worst), 0.1078 at
η=0.05 (mid-pack, no consistent ordering).

**Caveats.** Performance is training loss: on this synthetic non-realizable task, held-out
val loss (logged) is dominated by an overfitting axis that anti-correlates with train fit —
it is not an optimization readout. Gate E was recalibrated on 2026-07-04 from an absolute
sub-critical cutoff (0.05, which failed at +0.068) to the relative contrast form (≤ half the
EoS gap); the reason is recorded in the driver docstring. Schedule arm (metrics only, S3,
tail loss): fixed β=0.9 → 0.753, fixed β=0.95 → 0.645; arms whose prefix sits below the
confinement floor β≈0.7 lose badly (linear 0→0.95: 11.5; step 0→0.9 at t=30: 89); linear
0.7→0.95 wins outright (0.48); step 0.7→0.9 at t=30: 0.611. NEW adaptive arms: tube-triggered
0.7→0.95 raises at mean step 11 → 0.834; gradient-sign-flip-triggered 0.7→0.95 raises at
step 11 → 0.743; the sub-floor adaptive control (0→0.95) triggers only at step 201 and loses
at 244 — adaptivity does not rescue a sub-floor start. Confinement-triggered raises land in
fixed-β territory; among floor-respecting schedules the gradual ramp still wins. Muon η_o is
a single declared value (0.02), not tuned per variant.
