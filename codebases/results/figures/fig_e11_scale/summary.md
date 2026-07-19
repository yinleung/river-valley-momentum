# fig_e11_scale — mini-batch nanoGPT: the momentum regimes at scale (E11)

**Feeding run** (latest for task `nanogpt`): `nanogpt_closedloop_lr4_b6_seeds3_T2000_c8c5ccf4`
(upstream karpathy/nanoGPT @ 3adf61e, pristine; 0.40M-param char-GPT, shakespeare_char,
batch 32×128, T=2000, MPS; EMA-SGDM on all parameters, no clipping; probe window = steps
1200–1455 of the last block's mlp.c_fc gradient).

**Panels.** (a) tail training loss (mean of last 200 steps) vs β per lr, 3 seeds,
surviving-seed means ± std; legend carries each lr's raw-stream HFER at β=0. (b) seed-0
training-batch loss curves at lr=0.4: β=0 diverges within steps (×), β=0.5 survives with
instability spikes, β∈{0.9, 0.95} train smoothly. (c) best-β benefit over β=0 (bars) and
HFER(G, β=0) (line) vs lr, with the white-stream baseline (0.402) dashed. (d) Muon
pre/post-polar final loss at background lr ∈ {0.2 (full), 0.05 (faint)}, η_o=0.02,
polar-only dashed per lr.

**Reproduce.**
```
cd codebases && python scripts/run_e11_scale.py    (~2 h on MPS)
cd figures   && python fig_e11_scale.py
```

**Headline numbers.** Gates A, C, E PASS; gates B, D FAIL as operationalized (kept, not
recalibrated — see caveats). Gate A: HFER(G, β=0) = 0.821/0.823/0.838 at lr=0.05/0.1/0.2 vs
white 0.402 — mini-batch transformer streams are hill-dominated at every stable lr (no
smooth sub-critical regime, unlike the full-batch MLP). Gate C + the regime dose-response:
best-β benefit over β=0 rises 0.8% → 3.5% → 11.9% → 21.8% across lr = 0.05/0.1/0.2/0.4.
The sweep-best cell is (lr=0.4, β=0.95) at train 1.946 — a learning rate where β=0 diverges
on one of three seeds and its survivors are among the worst cells (2.488). Gate E: pre-polar
beats post-polar 3/3 at lr=0.05 (median gap +0.023) and 2/3 at lr=0.2 (+0.035; post-polar
wins at β=0.9); polar-only mid-pack. Val loss TRACKS train loss (Spearman 0.90 over all
cells; per-row 0.83–1.0; best-train cell = best-val cell, val 2.018): on a real corpus the
optimization gain is not paid back in generalization, unlike the synthetic E7 diagnostic.

**Caveats.** Gate B required β=0 to diverge on every seed at lr=0.4; it diverged on 1/3
(seeds differ in batch order), so the CL-1-style boundary is seed-marginal at this scale —
the recorded claim is "β=0 marginally unstable and worst, momentum best", not a sharp
threshold. Gate D (mechanism-score rank prediction within lr rows) fails: median Spearman
−0.09 (rows −0.54/−0.09/+0.37) — the toy-calibrated score does not resolve the small
within-row loss spreads from a single probe seed; reported as a limitation, not softened.
Muon gaps are small (+0.02/+0.03) relative to the toy MLP's (+0.22 at EoS). β=0.99 columns
show the warm-up lag at every lr (CL-3′). No gradient clipping anywhere; divergence guard
at 3× the initial loss.
