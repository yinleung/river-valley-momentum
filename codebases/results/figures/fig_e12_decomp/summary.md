# fig_e12_decomp — decomposition of the mini-batch stream (E12)

**Feeding run** (latest for task `nanogpt`, probe `decomp`):
`nanogpt_decomp_lr3_seeds3_LB1024_k16_4f72b186`.

**Panels.** nanoGPT probe as in E11 (0.40M-param char-GPT, shakespeare_char, batch 32×128,
T=2000, probe window steps 1200–1455 of the last block's mlp.c_fc gradient), plain SGD
(β=0), lrs ∈ {0.05, 0.1, 0.2} × 3 seeds. At each probe step a 1024-sequence large-batch
gradient ĝ^LB at the same iterate splits the stream g^mb = ĝ^LB + ξ̂ (state stream +
sampling residual); top-16 restricted-Hessian eigenpairs at the window midpoint (block power
iteration on MPS, 50 iters, 512-seq eigen-batch). (a) per-frequency Frobenius energy of the
three streams at lr=0.2, seed mean (log), high band ω ≥ 0.6π shaded. (b) HFER by stream per
lr, seeds as dots, white baseline dashed. (c) high-band shares per lr: state share of the
raw high band, and the top-16 subspace share of the state high band. (d) per-direction HFER
of ⟨ĝ^LB, v_i⟩ vs ηλ_i (restricted eigenvalues), all lrs × seeds, white dashed.

**Reproduce.**
```
cd codebases && python scripts/run_e12_decomp.py       # ~20 min on MPS (--pilot for one run)
cd figures   && python fig_e12_decomp.py
```

**Headline numbers** (3-seed means from `metrics.json`). All three E12 gates PASS.
Gate A (state dominance): share_LB = E_h(LB)/(E_h(LB)+E_h(ξ)) = 0.829/0.842/0.927 at
lr = 0.05/0.1/0.2 (≥ 0.5 gated); cross terms at most 0.138 of the raw high band.
Direct accounting of the raw stream's high band (derived from the cached `spec_mb/lb/xi`
arrays, high band ω ≥ 0.6π, i.e. `Eh_s = spec_s[..., ω≥0.6π].sum(-1)` and ratios to
`Eh_mb`; quoted in the paper §E12): state 0.722/0.734/0.860, cross 0.128/0.129/0.074,
residual 0.149/0.137/0.066 — the reconstruction reproduces `share_lb` to all printed
digits, which validates the recipe. Gate B (residual whiteness): HFER(ξ̂) =
0.400/0.404/0.409 vs white 0.402 (±0.05 gated). Gate C (curvature alignment): HFER in/out =
0.880/0.887/0.942 vs 0.561/0.591/0.814; top-16 of 65,536 dims (2.4e-4) capture
share_high_in = 0.917/0.885/0.824 of the state high band. Report D: restricted η·λ_top =
0.27/0.42/0.21 (per-run range 0.06–0.47), far below 2 — the oscillating mode is a
network-level object (restriction only lower-bounds full sharpness by interlacing); the
restricted λ_i are alignment evidence, not the per-direction constants of the T2 spectrum.
Consistency: seed-0 HFER(g^mb) = 0.821/0.823/0.835 vs E11's published 0.821/0.823/0.838
(MPS nondeterminism at the third digit). Declared estimator bias: E‖ξ̂‖² overstates E‖ξ‖²
by 1/32 and whitens ĝ^LB by the same relative energy — both push against gates A/C.

**Caveats.** Eigenpairs are estimated once (window midpoint) and applied across the 256-step
window; subspace drift over the window is not measured. Power-iteration residuals ≤ 0.012 on
the top four directions; deeper directions are less converged. The paper (§E12) quotes the
seed-mean numbers above.
