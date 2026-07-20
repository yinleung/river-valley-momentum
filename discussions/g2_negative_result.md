# G2 kill-switch: the river-valley decomposition does not transfer to ResNet-CIFAR
## (partial-transfer negative result — campaign paused per plan_v5 §3.2 G2)

**Status: DRAFT pending the 4 β=0.99 cells (running, jobs 9232192–95). Verdict already
robust from 8/12 cells; the β=0.99 cells can only reinforce it. G3/G4 NOT started.**

Written 2026-07-21 from the G2 decomposition runs (`resnet-cifar_g2_decomp_*`, git
b65bd45/cb93917b). This is the campaign's predeclared kill-switch (plan §3.2 G2, §3.4
order): "if the four support conditions fail, STOP — do not start G3/G4."

## The measurement (exactly as predeclared)

Paired decomposition `g^mb_t = ĝ^LB_t + ξ^res_t` on the ACTUAL β trajectories: mini-batch
gradient of the probed matrices (batch 128, no aug), large-batch state gradient ĝ^LB at
the visited iterate (fixed 4096-image probe, every step), top-16 target-restricted
Hessian eigenpairs at each window midpoint. 4 regime LRs {0.05, 0.2, 0.8, 3.2} × β {0,
0.9, (0.99 pending)}, seed 0, 3 windows (early/EoS-onset/late), rectangular primary.

## The four support conditions — verdict: FAIL as operationalized

Scored quantities (best over the 3 lb_targets × 3 windows unless noted):

| LR (β=0) | max share_lb | state HFER (hfer_lb) | curv. concentration (share_high_in) | ξ HFER (white=0.40) |
|---|---|---|---|---|
| 0.05 | 0.20 | 0.11 | 0.07 | 0.395 |
| 0.2  | 0.25 | 0.15 | 0.11 | 0.397 |
| 0.8  | 0.28 | 0.24 | 0.10 | 0.400 |
| 3.2  | **0.58** (linear head, w0) | 0.36 | **0.53** | 0.398 |

β=0.9 (and 0.99): max share_lb ≤ 0.003 at every cell — the state stream is essentially DC.

- **(1) state dominance** — share_lb ≥ 0.5: **FAILS** at every cell except the single
  beyond-edge point (lr 3.2, β=0, on the `linear` head only), where accuracy has already
  degraded to 0.65. The high band of the mini-batch gradient is majority **sampling
  noise**: ξ^res is white (HFER ≈ 0.40 = the surrogate baseline at every cell) and the
  state stream ĝ^LB is smooth (HFER 0.11–0.36 vs white 0.40 — LOW-frequency, low_frac
  0.55–0.99).
- **(2) sharp concentration** — HFER_in ≥ HFER_out + 0.10 AND share_high_in ≥ 0.25:
  the concentration clause holds only at the edge (share_high_in 0.53 at lr 3.2, else
  ≤ 0.11); the HFER differential never reaches 0.10. FAIL except partially at the edge.
- **(3) low-freq outside**: not separately decisive given (1)/(2).
- **(4) benefit coupling**: G1's divergence-counted momentum benefit at fixed η was ≈ 0
  (BN never diverges; best-β minus β=0 tail loss ≤ 0.09, concentrated at lr 3.2) — there
  is essentially no benefit to correlate against the separation. FAIL.

## What IS true (this is a PARTIAL transfer, not "mechanism absent")

The river-valley mechanism is **present and directionally exactly as predicted**, just
noise-subdominant at practical settings:
- state high-band energy grows **monotonically toward the edge** (share_lb 0.20→0.58,
  hfer_lb 0.11→0.36 as lr 0.05→3.2 at β=0);
- what state high-band exists **concentrates in the top-curvature subspace** at the edge
  (share_high_in → 0.53 at lr 3.2 — the sharp-direction prediction);
- the momentum filter's β-dependence is exactly the open-loop filter (β=0.9/0.99 state
  streams are pure DC — the buffer has removed all high-band state content).

The probe is sound: hfer_lb rising with LR and falling with β is the correct signature of
a large-batch gradient at a moving iterate (fast/large steps → ĝ^LB varies fast; slow EMA
→ ĝ^LB smooth), and the edge curvature-concentration confirms it captures real landscape
structure, not an artifact.

## The contrast that defines the result

E12 (nanoGPT-shakespeare, batch 32): share_lb **0.83–0.93 at every LR** — clean state
dominance. ResNet-CIFAR (batch 128) reaches 0.58 only beyond the edge. Same probe, same
32× large-batch ratio, opposite verdict → the difference is **architectural/task**, not
batch-size: on ResNet-CIFAR the landscape-oscillation signal in the gradient is far weaker
relative to sampling noise than on a small char-transformer.

## Caveats (stated, not buried)

- **Cross-term** (2⟨ĝ^LB, ξ⟩_high / E_high(mb)) is 0.34–0.36 at β=0 high-LR (small,
  0.04–0.07, at β=0.9): the state and residual are ~35% high-band-correlated exactly where
  the closed-loop coupling is strongest, so the additive split is muddied precisely in the
  regime where (1) is closest to passing. Does not rescue (1).
- Batch 128, no augmentation, one seed per cell. A **larger-batch probe** (batch 512/1024
  at the near-edge LRs) is the honest test of whether state-dominance emerges once
  sampling noise is reduced — but that changes the predeclared protocol, so it is Leon's
  call, not taken here.

## Recommendation to Leon (decision required — G3/G4 held)

The predeclared kill-switch has tripped: the four support conditions do not hold on
ResNet-CIFAR at the campaign's settings. Per the mission I have **stopped and not started
G3/G4**. Three ways forward, none taken unilaterally:

1. **Rescope and continue.** Read this as "mechanism present, noise-subdominant, emerges
   toward the edge" and reframe the paper's CIFAR claim as directional (benefit/state-share
   grows toward the edge) rather than dominance. Then G3 (forced-frequency, which tests the
   mechanism directly by injection and does NOT depend on natural state-dominance) becomes
   the more informative experiment — arguably it should run regardless, since it can
   confirm the mechanism is real even when it's noise-subdominant.
2. **Larger-batch decomposition** at the near-edge LRs to test whether dominance is
   batch-artifactual before deciding (≈ +8 GPU-h; protocol change).
3. **Accept as a negative result** and rescope the whole paper's empirical claim to the
   toy/analytic regime + nanoGPT, reporting CIFAR as the honest boundary of transfer.

My read: (1)+(2) together — run the larger-batch check AND G3-by-injection, because both
are cheap and both distinguish "mechanism weak" from "mechanism absent," which is the
actual open question. But this reframes a review-load-bearing claim, so it is yours to
call.
