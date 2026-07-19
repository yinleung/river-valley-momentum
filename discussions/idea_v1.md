# Research Plan: Momentum as Temporal Filtering in River-Valley Optimization

## 0. Core Idea

We want to develop a general theory that interprets **momentum as a temporal filter for gradient streams**.

The current Muon paper already shows one important instance: momentum suppresses perturbations before Muon’s polar/orthogonalization step, creating a cleaner spectral structure and improving signal alignment. This motivates the principle:

[
\textbf{filter first, then nonlinearize.}
]

The earlier frequency-domain paper gives the natural mathematical language: momentum is a recursive filter with transfer function

[
H(z)=\frac{v}{1-u z^{-1}},
]

and magnitude response

[
|H(\omega)|
===========

\frac{|v|}
{\sqrt{1-2u\cos\omega+u^2}}.
]

This lets us discuss how momentum attenuates or amplifies different temporal gradient frequencies. 

The new research direction is:

[
G_t
===

G_t^{\mathrm{slow}}
+
G_t^{\mathrm{fast}}
+
\Xi_t,
]

where:

* (G_t^{\mathrm{slow}}): useful slowly varying optimization signal,
* (G_t^{\mathrm{fast}}): rapidly changing/high-frequency gradient component,
* (\Xi_t): stochastic mini-batch noise.

The river-valley landscape gives a concrete geometric reason why this decomposition should exist: the **river direction** changes slowly, while the **hill direction** may oscillate rapidly under large learning rates.

---

# 1. Main Hypothesis

## 1.1 Scientific Hypothesis

Momentum improves optimization when useful gradient components are temporally slower than harmful or less useful components.

More concretely:

[
m_t = \beta m_{t-1} + (1-\beta)g_t
]

should satisfy:

[
m_t
\approx
g_t^{\mathrm{river}}
+
\text{attenuated } g_t^{\mathrm{hill}}
+
\text{attenuated noise}.
]

So momentum should improve the **river-to-hill signal ratio**:

[
\frac{|P_{\mathcal R} m_t|}
{|P_{\mathcal H} m_t|}

>

\frac{|P_{\mathcal R} g_t|}
{|P_{\mathcal H} g_t|},
]

where (P_{\mathcal R}) projects onto the river direction and (P_{\mathcal H}) projects onto the hill direction.

## 1.2 Expected Big-Picture Contribution

The eventual paper should argue:

> Momentum is not only acceleration and not only stochastic denoising. It is a temporal spectral filter that preserves slowly varying optimization signal while suppressing rapidly varying gradient components caused by stochastic noise, high-curvature oscillation, and nonlinear update instability.

---

# 2. Experimental Plan

The experiments should go from **simple controlled demos** to **real neural-network diagnostics**.

The first goal is not to prove the theory, but to verify whether the proposed mechanism is empirically visible.

---

# Experiment E1: Straight River-Valley Filtering Demo

## Goal

Show that in a simple river-valley landscape, momentum with larger (\beta) suppresses hill-direction oscillation while preserving river-direction progress.

## Landscape

Use a two-dimensional loss:

[
L(x,y)
======

\phi(x)+\frac{\lambda}{2}y^2,
]

where:

* (x): river coordinate,
* (y): hill coordinate,
* (\lambda \gg 1): steep hill curvature.

A simple choice:

[
\phi(x)=\frac{\mu}{2}(x-x^\star)^2,
\qquad
0<\mu\ll \lambda.
]

Then:

[
\nabla L(x,y)
=============

\begin{bmatrix}
\mu(x-x^\star) \
\lambda y
\end{bmatrix}.
]

## Update Rule

For each (\beta\in{0,0.5,0.9,0.95,0.99}), run:

[
m_t = \beta m_{t-1}+(1-\beta)\nabla L(w_t),
]

[
w_{t+1}=w_t-\eta m_t.
]

Choose (\eta\lambda>1) but not too large, so that the hill coordinate oscillates:

[
y_{t+1}\approx (1-\eta\lambda)y_t.
]

When (1-\eta\lambda<0), the hill component alternates sign, creating a high-frequency temporal gradient component.

## Plots

Generate:

1. Contour plot of (L(x,y)) with trajectories for different (\beta).
2. Time series of raw hill gradient (g_{t,y}).
3. Time series of momentum hill component (m_{t,y}).
4. Temporal spectrum of (g_{t,y}) and (m_{t,y}).
5. River alignment over time.

## Metrics

### Hill Suppression Ratio

[
\mathrm{HSR}(\beta)
===================

\frac{\sum_t |m_{t,y}|^2}
{\sum_t |g_{t,y}|^2}.
]

Expected:

[
\mathrm{HSR}(0.95) < \mathrm{HSR}(0.9) < \mathrm{HSR}(0.5) < \mathrm{HSR}(0).
]

### River Alignment

The river direction is (r=(1,0)). Compute:

[
\mathrm{Align}_{\mathcal R}(g_t)
================================

\frac{|\langle g_t,r\rangle|}
{|g_t|},
]

[
\mathrm{Align}_{\mathcal R}(m_t)
================================

\frac{|\langle m_t,r\rangle|}
{|m_t|}.
]

Expected:

[
\mathrm{Align}_{\mathcal R}(m_t)

>

\mathrm{Align}_{\mathcal R}(g_t)
]

for moderate or large (\beta).

### Distance to River

[
d_t = |y_t|.
]

Expected: moderate (\beta) should reduce oscillation around the valley floor.

## Success Criteria

This experiment is successful if:

* high (\beta) clearly suppresses the hill-direction oscillation;
* the momentum direction aligns better with the river direction than the raw gradient;
* temporal FFT shows reduced high-frequency energy in (m_{t,y});
* moderate (\beta), such as (0.9) or (0.95), improves trajectory stability.

## Failure Criteria

This experiment fails if:

* (m_t) does not suppress the hill component;
* river alignment does not improve;
* all (\beta) values behave nearly identically;
* improved performance is visible only in final loss but not in mechanism metrics.

---

# Experiment E2: Curved River-Valley Filtering Demo

## Goal

Test whether momentum still helps when the river bends, and whether overly large (\beta) causes lag.

## Landscape

Use:

[
L(x,y)
======

\phi(x)
+
\frac{\lambda}{2}(y-f(x))^2.
]

Suggested choices:

[
f(x)=a\sin(kx)
]

or

[
f(x)=a x^2.
]

The river floor is:

[
y=f(x).
]

## River Tangent

The local river tangent is:

[
r(x)
====

\frac{(1,f'(x))}
{\sqrt{1+f'(x)^2}}.
]

The hill normal is:

[
n(x)
====

\frac{(-f'(x),1)}
{\sqrt{1+f'(x)^2}}.
]

## Metrics

### River Alignment

[
\mathrm{Align}_{\mathcal R}(m_t)
================================

\frac{|\langle m_t,r(x_t)\rangle|}
{|m_t|}.
]

### Hill Energy

[
\mathrm{HillEnergy}(m)
======================

\sum_t |\langle m_t,n(x_t)\rangle|^2.
]

### Lag Error

Compare the momentum direction to the current river tangent:

[
\mathrm{Lag}(m_t)
=================

1-
\frac{|\langle m_t,r(x_t)\rangle|}
{|m_t|}.
]

## Expected Result

Moderate (\beta), such as (0.9) or (0.95), should perform best.

Very large (\beta=0.99) may over-smooth and lag behind the bending river.

Expected qualitative pattern:

[
\beta=0
\quad
\text{has large hill oscillation},
]

[
\beta=0.9,0.95
\quad
\text{filter hill oscillation and follow the river well},
]

[
\beta=0.99
\quad
\text{filters strongly but may lag}.
]

## Success Criteria

This experiment is successful if it shows a **filtering-lag tradeoff**:

[
\text{larger }\beta
\Rightarrow
\text{less high-frequency hill energy}
]

but

[
\text{too large }\beta
\Rightarrow
\text{larger river-following lag}.
]

This would be very useful theoretically because it explains why the best momentum coefficient should not always be maximal.

---

# Experiment E3: River-Valley with Stochastic Noise

## Goal

Show that momentum filters both:

1. deterministic high-frequency hill oscillation;
2. stochastic mini-batch-like perturbation.

## Stochastic Gradient

Use:

[
g_t
===

\nabla L(w_t)+\xi_t.
]

Try three noise types:

1. Gaussian isotropic noise;
2. anisotropic noise stronger in hill directions;
3. heavy-tailed finite-variance noise.

## Metrics

Use the same metrics as E1/E2, plus:

### Noise Suppression Ratio

If the exact gradient is known:

[
\mathrm{NSR}(\beta)
===================

\frac{\sum_t |m_t-\nabla L(w_t)|^2}
{\sum_t |g_t-\nabla L(w_t)|^2}.
]

Expected:

[
\mathrm{NSR}(\beta)
\downarrow
\quad
\text{as }
\beta
\uparrow.
]

## Success Criteria

Successful if momentum simultaneously:

* suppresses stochastic noise;
* suppresses deterministic hill oscillation;
* preserves useful river progress.

This is important because the new theory should go beyond the current perturbation-only model in the Muon paper. 

---

# Experiment E4: Frequency-Domain Validation on Toy Landscapes

## Goal

Directly verify the frequency-response prediction.

## Procedure

For each trajectory, collect:

[
g_1,\dots,g_T,
\qquad
m_1,\dots,m_T.
]

Apply finite-window DFT to each component:

[
\widehat g(\omega_\ell),
\qquad
\widehat m(\omega_\ell).
]

Compare empirical ratio:

[
R_{\mathrm{emp}}(\omega_\ell)
=============================

\frac{|\widehat m(\omega_\ell)|}
{|\widehat g(\omega_\ell)|}
]

with theoretical magnitude response:

[
|H_\beta(\omega_\ell)|
======================

\frac{1-\beta}
{\sqrt{1-2\beta\cos\omega_\ell+\beta^2}}.
]

## Expected Result

For fixed gradient streams, the empirical ratio should roughly follow:

[
R_{\mathrm{emp}}(\omega)
\approx
|H_\beta(\omega)|.
]

Boundary effects may appear near the beginning/end of the window.

## Success Criteria

Successful if:

* low-frequency ratio is close to (1);
* high-frequency ratio is close to ((1-\beta)/(1+\beta));
* empirical attenuation increases with (\beta);
* the result holds for both straight and curved river-valley landscapes.

This experiment directly connects the river-valley demo to the frequency-domain paper. 

---

# Experiment E5: Matrix-Valued River-Valley / Muon-Style Toy Demo

## Goal

Connect the river-valley filtering story to Muon and nonlinear direction extraction.

## Model

Construct matrix gradients:

[
G_t
===

S_t
+
H_t
+
\Xi_t,
]

where:

* (S_t): slowly varying low-rank signal matrix;
* (H_t): high-frequency oscillatory matrix component;
* (\Xi_t): stochastic perturbation.

Example:

[
S_t = U \Lambda_t V^\top,
]

with slowly varying (\Lambda_t), and

[
H_t = (-1)^t A
]

or a sinusoidal high-frequency matrix mode.

## Compare Pipelines

Use the three pipelines from the Muon paper:

[
\text{Pre-polar: } O(M_t),
]

[
\text{Post-polar: } \widetilde M_t
==================================

\beta \widetilde M_{t-1}
+
(1-\beta)O(G_t),
]

[
\text{Polar-only: } O(G_t).
]

## Metrics

### Signal Alignment

[
\mathrm{Align}(A,S)
===================

\frac{\langle A,O(S)\rangle_F}
{\min(m,n)}.
]

### Subspace Alignment

Use principal angle error:

[
|\sin\Theta(U_A,U_S)|_2.
]

### Spectral Gap

Measure:

[
\sigma_r(M_t)-\sigma_{r+1}(M_t).
]

## Expected Result

Pre-polar should dominate:

[
\mathrm{Align}(O(M_t),S_t)

>

\mathrm{Align}(\widetilde M_t,S_t)
]

and

[
\mathrm{Align}(O(M_t),S_t)

>

\mathrm{Align}(O(G_t),S_t).
]

This would generalize the current Muon paper from stochastic perturbation filtering to temporal high-frequency filtering. 

## Success Criteria

Successful if Pre-polar improves when (H_t) is high-frequency, even when (H_t) is not zero-mean stochastic noise.

This is a key test of the new idea.

---

# Experiment E6: Real Neural-Network Gradient Frequency Diagnostics

## Goal

Check whether real training gradients contain the slow/fast structure predicted by the theory.

## Candidate Settings

Use small-to-medium experiments first:

* CIFAR-10 ResNet-18;
* CIFAR-100 ResNet-50;
* NanoGPT small training;
* Muon training if available.

## Procedure

At checkpoints, freeze the model and collect a gradient buffer:

[
G_1,\dots,G_K.
]

Also collect live trajectory buffers during training.

For selected layers, compute:

1. temporal DFT of gradient entries or low-rank projections;
2. temporal DFT of singular values;
3. temporal DFT of projections onto estimated signal subspace;
4. temporal DFT of residual components.

## Signal Proxy Options

Use one or more of:

### Option A: Mean Gradient Proxy

[
\bar G = \frac{1}{K}\sum_{t=1}^K G_t.
]

Slow signal proxy:

[
G_t^{\mathrm{slow}}\approx \bar G.
]

### Option B: Low-Pass Proxy

Use a very large moving average as the slow component.

### Option C: Hessian-Based Proxy

Approximate sharp/hill directions using top Hessian eigenvectors.

This is more expensive but more directly connected to river-valley geometry.

## Metrics

### High-Frequency Energy Ratio

[
\mathrm{HFER}(G)
================

\frac{
\sum_{\omega\in\Omega_{\mathrm{high}}}
|\widehat G(\omega)|*F^2
}{
\sum*{\omega}
|\widehat G(\omega)|_F^2
}.
]

### Momentum Suppression Ratio

[
\mathrm{MSR}(\beta)
===================

\frac{
\sum_{\omega\in\Omega_{\mathrm{high}}}
|\widehat M(\omega)|*F^2
}{
\sum*{\omega\in\Omega_{\mathrm{high}}}
|\widehat G(\omega)|_F^2
}.
]

### Signal Alignment Improvement

[
\Delta \mathrm{Align}
=====================

## \mathrm{Align}(M,\bar G)

\mathrm{Align}(G,\bar G).
]

## Expected Result

* raw gradients have nontrivial high-frequency energy;
* momentum suppresses high-frequency energy according to (|H_\beta(\omega)|);
* filtered gradients align better with the estimated slow signal;
* the effect is stronger in later training.

## Success Criteria

Successful if at least two real settings show:

* clear high-frequency suppression by momentum;
* improved alignment with slow/mean gradient;
* consistency across layers or checkpoints;
* qualitative agreement with toy river-valley results.

## Failure Criteria

This direction weakens if real gradients do not show meaningful temporal frequency structure, or if momentum does not improve slow-signal alignment.

---

# Experiment E7: Optimizer-Level Tests

## Goal

Check whether the filtering mechanism predicts actual optimizer behavior.

## Compare

For SGDM or Muon:

[
\beta\in{0,0.5,0.8,0.9,0.95,0.99}.
]

Also test schedules:

1. fixed (\beta);
2. increasing (\beta);
3. decreasing (\beta);
4. adaptive (\beta) based on high-frequency energy.

## Metrics

Use both performance and mechanism metrics:

* validation loss;
* training loss;
* high-frequency gradient energy;
* alignment with slow gradient;
* post-decay loss drop;
* for Muon: Pre-polar vs Post-polar alignment.

## Expected Result

Momentum values or schedules that best suppress harmful high-frequency components while avoiding excessive lag should perform best.

## Success Criteria

The idea is strongly supported if the mechanism metrics predict performance better than (\beta) alone.

For example, if the best validation loss corresponds to the best balance of:

[
\text{low hill/high-frequency energy}
\quad+\quad
\text{low lag}
\quad+\quad
\text{good river alignment}.
]

---

# 3. Theory Plan

The theory should be developed after E1–E4 confirm that the mechanism exists in controlled settings.

---

# Theory T1: Finite-Window Momentum Filtering Theorem

## Goal

Make the frequency-domain argument rigorous for finite training windows.

## Setup

Let (g_t) be a vector- or matrix-valued sequence in a Hilbert space.

Define EMA momentum:

[
m_t
===

(1-\beta)\sum_{s=0}^{t-1}\beta^s g_{t-s}.
]

On a window (t=1,\dots,T), use finite DFT:

[
\widehat g(\omega_\ell),
\qquad
\widehat m(\omega_\ell).
]

## Target Theorem

For high-frequency band (\Omega_{\mathrm{high}}):

[
\sum_{\omega_\ell\in\Omega_{\mathrm{high}}}
|\widehat m(\omega_\ell)|^2
\le
\rho_{\mathrm{high}}(\beta)^2
\sum_{\omega_\ell\in\Omega_{\mathrm{high}}}
|\widehat g(\omega_\ell)|^2
+
\mathrm{BoundaryError},
]

where:

[
\rho_{\mathrm{high}}(\beta)
===========================

\sup_{\omega\in\Omega_{\mathrm{high}}}
|H_\beta(\omega)|.
]

Similarly, for low frequencies:

[
\sum_{\omega_\ell\in\Omega_{\mathrm{low}}}
|\widehat m(\omega_\ell)-\widehat g(\omega_\ell)|^2
\le
\epsilon_{\mathrm{low}}(\beta)^2
\sum_{\omega_\ell\in\Omega_{\mathrm{low}}}
|\widehat g(\omega_\ell)|^2
+
\mathrm{BoundaryError}.
]

## Proof Tools

* Z-transform for intuition;
* finite DFT for rigorous proof;
* Parseval identity;
* convolution theorem;
* boundary/transient control.

## Success Criteria

This theorem is successful if it gives explicit constants depending on:

[
\beta,
\qquad
\Omega_{\mathrm{high}},
\qquad
T.
]

---

# Theory T2: Stochastic Noise Filtering Theorem

## Goal

Unify frequency filtering with stochastic perturbation filtering.

## Setup

Let:

[
g_t=s_t+h_t+\xi_t.
]

Assume:

[
\mathbb E[\xi_t]=0,
]

and weak temporal orthogonality or mixing.

## Target Bound

[
\mathbb E|m_t - \mathcal F_\beta(s+h)*t|^2
\le
\frac{\sigma^2}{N*{\mathrm{eff}}},
]

where:

[
N_{\mathrm{eff}}
================

\frac{1+\beta}{1-\beta}.
]

This connects to the effective sample size argument used in the Muon paper. 

## Success Criteria

Successful if the theorem combines:

[
\text{deterministic high-frequency attenuation}
+
\text{stochastic variance reduction}.
]

---

# Theory T3: River-Valley Generates High-Frequency Hill Gradients

## Goal

Show mathematically why the river-valley landscape naturally creates slow/fast gradient decomposition.

## Straight Valley Case

Use:

[
L(x,y)
======

\phi(x)+\frac{\lambda}{2}y^2.
]

For gradient descent in the hill direction:

[
y_{t+1}
=======

(1-\eta\lambda)y_t.
]

If:

[
1<\eta\lambda<2,
]

then:

[
1-\eta\lambda<0,
]

so (y_t) alternates sign and the hill gradient is high-frequency.

## Target Statement

The hill gradient has dominant temporal frequency near (\omega=\pi), while the river gradient varies slowly when (\mu\ll\lambda).

Then momentum attenuates hill gradients by approximately:

[
|H_\beta(\pi)|
==============

\frac{1-\beta}{1+\beta}.
]

## Curved Valley Case

Use:

[
L(x,y)
======

\phi(x)
+
\frac{\lambda}{2}(y-f(x))^2.
]

Need perturbation analysis showing that when the river curvature is not too large, the hill direction still produces higher-frequency oscillations than the river tangent direction.

## Success Criteria

Successful if we can prove a clean theorem for the straight valley and a controlled approximation for the curved valley.

---

# Theory T4: Momentum Improves River Alignment

## Goal

Turn filtering into an optimization-relevant statement.

## Desired Result

Assume:

[
g_t = g_t^{\mathcal R}+g_t^{\mathcal H},
]

where:

* (g_t^{\mathcal R}) is low-frequency;
* (g_t^{\mathcal H}) is high-frequency.

Then:

[
\frac{|m^{\mathcal H}|*{2,T}}
{|m^{\mathcal R}|*{2,T}}
\le
\kappa(\beta)
\frac{|g^{\mathcal H}|*{2,T}}
{|g^{\mathcal R}|*{2,T}},
]

with:

[
\kappa(\beta)<1.
]

This implies improved river alignment.

## Success Criteria

Successful if the theorem gives a condition like:

[
\rho_{\mathrm{high}}(\beta)
<
1-\epsilon_{\mathrm{low}}(\beta),
]

meaning high-frequency attenuation dominates low-frequency distortion.

---

# Theory T5: Filter-First Before Nonlinear Direction Maps

## Goal

Generalize the Muon result.

## Setup

Let (\mathcal D(\cdot)) be a nonlinear direction map, such as:

* normalization:
  [
  \mathcal D(g)=g/|g|;
  ]
* sign:
  [
  \mathcal D(g)=\mathrm{sign}(g);
  ]
* polar factor:
  [
  \mathcal D(G)=O(G).
  ]

Compare:

[
\text{Filter-first: }
\mathcal D(m_t)
]

against

[
\text{Nonlinearize-first: }
\tilde m_t
==========

\beta \tilde m_{t-1}
+
(1-\beta)\mathcal D(g_t).
]

## Target Claim

If momentum improves the signal-to-perturbation ratio before applying (\mathcal D), then:

[
\mathcal D(m_t)
]

has better alignment with the slow signal than nonlinearize-first baselines.

For Muon, this recovers and extends the current Pre-polar advantage. 

## Proof Tools

* Lipschitz or perturbation bound for (\mathcal D);
* Wedin theorem for polar factor;
* spectral gap bound;
* signal alignment comparison.

## Success Criteria

Successful if we can prove at least one strong version for the polar factor and one simpler version for normalized SGD.

---

# Theory T6: Momentum Scheduling from Filtering-Lag Tradeoff

## Goal

Explain why fixed large (\beta) may not always be best.

## Desired Bound

A general error decomposition:

[
|m_t-s_t|
\le
\underbrace{\mathrm{Lag}*\beta(s_t)}*{\text{slow signal bias}}
+
\underbrace{\mathrm{HighFreq}*\beta(h_t)}*{\text{fast component leakage}}
+
\underbrace{\mathrm{Noise}*\beta(\xi_t)}*{\text{stochastic variance}}.
]

Expected scaling:

[
\mathrm{Lag}_\beta(s_t)
\uparrow
\quad \text{as } \beta\uparrow,
]

[
\mathrm{HighFreq}_\beta(h_t)
\downarrow
\quad \text{as } \beta\uparrow,
]

[
\mathrm{Noise}_\beta(\xi_t)
\downarrow
\quad \text{as } \beta\uparrow.
]

## Success Criteria

Successful if this theorem naturally motivates:

* increasing (\beta) schedules;
* adaptive (\beta) based on frequency diagnostics;
* avoiding excessively large (\beta) when the signal changes quickly.

---

# 4. Step-by-Step Execution Plan for LLM Agents

## Phase 1: Build Controlled Toy Experiments

### Task 1.1: Implement Straight Valley

**Input:** loss (L(x,y)=\frac{\mu}{2}(x-x^\star)^2+\frac{\lambda}{2}y^2)

**Steps:**

1. Set (\mu=0.1), (\lambda=10), (x^\star=5).
2. Choose starting point, e.g. ((x_0,y_0)=(-3,1)).
3. Choose learning rate (\eta) such that (1<\eta\lambda<2).
4. Run (\beta\in{0,0.5,0.9,0.95,0.99}).
5. Save trajectories, gradients, momentum buffers.

**Output:**

* `trajectory_beta_*.csv`
* contour plots;
* hill gradient time-series;
* FFT plots;
* metric table.

**Decision Gate:** continue only if hill oscillation is visible for (\beta=0).

---

### Task 1.2: Implement Curved Valley

**Input:**

[
L(x,y)=\frac{\mu}{2}(x-x^\star)^2+\frac{\lambda}{2}(y-a\sin(kx))^2.
]

**Steps:**

1. Use same (\beta) grid.
2. Compute river tangent (r(x)) and hill normal (n(x)).
3. Track river alignment, hill energy, and lag.

**Output:**

* curved trajectory plots;
* river alignment table;
* lag table.

**Decision Gate:** continue if moderate (\beta) improves alignment but (\beta=0.99) shows lag or diminished progress.

---

### Task 1.3: Add Noise

**Steps:**

1. Add Gaussian noise.
2. Add anisotropic hill noise.
3. Add heavy-tailed finite-variance noise.
4. Repeat E1/E2 over at least 10 seeds.

**Output:**

* mean ± std metric tables;
* plots with confidence bands.

**Decision Gate:** continue if filtering effect survives noise.

---

## Phase 2: Frequency-Domain Verification

### Task 2.1: Empirical Transfer Function

**Steps:**

1. Collect (g_t,m_t).
2. Compute DFT.
3. Plot empirical ratio:
   [
   |\widehat m(\omega)|/|\widehat g(\omega)|.
   ]
4. Overlay theoretical (|H_\beta(\omega)|).

**Output:**

* frequency response validation plots.

**Decision Gate:** continue if empirical response roughly follows theoretical response.

---

## Phase 3: Matrix/Muon Toy Experiment

### Task 3.1: Matrix Gradient Generator

**Steps:**

1. Generate slow low-rank signal (S_t).
2. Generate fast oscillatory matrix component (H_t).
3. Add stochastic perturbation (\Xi_t).
4. Construct (G_t=S_t+H_t+\Xi_t).

### Task 3.2: Compare Pipelines

Run:

* Pre-polar;
* Post-polar;
* Polar-only.

**Output:**

* signal alignment curves;
* subspace alignment curves;
* spectral gap curves.

**Decision Gate:** continue if Pre-polar dominates when the disturbance is high-frequency, not only stochastic.

---

## Phase 4: Real Neural-Network Diagnostics

### Task 4.1: Stationary Gradient Buffers

**Steps:**

1. Train small ResNet or NanoGPT.
2. Save checkpoints.
3. Freeze model.
4. Collect (K) mini-batch gradients for selected layers.
5. Compute temporal spectrum and momentum-filtered spectrum.

**Output:**

* high-frequency energy ratio;
* momentum suppression ratio;
* slow-gradient alignment.

### Task 4.2: Live Trajectory Buffers

**Steps:**

1. During training, store sliding gradient buffers.
2. Repeat frequency and alignment diagnostics.
3. Compare early/mid/late training.

**Decision Gate:** continue if real gradients show stable slow/fast structure.

---

## Phase 5: Theorem Development

### Task 5.1: Prove Pure Filtering Lemma

Use finite DFT and Parseval.

**Output:** theorem with explicit (\rho_{\mathrm{high}}(\beta)).

### Task 5.2: Prove River-Valley Frequency Separation

Start with straight valley, then curved valley.

**Output:** theorem showing hill gradients are higher-frequency than river gradients under large LR.

### Task 5.3: Prove River Alignment Improvement

Combine Task 5.1 and 5.2.

**Output:** theorem proving improved river-to-hill ratio after momentum.

### Task 5.4: Prove Filter-First Nonlinear Direction Result

Start with normalized SGD, then polar/Muon.

**Output:** theorem generalizing “denoise first, orthogonalize later” to “filter first, nonlinearize later.”

---

# 5. Overall Criteria of Success

## Minimal Success

The project is minimally successful if:

1. toy river-valley experiments show clear hill oscillation filtering;
2. frequency-domain plots match the EMA magnitude response;
3. a finite-window filtering theorem is proven;
4. the result gives a convincing bridge from the old frequency-domain paper to the Muon paper.

This would already be a good workshop or early-stage theory paper.

## Strong Success

The project is strongly successful if:

1. real neural-network gradients show slow/fast temporal decomposition;
2. momentum improves alignment with slow gradient components;
3. Muon Pre-polar advantage extends to deterministic high-frequency perturbations;
4. the theory proves river alignment improvement in a river-valley model;
5. filtering-lag tradeoff explains why momentum scheduling helps.

This could become a full conference paper.

## Very Strong Success

The project is very strong if:

1. the theory covers vector optimizers and matrix optimizers;
2. the “filter first, nonlinearize later” principle applies beyond Muon;
3. the diagnostics predict actual optimizer performance;
4. a new momentum schedule or adaptive (\beta) rule follows naturally and improves training.

This could become a broad theoretical and empirical paper on momentum filtering.

---

# 6. Failure Criteria and How to Interpret Failure

## Failure Mode 1: Toy River-Valley Demo Fails

If momentum does not suppress hill oscillations in the toy model, the core idea is in trouble.

Possible causes:

* learning rate not in oscillatory regime;
* valley curvature too small;
* metrics are wrong;
* momentum update convention is mismatched.

Action: debug toy setup first before abandoning the idea.

## Failure Mode 2: Toy Demo Works but Real Gradients Do Not

This means river-valley filtering may be a valid mechanism but not dominant in practical networks.

Action: narrow the paper to a theoretical mechanism paper, or search for architectures/layers/training phases where the mechanism appears.

## Failure Mode 3: Momentum Filters High Frequency but Does Not Improve Optimization

This means high-frequency components are not necessarily harmful.

Action: refine the theory. The claim should become conditional:

[
\text{momentum helps only when high-frequency components are less aligned with useful progress}.
]

## Failure Mode 4: Very Large (\beta) Always Wins

This weakens the filtering-lag tradeoff story.

Action: test curved rivers, rapidly changing signals, and early training. If very large (\beta) still always wins, the theory may focus more on denoising/filtering than scheduling.

## Failure Mode 5: No Clean Theorem for Curved Valley

This is acceptable.

Action: keep the rigorous theorem for straight/local valley and present curved valley as empirical evidence or local approximation.

---

# 7. Suggested Paper Structure

## Title Candidates

1. **Momentum Filters the River: A Temporal-Spectral Theory of Momentum in Optimization**
2. **Momentum as Gradient Stream Filtering: From River-Valley Dynamics to Matrix Optimizers**
3. **Filter First, Then Update: A Temporal Frequency View of Momentum**

## Paper Outline

### Section 1: Introduction

* Momentum is usually viewed as acceleration or stochastic denoising.
* We argue it is more generally a temporal filter.
* River-valley landscapes provide a geometric source of high-frequency gradient components.
* Muon shows why filtering before nonlinear direction extraction matters.

### Section 2: Frequency-Domain Momentum Filtering

* Review EMA/SGDM as a filter.
* Define low/high temporal gradient components.
* Prove finite-window filtering theorem.

### Section 3: River-Valley Dynamics

* Define straight and curved river-valley landscapes.
* Show hill gradients can be high-frequency.
* Prove momentum improves river alignment under frequency separation.

### Section 4: Filter-First Principle for Nonlinear Optimizers

* General direction maps.
* Normalized SGD case.
* Muon/polar case.
* Connect to Pre-polar vs Post-polar.

### Section 5: Controlled Experiments

* Straight river demo.
* Curved river demo.
* Noisy river demo.
* Matrix/Muon toy demo.

### Section 6: Real Training Diagnostics

* CIFAR/NanoGPT gradient spectra.
* Momentum suppression.
* Signal alignment.
* Muon ordering.

### Section 7: Discussion

* Filtering-lag tradeoff.
* Momentum scheduling.
* Limitations.
* Future adaptive momentum design.

---

# 8. Final Research Claim We Want to Support

The final paper should aim to justify the following claim:

> Momentum improves deep learning optimization not only by accelerating deterministic dynamics or averaging stochastic noise, but by acting as a temporal spectral filter on the gradient stream. In river-valley landscapes, useful river-direction gradients vary slowly, while hill-direction gradients often oscillate rapidly. Momentum suppresses these high-frequency hill components and improves alignment with the river direction. When a nonlinear direction map such as normalization, sign, or polar orthogonalization is applied, filtering before the nonlinear map produces a cleaner and more reliable update.

That is the coherent story connecting:

[
\text{frequency-domain momentum}
\rightarrow
\text{river-valley landscape}
\rightarrow
\text{momentum filtering theory}
\rightarrow
\text{Muon filter-first principle}.
]
