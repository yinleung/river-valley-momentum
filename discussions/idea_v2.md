# Future instruction: improve `Momentum Filters the River` for top-AI-conference submission

## 0. Core revision goal

Revise the paper so that the main claim is not:

[
\text{larger } \beta \Rightarrow \text{better optimization}
]

but rather:

[
\text{larger } \beta \Rightarrow \text{stronger high-frequency hill filtering},
]

and this improves river-valley optimization **only in the predicted regime**: noisy or sustained hill excitation, large learning rate, sufficient valley conditioning, and not-too-large memory relative to river curvature/traversal time.

The paper already states the filtering-lag tradeoff: (\rho_{\rm high}(\beta)) decreases while (\epsilon_{\rm low}(\beta)) increases, so maximal (\beta) is not always optimal. The revision should make this the central empirical message, not a side remark. 

---

## 1. Add a direct “Does larger (\beta) improve river-valley optimization?” experiment

### 1.1 Add a new main experiment: `E_beta_improves_river_valley`

Create a new controlled experiment whose only purpose is to answer:

> In the river-valley regime where the theory predicts benefit, does increasing (\beta) improve actual optimization metrics?

Use the straight noisy river valley first, because it removes curved-river lag and directly matches Proposition 3.

### Experimental setup

Use

[
L(x,y)=\frac{\mu}{2}(x-x^\star)^2+\frac{\lambda}{2}y^2,
]

with noisy hill gradients:

[
g_t=\nabla L(w_t)+\xi_t,\qquad \xi_t\sim \mathcal{N}(0,\sigma^2 I).
]

Sweep:

```text
beta_grid = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995]
eta_lambda_grid = [0.6, 1.2, 1.6, 1.8, 2.2, 2.5]
condition_grid = [10, 100, 1000]  # lambda / mu
noise_grid = [0.5, 1.0, 2.0]
seeds = 32
T = long enough for stationarity, e.g. 5000 or 10000
burn_in = first 30% steps
```

For each cell, report:

```text
tail_loss
stationary_hill_variance
rms_distance_to_river_floor
tube_escape_frequency
river_progress
river_alignment
high_frequency_energy_ratio
momentum_suppression_ratio
predicted_CL_variance
predicted_stability_indicator: eta * lambda < 2 * T_eff
```

### Expected result

In the **straight noisy valley**, for large (\eta\lambda), increasing (\beta) should monotonically or near-monotonically reduce stationary hill variance and tail loss until finite-horizon warmup effects appear. The theory predicts:

[
\operatorname{Var}(y_\infty)=
\frac{\eta\sigma^2}
{\lambda(2-\eta\lambda/T_{\rm eff})}.
]

This is already in the paper, but we need a clear optimization-quality plot, not only a mechanism plot. 

### Required figures

Add a new Figure, preferably early in the experiments section:

**Figure: Larger (\beta) improves river-valley optimization in the predicted noisy large-LR regime**

Panels:

1. (\beta) vs tail loss.
2. (\beta) vs stationary hill variance.
3. (\beta) vs tube escape frequency.
4. measured hill variance / predicted Proposition 3 variance.
5. heatmap of best (\beta) over ((\eta\lambda,\lambda/\mu)).
6. stability map: observed divergence vs predicted condition (\eta\lambda < 2T_{\rm eff}).

### Success criteria

The experiment succeeds if:

```text
1. For eta*lambda near or above 2, beta=0 is unstable or high-variance.
2. Moderate/high beta stabilizes the trajectory and reduces tail loss.
3. Measured stationary hill variance matches Proposition 3 within a reasonable tolerance.
4. Observed divergence boundary matches Proposition 2.
5. In small eta*lambda regimes, beta provides little improvement, consistent with the theory.
```

Do not claim global monotonicity. The correct claim is:

```text
Increasing beta improves optimization in the noisy large-learning-rate straight-valley regime,
but the improvement is regime-scoped.
```

---

## 2. Add the complementary curved-valley “too-large (\beta) hurts” experiment

The current curved-valley experiment already suggests that (\beta=0.9) is good but (\beta=0.99) lags the river. Make this clearer and more systematic.

### Add experiment: `E_beta_window_curved_valley`

Use

[
L(x,y)=\phi(x)+\frac{\lambda}{2}(y-f(x))^2,
\qquad
f(x)=a\sin(kx).
]

Sweep:

```text
beta_grid = dense grid from 0 to 0.999
bend_frequency k = [0.3, 0.6, 0.9, 1.2, 1.8]
condition_number lambda/mu = [10, 100, 1000]
eta_lambda = [1.6, 1.8, 2.2, 2.5]
seeds = 16 or 32
```

Metrics:

```text
tail_loss
river_progress
hill_normal_energy
river_following_lag = 1 - AlignR(m)
tube_escape_frequency
best_beta
predicted_beta_window
```

### Expected result

For curved valleys, the curve should be U-shaped:

[
\beta \text{ too small} \Rightarrow \text{insufficient hill filtering},
]

[
\beta \text{ moderate/high} \Rightarrow \text{best river tracking},
]

[
\beta \text{ too large} \Rightarrow \text{river lag}.
]

This directly supports the paper’s “filtering-lag tradeoff.” The current paper already says the straight-valley closed-loop floor distance is best at (\beta=0.5), while for (\beta\ge 0.9) the underdamped mode rings longer. It also shows curved-valley river alignment peaking near (\beta=0.9), then degrading at (\beta=0.99). These need to become a single coherent beta-window experiment rather than scattered observations.  

---

## 3. Add a theory-to-experiment prediction table

Add a table called:

```text
Table: Theory predictions and experimental validations
```

Rows:

```text
Theorem 1: EMA high-frequency attenuation
Prediction: MSR tracks |H_beta(pi)|^2
Experiment: E4 frequency validation

Theorem 2: straight valley emits Nyquist hill and DC river
Prediction: hill peak at pi, river peak at 0
Experiment: E1

Proposition 2: stability iff eta*lambda < 2*T_eff
Prediction: divergence boundary shifts with beta
Experiment: new E_beta_improves_river_valley + E8

Proposition 3: stationary hill variance formula
Prediction: measured variance matches formula
Experiment: new straight noisy valley sweep

Remark 1 / Remark 3: beta window
Prediction: optimal beta increases with conditioning but decreases with river curvature
Experiment: new curved valley sweep

Theorem 4: filter-first before polar map
Prediction: pre-polar has lower subspace error and better update alignment
Experiment: E5 + expanded Muon experiment
```

This will help reviewers see that each theorem has a corresponding test.

---

## 4. Strengthen the Muon-specific experiments

The current Muon evidence is too small. The paper has a matrix toy and a small MLP first-layer Muon-style test, but for a top AI conference, reviewers will ask whether this matters for real optimizers. The paper itself says Theorem 4 is about the buffer feeding the direction map, not a nonlinear trajectory theorem, and that the real-network evidence is only one small full-batch network. 

### Add experiment: closed-loop Muon variants

Compare:

```text
Muon-pre-polar: O(EMA(G_t))
Muon-post-polar: EMA(O(G_t))
Muon-no-momentum: O(G_t)
SGDM / AdamW baseline if appropriate
```

Run on:

```text
1. controlled matrix sensing river-valley task
2. small MLP edge-of-stability task
3. CIFAR-10 small CNN or ResNet
4. NanoGPT small-scale language modeling task, if compute permits
```

For every run, log:

```text
training loss
validation loss
gradient-stream HFER
update-stream HFER
slow-gradient/update alignment
pre-polar vs post-polar update alignment
singular subspace stability
spectral gap of raw gradient matrix and momentum matrix
```

### Required result

The paper should not only show that pre-polar wins in a toy matrix model. It should show that:

```text
When the gradient stream is high-frequency dominated, pre-polar Muon beats post-polar Muon.
When the stream is low-frequency dominated, the gap shrinks.
```

That would directly validate the mechanism.

---

## 5. Add large-scale or at least realistic mini-batch evidence

The current real-network experiment is full-batch and small. Add at least one mini-batch setting.

Recommended options:

```text
Option A: CIFAR-10 / ResNet-18 or small ConvNet
Option B: NanoGPT small model on a character or OpenWebText subset
Option C: modded-nanogpt optimization track if feasible
```

For each task, sweep:

```text
beta = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
learning_rate = at least 3 values around edge-of-stability
seeds = at least 3, preferably 5
```

Keep all else fixed.

Report:

```text
training loss
validation loss
tokens/sec or wall-clock if relevant
gradient HFER
mechanism score
rank correlation between mechanism score and final loss
```

Important: separate optimization from generalization. The current paper notes that the lowest training loss can correspond to worse held-out loss. Keep this honest and state:

```text
The theory predicts optimization dynamics, not necessarily generalization.
```

---

## 6. Add beta-schedule experiments

The paper’s discussion mentions floor-respecting warmup and confinement onset. Turn this into a real experiment.

Compare:

```text
fixed beta: 0, 0.5, 0.8, 0.9, 0.95, 0.99
linear beta ramp: 0 -> 0.9
floor-respecting ramp: 0.7 -> 0.95
late beta increase after detected confinement
adaptive beta based on HFER or tube metric
```

For adaptive beta, implement:

```text
if HFER > threshold and river_alignment_raw is low:
    increase beta
else:
    keep beta moderate
```

or simpler:

```text
beta_t = beta_low before confinement
beta_t = beta_high after confinement
```

Use both toy valleys and one real network diagnostic.

Expected claim:

```text
A beta schedule should not be clock-based only; it should respect confinement / high-frequency onset.
```

---

## 7. Improve the theory: prove actual optimization improvement, not only filtering improvement

The paper’s theory currently proves strong filtering facts and some closed-loop variance facts. But reviewers may ask:

> Where exactly do you prove that larger (\beta) improves optimization?

Add a proposition like:

```text
Proposition: In the straight noisy river valley, for fixed eta*lambda in the stable regime,
the stationary expected hill loss is monotonically decreasing in beta under EMA normalization.
```

This follows from Proposition 3:

[
\mathbb{E}\left[\frac{\lambda}{2}y_\infty^2\right]
==================================================

\frac{\eta\sigma^2}{2(2-\eta\lambda/T_{\rm eff})}.
]

Then explicitly state the conditions:

```text
1. straight valley
2. sustained white gradient noise
3. fixed EMA-normalized step convention
4. sufficiently long horizon / stationarity
5. river drift preserved to first order
```

Also add a finite-time version or at least a burn-in discussion, because very large (\beta) has startup lag.

---

## 8. Generalize Theorem 4 beyond exactly ((-1)^tA)

Theorem 4 is good but too special. Generalize it to:

[
G_t=S_t+A_t,
]

where (S_t) is low-frequency / slowly varying and (A_t) is high-frequency band-limited.

Target theorem:

```text
If S_t has most energy below omega_0 and A_t has most energy above omega_c,
then EMA improves the signal-to-disturbance ratio before the polar map by roughly
(1 - epsilon_low(beta)) / rho_high(beta),
up to boundary and burn-in terms.
```

Then apply Wedin to show subspace stability of the filtered matrix.

This would connect Theorem 1 and Theorem 4 directly. Right now Theorem 4 is an elegant canonical Nyquist case, but the general band-limited theorem would be much more convincing.

---

## 9. Handle the finite-window boundary term more carefully

The finite-window DFT identity includes the boundary term:

[
B=\frac{\beta}{1-\beta}m_T.
]

The paper says this is small once the filter has settled, but that is not always true, especially for persistent low-frequency signals. Add one of the following:

```text
Option A: prove a boundary-normalized theorem where the boundary term divided by window energy vanishes under explicit assumptions.
Option B: use burn-in and analyze only the settled window.
Option C: use tapered windows and report that boundary artifacts are controlled.
Option D: present both exact finite-window theorem and infinite-stationary transfer theorem.
```

In experiments, always state:

```text
window type
burn-in
whether DFT is rectangular or Hann
how boundary effects are handled
```

---

## 10. Clarify EMA normalization vs heavy-ball normalization

The paper’s conclusions depend on the convention:

```text
EMA update: w_{t+1} = w_t - eta * m_t, m_t = beta m_{t-1} + (1-beta) g_t
```

versus heavy-ball-style conventions where the effective step changes with (\beta).

Add a short subsection:

```text
Normalization conventions and what is held fixed
```

Include:

```text
1. fixed eta under EMA normalization
2. fixed eta_HB under heavy-ball normalization
3. scale-invariant Muon/polar updates
```

Then add a small ablation:

```text
Compare beta sweeps under both normalizations.
Show which claims are invariant and which are convention-dependent.
```

This will prevent reviewer confusion.

---

## 11. Improve experimental statistics

For every nontrivial experiment:

```text
use at least 8 seeds for toy experiments
use at least 3 seeds for real ML experiments
report mean ± std or confidence intervals
report exact beta/lr grids
report diverged runs explicitly
avoid hiding failed cells
```

For figures:

```text
plot individual seed traces faintly
plot aggregate mean boldly
include predicted theory curves when available
```

For tables:

```text
include best beta
include predicted beta window
include observed beta window
include Spearman correlation between mechanism score and final loss
```

---

## 12. Improve the paper’s structure

Suggested revised structure:

```text
1. Introduction
   - Motivation
   - One-sentence claim: momentum filters temporal gradient frequencies
   - Main result: beta window, not monotonic beta

2. EMA as a temporal filter
   - Transfer function
   - finite-window theorem
   - boundary caveat

3. River-valley emits high-frequency hill gradients
   - deterministic separation
   - stochastic stationary spectrum
   - confinement/DC lemma

4. Closed-loop optimization consequences
   - stability threshold
   - stationary hill variance
   - actual optimization-improvement proposition

5. Filter-first nonlinear maps and Muon
   - canonical Nyquist theorem
   - general band-limited theorem if added
   - relation to polar/Muon

6. Experiments
   6.1 Does larger beta improve optimization? straight noisy valley
   6.2 Why not beta -> 1? curved-valley beta window
   6.3 frequency-domain validation
   6.4 Muon/filter-first validation
   6.5 real-network / mini-batch validation

7. Discussion and limitations
```

---

## 13. Rewrite the main claim carefully

Replace broad claims like:

```text
Momentum improves river-valley optimization by filtering the hills.
```

with:

```text
Momentum improves river-valley optimization when the hill component is sustained and high-frequency, while the river component is slow enough to remain inside the passband. The benefit is therefore regime-scoped and produces a beta window rather than a monotonic preference for beta -> 1.
```

This is much safer and more aligned with the current theory.

---

## 14. Minimal acceptance-oriented checklist

Before submission, ensure the paper has:

```text
[ ] One direct figure showing larger beta improves optimization in the predicted straight noisy valley regime.
[ ] One direct figure showing too-large beta hurts in curved valleys because of lag.
[ ] A theory-to-experiment table mapping each theorem to one experiment.
[ ] A new proposition translating Proposition 3 into expected hill-loss improvement.
[ ] A stronger Theorem 4 generalization or a clear statement that it is only a canonical Nyquist case.
[ ] Larger or mini-batch ML evidence beyond one full-batch MLP.
[ ] Muon pre-polar vs post-polar closed-loop experiments on at least one realistic task.
[ ] Seed variance / confidence intervals for all key plots.
[ ] Clear treatment of boundary terms and burn-in.
[ ] Clear distinction between optimization gain and generalization gain.
[ ] Clear normalization convention for momentum.
```

## My priority ranking

The most important improvements are:

1. **Add the direct larger-(\beta) river-valley optimization experiment.**
2. **Add the curved-valley beta-window experiment.**
3. **Add realistic mini-batch / Muon evidence.**
4. **Prove or explicitly state an optimization-improvement proposition from Proposition 3.**
5. **Clarify Theorem 4’s scope and, ideally, generalize it from exact Nyquist disturbance to band-limited disturbance.**

These changes would make the paper much more convincing because reviewers would see not only that momentum filters the gradient stream, but also **when that filtering actually improves optimization and when it fails**.
