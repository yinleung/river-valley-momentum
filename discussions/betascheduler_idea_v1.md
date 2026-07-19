# idea_v1: Momentum-Coefficient Scheduling for Muon

## 0. One-line summary

We study **Muon with momentum** under a fixed NanoGPT training setup and fixed learning-rate schedule, changing only the **Muon momentum coefficient scheduler**. The goal is to test whether a time-varying or feedback-controlled momentum coefficient can improve optimization efficiency over the standard fixed momentum coefficient.

---

## 1. Scope and non-goals

### Scope

We only consider Muon-style hidden-matrix optimization with momentum. The update of interest is the Muon momentum pipeline:

```math
M_t = \beta_t M_{t-1} + (1-\beta_t)G_t,
```

followed by Muon’s orthogonalization / Newton-Schulz polar-style update.

In practical Muon implementations with Nesterov-style momentum, the update may instead use a combination such as

```math
U_t = \operatorname{NS}\big((1-\nu_t)G_t + \nu_t M_t\big)
```

or an equivalent `grad.lerp(momentum, mu)` form. In this project, we should keep the exact original Muon update form used by the benchmark script and only replace the scalar momentum coefficient `mu` by a scheduler.

### Non-goals

We do **not** modify:

- the learning-rate scheduler;
- the model architecture;
- the dataset;
- the batch size;
- the number of forward-backward passes per step;
- the Muon orthogonalization method / Newton-Schulz iteration count;
- auxiliary AdamW settings, unless the benchmark baseline already uses them and they remain fixed;
- any outer Schedule-Free optimizer sequence \(x_t,z_t,y_t\).

The central variable is only:

```math
\beta_t \quad \text{or equivalently} \quad T_t^{\mathrm{mom}} = \frac{1}{1-\beta_t}.
```

---

## 2. Motivation

The frequency-domain momentum view interprets momentum as a filter over the gradient stream. A larger momentum coefficient corresponds to a longer effective memory and stronger low-pass filtering. Prior frequency-domain analysis suggests that the momentum coefficient should not necessarily be fixed: early training may benefit from preserving more raw-gradient/high-frequency behavior, while late training may benefit from stronger filtering.

The Schedule-Free papers are useful here not because we want to use Schedule-Free as the optimizer, but because they emphasize an **anytime averaging principle**: avoid relying on a fixed final training horizon when possible. The direct Schedule-Free optimizer introduces model-iterate averaging and evaluates gradients at an interpolated point. We will not use that mechanism. Instead, we borrow only the idea that an averaging coefficient can evolve without requiring the final stopping time.

The project question is:

> Can we design a better Muon momentum coefficient scheduler than a fixed \(\beta\), while keeping all other training details fixed?

We compare four families:

1. static beta baselines;
2. naive horizon-dependent beta pre-schedules;
3. Schedule-Free-inspired horizon-free beta schedules;
4. negative-feedback adaptive beta schedules.

---

## 3. Core experimental comparison

Let

```math
T_t^{\mathrm{mom}} = \frac{1}{1-\beta_t}.
```

We should report and tune schedules in terms of \(T_t^{\mathrm{mom}}\), not only \(\beta_t\), because \(\beta\)-space is highly nonlinear. For example:

```math
\beta=0.90 \Rightarrow T=10,
\quad
\beta=0.95 \Rightarrow T=20,
\quad
\beta=0.98 \Rightarrow T=50,
\quad
\beta=0.99 \Rightarrow T=100.
```

A small visual change in \(\beta\) near 1 can mean a large change in memory length.

### 3.1 Static beta baselines

Fixed momentum coefficient:

```math
\beta_t = \beta.
```

Candidate grid:

```math
\beta \in \{0.85, 0.90, 0.95, 0.97, 0.98, 0.99\}.
```

The most important baseline is the official benchmark/default Muon momentum, likely around \(\beta=0.95\), depending on the pinned script.

Purpose:

- determine the best fixed beta under the selected benchmark setup;
- test whether scheduling actually beats a well-tuned constant beta.

### 3.2 Naive beta pre-scheduling

These schedules use the known training horizon \(S\) or total number of training steps. They are not schedule-free, but they are important upper/lower comparison points.

#### Option A: linear beta schedule

```math
\beta_t = \beta_{\min} + (\beta_{\max}-\beta_{\min})\frac{t}{S}.
```

This is easy but probably not ideal because beta is nonlinear.

#### Option B: log-linear schedule in \(1-\beta\)

```math
1-\beta_t
= (1-\beta_{\min})^{1-t/S}(1-\beta_{\max})^{t/S}.
```

Equivalently,

```math
\log(1-\beta_t)
= \left(1-\frac{t}{S}\right)\log(1-\beta_{\min})
+ \frac{t}{S}\log(1-\beta_{\max}).
```

This is closer to scheduling the memory scale multiplicatively.

#### Option C: linear effective-window schedule

```math
T_t^{\mathrm{mom}}
= T_{\min} + (T_{\max}-T_{\min})\frac{t}{S},
\qquad
\beta_t = 1 - \frac{1}{T_t^{\mathrm{mom}}}.
```

Recommended naive pre-schedule baseline:

```math
T_{\min}\in\{5,10\},
\quad
T_{\max}\in\{50,100\}.
```

Purpose:

- test the simple heuristic “small beta early, large beta late”;
- estimate how much benefit is possible if the final horizon is known.

### 3.3 Schedule-Free-inspired beta scheduler

We do **not** use Schedule-Free optimizer. We only borrow the averaging-coefficient idea.

Schedule-Free averaging uses coefficients such as

```math
c_{t+1}=\frac{1}{t+1}
```

or more generally front-weighted variants. If we transfer only this idea to momentum,

```math
M_t = (1-c_t)M_{t-1}+c_tG_t,
```

then

```math
\beta_t = 1-c_t.
```

The raw translation would be:

```math
\beta_t = \frac{t}{t+1},
\qquad
T_t^{\mathrm{mom}} = t+1.
```

This is likely too aggressive for Muon because the momentum window grows without bound and can become stale. So the practical Schedule-Free-inspired version should be capped or softened.

#### Recommended form: capped anytime memory growth

```math
T_t^{\mathrm{mom}}
= \min\left(T_{\max},\; T_{\min}+\frac{t}{\tau}\right),
\qquad
\beta_t = 1 - \frac{1}{T_t^{\mathrm{mom}}}.
```

Alternative smooth version:

```math
T_t^{\mathrm{mom}}
= T_{\min} + (T_{\max}-T_{\min})\frac{t}{t+\tau},
\qquad
\beta_t = 1 - \frac{1}{T_t^{\mathrm{mom}}}.
```

Suggested grid:

```math
T_{\min}\in\{5,10\},
\quad
T_{\max}\in\{50,100\},
\quad
\tau\in\{100,250,500,1000\}\;\text{steps}.
```

Purpose:

- horizon-free deterministic beta scheduling;
- test whether a Schedule-Free-style anytime memory rule works for Muon momentum;
- compare directly with horizon-dependent pre-scheduling.

### 3.4 Negative-feedback beta scheduler

This family is adaptive rather than deterministic. The scheduler reacts to cheap statistics of the current gradient and momentum buffer.

The main controlled variable should be the effective memory window:

```math
T_t^{\mathrm{mom}} = \frac{1}{1-\beta_t}.
```

#### Cheap alignment controller

Compute gradient-momentum alignment:

```math
a_t =
\frac{\langle G_t,M_{t-1}\rangle_F}
{\|G_t\|_F\|M_{t-1}\|_F+\epsilon}.
```

Update log-window:

```math
\log T_{t+1}^{\mathrm{mom}}
=
\operatorname{clip}
\left(
\log T_t^{\mathrm{mom}} + \rho(a_t-a_\star),
\log T_{\min},
\log T_{\max}
\right),
```

then

```math
\beta_{t+1}=1-\frac{1}{T_{t+1}^{\mathrm{mom}}}.
```

Interpretation:

- if \(a_t>a_\star\), the current gradient agrees with momentum, so a longer memory is safe;
- if \(a_t<a_\star\), the momentum may be stale or unstable, so reduce memory.

Suggested hyperparameters:

```math
T_{\min}\in\{5,10\},
\quad
T_{\max}\in\{50,100\},
\quad
\rho\in\{0.005,0.01,0.02\},
\quad
 a_\star\in\{0.1,0.2,0.3\}.
```

#### Optional noise/staleness controller

Maintain cheap EMAs of scalar statistics:

```math
v_t = \operatorname{EMA}(\|G_t\|_F^2),
```

```math
n_t = \frac{\max(v_t-\|M_t\|_F^2,0)}{v_t+\epsilon}.
```

Then update:

```math
\log T_{t+1}^{\mathrm{mom}}
=
\operatorname{clip}
\left(
\log T_t^{\mathrm{mom}}
+\rho_n(n_t-n_\star)
-\rho_a(a_\star-a_t)_+,
\log T_{\min},
\log T_{\max}
\right).
```

This says: increase filtering when raw gradients look noisy, but decrease filtering when the current gradient and momentum disagree.

Purpose:

- horizon-free and adaptive;
- potentially better than deterministic schedules because it can reduce beta when momentum becomes stale;
- computational overhead is cheap: Frobenius dot products and norms over existing gradients/momentum buffers.

---

## 4. Benchmark choice: Modded-NanoGPT Track 3 Optimization

We propose to evaluate on the **Modded-NanoGPT Optimization Benchmark**, specifically:

```text
https:// qw.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization
```

This track is suitable because its stated goal is to find efficient optimizers by minimizing the number of training steps needed to reach the target validation loss, not minimizing wall-clock time. This matches our project: beta scheduling is an optimizer-level change.

### 4.1 Why this benchmark fits

- It already uses Muon-style optimization in the benchmark history.
- It targets NanoGPT / GPT-2-style language-model training, which is a meaningful testbed for Muon.
- It evaluates optimization efficiency by **steps to reach validation loss 3.28**, which is directly relevant to optimizer design.
- The rules explicitly allow modifying the optimization algorithm and hyperparameters/schedules, while fixing dataset, batch size, architecture, and one forward-backward pass per step.

### 4.2 Rules we must obey

To keep results valid under the track, we must obey the following.

1. **Do not modify the dataset.** Use the benchmark-provided FineWeb data preparation and train/validation split.
2. **Do not modify the batch size.** Keep the official batch size in the selected benchmark script.
3. **Do not modify the architecture.** Keep the official NanoGPT architecture exactly fixed.
4. **Do not use more than one forward-backward pass per step.** Our beta scheduler must run after the same gradient computation and must not require additional gradients.
5. **Target validation loss:** runs must attain below 3.28 validation loss.
6. **Statistical significance:** use the track’s condition

```math
(3.28-\mu)\sqrt{n} \ge 0.004,
```

where \(\mu\) is the mean validation loss over \(n\) non-cherry-picked runs.

7. **Reproducibility:** all code needed to reproduce a run should be included in the logfile. Do not import third-party optimizer libraries.
8. **No validation-loss p-hacking:** do not choose per-run stopping points based on validation loss. If selecting an earlier stopping step, use the same stopping criterion/step across all trials.
9. **Pin the commit.** Because this repository evolves quickly, we should record the exact git commit hash, training script, PyTorch version, CUDA version, GPU type, and number of GPUs.

### 4.3 Suggested benchmark protocol

We should start with the official/simple Muon setup from the track, then edit only the Muon momentum coefficient logic.

Proposed protocol:

1. Clone the benchmark repo and pin a commit.
2. Prepare data using the official command.
3. Select one official Muon-based script as the base script.
4. Create a minimal patch that adds a `beta_scheduler` option to the Muon optimizer.
5. Keep all non-beta hyperparameters fixed.
6. Run initial sweeps with small seed counts, e.g. \(n=1\) or \(n=2\), to remove bad schedules.
7. Run promising schedules with larger seed counts, e.g. \(n=5,8,10\), depending on compute.
8. Report:
   - mean validation loss at fixed step count;
   - earliest step satisfying the benchmark statistical criterion;
   - beta/window curves;
   - gradient/momentum diagnostics.

### 4.4 Main metric

Primary metric:

```text
Fewest training steps to statistically satisfy validation loss < 3.28.
```

Secondary metrics:

- validation loss at fixed step count, e.g. 3125 steps;
- mean and standard error over seeds;
- pairwise comparison against best fixed-beta baseline;
- stability / NaN rate;
- wall-clock overhead, even though wall-clock is not the target.

---

## 5. Implementation details

### 5.1 Where to modify

The Muon update usually contains a line like:

```python
momentum.lerp_(grad, 1 - mu)
```

and, for Nesterov Muon, something like:

```python
update = grad.lerp(momentum, mu)
```

The change should be minimal:

```python
mu_t = beta_scheduler(step, group, state, grad, momentum)
momentum.lerp_(grad, 1 - mu_t)
update = grad.lerp(momentum, mu_t)  # if original Muon uses Nesterov
```

Important: if the original benchmark Muon uses the same `mu` in both the EMA update and the Nesterov interpolation, we should initially schedule both together to preserve the original implementation semantics. Later, we can ablate “EMA beta only” versus “Nesterov interpolation beta only,” but this should not be in v1.

### 5.2 Per-parameter or global beta?

For v1, use a **global scalar beta schedule** shared across all Muon-optimized hidden matrices.

Reasons:

- cleaner comparison;
- cheaper;
- less risk of overfitting;
- easier to explain.

For negative feedback, compute the alignment statistic globally across all Muon parameters by accumulating dot products and norms:

```math
a_t =
\frac{\sum_i \langle G_{t,i},M_{t-1,i}\rangle_F}
{\sqrt{\sum_i\|G_{t,i}\|_F^2}\sqrt{\sum_i\|M_{t-1,i}\|_F^2}+\epsilon}.
```

This avoids a different beta per matrix in v1.

### 5.3 Distributed training issue

The benchmark uses distributed training. If computing global alignment, we must all-reduce scalar accumulators across ranks before updating the global beta. This is cheap compared with gradient all-reduce.

If we want the simplest implementation, first avoid feedback and test deterministic schedules. For feedback, implement scalar all-reduce carefully.

### 5.4 Logging

Log the following every validation step, or every 25/125 steps matching benchmark logging:

```text
step
val_loss
mu_t / beta_t
T_mom = 1/(1-beta_t)
global grad norm for Muon params
global momentum norm for Muon params
global cos(G, M)
optional: update norm after Muon orthogonalization
```

This logging does not change the training rule and should be cheap.

---

## 6. Initial experiment matrix

### Phase 1: sanity check and static beta baselines

Run fixed beta:

```math
\beta\in\{0.85,0.90,0.95,0.97,0.98,0.99\}.
```

Use \(n=1\) or \(n=2\) first.

Goal:

- locate best constant beta;
- identify instability at too-large or too-small beta;
- establish the true baseline that schedules must beat.

### Phase 2: deterministic schedules

Run:

1. horizon-dependent log-linear schedule \(\beta_{\min}\to\beta_{\max}\);
2. horizon-dependent linear-window schedule;
3. Schedule-Free-inspired horizon-free window growth.

Candidate settings:

```math
T_{\min}\in\{5,10\},
\quad
T_{\max}\in\{50,100\},
\quad
\tau\in\{100,250,500,1000\}.
```

Goal:

- compare horizon-dependent vs horizon-free deterministic scheduling;
- see whether increasing memory is consistently better than fixed beta.

### Phase 3: negative feedback schedules

Start with alignment controller:

```math
\log T_{t+1}^{\mathrm{mom}}
=
\operatorname{clip}
\left(
\log T_t^{\mathrm{mom}} + \rho(a_t-a_\star),
\log T_{\min},
\log T_{\max}
\right).
```

Candidate settings:

```math
T_{\min}=10,
\quad
T_{\max}=100,
\quad
\rho\in\{0.005,0.01,0.02\},
\quad
 a_\star\in\{0.1,0.2,0.3\}.
```

Goal:

- test whether adaptive scheduling improves over deterministic schedules;
- inspect whether beta increases early and stabilizes, or decreases during rapid training phases.

### Phase 4: statistical validation

For the top 2-3 schedules:

- run enough seeds to satisfy the benchmark rule;
- compare against the best fixed-beta baseline with the same number of seeds if possible;
- report both “validity vs 3.28” and pairwise improvement over the fixed beta.

---

## 7. Expected outcomes and interpretation

### Possible outcome A: fixed beta wins

Then the conclusion is still useful: Muon on this benchmark may already use a near-optimal momentum horizon, and scheduling provides little benefit. We should report this as a negative result and inspect diagnostics.

### Possible outcome B: naive pre-scheduling wins

This supports the heuristic that Muon benefits from weak filtering early and strong filtering late, but the best rule may still depend on the training horizon.

### Possible outcome C: Schedule-Free-inspired scheduler matches or beats pre-scheduling

This would be strong: it suggests that an anytime deterministic memory-growth rule is sufficient, and we can avoid tuning beta schedules to the final step count.

### Possible outcome D: negative feedback wins

This is the most interesting research outcome. It would suggest that the momentum filter should not merely increase monotonically; it should respond to gradient-momentum agreement and reduce memory when the signal drifts.

---

## 8. Things to be careful about

1. **Do not compare weak baselines.** The best fixed beta may not be the benchmark default.
2. **Schedule the memory window, not beta directly.** Always report \(T_t^{\mathrm{mom}}=1/(1-\beta_t)\).
3. **Nesterov coupling matters.** If the code uses `mu` in both momentum update and Nesterov interpolation, schedule both in v1; ablate later.
4. **Keep LR schedule fixed.** Otherwise the experiment becomes a joint LR-beta scheduling study.
5. **Avoid validation-based tuning inside a run.** This would violate the benchmark spirit/rules.
6. **Beware overfitting to Track 3.** Use Track 3 as the main benchmark, but eventually confirm on a second setting if the idea works.
7. **Pin repo commit and script.** The benchmark changes quickly.
8. **Keep code simple.** The benchmark explicitly values reproducible self-contained scripts; avoid unnecessary “barnacles.”

---

## 9. Minimal naming

Possible method names:

- **BetaSched-Muon**: generic umbrella name.
- **SF-MuonBeta**: Schedule-Free-inspired deterministic beta scheduling.
- **FB-MuonBeta**: feedback-controlled beta scheduling.
- **AIM-Muon**: Adaptive Inner Momentum Muon.

For now, use neutral names:

```text
Muon-StaticBeta
Muon-PreBeta
Muon-SFBeta
Muon-FBBeta
```

---

## 10. v1 claim to test

> For Muon, the learning-rate schedule controls update amplitude, while the momentum coefficient controls the temporal filtering horizon of the matrix passed to orthogonalization. Keeping the LR schedule fixed, we test whether scheduling this filtering horizon improves NanoGPT optimization efficiency. We compare fixed beta, horizon-dependent beta schedules, Schedule-Free-inspired horizon-free memory growth, and negative-feedback adaptive memory control under the Modded-NanoGPT Track 3 optimization benchmark.
