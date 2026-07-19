



## Overall assessment

The paper develops a coherent temporal-frequency explanation of momentum: steep directions in a river-valley landscape generate rapidly oscillating gradients, while progress along the river generates slowly varying gradients; EMA momentum suppresses the former while retaining the latter. It then studies how this filtering interacts with the optimization dynamics themselves and attempts to extend the argument to Muon-style polar updates. fileciteturn0file0

My assessment is:

> **The central straight-valley theory is largely correct and potentially publishable. The present manuscript is not yet ready for JMLR because the curved-valley results are stated more strongly than proved, and the claimed transfer from buffer subspace recovery to the polar/Muon update is not mathematically established.**

**Current JMLR-level recommendation: reject and encourage resubmission after major revision.**  
**Overall rating: 5/10.**

---

# 1. Main idea

The paper combines four observations.

### 1. Momentum is a temporal low-pass filter

For

\[
m_t=\beta m_{t-1}+(1-\beta)g_t,
\]

the response to a temporal frequency \(\omega\) is

\[
H_\beta(\omega)
=
\frac{1-\beta}{1-\beta e^{-i\omega}}.
\]

Thus:

- \(H_\beta(0)=1\): a constant or slowly varying gradient passes through.
- \(|H_\beta(\pi)|=(1-\beta)/(1+\beta)=1/T_{\mathrm{eff}}\): a step-by-step sign flip is strongly attenuated.
- Increasing \(\beta\) improves high-frequency suppression but increases lag on nonzero low frequencies.

The paper correctly emphasizes that a filter theory is useful only after explaining where the input spectrum comes from.

### 2. A river valley generates frequency separation

For a quadratic straight valley,

\[
L(x,y)=\frac{\mu}{2}(x-x^\star)^2+\frac{\lambda}{2}y^2,
\qquad \mu\ll\lambda,
\]

gradient descent gives

\[
y_{t+1}=(1-\eta\lambda)y_t.
\]

When \(1<\eta\lambda<2\), the hill coordinate changes sign every step. Its gradient is therefore concentrated near the Nyquist frequency \(\omega=\pi\). The slowly evolving river coordinate is concentrated near DC.

Momentum consequently tends to preserve river progress while suppressing hill bouncing.

### 3. Closing the loop changes the conclusion

The gradient sequence is not actually an exogenous signal: momentum changes the iterate, which changes future gradients. The paper derives the hill recursion

\[
y_{t+1}
=
\bigl(1+\beta-\eta(1-\beta)\lambda\bigr)y_t
-\beta y_{t-1}
-\eta(1-\beta)\xi_t.
\]

From this it obtains:

- Stability under the normalized EMA convention:

\[
\eta\lambda
<
2\frac{1+\beta}{1-\beta}
=
2T_{\mathrm{eff}}.
\]

- Stationary hill variance:

\[
\operatorname{Var}(y_\infty)
=
\frac{\eta\sigma^2}
{\lambda\left(2-\eta\lambda/T_{\mathrm{eff}}\right)}.
\]

- A frequency-dependent forced response: Nyquist forcing is suppressed, DC forcing is not, and forcing near the system's underdamped mode can be amplified.

This leads to a **good-\(\beta\) window**, rather than a universal preference for \(\beta\to1\).

### 4. Filtering before a nonlinear direction map

The paper finally argues that, for matrix gradients,

\[
G_t=S+(-1)^tA,
\]

EMA suppresses \(A\) before applying the polar factor, making the buffer's dominant singular subspaces better reflect \(S\). It uses Wedin-type perturbation bounds and extends the argument to band-limited and band-energy disturbances.

This final step is where the largest theoretical problem arises.

---

# 2. Mathematical audit

## 2.1 Results that appear correct

### Proposition 1: exact finite-window transfer identity

The derivation on page 9 is correct. Exchanging the sums gives an exact DFT relationship with a finite-window boundary term:

\[
\widehat m(\omega_\ell)
=
H_\beta(\omega_\ell)
\left(
\widehat g(\omega_\ell)
-
e^{-i\omega_\ell(T+1)}B
\right).
\]

The boundary term is important, particularly when \(T\) is not substantially larger than the EMA memory. The manuscript recognizes this rather than pretending that the infinite-stream transfer identity is exact on a finite window.

### Theorem 1: finite-window band filtering

The high-band and low-band bounds follow correctly from Proposition 1, the triangle inequality, and the monotonicity of \(|H_\beta(\omega)|\). The expressions

\[
\rho_{\mathrm{high}}(\beta)
=
\frac{1-\beta}
{\sqrt{1-2\beta\cos\omega_c+\beta^2}}
\]

and

\[
\epsilon_{\mathrm{low}}(\beta)
=
\frac{2\beta\sin(\omega_0/2)}
{\sqrt{1-2\beta\cos\omega_0+\beta^2}}
\]

are correct.

One qualification is that Theorem 1 does not necessarily imply contraction on a short window because the boundary term may dominate. The paper mostly handles this qualification properly.

### Theorem 2: deterministic frequency separation

For the straight quadratic valley, the proof that the alternating hill transient is maximized at \(\omega=\pi\), while the positive river transient is maximized at \(\omega=0\), is valid.

The phrase “raises the river-to-hill energy ratio by a factor up to \(((1+\beta)/(1-\beta))^2\)” should, however, be read as an open-loop, near-pure-tone limit. It is not a general closed-loop guarantee.

### Theorem 3: stationary hill-gradient spectrum

This is one of the cleanest results in the manuscript. For

\[
y_{t+1}=a_0y_t-\eta\xi_t,
\qquad
g_t=\lambda y_t+\xi_t,
\qquad
a_0=1-\eta\lambda,
\]

the transfer from \(\xi\) to \(g\) is indeed

\[
\frac{1-z^{-1}}{1-a_0z^{-1}},
\]

giving

\[
S_g(\omega)
=
\sigma^2
\frac{|1-e^{-i\omega}|^2}
{|1-a_0e^{-i\omega}|^2}.
\]

The zero at DC, monotonicity, endpoint values and total variance are correct.

A useful interpretive caveat is that \(S_g(0)=0\) is partly a closed-loop telescoping identity:

\[
g_t=\frac{y_t-y_{t+1}}{\eta}.
\]

It does not, by itself, prove that every observed high-frequency gradient component is a geometric “hill mode.”

### Corollary 1 and Lemma 1

The rearrangement/Chebyshev inequality used for the white-noise ceiling is appropriate, given the monotone spectrum assumption.

The confinement-versus-DC lemma is also correct and offers a useful assumption-light observation.

### Propositions 2–4 and Corollary 2

The straight-valley closed-loop calculations appear correct:

- The Jury stability threshold is correct.
- The AR(2) stationary variance is correct.
- The stationary loss reduction \(\eta\lambda/2\) is correct under the stated limit \(\beta\to1\), when \(\eta\lambda<2\).
- The DC and Nyquist forced gains are correct.
- The resonant asymptotic

\[
|G_\beta(\theta_\beta)|
\asymp
\sqrt{\frac{\eta}{\lambda(1-\beta)}}
\]

is also correct.

These results are, in my view, the strongest part of the paper.

Their scope is nevertheless narrow:

- normalized EMA update at fixed \(\eta\);
- straight quadratic direction;
- additive, state-independent white noise;
- stationarity and sufficient burn-in.

The fact that the \(\beta\)-sweep reverses under fixed heavy-ball learning rate is not a minor technicality. It substantially limits any headline claim that “larger momentum reduces hill loss.”

---

## 2.2 Proposition 5 needs correction

Part (a), the one-step Taylor identity for the curved valley, is correct.

Part (b) correctly derives an approximate frozen-coefficient recursion:

\[
d_{t+1}
\quad\text{behaves like the straight hill recursion with}\quad
\lambda_{\mathrm{loc}}
=
\lambda(1+f'^2),
\]

together with forcing \(-f'\phi'(x_t)\).

The problem is the subsequent exact claim for a linear floor \(f(x)=cx\):

\[
\mathbb E[d_\infty]
=
\frac{c\phi'}{\lambda_{\mathrm{loc}}},
\qquad
\operatorname{Var}(d_\infty)
=
\frac{\eta\sigma^2}
{\lambda(2-\eta\lambda_{\mathrm{loc}}/T_{\mathrm{eff}})}.
\]

Even when \(f'=c\) is constant, the forcing

\[
-c\phi'(x_t)
\]

is generally not constant. Under the paper's earlier choice

\[
\phi(x)=\frac{\mu}{2}(x-x^\star)^2,
\]

\(x_t\) and \(d_t\) form a coupled stochastic linear system. One cannot directly apply Proposition 3 to \(d_t\) while ignoring the stochastic forcing through \(x_t\).

For example, already at \(\beta=0\), solving the complete two-dimensional Lyapunov equation gives

\[
\operatorname{Var}(d_\infty)
=
\frac{
\eta\sigma^2(2-\eta\mu)
}{
\lambda\left[
4-2\eta\lambda(1+c^2)-2\eta\mu+\eta^2\lambda\mu
\right]
},
\]

whereas the manuscript's expression would be

\[
\frac{\eta\sigma^2}
{\lambda\left[2-\eta\lambda(1+c^2)\right]}.
\]

These are generally unequal. They become close when the river curvature is sufficiently small, which explains why the approximation may work empirically, but the formula is not exact as currently stated.

The proposition can be repaired in one of three ways:

1. Assume \(\phi'\) is constant, corresponding to a constant-slope river potential.
2. Treat \(x_t\) as an exogenous slowly varying process and label the result quasi-static.
3. Solve the full coupled linear system for quadratic \(\phi\).

A related issue is that pointwise frozen stability,

\[
\eta\lambda(1+f'(x)^2)<2T_{\mathrm{eff}},
\]

is not a global curved-valley stability theorem. Products of time-varying matrices need not be stable merely because every frozen matrix is stable. The paper acknowledges that a global theorem is open, but several surrounding claims and the abstract state the curved-valley prediction too definitively.

---

## 2.3 The polar/Muon conclusion does not follow from Theorems 4–5

The EMA calculations inside Theorems 4 and 5 are mostly correct. In particular, for

\[
G_t=S+(-1)^tA,
\]

the buffer disturbance coefficient decays to \(1/T_{\mathrm{eff}}\), and the Wedin bounds can show that the **top-\(r\) singular subspaces of \(M_t\)** approach those of \(S\), subject to a gap.

What is not justified is the statement that the **polar update inherits this recovery**.

The exact polar factor flattens every nonzero singular value to one. It therefore erases the singular-value ordering used by the theorem.

Consider the paper's own orthogonal rank-one construction:

\[
S=\sigma uv^\top,
\qquad
A=\alpha u'v'^\top,
\qquad
u\perp u',\;v\perp v'.
\]

For every nonzero \(\varepsilon\),

\[
\operatorname{polar}(S+\varepsilon A)
=
uv^\top+\operatorname{sign}(\varepsilon)u'v'^\top
\]

under the corresponding partial-polar interpretation. The amplitude \(|\varepsilon|\) disappears. Reducing the disturbance from \(\alpha\) to \(\alpha/T_{\mathrm{eff}}\) does not make this polar output converge to \(uv^\top\) unless the disturbance becomes exactly zero or a rank truncation/regularization is introduced.

Consequently:

- Theorems 4–5 prove a result about the input buffer \(M_t\).
- They do not prove convergence or improved alignment of \(\operatorname{polar}(M_t)\).
- Proposition 6 introduces an implicit “top-\(r\) readout,” but ordinary Muon has no such rank-\(r\) readout.
- Saying that ties in the polar output should be resolved using the input ordering is not an intrinsic property of the polar output.

This is a central issue because the manuscript repeatedly calls these “filter-first theorems for the polar map.”

A rigorous repair would require one of the following:

- **Full-rank signal:** assume \(\sigma_{\min}(S)>0\) and apply an actual polar-factor perturbation bound to  
  \(\|\operatorname{polar}(S+E)-\operatorname{polar}(S)\|\).
- **Truncated polar map:** explicitly study  
  \(D_r(M)=U_rV_r^\top\), for which top-\(r\) subspace recovery is operational.
- **Regularized polar map:** use a map such as  
  \[
  D_\epsilon(M)=M(M^\top M+\epsilon^2I)^{-1/2},
  \]
  under which sufficiently small singular components remain small.

Without such a change, the Muon component is a buffer-subspace observation rather than a theorem about the Muon update.

---

## 2.4 The claimed unification with acceleration is incomplete

The discussion says that acceleration, denoising and filtering “coincide.”

The denoising and temporal-filter interpretations do connect cleanly. Classical momentum acceleration does not follow merely from

\[
H_\beta(0)=1.
\]

Under normalized EMA, DC gain is exactly one, not greater than one. Heavy-ball acceleration arises from the coupled second-order dynamics and parameter tuning, not simply from passing low frequencies. Under the unnormalized heavy-ball convention, persistent gradients receive a gain \(1/(1-\beta)\), but that is a different normalization.

The paper should present filtering as complementary to acceleration rather than claiming that the three are essentially the same mechanism.

---

# 3. Experimental assessment

## Strengths

The forced-response experiment on pages 22–23 is particularly well designed. It tests:

- a passband disturbance that momentum should not remove;
- a Nyquist disturbance that it should remove;
- a resonant disturbance for which momentum should hurt.

This is considerably stronger than showing only that increasing \(\beta\) reduces some empirical variance.

The large-batch state-versus-residual decomposition on pages 30–33 is also directionally valuable. It attempts to distinguish sampling noise from trajectory-induced oscillation rather than labeling every high-frequency component “noise.”

The manuscript also reports negative results, including:

- deterministic ringing at large \(\beta\);
- the failure of the mechanism score to rank \(\beta\) at transformer scale;
- a training/generalization disagreement in one network.

That improves credibility.

## Weaknesses

### 1. Much of the synthetic validation is self-confirmation

Several experiments simulate exactly the linear AR(2) model from which the formulas were derived and then show that the empirical covariance or response matches the formula. These are useful implementation checks, but they are not independent evidence that neural-network optimization is adequately modeled by the theory.

### 2. The real-network evidence is too small for the breadth of the claims

The main non-toy experiments are:

- one small two-layer full-batch network;
- a 0.40M-parameter character-level nanoGPT;
- three seeds;
- one probed matrix for the Muon comparisons.

That is insufficient for broad claims about language-model training or practical Muon optimization at JMLR level.

### 3. The Muon experiment does not test practical Muon

The manuscript itself admits that it does not include:

- all matrix layers;
- Newton–Schulz orthogonalization;
- tuned Muon baselines;
- realistic model scale.

Given that the corresponding theorem does not actually control the polar output, stronger empirical evidence is especially important.

### 4. HFER is not enough to establish temporal whiteness

A residual HFER close to the white-bin fraction shows that its energy is distributed similarly across the chosen low/high split. It does not establish that the residual is temporally independent or white. Autocorrelation tests, full spectral confidence bands, and sensitivity to the \(0.6\pi\) cutoff would be needed.

### 5. Surviving-seed means are problematic

At learning rate \(0.4\), the \(\beta=0\) arm diverges on one of three seeds, while reported means use surviving runs. A percentage improvement relative to this baseline can be biased. Divergence should be incorporated into the performance criterion rather than dropping the failed seed.

### 6. Unreported schedule results appear in the Discussion

Page 32 contains quantitative claims about an “E7 schedule arm,” including factors of 15 and 118, confinement detectors and several warmup schedules. I could not find a corresponding experimental subsection, figure, table or methodology earlier in the manuscript.

Those claims should either be fully documented or removed.

### 7. Reproducibility is not currently demonstrated

The manuscript refers to local `codebases/`, cached run identifiers and JSON records, but the supplied paper does not provide a public repository or supplementary archive. I therefore cannot independently verify the reported experiments.

---

# 4. JMLR-style rating

| Criterion | Rating | Assessment |
|---|---:|---|
| Originality | **7/10** | The landscape-to-spectrum-to-filter loop is a worthwhile and reasonably novel synthesis. |
| Technical depth | **7/10** | Broad collection of exact filter, AR(2), forced-response and perturbation analyses. |
| Correctness | **5/10** | Straight-valley results are strong; curved-valley exactness and polar transfer contain substantial gaps. |
| Significance | **6/10** | Potentially useful for understanding momentum and coefficient scheduling, but practical conclusions remain limited. |
| Empirical support | **4.5/10** | Excellent controlled experiments, but inadequate scale and breadth for the larger claims. |
| Clarity | **6/10** | Generally self-contained, but overlong and sometimes mixes theorem, approximation and empirical heuristic. |
| Reproducibility | **4/10** | Numerous run identifiers, but no verifiable public artifact in the manuscript. |
| **Overall** | **5/10** | Interesting paper with a solid core, but not JMLR-ready. |

## Recommendation

**Reject in the present form; encourage a substantially revised resubmission.**

The paper could become a strong JMLR submission after:

1. Correcting Proposition 5 and clearly separating exact, frozen-coefficient and empirical statements.
2. Replacing the current polar-subspace argument with a genuine theorem about the polar output, or removing the Muon theorem claim.
3. Reframing normalization-dependent results so that the practical scope is unmistakable.
4. Documenting or removing the missing schedule experiments.
5. Adding public code and substantially stronger neural-network and Muon experiments.
6. Shortening the paper and moderating claims such as “experiments confirm every prediction” and “the three views coincide.”

The most defensible paper today is a narrower one centered on **Theorems 1–3 and Propositions 2–4: momentum filtering in a straight river valley, with exact closed-loop frequency response and a filtering–lag–resonance tradeoff**. That core is substantially stronger than the current curved-valley and Muon extensions. 

Yes. For the present paper, GPU-based neural-network training should be the next empirical stage.

A purely theoretical JMLR paper does not necessarily require large-scale deep-learning experiments. However, this manuscript makes claims about edge-of-stability training, practical momentum behavior, and Muon-style nonlinear updates. Without GPU experiments, the evidence remains concentrated on models whose dynamics are very close to the assumptions used in the proofs. The manuscript itself identifies practical Muon training, Newton–Schulz orthogonalization, all matrix layers, and larger-scale training as untested limitations. 

The goal should not simply be “train a larger model and report better loss.” The experiments need to test the specific causal chain:

[
\text{large learning rate}
\rightarrow
\text{high-frequency state-gradient oscillation}
\rightarrow
\text{momentum attenuation}
\rightarrow
\text{better stability/optimization},
]

with excessive momentum eventually causing lag or resonance.

# 1. Core hypotheses the GPU experiments must test

| Theoretical claim                                | Required deep-learning evidence                                                                                                 |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Steep directions emit high-frequency gradients   | High-frequency gradient energy should concentrate in high-curvature Hessian directions                                          |
| The high-frequency component is landscape-driven | It should appear in a large-batch mean-gradient stream, not only in mini-batch sampling noise                                   |
| Momentum helps more at large effective curvature | Its benefit should grow with learning rate, sharpness, or state-gradient HFER                                                   |
| Momentum extends stability                       | Some learning rates unstable at (\beta=0) should become stable with momentum                                                    |
| There is a finite good-(\beta) window            | Moderate (\beta) should outperform both insufficient filtering and excessive-memory regimes                                     |
| Filtering is frequency-specific                  | Low-frequency perturbations should remain, Nyquist perturbations should be removed, and resonant perturbations may be amplified |
| Filter-first matters for Muon                    | Momentum-before-polar should beat polar-before-momentum specifically in high-frequency regimes                                  |

These should become the organizing hypotheses of the experimental section.

---

# 2. Experiment A: GPU closed-loop sweep on an image model

## Recommended setup

Use a conventional model where curvature diagnostics remain computationally manageable:

* ResNet-18 on CIFAR-10 and CIFAR-100.
* Optionally one second architecture, such as a small ViT, after the ResNet experiment works.
* No data augmentation in the primary mechanism experiment, or use a fixed augmentation sequence across optimizer arms.
* Three seeds for the complete sweep; five seeds for headline comparisons.

Use the paper's normalized EMA update:

[
m_t=\beta m_{t-1}+(1-\beta)g_t,\qquad
w_{t+1}=w_t-\eta m_t.
]

Suggested grid:

[
\beta\in{0,0.5,0.9,0.95,0.99},
]

with learning rates covering:

1. a clearly subcritical regime;
2. an intermediate regime;
3. a near-edge-of-stability regime;
4. a regime where (\beta=0) is unstable but some (\beta>0) values remain stable.

Do not choose these learning rates only from conventional recipes. First perform a coarse learning-rate scan.

## What to measure

For each run:

* training and validation loss;
* accuracy;
* divergence and loss-spike frequency;
* gradient norm and momentum norm;
* temporal gradient spectrum on several representative layers;
* high-frequency energy ratio;
* momentum suppression ratio;
* alignment of (m_t) with a large-batch gradient proxy;
* approximate top Hessian eigenvalue;
* projection of the gradient stream onto the top Hessian eigenspace.

## Main predicted result

At low learning rates, the temporal gradient stream should be relatively smooth, and changing (\beta) should have limited benefit.

Near the stability boundary:

* gradient HFER should increase;
* the state-gradient high-frequency component should concentrate in sharp directions;
* moderate or large (\beta) should improve training stability and loss;
* excessively large (\beta) should eventually incur lag or inadequate burn-in.

The strongest plot would be:

[
\text{momentum benefit}
\quad\text{versus}\quad
\eta\widehat\lambda_{\max}
]

or versus the measured state-gradient HFER, rather than merely plotting performance against (\beta).

---

# 3. Experiment B: separate hill oscillation from sampling noise

This is probably the most important mechanism experiment.

For a short diagnostic window, at every visited iterate (w_t), compute:

[
g_t^{\mathrm{mb}}
=================

\bar g_t^{\mathrm{LB}}
+
\xi_t^{\mathrm{res}},
]

where:

* (g_t^{\mathrm{mb}}) is the actual training mini-batch gradient;
* (\bar g_t^{\mathrm{LB}}) is a gradient evaluated on a fixed large probe batch at the same (w_t);
* (\xi_t^{\mathrm{res}}) is their difference.

This does not need to run throughout training. Use perhaps two or three diagnostic windows:

* early training;
* near the onset of edge-of-stability behavior;
* late training.

Each window should be long relative to the momentum memory. For example, (\beta=0.99) has

[
T_{\mathrm{eff}}=\frac{1+\beta}{1-\beta}=199,
]

so a 128-step window is inadequate. Diagnostic windows should preferably be at least 1024–2048 steps when testing (\beta=0.99), or the analysis should explicitly retain finite-window effects.

## Required analyses

Compute temporal spectra for:

[
g_t^{\mathrm{mb}},\qquad
\bar g_t^{\mathrm{LB}},\qquad
\xi_t^{\mathrm{res}}.
]

Then estimate a top-(k) Hessian eigenspace using Hessian-vector products or block Lanczos at the midpoint of the window. Decompose the large-batch state gradient into

[
\bar g_t^{\mathrm{LB}}
======================

P_{\mathrm{top}}\bar g_t^{\mathrm{LB}}
+
(I-P_{\mathrm{top}})\bar g_t^{\mathrm{LB}}.
]

The paper's river-valley interpretation receives strong support only when:

1. most high-band energy is in (\bar g_t^{\mathrm{LB}}), not merely in (\xi_t^{\mathrm{res}});
2. the high-band state component is disproportionately concentrated in (P_{\mathrm{top}});
3. low-frequency content is relatively stronger outside the sharp subspace;
4. the momentum benefit increases as this state-gradient separation becomes more pronounced.

A flat spectrum of the residual alone is not sufficient to claim temporal independence. Also report:

* autocorrelation;
* cross-step covariance;
* spectra using multiple frequency cutoffs;
* confidence intervals from surrogate white-noise sequences.

---

# 4. Experiment C: real-network forced-frequency controls

The forced-response experiment is one of the paper's most distinctive theoretical contributions. It should be transferred from the toy valley to a real network.

At a selected training regime, estimate a sharp direction (v), for example the top Hessian eigenvector. Modify the gradient by

[
\widetilde g_t
==============

g_t+A\cos(\omega t)v,
]

using a sufficiently small (A) that the perturbation remains diagnostic rather than dominating training.

Test three frequency regimes:

1. **Low frequency:** (\omega\approx 0).
2. **Nyquist:** (\omega=\pi), giving alternating signs.
3. **Near-resonant frequency:** estimated using the observed gradient spectrum or the local quadratic approximation.

Measure:

* the amplitude of the induced parameter response;
* projected loss and gradient response along (v);
* training-loss degradation;
* dependence on (\beta).

The theory predicts:

* little or no removal of low-frequency bias;
* strong attenuation of the Nyquist perturbation;
* possible amplification near the optimizer's underdamped mode.

This is much stronger evidence than showing that high (\beta) simply lowers gradient variance. It tests whether the mechanism truly depends on temporal frequency.

A practical version can freeze the learning-rate schedule and run only a few hundred diagnostic steps from the same checkpoint for each perturbation arm.

---

# 5. Experiment D: language-model training

After the image-model mechanism is verified, transfer the result to a decoder-only transformer.

## Suggested progression

### Diagnostic scale

Use a roughly 10M–30M parameter GPT on a public text corpus. This scale permits:

* broad ((\eta,\beta)) sweeps;
* several seeds;
* gradient recording;
* large-batch probes;
* layerwise curvature approximations.

### Confirmation scale

Use a 50M–100M parameter model for a smaller number of selected configurations:

* low-HFER regime;
* high-HFER regime;
* best moderate (\beta);
* excessively high (\beta);
* unstable (\beta=0) regime stabilized by momentum.

The purpose of the larger run is confirmation, not hyperparameter exploration.

## Required comparisons

First use ordinary EMA-SGDM, before introducing Muon. This separates the generic temporal-filtering claim from nonlinear orthogonalization.

Report:

* training loss or perplexity;
* validation loss;
* time or tokens to a fixed target;
* divergence rate;
* state-gradient and residual spectra;
* layerwise HFER;
* sharpness proxies;
* momentum benefit as a function of learning rate.

A central predicted result is:

[
\text{best-}\beta\text{ improvement over }\beta=0
]

should grow as the learning rate approaches the high-frequency/stability regime.

However, the analysis should not require a perfectly monotonic relationship. Deep networks are nonstationary, and different layers may occupy different regimes.

---

# 6. Experiment E: practical Muon filter-first test

This experiment needs to be materially stronger than a one-matrix toy comparison.

## Pipelines

Compare:

### Pre-polar

[
M_t=\beta M_{t-1}+(1-\beta)G_t,
\qquad
U_t=\operatorname{NS\text{-}polar}(M_t).
]

### Post-polar

[
P_t=\operatorname{NS\text{-}polar}(G_t),
\qquad
\widetilde M_t
==============

\beta\widetilde M_{t-1}+(1-\beta)P_t.
]

### Polar-only

[
U_t=\operatorname{NS\text{-}polar}(G_t).
]

### Optional raw-momentum control

[
U_t=M_t.
]

Use:

* the same Newton–Schulz implementation;
* all eligible 2D matrix parameters;
* identical handling of embeddings, normalization parameters and biases;
* identical batch sequences;
* equalized computational budgets.

## Diagnostics

Do not measure only the singular subspaces of the input matrix. Because the theoretical polar claim currently has a gap, measure the actual update produced by each pipeline:

* alignment with a large-batch update proxy;
* alignment with (\operatorname{polar}(\bar G_t^{\mathrm{LB}}));
* descent obtained on a large evaluation batch after a small virtual step;
* update-stream HFER;
* singular-subspace stability;
* training and validation loss.

## Key falsifiable prediction

In a high-HFER regime:

[
\text{pre-polar}

>

\text{post-polar}
\approx
\text{polar-only}
]

in update quality and optimization.

In a smooth low-HFER regime, the pre/post gap should shrink or disappear. This regime interaction is more convincing than pre-polar winning everywhere.

A caveat remains: better empirical performance would not repair the mathematical gap concerning the exact polar factor. The theorem and experiment should be presented as separate evidence unless the theory is revised.

---

# 7. Experiment F: the finite (\beta) window

The paper should demonstrate both sides of the window in deep training.

### Lower edge

Too little momentum:

* inadequate suppression of sharp-direction oscillations;
* instability or larger training loss.

### Upper edge

Too much momentum:

* large phase lag;
* long burn-in;
* slower response to changing gradients;
* potentially resonant or underdamped behavior.

Useful measurements include:

[
1-\cos(m_t,\bar g_t^{\mathrm{LB}})
]

as an instantaneous lag proxy, together with high-band suppression.

The strongest result would show that the best (\beta) approximately minimizes a mechanism score such as

[
\text{high-band residual}
+
c\cdot\text{low-frequency lag},
]

but this score should be evaluated prospectively. It should not be tuned separately on every setting after seeing the final loss.

---

# 8. Normalization and fairness

This is essential.

The paper's main monotonicity results hold under normalized EMA momentum at fixed (\eta):

[
m_t=\beta m_{t-1}+(1-\beta)g_t,
\qquad
w_{t+1}=w_t-\eta m_t.
]

Standard deep-learning SGDM often uses

[
v_t=\beta v_{t-1}+g_t,
\qquad
w_{t+1}=w_t-\eta_{\mathrm{HB}}v_t.
]

Holding (\eta_{\mathrm{HB}}) fixed while changing (\beta) changes the effective step scale. Therefore use two separate protocols:

### Theory-faithful protocol

Hold normalized (\eta) fixed across (\beta). This directly tests the propositions.

### Practical-optimizer protocol

Use conventional SGDM or Muon and tune the learning rate for each (\beta), or apply the conversion

[
\eta_{\mathrm{HB}}=\eta(1-\beta).
]

Do not mix these protocols in one performance curve.

---

# 9. Statistical and spectral requirements

For JMLR-quality evidence:

* Use at least three seeds throughout and five seeds for headline claims.
* Count divergence as failure; do not report only surviving-seed means.
* Use identical batches and initialization across paired optimizer arms.
* Report confidence intervals.
* Use diagnostic windows much longer than (T_{\mathrm{eff}}).
* Report both rectangular-window spectra, which align with the exact theorem, and optionally tapered-window spectra for robustness.
* Test multiple high-frequency cutoffs rather than relying only on (0.6\pi).
* Report the complete ((\eta,\beta)) grid, including failed configurations.
* Separate exploratory sweeps from final confirmatory experiments.

---

# 10. Minimum viable JMLR experimental package

I would regard the following as the minimum serious package:

1. **ResNet-18/CIFAR LR–(\beta) sweep** showing that momentum benefit grows in the high-curvature/high-HFER regime.
2. **Large-batch state-versus-residual decomposition** showing that the high band is predominantly trajectory-generated.
3. **Hessian-subspace analysis** showing concentration of high-frequency state gradients in sharp directions.
4. **Real-network forced-frequency intervention** reproducing passband neutrality, Nyquist suppression and resonant harm.
5. **Transformer training with EMA-SGDM** confirming regime transfer.
6. **Practical all-layer Muon comparison** using Newton–Schulz: pre-polar, post-polar and polar-only.
7. **A finite-(\beta) window result**, not merely monotonic improvement with momentum.

The image and language experiments serve different purposes. The image model provides tractable mechanism diagnostics; the transformer establishes relevance to the optimization setting motivating the paper.

# Recommended order

The most efficient sequence is:

1. ResNet LR–(\beta) sweep.
2. State/residual and curvature decomposition on selected ResNet runs.
3. Forced-frequency intervention.
4. Small-GPT EMA-SGDM sweep.
5. Practical Muon pre/post comparison.
6. Larger confirmation runs only after the mechanism survives the earlier tests.

This avoids spending substantial GPU compute on a scaling experiment before knowing whether the key measurable predictions survive in an ordinary neural network.
