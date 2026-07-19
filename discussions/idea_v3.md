The two papers [1,2] are studying two different failure modes of the same pipeline,

[
G_t
;\xrightarrow{\text{temporal aggregation}};
M_t
;\xrightarrow{\text{spectral transformation}};
D_t
;\xrightarrow{\text{parameter update}};
X_{t+1}.
]

[1] references/momentum-filtering/Denoise First, Orthogonalize Later- Understanding Momentum in Muon via Spectral Filtering.pdf
[2] references/river-valley/Towards Understanding the Power and Limits of the Muon Optimizer- A River-Valley Perspective.pdf

The momentum-filtering paper studies mainly the first arrow and asks:

> What matrix should be given to the polar factor?

Its answer is that a temporally averaged momentum matrix can separate persistent signal from perturbation before orthogonalization; otherwise the polar map can destroy the amplitude gap that distinguishes signal from noise. 

The river-valley paper studies mainly the second and third arrows and asks:

> Even with a useful direction, when is full spectral normalization beneficial or harmful?

Its answer is that strong orthogonalization can help rapid progress through anisotropic landscapes, but near the target the loss of residual-scale information causes overshoot and oscillation. 

So, conceptually, I would say:

> **One paper studies temporal filtering before spectral normalization; the other studies the optimization consequences of spectral normalization after temporal filtering.**

That does suggest a joint theory.

---

# 1. I would formulate the unified optimizer with two independent controls

Let

[
M_t
===

\beta_t M_{t-1}
+
(1-\beta_t)G_t
]

and then replace the binary choice

[
M_t
\quad\text{versus}\quad
\operatorname{polar}(M_t)
]

by a continuous family

[
D_t=\Phi_{\alpha_t}(M_t),
\qquad \alpha_t\in[0,1].
]

Here:

* (\beta_t) controls **temporal filtering strength**;
* (\alpha_t) controls **spectral flattening strength**.

A mathematically natural example is, for

[
M=U\operatorname{Diag}(\sigma_i)V^\top,
]

to define something like

[
\Phi_\alpha(M)
==============

U
\operatorname{Diag}\left(\sigma_i^{,1-\alpha}\right)
V^\top,
]

with appropriate global normalization so that changing (\alpha) does not merely change the effective learning rate.

Then:

[
\alpha=0
\quad\Longrightarrow\quad
\Phi_0(M)=M
]

is momentum-GD-like, while

[
\alpha=1
\quad\Longrightarrow\quad
\Phi_1(M)=UV^\top
]

is full Muon/polar behavior.

This gives the two-dimensional control problem

[
\boxed{
(\beta_t,\alpha_t)
==================

\text{temporal filtering strength}
\times
\text{spectral normalization strength}.
}
]

I think this is a much cleaner unification than talking about the “magnitude of the polar factor.”

There is one important technical correction here: **multiplying the polar factor by a scalar does not change the degree of orthogonalization**. That is essentially only a learning-rate change. To interpolate between Muon and momentum-GD, you need to change the singular-value transformation itself.

---

# 2. Your proposed schedule is plausible, but one part needs qualification

Your intuition is:

[
\begin{array}{c|cc}
& \beta & \alpha\
\hline
\text{early} & \text{small} & \text{large}\
\text{late} & \text{large} & \text{small}
\end{array}
]

I think the **(\alpha) part is strongly justified**.

The **(\beta) part is plausible but not yet implied by these two papers**.

## Early stage: large (\alpha)

This is the most convincing part.

Suppose the landscape is very anisotropic. In a crude diagonal model,

[
g_i=\lambda_i x_i.
]

Under the power transformation,

[
d_i
===

\operatorname{sign}(g_i)|g_i|^{1-\alpha},
]

the spectral disparity is changed roughly from

[
\frac{\lambda_{\max}}{\lambda_{\min}}
]

to

[
\left(
\frac{\lambda_{\max}}{\lambda_{\min}}
\right)^{1-\alpha}.
]

Thus increasing (\alpha) reduces anisotropy.

At (\alpha=1), the magnitude differences disappear completely.

That is almost exactly the mechanism emphasized by the river-valley paper: Muon can move rapidly along weak but information-bearing spectral directions instead of being dominated by the strongest curvature directions. 

So:

[
\boxed{\text{early stage: strong spectral flattening is well motivated.}}
]

---

## Late stage: small (\alpha)

This is also strongly motivated.

Consider the one-dimensional quadratic

[
f(x)=\frac{\lambda}{2}x^2.
]

Under the interpolated update

[
x^+
===

## x

\eta
\operatorname{sign}(x)
|\lambda x|^{1-\alpha},
]

a descent condition roughly requires

[
\eta
<
C
\frac{|x|^\alpha}{\lambda^{1-\alpha}}.
]

The important term is

[
|x|^\alpha.
]

Therefore:

* (\alpha=0): safe step size remains (O(1)) as (x\to0);
* (0<\alpha<1): safe step size shrinks like (|x|^\alpha);
* (\alpha=1): safe step size shrinks linearly with residual magnitude.

This creates a continuous interpolation of the river-valley paper's result.

In my opinion, this may be one of the cleanest theoretical bridges between the papers:

[
\boxed{
\text{spectral flattening strength }\alpha
\quad\Longleftrightarrow\quad
\text{degree of residual-scale destruction}.
}
]

Full Muon has maximum anisotropy correction but maximum late-stage scale blindness. Raw momentum has no spectral flattening but retains residual magnitude.

That produces a real trade-off, not merely a heuristic schedule.

---

# 3. Why small (\beta) early and large (\beta) late is plausible

Here I think the right argument is not simply:

> early = exploration, therefore low momentum.

That argument is too loose.

A stronger argument comes from **signal tracking versus noise suppression**.

Suppose

[
G_t=S_t+\Xi_t,
]

where (S_t) is the current useful gradient signal and (\Xi_t) is stochastic perturbation.

EMA momentum produces

[
M_t
===

\underbrace{\operatorname{EMA}*\beta(S_t)}*{\text{filtered signal}}
+
\underbrace{\operatorname{EMA}*\beta(\Xi_t)}*{\text{filtered noise}}.
]

Increasing (\beta):

1. decreases stochastic noise;
2. but increases lag when (S_t) itself changes rapidly.

The momentum-filtering paper proves the first benefit under a coherent or slowly drifting signal model: larger effective window size suppresses perturbation and improves singular-subspace reliability. 

But once the signal changes quickly, there is another error:

[
M_t-S_t
=======

\underbrace{
\operatorname{EMA}*\beta(S)*t-S_t
}*{\text{tracking bias / lag}}
+
\underbrace{
\operatorname{EMA}*\beta(\Xi)*t
}*{\text{noise}}.
]

This is, in my view, the missing term in the current momentum-filtering theory.

In a simple random-walk signal model,

[
S_t=S_{t-1}+W_t,
]

one can get an EMA tracking error of approximately the form

[
\operatorname{MSE}(\beta)
\sim
\frac{
\beta^2 q
+
(1-\beta)^2 r
}{
1-\beta^2
},
]

where:

* (q) measures signal drift;
* (r) measures observation noise.

Then:

* large (q/r): faster-changing signal (\Rightarrow) smaller optimal (\beta);
* small (q/r): stable signal with substantial noise (\Rightarrow) larger optimal (\beta).

This gives a principled version of your proposal:

[
\boxed{
\beta_t
\text{ should be controlled by the local signal-drift/noise ratio.}
}
]

If early training has rapidly changing gradient directions and later training has more persistent local signal, then a small-to-large (\beta) schedule follows naturally.

But this empirical premise needs to be demonstrated rather than assumed.

---

# 4. My actual preferred schedule is slightly more nuanced than two phases

I would expect something closer to three regimes:

| Phase  | Signal drift |            Anisotropy problem | Precision need | Preferred behavior                      |
| ------ | -----------: | ----------------------------: | -------------: | --------------------------------------- |
| Early  |         high |                          high |            low | moderate/low (\beta), high (\alpha)     |
| Middle |        lower |                    still high |         medium | larger (\beta), moderate/high (\alpha)  |
| Late   |          low | less important than precision |           high | large or moderate (\beta), low (\alpha) |

So rather than only

[
(\beta_{\rm low},\alpha_{\rm high})
\rightarrow
(\beta_{\rm high},\alpha_{\rm low}),
]

I would initially test

[
\boxed{
(\beta_{\rm low},\alpha_{\rm high})
\rightarrow
(\beta_{\rm high},\alpha_{\rm high})
\rightarrow
(\beta_{\rm high},\alpha_{\rm low}).
}
]

Why?

Because the filtering paper suggests that larger (\beta) can improve the matrix presented to orthogonalization. Therefore, there may be a middle phase where **strong denoising and strong orthogonalization are complementary**, not contradictory. 

Then only in the refinement phase do we reduce (\alpha), because spectral magnitude becomes important for step calibration.

---

# 5. I think the strongest unified theory would have this form

The full gradient model should combine both papers:

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

* (S_t): useful river/signal component;
* (H_t): structured nuisance or hill component;
* (\Xi_t): stochastic perturbation.

Then:

[
M_t
===

\operatorname{EMA}_{\beta}(G)_t.
]

The theory could proceed in four layers.

## Layer A: temporal filtering theorem

Bound

[
|M_t-S_t|
]

as

[
\underbrace{
|\operatorname{EMA}*{\beta}(S)*t-S_t|
}*{\text{signal drift / lag}}
+
\underbrace{
|\operatorname{EMA}*{\beta}(H)*t|
}*{\text{structured nuisance}}
+
\underbrace{
|\operatorname{EMA}_{\beta}(\Xi)*t|
}*{\text{stochastic noise}}.
]

The current filtering paper essentially develops the last part and signal persistence. The extension would explicitly include drift.

---

## Layer B: partial orthogonalization theorem

Study

[
D_t=\Phi_{\alpha}(M_t).
]

Prove two opposing effects of increasing (\alpha):

### Benefit

It reduces spectral imbalance and improves movement along weak informative directions.

Something like

[
\kappa_{\rm effective}
\approx
\kappa^{1-\alpha}.
]

### Cost

It reduces residual scale sensitivity and shrinks the safe late-stage step size approximately like

[
\eta_{\max}
\propto
|X_t-X^\star|^\alpha.
]

This would provide a continuous generalization of the river-valley paper's GD-versus-Muon dichotomy.

---

## Layer C: interaction theorem

This is potentially the most novel part.

The quality of

[
\Phi_\alpha(M_t)
]

depends jointly on:

[
\beta
\quad\text{and}\quad
\alpha.
]

A stronger (\alpha) makes spectral perturbations more consequential because it modifies singular values more aggressively. Therefore stronger orthogonalization may require a cleaner input matrix.

Conceptually:

[
\boxed{
\text{larger }\alpha
\Rightarrow
\text{greater need for sufficient temporal filtering}.
}
]

This creates a coupling constraint:

[
\alpha
\leq
\alpha_{\max}
\left(
\text{spectral gap of } M_t
\right).
]

Or conversely,

[
\beta
\geq
\beta_{\min}(\alpha,\mathrm{SNR},\mathrm{drift}).
]

That would be a much deeper result than simply proposing two schedules independently.

---

## Layer D: adaptive control

Then the final optimizer becomes

[
\beta_t
=======

\mathcal B(
\widehat{\text{drift}},
\widehat{\text{noise}}
),
]

[
\alpha_t
========

\mathcal A(
\widehat{\text{anisotropy}},
\widehat{\text{residual scale}},
\widehat{\text{spectral reliability}}
).
]

The conceptual interpretation becomes:

> **(\beta) decides how much history to trust.
> (\alpha) decides how much spectral magnitude to trust.**

I think that is a very clean unifying principle.

---

# 6. One thing I would be careful about

I would not claim yet:

> “The two papers prove that early training should use low (\beta) and late training should use high (\beta).”

They do not.

The momentum-filtering paper actually provides evidence that larger (\beta) improves signal alignment under its assumptions, and its trajectory probes report this behavior beyond strictly stationary settings. 

Therefore, the scientifically safer claim is:

> The two papers jointly motivate **separating temporal filtering strength from spectral normalization strength**. They strongly motivate decreasing spectral normalization toward the late stage. A low-to-high momentum schedule is an additional hypothesis that follows if the useful gradient signal becomes more slowly varying relative to noise over training.

That hypothesis is quite testable.

---

# 7. My honest assessment of the research idea

I think the basic idea makes sense and is stronger than merely combining two papers.

The most interesting research question is not:

> Should (\beta) increase and orthogonalization decrease?

It is:

> **Can Muon be understood as a composition of a temporal filter and a spectral filter, and can their strengths be jointly optimized according to the changing time-frequency and spectral geometry of training?**

That gives you a two-axis theory:

[
\boxed{
\text{time axis: }\beta
\qquad
\text{spectral axis: }\alpha
}
]

with two independent but interacting trade-offs:

[
\beta:
\quad
\text{denoising}
\leftrightarrow
\text{tracking lag},
]

[
\alpha:
\quad
\text{anisotropy correction}
\leftrightarrow
\text{residual-scale preservation}.
]

My current view is that this is probably the cleanest way to unify your momentum-filtering work with the river-valley analysis. The strongest theoretical result would be a **joint phase diagram in ((\beta,\alpha))** showing which regime is optimal as a function of noise, signal drift, anisotropy, and distance to the target.
