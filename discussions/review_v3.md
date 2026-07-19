## Overall review judgment

The modified version is **substantially stronger** than the previous draft. It now has the missing direct experiment: in E9, the paper explicitly sweeps (\beta), measures tail hill loss, compares it to the Corollary 1 prediction, and separates EMA-normalized momentum from heavy-ball-normalized momentum. It also adds a beta-window curved-valley experiment, a theory-to-experiment map, a band-limited filter-first theorem, and a small mini-batch transformer test. These additions directly address the earlier concern that the paper did not clearly show whether larger (\beta) improves river-valley optimization. 

My top-conference-level assessment would be:

**Recommendation: borderline accept / weak accept as a theory-mechanism paper, but not yet a strong accept.**

For a journal, I would say **major revision**, mainly because the theory is cleanest for toy river-valley models while the large-scale evidence is still quite small.

The paper is promising and intellectually coherent. The main risk is that reviewers may see it as a **unification of several recent ideas** rather than a wholly new mechanism. The paper’s best defense is to emphasize that the novelty is not “momentum is a filter” alone, but:

[
\text{momentum filter} + \text{river-valley geometry} + \text{closed-loop stability/variance} + \text{Muon filter-first order}.
]

---

## Main idea

The paper argues that EMA momentum is a temporal low-pass filter on the gradient stream:

[
m_t=\beta m_{t-1}+(1-\beta)g_t.
]

Low-frequency gradient components pass through, while rapidly alternating components are suppressed. The river-valley landscape supplies a geometric source of this frequency separation: river motion is slow / DC-like, while hill-wall bouncing at large learning rate creates high-frequency, near-Nyquist gradient oscillations. The paper then argues that momentum improves optimization when the hill component is sustained and high-frequency, while the river component remains slow enough to avoid filter lag. The paper explicitly states that the benefit is a **(\beta)-window**, not a monotone preference for (\beta\to 1). 

The Muon part extends the same logic to nonlinear direction maps: if one applies the polar factor after filtering, the polar map receives a cleaner matrix; if one applies the polar map before filtering, high-frequency perturbations can already distort the singular subspace.

---

## Novelty assessment

### What is not fully novel

The individual ingredients already exist.

First, momentum-as-filter / signal-processing views already exist. The paper itself cites Li et al. 2025, which interprets momentum as a time-varying frequency-domain filter whose response depends on the momentum coefficient. That work also reports that high-frequency components can help early but hurt late. ([arXiv][1]) There are also earlier signal-processing optimizer papers, such as “Signal Processing Meets SGD: From Momentum to Filter,” which uses Wiener filtering ideas to improve first-moment estimation. ([arXiv][2])

Second, the river-valley landscape is not new. Wen et al. introduced the river-valley explanation for warmup-stable-decay schedules: high learning rate causes oscillations in steep mountain directions while still allowing progress along the river. ([arXiv][3])

Third, momentum and averaging in river-valley settings are already being discussed. The Schedule-Free / “Through the River” line says that averaging can cancel oscillations along hill directions and that larger (\beta_1) suppresses hill-oscillation variance. ([arXiv][4])

Fourth, the Muon-specific “filter before orthogonalization” idea is close to the user’s earlier Muon momentum paper. “Denoise First, Orthogonalize Later” already argues that momentum in Muon acts as a spectral filter that suppresses perturbations before the polar/orthogonalization step. ([arXiv][5]) Shen et al. also study Muon through a river-valley perspective, although their momentum treatment is more assumption-based. ([arXiv][6])

### What is novel enough

The paper’s novelty is in the **synthesis and sharpening**:

1. It connects the frequency-domain momentum view to a concrete geometric source: river-valley hill oscillations.
2. It computes the emitted hill-gradient spectrum in a simple closed-loop stochastic model.
3. It derives exact stability and stationary-variance formulas under EMA-normalized momentum.
4. It translates variance into a clean optimization statement: stationary hill loss decreases with (\beta) in the predicted straight noisy valley regime.
5. It clarifies why this does not imply (\beta\to1): curved valleys and finite burn-in create a (\beta)-window.
6. It extends the Muon filter-first story from stochastic signal-plus-noise to deterministic Nyquist and band-limited disturbances.

So I would describe the novelty as **moderate-to-strong for a theory/mechanism paper**, but not as a completely independent new optimizer theory. It is best positioned as:

> A unifying temporal-spectral theory explaining when momentum helps in river-valley optimization and why filter-first nonlinear direction maps such as Muon benefit from momentum.

---

## Strengths

The strongest improvement is E9. The paper now directly asks whether larger (\beta) improves river-valley optimization. In the straight noisy valley, it reports that measured tail hill loss decreases with (\beta), that the maximal relative reduction matches the (\eta\lambda/2) prediction, and that the effect is modest at small (\eta\lambda) but large near instability. It also shows that the direction of the (\beta)-sweep flips under fixed heavy-ball normalization, which is an important honesty point. 

The theory-to-experiment map is also a good addition. It makes the paper easier to review because each theorem is tied to a falsifiable experimental prediction: transfer response, frequency separation, stability boundary, stationary hill loss, beta window, and pre-polar subspace recovery. 

The paper is also careful about scope in several places. It explicitly separates open-loop filtering from closed-loop damping, noting that momentum attenuates exogenous high-frequency content but does not damp the deterministic hill transient it creates; in fact, high (\beta) can ring longer.  This is important and should remain.

The addition of Theorem 5 is conceptually useful. Theorem 4’s exact ((-1)^tA) disturbance was elegant but too special; the band-limited version better connects the finite-window filter theorem to the polar/Muon story. 

The mini-batch transformer experiment is a meaningful step beyond pure toys. It is still small, but it gives the paper a more top-conference-compatible empirical bridge: 0.40M-parameter character-level GPT, multiple learning rates, (\beta)-sweeps, HFER diagnostics, and pre-/post-polar comparisons. 

---

## Main weaknesses

### 1. The real deep-learning evidence is still too small

E11 is helpful, but a 0.40M character-level GPT on tiny-Shakespeare with 3 seeds is not enough to convince skeptical reviewers that the mechanism matters for modern training. The paper itself admits that production scale, Adam-preconditioned updates, and learning-rate schedules remain untested. 

For a top conference, this is the biggest acceptance risk. The theory and toys are good, but reviewers may ask: “Does this explain anything beyond a carefully constructed toy and a tiny GPT?”

### 2. HFER in mini-batch training may conflate hill oscillation with stochastic noise

In full-batch edge-of-stability experiments, a high-frequency peak is easier to interpret as deterministic hill bouncing. In mini-batch training, high-frequency gradient content can simply be batch noise. The paper says the mini-batch gradient stream is hill-dominated at every stable learning rate, but without a decomposition into large-batch gradient, mini-batch noise, and curvature-aligned components, the term “hill-dominated” may be too strong.

A reviewer could say: “You measured high-frequency content, but you did not prove it is river-valley hill motion rather than stochastic sampling noise.”

### 3. The curved-valley theory is still mostly local / heuristic

The straight valley theory is clean. The curved valley is empirically explored, but the theory remains a local approximation. E10 even finds a correction: as bend frequency rises, river speed collapses, so the naive prediction that the upper (\beta)-edge should fall with bend frequency is incomplete. This is actually interesting, but it also shows the curved-valley theory is not closed yet. 

### 4. Theorem 5 is useful but still stylized

Theorem 5 assumes finite signed frequency sets, explicit spectral gaps, and for the strongest subspace claim, slow drift confined to the signal subspaces. This is mathematically reasonable, but still far from real nonstationary matrix gradients. It would be stronger if formulated for finite-window energy bands using Parseval-style bounds, not just finite Fourier sums with conservative amplitude sums.

### 5. Muon claims are not yet tested in realistic Muon training

The pre-polar vs post-polar comparison is important, but the current mini-batch Muon-style test applies variants to a probed matrix / limited setting rather than full practical Muon training across all matrix-shaped layers with standard implementation details. Since Muon is now an active optimizer line, reviewers may expect stronger baselines and practical settings. Muon itself is a real optimizer for hidden layers and has been used in NanoGPT/CIFAR speedrun settings. ([Keller Jordan][7]) There is also a rapidly growing Muon theory literature, including spectral-flattening, fractional Muon powers, Nesterov/inexact polar decomposition, and other variants. ([arXiv][8])

---

## Technical correctness

I do not see an obvious fatal mathematical error. The EMA transfer function, finite-window identity, straight-valley frequency separation, AR(2) stability threshold, stationary variance, and stationary hill-loss corollary all look algebraically sound.

The parts I would ask the authors to tighten are:

1. **Boundary term formalization.** Remark 2 is much improved, but the phrase “per-bin relative weight is (O(T_{\rm eff}^2/T))” should specify whether this is squared relative energy or amplitude-level ratio. It also relies on (S(\omega_\ell)) being bounded away from zero, which should be emphasized.

2. **Theorem 5 constants.** The use of (\ell_1)-style sums (a=\sum|C_\omega|*2), (\alpha=\sum|D*\omega|_2) is safe but conservative. A reviewer may ask why not use energy norms or random-phase/orthogonality assumptions to obtain sharper bounds.

3. **Stationary noise assumption.** Proposition 3 assumes white, additive, state-independent noise. Real mini-batch noise is state-dependent, anisotropic, and temporally correlated. This is acceptable for a model theorem, but the paper should avoid implying the formula quantitatively explains real training without qualification.

4. **EMA vs heavy-ball convention.** The paper now addresses this, which is good. But the abstract and introduction should be very explicit that the monotone hill-loss decrease is under the fixed-(\eta) EMA normalization, not under every conventional momentum parameterization. 

---

## What to improve next

### Highest priority: strengthen real-scale evidence

Run at least one experiment that reviewers recognize as non-toy:

* NanoGPT optimization track or modded-nanogpt optimization track.
* CIFAR-10 ResNet / ConvNet with SGDM and Muon variants.
* A small but standard language-modeling setup beyond tiny-Shakespeare, ideally with multiple seeds and a practical tokenizer/dataset.

For each setting, report:

[
\beta,\quad \eta,\quad \text{train loss},\quad \text{validation loss},\quad \text{HFER},\quad \text{large-batch gradient HFER},\quad \text{mechanism score}.
]

The key experiment should not just show that (\beta=0.9) works. It should show:

[
\text{momentum benefit grows when high-frequency structured gradient content grows}.
]

### Separate stochastic noise from hill oscillation

Add a decomposition experiment:

1. Record mini-batch gradients (g_t^{mb}).
2. Periodically estimate a large-batch or full-batch proxy (g_t^{LB}).
3. Decompose:

[
g_t^{mb} = g_t^{LB} + \xi_t.
]

Then compute HFER separately for (g_t^{LB}) and (\xi_t). If the high-frequency signal is mostly in (g_t^{LB}), the river-valley hill story is strongly supported. If it is mostly in (\xi_t), the story becomes ordinary denoising rather than hill filtering.

This is probably the single most important diagnostic improvement.

### Add Hessian / curvature alignment diagnostics

To justify “hill” rather than just “high-frequency,” measure whether high-frequency gradient components align with high-curvature directions. For example:

* estimate top Hessian eigenvectors or top Gauss-Newton directions;
* project gradient stream onto top-curvature subspace and its orthogonal complement;
* show high-frequency power is concentrated in high-curvature directions;
* show momentum suppresses the high-curvature oscillatory component while preserving low-curvature progress.

This would directly connect temporal frequency to landscape geometry.

### Expand Muon experiments

Run full optimizer comparisons:

[
\text{Muon pre-polar},\quad
\text{post-polar Muon},\quad
\text{polar-only},\quad
\text{SGDM},\quad
\text{AdamW if appropriate}.
]

Use standard Muon implementation details, including Newton-Schulz approximation and Nesterov if that is the standard baseline. If the theory excludes Nesterov, then include both non-Nesterov and Nesterov variants and state the gap.

The key claim should be:

> Pre-polar wins when the raw matrix-gradient stream has structured high-frequency disturbance; the gap shrinks when the stream is smooth or noise-dominated.

### Generalize the curved-valley theory

The current curved-valley story is empirically interesting but theoretically incomplete. A stronger theorem would model

[
L(x,y)=\phi(x)+\frac{\lambda}{2}(y-f(x))^2
]

and derive explicit conditions involving (f'(x)), (f''(x)), (\eta), (\lambda), (\mu), and (T_{\rm eff}). The goal would be to formalize when:

[
\text{larger }\beta \Rightarrow \text{better hill suppression},
]

and when:

[
\text{too large }\beta \Rightarrow \text{river tangent lag}.
]

Even a local perturbation theorem would improve the paper.

### Make Theorem 5 more realistic

Replace or supplement the finite-tone theorem with a finite-window band-energy theorem:

[
G_t = S_t + A_t,
]

where (S_t) has most DFT energy below (\omega_0), (A_t) has most DFT energy above (\omega_c), and leakage is bounded. Then prove that the filtered matrix has an improved signal-to-disturbance ratio up to boundary and leakage terms.

This would align more naturally with the empirical HFER/MSR diagnostics.

### Add ablations that can falsify the mechanism

Include negative controls:

1. High-frequency component in the river direction: momentum should hurt or lag.
2. Low-frequency disturbance in hill direction: momentum should not remove it well.
3. Same (\beta), different frequency content: performance should follow frequency content, not (\beta) alone.
4. Same HFER, different curvature alignment: if only curvature-aligned high frequency predicts benefit, that strengthens the river-valley claim.

### Improve positioning

The paper should not market itself as “momentum as a filter,” because that is already known. It should market itself as:

> A closed-loop temporal-spectral theory explaining why river-valley landscapes emit filterable hill gradients, when (\beta) improves optimization, when it hurts, and why filter-first nonlinear maps such as Muon benefit.

That is more precise and more defensible.

---

## Likely reviewer concerns

A strong reviewer might write:

1. “Momentum-as-filter is not new.”
2. “River-valley landscapes are not new.”
3. “The Muon filter-first story overlaps with prior work.”
4. “The main theory is on a linear quadratic valley.”
5. “The mini-batch experiment is too small and may measure stochastic noise rather than hill oscillation.”
6. “The paper explains optimization, not generalization.”

The best response is not to deny these. The best response is to make the contribution precise:

> Existing work gave the filter, the valley, and the Muon subspace story separately. This paper derives the temporal spectrum emitted by the valley, closes the loop with exact stability/loss formulas, explains the (\beta)-window, and connects the same mechanism to filter-first nonlinear maps.

---

## Final assessment

The modified paper is now a coherent and plausible top-conference submission. The idea is **not entirely new at the component level**, but the **unified mechanism is novel enough** if presented carefully. The math mostly makes sense, and the new experiments directly address the previous gap about larger (\beta).

To move from borderline/weak accept to strong accept, the paper needs one of the following:

1. a substantially stronger real-training experiment, or
2. a sharper theorem for curved/nonlinear river valleys, or
3. a convincing decomposition showing that high-frequency mini-batch gradients are truly curvature/hill-driven rather than just stochastic noise.

The most efficient next step is the decomposition experiment: large-batch gradient proxy, mini-batch noise residual, frequency spectra for both, and Hessian-direction alignment. That would directly test the central claim and make the paper much harder to dismiss as “momentum just denoises noisy gradients.”

[1]: https://arxiv.org/abs/2411.19671?utm_source=chatgpt.com "On the Performance Analysis of Momentum Method: A Frequency Domain Perspective"
[2]: https://arxiv.org/abs/2311.02818?utm_source=chatgpt.com "Signal Processing Meets SGD: From Momentum to Filter"
[3]: https://arxiv.org/abs/2410.05192?utm_source=chatgpt.com "Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective"
[4]: https://arxiv.org/html/2507.09846v3?utm_source=chatgpt.com "Understanding the Benefit of Schedule-Free Methods for ..."
[5]: https://arxiv.org/abs/2606.03899?utm_source=chatgpt.com "Denoise First, Orthogonalize Later: Understanding Momentum in Muon via Spectral Filtering"
[6]: https://arxiv.org/html/2606.21514v1?utm_source=chatgpt.com "Towards Understanding the Power and Limits of the Muon ..."
[7]: https://kellerjordan.github.io/posts/muon/?utm_source=chatgpt.com "Muon: An optimizer for hidden layers in neural networks"
[8]: https://arxiv.org/abs/2605.13079?utm_source=chatgpt.com "[2605.13079] Spectral Flattening Is All Muon Needs"
