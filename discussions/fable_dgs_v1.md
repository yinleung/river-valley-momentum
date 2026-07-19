# Fable deep-dive v1: does momentum filtering survive the river valley?

*2026-07-03. Written after reading all seven reference PDFs (including the new
`references/river-valley/` folder), `latex/main.tex` (the current draft), and `discussions/idea_v1.md`.
Independent assessment requested by Leon: (Q1) is the general momentum-filtering idea valid once we
drop the time-invariant-signal + BVMZOS (bounded-variance, mean-zero, temporally orthogonal
sequence) gradient model of the Muon paper? (Q2) is the current draft good, or does it need
modification? All new analytical claims below were verified numerically; script with captured
output alongside this note (`fable_dgs_v1_check.py`), key numbers quoted inline. Notation
throughout: EMA momentum $m_t=\beta m_{t-1}+(1-\beta)g_t$ with
$T_{\mathrm{eff}}=(1+\beta)/(1-\beta)$; in the valley models, $\eta$ = step size, $\lambda$ = hill
curvature, $\mu$ = river curvature, $\sigma$ = gradient-noise scale.*

---

## 0. TL;DR

**Q1 — Yes, the idea is valid, but it is a three-layer claim and the layers have different
epistemic status.** The filter layer is unconditionally true; it removes the BVMZOS assumption and
additionally covers deterministic high-frequency streams
(a deterministic Nyquist disturbance is attenuated at rate $1/T_{\mathrm{eff}}$ in operator norm — *faster*
than the $T^{-1/4}$ the stochastic model gives). The landscape layer (river/hill frequency
separation) is true well beyond the proved straight-valley edge-of-stability (EoS) case — I sketch two strengthenings
below, one of which needs almost no assumptions. The optimization layer is the delicate one: the
mechanism provably improves the *direction quality of the buffer* (which is exactly what matters
for Muon-class optimizers), and at the *trajectory* level it helps under sustained noise at large
learning rate — but it does **not** damp the deterministic self-generated oscillation (the draft's
underdamped-hill remark is correct and must stay). The one-sentence version:

> Momentum filtering is a theorem about the gradient *stream*; it becomes a theorem about
> *optimization* either through a nonlinear direction map (Muon: filter-first) or through noise at
> large learning rate (stochastic river valley) — and only there.

**Q2 — The draft is a sound and honest skeleton, not yet a competitive 2026 paper.** Three things
changed my assessment while reading the new references: (i) the June-2026 Muon river-valley paper
(Shen et al.) *assumes* exactly what this project proves — a gift for positioning, but only if we
cite and use it; (ii) the closed-loop remark currently tells only the negative half of the story —
the positive half (stability threshold $2T_{\mathrm{eff}}$, stationary hill variance $\propto 1/(2-\eta\lambda/T_{\mathrm{eff}})$,
river drift preserved) is a five-line computation away and is, in my view, the single strongest
addition the paper could make; (iii) the filter-first section rests on a paragraph of argument plus
a toy — the deterministic version of T5 is essentially free and should be a theorem. Prioritized
list in §6.

---

## 1. What exactly is being loosened

The Muon paper (`li2026denoise`, arXiv:2606.03899) proves its results under:

- **A1(a) Coherent signal**: $G_t^{\mathrm{sig}} = \sum_k \lambda_k(t)\,u_k v_k^\top$ with
  *time-invariant* orthonormal families $\{u_k\},\{v_k\}$; **A3** strengthens to fully
  time-invariant $G^{\mathrm{sig}}$.
- **A1(b) BVMZOS perturbation**: $\mathbb E[\Xi_t]=0$, $\mathbb E\|\Xi_t\|_F^2\le\eta$, and
  $\mathbb E\langle\Xi_{t_1},\Xi_{t_2}\rangle_F=0$ for $t_1\ne t_2$ (temporal orthogonality).
- **A2 Signal persistence**: the momentum-filtered coordinate stays bounded away from zero.

The river-valley hill disturbance $H_t=(-1)^tA$ (or any sustained oscillation) violates **every
axis of A1(b)**: it is not mean-zero ($\mathbb E[\Xi_t]=\Xi_t\ne0$), and consecutive terms are
maximally *anti*-correlated, not orthogonal ($\langle H_t,H_{t+1}\rangle_F=-\|A\|_F^2$). So the
river-valley setting is a genuine extension, not a special case — the project premise is sound.
The two loosening axes are:

| BVMZOS-world assumption | Loosened replacement | Where it lives in the draft |
|---|---|---|
| perturbation mean-zero + temporally orthogonal (white in expectation) | perturbation is *high-frequency* (any coloring, deterministic allowed) | Thm 2 (filter), high band |
| signal time-invariant (DC) | signal *slowly varying* (low band, distortion $\varepsilon_{\mathrm{low}}$) | Thm 2(b), E2 lag, tradeoff remark |

A useful bookkeeping fact I noticed: the Muon paper's ubiquitous constant $2T-1$ (with
$T=1/(1-\beta)$) *equals* the draft's $T_{\mathrm{eff}}=(1+\beta)/(1-\beta)$ exactly. The two papers already
speak the same quantitative language; the draft could say so in one line when citing Theorem 1.

## 2. Reading notes (what each reference contributes to the question)

**`li2026denoise` (the target).** Momentum opens a spectral gap: head singular values pinned at
$c_{\mathrm{sig}}\lambda_k - \sqrt{\eta}(2T-1)^{-1/4}$, tail crushed to $\sqrt{\eta}(2T-1)^{-1/4}$,
w.p. $1-(2T-1)^{-1/2}$; Wedin then gives subspace reliability $O(T^{-1/4})$; Pre-polar dominates
Post-polar and Polar-only in expected alignment (Thms 2–3), with Polar-only alignment vanishing as
$m\to\infty$ in the low-SNR rank-1 model. Note the rate: the *stochastic* perturbation dies at
$T^{-1/4}$ — slow, and forced through a Markov/Frobenius argument. Keep this number in mind for §3.1.

**Shen et al., arXiv:2606.21514 (new, June 19; the critical one to check for overlap).** Mixed-spiked
matrix sensing + river-valley decomposition. Momentum-free analysis: simplified Muon (msign) makes
*linear* early progress along the river but overshoots and oscillates near the river bottom
(constant-magnitude updates); GD refines geometrically; hence "Muon early, GD/AdamW late". When they
do add momentum (Sec. 4, Thms 4.1–4.2), its role is **assumed, not analyzed**: "the stored momentum
$M_0$ has to point mostly in the river direction", and their early-stage rate carries an unexplained
alignment factor $\rho$ between the momentum direction and the river gradient. **No frequency
analysis, no pre/post-polar comparison, no explanation of why momentum aligns with the river.**
Verdict: complementary, not competing — and their assumption is precisely the statement our
Thm (separation) + Thm (filter) discharge. This is the strongest "why now" hook available to the
paper and it is currently uncited.

**Wen et al., arXiv:2410.05192 (WSD).** River = direction of the smallest Hessian eigenvector;
GD/gradient flow converge to and track the river; SGD with mountain-direction noise bounces, with
stable-phase loss gap $(d-1)\eta\sigma^2/2$; decay descends the wall. Two facts matter for us:
(i) their theorems live in the *small-LR contractive* regime ($\eta<\gamma/2\gamma_{\max}^2$) — the
hill bouncing is noise-driven, **not** the deterministic EoS flip the draft's Thm (separation)
proves; (ii) there is **no momentum** anywhere. So the draft's current story covers the EoS regime
and the WSD regime only via "noise = BVMZOS"; §3.2 below closes that gap properly.

**Song et al., arXiv:2507.09846 (Through the River, NeurIPS 2025).** The closest existing
trajectory-level claim: for Schedule-Free AdamW, the $y_t$ iterates run at EoS (stability threshold
$2/((1-\beta_1)\gamma)$ — note the $1/(1-\beta)$), and a central-flow computation gives a hill
oscillation variance $\sigma^2(\beta_1)$ that *decreases* as $\beta_1\to1$: "larger $\beta_1$
suppresses oscillations along the hill directions". Mechanism: in SF's $y$-update the instantaneous
gradient enters scaled by $(1-\beta)$, so momentum shrinks the effective step — *not* spectral
filtering, and SF $\ne$ EMA-SGDM (the gradient is evaluated at an interpolated point; the averaging
iterate $x_t$ is a Polyak–Ruppert-style average). Also useful: their refined SF *decouples* the
momentum coefficient from the averaging window — independent evidence for the draft's point that one
$\beta$ is doing two jobs (filtering vs lag). Must-cite, must-differentiate.

**`li2025frequency` (ICLR 2025).** Establishes the filter language (transfer function, orthodox vs
unorthodox systems) and the empirical finding that high-frequency gradient content helps *early*
and hurts *late*; FSGDM is a heuristic built on that. It contains no landscape model and no
finite-window theorems — the draft's T1 is precisely the missing rigor, and the river valley is
precisely the missing "which components are high-frequency and why". The early/late finding also
imposes a claim discipline on us (§4, threat iii).

**Defazio et al. 2405.15682, Defazio 2605.19095.** Background for the averaging-vs-momentum
unification remark (§5, D2); no direct overlap.

**Added to `references/` today** (folder `edge-of-stability/`, manifest updated): Cohen et al.
central flows (arXiv:2410.24206) — the formal tool behind Song et al.'s oscillation claim; and
arXiv:2604.14108 (*Momentum Further Constrains Sharpness at the Edge of Stochastic Stability*) —
batch-sharpness plateaus $2(1-\beta)/\eta$ (small batch) vs $2(1+\beta)/\eta$ (large batch,
heavy-ball form). The latter matters because it means the momentum-shifted stability threshold in
§3.3 is *known* — the draft should derive it from its own $2\times2$ map and cite, not claim it.

An arXiv/web sweep for "river valley + momentum", "momentum spectral filter", and adjacent phrases
found no occupant of the intersection this project targets. The neighbors triangulate it: Shen
(river-valley Muon, momentum assumed), Song (hill-oscillation suppression, SF-specific, no spectra),
2604.14108 (momentum EoS thresholds, no landscape decomposition, no filtering), li2025frequency
(filtering, no theory), li2026denoise (filtering theory, no landscape). The niche is real and open.

## 3. Independent analysis: the three layers of the claim

### 3.1 Layer 1 — the filter (unconditional; survives any loosening)

Prop. (exact windowed transfer) and Thm. (band filtering) in the draft hold for *any* stream in any
Hilbert space — they are algebra, not probability. This layer doesn't merely "survive" the
loosening; it is *sharper* on the loosened inputs than the BVMZOS theory is on its own inputs:

- **White (BVMZOS) noise**: energy attenuation $\mathbb E\|\mathrm{EMA}(\xi)\|^2 = \sigma^2/T_{\mathrm{eff}}$;
  operator-norm tail dies at $(T_{\mathrm{eff}})^{-1/4}$ (li2026denoise Thm 1, probabilistic).
- **Deterministic Nyquist tone** $h_t=(-1)^tA$: the EMA has the *exact closed form*
  $$m_t^h = (-1)^t\,\frac{(1-\beta)\bigl(1-(-\beta)^t\bigr)}{1+\beta}\,A,$$
  verified to machine precision, so
  $\|m_t^h\|_{\mathrm{op}} = \frac{(1-\beta)\lvert1-(-\beta)^t\rvert}{1+\beta}\,\|A\|_{\mathrm{op}}
  \le 2\|A\|_{\mathrm{op}}/T_{\mathrm{eff}}$, settling to $\|A\|_{\mathrm{op}}/T_{\mathrm{eff}}$ once the
  $(-\beta)^t$ transient dies — **deterministic and in operator norm**: no Markov, no
  probability, and rate $T_{\mathrm{eff}}^{-1}$ instead of $T_{\mathrm{eff}}^{-1/4}$. Energy attenuation $1/T_{\mathrm{eff}}^2$.
- **Slowly drifting signal**: passes with distortion $\varepsilon_{\mathrm{low}}(\beta)\approx\beta\omega_0/(1-\beta)$
  — this is what replaces "time-invariant signal" (A3) and "signal persistence" (A2): persistence
  becomes "the signal band sits inside the passband".

So the structured, deterministic, high-frequency disturbance — the one BVMZOS excludes — is the
*easiest* case for the filter, not the hardest. This asymmetry ($T_{\mathrm{eff}}^{-1}$ vs $T_{\mathrm{eff}}^{-1/4}$) is
worth stating explicitly in the paper; it is the cleanest quantitative sense in which "denoise
first" generalizes to "filter first".

### 3.2 Layer 2 — the landscape (river valleys really do emit the right spectrum)

The draft proves the straight-valley EoS case: hill stream concentrated at $\omega=\pi$, river at
DC, for $1<\eta\lambda<2$. Two strengthenings make this layer robust rather than exemplary:

**(a) The noise-driven regime (unifies with Wen et al.).** In the WSD picture the hill is
contractive ($\eta\lambda<1$) and bounces on noise. Take the scalar hill model
$g_t=\lambda y_t+\xi_t$, $y_{t+1}=(1-\eta\lambda)y_t-\eta\xi_t$, $\xi_t$ white. Then the sampled
hill-gradient stream has stationary spectral density
$$S_{g^{\mathrm{hill}}}(\omega) \;=\; \sigma^2\,
\frac{\lvert 1-e^{-i\omega}\rvert^2}{\lvert 1-(1-\eta\lambda)e^{-i\omega}\rvert^2},$$
(the $\lambda y$ and $\xi$ contributions combine into an exact *differencing* — verified
numerically: band-averaged empirical/theory ratios $1.00\pm0.04$, DC bin $\sim10^{-6}$ of the mean
energy). Read it: the hill stream **always has zero DC mass**, is high-pass with cutoff
$\omega\sim\eta\lambda$, is $\approx$ white for $\eta\lambda\approx1$, and peaks at Nyquist iff
$\eta\lambda>1$. **[Erratum 2026-07-03, same day: the last clause is false — the argmax is $\pi$
for *every* $\eta\lambda\in(0,2)$ ($dS/d\cos\omega<0$ identically); the correct dichotomy is
concentration, $S(\pi)/S(\pi/2)=2(1+a^2)/(1+a)^2\ge2$ iff $\eta\lambda\ge1$. The periodogram
"peak at $\omega=2.66$" in the check script is sampling noise on a near-flat spectrum. Corrected
statement and proof in `theory_cl_t2_t5.md` T2; see `fable_wlog_v1.md` §4.1.]** So the frequency separation is not an EoS artifact: it degrades gracefully from
"Nyquist tone" (EoS) through "white above cutoff $\eta\lambda$" (WSD regime), and momentum's
river-to-hill improvement degrades correspondingly from $T_{\mathrm{eff}}^2$ toward $T_{\mathrm{eff}}$. One formula covers
both referenced landscapes' regimes.

This also yields the **principled $\beta$ window** the plan's T6 asked for: the EMA's half-power
point sits at $\omega=\Theta(1/T_{\mathrm{eff}})$ (about $1-\beta$; constants ignored throughout this
display), so filtering separates river from hill exactly when
$$\eta\mu \;\ll\; \frac{1-\beta}{1+\beta} \;\lesssim\; \eta\lambda,$$
i.e. the momentum memory should exceed the hill relaxation time $1/(\eta\lambda)$ but stay below the
river traversal time $1/(\eta\mu)$. Optimal $\beta$ ranges widen with the valley conditioning
$\lambda/\mu$ — a falsifiable prediction E3's grid could already test.

**(b) A nearly assumption-free separation (proposed new lemma, "T3′").** Telescoping the update
$w_{t+1}=w_t-\eta m_t$ gives $\sum_{t=1}^{T} m_t = (w_0-w_T)/\eta$ *exactly*, and via the draft's own
Prop. (transfer) at $\omega=0$, $\hat g(0)=\hat m(0)+B$ with $B=\tfrac{\beta}{1-\beta}m_T$ the
draft's boundary vector. Project onto any direction: if the hill coordinate stays **confined** in a
tube of radius $R$ (which, for a bounded gradient stream, also bounds the terminal buffer
$m_T^{\mathrm{hill}}$) while the river coordinate **travels** a distance $D_T$, then
$$\bigl|\hat g^{\mathrm{hill}}(0)\bigr| \le \frac{2R}{\eta} + \frac{\beta}{1-\beta}\bigl|m_T^{\mathrm{hill}}\bigr|
\quad\text{(bounded in }T\text{)}, \qquad
\bigl|\hat g^{\mathrm{river}}(0)\bigr| \approx \frac{D_T}{\eta} \quad(\text{grows with }T).$$
"The river is where the net displacement happens; hills are where you bounce" — no quadratic model,
no noise model, no EoS condition. To upgrade the DC bin to a low-frequency *band* one adds a single
mixing assumption (hill autocorrelation time $\tau \Rightarrow$ spectral mass below $\omega_0$ small
for $\omega_0\ll1/\tau$), whose concrete instance is exactly (a). I recommend this as the anchor
lemma of the landscape section: it is what makes the story survive "real landscapes are not
quadratic valleys".

**(c) A caveat for multi-dimensional hills.** With many hill eigenvalues $\lambda_i$, only those
with $\eta\lambda_i$ near or above 1 emit near-Nyquist content; small-$\lambda_i$ hills emit
low-frequency (but still DC-free) content that momentum passes. The separation statement should be
per-direction, ordered by $\eta\lambda_i$ — one sentence in the paper, but a reviewer will ask.

### 3.3 Layer 3 — optimization (where validity is conditional, and the draft must aim its claims)

This is the layer where the naive syllogism ("momentum filters hills ⇒ momentum optimizes better")
fails, and where I probed hardest. The truth splits into three regimes, all derivable from the same
$2\times2$ hill map $(y_t,m_{t-1})\mapsto(y_{t+1},m_t)$ that the draft's closed-loop remark already
analyzes ($y_{t+1}=(1+\beta-\eta(1-\beta)\lambda)y_t-\beta y_{t-1}$):

1. **Deterministic transient (the draft's remark — correct, keep it).** In the complex-pole regime
   the poles have modulus $\sqrt\beta$: the self-generated hill transient is underdamped and rings
   *longer* as $\beta\to1$. E1's $d_{\mathrm{rms}}$ non-monotonicity is this. Momentum does not damp what it
   itself drives.

2. **Stability threshold (currently missing — the positive deterministic half).** Jury conditions on
   the same map give stability iff
   $$\eta\lambda \;<\; 2\,\frac{1+\beta}{1-\beta} \;=\; 2\,T_{\mathrm{eff}},$$
   and at the threshold the unstable mode is again a period-2 flip (root $\to-1$), so the
   Nyquist-hill picture survives with momentum in the loop, at the shifted threshold. (Verified:
   simulated blow-up bracketing $2T_{\mathrm{eff}}$ to $\pm1\%$.) In sharpness language this is the known
   $2(1+\beta)/\eta_{\mathrm{HB}}$ momentum-EoS threshold (cite Cohen et al. 2021 and
   arXiv:2604.14108); the draft's contribution would be deriving it inside its own framework and
   converting it: at fixed $\eta$, momentum buys $T_{\mathrm{eff}}\times$ more headroom against the steepest
   hills — headroom that a large-LR river-valley schedule spends on river speed.

3. **Sustained noise (the missing closed-loop theorem, and I think the best one).** Solving the
   stationary variance of the same map driven by white gradient noise (AR(2) algebra, exact):
   $$\operatorname{Var}(y_\infty) \;=\; \frac{\eta\sigma^2}{\lambda}\cdot
   \frac{1}{\,2-\eta\lambda/T_{\mathrm{eff}}\,},$$
   verified against simulation to three digits (e.g. $\eta\lambda=1.8$, $\beta\in\{0,0.95\}$:
   predicted ratio $0.102$, measured $0.101$; monotone decreasing in $\beta$ in all regimes
   tested, $\eta\lambda\in\{0.3,1.0,1.8\}$). Meanwhile the *river drift is preserved exactly to
   first order*: the slow mode of the same recursion contracts at $1-\eta\mu$ per step, independent
   of $\beta$. Read the formula:
   - $\eta\lambda\ll1$: the factor is $\approx(2-\eta\lambda)/2$ — momentum changes (almost)
     nothing. This *recovers* the folklore/empirics that momentum is marginal at small LR under
     noise (Wang et al., cited in li2026denoise's intro) — a consistency check, and a limitation
     the paper can now state quantitatively instead of defensively.
   - $\eta\lambda\to2$: SGD's variance diverges while EMA-SGDM's stays $\approx\eta\sigma^2/2\lambda$
     — the reduction factor $(2-\eta\lambda)/(2-\eta\lambda/T_{\mathrm{eff}})$ is unbounded. Momentum's
     closed-loop benefit is **specifically a large-learning-rate, river-valley-regime effect**.
   - Combined with (2): momentum both *extends* the usable LR range by $T_{\mathrm{eff}}$ and *suppresses* the
     stationary hill energy inside it, at no first-order cost to river speed (the residual cost is
     the higher-order, $\beta$-dependent lag of the tradeoff remark). That is the trajectory-level
     content of "momentum filters the river", and it is provable in five lines from the map the
     draft already displays.

   (Convention caveat: all of this is the EMA normalization the draft and Muon use. In heavy-ball
   normalization, fixed $\eta$ means growing effective step $\eta/(1-\beta)$ and the conclusion
   inverts — worth one remark, since the two conventions coincide for Muon's scale-invariant polar
   step but *not* for SGDM trajectories. This is also precisely why Song et al.'s SF and our
   EMA-SGDM agree qualitatively: both keep the slow-direction effective step $\beta$-independent.)

4. **The nonlinear direction map (Muon; where the loosened theory is most valuable).** For
   normalized/sign/polar updates, the update direction is a function of the *buffer*, so Layer-1
   filtering transfers directly to update quality — the closed-loop subtleties above don't
   intervene. Here the loosening pays off hardest: with the exact Nyquist form of §3.1, the tail of
   the buffer is bounded *deterministically* at $\|A\|/T_{\mathrm{eff}}$, so the spectral gap needed by the
   Wedin route of li2026denoise holds with better constants and no probability, giving a
   filter-first theorem for a disturbance that is high-frequency but *not noise* (T5). E5 already
   shows the effect empirically (pre-polar subspace error 0.114 vs post-polar 0.133 vs polar-only
   0.508 at $\beta=0.9$). T5's proof is genuinely low-hanging: the only new ingredients are the
   closed form above and a head-singular-value lower bound, then cite Wedin.

**Verdict on Q1.** The general idea is valid, with this claim discipline: (i) never claim momentum
damps the deterministic oscillation it drives (it doesn't; E1 shows it); (ii) state trajectory-level
benefits in the stochastic large-LR regime, where they are provable and large; (iii) route all
direction-quality claims through the buffer and the nonlinear map, which is where Muon lives; (iv)
keep the filtering-lag tradeoff as the reason maximal $\beta$ is wrong. Within that discipline the
loosened theory is not just surviving — it is stronger and more general than the BVMZOS original on
its home turf.

## 4. What could still hurt it (threats, honestly)

1. **Scale transfer.** E6 is one small full-batch MLP at EoS. Whether real minibatch training at
   scale has hill-dominated gradient streams (HFER $\gg$ white baseline) is exactly the open E7/E6′
   question; Shen et al. and Song et al. both anchor their stories in LLM pretraining runs, and a
   venue reviewer will compare. This is the project's largest genuine risk, and it is an
   experiment-shaped risk, not a theory-shaped one.
2. **"High frequency = harmful" is regime-dependent.** li2025frequency's own empirics say
   high-frequency content helps *early*. The river-valley story actually predicts this: before the
   iterate reaches the tube, the hill gradient is the useful descent signal; confinement (and hence
   T3′'s DC argument) only holds after entry. The paper should make this reconciliation explicit —
   it turns an apparent self-contradiction between Leon's own two papers into a prediction
   (filtering starts paying once the iterate is river-confined; cf. $\beta$-warmup practice).
3. **Momentum ≠ unique low-pass filter.** Anything DC-preserving and high-cutting works (weight
   averaging, EWA, SF's $x_t$…). I'd frame this as a feature — the theory *predicts* the empirical
   near-equivalence of momentum, averaging, and LR decay in river valleys (Wen's decay, Song's
   averaging, our filtering are three implementations of the same spectral operation) — but the
   related-work section must own it before a reviewer does.
4. **Nesterov.** Muon implementations often use Nesterov momentum; li2026denoise explicitly excludes
   it, and so does the draft. Fine to inherit the exclusion, but say so.

## 5. Q2: assessment of `latex/main.tex`

**What is good (keep):** the exact windowed-transfer identity as the technical spine; the honest
open-loop/closed-loop remark; measurement definitions centralized; every number traced to run
records; the E1–E6 arc matching theory to controlled evidence; register discipline (WRITING.md
compliant as far as I can tell).

**What is not yet good enough, in priority order:**

- **P0 — Complete the closed-loop story (§3.3(2)+(3)).** Currently Remark (closedloop) concedes the
  deterministic case and offers nothing back. Add the threshold $\eta\lambda<2T_{\mathrm{eff}}$ (three lines,
  cite Cohen 2021 + arXiv:2604.14108 for the known sharpness form) and the stationary-variance
  proposition $\operatorname{Var}(y)=\frac{\eta\sigma^2}{\lambda(2-\eta\lambda/T_{\mathrm{eff}})}$ with
  river-drift preservation. Without this, the paper's title claim is open to the rebuttal "in your
  own E1 the trajectory is *worse* at large $\beta$"; with it, the rebuttal becomes a regime map.
- **P0 — Related work.** The draft cites no 2025–2026 neighbor except wen2024river. Minimum set:
  Shen et al. 2606.21514 (use their momentum-alignment assumption as the explicit gap we close —
  strongest positioning available), Song et al. 2507.09846 (differentiate: SF-specific effective-step
  mechanism, no spectra), arXiv:2604.14108 + Cohen et al. central flows (closed-loop EoS context),
  Defazio et al. (averaging unification, threat 3), Davis–Drusvyatskiy ravine (landscape kin).
- **P1 — Prove T5 (deterministic filter-first).** §3.3(4). Turns Sec. "Filter-First" from argued+toy
  into theorem+toy, and its rate ($T_{\mathrm{eff}}^{-1}$, deterministic) is *better* than the stochastic
  original — a headline, not a patch.
- **P1 — Stochastic-regime landscape proposition (§3.2(a)) and/or the confinement lemma (§3.2(b)).**
  Either one extends Thm (separation) beyond the EoS straight valley; together they cover the WSD
  regime and give the principled $\beta$ window ($\eta\mu\ll(1-\beta)/(1+\beta)\lesssim\eta\lambda$).
  The confinement lemma is the more distinctive of the two.
- **P2 — E7 and a minibatch E6.** The remaining Strong-Success item; after Shen/Song, some
  optimizer-level evidence (even NanoGPT-small scale: $\beta$ sweep, mechanism metrics vs
  validation loss; ideally one Muon pre/post-polar run at EoS) is what separates "workshop" from
  "conference" in 2026.
- **P2 — Small text items.** (i) Abstract/intro: one clause scoping the trajectory claims to the
  stochastic/large-LR regime, so the closed-loop remark doesn't read as a retraction. (ii) State
  "raises the river-to-hill ratio" explicitly as a property of the *buffer/stream* (it is, in
  Thm separation's wording, but the abstract's unqualified phrasing invites misreading). (iii)
  Thm (separation) maximizes $|\hat g(\omega)|$ over continuous $\omega$ while Thm (filter) uses
  grid bands — harmless, but align the statements ($\omega=\pi$ is a grid point only for even $T$).
  (iv) One line noting $(2T-1)=T_{\mathrm{eff}}$ when first citing li2026denoise's Theorem 1. (v) When adding
  citations, update `latex/references.bib` and the manifest per `references/README.md`.

**Is the current version "good"?** As the Minimal-Success milestone (idea_v1.md §5): yes — it
delivers items 1–4 of that bar cleanly. As a submission into the mid-2026 conversation: not yet —
it is one closed-loop section, one theorem (T5), one related-work pass, and one experiment (E7)
short. None of the four requires new machinery; the first three require roughly a week of work and
reuse objects already in the paper.

## 6. Recommended execution order

1. Closed-loop proposition pair (threshold + stationary variance) → new subsection or upgraded
   Remark in Sec. River-Valley Dynamics. [theory, small]
2. T5 deterministic filter-first theorem → Sec. Filter-First. [theory, small–medium]
3. Related-work paragraph + repositioned contribution bullet ("we prove the alignment property that
   Shen et al. assume"). [writing, small]
4. Confinement lemma T3′ + stochastic spectrum proposition → generalize Sec. River-Valley.
   [theory, medium]
5. E7 (+ minibatch E6 variant), per idea_v1.md's remaining plan. [experiments, the long pole]
6. Abstract/scope clause and text items. [writing, trivial]

Items 1–2 change what the paper *is* (from "filtering exists and is visible" to "filtering, its
closed-loop consequences, and its Muon consequence, each proved in its regime"); item 3 changes how
it lands; items 4–5 decide the venue tier. §§7–8 (added in a second pass at Leon's request) expand
this into the headline-figure design and the full validation + theorem program.

---

## 7. A Figure-2-style headline visualization (proposed E8)

*Question from Leon: can we get the analogue of Shen et al.'s Figure 2 — the one-glance 2D
trajectory picture — showing the benefit of momentum with stronger $\beta$?*

**Answer: yes, and the pilot run confirms it — but only with the right figure design.** Shen et
al.'s Figure 2 works because it is a *regime story* told through trajectory families on one
2D slice (Muon: fast but oscillatory; GD-type: slow but accurate; hybrid: both), with
loss-vs-iteration insets. The naive transplant — a clean deterministic valley with a $\beta$
sweep — would show **moderate $\beta$ winning, not strong $\beta$** (E1's floor distance is
best at $\beta=0.5$), and would hand a reviewer the counter-figure to our own title. The theory of
§3.3 says exactly where "stronger $\beta$ wins" is visible: **under sustained excitation** (noise,
or a bending river that continuously re-forces the hill mode) **at large $\eta\lambda$**. I ran a
pilot on the E2/E3 landscape ($f(x)=2\sin(0.9x)$, $\lambda/\mu=100$; 8 noisy seeds at $\sigma=2$
plus one deterministic control; Claim 5 in `fable_dgs_v1_check.py`) to verify the design before
proposing it:

| panel | setting | $\beta=0$ | $\beta=0.5$ | $\beta=0.9$ | $\beta=0.99$ | $\beta=0.999$ |
|---|---|---|---|---|---|---|
| (a) noisy, $\eta\lambda=1.8$ | hill-offset rms (tail) | $4.40$ | $0.52$ | $\mathbf{0.19}$ | $0.35$ | $0.59$ |
| (a) | mean loss (tail) | $286$ | $3.2$ | $\mathbf{0.50}$ | $2.1$ | $3.9$ |
| (b) $\eta\lambda=2.5>2$ | outcome | diverges $8/8$ | rms $1.99$ | rms $\mathbf{0.25}$, $x_T\!\approx\!4.9$ | rms $0.31$, lags | — |
| (control) same bend, clean ($\sigma=0$) | hill-offset rms | $3.66$ | $0.30$ | $\mathbf{0.0082}$ | $0.28$ | — |

Three findings that shape the figure:

1. **The money panel is (b), the stability extension.** At $\eta\lambda=2.5$, plain SGD diverges on
   every seed while $\beta=0.9$ tracks the bending river to rms $0.25$ and reaches the target
   ($x_T=4.9$ of $x^\star=5$). "GD cannot even run here; momentum can" is the strongest one-glance
   claim we own, it is *predicted* by the $\eta\lambda<2T_{\mathrm{eff}}$ threshold, and it has no
   analogue in Shen et al. (their momentum-free Muon panel plays a different game).
2. **The closed-form is quantitatively honest inside the tube — and momentum is what puts you in
   the tube.** At $\beta=0.9$ the measured rms matches CL-2's prediction to $3\%$ (panel (a):
   $0.189$ vs $0.194$; panel (b): $0.246$ vs $0.231$), while at $\beta=0$ the iterate *escapes the
   linear tube entirely* ($4.40$ measured vs $0.60$ predicted — it bounces across several sine
   periods, loss $286$). So the honest caption is not "momentum shrinks the tube by factor
   $\sqrt{(2-\eta\lambda)/(2-\eta\lambda/T_{\mathrm{eff}})}$" but the stronger "*appropriately
   chosen* momentum ($\beta=0.9$ here) restores the tube-confined regime that river-valley analyses
   (ours and Wen et al.'s) presuppose, and inside it the variance formula is exact." Not generic
   momentum: in panel (b), $\beta=0.5$ is stable yet still partially escaped (rms $1.99$, loss
   $65$); the escape-frequency-vs-$\beta$ curve of §8.2 item 2 is the proper quantification.
3. **The lag edge must be in the figure.** $\beta=0.999$ is strictly worse on every metric and ends
   far behind: $x_T=-0.8$ from $x_0=-3$ — barely a quarter of the river distance covered, versus
   $x_T=3.5$ (over $80\%$) at $\beta=0.9$.
   Including it converts the figure from an advocacy plot into the regime map the theory predicts —
   benefit monotone in $\beta$ until the filter memory $T_{\mathrm{eff}}$ outruns the bend, then
   reversal. That is the filtering–lag tradeoff drawn as trajectories.

One refinement of §3.3(1) that the deterministic control suggests: on a *curved* clean valley,
large $\beta$ wins deterministically too (rms $0.0082$ at $\beta=0.9$ vs $3.66$ at $\beta=0$; E2's
alignment numbers agree). The natural reading is that the bend term $\mathcal O(\eta\dot x f')$
re-excites the hill mode every step, so curvature behaves like sustained forcing and the
"momentum doesn't damp its own transient" caveat is isolated in the *straight-and-clean* valley
(E1). The control does not by itself separate the forcing effect from the homogeneous transient —
a straight-noisy panel or a curvature ($k$) sweep in E8 would — but the hypothesis is consistent
with all four settings we now have (E1, E2, panel (a), control), and the draft's closed-loop remark
should at least stop reading as if it applied to river valleys generally.

**Proposed figure (E8).** Three panels on the curved landscape, each with trajectories over the
loss contour plus a loss-vs-iteration inset, Shen-style: (a) noisy at $\eta\lambda=1.8$,
$\beta\in\{0,0.5,0.9,0.999\}$ — cloud, loose tube, tight tube, lagging tail; (b) $\eta\lambda=2.5$,
$\beta\in\{0,0.5,0.9\}$ — divergence vs tracking, with the CL-2 rms guide; (c) the caveat panel:
the clean *straight* valley, where moderate $\beta$ wins — this is E1's existing result
($d_{\mathrm{rms}}$ minimized at $\beta=0.5$, worst at $\beta=0.99$; note it is *not* the pilot's
curved control, which behaves like (a)) — or, alternatively, the $\beta$-sweep of tail rms against
the CL-2 curve. Implementation is cheap under the codebase standard: one driver
(`scripts/run_e8_headline.py`, reusing `rivervalley_sim` + the E3 noise models, $\ge8$ seeds,
cached run records) and one figure module (`figures/fig_e8_headline.py`); captions declare panels
and guides per WRITING.md, numbers from `metrics.json`. Decision gates: $\beta=0$ diverges at
$\eta\lambda=2.5$ on all seeds; in-tube rms matches CL-2 within $\sim10\%$; lag reversal appears by
$\beta=0.999$. An optional companion (E8b) would render E5 as trajectories in Shen-style spectral
coordinates (slow-signal coefficient vs oscillatory-mode coefficient) for pre-polar vs polar-only —
the matrix/Muon version of the same picture.

## 8. Further plan: theorem program and validation program

### 8.1 Theorem program (statements, proof routes, effort, where they land)

| # | statement (informal) | proof route | effort | lands in |
|---|---|---|---|---|
| CL-1 | closed-loop hill stability iff $\eta\lambda<2T_{\mathrm{eff}}$; flip (Nyquist) mode at threshold; poles $\sqrt\beta$ in the complex regime | Jury conditions on the $2\times2$ companion map already in the draft's remark | trivial | Sec. River-Valley, remark → proposition; cite Cohen 2021 + arXiv:2604.14108 for the known sharpness form |
| CL-2 | stationary hill variance $\operatorname{Var}(y)=\frac{\eta\sigma^2}{\lambda(2-\eta\lambda/T_{\mathrm{eff}})}$, monotone $\downarrow$ in $\beta$; river mode contracts at $1-\eta\mu$ to first order, $\beta$-free | AR(2) stationary variance / $2\times2$ discrete Lyapunov; eigenvalue perturbation for the slow mode; $d$-dim corollary $\sum_i\frac{\eta\sigma^2}{2(2-\eta\lambda_i/T_{\mathrm{eff}})}$ against Wen's $(d{-}1)\eta\sigma^2/2$ | small | same section; supplies the guides for E8(a,b) |
| T5 | filter-first for the deterministic hill: with $G_t=S+(-1)^tA$, pre-polar subspace error $\lesssim \frac{\|A\|/T_{\mathrm{eff}}}{\sigma_r(S)-\|A\|/T_{\mathrm{eff}}}\to0$, while polar-only is stuck at $\approx\frac{\|A\|}{\sigma_r(S)-\|A\|}$ (and fails outright once $\|A\|\ge\sigma_r(S)$); post-polar limit computable exactly (EMA of a period-2 sequence of polar factors $\to$ midpoint of $\mathcal O(S{+}A),\mathcal O(S{-}A)$ as $\beta\to1$ in steady state, generically not orthogonal) | exact EMA-of-Nyquist form (§3.1) + Weyl for the head + Wedin; post-polar via the closed-form limit, worked rank-1/2$\times$2 example for the lower bound | small–medium (pre-polar mechanical; post-polar example needs care) | Sec. Filter-First, argued → theorem |
| T2 | sampled hill-gradient spectrum $S(\omega)=\sigma^2\frac{|1-e^{-i\omega}|^2}{|1-(1-\eta\lambda)e^{-i\omega}|^2}$ (zero DC, cutoff $\eta\lambda$, Nyquist-peaked iff $\eta\lambda>1$ *[erratum: Nyquist-**concentrated** iff $\eta\lambda\ge1$; argmax is always $\pi$ — see §3.2(a) erratum]*); corollary: river-to-hill band ratio improves by $T_{\mathrm{eff}}$–$T_{\mathrm{eff}}^2$, and the $\beta$ window $\eta\mu\ll(1-\beta)/(1+\beta)\lesssim\eta\lambda$ | closed-loop transfer of white noise through the stable AR(1); stationarity by construction | small | Sec. River-Valley, stochastic subsection; unifies with Wen et al.'s regime |
| T3′ | confinement lemma: confined coordinate $\Rightarrow$ DC mass of its gradient stream bounded ($\le 2R/\eta+\frac{\beta}{1-\beta}\lvert m_T\rvert$) while a traveled coordinate's DC mass grows $\propto D_T/\eta$; band version under a mixing/autocorrelation-time assumption | telescoping $+$ Prop. (transfer) at $\omega=0$; band version: spectral mass bound below $\omega_0\ll1/\tau$, with T2 as the concrete instance | small (DC) / medium (band) | anchor lemma of Sec. River-Valley; the assumption-light generalization |
| T6 (opt.) | optimal-$\beta$/schedule corollary: minimize CL-2 floor $+$ lag bias on a bending river; motivates increasing-$\beta$ and the window above | combine CL-2 with the $\varepsilon_{\mathrm{low}}$ lag term of Thm (filter)(b) on the curved model | medium | Discussion, possibly formal |

Deliberately *not* attempted: a clean curved-valley closed-loop theorem (idea_v1 failure mode 5
already accepts a local approximation — E8 makes it empirical instead), and a full nonlinear Muon
trajectory theorem (out of reach; the buffer-level T5 plus E5/E8b is the defensible scope). One
optional remark-level item suggested by the pilot: a "tube-restoration" statement (momentum keeps
the escape probability small exactly when CL-2's rms is small against the tube radius) — I would
validate this empirically (escape frequency vs $\beta$) rather than prove it.

### 8.2 Validation program (experiments, gates, falsifiers)

1. **E8 headline figure** (§7). *Gates:* divergence/tracking split at $\eta\lambda=2.5$; in-tube rms
   vs CL-2 within $\sim10\%$; lag reversal by $\beta=0.999$. *Falsifier:* if in-tube rms misses
   CL-2 badly, the noise-entry model (noise added to the gradient vs to the iterate) or the EMA
   convention is wrong somewhere — stop and reconcile before any theorem lands in the paper.
2. **E3 extension — CL-2 heatmap.** Grid over $(\eta\lambda,\beta)$: measured/predicted stationary
   rms, plus escape frequency; a $\lambda/\mu$ sweep to test the T2/T6 prediction that the good-$\beta$
   window *widens with valley conditioning*. Reuses `run_e3` machinery. *Gate:* ratio $1.0$ within
   seed error in the tube-confined cells.
3. **E5 extension — T5 constants.** Overlay $\sigma_{r+1}(M_t)$ against the $\|A\|/T_{\mathrm{eff}}$
   guide and $\sin\Theta$ against the Wedin bound (the E4 treatment, applied to E5). *Gate:*
   constants track within a factor $\sim2$ (transient $(-\beta)^t$ term included).
4. **E6 extension — confinement onset.** HFER on early-vs-late training windows of the existing E6
   stream (pre- vs post-river-entry), testing the reconciliation of threat (ii): high-frequency
   content is signal *before* confinement, nuisance *after*. Likely needs no new runs, only a
   windowed re-analysis of the cached E6 arrays.
5. **E7 optimizer-level tests** (unchanged from idea_v1; the last Strong-Success gate): $\beta$
   sweeps with performance *and* mechanism metrics on the toy + a small NN, ideally one Muon
   pre/post-polar run at EoS; the claim to test is that mechanism metrics predict performance
   better than $\beta$ alone.
6. **Paper integration.** Port CL-1/CL-2/T5 (+T2/T3′ as they land) and E8 into `latex/main.tex`
   with the §5 P0 related-work pass; update `latex/references.bib` and the references manifest.

Suggested order: (CL-1, CL-2, E8) as one sitting — they form a self-contained
"closed-loop section + headline figure" unit; then (T5, E5-overlay); then (T2, T3′, E3-heatmap);
then E6-onset; E7 last and longest. Items 1–3 of this list are what §6's items 1–2 look like once
made concrete.

---

*Files touched: this document; `discussions/fable_dgs_v1_check.py` (verification script + captured
output, Claims 1–5: closed-loop threshold bracketing; AR(2) variance vs formula, 3-digit agreement;
hill-spectrum band ratios $1.00\pm0.04$; exact EMA-of-Nyquist form, machine precision; E8 pilot,
8 seeds on the curved noisy valley); `references/edge-of-stability/` (two PDFs added);
`references/README.md` (manifest backfilled for all nine PDFs). Codex second-opinion review:
round 1 (§§0–6) PASS with minor nits, applied; round 2 (§§7–8) FAIL with two blocking findings
(a mislabeled evidence row and a false "net backward" reading of $x_T=-0.8$), both fixed along
with three softer findings; round 3 confirmation PASS.*
