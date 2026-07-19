# Project Criterion (latex/CRITERION.md)

The thin, paper-specific child of `WRITING.md`. It holds only what the general standard cannot. The general standard is the parent: where this file is silent, the parent applies; where they conflict, **this file wins**. Keep it short — add an entry only when a rule is genuinely
project-specific.

## Canonical-term table
One phrasing per concept; use it everywhere.

| Concept | Canonical phrasing | Where defined |
|---------|--------------------|---------------|
| slow useful direction | river direction | Background (river-valley landscapes) |
| steep oscillatory direction | hill direction | Background (river-valley landscapes) |
| coupled momentum recursion | EMA momentum $m_t=\beta m_{t-1}+(1-\beta)g_t$ | Background (streams), Def. (ema) |
| EMA frequency response | magnitude response $|H_\beta(\omega)|$ | eq. (transfer) |
| EMA's equivalent uniform-average size | effective averaging length $\Teff=\tfrac{1+\beta}{1-\beta}$ | Rem. (teff) |
| finite-run frequency transform | windowed transform $\hat s(\omega)$; windowed DFT on the grid $\omega_\ell=2\pi\ell/T$ | Background (windowed DFT) |
| pure single-frequency stream | tone $e^{i\omega t}v$ (real form $A\cos(\omega t+\varphi)$) | Background (geometric modes) |
| constant component | DC ($\omega=0$) | Background (geometric modes) |
| fastest expressible oscillation | Nyquist frequency $\omega=\pi$, the stream $(-1)^t$ | Background (geometric modes) |
| exact EMA response to one tone | per-tone identity, eq. (pertone) | Background (tone response) |
| filter before the nonlinear map | filter-first (principle) | Sec. Filter-First |
| filter-then-orthogonalize | pre-polar | Sec. Filter-First |
| orthogonalize-then-filter | post-polar | Sec. Filter-First |
| no-momentum orthogonalization | polar-only | Sec. Filter-First |
| hill-energy attenuation | hill suppression ratio (HSR) | Measurements |
| noise residual after filtering | noise suppression ratio (NSR) | Measurements |
| fraction of high-band energy | high-frequency energy ratio (HFER) | Measurements |
| high-band momentum/raw ratio | momentum suppression ratio (MSR) | Measurements |
| attenuation vs lag balance | filtering--lag tradeoff | Sec. Discussion |
| filter on a given stream vs poles of the driven recursion | open-loop filtering vs closed-loop damping | Sec. Closed-Loop Consequences (Rem.) |
| hill mode of the driven recursion | underdamped hill mode, closed-loop poles of modulus $\sqrt\beta$ | Sec. Closed-Loop Consequences (Rem.) |
| large-LR oscillation regime | edge of stability | Background (river-valley landscapes) |
| momentum-shifted stability bound | closed-loop stability threshold $\eta\lambda<2\Teff$ | Prop. (stability) |
| noise-driven hill floor | stationary hill variance $\operatorname{Var}(y_\infty)$ | Prop. (variance) |
| assumption-light separation | confinement lemma (DC mass) | Lem. (confinement) |
| leaving the valley tube | tube-escape frequency | Measurements |
| admissible momentum range | $\beta$ window $\eta\mu\ll1/\Teff\lesssim\eta\lambda$ | Rem. (window) |
| monotone-spectrum filtering bound | white-noise ceiling $\operatorname{Var}(m)/\operatorname{Var}(g)\le1/\Teff$ | Cor. (ceiling) |
| minimum confining momentum | confinement floor (smallest β keeping the trajectory in the tube) | Sec. Discussion (E7 schedule arm); measured vs bend frequency in E10 |
| per-conditioning scoring span | traveling horizon $3/(\eta\mu)$ | Sec. closed-loop grid |
| optimization form of CL-2 | stationary hill loss $\mathbb{E}[(\lambda/2)y_\infty^2]$ | Cor. (hillloss) |
| size of the β improvement | maximal relative reduction $\eta\lambda/2$ | Cor. (hillloss) |
| time to stationarity | burn-in horizon $(\Teff/2)\log(1/\varepsilon)$ | Rem. (burnin) |
| finite-window boundary size | boundary weight $\Ocal(\Teff^2/T)$ | Rem. (boundary) |
| filter-first beyond Nyquist | band-limited filter-first | Thm. (bandlimited) |
| step conventions | EMA normalization (fixed $\eta$) vs heavy-ball normalization (fixed $\eta_{\mathrm{HB}}=\eta(1-\beta)$) | Background (tone response, step conventions); ablation E9 |
| per-step river displacement | river speed $\vriv$ | Measurements |
| distance to the bending floor | hill offset $d_t=y_t-f(x_t)$ | Background (river-valley landscapes) |
| curvature cost on traversal | river-speed collapse | Sec. E10 |
| closed-loop response per frequency | forced-response gain $\lvert G_\beta(\omega)\rvert=\eta(1-\beta)/\lvert p(e^{i\omega})\rvert$ | Prop. (forced) |
| underdamped-mode angle | hill-mode frequency $\theta_\beta$ | Prop. (forced) |
| bend-sharpened curvature | local hill sharpness $\lambda_{\mathrm{loc}}=\lambda(1+f'^2)$ | Prop. (curved) |
| curvature's passband cost | tracking offset $f'\phi'/\lambda_{\mathrm{loc}}$ | Prop. (curved) |
| mean-loss gradient along visited iterates | state stream (large-batch probe $\bar g^{\mathrm{LB}}$) | Measurements; Sec. E12 |
| noise part of the mini-batch stream | sampling residual $\xi^{\mathrm{res}}$ | Measurements; Sec. E12 |
| where the high band lives | high-band state share | Measurements; Sec. E12 |
| hill directions at scale | curvature subspace (top-$k$ restricted-Hessian eigenvectors) | Measurements; Sec. E12 |
| energy-hypothesis filter-first | band-energy filter-first, low-fraction bound $\gamma$ | Cor. (bandenergy) |
| tone falsification arms | forced-disturbance controls | Sec. E13 |
| spectral maps covered by the buffer theorems | monotone singular-value map $\Ubf\,h(\Sigmabf)\Vbf^\top$, $h\ge0$ nondecreasing | Rem. (mapscope) |
| interpolation between raw buffer and polar factor | power family $\sigma\mapsto\sigma^{1-q}$ | Rem. (mapscope) |
| strength labels on claims | statement classes: exact / quasi-static / reading / measured | Sec. Intro (Statement classes); Table (map) |
| linear-floor exact curved solution | modal solution: two straight loops at the Hessian eigenvalues $\nu_1\ge\nu_2$ | Prop. (curved)(b) |
| status of the pointwise curved threshold | frozen diagnostic (necessary, not sufficient); exact boundary $\eta\nu_1<2\Teff$ | Rem. (frozenexact) |
| fixed-$\eta_{\mathrm{HB}}$ sweep result | heavy-ball normalization corollary; optimum $\beta^\star=\max(0,\sqrt{\eta_{\mathrm{HB}}\lambda}-1)$ | Cor. (hb) |
| two-level Muon register | buffer theorems / update theorems / necessity example | Sec. Filter-First (buffer, update) |
| operator practical Muon runs | deployed Newton--Schulz operator $\Dcal_{\mathrm{NS}}$ | eq. (deployed) |
| scalar action of the deployed operator | NS response $h_{\mathrm{NS}}=\psi^{\circ K}$, slope bound $L_h=\sup_{[0,1]}\lvert h_{\mathrm{NS}}'\rvert=\psi_1^K$ | eq. (deployed); Thm. (nsupdate) |
| annulus hypothesis of the NS theorem | Frobenius floor $\kappa$ | Thm. (nsupdate) |
| amplitude below which NS transfer is tight | the knee ${\sim}(\kappa+\varepsilon_{\mathrm{NS}})/(2L_h)$ | Rem. (knee) |

## Banned constructions / legacy names (remove on sight)
- AI-like editorial (WRITING.md B4): "comprehensive", "thorough", "clearly demonstrates",
  "as expected", "beautifully matches", "matches exactly", figure-reading verbs.
- Do not coin slash-compounds or decorative adjectives for the named quantities above.
- "Experiments confirm each/every prediction" and cognates (review_v5 R6): experiments
  *measure* or *test* predictions; agreement is stated per prediction with its numbers.
- "the reduction is exact" for any curved-valley statement other than Prop. (curved)(b)'s
  linear-floor modal solution; the quasi-static part is always labeled as such.
- Never quote a percentage against a baseline cell containing a diverged seed
  (divergence-counted criterion, Measurements ¶ Statistics).

## Register rules (project style)
- Fix $T_{\mathrm{eff}}=\tfrac{1+\beta}{1-\beta}$ and $|H_\beta(\pi)|=\tfrac{1-\beta}{1+\beta}$ as the
  canonical forms; do not let equivalent rewrites drift in.
- Momentum coefficient is always $\beta$; high/low band cut frequencies $\omega_c,\omega_0$.
- Spell it "underdamped" (no hyphen); closed-loop hill poles have modulus $\sqrt\beta$ (fixed form).
- Cross-reference with `\Cref` consistently (theorems, propositions, remarks, figures, equations,
  sections); refer to experiments by their canonical labels E1--E13 (plus the closed-loop grid).
- Matrix-valued streams and their factors are bold ($\Gbf_t,\Sbf,\Abf_t,\Mbf_t,\Xibf_t,\Ubf,\Sigmabf,\Vbf$);
  the polar factor is written $\polar(\cdot)$; $\Ocal$ is big-O only.
  Exception: the $\Hbb$-valued generic objects $g_t,m_t,\tilde m_t$ stay italic even when instantiated
  on matrices --- bold marks objects *defined* as matrices.
- The hat is reserved for the windowed transform $\hat s(\omega)$; estimator streams carry bars/tags
  ($\bar g^{\mathrm{LB}}$, $\xi^{\mathrm{res}}$); PSDs are $S_g(\omega)$ with the subscript naming the stream.
- Fixed scalar names: closed-loop scalar $\chi=\eta(1-\beta)\lambda$; hill transient $q_{\mathrm{h}}=\eta\lambda-1$;
  $\beta=0$ hill factor $a_0=1-\eta\lambda$; forcing stream $u_t$; hill offset $d_t$; river speed $\vriv$;
  bend amplitude $a_{\mathrm{b}}$; Markov step fraction $\tau$; generic stream dummy $s_t$.
- New fixed symbols (2026-07-19): linear-floor Hessian eigenvalues $\nu_1\ge\nu_2$; modal weights
  $b_1,b_2$ (statement-scoped in Prop. (curved)); NS polynomial $\psi$ with coefficients
  $(\psi_1,\psi_3,\psi_5)=(3.4445,-4.7750,2.0315)$, iteration count $K$ ($=5$ deployed), guard
  $\varepsilon_{\mathrm{NS}}=10^{-7}$, response $h_{\mathrm{NS}}$, slope bound $L_h$, Frobenius
  floor $\kappa$. The NS deployed slope is quoted as $\psi_1^K\approx485$ (not 484.88...).

## Known-dead cross-reference labels (maintained during structural edits)
- (none pending --- 2026-07-18: `sec:river` was split into `sec:spectrum` + `sec:loop` and
  `sec:filter`'s preliminaries moved to `sec:background`/`sec:modes`/`sec:dft`/`sec:ema`;
  every `\Cref` was retargeted in the same edit batch.)
