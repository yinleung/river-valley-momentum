# Task: Restructure and notation-audit “Momentum Filters the River” for JMLR-level clarity

You are revising the LaTeX source of the paper:

**Momentum Filters the River: A Temporal–Spectral Theory of Momentum in Optimization**

The primary goals are:

1. Introduce a rigorous, self-contained **Background and Preliminaries** section.
2. Clearly separate standard signal-processing/optimization background from the paper’s original contributions.
3. Perform a paper-wide notation audit and eliminate notation reuse that creates ambiguity.
4. Preserve all valid mathematical claims, proofs, experiments, numerical results, and citations unless an actual inconsistency is found.
5. Improve the paper toward the clarity, self-containment, and theorem organization expected of a JMLR theory paper.

Do not merely perform stylistic rewriting. First understand the complete paper, its theorem dependencies, and the role of each symbol.

---

## Phase 1: Inspect before modifying

Read the full LaTeX project, including:

- the main paper;
- all theorem and proof environments;
- appendices, if any;
- bibliography;
- figure and table references;
- experiment descriptions;
- custom macros.

Before editing, produce an internal modification plan containing:

1. the current section structure;
2. which paragraphs are standard background;
3. which propositions, theorems, corollaries, and remarks are original;
4. theorem dependency order;
5. all symbols with multiple meanings;
6. all hard-coded theorem, equation, figure, table, or section numbers.

Use LaTeX labels and references rather than hard-coded numbering.

Do not change theorem statements silently. If a mathematical issue is discovered, record it explicitly and propose a correction with justification.

---

## Phase 2: Create a Background and Preliminaries section

Insert a new section after Related Work:

# Background and Preliminaries

This section must introduce only standard or previously established material needed to understand the paper. It should be mathematically precise but readable to an optimization researcher without prior signal-processing training.

Use the following subsection structure, adjusting titles slightly if necessary.

### 3.1 Optimization-indexed temporal streams

Define a stream as a sequence indexed by optimization step:

\[
s_1,\ldots,s_T,
\]

where each entry may be a scalar, vector, or matrix.

Explain clearly that the paper studies **temporal frequency across optimizer steps**, not spatial image frequency, Hessian eigenvalues, or matrix singular values.

Add a short paragraph distinguishing:

- temporal frequency spectrum: indexed by \(\omega\);
- curvature/Hessian spectrum: indexed by \(\lambda_i\);
- matrix singular spectrum: indexed by \(\sigma_i\).

Use these terms consistently throughout the paper.

### 3.2 Discrete-time linear time-invariant systems and geometric modes

Introduce the time-shift operator and explain why geometric sequences

\[
s_t=z_0^t v
\]

are eigenmodes of a one-step shift:

\[
s_{t-1}=z_0^{-1}s_t.
\]

Clarify that this is a probe/eigenmode analysis, not an assumption that an arbitrary gradient stream globally has one geometric form.

Explicitly distinguish:

- \(z_0\): the parameter of one geometric mode;
- \(z\): the variable of a \(z\)-transform or transfer function;
- \(z=e^{i\omega}\): restriction to the unit circle for persistent sinusoidal frequency response.

Explain:

\[
z_0=r e^{i\omega}
\]

with \(r\) controlling growth or decay and \(\omega\) controlling temporal oscillation.

### 3.3 The \(z\)-transform, transfer functions, and the unit circle

Introduce the unilateral or bilateral \(z\)-transform convention used in the paper and use it consistently.

Explain:

- transfer functions;
- poles and zeros;
- region of convergence where relevant;
- why \(|z|=1\) corresponds to persistent frequencies;
- why off-unit-circle points describe growing or decaying geometric modes;
- why stability roots are determined by a characteristic polynomial rather than freely selected.

Keep this subsection concise. Do not turn it into a full signal-processing textbook.

### 3.4 DFT and finite-window frequency analysis

Define the length-\(T\) DFT grid:

\[
\omega_\ell=\frac{2\pi\ell}{T}.
\]

Explain that the grid tones form an orthogonal basis for every length-\(T\) stream. State:

- the DFT;
- the inverse DFT;
- conjugate symmetry for real streams;
- Parseval’s identity;
- DC and Nyquist frequencies;
- folded one-sided frequencies;
- band energy.

Explain that a general gradient stream need not equal \(z_0^t v\); it is represented as a sum of unit-circle DFT modes.

Also explain the finite-window caveats:

- the DFT implicitly represents a periodic extension of the window;
- startup and endpoint effects can create boundary discrepancies;
- tapered windows reduce leakage but alter exact identities.

Do not prove the paper’s novel finite-window EMA boundary identity here. State only the standard issue and refer forward to the paper’s exact result.

### 3.5 EMA momentum conventions

Define EMA-normalized momentum:

\[
m_t=\beta m_{t-1}+(1-\beta)g_t.
\]

Define the optimizer update convention used in the paper and distinguish it from fixed-step heavy-ball normalization.

State the conversion between the two conventions.

Define:

\[
T_{\mathrm{eff}}=\frac{1+\beta}{1-\beta}
\]

and explain its standard variance/mean-age interpretation.

The elementary steady-state EMA transfer function may be derived here, or briefly at the beginning of the next section. Choose one location only; do not duplicate the derivation.

If placed in Background, clearly label it as a standard fact and ensure that the first original result in the next section is the exact finite-window identity.

### 3.6 River-valley landscapes and edge-of-stability behavior

Introduce the straight valley:

\[
L(x,y)=\phi(x)+\frac{\lambda}{2}y^2
\]

and the curved valley:

\[
L(x,y)=\phi(x)+\frac{\lambda}{2}(y-f(x))^2.
\]

Define:

- river coordinate/direction;
- hill coordinate/direction;
- river curvature scale;
- hill curvature;
- condition ratio;
- confinement/tube;
- edge-of-stability regime.

Summarize only prior river-valley results needed later.

Do not include the paper’s novel statements that:

- the hill gradient is Nyquist-concentrated;
- the stationary hill-gradient spectrum has the displayed exact density;
- momentum raises the river-to-hill ratio;
- momentum changes stationary hill loss.

Those belong in the main results sections.

### 3.7 Polar factor and subspace perturbation tools

Briefly define:

- thin SVD;
- polar factor;
- principal-angle/subspace error;
- Weyl inequalities;
- Wedin’s sin-\(\Theta\) theorem.

State only the versions actually used later.

Avoid proving classical results.

### 3.8 Notation summary

Add a compact notation table organized into:

1. optimization dynamics;
2. temporal-frequency quantities;
3. river-valley quantities;
4. matrix/subspace quantities;
5. experimental metrics.

Include symbol, meaning, domain, and first-use section.

---

## Phase 3: Restructure the main technical sections

After Background, organize the paper approximately as follows:

### 4 Momentum as a Temporal Filter

This section should begin the paper’s original technical development.

Suggested subsections:

- 4.1 EMA response and finite-window setup
- 4.2 Exact finite-window transfer identity
- 4.3 Finite-window band-filtering theorem
- 4.4 Boundary term and filtering–lag tradeoff

Do not repeat the full DFT tutorial from Background.

Make explicit which statement is standard and which result is new.

### 5 Spectrum Generated by River-Valley Dynamics

Place together:

- deterministic river/hill frequency separation;
- stationary hill-gradient spectrum;
- white-noise ceiling;
- confinement/DC lemma.

### 6 Closed-Loop Consequences

Place together:

- closed-loop recurrence;
- stability threshold;
- stationary variance and hill loss;
- burn-in;
- forced-response analysis;
- resonance;
- curved-valley reduction;
- beta-window consequences.

### 7 Filter-First Nonlinear Direction Maps

Place together:

- deterministic Nyquist theorem;
- polar-only counterexample;
- band-limited theorem;
- band-energy result;
- scope beyond the polar factor.

Subsequent experiment and discussion sections may retain their current broad organization.

Ensure every theorem is introduced only after all required notation and assumptions have been defined.

---

## Phase 4: Paper-wide notation audit

Create a symbol registry before renaming anything. For each symbol, record every definition and scope.

Prioritize eliminating the following collisions.

### A. Transfer functions and gradient matrices

Current risk:

- \(G_t\): gradient matrix;
- \(G_\beta(\omega)\): closed-loop forced-response gain.

Use visually distinct notation. Recommended example:

- \(\mathbf G_t\): gradient matrix;
- \(\mathcal R_{\beta,\lambda}(z)\) or \(\mathcal C_{\beta,\lambda}(z)\): closed-loop response.

Keep \(H_\beta(z)\) or \(\mathcal H_\beta(z)\) exclusively for the open-loop EMA transfer function.

### B. Signal, spectral density, and transforms

Current risk:

- \(S\): signal matrix;
- \(S_g(\omega)\): spectral density;
- \(S(z)\): transform of the forcing input.

Recommended distinction:

- \(\mathbf S\): signal matrix;
- \(\mathcal S_g(\omega)\): power spectral density;
- \(U(z)\) or \(\mathcal U(z)\): transform of the forcing sequence.

Never use the same plain capital letter for these three roles.

### C. Disturbance amplitude and disturbance matrices

Current risk:

- \(A\): scalar forcing amplitude;
- \(A\): disturbance matrix;
- \(A_t\): disturbance stream.

Recommended distinction:

- \(a_{\mathrm{forc}}\): scalar forcing amplitude;
- \(\mathbf A\): fixed disturbance matrix;
- \(\mathbf E_t\) or \(\mathbf A_t\): disturbance stream, but reserve the chosen symbol globally.

### D. Hill offset and spectral gap

Current risk:

- \(\delta_t\): curved-valley hill offset;
- \(\delta_t\): singular-value separation/gap.

This collision must be removed.

Recommended:

- \(h_t=y_t-f(x_t)\) or \(d_t=y_t-f(x_t)\): hill offset;
- \(\Delta_t^{\mathrm{gap}}\): spectral separation.

### E. Euler’s number versus shorthand scalar

Current risk:

\[
e=\eta(1-\beta)\lambda
\]

appears near \(e^{i\omega}\).

Rename the scalar, for example:

\[
\chi_\lambda:=\eta(1-\beta)\lambda.
\]

Do not use plain \(e\) as an auxiliary optimization parameter.

### F. Decay factor and high-band contraction

Current risk:

- \(\rho=\eta\lambda-1\): hill transient factor;
- \(\rho_{\mathrm{high}}(\beta)\): high-band contraction.

Rename the transient factor, for example:

\[
q_{\mathrm h}:=\eta\lambda-1.
\]

Reserve \(\rho_{\mathrm{high}}\) for filter contraction.

### G. Tone amplitude, river speed, and singular vectors

Current risk:

- \(v\): tone amplitude;
- \(v\): river speed;
- \(V\): right singular vectors.

Recommended:

- \(a\) or \(u\): tone amplitude, according to whether scalar or vector;
- \(v_{\mathrm{river}}\): river speed;
- \(\mathbf V\): right singular-vector matrix.

### H. Rank and radial modulus

Reserve \(r\) for only one persistent meaning. Recommended:

- \(r_{\mathrm s}\): signal rank;
- \(r_z=|z|\) or simply \(|z|\): complex-mode radius.

Do not use \(r\) simultaneously as rank, radius, river tangent, and summation index in nearby derivations.

### I. Noise scale and singular values

Use:

- \(\sigma_\xi^2\): noise variance;
- \(\sigma_j(\mathbf S)\): singular values.

Never write a bare \(\sigma\) where the meaning could be either.

### J. Generic stream versus river coordinate

Do not use \(x_t\) for both a generic temporal stream and the river coordinate.

Recommended:

- \(s_t\): generic stream;
- \(x_t\): river coordinate.

### K. Matrix dimensions versus momentum

Avoid dimensions \(m,n\) when \(m_t\) is the momentum buffer.

Recommended:

\[
\mathbf G_t\in\mathbb R^{d_1\times d_2}.
\]

### L. Direction map versus displacement

Use:

- \(\mathcal D\): nonlinear direction map;
- \(\Delta_T\): net displacement.

### M. Hilbert space versus transfer function

Use:

- \(\mathcal X\) or \(\mathscr H\): ambient Hilbert space;
- \(H_\beta\) or \(\mathcal H_\beta\): transfer function.

Ensure visual distinction is clear in the chosen font.

### N. Frequencies and thresholds

Keep:

- \(\omega\): temporal frequency;
- \(\omega_\ell\): DFT grid frequency;
- \(\theta_\beta\) or \(\omega_{\mathrm{mode},\beta}\): closed-loop modal frequency.

Do not reuse \(\theta\) for the “fraction of bad steps” in the band-energy corollary. Rename that probability/measure parameter to \(\varepsilon_{\mathrm{step}}\), \(\delta_{\mathrm{step}}\), or similar.

### O. Observation length

Reserve \(T\) for the finite observation/training window and \(T_{\mathrm{eff}}\) for EMA effective memory.

When discussing notation from prior work that uses \(T\) differently, rename the imported quantity locally and explain the conversion.

---

## Phase 5: Add conceptual clarification boxes or remarks

Add brief, formal clarifications at appropriate places.

### Clarification 1: A geometric mode is not a global gradient model

State:

> The calculation with \(g_t=z_0^t v\) characterizes one eigenmode of a linear time-invariant operator. A general finite gradient stream need not have this form; the DFT represents it as a sum of unit-circle modes.

### Clarification 2: Three roles of \(z\)

State the distinction between:

1. mode parameter \(z_0\);
2. transform variable \(z\);
3. frequency-response point \(z=e^{i\omega}\).

### Clarification 3: Open-loop versus closed-loop

State:

- \(H_\beta(z)\) describes the map from a prescribed gradient stream to the EMA buffer;
- the optimizer trajectory requires the combined loss–optimizer closed-loop recurrence;
- inertia, ringing, and stability are determined by closed-loop roots, not by the open-loop EMA response alone.

### Clarification 4: Three spectral domains

State explicitly that temporal frequency, Hessian curvature, and matrix singular values are distinct axes connected by the theory but not interchangeable.

---

## Phase 6: Preserve and verify the mathematics

After restructuring:

1. verify every equation after symbol renaming;
2. verify dimensions of all vector and matrix expressions;
3. verify all conjugates in DFT orthogonality and real-stream symmetry;
4. verify all norms and inner products;
5. verify every use of Parseval, Minkowski, Weyl, Wedin, Chebyshev, and Markov;
6. verify the convention for unilateral versus finite-window transforms;
7. verify all initialization and boundary terms;
8. verify the relationship between EMA and heavy-ball normalization;
9. verify that all asymptotic statements specify the order of limits;
10. verify all stationary claims list their stability and noise assumptions.

Do not weaken or strengthen theorem claims merely for prose smoothness.

If an assumption is needed but currently implicit, add it explicitly.

---

## Phase 7: Avoid duplication and excessive tutorialization

The Background must be self-contained but economical.

Do not repeat:

- the full tone derivation in both Background and Section 4;
- DFT definitions in multiple later sections;
- the open-loop/closed-loop distinction in several long paragraphs;
- river-valley definitions each time a new result is introduced.

After a concept is defined in Background, later sections should refer back to it.

Move lengthy standard algebra to an appendix if it interrupts the main theorem flow, while retaining the essential intuition in the main text.

The resulting paper should remain a research article, not become a signal-processing textbook.

---

## Phase 8: Deliverables

Return:

1. the revised LaTeX source;
2. a compiled PDF;
3. a concise changelog;
4. a symbol-renaming table with old and new notation;
5. a section-movement table showing where each original paragraph moved;
6. a list of any mathematical or logical issues discovered;
7. a list of all claims that required added assumptions;
8. confirmation that all labels, references, figures, tables, citations, and equations compile correctly.

---

## Acceptance criteria

The revision is successful only if:

- a reader unfamiliar with DFT and the \(z\)-transform can understand why complex modes are used;
- the reader understands that a general gradient stream is not assumed to equal one \(z_0^t v\);
- the reader understands why frequency response uses the unit circle;
- the reader understands what off-unit-circle modes and closed-loop roots mean;
- standard background and novel results are visually and structurally separated;
- temporal, Hessian, and singular-value spectra cannot be confused;
- every important symbol has one stable meaning within the paper;
- no theorem or experimental conclusion changes accidentally;
- the LaTeX project compiles without unresolved references or notation inconsistencies;
- the final exposition is self-contained and appropriate for a JMLR theory submission.