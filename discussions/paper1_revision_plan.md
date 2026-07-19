# Paper 1 revision plan — positioning edits for main.tex

**Status:** plan, not yet executed. Drafted 2026-07-07 from the idea_v3 / Shen-et-al. discussion.
**Inputs:** `latex/main.pdf` (2026-07-04 build), `discussions/idea_v3.md`,
`references/river-valley/Towards Understanding the Power and Limits of the Muon Optimizer- A River-Valley Perspective.pdf` (Shen et al., arXiv:2606.21514, cited as [10]).
**Governing standards once edits land in `latex/`:** `WRITING.md` + `CRITERION.md` (CRITERION wins on conflict).

## 1. Goals and non-goals

Goals — three small positioning edits:

1. Engage Shen et al.'s **limit-side** results (late-stage overshoot under the polar map), not only their early-stage momentum assumption.
2. Plant the **direction-map-scope remark**: the buffer-level filter-first theorems hold for any monotone singular-value map, of which the polar factor is the extreme member. This is the bridgehead the scheduler paper will lean on.
3. One sentence each in Discussion (schedules) and Limitations connecting the temporal-axis schedule to the spectral-axis schedule.

Non-goals (explicitly out of scope for this revision):

- No new theory, no new experiments, no new numbers beyond citation-sourced facts from [10].
- Abstract and Contribution bullets untouched.
- No restructuring; no changes to Sections 3–4 or 6–9.
- No α/partial-orthogonalization content beyond the scope remark — trajectory-level claims stay in the scheduler paper.

Estimated diff: ~15–20 lines of `main.tex`, 1 optional bib entry, 2 rows in `CRITERION.md`.

## 2. Edit E1 — Related Work, river-valley paragraph

**Location:** §2, river-valley paragraph, appended after "…the landscape emits a frequency-separated stream and the EMA passes the river while attenuating the hills."

**Pre-registered draft text** (adjust flow at edit time; keep content):

> Their trajectory theorems are momentum-free (a simplified Muon), and their limit result is the
> complement of their early-stage rate: under the polar map the admissible step size shrinks with
> the residual scale, so Muon overshoots near the optimum where gradient descent, whose admissible
> step is residual-independent, keeps refining; switching from Muon to AdamW at a fixed step
> improves final validation loss in their 250M-parameter runs. Read against this paper: they vary
> the direction map across training phases with the temporal filter assumed river-aligned; we fix
> the direction map and analyze the temporal filter, with the alignment they assume derived from
> the landscape's frequency separation. The two controls are independent, and their joint schedule
> is untested (Section 10).

**Fact-verification table (B6 applied to citation claims; page refs are the Shen et al. PDF):**

| Claim in draft | Source |
|---|---|
| simplified Muon removes momentum | p.2 ("whose update removes momentum"); §4 keeps momentum only as assumption 1 (p.7) |
| admissible step shrinks with residual scale | Eq. (13), p.8: safe range `0 < η < 2ay = O(a)` |
| overshoot by second step vs GD residual-independent η₀ = O(1) | Theorem 4.2 (informal), p.8 |
| Muon→AdamW switch, 250M params, better final val loss | §5 + Figure 4, p.9 (1.5k/4k switch; both AdamW peak LRs beat all Muon-only baselines) |

**Style checks:** "polar map"/"direction map" are canonical (§5 terms) — do not write "msign".
No banned editorial (B4). Forward references to Section 10 mirror existing §2 practice.

## 3. Edit E2 — new remark at the end of Section 5 (the scope remark)

**Location:** after Remark 8, before the closing paragraph of §5 ("Whenever momentum improves the
signal-to-disturbance ratio before D…"). New `\label{rem:mapscope}`; it will number as Remark 9.

**Pre-registered draft text:**

> **Remark 9 (Scope of the direction map).** Theorems 4 and 5 and Corollary 3 constrain the buffer
> M_t, the input to D; they use the polar map only through two facts: O(M_t) = UV^⊤ shares M_t's
> singular subspaces, and the signal is read out of M_t's singular-value ordering (Proposition 6).
> Both facts hold for every map M_t = UΣV^⊤ ↦ U h(Σ) V^⊤ with h nonnegative and nondecreasing
> applied entrywise to the singular values: such a map carries M_t's singular subspaces per index
> and maps a descending singular-value sequence to a descending one, so the subspace-identification
> conclusions transfer verbatim, with ties read from M_t's input ordering exactly as in
> Proposition 6. The polar factor is the flattest member, h(σ) = 1; h(σ) = σ recovers the raw
> buffer up to the step size; the fractional powers h(σ) = σ^{1−u}, u ∈ (0,1), interpolate.
> Trajectory-level consequences of the interpolation are outside this paper's scope (Section 10).

**Math nuances to preserve at edit time:**

- h nondecreasing (not strictly increasing) suffices: ordering is preserved weakly; ties/collapse
  (h constant at the polar end) are handled the same way Prop 6 already reads Muon — the rank-r
  readout relies on the **input** ordering of M_t. The draft sentence covers this; do not drop it.
- **Symbol choice:** exponent symbol is a problem — α (Thm 5 high-band amplitude), p (AR(2)
  characteristic polynomial), s (forcing tone), a, c, θ, r are all taken. Draft uses **u**, which
  appears only as a proof-local dummy index. Flag to Codex; if it objects, spell the family in
  words and drop the formula.
- Optional citation: single-step Shampoo with exponent −1/4 per side is exactly the polar factor,
  and fractional exponents give the σ^{1−u} family. If we name Shampoo, add
  `references.bib` entry (Gupta, Koren, Singer, ICML 2018). Decision at edit time; naming it is
  honest lineage, omitting it keeps the remark self-contained. Default: **cite it**.

## 4. Edit E3 — Discussion and Limitations, one sentence each

**E3a, §10 schedules paragraph**, appended after "…among floor-respecting schedules the gradual
ramp remains the best arm." The paper already says "a confinement question, not a clock question",
so this diction is native:

> Shen et al. [10] schedule the complementary control — the direction map, switched from the polar
> factor to AdamW at a fixed step; whether that switch is also a confinement-type question, and how
> the two schedules compose, is untested.

**E3b, Limitations paragraph**, extend the final sentence's list:

> Production scale, Adam-preconditioned updates, learning-rate schedules, and trajectory-level
> behavior of the Remark 9 family remain untested, and are the runs the Integration Contract makes
> cheap.

## 5. CRITERION.md updates

Add to the canonical-term table:

| Concept | Canonical phrasing | Where defined |
|---|---|---|
| spectral maps covered by the buffer theorems | monotone singular-value map U h(Σ)V^⊤, h ≥ 0 nondecreasing | Sec. Filter-First (Rem. mapscope) |
| interpolation between raw buffer and polar factor | power family σ ↦ σ^{1−u} | Sec. Filter-First (Rem. mapscope) |

Known-dead-label list: expected empty (no structural moves). Verify anyway after edit.

## 6. Mechanics and acceptance checklist

Order of operations:

1. Edit `latex/main.tex` (E1, E2, E3a, E3b), `latex/references.bib` (Shampoo entry if kept),
   `latex/CRITERION.md` (term rows).
2. `cd latex && make` (Tectonic); confirm clean compile and that every touched `\Cref` resolves.
3. Run the WRITING.md Part C audit on the four touched paragraphs (terms defined, no editorial,
   no duplication, refs resolve, canonical spellings, numbers traced — here: the four [10] facts).
4. `codex review` on the diff, judged against `latex/WRITING.md` + `CRITERION.md`. Resolve
   blocking findings or surface them with reasons.
5. Rebuild `main.pdf`; confirm page count/remark numbering; done.

Acceptance criteria:

- [ ] All four [10]-sourced facts match the table in §2 above.
- [ ] Remark 9's mathematical claim is exactly the two-facts transfer, with the tie/ordering caveat.
- [ ] No new symbols colliding with the existing symbol table; u flagged in the Codex request.
- [ ] Abstract, contributions, Table 1 unchanged.
- [ ] Codex findings resolved or surfaced.
