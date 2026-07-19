# Paper-Writing Criterion (latex/WRITING.md)

A reusable, project-agnostic standard for writing a theory + experiment paper. It has two layers: **macro structure** (the skeleton of the document) and **micro prose** (sentence, caption, and remark rules). Both are distilled from a paper that paired rigorous theory with
empirical validation.

---

## Part A — General structural principles

A handful of invariants that hold for any research paper, regardless of area. They constrain *how ideas relate*, not which sections exist — a theory paper, an empirical study, and a systems paper each satisfy them with different layouts. This file does **not** prescribe a section
skeleton; a genre-specific example lives in the Appendix, to be used only if it fits.

1. **Gap before contribution.** Establish what is missing or unresolved before stating what you add. The reader should feel the gap before they read the fix.
2. **Evidence near its claim.** Keep each claim adjacent to the evidence that supports it — do not bunch all support at the end. The reader should never hold a claim in mind across many pages to find its justification.
3. **Define before use.** Introduce every term, symbol, and object before it appears in an argument, and interpret it in plain words. Name it once, then reuse that name.
4. **State limitations explicitly.** Say what the work does *not* establish and what an extension would require. Flag a boundary in place rather than letting the reader find it.
5. **One canonical structure per recurring element.** Whatever repeats — a result block, a benchmark row, a related-work cluster — give it one consistent shape and keep it. Do not let parallel things drift into different forms.

---

## Part B — Micro prose rules

### B1. No invented terms
Use only terms already defined in the paper or its glossary/appendix. Do not coin new
compounds on the fly (e.g. slash-compounds, decorative adjectives). Maintain the
canonical-term table in the project `CRITERION.md`; if a concept needs a name, add it there
once and use it everywhere.

### B2. Single source of truth
Each fact, definition, and number lives in **exactly one place**; everything else points to it.
Do not restate a formula once it is defined — name the quantity and cite the section. If
`R(T)` is defined in the body, captions and other remarks write `R(T)`, never the formula.

### B3. Caption / body / remark — division of labor
- **Caption = declarative.** Data source (layer, step, sample size), each panel's metric by
  canonical name or formula, the axis variable, dashed reference curves, the meaning of
  color/marker conventions, and error-bar definitions. One short reading guide clause is fine.
- **Body = interpretive.** What the reader should conclude — the validation claim, the
  qualitative reading, the cross-setting comparison.
- **Remark = load-bearing only.** A definition, an inversion (turning an upper bound into a
  lower bound), a special case, or a pointer to a sharper result. Cut editorial framing.

If you find reading-prose in a caption, move it to the body. If you find a definition in a body
sentence, move it to a Remark.

### B4. No AI-like / chatty editorial
Drop on sight: "comprehensive", "thorough", "quick validation", "clearly demonstrates", "as
expected", "is expected", "beautifully/qualitatively matches", "matches exactly", and
figure-"reading" verbs ("as we read off Figure X", "the trajectory reading mirrors…").

Test: if a sentence states a mathematical fact, keep it. If it frames or editorializes, cut it.
Keep the project's banned-phrase list in `CRITERION.md` and remove repeat offenders on sight.

### B5. Cross-references flow one direction
Figures cite definitions; definitions do **not** forward-list the figures that use them. After
any structural edit, search every `\ref{}` you touched and confirm the label resolves; during a
reorganization, keep a running list of known-dead labels until they are fixed.

### B6. Numerical claims are verified against source data
Every sentence with a specific number is checked against the cached result file
(JSON/CSV/array) **before shipping**. Prefer per-cell measured numbers over narrative
summaries. A number that cannot be traced to a source artifact does not go in the paper.

### B7. Project register / style sheet
Maintain a short list of register rules in `CRITERION.md` and enforce them mechanically — one
phrasing per concept, a fixed convention for recurring symbols, and any banned punctuation or
constructions (for example, a no-semicolon rule, or fixing `T = 1/(1-β)` rather than letting an
equivalent form drift in). The general rule is consistency: vary nothing that has a canonical
form.

---

## Part C — Pre-ship audit checklist
Run before shipping any paragraph, caption, or remark:

1. Does every term appear elsewhere in the paper or its glossary? If not, replace or define.
2. Is any sentence editorial or chatty? If yes, cut.
3. Does any content duplicate a body pointer, a remark, or a caption?
4. Does each caption *declare* what is plotted, rather than interpret it?
5. Do all `\ref{}` targets resolve?
6. Is each canonical term spelled the canonical way?
7. Is every number checked against its source artifact?
8. Are the project register/style-sheet rules satisfied (banned phrases, punctuation, symbols)?

---

## Relationship to the project-specific `CRITERION.md`
This file is the parent standard. Each paper keeps a thin child file holding only:

- the **canonical-term table** (concept → exact phrasing → where defined),
- the **legacy / banned-construction list** (names never to reintroduce, phrases to cut),
- the **known-dead-label list** maintained during structural edits.

When the child drifts from the parent on a general rule, the parent wins.
