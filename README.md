# betascheduler

Research project on a **scheduler for the momentum coefficient (β)** through the lens of
**momentum / spectral filtering** — paper writing and experiments in one repo.

> Status: scaffolding. The method is not yet pinned down (see `discussions/`), so
> `codebases/core/` still carries the generic analysis template and will be tailored once
> the idea is fixed.

## Layout

| Path | What it holds | Governed by |
|------|---------------|-------------|
| `latex/` | The paper: `main.tex`, `preamble.tex`, `references.bib`. | `latex/WRITING.md` + `latex/CRITERION.md` |
| `codebases/` | Code, experiments, figures, results. | `codebases/CODING.md` |
| `discussions/` | Research notes and idea drafts (not yet a formal standard). | — |
| `references/` | Source papers. **PDFs are git-ignored**; see `references/README.md`. | — |

## How we work — Claude executes, Codex reviews

This repo is set up for two agents with a strict division of labor:

- **Claude** implements, runs, and edits.
- **Codex** reviews and **never edits files** — it judges changes against the project's own
  standard, not generic taste.

Each subtree carries paired entry points: `CLAUDE.md` (Claude's mode + imported standard) and
`AGENTS.md` (Codex's reviewer rubric). Routing starts at the root `CLAUDE.md` / `AGENTS.md` and
narrows as you descend. **Launch your agent from the subtree you're working in** (`cd latex` or
`cd codebases`) so the right standard loads deterministically.

After any non-trivial change, run `codex review` on the diff against the relevant standard
(`latex/WRITING.md` for prose, `codebases/CODING.md` for code). Resolve blocking findings or
surface them.

## Quickstart

**Compile the paper** — uses [Tectonic](https://tectonic-typesetting.github.io) (one self-contained binary, no TeX Live install):
```bash
cd latex && make        # or: tectonic main.tex
```

**Add an experiment task** — drop the upstream repo into `codebases/<task>/upstream/` (kept
pristine), then have an agent write the `adapter/`, `config.yaml`, and `AGENT_NOTES.md` against
the Integration Contract. Full recipe: `codebases/CODING.md` → "Quickstart: add a new task".

## Data & sharing policy

This repo ships **code, standards, and lightweight provenance — never datasets**. Git-ignored:
datasets and heavy artifacts (`data/`, `*.pt`, `*.npz`, checkpoints, …), experiment-tracker
output (`wandb/`), each task's `upstream/`, the figure cache (`results/cache/`), rendered figures
(`figures/out/`), LaTeX build products, and the copyrighted reference PDFs.

**Tracked** (so figures rebuild and provenance survives): `results/index/runs.csv` and every
`results/figures/<label>/summary.md`. See `.gitignore` for the exact rules.

## Collaborators

Private repo. To get a teammate started: grant them access on GitHub, have them clone, and point
them at this README + the subtree standards. Reference PDFs are fetched individually
(`references/README.md`).
