# Research Codebase Organization (codebases/CODING.md)

A reusable layout for a research project that (a) runs the same analysis across many tasks/datasets/models, (b) renders many figures from those results, and (c) must record every experiment under one queryable scheme. It is optimized for two readers: **a future you**, and
**an agent integrating the next task**. Both need legible structure, non-invasive integration, and a single record of every result.

## Guiding principles
1. **Drop-in, don't rewire.** A new task is a repo dropped into `codebases/`, integrated by an agent through a small fixed contract — not by editing shared code.
2. **Pristine upstream.** External code is never edited. Integration lives in an adapter beside it, so upstream stays pullable and "their code" vs "our glue" is always clear.
3. **Write analysis once.** Probe protocols, metrics, and momentum logic live in `core/` and are imported by every task. They are never re-implemented per dataset.
4. **One record of every run.** Every run logs to the tracker under a unified name; figures read a local mirror of that record, never recompute experiments.
5. **Version with git, not filenames.** No `_v2`, no `figures_new2`. One output dir, one naming scheme.

---

## Directory layout

```
codebases/                  # the code subtree root — this standard governs everything here
  CLAUDE.md                 # Claude entry: imports CODING.md + sets "code mode"
  AGENTS.md                 # Codex entry: reviewer role, reads CODING.md as its rubric
  CODING.md              # this standard
  CODEBASE_INDEX.md         # one line per integrated task: name, source, status, date
  <task-name>/              # a dropped-in repo plus its adapter (one folder per task)
    upstream/               # the GitHub repo as-downloaded; NEVER edited
    adapter/                # the ONLY code we author: satisfies the Integration Contract
    config.yaml             # task config (target matrix, batch, steps, sweep grids, ...)
    AGENT_NOTES.md          # agent-authored: entry points, gradient flow, gotchas, smoke test
  core/                     # written once, task-agnostic; imported by adapters
    probe.py                # stationary + trajectory gradient-collection protocols
    metrics.py              # the analysis metrics (one definition each, matching the paper)
    momentum.py             # buffer recursion + the pipeline variants under comparison
    logging.py              # the tracker wrapper: unified run naming + config/metric/artifact logs
  figures/                  # one home for all rendering
    style.py                # rcParams, colors, markers — applied everywhere, no per-figure drift
    fig_<label>.py          # one PURE module per paper figure: run records -> figure
    out/                    # the single output dir for rendered pdf/png
  results/                  # the record-everything layer
    index/runs.csv          # offline index mirrored from the tracker API
    cache/<run-id>/         # cached metrics + artifacts for offline figure builds
    figures/<label>/        # per-figure provenance: data/ + data/history/ + summary.md
  configs/                  # shared / cross-task experiment configs
  scripts/                  # HPC / cluster submission scripts
```

---

## Pillar 1 — `codebases/` and agent integration

### The model
To add a task: drop the repository into `codebases/<name>/upstream/` untouched. An agent then
authors **three things only** — an `adapter/`, a `config.yaml`, and an `AGENT_NOTES.md`. The
upstream tree is never modified.

### The Integration Contract
`core/` calls a small, stable interface that each adapter implements by *wrapping* upstream
code. Keep this contract minimal and fixed — it is the surface an agent targets. A typical
contract for a gradient-analysis project:

| Function | Returns | Purpose |
|---|---|---|
| `build_model(config)` | model on device, weights freezable | load the task's model |
| `data_loader(config)` | iterable of batches | the task's data stream |
| `target_modules(model, config)` | named weight matrices | which matrices to probe |
| `gradient_stream(model, data, target, config)` | yields per-step gradient `G_t` | the core observable |
| `end_to_end_train(model, data, config)` *(optional)* | training curve | full-training comparisons |

The adapter's job is to satisfy these by calling into `upstream/` — never by copying or forking
upstream files.

### `AGENT_NOTES.md` (agent-authored, per codebase)
The agent records what it learned so the next agent starts from notes, not from scratch:

- repository map: entry points, where the training loop lives, the model definition;
- **gradient flow**: how to obtain a target matrix's per-step gradient, and how to freeze weights;
- config knobs that matter (batch shape, sequence length, precision);
- dependency quirks / environment notes;
- the **smoke-test command** that proves the integration works.

### Agent workflow to add a task
1. Drop the repo into `codebases/<name>/upstream/`.
2. Agent analyzes it and writes `AGENT_NOTES.md` (map + integration plan).
3. Agent writes `adapter/` implementing the Integration Contract by wrapping upstream.
4. Agent writes `config.yaml`.
5. **Smoke test**: run the probe for a few steps on one target; confirm a run logs to the
   tracker and a figure can read it back from the local cache.
6. **Register**: add one line to `CODEBASE_INDEX.md`.

### Why non-invasive integration matters
Upstream stays pullable; blame is clean between external code and our glue; and agents integrate
reliably because the contract is small and fixed rather than discovered anew each time.

---

## Pillar 2 — `core/` written once
- `probe.py` — the collection protocols (e.g. a fixed-model **stationary** probe and a sliding
  **trajectory** buffer during training). Identical downstream analysis for every task.
- `metrics.py` — the analysis metrics, each defined exactly once and matching the paper's
  Measurements appendix.
- `momentum.py` — the buffer recursion and the pipeline variants being compared.
- `logging.py` — the tracker wrapper so every task logs identically (see Pillar 3).

**Dependency direction is one-way:** adapters import `core/`; `core/` never imports tasks.

---

## Pillar 3 — `results/`: record every run (tracker + offline mirror)

**Tracker = Weights & Biases** (MLflow is a drop-in alternative for the same role). Every run
logs its full config, its metrics (per step / checkpoint), and its heavy outputs as artifacts.

### Unified run identity
Make every run self-describing and sortable:

- `name = <task>_<probe>_<key-params>_<git-sha8>` — e.g. `nanogpt_stationary_s3000_K500_b3a9f1c2`
- `group = <task>`, `job_type = <probe>`
- `tags = [<task>, <probe>, key flags]`
- `config = ` full parameter dict **plus the git SHA**, so every result ties to a commit.

### Offline mirror (required for HPC + reproducible figures)
Figures must never call the tracker API at render time. A small sync script:

1. pulls the run table to `results/index/runs.csv` (the offline, queryable index), and
2. caches each run's metrics + artifacts under `results/cache/<run-id>/`.

Figures read `results/cache/`. On nodes without network egress, log with `wandb offline` and
sync afterward.

### Per-figure provenance (the human narrative)
Keep a `results/figures/<label>/summary.md` for every figure: what it plots, **which run-ids
feed it**, the exact reproduce command, known caveats, and a `data/history/` holding superseded
vintages. This is the pattern that already proved its worth — pair it with the machine index,
never replace it.

**Net:** the tracker gives live comparison; `runs.csv` gives an offline queryable index;
`summary.md` gives the per-figure story. Three views, one record.

---

## Pillar 4 — `figures/`: one home, one module per figure
- `style.py` centralizes rcParams, colors, and markers; apply it everywhere with no per-figure
  deviation.
- Each paper figure is **one pure module** `fig_<label>.py` exposing `build()`, which takes run
  records from `results/cache/` and writes `<label>.pdf`/`.png` to `out/`. Pure means: no
  experiment recomputation and no network.
- **One naming scheme**, tied to the paper's `\label` keys. Migrate any legacy schemes to it.
- Version through git. Never create `_v2` modules or parallel `figures_new*/` output dirs.

---

## Anti-patterns (lessons to encode)
- **Version suffixes in filenames** (`exp_*_v2.py`, `*_v7_reconstruct.py`) and **parallel output
  dirs** (`figures_new2/`, `figures_cr/`) — git already versions; delete on sight.
- **Coexisting figure-naming epochs** (`fig1…`, `fig_app_*`, `fig_ch*`) — pick one scheme and
  migrate.
- **Analysis re-implemented per dataset** — it belongs in `core/`, written once.
- **Editing upstream code to integrate** — wrap it in `adapter/` instead.
- **Provenance only in prose** — pair every `summary.md` with the machine index / tracker.
- **Renderers that recompute experiments** — figures read records; they never run models.
- **No global index** — if finding a run means opening figure folders one by one, add the index.

---

## Quickstart: add a new task (agent prompt)
> A repo is in `codebases/<name>/upstream/`. Read it and write `AGENT_NOTES.md` mapping its
> entry points and gradient flow. Then implement `codebases/<name>/adapter/` against the
> Integration Contract in `../CODING.md` (the code standard at the `codebases/` root),
> wrapping upstream without editing it. Write
> `config.yaml`. Run the smoke test, confirm a run logs to the tracker and a figure reads it
> from `results/cache/`, then add a line to `CODEBASE_INDEX.md`.

## Quickstart: add a new figure
1. Create `figures/fig_<label>.py` with a pure `build(records) -> writes out/<label>.{pdf,png}`.
2. Pull the feeding runs into `results/cache/` via the sync script.
3. Write `results/figures/<label>/summary.md` (what it plots, run-ids, reproduce command, caveats).
4. Render; commit the module and the provenance together.
