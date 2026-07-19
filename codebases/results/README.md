# results/ — record every run

See `../CODING.md` Pillar 3. The tracker (Weights & Biases by default) is the live system of
record; this directory is its offline, reproducible mirror.

- `index/runs.csv` — the run table mirrored from the tracker API (the offline queryable index).
- `cache/<run-id>/` — cached metrics + artifacts per run, so figures build offline.
- `figures/<label>/` — per-figure provenance: a `summary.md` (what it plots, feeding run-ids,
  reproduce command, caveats) plus `data/` and `data/history/` for superseded vintages.

Unified run name: `<task>_<probe>_<key-params>_<git-sha8>`. Figures read `cache/`, never the live
API. On nodes without network egress, log with `wandb offline` and sync afterward.
