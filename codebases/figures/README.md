# figures/ — one home for all rendering

See `../CODING.md` Pillar 4.

- `style.py` — rcParams, colors, markers. Applied everywhere; no per-figure deviation.
- `fig_<label>.py` — one **pure** module per figure: `build(records) -> writes out/<label>.{pdf,png}`.
  Reads run records from `../results/cache/`; never recomputes an experiment or hits the tracker.
- `out/` — the single output directory for rendered figures.

Figure module names follow the paper's `\label` keys. Version through git: no `_v2` modules and
no parallel `figures_new*/` directories.
