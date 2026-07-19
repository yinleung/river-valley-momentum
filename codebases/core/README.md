# core/ — written once, task-agnostic

Shared library imported by every task's `adapter/`. **`core/` never imports tasks** — the dependency points one way. See `../CODING.md` Pillar 2.

- `metrics.py` — the analysis metrics, one definition each (mirror the paper's Measurements appendix).
- `momentum.py` — EMA recursion, transfer function, and the Muon-style polar pipelines.
- `landscapes.py` — river-valley losses and the gradient-noise models (E1–E4, E7, E8).
- `closedloop.py` — theory guides from CL-1/CL-2/CL-3/T2/T5 (`discussions/theory_cl_t2_t5.md`, `discussions/theory_cl3_t5b.md`), one definition per dashed guide curve.
- `probe.py` — trajectory / stationary gradient-collection protocols.
- `logging.py` — the tracker wrapper: unified run naming + config / metric / artifact logging.
