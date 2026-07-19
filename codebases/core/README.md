# core/ — written once, task-agnostic

Shared library imported by every task's `adapter/`. **`core/` never imports tasks** — the dependency points one way. See `../CODING.md` Pillar 2.

- `metrics.py` — the analysis metrics, one definition each (mirror the paper's Measurements appendix).
- `momentum.py` — EMA recursion, transfer function, and the Muon-style polar pipelines.
- `landscapes.py` — river-valley losses and the gradient-noise models (E1–E4, E7, E8).
- `closedloop.py` — theory guides from CL-1/CL-2/CL-3/T2/T5 (`discussions/theory_cl_t2_t5.md`, `discussions/theory_cl3_t5b.md`), one definition per dashed guide curve.
- `probe.py` — trajectory / stationary gradient-collection protocols.
- `logging.py` — the tracker wrapper: unified run naming (git sha on clean trees, content
  hash on dirty — plan_v5 B3) + config / metric / artifact logging.

GPU campaign modules (plan_v5 §3.1 B2/B5; torch-importing, so NOT eagerly imported by
`core/__init__` — laptop numpy-only figure builds stay torch-free):

- `device.py` — device resolution, seeding, determinism policy, bf16-autocast policy (B2).
- `looprunner.py` — THE instrumented closed-loop runner every task adapter delegates to
  (optimizer conventions ema/hb, divergence discipline, window/sketch/LB/HVP orchestration).
- `spectral_stream.py` — chunk-generated JL sketches, raw-window recorder, stream stats.
- `hvp.py` — HVP operator, block power iteration, full-model Lanczos, sharpness tracker.
- `forced.py` — G3 tone injection + lock-in readout (E13 conventions at network scale).
- `lbprobe.py` — fixed large-batch probe gradient + E12 state/residual band accounting.
