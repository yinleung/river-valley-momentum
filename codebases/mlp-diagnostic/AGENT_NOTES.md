# mlp-diagnostic — agent notes

First-party task (no upstream repo) for Experiment E6: a real-network trajectory-gradient frequency
diagnostic. The model and the Integration Contract both live in `adapter/contract.py`; `core/` and
the experiment driver stay framework-free and call the contract.

## Repository map
- `adapter/contract.py` — the whole task:
  - `TwoLayerMLP` — `fc2(relu(fc1(x)))`, double precision, CPU.
  - `build_model(config)` / `data_loader(config)` / `target_module(model, config)` / `full_loss`.
  - `make_step_fn(model, data, target, config)` → `step_fn()` runs ONE full-batch GD step and
    returns the probed layer's gradient as numpy (the contract surface `core.probe` drives).
  - `val_data_loader(config)` — held-out batch (seed+101), used by E7 for reference val loss.
  - `end_to_end_train(model, data, target, config, opt)` — the Contract's optional end-to-end
    member (E7): closed-loop training under `{"kind": "sgdm", "beta": b}` (every parameter
    EMA-SGDM at `config["eta"]`) or `{"kind": "muon", "beta": b, "variant": "pre"|"post",
    "eta_o": lr}` (target matrix takes the orthogonalized update, others plain GD). Returns
    the loss curve, raw target-gradient stream, realized buffer stream, val loss, and a
    divergence flag (stops when loss exceeds 1e3 x initial).
- `config.yaml` — documents the task config; the canonical values live inline in the `CFG` dict of
  `scripts/run_e6_trajectory.py` (kept in sync, and the full config is written to each run record).

## Gradient flow
- Target matrix: `model.fc1.weight` (shape `d_hidden × d_in`).
- Per step: `model.zero_grad(); loss.backward()`; read `target.grad` (clone to numpy) BEFORE the
  in-place GD update `p -= eta * p.grad`. The returned stream is the `β=0` (plain GD) gradient
  sequence; the EMA momentum filter is applied open-loop in analysis (idea_v1.md Task 4.2).
- Freezing weights for a stationary probe: hold the model fixed and resample mini-batches; not used
  by E6 (full-batch trajectory probe), but `core.probe.collect_stationary` supports it.

## Knobs that matter
- `eta` — set near the **edge of stability**. With the non-realizable target `y=sin(freq·⟨w,x⟩)`,
  `eta≈0.3` gives sustained ω=π oscillation (HFER≈0.99) while the loss still descends; `eta≲0.2`
  is smooth (HFER≈0); `eta≳1` diverges.
- `freq` — target frequency; higher = harder/sharper landscape. `freq=3` is enough to enter EoS.
- `cond` — input condition number (anisotropy). `seed` — model + data seed (full determinism).

## Smoke test
```
cd codebases && python scripts/run_e6_trajectory.py
```
Expect: `loss 0.51 -> 0.31`, `HFER(G)≈0.99` (white baseline ≈0.41), spectral peak at `ω/π=1.000`,
MSR(β) matching `|H_β(π)|²`, slow-gradient alignment rising from 0.26 (raw) to ~0.82 (β=0.99), and
`decision gate ... PASS`. One run logs to `results/cache/` and a row to `results/index/runs.csv`.
