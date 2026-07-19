# Agent notes — nanogpt task (E11 scale transfer)

## Repository map (upstream)
- `upstream/` = karpathy/nanoGPT at `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`
  (2025-11-12, shallow clone), **never edited**.
- `upstream/model.py` — `GPTConfig` dataclass + `GPT` module; forward returns
  `(logits, loss)` when targets are given. Blocks live at `model.transformer.h[i]`, each
  with `.attn` (`c_attn`, `c_proj`) and `.mlp` (`c_fc`: 4·n_embd × n_embd, `c_proj`).
- `upstream/train.py` — reference loop (AdamW, grad clip, DDP); we do NOT use it — the
  adapter runs its own plain closed-loop so the optimizer is exactly the paper's EMA-SGDM /
  Muon variants, with no clipping or decay in the way of the theory.
- `upstream/data/shakespeare_char/prepare.py` — downloads tiny-shakespeare and writes
  `train.bin` / `val.bin` (uint16 char ids) + `meta.pkl` (vocab 65) into its own directory.
  These generated artifacts are upstream's intended workflow, not edits. Run once.

## Gradient flow / integration
- `adapter/contract.py` implements the Integration Contract: `build_model`, `data_loader`
  (memmap bins → seeded random (x, y) batches; batch order is a function of
  `config["seed"]`, so arms at the same seed see identical data), `target_module`
  (last block's `mlp.c_fc.weight`, 512×128 at n_embd=128), `end_to_end_train` (SGDM on all
  parameters, or Muon pre/post/polar on the target with plain SGD elsewhere).
- Per-step target gradient = `target.grad` after `loss.backward()` on the mini-batch; the
  probe window `[probe_start, probe_start+probe_len)` streams float32 copies to CPU.
- Divergence guard: batch loss non-finite or > 3× the first loss stops the run (no gradient
  clipping — instability must be observable for CL-1 claims).

## Config knobs that matter
- `n_layer=2, n_head=4, n_embd=128, block_size=128, batch_size=32` → 0.40M params.
- `device: auto` → MPS when available (float32; MPS has no float64), else CPU.
- `probe_len=256` gives the DFT window; keep it a contiguous mid-training stretch.
- Pilot (300 steps, seed 0, plain SGD): stable at lr ≤ 0.2 (tail ≈ 2.5–3.0), diverges at
  lr ≥ 0.4 within ~4 steps; raw-stream HFER 0.87–0.98 across lr 0.02–0.2 (white ≈ 0.41);
  at lr=0.4, β=0.9/0.95 train fine (tail ≈ 2.45/2.48) and their realized streams are
  smooth (HFER 0.24/0.11) — closed-loop damping removes the oscillation it filters.

## Dependency / environment quirks
- MPS non-determinism is minor but nonzero; runs record seeds and the code sha. SVDs for
  the Muon variants run on CPU (numpy) — MPS SVD support is not needed.
- `prepare.py` needs network once (`requests`); afterwards everything is offline.

## Decomposition probe (E12, adapter/decomp.py)
- `run_decomp(config)` mirrors the SGDM β=0 path of `contract.end_to_end_train` (same rngs,
  same batch sequence per seed) and adds, inside the probe window: a large-batch gradient
  `ĝ^LB` of the target at the same iterate (config `lb_batch`, chunked by `eig_chunk`; drawn
  from a `seed+1000` loader so the training stream is untouched; all param grads are
  saved/restored around the probe), and top-`eig_k` restricted-Hessian eigenpairs at the
  window midpoint (block power iteration; double backward WORKS on MPS in torch 2.10 —
  CPU-copy fallback kept). ~2 min per 2000-step run at lb_batch=1024, k=16.
- Driver `scripts/run_e12_decomp.py` (gates A/B/C predeclared) reduces everything in-driver
  and caches only small arrays — raw (256, 512, 128) streams are NOT stored (rerunnable).

## Smoke test
```
cd codebases/nanogpt/upstream/data/shakespeare_char && python prepare.py   # once
cd codebases && python - <<'EOF'
import sys; sys.path.insert(0, 'nanogpt/adapter')
import contract
cfg = dict(seed=0, n_layer=2, n_head=4, n_embd=128, block_size=128, vocab_size=65,
           batch_size=32, steps=60, eval_iters=20, probe_start=20, probe_len=16,
           device='auto', lr=0.2)
model = contract.build_model(cfg)
out = contract.end_to_end_train(model, contract.data_loader(cfg),
                                contract.target_module(model, cfg), cfg,
                                dict(kind='sgdm', beta=0.9))
print(out['losses'][0], '->', out['losses'][-1], 'val', out['val_loss'])
EOF
```
Full driver: `python scripts/run_e11_scale.py` (~1.5–2 h on MPS; logs task `nanogpt`).
