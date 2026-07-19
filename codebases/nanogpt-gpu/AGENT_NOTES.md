# Agent notes — nanogpt-gpu task (G4/G5/G6)

## Repository map
- **No upstream copy of its own**: imports the vendored `../nanogpt/upstream` (karpathy
  nanoGPT @ `3adf61e`, never edited) — this task is the plan's "config extension of the
  existing nanogpt task" at GPT-2 scale. `model.py` provides `GPT`/`GPTConfig`; upstream
  `train.py` unused (the closed loop is `core.looprunner`, exactly the paper's optimizers).
- Data: FineWeb 10BT GPT-2 shards in llm.c format at
  `/work/gs26/s26001/modded-nanogpt/data/fineweb10B` (32 train shards x 100M tokens + val;
  pre-staged before this campaign; `kjj0/fineweb10B-gpt2`). Header = 256 int32
  (magic 20240520, version 1, ntok); tokens uint16 from byte 1024. NOTE: this is classic
  FineWeb, not FineWeb-Edu — the same corpus the modded-nanogpt speedrun (G5's external
  baseline) trains on; flagged to Leon at the Phase-0 gate.

## Gradient flow / integration
- `adapter/contract.py` implements the Contract; `end_to_end_train` = `core.looprunner`
  under GPT hooks. Training forward under bf16 autocast (CUDA); probe paths fp32 on the
  bare module. Divergence guard per looprunner (rank-local — synchronize before any DDP
  run where divergence is a live outcome; see docstring).
- Probed matrices: `mlp.c_fc` at layers {0, mid, last} + `attn.c_attn` at mid.
  m-stream reconstruction from window-start snapshot as in resnet-cifar.
- Determinism: batch order = f(seed, rank); probe generators at seed offsets lb 1000,
  eig 2000, lam 3000, val 4000. LayerNorm only -> no BN caveats; probe_mode train==eval.
- DDP (8-GPU node): `ddp_setup(model)` after `build_model`; torchrun sets RANK/WORLD_SIZE/
  LOCAL_RANK; NCCL_SOCKET_FAMILY=AF_INET (pjsub template). batch_size is PER-RANK.
  Instrumentation asserts world==1 (mechanism probes live on the 20M single-GPU rung).

## Config knobs that matter
- `models.gpt20m`: 6L/6H/384d (~29.9M params incl. tied embeddings, ~10.6M non-embedding)
  batch 64x1024 tokens; `models.gpt124m`: GPT-2 small 12L/12H/768d, per-rank batch 24.
- Windows >= 2048 for beta = 0.99 (T_eff = 199; §3.0.3). 20M probe batches: lb_batch in
  SEQUENCES (x1024 tokens each) — lb_batch 256 sequences ~ 262k tokens per probe.

## Dependency / environment quirks
- Shards dir is READ-ONLY input from another project; never write there. If more tokens
  are ever needed: `N_CHUNKS=48 bash scripts/wisteria/stage_data.sh` extends in place
  (103 chunks max). Val shard is a single fixed 100M-token file.
- bf16 autocast: loss reported fp32; probe forwards bypass autocast entirely.

## Smoke test
```
cd codebases && python - <<'EOF'
import sys; sys.path.insert(0, 'nanogpt-gpu/adapter')
import contract
cfg = dict(seed=0, n_layer=6, n_head=6, n_embd=384, block_size=1024, vocab_size=50304,
           batch_size=8, steps=30, lr=0.05, device='auto', eval_iters=4, eval_batch=8,
           windows=[[8, 16]], sketch_k=32, lam_every=15, lam_batch=16, lam_chunk=8,
           lb_batch=32, lb_chunk=8, lb_every=8)
model = contract.build_model(cfg)
data = contract.data_loader(cfg)
targets = contract.target_modules(model, cfg)
out = contract.end_to_end_train(model, data, targets, cfg, dict(kind='ema', beta=0.9))
print('loss', out['losses'][0], '->', out['losses'][-1], '| val', out['val_loss'],
      '| lam', out['lam_trace'][-1] if len(out['lam_trace']) else None)
EOF
```
