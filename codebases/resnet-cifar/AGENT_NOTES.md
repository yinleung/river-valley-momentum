# Agent notes — resnet-cifar task (G1/G2/G3)

## Repository map (upstream)
- `upstream/` = kuangliu/pytorch-cifar at `49b7aa9` (2021-02-25), **never edited**
  (clone history kept locally as `upstream/.git-upstream-49b7aa9`, gitignored).
- `upstream/models/resnet.py` — the canonical CIFAR ResNet-18: 3×3 stem, no maxpool,
  4 stages [64,128,256,512], BasicBlock ×2 each, BN everywhere, `linear` head. 11.17M
  params. We import `models.resnet.ResNet18` only; upstream `main.py`/`utils.py` unused.
- `data/` (gitignored): CIFAR-10/100 python-pickle batches, staged by
  `scripts/wisteria/stage_data.sh` on the login node.

## Gradient flow / integration
- `adapter/contract.py` implements the Contract. The instrumented loop
  `end_to_end_train(model, data, targets, config, opt, injector, raw_sink)` runs EMA-SGDM
  (`opt={"kind":"ema","beta":b}`, P-thy) or conventional heavy-ball (`"hb"`, P-prac;
  eta_HB = eta*(1-beta) conversion) on ALL parameters, fp32, no clipping; divergence guard
  = non-finite or >3× initial loss; spike counter = steps with loss > 2× trailing median.
- Probed matrices (`target_modules`): conv1 (stem), layer2.0.conv1, layer3.0.conv1,
  layer4.1.conv2, linear — 5 across depth. Raw fp16 windows record `target.grad` per
  window step (post-injection when G3 forcing is on). The momentum stream is NOT stored:
  the window-start buffer snapshot `m0` + the G stream reconstruct it exactly via
  m_t = beta*m_{t-1} + coef_g*g_t (coef_g = 1-beta for ema, 1 for hb).
- Probes are state-neutral: LargeBatchProbe and the HVP paths save/restore every param
  .grad and every module buffer (BN running stats). Probe forwards run TRAIN-mode BN on
  fixed chunks (chunk size declared in config) — the training-loss geometry, not eval
  running stats; the `norm: gn` control arm is where no BN caveat applies at all.
- Sharpness: `core.hvp.SharpnessTracker`, full-model, fixed 512-example subset
  (seed_offset 3000), warm-started power iteration, cadence `lam_every`. Full Lanczos
  top-16 (its own fixed 2048-example batch, seed_offset 2000) only at window midpoints
  when `eig_at_mid` (G2). chi = eta_eff*(1-beta)*lam_hat logged as `chi_trace`.
- Data determinism: whole CIFAR resident on device; batch draw = seeded rng (seed+7),
  sampling with replacement; val draw rng seed+8; augmentation rng seed+5000 (fixed
  per-seed sequence, shared across arms; default OFF for mechanism runs). Paired arms at
  one seed therefore see identical batch AND augmentation sequences, and probe cadence
  never advances the training generator (probe batches come from their own generators:
  lb 1000, eig 2000, lam 3000).

## Config knobs that matter
- `batch_size: 128`, `steps: 10000`, `windows` = 3×2048 (>= 10*T_eff at beta=0.99).
- `lb_every`: 256 in G1 (alignment scalars), 1 in G2 (full decomposition stream).
- Memory: G2 windows are spilled via `raw_sink` as they complete (~5.5 GB fp16 per
  2048-window over all 5 targets; lb_targets restricts the GLB stream to 3 matrices).
- GN arm: `norm: gn` (GroupNorm(32) surgery on the built model — upstream untouched).

## Dependency / environment quirks
- Needs only torch/torchvision-free stack: torch + numpy (torchvision unused — CIFAR is
  read from the pickles directly).
- CUDA determinism: drivers must call core.device.seed_all + setup_determinism; pjsub
  templates export CUBLAS_WORKSPACE_CONFIG=:4096:8.

## Smoke test
```
cd codebases && python - <<'EOF'
import sys; sys.path.insert(0, 'resnet-cifar/adapter')
import contract
cfg = dict(seed=0, dataset='cifar10', batch_size=128, steps=40, lr=0.1,
           device='auto', windows=[[10, 16]], sketch_k=32, lam_every=20,
           lam_batch=256, lam_chunk=256, lb_batch=512, lb_chunk=256, lb_every=8)
model = contract.build_model(cfg)
data = contract.data_loader(cfg)
targets = contract.target_modules(model, cfg)
out = contract.end_to_end_train(model, data, targets, cfg, dict(kind='ema', beta=0.9))
print('loss', out['losses'][0], '->', out['losses'][-1], '| val', out['val_loss'],
      '| lam', out['lam_trace'][-1] if len(out['lam_trace']) else None,
      '| raw', {k: v.shape for k, v in out['raw'].items()})
EOF
```
