# Wisteria campaign notes (state file)

Running state for the plan_v5 §3 GPU campaign. Started 2026-07-19 (session 1). Update at every
session end: status, node-hours spent, open problems, next actions.

## 1. Cluster facts (verified 2026-07-19 on wisteria03; commands in parentheses)

### Resource groups — Aquarius (project gs26; `pjstat --rsc -x`)

| rscgrp | nodes/GPUs | max elapse | use for |
|---|---|---|---|
| `share-debug` | 1 node, 1/2/4 GPU | 30 min | smoke tests, nvidia-smi checks |
| `share-short` | 1 node, 1/2/4 GPU | 2 h | calibration, preflight |
| `share` → share-1 | 1 GPU | **48 h** | G1/G2/G3/G6 singles (the bulk) |
| `share` → share-2 | 2 GPU | 48 h | — |
| `share` → share-4 | 4 GPU | 24 h | — |
| `short-a` | 1–2 nodes | 2 h | quick full-node tests |
| `regular-a` → small-a | 1–2 nodes | 48 h | 124M DDP runs (G4 conf, G5) |
| `regular-a` → medium-a | 3–4 nodes | 48 h | not needed |
| `regular-a` → large-a | 5–8 nodes | 24 h | not needed |

Submission syntax (from Leon's working scripts, e.g. `momentum_probe/experiments/nanogpt/scripts/run_job_probe_sgdm.sh`):
`#PJM -L rscgrp=share` + `#PJM -L gpu=N` (share family; the share-1/2/4 subgroup is selected by
the gpu count) or `#PJM -L rscgrp=regular-a` + `#PJM -L node=1`; plus `#PJM -g gs26`,
`#PJM -L elapse=H:MM:SS`, `#PJM -j`, `#PJM -o <log>`.

### Job caps — Aquarius (`pjstat --limit`)

- ACCEPT 16 (queued+running per project), RUN 8 concurrent, 64 GPUs concurrent, Max-node(A) 8.
- **Bulk jobs disabled on Aquarius** (BULK_ACCEPT 0) — submit loops, ≤16 in queue at a time.

### Token budget (`show_token`, Users_Guide_en_20240524.pdf §1.6.1 Table 1-5)

- Aquarius charge: **tokens = requested GPUs × elapsed hours × 1.5** (share and regular alike;
  a full node = 8 GPUs = 12 token-h/h). Odyssey coefficient 1.0; 3.0 only for priority groups.
- Group gs26: limit 17,280 token-h, used 12,352.4 (71%) at session start → **≈4,928 remaining**,
  expiry 2027-03-31. Campaign plan ≈775 A100-h ≈ 1,163 tokens (1,510 with +30% contingency) —
  ≈31% of the remaining budget. Fits.
- At 100% usage queued jobs are held; running jobs finish. `pjstat` shows expected token per job.
- **Scheduled system stop: 2026-07-31 09:00 JST** (pjstat banner, seen 2026-07-19 — 11.5 days
  out). Long jobs must complete before it; plan G4/G5 124M full-node runs accordingly.

### Storage (`show_quota`)

- `/work/gs26` (= symlink to `/work/02/gs26`): group quota 8 TiB, 3.78 TiB used. **File-count
  quota 4.10 M, 3.06 M used (75%) — the binding constraint.** Datasets must be tarballs /
  binary shards, never extracted image trees. Monitor with `show_quota` at session ends.
- `/home` 50 GiB/user (45.5 used — do not put anything there). `/data/scratch/gs26` 320 GiB,
  purged (fine for transient only).

### Nodes, software, network

- Aquarius node: 2× Xeon 8360Y + 8× A100-40GB SXM4. Driver **570.148.08, CUDA 12.8 capable**
  (nvidia-smi in `_mng_probe/records/track_1_short/2025-10-04_Backout/` logs; re-verify in smoke job).
- Modules (login + aquarius nodes, x86_64): `gcc/8.3.1`, `python/3.11.7` (also 3.10.13),
  `cuda/12.6` … `cuda/10.1`, cudnn ≤8.9.4, nccl ≤2.18.5. Load pattern:
  `module purge && module load gcc/8.3.1 python/3.11.7` (+ `cuda/12.x` only if a source build
  ever needs nvcc — pip torch wheels bundle their own CUDA runtime).
  `export LD_LIBRARY_PATH=/work/opt/local/x86_64/apps/gcc/8.3.1/python/3.11.7/lib:$LD_LIBRARY_PATH`
  is required for the venv python to find libpython (Leon's scripts do this).
- **Login node has direct outbound internet** (PyPI, download.pytorch.org, HF all reachable; no
  proxy needed). GitHub SSH works (`ssh -T git@github.com` → authenticated as yinleung), so
  `git push origin` works from the login node.
- **Compute nodes have no internet** (Leon's `install_env.sh` via `proxy.f.u-tokyo.ac.jp:8080`
  fails: DNS unresolvable from nodes — that install log is a failure record, not a recipe).
  Everything is staged from the login node.
- DDP quirk from Leon's scripts: `export NCCL_SOCKET_FAMILY=AF_INET` and
  `GLOO_SOCKET_FAMILY=AF_INET`; launch via `torchrun --standalone --nproc_per_node=N`.

### Pre-staged data found on /work (reuse, don't re-download)

- **FineWeb 10BT sample, GPT-2 tokens, llm.c shard format** (`kjj0/fineweb10B-gpt2`; magic
  20240520, version 1, 256×int32 header then uint16 tokens; 100M tokens/shard):
  `/work/gs26/s26001/modded-nanogpt/data/fineweb10B/` — 32 train shards (3.2B tokens) + 1 val.
  A second partial copy (10 files) in `/work/gs26/s26001/nanogpt/data/fineweb10B/`.
- Note: plan B1 names FineWeb-**Edu** sample-10BT or OpenWebText. The staged corpus is classic
  FineWeb 10BT — the same corpus the modded-nanogpt speedrun (G5's externally-tuned baseline)
  trains on, so using it keeps G4/G5 on one dataset. Flagged to Leon at the Phase-0 gate;
  FineWeb-Edu backup download deferred until Leon weighs in (avoids duplicate 6 GB + file-count
  pressure).

## 2. Decisions taken this session

- Venv: `.venv-a100` at repo root (gitignored), python/3.11.7 module + pip torch cu126 wheels.
  The revision_v5 `.venv` (python3.8, for sympy checks) is untouched and remains valid for its
  original purpose.
- resnet-cifar task: `kuangliu/pytorch-cifar` dropped into `upstream/` (canonical CIFAR
  ResNet-18, 3×3 stem); adapter implements the Integration Contract.
- nanogpt GPU configs: extend the existing `nanogpt` task with a fineweb loader in the adapter
  (upstream untouched); 20M/124M GPT-2-BPE configs. Data path points at the staged
  modded-nanogpt shards (read-only).

## 3. Raw-window locations (diagnostic streams stay on /work, never in git)

- (none yet)

## 4. Node-hours / tokens spent (campaign ledger)

| date | what | jobs | GPU-h | tokens |
|---|---|---|---|---|
| (pending first submission) | | | | |

## 5. Status / next actions

- Phase 0 in progress (2026-07-19): cluster facts verified; B1 done (venv torch 2.11.0+cu128
  verified against driver 570; wisteria scripts: setup_env / submit / templates / stage_data /
  sync_results; CIFAR-10 staged+md5-verified, CIFAR-100 downloading (cs.toronto slow),
  FineWeb pre-staged); B2+B3+B5 written as one core/ batch (device, logging git-sha,
  looprunner, spectral_stream, hvp, forced, lbprobe) + resnet-cifar & nanogpt-gpu adapters +
  preflight/smoke/calibrate drivers. CPU micro-smokes of both adapters PASS on the login
  node (all instrumentation paths exercised; λ̂ tracker shift-fix verified; SDPA MATH
  backend needed for GPT HVP double-backward — flash kernel lacks it).
- Next: commit the batch → Codex review → GPU smoke (share-debug) + preflight (share-short)
  + calibration (share-short & short-a 8-GPU) → Phase-0 gate report to Leon.
