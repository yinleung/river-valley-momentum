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

## 3. B4 calibration table (measured 2026-07-19; A100-40GB, driver 535.54.03, torch 2.11.0+cu128)

Component times (1 GPU; determinism policy ON: TF32 off, cudnn.benchmark off):

| component | resnet-18/CIFAR (batch 128, fp32) | gpt20m 6L/384d (batch 32×1024, bf16) |
|---|---|---|
| plain train step | 54.1 ms | (calib1 rerun pending) |
| window step (raw fp16 + JL k=128) | 80.5 ms | — |
| LB probe (2048@512 / 256 seq@16) | 0.92 s | — |
| LB probe G2 (4096@512) | 1.79 s | — |
| λ̂ tracker warm (512-ex subset) | 3.57 s | — |
| λ̂ tracker cold start | 24.3 s | — |
| full-model Lanczos k=16 m=48 @2048 | 229 s | — |

- **gpt124m ×8 GPU DDP (batch 24/rank ×1024, bf16): 493.7 ms/step = 398k tokens/s**
  (short-a calib8, job 9224255). ~12% MFU — no torch.compile (determinism/simplicity);
  compile is the fallback lever if G4/G5 budgets force it.
- Flash-attention BACKWARD is nondeterministic (warn-only policy); probe paths use MATH
  SDPA (double-backward + deterministic). Accepted + recorded; cached-arrays rule applies.

### Family projections vs plan §3.4 budgets (from components; runs × per-run)

| family | budget h | projected | verdict |
|---|---|---|---|
| G1 sweep (~181 runs × ~0.26 h) | 110 | **~47** | ok (0.43×) |
| G2 decomposition (12 × ~3.9 h) | 55 | **~47** | ok (0.85×) |
| G3 forced (2 ckpts + 132 arms) | 15 | **~2** | ok |
| G4 20M diagnostic | 75 | ~10–15 (pending gpt20m step time) | ok |
| G4 124M confirmation (15 × 2.5B tok ÷ 398k/s × 8 GPU) | 90 | **~209** | **2.3× OVER → pause-and-surface (§5)** |
| G5 124M arms (same rate; recompute at G5 session) | 175 | likely similar overrun factor | surface with G4 |
| G6 window | 55 | fine at measured rates | ok |

## 4. Raw-window locations (diagnostic streams stay on /work, never in git)

- G1 grid: seed-0 cells keep the EoS-onset window (index 1) raw G for layer3.0.conv1 +
  linear under `codebases/results/raw/grid_*/` (~11 GB total; policy in run_g1_resnet.py).
- G2: full G+GLB fp16 windows per cell under `codebases/results/raw/decomp_*/`
  (~50–100 GB transient; delete after the paper's numbers freeze).
- G3 checkpoints + probe vectors: `codebases/results/ckpt/g3_*.pt|_probe.npz`.

## 4b. G1 findings (2026-07-20) — two that change how gates are reported

**F1 — BN never diverges at β=0 (scan + grid, lr 0.0125 → 6.4, 10k steps).** Card gate (b)
("some LR where β=0 diverges on all seeds and β≥0.9 trains") is **not testable on the BN
arm**: BatchNorm's rescaling invariance removes hard divergence over the whole ladder.
Reported as a finding, and gate (b) moves to the GroupNorm control (predeclared before the
grid launched, commit 039694f).

**F2 — GroupNorm sign-flips gate (b): momentum DEstabilizes at fixed η.** GN control
(headline cells, 3 seeds):

| cell | β=0 | β=0.9 |
|---|---|---|
| lr 0.8 | 0/3 diverged, acc 0.786 | **3/3 diverged** |
| lr 3.2 | 0/3 diverged, acc 0.677 | 1/3 diverged, acc 0.747 (surv.) |

Mechanism, from the instrumentation (not inferred): **every GN divergence occurs at step
10–11**, and λ̂(init) ≈ 616–707 for ALL GN runs, survivors included. So both arms start
enormously above their thresholds (β=0 at lr 0.8: χ = 565 vs threshold 2); the β=0 arms
survive by catapulting — 200–450 counted loss spikes — while the β=0.9 arms die during the
same transient. Reading: EMA at fixed η spreads the corrective response over T_eff ≈ 19
steps, so the iterate escapes before the correction lands. Note the non-monotonicity — at
lr 3.2 the β=0.9 arm mostly survives (bigger first step escapes to a flatter region faster).

Scope, stated plainly: this is an **initialization-transient** effect, OUTSIDE the
stationary/local regime the propositions describe (they linearize around a valley). It is
NOT evidence against the stationary threshold claim, and it must not be reported as such —
but it IS a real sign-flip of the gate as operationalized, so per the G1 card it is reported
as a negative result and the theory's scope wording is revisited before G4.
**For Leon (scope decision, not taken here):** the natural follow-up is a short-warmup GN arm
(standard practice; isolates transient from stationary). It is not in the predeclared card,
so it is not being run unilaterally — say the word and it is ~1 GPU-h.

## 5. Node-hours / tokens spent (campaign ledger)

| date | what | jobs | GPU-h | tokens (≈GPU-h×1.5) |
|---|---|---|---|---|
| 2026-07-19 | Phase 0 bring-up: smoke ×3, cudacheck, preflight ×4, calib1 ×3, calib8 (8-GPU) | 9224222–9224312 | ≈1.5 | ≈2.3 |

## 6. Status / next actions

- **Phase 0 COMPLETE, gate PASS (2026-07-19).** Evidence:
  - Smoke (share-debug GPU): PASS ×3, **bitwise-deterministic** ResNet repeat; full cache
    records (config/metrics/arrays + runs.csv rows w/ git_sha + stage) for both tasks.
  - preflight resnet-cifar: **PASS** (P0/P1/P2) — HFER(0.6π) 0.47→0.54 over lr 0.0125→1.6
    (white q97.5 = 0.418), PSD Spearman 0.68–0.93, ρ₁ −0.12→−0.22. Regime present.
  - preflight nanogpt-gpu (20M, FineWeb): **PASS** — HFER 0.83–0.86, Spearman 0.70–0.96,
    ρ₁ −0.60→−0.72. Regime emphatically present.
  - Codex reviews: round 1 FAIL → all 3 findings fixed (path bug; results/-aware dirty
    check; DDP-synchronized divergence guard); round 2 on G2/G3 drivers pending.
  - Divergence guard revised to SUSTAINED 3×-for-10-steps (instant rule misclassified
    recoverable step-1 spikes at lr ≥ 0.2); preflight P2 revised to the sign test — both
    documented in-file, before any campaign measurement.
- **DECISION NEEDED FROM LEON (pause-and-surface, §3.4 +30% rule): G4 124M confirmation
  projects ~209 GPU-h vs 90 budgeted (2.3×)** at the measured 398k tok/s (15 runs × 2.5B
  tokens). Options: (a) accept the overrun — total campaign ≈ 900 A100-h ≈ 1,350 tokens,
  still ~27% of the group's remaining 4,928 (budget headroom is real); (b) 2 seeds instead
  of 3 at 124M (~140 GPU-h); (c) full 2.5B only for the pre-registered predicted-best cell,
  1.25B for the other 4 configs (~120 GPU-h); (d) add torch.compile for the 124M runs
  (12% MFU today; likely ≥2× step-rate, but adds a compile-vs-eager consistency check to
  the protocol). G5 (175 h, 124M-heavy) will be re-projected at its session with the same
  lever set. No G4/G5 job launches until Leon picks.
- G1 STARTED: coarse scan (job 9224311, 10 LRs × 10k steps) running; grid follows the
  scan classification + preregister + commit, per the predeclaration protocol.
- Next session entry point: read this file §6, then `pjstat`; if the scan finished,
  review its classification table in `results/joblogs/g1scan.*.log`, commit grid_lrs,
  run `--stage preregister`, commit, then launch grid rows (submit.sh, 5 jobs `--rows`).
