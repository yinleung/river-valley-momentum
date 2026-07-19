# Wisteria campaign prompts (paste-ready)

Prompts for Claude Code sessions on Wisteria executing `plan_v5.md` §3. Written 2026-07-19, after
revision_v5 (paper work done; campaign not started). Paste the fenced block; fill `<...>` slots.

---

## Session 1 — Phase 0 bring-up, then G1 → G2 (kill-switch) → G3

```
You are on the Wisteria/BDEC-01 cluster (login node), in the repo clone at <repo path, e.g.
/work/<group>/<user>/river-valley-momentum>. This is a code/experiment session: read
codebases/CLAUDE.md (which imports CODING.md) first, then discussions/plan_v5.md — your goal is
its §3 GPU campaign. The EXECUTION STATUS block at the top of plan_v5.md tells you what is
already done: the paper pass (revision_v5) is complete and Codex-passed, so do NOT redo any
theory or writing work and do NOT edit latex/ in this session.
discussions/revision_v5_changelog.md records the completed paper pass.

Mission, strictly in this order, with a stop-and-report gate after each numbered step:

1. Phase 0 (plan §3.1). First verify every cluster fact marked "?" in §3.1 (Aquarius resource
   group names, elapse ceilings, our group's node-hour budget and charge rates, concurrent-job
   caps — use pjstat/pjsub docs and the accounting tools) and record them in a new
   discussions/wisteria_notes.md. Then: B1 pjsub templates + environment (python >= 3.10 venv on
   /work with torch cu12x wheels — the existing revision_v5 .venv is python3.8 and is NOT
   sufficient for torch; build fresh from a newer module or miniforge); B2 CUDA port +
   determinism policy and B3 git-SHA logging as ONE batched core/ commit (the code_sha churn
   rule, plan §3.1); B4 the throughput calibration table; B5 the instrumentation modules
   (core/spectral_stream.py, core/hvp.py, core/forced.py, core/lbprobe.py,
   scripts/preflight_stream.py) and the resnet-cifar task per the Integration Contract
   (upstream pristine, adapter-only, AGENT_NOTES.md, CODEBASE_INDEX.md row).
   GATE: a 1-GPU smoke job on the share resource group completes, logs a full cache record
   (config.json/metrics.json/arrays.npz + runs.csv row with git SHA and stage flag), and
   preflight_stream passes on resnet-cifar and on the nanogpt task.

2. G1 (plan §3.2 card): coarse LR scan first, then the P-thy grid, then the P-prac companion,
   CIFAR-100 thinned sweep, GroupNorm control at headline cells. The ten protocol invariants of
   §3.0 are binding (two protocols never mixed; divergence counts as failure; windows >> T_eff;
   assumption diagnostics logged; exact-simulator predictions committed BEFORE launch). Gates
   are predeclared in the run configs. The EoS-style sharpness overlay is the headline figure's
   data — collect it for every grid cell.

3. G2 (plan §3.2 card): paired state/residual decompositions on actual beta trajectories.
   THIS IS THE CAMPAIGN KILL-SWITCH: if the four support conditions fail, STOP — do not start
   G3/G4 — write the negative-result summary and report to Leon with the evidence.

4. G3 (plan §3.2 card): the forced-frequency beta x omega grid from common checkpoints.

Standing rules for the whole session:
- Cache/logging discipline per CODING.md Pillar 3 and the acceptance criterion in plan §3.1 B3.
  Figures are NEVER rendered here — Leon renders locally from the synced cache. Raw diagnostic
  windows stay on /work; record their paths in wisteria_notes.md; only distilled arrays.npz go
  into results/cache/.
- Codex review per the root CLAUDE.md protocol for every non-trivial code change, judged
  against codebases/CODING.md.
- git: commit at each numbered-step boundary (and when a full experiment family's cache lands)
  and push to origin, so Leon can pull results locally. Never commit raw window data.
- Budgets per plan §3.4. If B4 calibration shows any family will exceed its budget by more
  than 30%, pause that family and surface options to Leon instead of silently thinning grids.
- Compute nodes have no internet: all downloads (datasets, wheels, tokenizer assets) happen on
  the login node in Phase 0.
- End of session: update discussions/wisteria_notes.md with status, node-hours spent, open
  problems, and the next action.
```

## Session 2+ — continuation template (G4 → G5 → G6)

```
Continue the plan_v5 §3 campaign on Wisteria, repo at <repo path>. Read, in order:
discussions/wisteria_notes.md (current state), discussions/plan_v5.md §3 (the spec; EXECUTION
STATUS block at top), and results/index/runs.csv. Completed so far: <e.g. Phase 0 + G1–G3, all
gates PASS>. Next, per the §3.4 order: <G4 diagnostic 20M sweep | G5 20M pipeline tunes |
G4 124M confirmation (pre-registered predicted-best cell — write it into the run config from
the 20M fits BEFORE launching) | G5 124M mechanism arms + external baselines | G6 beta window +
horizon arm>. Same standing rules as session 1: §3.0 invariants binding, predeclared gates,
cache discipline, Codex review, commit+push at family boundaries, pause-and-surface at +30%
budget overrun, figures rendered only by Leon locally. End by updating wisteria_notes.md.
```

## Not for Wisteria sessions

- Paper edits from the results (new experimental sections, Table 2 G-rows, abstract's GPU
  sentence — plan W2/W5 leftovers) happen AFTER the campaign, in a latex/-scoped session on
  the laptop, governed by latex/WRITING.md + CRITERION.md.
- T2 (global stability) and T6 (noise lemma) are stretch theory — separate login-node or
  laptop sessions with proof-writer + check scripts, not campaign work.
- The response letter is assembled last (concede-and-extend on the reviewer's β=0 formula —
  see the correction note in plan §2 T1).
