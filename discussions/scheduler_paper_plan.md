# Scheduler paper plan — regime-triggered scheduling of Muon's two knobs

**Status:** plan v1, drafted 2026-07-07.
**Provenance:** supersedes `betascheduler_idea_v1.md` (imported from the original betascheduler
repo — this repo's namesake mission) and absorbs `idea_v3.md`'s (β,α) program as *measurement +
policy*, per the merge decision: one empirical-side paper; Layer B/C theorems parked (§11).
**Relation to paper 1:** consumes its dials (confinement floor, lag top edge, resonance guard),
its toy harness (E9/E10/E13 style), and its Remark-9 scope edit (`paper1_revision_plan.md`).

---

## 1. Thesis and claim

> Optimizer schedules encode regime changes, and the regimes are cheaply measurable — so schedules
> should be triggered by dials, not clocks. For Muon: the momentum coefficient should track the
> valley's β window (floor: confinement onset; top: drift/lag), and the orthogonalization strength
> should fall when the refinement regime begins. Both triggers are cheap gradient statistics, and
> they beat hand-tuned clock schedules on a serious benchmark — and, unlike clock schedules, they
> transfer when the training horizon changes.

Working titles (neutral until results): *Dials, Not Clocks: Regime-Triggered Scheduling for Muon* /
*Scheduling the Two Knobs of Muon by Measurement*. Arm names stay neutral (`Muon-StaticBeta`,
`Muon-PreBeta`, `Muon-SFBeta`, `Muon-WTBeta`, `Muon-AlphaClock`, `Muon-AlphaTrig`, …).

"General" means the **framework** generalizes (any knob gets a dial), demonstrated on Muon's two
knobs on one serious benchmark plus one horizon-transfer setting. Not a survey across optimizers.

## 2. Scope and non-goals

In scope: Muon momentum coefficient β_t (equivalently T_t = 1/(1−β_t)); orthogonalization strength
(three implementations, §3.2); their joint schedule; trigger-vs-clock ablations; horizon transfer.

Non-goals (v1):
- LR schedule fixed to the pinned script's (the third knob; regime-explained by Wen et al. — one
  positioning paragraph, zero experiments). Exception: parked stretch arm, §10.
- No per-layer / per-matrix β; global scalar only.
- No architecture, data, batch, or fwd/bwd-count changes (Track 3 rules).
- Adam's β₂ / Shampoo accumulation window: one discussion sentence ("the same knob elsewhere").
- No new theorems. Toy-model measurements + conjecture only; proofs go to the parked companion.

## 3. The two knobs and their dials

### 3.1 β knob (temporal filter strength)

Schedule and report in **T-space** (T = 1/(1−β)); controllers step `log T`, clipped to
[log T_min, log T_max]. Adiabaticity: per-step |Δlog T| ≤ 0.02, so the fixed-β guides from paper 1
apply piecewise (state this in the paper; the schedule moves slowly relative to T_eff).

Dials (all cheap scalars over Muon params, all-reduced across ranks):

| Dial | Statistic | Regime it detects | Response |
|---|---|---|---|
| confinement / hill dominance | `s_t = EMA_w( ⟨G_t,G_{t−1}⟩ / (‖G_t‖‖G_{t−1}‖) )` — successive-gradient anticorrelation; equivalently sign-flip fraction or windowed HFER of a scalar probe | post-confinement hill oscillation (paper 1 E6/E7: filtering pays after confinement) | s_t below threshold ⇒ **raise** log T |
| drift / lag guard | `d_t = cos( M_t , SMA_w(G) )` — buffer vs short simple average (low band) of recent gradients | river bend; buffer lagging the slow signal (paper 1: kv·T_eff ≲ 1 top edge) | d_t below threshold ⇒ **lower** log T |
| noise floor | `n_t = max(v_t − ‖M_t‖², 0)/v_t`, `v_t = EMA(‖G_t‖²)` (from idea_v1, direction correct) | broadband noise dominance | high n_t ⇒ raise log T (mildly; secondary) |

**Documented erratum from idea_v1 (kept as a falsification arm):** the raw alignment controller
`a_t = cos(G_t, M_{t−1})`, "agree ⇒ grow memory", conflates two regimes with **opposite** correct
responses. Post-confinement the raw gradient is Nyquist-dominated while the buffer is
river-aligned (paper 1 E6: raw slow-alignment 0.261 vs buffer 0.823), so a_t collapses exactly
where theory wants large T_eff; shrinking T then admits hill energy into M, driving
⟨G_t, M_{t−1}⟩ negative — positive feedback pinning T at T_min in the regime where filtering pays
most, and growing T in the early smooth phase. Prediction C2 (§7): it loses to window tracking and
pins at T_min after confinement. Run it; report it.

### 3.2 α knob (orthogonalization strength)

Three implementations, increasing in surgery:

1. **NS iteration count** k ∈ {5 (default), 3, 2, 1}: fewer Newton–Schulz iterations = partially
   flattened spectrum. Native to Muon, zero new code paths, crude granularity.
2. **Power map** Φ_u(M) = (MM^⊤)^{−u/2} M = U Σ^{1−u} V^⊤ (paper 1 Remark 9 family): exact SVD in
   toys; at scale via `eigh` on the Gram matrix (legal — Track 3 scores **steps**, not wall-clock;
   wall-clock reported as a secondary metric).
3. **Hard switch** Muon → SGDM (and Muon → AdamW as the Shen et al. reproduction arm).

**η(u) convention (stage-1 decision):** Φ_u output scale varies with u, so cross-u comparisons
need a declared convention — (a) fixed η; (b) update-RMS-matched η(u); (c) per-u best-tuned. The
toy stage picks the convention that makes toy→scale ordering transfer (paper 1's E9
normalization lesson: the *direction* of a sweep can invert with the convention; measure both).

**The α dial — open design question #1 (stage-1 deliverable).** Candidates:

| Candidate | Statistic | Cost | Risk |
|---|---|---|---|
| progress stall | per-step train-loss decrease ÷ update displacement, EMA'd | free (both logged) | noisy at small batch |
| overshoot detector | sign-flip rate of successive update projections onto the momentum direction | one dot product | conflates with hill oscillation — must band-separate as in §3.1 |
| river-energy rise | smoothed loss increase under frozen-direction step (Shen's E_{k+1} > E_k reading) | free | slow to fire |
| LR-decay coupling | fire when the pinned LR schedule enters decay | free | clock in disguise — baseline only, not a dial |

Selection criterion: on the toy suite the dial must fire **after** river progress stalls and
**before** oscillation harm accumulates, robustly across ηλ ∈ {1.2, 1.8, 2.5}, with one global
threshold. If no candidate passes, α-scheduling at scale degenerates to clock arms and the thesis
narrows to the β knob — an acceptable, honest fallback (declared in the paper).

## 4. Stage plan and gates

**Stage 0 — infra (blocking).**
Pin modded-nanogpt commit (record SHA, script, PyTorch/CUDA, GPU type/count). **Check the pinned
script for a built-in Muon momentum warmup** (records suggest ~0.85→0.95 over early steps): if
present, it — not fixed β — is the honest baseline arm. Fix idea_v1's URL (github.com). Port the
local cache logger; per-run JSON with `code_sha` + commit SHA (repo logging discipline). Secure
remote CUDA GPUs (Track 3 is not MPS-runnable); measure **R** = node-hours per baseline run.
*Gate G0: one reproduced baseline run within track tolerance; R known.*

**Stage 1 — toy suite (local, ~free).** Extend the paper-1 harness with Φ_u (new adapter module;
additive-only, per repo conventions in `codebases/CODING.md`). Deliverables: (a) (β,u) phase
diagram; (b) η(u) convention; (c) α-dial winner + frozen thresholds; (d) controller unit tests —
window-tracking must hold the E10-style window; raw-alignment arm must fail as predicted; E13-style
forced-disturbance controls must not fool either; (e) ≤ 2 frozen controller configs for scale.
*Gate G1: dials frozen; anything unfrozen does not board the GPU.*

**Stage 2 — scale, β only (the old project-2 core; publishable alone).** Families F1–F4 on
Track 3 (§5.2), three phases: screening (n=1–2) → refinement (top-3, n=5) → validation (top-2,
n≈10, track statistics rule).
*Gate G2 — SHIP-OR-EXTEND: if compute/results disappoint, stage 2 ships as the β-scheduler
benchmark paper; else continue.*

**Stage 3 — scale, α + joint + horizon transfer (§5.3).**

**Stage 4 — parked stretch (do not let it leak into scope):** regime-triggered LR-decay *onset*
(replacing the one clock everyone uses). Spectacular if it works; strictly after G2/G3.

## 5. Experiment grids

### 5.1 Stage 1 (toy) grids

Landscapes: straight noisy valley + curved valley f(x) = 2 sin(kx), k ∈ {0.6, 0.9, 1.2}; σ = 2.

| Axis | Values |
|---|---|
| β | {0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99} |
| u (spectral power) | {0, 0.25, 0.5, 0.75, 1} |
| ηλ | {0.6, 1.2, 1.8, 2.5} |
| λ/μ | {10, 100, 1000} |
| seeds | 8 (T = 6000, tail half scored) |

Full factorial ≈ 480 cells × 8 seeds — cheap locally. Metrics: tail hill loss, tube escape, river
speed, and the u-dependent stationary floor (expected qualitative shape: limit-cycle amplitude
shrinks as u → 0; measure, don't assert a formula). η(u) run twice: fixed-η and RMS-matched arms.
MPS caveat from repo practice: derived numbers traced to cached arrays, not re-runs.

Controller tests on the curved valley: window-tracking vs raw-alignment vs fixed-β, 16 seeds,
plus forced Nyquist/passband/resonance tones (E13 pattern) — a controller must raise T under a
Nyquist tone, do nothing at passband, and never chase the resonance frequency.

### 5.2 Stage 2 (scale, β) — families and run counts

All runs: pinned script, LR schedule untouched, global scalar β, T-space logging,
T_min = 5, T_max = 100 clips unless stated. n = seeds.

| Family | Configs | n | Runs |
|---|---|---|---|
| F1 static | β ∈ {0.85, 0.90, 0.95, 0.97, 0.98, 0.99} + script default + built-in warmup (if present) | 2 | ≤ 16 |
| F2 pre-schedules | log-linear in (1−β): T 5→50, 5→100, 10→50, 10→100; linear-in-T: 10→50, 10→100; floor-respecting ramp (anchor from diagnostic run): 3 variants | 2 | 18 |
| F3 horizon-free growth | T_t = min(T_max, T_min + t/τ): T_min ∈ {5,10} × T_max ∈ {50,100} × τ ∈ {125, 250, 500, 1000} | 1 | 16 |
| F4 feedback | window-tracking: 2 frozen configs from G1 (each: ρ_up, ρ_down ∈ {0.005, 0.02} pre-pruned in toys) | 2 | 4 |
| F4′ falsification | idea_v1 raw-alignment controller, its two recommended settings | 1 | 2 |
| diagnostics | fixed default β, full dial logging (locates confinement onset + floor at scale; anchors F2's ramp) | 2 | 2 |

Screening ≈ **58 runs**. Refinement: top-3 + best static, n=5 ⇒ **20**. Validation: top-2, n≈10
with the track rule (3.28 − μ)·√n ≥ 0.004 ⇒ **20**. Stage-2 total ≈ **100 R**.

### 5.3 Stage 3 (scale, α + joint + transfer)

| Block | Arms | n | Runs |
|---|---|---|---|
| α single-knob | NS-count ramp-down {trigger, clock} × 2 shapes; Φ_u ramp {trigger, clock} × 2 shapes; hard switch Muon→SGDM at {50%, 75%}; Muon→AdamW (Shen repro) at 75%; trigger-switch | 2 | ~24 |
| trigger-vs-clock (thesis test) | best α schedule *shape* held fixed, only the trigger mechanism varied | 2 | 4 |
| joint (β,α) | ≤ 6 configs chosen from the stage-1 phase diagram (best F-family β schedule × best α schedule; ± one-knob-only controls) | 2 | 12 |
| horizon transfer | step budgets {0.75×, 1.5×} × {best triggered, best clock, best static, built-in warmup} | 2 | 16 |
| validation | top-2 overall, n = 8 | 8 | 16 |

Stage-3 total ≈ **72 R**. Grand total ≈ **170–180 R** (+ stage-4 if opened). R measured at G0;
if R ≈ 0.2–1.0 node-hours, the whole campaign is ~35–180 node-hours — refine after G0 and
re-scope family sizes if needed (screening families are the flex zone; validation counts are not).

**Horizon-transfer is the thesis-critical block:** clock schedules tuned at 1.0× should degrade at
0.75×/1.5×; triggered schedules should not. If triggers don't transfer better, the thesis fails
and the paper reports that.

## 6. Metrics, logging, statistics

Primary: fewest steps to statistically satisfy val < 3.28 (track rule). Secondary: val at fixed
step count; pairwise Δ vs best static AND vs built-in warmup; stability/NaN rate; wall-clock
overhead (report honestly, especially for Φ_u `eigh`); dial/knob traces.

Log every 25 steps (or the script's eval cadence): step, val loss (at evals), train shard loss,
β_t, T_t, u_t / NS-count, all dial values (s_t, d_t, n_t, α-dial), global grad/momentum norms,
cos(G, M), update RMS, wall-clock. One JSON per run, `code_sha` + commit SHA + seed; no
validation-based per-run stopping (track rule 8); stopping criteria fixed across arms.

Nesterov coupling: the pinned Muon uses μ in both the buffer update and the blend
(`buf.lerp_(g, 1−μ); g.lerp_(buf, μ)`). v1 schedules both together (preserve script semantics);
"buffer-only vs blend-only" is a stage-3 ablation at most. Note in the paper: the blend re-injects
(1−μ)G_t of raw high band, so paper-1 buffer constants transfer approximately.

## 7. Claims map (each claim → the arms that can falsify it)

| # | Claim | Falsified by |
|---|---|---|
| C1 | regime-triggered β beats best static **and** built-in warmup on steps-to-target | F4 vs F1 (validation phase) |
| C2 | window tracking beats the raw-alignment controller, which pins at T_min post-confinement | F4 vs F4′ + dial traces |
| C3 | triggered α matches/beats the clock switch at 1.0× and transfers at 0.75×/1.5× where clocks degrade | α block + horizon transfer |
| C4 | joint (β,α) ≥ each knob alone | joint block vs single-knob bests |
| C5 | the toy (β,u) phase diagram predicts the scale ordering (regime transfer, as in paper 1 E11) | stage-1 diagram vs stage-2/3 outcomes |

Interpretation discipline (from idea_v1 §7, kept): every negative outcome is reportable —
"a well-tuned constant suffices at this scale" (C1 fails), "staleness heuristics misfire" (C2
informative either way), "α needs no dial beyond the LR clock" (C3 fails). No outcome is wasted.

## 8. Risks and mitigations

- **Compute access** (blocking): remote CUDA needed; decide provider/budget at G0.
- **Confinement onset too early at scale to matter:** paper 1 E11 found hill-dominated streams at
  every stable LR from the start — the β-warmup window may be tiny; the win must then come from
  the drift guard + α knob. Diagnostic runs (F-diagnostics) answer this in week one.
- **Controller meta-hyperparameters:** controllers have their own knobs (ρ's, thresholds) — count
  this tuning honestly; freeze at G1; report sensitivity in an appendix.
- **Φ_u instability / cost at scale:** fall back to NS-count as the α implementation.
- **Baseline strength:** the built-in warmup (if present) is the bar, not fixed β. Never compare
  against a weakened default (idea_v1 §8.1, kept).
- **Overfitting Track 3:** horizon-transfer doubles as the second setting; if results warrant, add
  one more corpus/scale at stage 4.
- **Time-varying-filter theory gap:** guides are fixed-β; adiabatic stepping (§3.1) is the stated
  bridge, verified empirically in toys.

## 9. Paper skeleton (target: empirical optimization track, ICLR/NeurIPS class)

1. Introduction — clocks everywhere (warmup, switch, decay); thesis: dials.
2. Two knobs, three dials — the pipeline, the window (from paper 1), the refinement trigger
   (from Shen et al.); what's measurable.
3. Toy phase diagram and controller falsification (stage 1).
4. Track 3 protocol (rules, pinned commit, statistics).
5. Results: β families (stage 2); α and joint (stage 3); horizon transfer.
6. Diagnostics: dial traces at scale; what fired when, vs the toy prediction.
7. Related work: Shen et al. (clock switch), Wen et al. (LR knob), schedule-free (anytime
   averaging), paper 1 (theory of the β window), Shampoo/Muon lineage (the u family).
8. Limitations: one benchmark family, global scalar knobs, no proofs (pointer to companion).

## 10. Parked (visible, not planned)

- Regime-triggered LR-decay onset (stage 4).
- Per-layer β/u; Adam β₂ as the same knob; joint (η, β, u) phase diagram.

## 11. Interface to the parked theory companion

Everything the companion (idea_v3 Layers A–C proofs) would need is logged by this plan: dial
traces (drift/noise proxies for Layer A), the (β,u) toy phase diagram (Layer B/C targets), η(u)
convention data. If the empirical paper lands and the proofs still feel worth it, the companion
starts from cached data, not new runs.

## 12. Immediate next actions

1. Stage 0 checklist: pin commit → inspect for built-in warmup → measure R → logging port.
2. Stage 1 implementation: Φ_u adapter module + controller module in `codebases/` (Codex-reviewed
   per repo protocol), then the 480-cell toy sweep.
3. Paper-1 edits land first (`paper1_revision_plan.md`) so Remark 9 exists to cite.
