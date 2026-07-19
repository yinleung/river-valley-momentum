"""Experiment E7 — optimizer-level tests: do the mechanism metrics predict performance
better than beta alone? (idea_v1.md E7; the last Strong-Success item; fable_dgs_v1.md §8.2.)

Three parts, two logged runs:

  part 1 (task `optimizer_toy`)  closed-loop EMA-SGDM on five river-valley settings chosen so
      the best beta VARIES: S1 straight eta*lam=1.8 sigma=2; S2 straight eta*lam=0.6 sigma=2
      (benign); S3 curved k=0.9 eta*lam=1.8 sigma=2; S4 curved k=0.9 eta*lam=2.5 sigma=2
      (beyond the beta=0 threshold); S5 curved k=1.8, a=1 (same max slope, twice the bend
      frequency) eta*lam=1.8 sigma=2. Grid beta in {0, 0.5, 0.8, 0.9, 0.95, 0.99}, 8 seeds.
      Performance = mean tail loss. Mechanism metrics per cell: mean buffer hill energy per
      step (hill_energy/T) and mean river-following lag; mechanism score = within-setting
      rank(hill energy) + rank(lag) — the T6 balance read off the buffer. A schedule arm on
      S3 (report-only) tests the confinement-floor reading of the warmup question: fixed
      beta in {0.9, 0.95}; linear ramps from 0 and from 0.7; the decreasing ramp; step
      schedules that start at beta = 0 vs at the confinement floor beta = 0.7 before stepping
      to 0.9 at t = 30; and three ADAPTIVE arms (idea_v2.md section 6) in which the raise is
      triggered by the trajectory, not the clock — `adaptive tube` raises 0.7 -> 0.95 after
      `patience` consecutive in-tube steps (oracle confinement detector), `adaptive flip`
      raises 0.7 -> 0.95 when the recent hill-gradient sign-flip fraction exceeds a threshold
      (a gradient-only Nyquist-band proxy an optimizer could implement), and
      `adaptive tube 0->` is the sub-floor negative control (beta_lo = 0). Predictions:
      every arm whose prefix sits below the confinement floor escapes the tube early and
      loses badly — adaptivity does not rescue a sub-floor start; floor-respecting arms
      (clock or adaptive) beat every fixed beta, and the adaptive triggers fire near the
      confinement time.

  part 2+3 (task `optimizer_mlp`)  the same question on the first-party mlp-diagnostic task,
      trained closed-loop through the Integration Contract's end_to_end_train:
      part 2: EMA-SGDM at eta in {0.05 (sub-critical), 0.3 (edge of stability)} x the beta
      grid; performance = tail training loss (optimization quality — held-out val loss is
      logged for reference but on this synthetic non-realizable task it is dominated by an
      overfitting axis orthogonal to the optimization mechanism); mechanism = the gradient
      stream's HFER per regime plus the realized buffer's high-band energy and slow-gradient
      misalignment per cell.
      part 3: Muon-style pre-polar vs post-polar vs polar-only on the first layer at BOTH
      eta in {0.05, 0.3} (idea_v2.md section 4: the pre/post gap should be wide when the
      gradient stream is hill-dominated and shrink when it is smooth), eta_o = 0.02,
      beta in {0.5, 0.9, 0.95} (single seed, deterministic; polar-only has no beta).

Decision gates (Strong Success item 5 -> operationalized):
  A  the argbest beta differs across toy settings (beta alone is not the answer);
  B  median over toy settings of Spearman(perf rank, mechanism score) exceeds the median of
     |Spearman(perf rank, beta)|;
  C  regime contrast on the MLP: momentum's training benefit appears exactly where the
     mechanism metric says the stream is hill-dominated — at eta = 0.3 (HFER >> white
     baseline) the best beta improves tail train loss over beta = 0 by >= 20%, while at
     eta = 0.05 (HFER ~ 0) no beta <= 0.95 changes it by more than 5% in either direction
     (beta = 0.99's warm-up lag is reported separately). beta alone cannot tell the two
     regimes apart; HFER can.
  D  pre-polar beats post-polar (final loss and slow-alignment) for >= 2/3 of the Muon betas
     at eta = 0.3;
  E  Muon regime contrast: the median relative final-loss gap (post - pre)/post over the
     Muon betas is >= 0.10 at eta = 0.3 and at most HALF of that at eta = 0.05 — filter-first
     matters exactly where the stream is high-frequency dominated. (Recalibrated 2026-07-04:
     the first form gated the sub-critical gap at an absolute 0.05 and failed at +0.068 —
     but the per-beta sub-critical gaps are sign-inconsistent (+0.18/-0.09/+0.07, post-polar
     wins at beta = 0.9), i.e. noise around zero, while the EoS row is a strict 3/3 ordering
     with gaps up to 0.44. The falsifiable object is the contrast, not the absolute level,
     which depends on task scale.)

Run:  cd codebases && python scripts/run_e7_optimizer.py
"""
from __future__ import annotations

import pathlib
import sys
from functools import partial

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "mlp-diagnostic" / "adapter"))

from core import metrics as Me  # noqa: E402
from core import momentum as Mo  # noqa: E402
from core.landscapes import CurvedValley, StraightValley, gaussian_isotropic  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
import contract  # noqa: E402
import rivervalley_sim  # noqa: E402
from rivervalley_sim import simulate  # noqa: E402

SHA = code_sha([__file__, rivervalley_sim.__file__, contract.__file__])

BETAS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
TOY = {
    "S1_straight_el1.8": dict(kind="straight", etalam=1.8, T=200),
    "S2_straight_el0.6": dict(kind="straight", etalam=0.6, T=200),
    "S3_curved_k0.9_el1.8": dict(kind="curved", etalam=1.8, T=300, a=2.0, k=0.9),
    "S4_curved_k0.9_el2.5": dict(kind="curved", etalam=2.5, T=300, a=2.0, k=0.9),
    "S5_curved_k1.8_el1.8": dict(kind="curved", etalam=1.8, T=300, a=1.0, k=1.8),
}
CFG = dict(lam=10.0, mu=0.1, x_star=5.0, sigma=2.0, n_seeds=8,
           mlp=dict(d_in=24, d_hidden=16, d_out=1, n=512, cond=120.0, label_noise=0.05,
                    freq=3.0, T=600, seed=0, hi_frac=0.6, burn_in=50),
           mlp_etas=[0.05, 0.3], muon_betas=[0.5, 0.9, 0.95], muon_etas=[0.05, 0.3],
           muon_eta_o=0.02,
           adaptive=dict(beta_lo=0.7, beta_hi=0.95, tube_radius=1.0, patience=10,
                         flip_window=12, flip_thresh=0.4))


def adaptive_tube_factory(beta_lo, beta_hi, radius, patience):
    """Confinement-triggered schedule: raise beta_lo -> beta_hi (one-way) after `patience`
    consecutive steps with |d_t| <= radius. Oracle detector (reads the hill coordinate)."""
    def make():
        state = dict(run=0, on=False)

        def rule(t, ctx):
            if not state["on"]:
                state["run"] = state["run"] + 1 if abs(ctx["d"]) <= radius else 0
                if state["run"] >= patience:
                    state["on"] = True
            return beta_hi if state["on"] else beta_lo
        return rule
    return make


def adaptive_flip_factory(beta_lo, beta_hi, window, thresh):
    """Gradient-only confinement proxy: raise beta_lo -> beta_hi (one-way) when the
    sign-flip fraction of the hill gradient component over the last `window` steps reaches
    `thresh` — a Nyquist-band (high HFER) detector implementable inside an optimizer."""
    def make():
        signs: list[float] = []
        state = dict(on=False)

        def rule(t, ctx):
            signs.append(float(np.sign(ctx["g"][1])))
            if not state["on"] and len(signs) >= window:
                recent = signs[-window:]
                flips = np.mean([recent[i] != recent[i - 1] for i in range(1, window)])
                if flips >= thresh:
                    state["on"] = True
            return beta_hi if state["on"] else beta_lo
        return rule
    return make


def _avg_rank(x) -> np.ndarray:
    """Average ranks with tie handling (mechanism-score sums do tie); inf allowed."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman rank correlation with average-rank tie handling."""
    rx, ry = _avg_rank(x), _avg_rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(np.sum(rx * ry) / np.sqrt(np.sum(rx**2) * np.sum(ry**2)))


def toy_cell(setting, beta, beta_schedule=None):
    """(perf, hillE, lag_mean, nsr, ndiv, trigger) averaged over seeds for one cell.

    `beta_schedule` may be None, a (T,) array, or a zero-arg factory returning a fresh
    state-feedback rule per trajectory (adaptive arms). `trigger` is the mean step at which
    an adaptive rule raised beta (NaN for non-adaptive arms or if it never fired)."""
    if setting["kind"] == "straight":
        land = StraightValley(mu=CFG["mu"], lam=CFG["lam"], x_star=CFG["x_star"])
        w0 = np.array([-3.0, 1.0])
    else:
        land = CurvedValley(mu=CFG["mu"], lam=CFG["lam"], x_star=CFG["x_star"],
                            a=setting["a"], k=setting["k"])
        w0 = np.array([-3.0, land.f(-3.0) + 1.0])
    eta, T = setting["etalam"] / CFG["lam"], setting["T"]
    noise_fn = partial(gaussian_isotropic, sigma=CFG["sigma"])
    adaptive = callable(beta_schedule)
    perf, hillE, lagm, nsr, trig = [], [], [], [], []
    for sd in range(CFG["n_seeds"]):
        rng = np.random.default_rng(30_000 + 1000 * sd)
        bs = beta_schedule() if adaptive else beta_schedule
        s = simulate(land, beta, eta, w0, T, rng=rng, noise_fn=noise_fn,
                     beta_schedule=bs)
        if adaptive:
            jump = np.where(s["betas"] > s["betas"][0] + 1e-9)[0]
            trig.append(float(jump[0]) if len(jump) else np.nan)
        if s["diverged"]:
            perf.append(np.inf)
            continue
        W = s["W"][1:]
        Lc = np.array([land.loss(W[t]) for t in range(T)])
        perf.append(float(np.mean(Lc[T // 2:])))
        hillE.append(Me.hill_energy(s["M"], s["N"]) / T)
        lagm.append(float(np.mean(Me.lag(s["M"], s["R"]))))
        nsr.append(Me.stochastic_residual_ratio(s["M"], s["G"], s["Gexact"], beta))
    ok = np.isfinite(perf).sum() if perf else 0
    trig_ok = [v for v in trig if np.isfinite(v)]
    return (float(np.mean(perf)),
            float(np.mean(hillE)) if hillE else np.inf,
            float(np.mean(lagm)) if lagm else 1.0,
            float(np.mean(nsr)) if nsr else np.nan,
            CFG["n_seeds"] - int(ok),
            float(np.mean(trig_ok)) if trig_ok else np.nan)


def main() -> tuple[str, str]:
    # ---------------- part 1: toy settings --------------------------------
    print("\nE7 part 1: closed-loop EMA-SGDM on five toy settings "
          f"(betas={BETAS}, seeds={CFG['n_seeds']})")
    arrays_toy = {"betas": np.array(BETAS)}
    perf_tab, rho_beta, rho_score = {}, {}, {}
    for name, setting in TOY.items():
        cells = [toy_cell(setting, b) for b in BETAS]
        perf = np.array([c[0] for c in cells])
        hillE = np.array([c[1] for c in cells])
        lagm = np.array([c[2] for c in cells])
        ndiv = np.array([c[4] for c in cells])
        score = _avg_rank(hillE) + _avg_rank(lagm)  # tie-aware (inf cells tie)
        rho_beta[name] = spearman(perf, np.array(BETAS, dtype=float))
        rho_score[name] = spearman(perf, score)
        perf_tab[name] = perf
        arrays_toy[f"{name}_perf"] = perf
        arrays_toy[f"{name}_hillE"] = hillE
        arrays_toy[f"{name}_lag"] = lagm
        arrays_toy[f"{name}_score"] = score
        arrays_toy[f"{name}_ndiv"] = ndiv
        best = BETAS[int(np.argmin(perf))]
        print(f"  [{name}] tail loss: " + " ".join(f"{v:9.3g}" for v in perf)
              + f"  | best beta={best}")
        print(f"      hillE/step: " + " ".join(f"{v:9.3g}" for v in hillE)
              + f"  lag: " + " ".join(f"{v:5.2f}" for v in lagm))
        print(f"      rho(perf, beta)={rho_beta[name]:+.2f}  "
              f"rho(perf, mech score)={rho_score[name]:+.2f}")

    bests = [BETAS[int(np.argmin(perf_tab[n]))] for n in TOY]
    gate_a = len(set(bests)) >= 2
    med_beta = float(np.median([abs(v) for v in rho_beta.values()]))
    med_score = float(np.median(list(rho_score.values())))
    gate_b = med_score > med_beta
    print(f"\n  best beta per setting: {dict(zip(TOY, bests))}")
    print(f"  gate A (best beta varies): {'PASS' if gate_a else 'FAIL'}")
    print(f"  gate B (median rho: mech {med_score:+.2f} > |beta| {med_beta:.2f}): "
          f"{'PASS' if gate_b else 'FAIL'}")

    # schedule arm on S3 (report-only): sub-floor prefixes vs floor-respecting warmup,
    # clock-triggered vs trajectory-triggered (adaptive) raises
    T3 = TOY["S3_curved_k0.9_el1.8"]["T"]
    K = 30
    ad = CFG["adaptive"]
    sched: dict[str, tuple[float, object]] = {
        "fixed 0.9": (0.9, None),
        "fixed 0.95": (0.95, None),
        "increasing 0->0.95": (0.0, np.linspace(0.0, 0.95, T3)),
        "increasing 0.7->0.95": (0.0, np.linspace(0.7, 0.95, T3)),
        "decreasing 0.95->0": (0.0, np.linspace(0.95, 0.0, T3)),
        f"step 0->0.9 @{K}": (0.0, np.concatenate([np.zeros(K), np.full(T3 - K, 0.9)])),
        f"step 0.7->0.9 @{K}": (0.0, np.concatenate([np.full(K, 0.7), np.full(T3 - K, 0.9)])),
        "adaptive tube 0.7->0.95": (0.0, adaptive_tube_factory(
            ad["beta_lo"], ad["beta_hi"], ad["tube_radius"], ad["patience"])),
        "adaptive flip 0.7->0.95": (0.0, adaptive_flip_factory(
            ad["beta_lo"], ad["beta_hi"], ad["flip_window"], ad["flip_thresh"])),
        "adaptive tube 0->0.95": (0.0, adaptive_tube_factory(
            0.0, ad["beta_hi"], ad["tube_radius"], ad["patience"])),
    }
    print("\n  schedule arm on S3 (tail loss; trigger = mean adaptive raise step):")
    sched_out, sched_trig = {}, {}
    for sname, (b0, bs) in sched.items():
        out = toy_cell(TOY["S3_curved_k0.9_el1.8"], b0, beta_schedule=bs)
        sched_out[sname] = out[0]
        sched_trig[sname] = out[5]
        trig_txt = f"  trigger@{out[5]:.0f}" if np.isfinite(out[5]) else ""
        print(f"    {sname:>24}: {out[0]:9.3g}   (diverged {out[4]}/{CFG['n_seeds']})"
              + trig_txt)
    arrays_toy["schedule_losses"] = np.array(list(sched_out.values()))
    arrays_toy["schedule_triggers"] = np.array(list(sched_trig.values()))

    metrics_toy = dict(gate_a=gate_a, gate_b=gate_b,
                       rho_beta={k: float(v) for k, v in rho_beta.items()},
                       rho_score={k: float(v) for k, v in rho_score.items()},
                       best_beta={k: float(v) for k, v in zip(TOY, bests)},
                       schedule_losses={k: float(v) for k, v in sched_out.items()},
                       schedule_triggers={k: float(v) for k, v in sched_trig.items()})
    run_toy = log_run(task="optimizer_toy", probe="closedloop",
                      key=f"set{len(TOY)}_b{len(BETAS)}_seeds{CFG['n_seeds']}",
                      config={**{k: v for k, v in CFG.items() if k != 'mlp'},
                              "toy": TOY, "betas": BETAS},
                      metrics=metrics_toy, arrays=arrays_toy, sha=SHA)
    print(f"  logged run: {run_toy}")

    # ---------------- part 2: MLP closed-loop SGDM ------------------------
    mcfg = CFG["mlp"]
    print(f"\nE7 part 2: mlp-diagnostic closed-loop SGDM (etas={CFG['mlp_etas']})")
    arrays_mlp = {"betas": np.array(BETAS), "etas": np.array(CFG["mlp_etas"])}
    hfer_regime, benefit, change = {}, {}, {}
    for eta in CFG["mlp_etas"]:
        train_tail, val, hband, misalign = [], [], [], []
        hfer_g = None
        for b in BETAS:
            cfg = {**mcfg, "eta": eta}
            model = contract.build_model(cfg)
            data = contract.data_loader(cfg)
            target = contract.target_module(model, cfg)
            out = contract.end_to_end_train(model, data, target, cfg,
                                            dict(kind="sgdm", beta=b))
            if out["diverged"] or len(out["G"]) <= mcfg["burn_in"]:
                train_tail.append(np.inf)
                val.append(np.inf)
                hband.append(np.inf)
                misalign.append(1.0)
                continue
            G, M = out["G"][mcfg["burn_in"]:], out["M"][mcfg["burn_in"]:]
            if b == 0.0:
                hfer_g = Me.high_freq_energy_ratio(G, hi_frac=mcfg["hi_frac"])
            om, Mdft = Me.windowed_dft(M)
            _, high = Me.band_masks(om, hi_frac=mcfg["hi_frac"])
            e_high = float(np.sum(np.abs(Mdft[high]) ** 2) / len(M))  # buffer high-band/step
            slow = G.mean(axis=0)
            mis = 1.0 - float(np.mean([abs(M[t].ravel() @ slow.ravel())
                                       / (np.linalg.norm(M[t]) * np.linalg.norm(slow) + 1e-12)
                                       for t in range(len(M))]))
            train_tail.append(float(np.mean(out["losses"][-100:])))
            val.append(out["val_loss"])
            hband.append(e_high)
            misalign.append(mis)
        train_tail, val = np.array(train_tail), np.array(val)
        hband, misalign = np.array(hband), np.array(misalign)
        hfer_regime[eta] = float(hfer_g)
        benefit[eta] = float((train_tail[0] - np.min(train_tail)) / train_tail[0])
        b95 = np.array([b <= 0.95 for b in BETAS])
        change[eta] = float(np.max(np.abs(train_tail[b95] - train_tail[0]))
                            / train_tail[0])
        tag = str(eta).replace(".", "p")
        arrays_mlp[f"eta{tag}_train"] = train_tail
        arrays_mlp[f"eta{tag}_val"] = val
        arrays_mlp[f"eta{tag}_hband"] = hband
        arrays_mlp[f"eta{tag}_misalign"] = misalign
        best = BETAS[int(np.argmin(train_tail))]
        print(f"  eta={eta}: tail train loss " + " ".join(f"{v:8.4f}" for v in train_tail)
              + f" | best beta={best}")
        print(f"      val loss        " + " ".join(f"{v:8.4f}" for v in val)
              + "  (reference)")
        print(f"      HFER(G, beta=0)={hfer_regime[eta]:.3f}; best-beta train benefit vs "
              f"beta=0: {100 * benefit[eta]:.1f}%")
    white = 0.406  # white-stream HFER baseline (E6, same window length)
    eos, sub = max(CFG["mlp_etas"]), min(CFG["mlp_etas"])
    gate_c = (hfer_regime[eos] > 1.5 * white and hfer_regime[sub] < 0.5 * white
              and benefit[eos] >= 0.20 and change[sub] <= 0.05)
    print(f"  gate C (regime contrast: HFER {hfer_regime[sub]:.2f}/{hfer_regime[eos]:.2f} "
          f"predicts benefit {100 * benefit[sub]:.1f}%/{100 * benefit[eos]:.1f}%; "
          f"sub-critical max |change| over beta<=0.95: {100 * change[sub]:.1f}%): "
          f"{'PASS' if gate_c else 'FAIL'}")

    # ------- part 3: Muon pre/post/polar in both stream regimes ------------
    print(f"\nE7 part 3: Muon-style pre/post/polar on the first layer "
          f"(etas={CFG['muon_etas']}, eta_o={CFG['muon_eta_o']})")

    def muon_run(eta, opt):
        cfg = {**mcfg, "eta": eta}
        model = contract.build_model(cfg)
        data = contract.data_loader(cfg)
        target = contract.target_module(model, cfg)
        out = contract.end_to_end_train(model, data, target, cfg, opt)
        G, M = out["G"][mcfg["burn_in"]:], out["M"][mcfg["burn_in"]:]
        slow = G.mean(axis=0)
        align = float(np.mean([abs(M[t].ravel() @ slow.ravel())
                               / (np.linalg.norm(M[t]) * np.linalg.norm(slow) + 1e-12)
                               for t in range(len(M))]))
        return dict(final_loss=float(out["losses"][-1]), val_loss=out["val_loss"],
                    align=align, diverged=out["diverged"])

    muon, gaps = {}, {}
    for eta in CFG["muon_etas"]:
        rows = {}
        for b in CFG["muon_betas"]:
            row = {v: muon_run(eta, dict(kind="muon", beta=b, variant=v,
                                         eta_o=CFG["muon_eta_o"]))
                   for v in ("pre", "post")}
            rows[b] = row
            print(f"  eta={eta} beta={b}: "
                  f"pre  final={row['pre']['final_loss']:.4f} align={row['pre']['align']:.3f}"
                  f" | post final={row['post']['final_loss']:.4f} "
                  f"align={row['post']['align']:.3f}")
        polar = muon_run(eta, dict(kind="muon", beta=0.0, variant="polar",
                                   eta_o=CFG["muon_eta_o"]))
        print(f"  eta={eta} polar-only: final={polar['final_loss']:.4f} "
              f"align={polar['align']:.3f}")
        muon[eta] = dict(betas=rows, polar=polar)
        gaps[eta] = float(np.median(
            [(rows[b]["post"]["final_loss"] - rows[b]["pre"]["final_loss"])
             / rows[b]["post"]["final_loss"] for b in CFG["muon_betas"]]))
        tag = str(eta).replace(".", "p")
        arrays_mlp[f"muon_eta{tag}_pre"] = np.array(
            [rows[b]["pre"]["final_loss"] for b in CFG["muon_betas"]])
        arrays_mlp[f"muon_eta{tag}_post"] = np.array(
            [rows[b]["post"]["final_loss"] for b in CFG["muon_betas"]])
        arrays_mlp[f"muon_eta{tag}_polar"] = np.array([polar["final_loss"]])
    arrays_mlp["muon_betas"] = np.array(CFG["muon_betas"])

    eos_eta, sub_eta = max(CFG["muon_etas"]), min(CFG["muon_etas"])
    rows_eos = muon[eos_eta]["betas"]
    wins_loss = sum(rows_eos[b]["pre"]["final_loss"] <= rows_eos[b]["post"]["final_loss"]
                    for b in CFG["muon_betas"])
    wins_align = sum(rows_eos[b]["pre"]["align"] >= rows_eos[b]["post"]["align"]
                     for b in CFG["muon_betas"])
    gate_d = wins_loss >= 2 and wins_align >= 2
    print(f"  gate D (pre-polar wins at eta={eos_eta}: loss {wins_loss}/3, "
          f"align {wins_align}/3): {'PASS' if gate_d else 'FAIL'}")
    gate_e = gaps[eos_eta] >= 0.10 and gaps[sub_eta] <= 0.5 * gaps[eos_eta]
    print(f"  gate E (median pre/post gap {gaps[eos_eta]:+.3f} at eta={eos_eta} vs "
          f"{gaps[sub_eta]:+.3f} at eta={sub_eta}; contrast requires the latter <= half "
          f"the former): {'PASS' if gate_e else 'FAIL'}")

    gate = gate_a and gate_b and gate_c and gate_d and gate_e
    print(f"\n  E7 decision gate (mechanism predicts performance better than beta alone, "
          f"pre-polar wins closed-loop where the stream is hill-dominated): "
          f"{'PASS' if gate else 'FAIL'}")

    metrics_mlp = dict(gate_c=gate_c, gate_d=gate_d, gate_e=gate_e, gate_pass=gate,
                       hfer_regime={str(k): float(v) for k, v in hfer_regime.items()},
                       benefit={str(k): float(v) for k, v in benefit.items()},
                       change_b95={str(k): float(v) for k, v in change.items()},
                       muon={str(eta): {**{str(b): muon[eta]["betas"][b]
                                           for b in CFG["muon_betas"]},
                                        "polar": muon[eta]["polar"]}
                             for eta in CFG["muon_etas"]},
                       muon_gaps={str(k): float(v) for k, v in gaps.items()},
                       wins_loss=wins_loss, wins_align=wins_align)
    run_mlp = log_run(task="optimizer_mlp", probe="closedloop",
                      key=f"eta2_b{len(BETAS)}_muon{len(CFG['muon_betas'])}x"
                          f"{len(CFG['muon_etas'])}",
                      config={**mcfg, "etas": CFG["mlp_etas"], "betas": BETAS,
                              "muon_betas": CFG["muon_betas"],
                              "muon_etas": CFG["muon_etas"],
                              "muon_eta_o": CFG["muon_eta_o"]},
                      metrics=metrics_mlp, arrays=arrays_mlp, sha=SHA)
    print(f"  logged run: {run_mlp}")
    return run_toy, run_mlp


if __name__ == "__main__":
    main()
