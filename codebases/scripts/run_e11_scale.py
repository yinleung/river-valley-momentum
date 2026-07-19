"""Experiment E11 — scale transfer: mini-batch transformer training (idea_v2.md sections
4 and 5; the "remaining risk" of fable_dgs_v1: do hill-dominated streams and the regime
claims survive real mini-batch training?).

Task `nanogpt` (drop-in karpathy/nanoGPT + adapter, CODING.md Pillar 1): a 0.4M-parameter
character-level GPT on shakespeare_char, EMA-SGDM and Muon-style closed-loop training on
Apple-silicon MPS. Batch order is seed-driven, no gradient clipping (divergence guard), EMA
normalization throughout. Config in ../nanogpt/config.yaml; pilot notes there.

Two parts, one logged run:

  part 1 (SGDM sweep)  lr x beta x 3 seeds. Performance = tail training loss (mean of the
      last 200 steps; optimization quality — held-out val loss logged and reported, kept
      separate per the optimization-vs-generalization limitation). Mechanism per (lr, beta):
      the probed matrix's raw-gradient HFER at beta = 0 (the regime label), the realized
      buffer's high-band energy, slow-gradient misalignment, MSR; mechanism score =
      within-row rank(buffer high-band) + rank(misalignment).
      Pilot findings the gates encode: mini-batch streams are hill-dominated at EVERY
      stable lr (HFER 0.87-0.98 vs white 0.41 — progressive sharpening keeps some
      directions at the edge even at small lr, unlike the full-batch MLP whose sub-critical
      stream was smooth), and the realized stream at (lr = 0.4, beta >= 0.9) is SMOOTH
      (HFER ~ 0.2): closed-loop momentum removes the oscillation it filters.

  part 2 (Muon variants)  pre/post/polar on the last block's mlp.c_fc at two background
      lrs, single seed. Both raw streams are hill-dominated (pilot), so filter-first should
      win at both; the pre/post gap is reported against the measured HFER.

Decision gates (predeclared):
  A  hill-dominated streams at scale: HFER(G, beta = 0) >= 1.5 x the white-stream baseline
     at every stable lr in the sweep (the scale-transfer claim itself).
  B  stabilization at scale: at lr = 0.4, beta = 0 diverges on every seed while
     beta in {0.9, 0.95} complete on every seed AND reach tail train loss no worse than
     the best (lr <= 0.2, beta = 0) cell + 10% — momentum extends the usable lr range and
     the extension pays (CL-1 at scale). Whether intermediate betas survive is reported,
     not gated: CL-1 places their thresholds between the two, not below lr = 0.4.
  C  momentum pays where the stream is hill-dominated: at lr = 0.2 the best-beta tail
     train loss improves on beta = 0 by >= 5% (3-seed means).
  D  mechanism score predicts the loss rank: median over stable-lr rows of
     Spearman(tail loss, mechanism score) >= +0.5.
  E  filter-first at scale: pre-polar final loss <= post-polar for >= 2/3 of the Muon betas
     at BOTH background lrs (both streams hill-dominated, so no gap shrinkage is predicted;
     the gap-vs-HFER pairing is reported).

Run:  cd codebases && python scripts/run_e11_scale.py          (~1.5-2 h on MPS)
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "nanogpt" / "adapter"))

from core import metrics as Me  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
import contract  # noqa: E402

SHA = code_sha([__file__, contract.__file__])
CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "nanogpt" / "config.yaml"
HI_FRAC = 0.6
TAIL = 200  # tail-loss window (steps)


def upstream_sha() -> str:
    """Short git SHA of the pristine nanoGPT upstream tree (part of the run identity)."""
    up = CFG_PATH.parent / "upstream"
    try:
        return subprocess.run(["git", "-C", str(up), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_cfg() -> dict:
    with open(CFG_PATH) as f:
        y = yaml.safe_load(f)
    base = dict(**y["model"], **y["data"], **y["train"])
    base["vocab_size"] = contract.vocab_size()
    return dict(base=base, sweep=y["sweep"], muon=y["muon"])


def train_once(base: dict, lr: float, seed: int, opt: dict) -> dict:
    cfg = {**base, "lr": lr, "seed": seed}
    model = contract.build_model(cfg)
    data = contract.data_loader(cfg)
    target = contract.target_module(model, cfg)
    return contract.end_to_end_train(model, data, target, cfg, opt)


def stream_metrics(G: np.ndarray, M: np.ndarray) -> dict:
    """HFER of the raw stream, buffer high-band energy/step, MSR, slow misalignment."""
    hfer = Me.high_freq_energy_ratio(G, hi_frac=HI_FRAC)
    om, Mdft = Me.windowed_dft(M)
    _, high = Me.band_masks(om, hi_frac=HI_FRAC)
    axes = tuple(range(1, Mdft.ndim))
    e_high = float(np.sum(np.sum(np.abs(Mdft) ** 2, axis=axes)[high]) / len(M))
    slow = G.mean(axis=0)
    mis = 1.0 - float(np.mean(
        [abs(M[t].ravel() @ slow.ravel())
         / (np.linalg.norm(M[t]) * np.linalg.norm(slow) + 1e-12) for t in range(len(M))]))
    msr = Me.momentum_suppression_ratio(G, M, hi_frac=HI_FRAC)
    return dict(hfer=hfer, hband=e_high, misalign=mis, msr=msr)


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    return float(np.sum(rx * ry) / den) if den > 0 else np.nan


def main() -> str:
    cfg = load_cfg()
    base, sweep, muon_cfg = cfg["base"], cfg["sweep"], cfg["muon"]
    lrs, betas, seeds = sweep["lrs"], sweep["betas"], sweep["seeds"]
    nL, nB, nS = len(lrs), len(betas), len(seeds)

    # white-stream HFER baseline for the probe window length (declared, seeded)
    rng = np.random.default_rng(0)
    white = Me.high_freq_energy_ratio(
        rng.standard_normal((base["probe_len"], 8, 8)), hi_frac=HI_FRAC)

    # ---------------- part 1: SGDM sweep ------------------------------------
    print(f"\nE11 part 1: nanogpt SGDM sweep (lrs={lrs}, betas={betas}, seeds={seeds}, "
          f"T={base['steps']}, white HFER baseline {white:.3f})")
    train_tail = np.full((nL, nB, nS), np.nan)
    val = np.full((nL, nB, nS), np.nan)
    ndiv = np.zeros((nL, nB), dtype=int)
    hfer0 = np.full(nL, np.nan)          # regime label: raw-stream HFER at beta = 0, seed 0
    hband = np.full((nL, nB), np.nan)    # buffer mechanism metrics (seed 0)
    misalign = np.full((nL, nB), np.nan)
    msr = np.full((nL, nB), np.nan)
    loss_curves = {}
    for li, lr in enumerate(lrs):
        for bi, b in enumerate(betas):
            for si, sd in enumerate(seeds):
                out = train_once(base, lr, sd, dict(kind="sgdm", beta=b))
                if si == 0:
                    loss_curves[f"loss_lr{str(lr).replace('.', 'p')}_b"
                                f"{str(b).replace('.', 'p')}"] = out["losses"]
                if out["diverged"]:
                    ndiv[li, bi] += 1
                    continue
                train_tail[li, bi, si] = float(np.mean(out["losses"][-TAIL:]))
                val[li, bi, si] = out["val_loss"]
                if si == 0:
                    sm = stream_metrics(out["G_win"], out["M_win"])
                    hband[li, bi], misalign[li, bi] = sm["hband"], sm["misalign"]
                    msr[li, bi] = sm["msr"]
                    if b == 0.0:
                        hfer0[li] = sm["hfer"]
        row = np.nanmean(train_tail[li], axis=-1)
        print(f"  lr={lr}: tail train " + " ".join(
            f"{v:7.3f}" if np.isfinite(v) else f"{'div':>7}" for v in row)
            + f"  | HFER(beta=0)={hfer0[li] if np.isfinite(hfer0[li]) else float('nan'):.3f}"
            + f"  ndiv={[int(v) for v in ndiv[li]]}")

    mean_tail = np.nanmean(train_tail, axis=-1)
    mean_val = np.nanmean(val, axis=-1)

    # gate A: hill-dominated streams at every stable lr
    stable_lr = [li for li in range(nL) if ndiv[li, 0] == 0]
    g_a = all(hfer0[li] >= 1.5 * white for li in stable_lr)
    print(f"  gate A (HFER(beta=0) >= 1.5x white at stable lrs "
          f"{[lrs[li] for li in stable_lr]}): {'PASS' if g_a else 'FAIL'}")

    # gate B: stabilization at scale (lr = 0.4 row)
    li4 = lrs.index(0.4)
    best_sub = np.nanmin(mean_tail[:li4, betas.index(0.0)])
    b09, b095 = betas.index(0.9), betas.index(0.95)
    g_b = (ndiv[li4, betas.index(0.0)] == nS
           and ndiv[li4, b09] == 0 and ndiv[li4, b095] == 0
           and max(mean_tail[li4, b09], mean_tail[li4, b095]) <= 1.10 * best_sub)
    print(f"  gate B (lr=0.4: beta=0 diverges, beta 0.9/0.95 run at "
          f"{mean_tail[li4, b09]:.3f}/{mean_tail[li4, b095]:.3f} vs best beta=0 "
          f"{best_sub:.3f}): {'PASS' if g_b else 'FAIL'}")

    # gate C: momentum pays at the hill-dominated edge lr
    li2 = lrs.index(0.2)
    benefit = {}
    for li in range(nL):
        r = mean_tail[li]
        if np.isfinite(r[0]):
            benefit[lrs[li]] = float((r[0] - np.nanmin(r)) / r[0])
    g_c = benefit.get(0.2, 0.0) >= 0.05
    print(f"  benefit vs lr: " + " ".join(f"{lr}:{100 * v:.1f}%"
                                          for lr, v in benefit.items()))
    print(f"  gate C (best-beta benefit >= 5% at lr=0.2): {'PASS' if g_c else 'FAIL'}")

    # gate D: mechanism score predicts the loss rank within stable-lr rows
    rhos = []
    for li in stable_lr:
        fin = np.isfinite(mean_tail[li]) & np.isfinite(hband[li])
        if fin.sum() < 4:
            continue
        score = (np.argsort(np.argsort(hband[li][fin]))
                 + np.argsort(np.argsort(misalign[li][fin]))).astype(float)
        rhos.append(spearman(mean_tail[li][fin], score))
    g_d = len(rhos) > 0 and float(np.median(rhos)) >= 0.5
    print(f"  gate D (median Spearman(loss, mechanism score) = "
          f"{np.median(rhos) if rhos else float('nan'):+.2f} over rows "
          f"{[f'{r:+.2f}' for r in rhos]}): {'PASS' if g_d else 'FAIL'}")

    # ---------------- part 2: Muon variants ---------------------------------
    m_lrs, m_betas, eta_o = muon_cfg["lrs"], muon_cfg["betas"], muon_cfg["eta_o"]
    print(f"\nE11 part 2: Muon pre/post/polar on mlp.c_fc (lrs={m_lrs}, eta_o={eta_o})")
    muon = {}
    muon_arrays = {}
    for lr in m_lrs:
        rows = {}
        for b in m_betas:
            row = {}
            for v in ("pre", "post"):
                out = train_once(base, lr, seeds[0],
                                 dict(kind="muon", beta=b, variant=v, eta_o=eta_o))
                sm = stream_metrics(out["G_win"], out["M_win"]) \
                    if not out["diverged"] else dict(hfer=np.nan, misalign=np.nan)
                row[v] = dict(final_loss=float(out["losses"][-1]) if len(out["losses"])
                              else float("inf"),
                              val_loss=out["val_loss"], diverged=out["diverged"],
                              hfer=sm["hfer"], misalign=sm["misalign"])
            rows[b] = row
            print(f"  lr={lr} beta={b}: pre final={row['pre']['final_loss']:.3f} "
                  f"mis={row['pre']['misalign']:.3f} | post "
                  f"final={row['post']['final_loss']:.3f} mis={row['post']['misalign']:.3f}")
        pol = train_once(base, lr, seeds[0], dict(kind="muon", beta=0.0, variant="polar",
                                                  eta_o=eta_o))
        polar_loss = float(pol["losses"][-1]) if len(pol["losses"]) else float("inf")
        print(f"  lr={lr} polar-only: final={polar_loss:.3f}")
        muon[lr] = dict(rows=rows, polar=polar_loss)
        tag = str(lr).replace(".", "p")
        muon_arrays[f"muon_lr{tag}_pre"] = np.array(
            [rows[b]["pre"]["final_loss"] for b in m_betas])
        muon_arrays[f"muon_lr{tag}_post"] = np.array(
            [rows[b]["post"]["final_loss"] for b in m_betas])
        muon_arrays[f"muon_lr{tag}_polar"] = np.array([polar_loss])

    g_e = True
    gaps = {}
    for lr in m_lrs:
        rows = muon[lr]["rows"]
        wins = sum(rows[b]["pre"]["final_loss"] <= rows[b]["post"]["final_loss"]
                   for b in m_betas)
        gaps[lr] = float(np.median(
            [(rows[b]["post"]["final_loss"] - rows[b]["pre"]["final_loss"])
             / rows[b]["post"]["final_loss"] for b in m_betas]))
        g_e &= wins >= 2
        print(f"  lr={lr}: pre-polar wins {wins}/{len(m_betas)}, median gap "
              f"{gaps[lr]:+.3f}")
    print(f"  gate E (pre-polar wins >= 2/3 at both lrs): {'PASS' if g_e else 'FAIL'}")

    gate = g_a and g_b and g_c and g_d and g_e
    print(f"\n  E11 decision gate: {'PASS' if gate else 'FAIL'}")

    arrays = dict(lrs=np.array(lrs), betas=np.array(betas),
                  train_tail=train_tail, val=val, ndiv=ndiv.astype(float),
                  hfer0=hfer0, hband=hband, misalign=misalign, msr=msr,
                  muon_betas=np.array(m_betas), muon_lrs=np.array(m_lrs),
                  white_baseline=np.array([white]), **muon_arrays, **loss_curves)
    metrics = dict(gate_pass=gate, gate_a=g_a, gate_b=g_b, gate_c=g_c, gate_d=g_d,
                   gate_e=g_e, white_baseline=white,
                   hfer0={str(lr): float(h) for lr, h in zip(lrs, hfer0)},
                   benefit={str(k): float(v) for k, v in benefit.items()},
                   rho_rows=[float(r) for r in rhos],
                   muon_gaps={str(k): float(v) for k, v in gaps.items()},
                   muon_detail={str(lr): {**{str(b): muon[lr]["rows"][b]
                                             for b in m_betas},
                                          "polar": muon[lr]["polar"]}
                                for lr in m_lrs},
                   mean_val_best={str(lrs[li]): float(np.nanmin(mean_val[li]))
                                  for li in range(nL)})
    run_id = log_run(task="nanogpt", probe="closedloop",
                     key=f"lr{nL}_b{nB}_seeds{nS}_T{base['steps']}",
                     config={**base, **sweep, "muon": muon_cfg,
                             "upstream_sha": upstream_sha()},
                     metrics=metrics, arrays=arrays, sha=SHA)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
