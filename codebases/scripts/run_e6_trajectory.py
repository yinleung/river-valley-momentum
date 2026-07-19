"""Experiment E6 — real-network trajectory-gradient frequency diagnostic (idea_v1.md, Phase 4).

Trains the first-party `mlp-diagnostic` task (tiny 2-layer MLP, ill-conditioned regression) full-
batch by gradient descent near the edge of stability, records the first-layer weight gradient
G_1..G_T, and asks whether real gradients carry the slow/fast temporal structure the toy theory
predicts. The non-trivial questions (the |H_beta| match is automatic by linearity of the EMA):

  1. does the gradient stream carry non-trivial high-frequency temporal energy (HFER), and is it
     concentrated near omega=pi like the river-valley hill?
  2. does open-loop EMA filtering improve alignment of the update with the slow (mean) gradient?

Decision gate (idea_v1.md): real gradients show stable slow/fast structure — here, HFER above a
white-stream baseline AND momentum improving slow-gradient alignment.

Confinement-onset extension (fable_dgs_v1.md §8.2 item 4): the same run, analysed in sliding
windows over the FULL stream (no burn-in drop), asks when the high-frequency structure
appears. The river-valley reading of li2025frequency's "high frequency helps early" is that
before the iterate is confined to the valley tube the steep-direction gradient IS the useful
descent signal (low temporal frequency, large magnitude), and only after entry does it become
the Nyquist oscillation that filtering should remove. Per window: HFER, mean loss, and the
open-loop EMA slow-alignment gain (align(m) - align(g) against the window mean). Onset =
first window with HFER >= 0.9. Gates:
    onset-a  the first window's HFER is below 0.5 (structure is not there from step 0);
    onset-b  at least half of the total loss decrease happens before the onset window;
    onset-c  the post-onset windows all have HFER >= 0.9 (stable oscillation regime);
    onset-d  the EMA alignment gain is larger after onset than before (filtering pays once
             confined).

Run:  cd codebases && python scripts/run_e6_trajectory.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "mlp-diagnostic" / "adapter"))

from core import metrics as Me  # noqa: E402
from core import momentum as Mo  # noqa: E402
from core import probe  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
import contract  # noqa: E402

BETAS = [0.5, 0.9, 0.95, 0.99]
CFG = dict(d_in=24, d_hidden=16, d_out=1, n=512, cond=120.0, label_noise=0.05, freq=3.0,
           eta=0.3, T=600, seed=0, hi_frac=0.6, burn_in=50,
           win_len=100, win_stride=50, onset_hfer=0.9, onset_beta=0.9)


def cosine_to_slow(stream: np.ndarray, slow: np.ndarray) -> float:
    """Mean cosine similarity of a (T,m,n) stream to a fixed slow matrix (flattened)."""
    s = slow.ravel()
    s = s / (np.linalg.norm(s) + 1e-12)
    out = []
    for t in range(stream.shape[0]):
        v = stream[t].ravel()
        out.append(abs(v @ s) / (np.linalg.norm(v) + 1e-12))
    return float(np.mean(out))


def main() -> str:
    model = contract.build_model(CFG)
    data = contract.data_loader(CFG)
    target = contract.target_module(model, CFG)
    step_fn = contract.make_step_fn(model, data, target, CFG)

    losses = []

    def step_with_loss():
        losses.append(contract.full_loss(model, data))
        return step_fn()

    loss0 = contract.full_loss(model, data)
    G_full = probe.collect_trajectory(step_with_loss, CFG["T"])  # (T, d_hidden, d_in)
    loss1 = contract.full_loss(model, data)
    loss_curve = np.asarray(losses)
    G = G_full[CFG["burn_in"]:]  # drop the initial fast decay; keep the stationary oscillation
    print(f"\nE6 mlp-diagnostic  (cond={CFG['cond']}, eta={CFG['eta']}, T={CFG['T']})")
    print(f"  loss {loss0:.4f} -> {loss1:.4f}   target gradient stream shape {G.shape}")

    # --- (1) high-frequency temporal energy vs a white-stream baseline -------
    hfer = Me.high_freq_energy_ratio(G, hi_frac=CFG["hi_frac"])
    rng = np.random.default_rng(0)
    white = rng.standard_normal(G.shape)
    hfer_white = Me.high_freq_energy_ratio(white, hi_frac=CFG["hi_frac"])
    # Frobenius temporal spectrum, to locate the dominant frequency
    om, Gdft = Me.windowed_dft(G, window="hann")
    spec = np.sqrt(np.sum(np.abs(Gdft) ** 2, axis=(1, 2)))
    peak = float(om[1:][np.argmax(spec[1:])] / np.pi)  # exclude DC

    # --- (2) does EMA improve alignment with the slow (mean) gradient? --------
    slow = G.mean(axis=0)
    align_g = cosine_to_slow(G, slow)
    msr, align_m = {}, {}
    for b in BETAS:
        M = Mo.ema_momentum(G, b)
        msr[b] = Me.momentum_suppression_ratio(G, M, hi_frac=CFG["hi_frac"])
        align_m[b] = cosine_to_slow(M, slow)

    # --- report + gate ------------------------------------------------------
    print(f"\n  (1) temporal HFER(G) = {hfer:.3f}   (white-stream baseline {hfer_white:.3f});"
          f"  Frobenius spectral peak at omega/pi = {peak:.3f}")
    print(f"  (2) slow-gradient alignment: raw g = {align_g:.3f}")
    print(f"      {'beta':>6} {'MSR(high)':>10} {'|H(pi)|^2':>10} {'align(m)':>9}")
    for b in BETAS:
        print(f"      {b:>6} {msr[b]:>10.2e} {((1-b)/(1+b))**2:>10.2e} {align_m[b]:>9.3f}")

    hf_present = hfer > 1.5 * hfer_white
    align_improves = any(align_m[b] > align_g + 1e-3 for b in BETAS)
    gate = hf_present and align_improves
    print(f"\n  high-freq structure present (HFER > 1.5x white): {hf_present}")
    print(f"  momentum improves slow-gradient alignment: {align_improves} "
          f"(best align(m)={max(align_m.values()):.3f} vs align(g)={align_g:.3f})")
    print(f"  decision gate (real gradients show slow/fast structure): {'PASS' if gate else 'FAIL'}")

    # --- (3) confinement onset: windowed re-analysis of the full stream ------
    W, St = CFG["win_len"], CFG["win_stride"]
    starts = np.arange(0, CFG["T"] - W + 1, St)
    hfer_w, dalign_w, loss_w = [], [], []
    for s0 in starts:
        Gw = G_full[s0:s0 + W]
        hfer_w.append(Me.high_freq_energy_ratio(Gw, hi_frac=CFG["hi_frac"]))
        slow_w = Gw.mean(axis=0)
        Mw = Mo.ema_momentum(Gw, CFG["onset_beta"])
        skip = W // 5  # drop the in-window EMA warm-up
        dalign_w.append(cosine_to_slow(Mw[skip:], slow_w) - cosine_to_slow(Gw[skip:], slow_w))
        loss_w.append(float(loss_curve[s0:s0 + W].mean()))
    hfer_w, dalign_w, loss_w = map(np.asarray, (hfer_w, dalign_w, loss_w))
    onset_idx = int(np.argmax(hfer_w >= CFG["onset_hfer"])) if np.any(
        hfer_w >= CFG["onset_hfer"]) else len(starts)
    onset_step = int(starts[onset_idx]) if onset_idx < len(starts) else CFG["T"]
    drop_total = loss_curve[0] - loss_curve[-1]
    drop_before = loss_curve[0] - loss_curve[min(onset_step, CFG["T"] - 1)]
    frac_before = float(drop_before / drop_total)
    print(f"\n  (3) confinement onset (windows len={W} stride={St}, onset = HFER >= "
          f"{CFG['onset_hfer']})")
    print(f"      window start: " + " ".join(f"{s:>6}" for s in starts))
    print(f"      HFER        : " + " ".join(f"{v:>6.3f}" for v in hfer_w))
    print(f"      dAlign(m-g) : " + " ".join(f"{v:>+6.3f}" for v in dalign_w))
    print(f"      mean loss   : " + " ".join(f"{v:>6.3f}" for v in loss_w))
    ga = hfer_w[0] < 0.5
    gb = frac_before >= 0.5
    gc = bool(np.all(hfer_w[onset_idx:] >= CFG["onset_hfer"])) if onset_idx < len(starts) else False
    gd = (float(np.mean(dalign_w[onset_idx:])) > float(np.mean(dalign_w[:max(onset_idx, 1)]))) \
        if 0 < onset_idx < len(starts) else False
    print(f"      onset at window {onset_idx} (step {onset_step}); loss-drop fraction before "
          f"onset = {frac_before:.2f}")
    print(f"      gates: a(first-window HFER<0.5)={ga} b(>=half drop before)={gb} "
          f"c(post-onset HFER>={CFG['onset_hfer']})={gc} d(dAlign larger after)={gd}"
          f"  -> {'PASS' if ga and gb and gc and gd else 'FAIL'}")
    onset_gate = ga and gb and gc and gd

    # --- log ----------------------------------------------------------------
    arrays = {"G_spectrum": spec, "omega": om, "betas": np.array(BETAS),
              "G_meanabs": np.abs(G).mean(axis=(1, 2)),
              "loss_curve": loss_curve, "win_starts": starts, "win_hfer": hfer_w,
              "win_dalign": dalign_w, "win_loss": loss_w}
    metrics = {
        "gate_pass": gate, "hfer": hfer, "hfer_white": hfer_white, "spectral_peak": peak,
        "align_g": align_g, "loss0": loss0, "loss1": loss1,
        "MSR": {str(b): msr[b] for b in BETAS},
        "align_m": {str(b): align_m[b] for b in BETAS},
        "onset_gate_pass": onset_gate, "onset_window": onset_idx, "onset_step": onset_step,
        "loss_drop_frac_before_onset": frac_before,
        "hfer_first_window": float(hfer_w[0]),
        "dalign_pre_onset": float(np.mean(dalign_w[:max(onset_idx, 1)])),
        "dalign_post_onset": float(np.mean(dalign_w[onset_idx:])) if onset_idx < len(starts)
        else float("nan"),
    }
    sha = code_sha([__file__, contract.__file__, probe.__file__])
    run_id = log_run(task="mlp_diagnostic", probe="trajectory",
                     key=f"cond{int(CFG['cond'])}_eta{str(CFG['eta']).replace('.', 'p')}_T{CFG['T']}",
                     config={**CFG, "betas": BETAS}, metrics=metrics, arrays=arrays, sha=sha)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
