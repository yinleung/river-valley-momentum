"""Experiment E12 — decomposition of the mini-batch stream (review_v3 "single most
important diagnostic"): is E11's high-band energy landscape-driven hill oscillation or
sampling noise?

Task `nanogpt`, probe `decomp` (adapter/decomp.py; contract.py untouched). At every stable
E11 learning rate, plain-SGD (beta = 0) runs record, over the same probe window as E11, the
probed matrix's mini-batch gradient g^mb_t together with a large-batch estimate ghat^LB_t of
the mean-loss gradient at the same iterate, giving the split

    g^mb_t = ghat^LB_t + xihat_t     (state stream + sampling residual),

plus the top-k eigenpairs of the target-restricted Hessian at the window midpoint (the
measured hill directions). Analysis (all in this driver; the cache stores reduced arrays):

  temporal   HFER of g^mb / ghat^LB / xihat; high-band energy share of the state stream
             share_LB = Eh(LB) / (Eh(LB) + Eh(xi)); cross-term fraction as a residual check.
             A temporally independent residual is spectrally flat, so HFER(xihat) should sit
             at the white baseline whatever its spatial structure; HFER above white must
             come from the state stream.
  geometric  project the state stream on the top-k curvature subspace: HFER inside vs
             outside, high-band energy share captured by k of m*n dimensions, per-direction
             HFER_i against eta*lambda_i, and eta*lambda_top per lr (edge proximity).

LB estimator bias (declared): xihat = xi - (LB sampling error), so E|xihat|^2 overstates
E|xi|^2 by a factor (1 + batch/lb_batch) = 1.03 and the LB stream carries a white
contamination of relative energy batch/lb_batch = 1/32 of the residual's; both push
AGAINST gate A (they move energy toward the residual and whiten the LB stream), so the
gate is conservative.

Decision gates (predeclared before the lr=0.2/seed-0 pilot and left unchanged by it;
pilot numbers in results/figures/e12_decomp/summary.md):
  A  state-stream dominance: share_LB >= 0.5 at every stable lr (3-seed mean) -- the
     majority of the raw stream's high-band energy is in the mean-loss gradient at the
     visited iterates, not in the sampling residual.
  B  residual whiteness: |HFER(xihat) - white| <= 0.05 at every lr (3-seed mean) -- the
     split is clean; the residual carries no temporal structure.
  C  curvature alignment: at every lr (3-seed mean), HFER(LB, in-subspace) >=
     HFER(LB, out-of-subspace) + 0.10 AND the top-k subspace (k/(m n) = 2.4e-4 of
     dimensions) captures >= 0.25 of the state stream's high-band energy.
  D  (report, not gated) eta*lambda_i spectrum per lr and per-direction HFER_i: the edge
     proximity that E11 attributed to progressive sharpening, now measured.

Run:  cd codebases && python scripts/run_e12_decomp.py [--pilot]     (~45 min full, MPS)
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "nanogpt" / "adapter"))

from core import metrics as Me  # noqa: E402
from core.logging import code_sha, log_run  # noqa: E402
import contract  # noqa: E402
import decomp  # noqa: E402

SHA = code_sha([__file__, decomp.__file__, contract.__file__])
CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "nanogpt" / "config.yaml"
HI_FRAC = 0.6
LRS = [0.05, 0.1, 0.2]          # the stable E11 lrs (beta = 0 completes on all seeds)
SEEDS = [0, 1, 2]
PROBE = dict(lb_batch=1024, eig_k=16, eig_iters=50, eig_batch=512, eig_chunk=128)


def load_base() -> dict:
    with open(CFG_PATH) as f:
        y = yaml.safe_load(f)
    base = dict(**y["model"], **y["data"], **y["train"])
    base["vocab_size"] = contract.vocab_size()
    return {**base, **PROBE}


def band_energy(x: np.ndarray) -> tuple[float, float]:
    """(high-band energy, total non-DC energy) of a stream, Frobenius over non-time axes."""
    omega, X = Me.windowed_dft(x)
    _, high = Me.band_masks(omega, hi_frac=HI_FRAC)
    axes = tuple(range(1, X.ndim))
    p = np.sum(np.abs(X) ** 2, axis=axes) if X.ndim > 1 else np.abs(X) ** 2
    return float(np.sum(p[high])), float(np.sum(p[omega > 0]))


def freq_energy(x: np.ndarray) -> np.ndarray:
    """Per-frequency Frobenius energy (for the spectrum panel)."""
    _, X = Me.windowed_dft(x)
    axes = tuple(range(1, X.ndim))
    return np.sum(np.abs(X) ** 2, axis=axes) if X.ndim > 1 else np.abs(X) ** 2


def analyse(out: dict, lr: float) -> dict:
    """All E12 reductions for one run; returns scalars + small arrays."""
    G_mb, G_lb = out["G_mb"].astype(np.float64), out["G_lb"].astype(np.float64)
    xi = G_mb - G_lb
    V = out["eigvecs"].astype(np.float64)                    # (k, m, n), orthonormal
    k = V.shape[0]
    hfer = {s: Me.high_freq_energy_ratio(a, hi_frac=HI_FRAC)
            for s, a in (("mb", G_mb), ("lb", G_lb), ("xi", xi))}
    eh = {s: band_energy(a)[0] for s, a in (("mb", G_mb), ("lb", G_lb), ("xi", xi))}
    share_lb = eh["lb"] / (eh["lb"] + eh["xi"])
    cross = (eh["mb"] - eh["lb"] - eh["xi"]) / eh["mb"]
    # curvature projection of the state stream
    c = np.einsum("tmn,kmn->tk", G_lb, V)                    # in-subspace coefficients
    r = G_lb - np.einsum("tk,kmn->tmn", c, V)                # out-of-subspace remainder
    hfer_in, hfer_out = (Me.high_freq_energy_ratio(a, hi_frac=HI_FRAC) for a in (c, r))
    eh_in, eh_out = band_energy(c)[0], band_energy(r)[0]
    share_high_in = eh_in / (eh_in + eh_out)
    c_xi = np.einsum("tmn,kmn->tk", xi, V)
    hfer_dir = np.array([Me.high_freq_energy_ratio(c[:, i], hi_frac=HI_FRAC)
                         for i in range(k)])
    return dict(
        hfer_mb=hfer["mb"], hfer_lb=hfer["lb"], hfer_xi=hfer["xi"],
        share_lb=share_lb, cross=cross,
        hfer_in=hfer_in, hfer_out=hfer_out, share_high_in=share_high_in,
        hfer_xi_in=Me.high_freq_energy_ratio(c_xi, hi_frac=HI_FRAC),
        hfer_dir=hfer_dir, eigvals=out["eigvals"], eig_resid=out["eig_resid"],
        eta_lam=lr * out["eigvals"],
        energy_lb=float(np.sum(G_lb ** 2)), energy_xi=float(np.sum(xi ** 2)),
        spec_mb=freq_energy(G_mb), spec_lb=freq_energy(G_lb), spec_xi=freq_energy(xi),
        tail_loss=float(np.mean(out["losses"][-200:])), eig_device=out["eig_device"])


def print_run(tag: str, a: dict) -> None:
    print(f"  {tag}: HFER mb/lb/xi = {a['hfer_mb']:.3f}/{a['hfer_lb']:.3f}/"
          f"{a['hfer_xi']:.3f}  share_LB={a['share_lb']:.3f}  cross={a['cross']:+.3f}")
    print(f"    curvature: HFER in/out = {a['hfer_in']:.3f}/{a['hfer_out']:.3f}  "
          f"share_high_in={a['share_high_in']:.3f}  eta*lam top3 = "
          + " ".join(f"{v:.2f}" for v in a["eta_lam"][:3])
          + f"  resid top3 = " + " ".join(f"{v:.3f}" for v in a["eig_resid"][:3])
          + f"  [{a['eig_device']}]")


def main(pilot: bool = False) -> str | None:
    base = load_base()
    rng = np.random.default_rng(0)
    white = Me.high_freq_energy_ratio(
        rng.standard_normal((base["probe_len"], 8, 8)), hi_frac=HI_FRAC)
    print(f"E12 decomposition (lrs={LRS}, seeds={SEEDS}, T={base['steps']}, "
          f"lb_batch={PROBE['lb_batch']}, k={PROBE['eig_k']}, white={white:.3f})")

    if pilot:
        t0 = time.time()
        out = decomp.run_decomp({**base, "lr": 0.2, "seed": 0})
        a = analyse(out, 0.2)
        print_run(f"pilot lr=0.2 seed=0 ({time.time() - t0:.0f}s)", a)
        return None

    runs: dict[float, list[dict]] = {lr: [] for lr in LRS}
    for lr in LRS:
        for sd in SEEDS:
            t0 = time.time()
            out = decomp.run_decomp({**base, "lr": lr, "seed": sd})
            if out["diverged"]:
                raise RuntimeError(f"unexpected divergence at lr={lr}, seed={sd}")
            a = analyse(out, lr)
            runs[lr].append(a)
            print_run(f"lr={lr} seed={sd} ({time.time() - t0:.0f}s)", a)

    def seed_mean(lr, key):
        return float(np.mean([r[key] for r in runs[lr]]))

    # gates
    g_a = all(seed_mean(lr, "share_lb") >= 0.5 for lr in LRS)
    g_b = all(abs(seed_mean(lr, "hfer_xi") - white) <= 0.05 for lr in LRS)
    g_c = all(seed_mean(lr, "hfer_in") >= seed_mean(lr, "hfer_out") + 0.10
              and seed_mean(lr, "share_high_in") >= 0.25 for lr in LRS)
    print(f"\n  gate A (share_LB >= 0.5 at every lr): "
          f"{[round(seed_mean(lr, 'share_lb'), 3) for lr in LRS]} "
          f"{'PASS' if g_a else 'FAIL'}")
    print(f"  gate B (|HFER(xi) - white| <= 0.05): "
          f"{[round(seed_mean(lr, 'hfer_xi'), 3) for lr in LRS]} "
          f"{'PASS' if g_b else 'FAIL'}")
    print(f"  gate C (HFER in >= out + 0.10 and share_high_in >= 0.25): "
          f"in {[round(seed_mean(lr, 'hfer_in'), 3) for lr in LRS]} "
          f"out {[round(seed_mean(lr, 'hfer_out'), 3) for lr in LRS]} "
          f"share {[round(seed_mean(lr, 'share_high_in'), 3) for lr in LRS]} "
          f"{'PASS' if g_c else 'FAIL'}")
    print(f"  report D (eta*lam top per lr): "
          f"{[round(float(np.mean([r['eta_lam'][0] for r in runs[lr]])), 2) for lr in LRS]}")
    gate = g_a and g_b and g_c
    print(f"\n  E12 decision gate: {'PASS' if gate else 'FAIL'}")

    k = PROBE["eig_k"]
    arrays = dict(
        lrs=np.array(LRS), seeds=np.array(SEEDS), white_baseline=np.array([white]),
        hfer_mb=np.array([[r["hfer_mb"] for r in runs[lr]] for lr in LRS]),
        hfer_lb=np.array([[r["hfer_lb"] for r in runs[lr]] for lr in LRS]),
        hfer_xi=np.array([[r["hfer_xi"] for r in runs[lr]] for lr in LRS]),
        share_lb=np.array([[r["share_lb"] for r in runs[lr]] for lr in LRS]),
        cross=np.array([[r["cross"] for r in runs[lr]] for lr in LRS]),
        hfer_in=np.array([[r["hfer_in"] for r in runs[lr]] for lr in LRS]),
        hfer_out=np.array([[r["hfer_out"] for r in runs[lr]] for lr in LRS]),
        hfer_xi_in=np.array([[r["hfer_xi_in"] for r in runs[lr]] for lr in LRS]),
        share_high_in=np.array([[r["share_high_in"] for r in runs[lr]] for lr in LRS]),
        energy_lb=np.array([[r["energy_lb"] for r in runs[lr]] for lr in LRS]),
        energy_xi=np.array([[r["energy_xi"] for r in runs[lr]] for lr in LRS]),
        hfer_dir=np.array([[r["hfer_dir"] for r in runs[lr]] for lr in LRS]),
        eta_lam=np.array([[r["eta_lam"] for r in runs[lr]] for lr in LRS]),
        eig_resid=np.array([[r["eig_resid"] for r in runs[lr]] for lr in LRS]),
        spec_mb=np.array([[r["spec_mb"] for r in runs[lr]] for lr in LRS]),
        spec_lb=np.array([[r["spec_lb"] for r in runs[lr]] for lr in LRS]),
        spec_xi=np.array([[r["spec_xi"] for r in runs[lr]] for lr in LRS]))
    metrics = dict(
        gate_pass=gate, gate_a=g_a, gate_b=g_b, gate_c=g_c, white_baseline=white,
        share_lb={str(lr): seed_mean(lr, "share_lb") for lr in LRS},
        hfer_mb={str(lr): seed_mean(lr, "hfer_mb") for lr in LRS},
        hfer_lb={str(lr): seed_mean(lr, "hfer_lb") for lr in LRS},
        hfer_xi={str(lr): seed_mean(lr, "hfer_xi") for lr in LRS},
        hfer_in={str(lr): seed_mean(lr, "hfer_in") for lr in LRS},
        hfer_out={str(lr): seed_mean(lr, "hfer_out") for lr in LRS},
        share_high_in={str(lr): seed_mean(lr, "share_high_in") for lr in LRS},
        cross_max=float(np.max([[abs(r["cross"]) for r in runs[lr]] for lr in LRS])),
        eta_lam_top={str(lr): float(np.mean([r["eta_lam"][0] for r in runs[lr]]))
                     for lr in LRS},
        eig_resid_top={str(lr): float(np.mean([r["eig_resid"][0] for r in runs[lr]]))
                       for lr in LRS},
        eig_device=runs[LRS[0]][0]["eig_device"],
        lb_batch=PROBE["lb_batch"], eig_k=k, tail_loss={
            str(lr): seed_mean(lr, "tail_loss") for lr in LRS})
    run_id = log_run(task="nanogpt", probe="decomp",
                     key=f"lr{len(LRS)}_seeds{len(SEEDS)}_LB{PROBE['lb_batch']}_k{k}",
                     config={**base, "lrs": LRS, "seeds": SEEDS},
                     metrics=metrics, arrays=arrays, sha=SHA)
    print(f"\n  logged run: {run_id}")
    return run_id


if __name__ == "__main__":
    main(pilot="--pilot" in sys.argv)
