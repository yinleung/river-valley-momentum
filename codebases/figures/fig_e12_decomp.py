"""Figure E12 — decomposition of the mini-batch stream: state vs residual, curvature split.

Pure module: reads the cached nanogpt decomp run -> out/fig_e12_decomp.{pdf,png}.
    cd figures && python fig_e12_decomp.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e12_decomp"
HI_FRAC = 0.6
LR_COLORS = {0.05: "#93c5fd", 0.1: "#3b82f6", 0.2: "#1e3a8a"}


def build() -> None:
    style.apply_style()
    rec = _data.load_task("nanogpt", probe="decomp")
    A = rec["arrays"]
    lrs = list(A["lrs"])
    white = float(A["white_baseline"][0])
    fig, ax = plt.subplots(1, 4, figsize=(19.5, 4.6))

    # (a) per-frequency Frobenius energy at lr = 0.2, seed mean
    li = lrs.index(0.2)
    F = A["spec_mb"].shape[-1]
    omega = np.linspace(0.0, 1.0, F)  # omega/pi on the rfft grid
    for key in ("mb", "lb", "xi"):
        st = style.DECOMP_STYLE[key]
        ax[0].plot(omega[1:], A[f"spec_{key}"][li].mean(axis=0)[1:],
                   color=st["color"], lw=style.LW.PRIMARY, label=st["label"])
    ax[0].axvspan(HI_FRAC, 1.0, color="0.92")
    ax[0].set(xlabel=r"$\omega/\pi$", ylabel="Frobenius energy", yscale="log",
              title="(a) spectra at lr=0.2")
    ax[0].legend()

    # (b) HFER of the three streams per lr, white baseline dashed
    x = np.arange(len(lrs))
    wd = 0.26
    for j, key in enumerate(("mb", "lb", "xi")):
        st = style.DECOMP_STYLE[key]
        vals = A[f"hfer_{key}"]
        ax[1].bar(x + (j - 1) * wd, vals.mean(axis=1), wd, color=st["color"],
                  label=st["label"])
        for si in range(vals.shape[1]):
            ax[1].plot(x + (j - 1) * wd, vals[:, si], ls="none", marker="o", ms=3,
                       color="k", alpha=0.5)
    ax[1].axhline(white, color=style.ROLE_COLORS["guide"], ls="--", lw=style.LW.GUIDE,
                  label="white")
    ax[1].set(xticks=x, xticklabels=[str(v) for v in lrs], xlabel="learning rate",
              ylabel="HFER", title="(b) HFER by stream")
    ax[1].legend(fontsize=10)

    # (c) high-band shares per lr: state share, then in-subspace share of the state stream
    for j, (key, lab, col) in enumerate((
            ("share_lb", "state share of high band", "#ea580c"),
            ("share_high_in", "top-16 subspace share of state high band", "#7c3aed"))):
        vals = A[key]
        ax[2].bar(x + (j - 0.5) * wd, vals.mean(axis=1), wd, color=col, label=lab)
        for si in range(vals.shape[1]):
            ax[2].plot(x + (j - 0.5) * wd, vals[:, si], ls="none", marker="o", ms=3,
                       color="k", alpha=0.5)
    ax[2].axhline(0.5, color=style.ROLE_COLORS["guide"], ls=":", lw=1.2)
    ax[2].set(xticks=x, xticklabels=[str(v) for v in lrs], xlabel="learning rate",
              ylabel="high-band energy share", ylim=(0, 1.05), title="(c) where the high band lives")
    ax[2].legend(fontsize=9)

    # (d) per-direction HFER of the state stream vs eta*lambda_i (restricted eigenvalues)
    for li_, lr in enumerate(lrs):
        ax[3].plot(A["eta_lam"][li_].ravel(), A["hfer_dir"][li_].ravel(), ls="none",
                   marker="o", ms=5, alpha=0.75, color=LR_COLORS[float(lr)],
                   label=f"lr={lr}")
    ax[3].axhline(white, color=style.ROLE_COLORS["guide"], ls="--", lw=style.LW.GUIDE,
                  label="white")
    ax[3].set(xlabel=r"$\eta\lambda_i$ (restricted)", ylabel=r"HFER of $\langle\bar g^{\mathrm{LB}},v_i\rangle$",
              title="(d) per-direction HFER")
    ax[3].legend(fontsize=10)

    fig.tight_layout()
    style.save_figure(fig, LABEL)
    print(f"wrote out/{LABEL}.pdf/.png from {rec['run_id']}")


if __name__ == "__main__":
    build()
