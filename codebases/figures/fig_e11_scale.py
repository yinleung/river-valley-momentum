"""Figure E11 — scale transfer: mini-batch nanoGPT training.

Pure module: reads the cached nanogpt run -> out/fig_e11_scale.{pdf,png}.
    cd figures && python fig_e11_scale.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e11_scale"
LR_COLORS = {0.05: "#0891b2", 0.1: "#1d4ed8", 0.2: "#7c3aed", 0.4: "#dc2626"}


def build() -> None:
    style.apply_style()
    rec = _data.load_task("nanogpt", probe="closedloop")
    A = rec["arrays"]
    lrs = [float(v) for v in A["lrs"]]
    betas = [float(v) for v in A["betas"]]
    n_seeds = A["train_tail"].shape[-1]

    fig, ax = plt.subplots(2, 2, figsize=(13.8, 9.2))

    # (a) tail train loss vs beta per lr (seed mean +- std; diverged cells absent)
    for li, lr in enumerate(lrs):
        m = np.nanmean(A["train_tail"][li], axis=-1)
        s = np.nanstd(A["train_tail"][li], axis=-1)
        hf = A["hfer0"][li]
        lab = rf"lr={lr}" + (f" (HFER {hf:.2f})" if np.isfinite(hf) else " (β=0 div)")
        ax[0, 0].errorbar(betas, m, yerr=s, marker="o", color=LR_COLORS[lr],
                          lw=style.LW.PRIMARY, label=lab)
    ax[0, 0].set(xlabel="β", ylabel="tail training loss",
                 title=f"(a) SGDM sweep ({n_seeds} seeds; surviving-seed means)")
    ax[0, 0].legend(fontsize=9)

    # (b) the stabilization row: loss curves at lr = 0.4
    for b, alpha in ((0.0, 1.0), (0.5, 0.75), (0.9, 1.0), (0.95, 1.0)):
        key = f"loss_lr0p4_b{str(b).replace('.', 'p')}"
        if key not in A:
            continue
        curve = A[key]
        c = style.beta_color(b)
        ax[0, 1].plot(np.arange(len(curve)), curve, color=c, lw=style.LW.PRIMARY,
                      alpha=alpha, label=f"β={b}")
        if len(curve) < rec["config"]["steps"]:
            ax[0, 1].plot([len(curve) - 1], [curve[-1]], marker="x", ms=11, mew=2.5,
                          color=c)
    ax[0, 1].set(xlabel="step", ylabel="training-batch loss", yscale="log",
                 title="(b) lr=0.4 (seed 0): β=0 diverges (×), β≥0.9 trains")
    ax[0, 1].legend(fontsize=9)

    # (c) benefit and stream HFER vs lr
    bmap = rec["metrics"]["benefit"]
    xs = [lr for lr in lrs if str(lr) in bmap]
    ax[1, 0].bar([str(lr) for lr in xs], [100 * bmap[str(lr)] for lr in xs],
                 color="#059669", width=0.5, label="best-β benefit vs β=0 (%)")
    ax2 = ax[1, 0].twinx()
    ax2.plot([str(lr) for lr in lrs], A["hfer0"], marker="o", color="#111111",
             lw=style.LW.PRIMARY, label=r"HFER($\mathbf{G}$, β=0)")
    ax2.axhline(float(A["white_baseline"][0]), ls="--", color="0.5", lw=style.LW.GUIDE,
                label="white baseline")
    ax2.set_ylabel("HFER", rotation=270, labelpad=18)
    ax2.set_ylim(0, 1.05)
    ax[1, 0].set(xlabel="lr", ylabel="benefit (%)",
                 title="(c) benefit and stream HFER vs lr")
    h1, l1 = ax[1, 0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1, 0].legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")

    # (d) Muon pre/post/polar at both background lrs
    mb = [float(v) for v in A["muon_betas"]]
    xpos = np.arange(len(mb))
    pre_c = style.PIPELINE_STYLE["pre_polar"]["color"]
    post_c = style.PIPELINE_STYLE["post_polar"]["color"]
    polar_c = style.PIPELINE_STYLE["polar_only"]["color"]
    m_lrs = [float(v) for v in A["muon_lrs"]]
    for lr, alpha in zip(sorted(m_lrs, reverse=True), (1.0, 0.35)):
        tag = str(lr).replace(".", "p")
        off = -0.27 if alpha == 1.0 else 0.09
        ax[1, 1].bar(xpos + off, A[f"muon_lr{tag}_pre"], width=0.18, color=pre_c,
                     alpha=alpha, label=f"pre-polar lr={lr}")
        ax[1, 1].bar(xpos + off + 0.18, A[f"muon_lr{tag}_post"], width=0.18, color=post_c,
                     alpha=alpha, label=f"post-polar lr={lr}")
        ax[1, 1].axhline(float(A[f"muon_lr{tag}_polar"][0]), ls="--", color=polar_c,
                         alpha=alpha, lw=style.LW.GUIDE, label=f"polar-only lr={lr}")
    ax[1, 1].set(xticks=xpos, xticklabels=[str(b) for b in mb], xlabel="β",
                 ylabel="final training loss", title="(d) Muon closed loop at scale")
    ax[1, 1].legend(fontsize=8, ncol=2)

    fig.suptitle("E11  mini-batch nanoGPT: hill-dominated streams and the momentum regimes "
                 "at scale", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
