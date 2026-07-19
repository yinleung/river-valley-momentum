"""Figure E7 — optimizer-level tests: mechanism metrics predict performance, beta alone does not.

Pure module: reads the cached optimizer_toy and optimizer_mlp runs -> out/fig_e7_optimizer.{pdf,png}.
    cd figures && python fig_e7_optimizer.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e7_optimizer"
SETTING_STYLE = {
    "S1_straight_el1.8": {"color": "#1d4ed8", "marker": "o", "label": "S1 straight 1.8"},
    "S2_straight_el0.6": {"color": "#0891b2", "marker": "v", "label": "S2 straight 0.6"},
    "S3_curved_k0.9_el1.8": {"color": "#059669", "marker": "s", "label": "S3 curved 1.8"},
    "S4_curved_k0.9_el2.5": {"color": "#dc2626", "marker": "D", "label": "S4 curved 2.5"},
    "S5_curved_k1.8_el1.8": {"color": "#b45309", "marker": "^", "label": "S5 curved k=1.8"},
}


def build() -> None:
    style.apply_style()
    toy = _data.load_task("optimizer_toy")
    mlp = _data.load_task("optimizer_mlp")
    betas = list(toy["arrays"]["betas"])

    fig, ax = plt.subplots(2, 2, figsize=(13.4, 9.2))

    # (a) toy: tail loss vs beta per setting
    for name, st in SETTING_STYLE.items():
        perf = toy["arrays"][f"{name}_perf"]
        shown = np.where(np.isfinite(perf), perf, np.nan)
        ax[0, 0].plot(betas, shown, marker=st["marker"], color=st["color"],
                      lw=style.LW.PRIMARY, label=st["label"])
    ax[0, 0].set(xlabel="β", ylabel="tail loss", yscale="log",
                 title="(a) toy settings: tail loss vs β")
    ax[0, 0].legend(fontsize=9)

    # (b) rank correlations: mechanism score vs beta, per setting
    names = list(SETTING_STYLE)
    rb = [toy["metrics"]["rho_beta"][n] for n in names]
    rs = [toy["metrics"]["rho_score"][n] for n in names]
    xpos = np.arange(len(names))
    ax[0, 1].bar(xpos - 0.18, np.abs(rb), width=0.36, color="0.6", label=r"$|\rho|$(loss, β)")
    ax[0, 1].bar(xpos + 0.18, rs, width=0.36, color="#059669",
                 label=r"$\rho$(loss, mechanism score)")
    ax[0, 1].set(xticks=xpos, xticklabels=[SETTING_STYLE[n]["label"] for n in names],
                 ylabel="Spearman correlation", title="(b) what predicts the loss rank")
    ax[0, 1].tick_params(axis="x", rotation=20, labelsize=9)
    ax[0, 1].axhline(0, color="0.5", lw=1)
    ax[0, 1].legend(fontsize=10)

    # (c) MLP regime contrast
    for eta, color in ((0.05, "#0891b2"), (0.3, "#dc2626")):
        tag = str(eta).replace(".", "p")
        tr = mlp["arrays"][f"eta{tag}_train"]
        hf = mlp["metrics"]["hfer_regime"][str(eta)]
        ax[1, 0].plot(betas, tr, marker="o", color=color, lw=style.LW.PRIMARY,
                      label=rf"$\eta={eta}$ (HFER {hf:.2f})")
    ax[1, 0].set(xlabel="β", ylabel="tail training loss",
                 title="(c) MLP: momentum pays only in the HFER regime")
    ax[1, 0].legend(fontsize=10)

    # (d) Muon closed-loop: pre/post final loss in both stream regimes, polar-only dashed
    mb = list(mlp["arrays"]["muon_betas"])
    xpos = np.arange(len(mb))
    pre_c = style.PIPELINE_STYLE["pre_polar"]["color"]
    post_c = style.PIPELINE_STYLE["post_polar"]["color"]
    polar_c = style.PIPELINE_STYLE["polar_only"]["color"]
    for eta, alpha, suff in ((0.3, 1.0, " (EoS)"), (0.05, 0.35, " (sub-critical)")):
        tag = str(eta).replace(".", "p")
        off = -0.27 if eta == 0.3 else 0.09
        ax[1, 1].bar(xpos + off, mlp["arrays"][f"muon_eta{tag}_pre"], width=0.18,
                     color=pre_c, alpha=alpha, label="pre-polar" + suff)
        ax[1, 1].bar(xpos + off + 0.18, mlp["arrays"][f"muon_eta{tag}_post"], width=0.18,
                     color=post_c, alpha=alpha, label="post-polar" + suff)
        ax[1, 1].axhline(float(mlp["arrays"][f"muon_eta{tag}_polar"][0]), color=polar_c,
                         alpha=alpha, ls="--", lw=style.LW.GUIDE,
                         label="polar-only" + suff)
    ax[1, 1].set(xticks=xpos, xticklabels=[str(b) for b in mb], xlabel="β",
                 ylabel="final training loss",
                 title="(d) Muon closed loop: the pre/post gap follows the regime")
    ax[1, 1].legend(fontsize=8, ncol=2)

    fig.suptitle("E7  optimizer-level tests: mechanism metrics vs β as performance predictors",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
