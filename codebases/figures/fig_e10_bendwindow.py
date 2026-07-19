"""Figure E10 — the beta window vs bend frequency on the curved valley.

Pure module: reads the cached bend_window run -> out/fig_e10_bendwindow.{pdf,png}.
    cd figures && python fig_e10_bendwindow.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e10_bendwindow"
_K_CMAP = mpl.colormaps["viridis"]


def k_color(k: float, k_max: float = 1.8):
    return _K_CMAP(0.10 + 0.80 * (k / k_max))


def build() -> None:
    style.apply_style()
    rec = _data.load_task("bend_window")
    A = rec["arrays"]
    betas = list(A["betas"])
    ks = list(A["ks"])
    etalams = list(A["etalams"])
    ei18 = etalams.index(1.8)

    fig, ax = plt.subplots(2, 2, figsize=(13.8, 9.2))

    # (a) travel loss vs beta per k (eta*lam = 1.8): the U-shapes
    for ki, k in enumerate(ks):
        row = A["loss"][ei18, ki]
        shown = np.where(np.isfinite(row), row, np.nan)
        ax[0, 0].plot(betas, shown, marker="o", color=k_color(k),
                      lw=style.LW.PRIMARY, label=f"k={k}")
        bb = A["best_beta"][ei18, ki]
        if np.isfinite(bb):
            bi = betas.index(float(bb))
            ax[0, 0].plot([bb], [row[bi]], marker="*", ms=16, color=k_color(k),
                          mec="k", mew=0.5)
    ax[0, 0].set(xlabel="β", ylabel="mean travel loss", yscale="log",
                 title=r"(a) U-shapes per bend frequency ($\eta\lambda=1.8$; $\star$ best)")
    ax[0, 0].legend(fontsize=9)

    # (b) the window map at eta*lam = 1.8
    W = A["in_window"][ei18]
    ax[0, 1].imshow(W, origin="lower", aspect="auto", cmap="Greens", vmin=0, vmax=1.4)
    for ki in range(len(ks)):
        fl, tp = A["floor_edge"][ei18, ki], A["top_edge"][ei18, ki]
        if np.isfinite(fl):
            ax[0, 1].plot(betas.index(float(fl)) - 0.38, ki, marker=">", color="#b45309",
                          ms=9)
        if np.isfinite(tp):
            ax[0, 1].plot(betas.index(float(tp)) + 0.38, ki, marker="<", color="#1d4ed8",
                          ms=9)
    ax[0, 1].set(xticks=range(len(betas)), xticklabels=[str(b) for b in betas],
                 yticks=range(len(ks)), yticklabels=[str(k) for k in ks],
                 xlabel="β", ylabel="bend wavenumber k",
                 title=r"(b) window map ($\eta\lambda=1.8$): floor $\rhd$, top $\lhd$")
    ax[0, 1].tick_params(axis="x", labelsize=8)

    # (c) confinement floor and window width vs k, both eta*lam
    for ei, el in enumerate(etalams):
        color = "#1d4ed8" if el == 1.8 else "#dc2626"
        ax[1, 0].plot(ks, A["floor_edge"][ei], marker="o", color=color,
                      lw=style.LW.PRIMARY, label=rf"floor, $\eta\lambda={el}$")
        ax[1, 0].plot(ks, A["top_edge"][ei], marker="^", ls=":", color=color,
                      lw=style.LW.RAW, alpha=0.7, label=rf"top, $\eta\lambda={el}$")
    ax[1, 0].set(xlabel="bend wavenumber k", ylabel="β",
                 title="(c) the window narrows from below")
    ax[1, 0].legend(fontsize=9)

    # (d) river speed collapses with k
    for ei, el in enumerate(etalams):
        color = "#1d4ed8" if el == 1.8 else "#dc2626"
        ax[1, 1].plot(ks, A["v_row"][ei], marker="o", color=color, lw=style.LW.PRIMARY,
                      label=rf"$\eta\lambda={el}$")
    ax[1, 1].set(xlabel="bend wavenumber k", ylabel="river speed $v_{\mathrm{riv}}$ (units/step)",
                 title="(d) curvature costs river speed")
    ax[1, 1].legend(fontsize=10)

    fig.suptitle("E10  curved valley: the good-β window vs bend frequency", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
