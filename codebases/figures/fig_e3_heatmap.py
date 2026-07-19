"""Figure E3-heatmap — CL-2 ratio grid, escape frequency, and the widening beta window.

Pure module: reads the cached cl2_grid run -> out/fig_e3_heatmap.{pdf,png}.
    cd figures && python fig_e3_heatmap.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e3_heatmap"


def build() -> None:
    style.apply_style()
    rec = _data.load_task("cl2_grid")
    A = rec["arrays"]
    etalams, betas_g = A["etalams"], A["betas_grid"]
    betas_f, conds = A["betas_fine"], A["conds"]

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.7))

    # (a) measured/CL-2 ratio heatmap on the straight valley
    ratio = A["grid_ratio"]
    im = ax[0].imshow(ratio, origin="lower", aspect="auto", cmap="RdBu_r",
                      vmin=0.8, vmax=1.2)
    for i in range(len(etalams)):
        for j in range(len(betas_g)):
            if A["grid_div"][i, j] == rec["config"]["n_seeds"]:
                ax[0].text(j, i, "div", ha="center", va="center", fontsize=9, color="k")
            elif np.isfinite(ratio[i, j]):
                ax[0].text(j, i, f"{ratio[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax[0].set(xticks=range(len(betas_g)), xticklabels=[str(b) for b in betas_g],
              yticks=range(len(etalams)), yticklabels=[str(e) for e in etalams],
              xlabel="β", ylabel=r"$\eta\lambda$",
              title="(a) straight valley: rms / CL-2 guide")
    fig.colorbar(im, ax=ax[0], shrink=0.85)

    # (b) escape frequency vs beta on the curved valley
    for el, color, marker in ((1.8, "#1d4ed8", "o"), (2.5, "#dc2626", "s")):
        tag = str(el).replace(".", "p")
        ax[1].plot(betas_f, A[f"esc{tag}_esc"], marker=marker, color=color,
                   lw=style.LW.PRIMARY, label=rf"$\eta\lambda={el}$")
    ax[1].set(xlabel="β", ylabel="escape frequency",
              title=f"(b) curved valley: tube escape (R={rec['config']['tube_radius']})")
    ax[1].legend(fontsize=10)

    # (c) good-beta window vs conditioning
    W = np.stack([A[f"window_in_{int(c)}"] for c in conds])
    ax[2].imshow(W, origin="lower", aspect="auto", cmap="Greens", vmin=0, vmax=1.4)
    ax[2].set(xticks=range(len(betas_f)), xticklabels=[str(b) for b in betas_f],
              yticks=range(len(conds)), yticklabels=[str(int(c)) for c in conds],
              xlabel="β", ylabel=r"$\lambda/\mu$",
              title="(c) good-β window (escape & lag criteria)")
    ax[2].tick_params(axis="x", labelsize=10)
    for ci in range(len(conds)):
        idx = np.where(W[ci] > 0)[0]
        if len(idx):
            ax[2].annotate("", xy=(idx.max() + 0.45, ci), xytext=(idx.min() - 0.45, ci),
                           arrowprops=dict(arrowstyle="<->", color="#065f46", lw=2))

    fig.suptitle("E3 extension  CL-2 exactness, tube restoration, and the widening β window",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
