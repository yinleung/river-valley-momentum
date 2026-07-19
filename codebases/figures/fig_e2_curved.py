"""Figure E2 — curved river-valley filtering-lag tradeoff.

Pure module: reads the cached curved_valley run -> out/fig_e2_curved.{pdf,png}.
    cd figures && python fig_e2_curved.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e2_curved"


def build() -> None:
    style.apply_style()
    rec = _data.load_task("curved_valley")
    A, cfg, M = rec["arrays"], rec["config"], rec["metrics"]
    betas = list(A["betas"])
    bi = {b: i for i, b in enumerate(betas)}

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a) curved contour + river floor + the well-behaved filtered trajectories.
    # beta=0 oscillates off-scale (x spans ~[-100, 150]); it is quantified in (b)-(d).
    W = A["W"]
    shown = [b for b in (0.9, 0.95, 0.99) if b in bi]
    xs = np.linspace(cfg["x0"] - 0.5, cfg["x_star"] + 1.5, 240)
    ys = np.linspace(-4.5, 4.5, 240)
    X, Y = np.meshgrid(xs, ys)
    floor = cfg["a"] * np.sin(cfg["k"] * X)
    L = 0.5 * cfg["mu"] * (X - cfg["x_star"]) ** 2 + 0.5 * cfg["lam"] * (Y - floor) ** 2
    ax[0, 0].contourf(X, Y, np.log1p(L), levels=20, cmap="Greys", alpha=0.30)
    ax[0, 0].plot(xs, cfg["a"] * np.sin(cfg["k"] * xs), "b-", lw=style.LW.GUIDE,
                  label="river floor y=f(x)")
    for b in shown:
        ax[0, 0].plot(W[bi[b], :, 0], W[bi[b], :, 1], lw=style.LW.PRIMARY,
                      color=style.beta_color(b), marker="o", markevery=40, markersize=4,
                      label=f"β={b}")
    ax[0, 0].text(0.03, 0.04, "β=0 oscillates off-scale", transform=ax[0, 0].transAxes,
                  fontsize=9, color="0.35")
    ax[0, 0].set(xlabel="river coordinate x", ylabel="hill coordinate y",
                 title="(a) filtered trajectories on the bending floor",
                 xlim=(cfg["x0"] - 0.5, cfg["x_star"] + 1.5), ylim=(-4.5, 4.5))
    ax[0, 0].legend(fontsize=9, loc="upper left")

    # (b) river alignment vs beta — inverted-U: best at moderate beta
    ag = [M["align_g"][str(b)] for b in betas]
    am = [M["align_m"][str(b)] for b in betas]
    ax[0, 1].plot(betas, ag, "o--", color="0.5", lw=style.LW.PRIMARY, label="raw $g_t$")
    ax[0, 1].plot(betas, am, "o-", color="#ea580c", lw=style.LW.MEAN, label="momentum $m_t$")
    ax[0, 1].set(xlabel="β", ylabel="river alignment", title="(b) alignment to current tangent",
                 ylim=(0, 1.05))
    ax[0, 1].legend(fontsize=9, loc="lower center")

    # (c) hill-normal energy vs beta (log)
    he = [M["hill_energy"][str(b)] for b in betas]
    ax[1, 0].semilogy(betas, he, "o-", color="#1d4ed8", lw=style.LW.PRIMARY)
    ax[1, 0].set(xlabel="β", ylabel="hill energy of $m$", title="(c) hill-normal energy")

    # (d) river-following lag vs beta
    lag = [M["lag"][str(b)] for b in betas]
    ax[1, 1].plot(betas, lag, "o-", color="#dc2626", lw=style.LW.PRIMARY)
    ax[1, 1].set(xlabel="β", ylabel="lag  1 - align($m$, r)", title="(d) river-following lag")

    fig.suptitle("E2  curved river-valley: alignment, hill energy, and lag vs β", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
