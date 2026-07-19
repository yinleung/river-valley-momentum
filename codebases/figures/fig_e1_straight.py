"""Figure E1 — straight river-valley filtering demo.

Pure module: reads the cached straight_valley run and renders out/fig_e1_straight.{pdf,png}.
Spectra are DFTs of the cached gradient/momentum streams (display post-processing, not an
experiment rerun). See ../CODING.md Pillar 4.

    cd figures && python fig_e1_straight.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402
from core import metrics as Me  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e1_straight"


def build() -> None:
    style.apply_style()
    rec = _data.load_task("straight_valley")
    A, cfg, M = rec["arrays"], rec["config"], rec["metrics"]
    betas = list(A["betas"])
    bi = {b: i for i, b in enumerate(betas)}

    fig, ax = plt.subplots(2, 3, figsize=style.grid_figsize(2, 3))

    # (a) contour + trajectories
    W = A["W"]
    xs = np.linspace(W[..., 0].min() - 0.5, W[..., 0].max() + 0.5, 200)
    ys = np.linspace(min(-1.2, W[..., 1].min()), max(1.2, W[..., 1].max()), 200)
    X, Y = np.meshgrid(xs, ys)
    L = 0.5 * cfg["mu"] * (X - cfg["x_star"]) ** 2 + 0.5 * cfg["lam"] * Y**2
    ax[0, 0].contour(X, Y, L, levels=18, colors="0.8", linewidths=0.6)
    for b in betas:
        ax[0, 0].plot(W[bi[b], :, 0], W[bi[b], :, 1], lw=style.LW.PRIMARY,
                      color=style.beta_color(b), label=f"β={b}")
    ax[0, 0].axhline(0, color="0.4", ls=":", lw=1)
    ax[0, 0].set(xlabel="river coordinate x", ylabel="hill coordinate y", title="(a) trajectories")
    ax[0, 0].legend(fontsize=9, loc="upper left")

    # (b) hill component: raw gradient (β=0) vs momentum buffer (β=0.9, 0.99)
    ax[0, 1].plot(A["Gexact"][bi[0.0], :, 1], color=style.beta_color(0.0), lw=style.LW.RAW,
                  label="raw $g_{t,y}$ (β=0)")
    for b in (0.9, 0.99):
        ax[0, 1].plot(A["M"][bi[b], :, 1], color=style.beta_color(b), lw=style.LW.PRIMARY,
                      label=f"$m_{{t,y}}$ (β={b})")
    ax[0, 1].set(xlabel="step t", ylabel="hill component", title="(b) hill component vs time")
    ax[0, 1].legend(fontsize=9)

    # (c) temporal spectrum of the hill component (rectangular window keeps the early transient)
    om, Gh = Me.windowed_dft(A["Gexact"][bi[0.0], :, 1], window="rect")
    ax[0, 2].semilogy(om / np.pi, np.abs(Gh), color=style.beta_color(0.0), lw=style.LW.RAW,
                      label="$|\\hat g_y|$ (β=0)")
    for b in (0.9, 0.99):
        _, Mh = Me.windowed_dft(A["M"][bi[b], :, 1], window="rect")
        ax[0, 2].semilogy(om / np.pi, np.abs(Mh), color=style.beta_color(b), lw=style.LW.PRIMARY,
                          label=f"$|\\hat m_y|$ (β={b})")
    ax[0, 2].axvspan(0.6, 1.0, color="0.85", alpha=0.5, zorder=0)
    ax[0, 2].set(xlabel="ω / π", ylabel="magnitude", title="(c) hill spectrum")
    ax[0, 2].legend(fontsize=9)

    # (d) HSR and MSR vs beta with the |H(π)|^2 guide
    hsr = [M["HSR"][str(b)] for b in betas]
    msr = [M["MSR"][str(b)] for b in betas]
    hpi2 = [((1 - b) / (1 + b)) ** 2 for b in betas]
    ax[1, 0].semilogy(betas, hsr, "o-", color="#1d4ed8", lw=style.LW.PRIMARY, label="HSR")
    ax[1, 0].semilogy(betas, msr, "s-", color="#dc2626", lw=style.LW.PRIMARY, label="MSR")
    ax[1, 0].semilogy(betas, hpi2, "k--", lw=style.LW.GUIDE, label=r"$|H_\beta(\pi)|^2$")
    ax[1, 0].set(xlabel="β", ylabel="hill-energy ratio", title="(d) hill suppression")
    ax[1, 0].legend(fontsize=9)

    # (e) river alignment of g vs m
    ag = [M["align_g"][str(b)] for b in betas]
    am = [M["align_m"][str(b)] for b in betas]
    ax[1, 1].plot(betas, ag, "o--", color="0.5", lw=style.LW.PRIMARY, label="raw $g_t$")
    ax[1, 1].plot(betas, am, "o-", color="#ea580c", lw=style.LW.MEAN, label="momentum $m_t$")
    ax[1, 1].set(xlabel="β", ylabel="river alignment", title="(e) river alignment", ylim=(0, 1.02))
    ax[1, 1].legend(fontsize=9)

    # (f) rms distance to the river floor
    dist = [M["dist_rms"][str(b)] for b in betas]
    ax[1, 2].plot(betas, dist, "o-", color="#059669", lw=style.LW.PRIMARY)
    ax[1, 2].set(xlabel="β", ylabel="rms |y|", title="(f) distance to river floor")

    fig.suptitle("E1  straight river-valley filtering", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
