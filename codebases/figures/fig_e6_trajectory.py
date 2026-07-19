"""Figure E6 — real-network gradients show the river-valley slow/fast structure.

Pure module: reads the cached mlp_diagnostic run -> out/fig_e6_trajectory.{pdf,png}.
    cd figures && python fig_e6_trajectory.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e6_trajectory"


def build() -> None:
    style.apply_style()
    rec = _data.load_task("mlp_diagnostic")
    A, M, cfg = rec["arrays"], rec["metrics"], rec["config"]
    betas = list(A["betas"])

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # (a) target-layer gradient magnitude over training (edge-of-stability oscillation)
    ax[0].plot(A["G_meanabs"], color="#374151", lw=1.0)
    ax[0].set(xlabel="step t (after burn-in)", ylabel=r"mean $|G_{t}|$",
              title="(a) gradient magnitude over training")

    # (b) temporal Frobenius spectrum: peak at omega = pi
    wf = A["omega"] / np.pi
    ax[1].semilogy(wf, A["G_spectrum"], color="#1d4ed8", lw=style.LW.PRIMARY)
    ax[1].axvspan(cfg["hi_frac"], 1.0, color="0.85", alpha=0.6, zorder=0)
    ax[1].axvline(M["spectral_peak"], color="#dc2626", ls="--", lw=1.5, label="spectral peak")
    ax[1].set(xlabel="ω / π", ylabel=r"$|\hat G(\omega)|_F$", title="(b) temporal spectrum")
    ax[1].legend(fontsize=9)

    # (c) slow-gradient alignment (left) and MSR (right) vs beta
    am = [M["align_m"][str(b)] for b in betas]
    msr = [M["MSR"][str(b)] for b in betas]
    hpi2 = [((1 - b) / (1 + b)) ** 2 for b in betas]
    ax[2].plot(betas, am, "o-", color="#ea580c", lw=style.LW.MEAN, label="align($m$, slow)")
    ax[2].axhline(M["align_g"], color="0.5", ls="--", lw=style.LW.PRIMARY,
                  label="align($g$, slow)")
    ax[2].set(xlabel="β", ylabel="slow-gradient alignment", title="(c) alignment and MSR",
              ylim=(0, 1.0))
    ax[2].legend(fontsize=9, loc="upper left")
    axr = ax[2].twinx()
    axr.semilogy(betas, msr, "s-", color="#1d4ed8", lw=style.LW.PRIMARY, label="MSR")
    axr.semilogy(betas, hpi2, "k:", lw=style.LW.GUIDE, label=r"$|H_\beta(\pi)|^2$")
    axr.set_ylabel("high-band MSR", color="#1d4ed8")
    axr.tick_params(axis="y", labelcolor="#1d4ed8")
    axr.legend(fontsize=9, loc="lower right")

    fig.suptitle("E6  real MLP gradients at the edge of stability", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
