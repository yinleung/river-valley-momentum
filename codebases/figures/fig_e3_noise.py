"""Figure E3 — momentum filters stochastic noise (three noise models, multi-seed).

Pure module: reads the cached noisy_valley run -> out/fig_e3_noise.{pdf,png}.
    cd figures && python fig_e3_noise.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e3_noise"
NOISES = ["gaussian", "anisotropic", "heavy_tailed"]


def build() -> None:
    style.apply_style()
    rec = _data.load_task("noisy_valley")
    A = rec["arrays"]
    betas = list(A["betas"])

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))

    # (a) NSR(noise) vs beta with the 1/N_eff floor
    floor = [(1 - b) / (1 + b) for b in betas]
    for nt in NOISES:
        nsr = A[f"{nt}_NSR"]
        st = style.NOISE_STYLE[nt]
        ax[0].errorbar(betas, nsr.mean(1), yerr=nsr.std(1), marker=st["marker"],
                       color=st["color"], lw=style.LW.PRIMARY, capsize=3, label=st["label"])
    ax[0].plot(betas, floor, "k--", lw=style.LW.GUIDE, label=r"$1/T_{\mathrm{eff}}$")
    ax[0].set(xlabel="β", ylabel="NSR (noise)", title="(a) stochastic noise suppression",
              yscale="log")
    ax[0].legend(fontsize=9)

    # (b) instantaneous-target NSR (declarative; interpretation lives in the body text)
    for nt in NOISES:
        nsr_i = A[f"{nt}_NSR_inst"]
        st = style.NOISE_STYLE[nt]
        ax[1].plot(betas, nsr_i.mean(1), marker=st["marker"], color=st["color"],
                   lw=style.LW.PRIMARY, label=st["label"])
    ax[1].set(xlabel="β", ylabel="NSR (instantaneous target)",
              title="(b) instantaneous-target NSR")
    ax[1].legend(fontsize=9)

    # (c) river-alignment improvement of momentum over raw gradient
    for nt in NOISES:
        am = A[f"{nt}_align_m"].mean(1)
        ag = A[f"{nt}_align_g"].mean(1)
        st = style.NOISE_STYLE[nt]
        ax[2].plot(betas, am - ag, marker=st["marker"], color=st["color"],
                   lw=style.LW.PRIMARY, label=st["label"])
    ax[2].axhline(0, color="0.6", ls="-", lw=1)
    ax[2].set(xlabel="β", ylabel="align($m$) − align($g$)",
              title="(c) align($m$) − align($g$) vs β")
    ax[2].legend(fontsize=9)

    fig.suptitle("E3  noisy river-valley: NSR and river alignment vs β", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
