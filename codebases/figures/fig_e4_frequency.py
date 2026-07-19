"""Figure E4 — empirical EMA transfer ratio vs theoretical |H_beta(omega)|.

Pure module: reads the cached frequency_validation run -> out/fig_e4_frequency.{pdf,png}.
The significant-bin mask is recomputed from the cached input stream (display post-processing).
    cd figures && python fig_e4_frequency.py
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

LABEL = "fig_e4_frequency"
STREAMS = ["synthetic", "straight", "curved"]
BETAS = [0.5, 0.9, 0.95, 0.99]


def build() -> None:
    style.apply_style()
    rec = _data.load_task("frequency_validation")
    A, cfg = rec["arrays"], rec["config"]
    omega = A["omega"]
    wf = omega / np.pi

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.7), sharey=True)
    for j, name in enumerate(STREAMS):
        _, G = Me.windowed_dft(A[f"{name}_stream"], window=cfg["window"])
        power = np.abs(G) ** 2
        sig = power > 0.01 * power.max()  # bins with enough energy for a meaningful ratio
        for b in BETAS:
            c = style.beta_color(b)
            ax[j].plot(wf, A[f"{name}_H_b{str(b).replace('.', 'p')}"], color=c,
                       lw=style.LW.GUIDE, zorder=2)
            R = A[f"{name}_R_b{str(b).replace('.', 'p')}"]
            ax[j].scatter(wf[sig], R[sig], s=12, color=c, alpha=0.7, zorder=3,
                          label=f"β={b}")
        ax[j].set(xlabel="ω / π", title=f"({chr(97+j)}) {name} stream", yscale="log",
                  ylim=(3e-3, 2))
        if j == 0:
            ax[j].set_ylabel(r"$R(\omega)=|\hat m|/|\hat g|$   vs   $|H_\beta(\omega)|$")
        ax[j].legend(fontsize=9, title="markers: empirical\nlines: theory", title_fontsize=8)

    fig.suptitle(r"E4  empirical transfer ratio $R(\omega)$ and EMA response $|H_\beta(\omega)|$",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
