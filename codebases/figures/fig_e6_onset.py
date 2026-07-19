"""Figure E6-onset — confinement onset: high-frequency structure appears after the descent.

Pure module: reads the cached mlp_diagnostic run's windowed arrays -> out/fig_e6_onset.{pdf,png}.
    cd figures && python fig_e6_onset.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e6_onset"


def build() -> None:
    style.apply_style()
    rec = _data.load_task("mlp_diagnostic")
    A = rec["arrays"]
    starts, hfer, dalign = A["win_starts"], A["win_hfer"], A["win_dalign"]
    onset = rec["metrics"]["onset_step"]

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.5))

    ax[0].plot(starts, hfer, marker="o", color="#1d4ed8", lw=style.LW.PRIMARY,
               label="windowed HFER")
    ax[0].axvline(onset, color="0.4", ls="--", lw=style.LW.GUIDE)
    ax[0].annotate("onset", (onset, 0.5), textcoords="offset points", xytext=(6, 0),
                   fontsize=11, color="0.3")
    tw = ax[0].twinx()
    tw.plot(A["loss_curve"], color=style.ROLE_COLORS["mean"], lw=style.LW.RAW, alpha=0.9)
    tw.set_ylabel("training loss", color=style.ROLE_COLORS["mean"])
    tw.tick_params(axis="y", colors=style.ROLE_COLORS["mean"])
    ax[0].set(xlabel="window start (step)", ylabel="HFER",
              title="(a) windowed HFER and the loss curve")
    ax[0].legend(fontsize=10, loc="center right")

    ax[1].plot(starts, dalign, marker="s", color="#7c3aed", lw=style.LW.PRIMARY)
    ax[1].axvline(onset, color="0.4", ls="--", lw=style.LW.GUIDE)
    ax[1].axhline(0, color="0.7", lw=1)
    ax[1].set(xlabel="window start (step)",
              ylabel=r"align$(m)$ $-$ align$(g)$ (β=0.9)",
              title="(b) EMA alignment gain per window")

    fig.suptitle("E6 onset  the Nyquist hill appears only after confinement; "
                 "filtering pays after it", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
