"""Figure E5 — filter-first (pre-polar) wins for high-frequency and stochastic disturbances.

Pure module: reads the cached matrix_muon run -> out/fig_e5_matrix.{pdf,png}.
    cd figures && python fig_e5_matrix.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e5_matrix"
PIPES = ["pre_polar", "post_polar", "polar_only"]  # array column order


def _series(A, scenario, metric):
    """(n_beta,) per-pipeline series for a scenario/metric from the stacked arrays."""
    betas = list(A["betas"])
    stacked = np.stack([A[f"{scenario}_b{str(b).replace('.', 'p')}_{metric}"] for b in betas])
    return betas, stacked  # stacked[bi, pipe_idx]


def build() -> None:
    style.apply_style()
    rec = _data.load_task("matrix_muon")
    A = rec["arrays"]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    panels = [
        (0, 0, "highfreq", "sinTheta", "(a) subspace error — high-freq $(-1)^t\mathbf{A}$", "sin Θ (↓)"),
        (0, 1, "stochastic", "sinTheta", "(b) subspace error — stochastic $\\Xi_t$", "sin Θ (↓)"),
        (1, 0, "highfreq", "align", "(c) signal alignment — high-freq", "alignment (↑)"),
        (1, 1, "stochastic", "align", "(d) signal alignment — stochastic", "alignment (↑)"),
    ]
    for i, j, scen, metric, title, ylab in panels:
        betas, stacked = _series(A, scen, metric)
        for pidx, pipe in enumerate(PIPES):
            st = style.PIPELINE_STYLE[pipe]
            ax[i, j].plot(betas, stacked[:, pidx], marker=st["marker"], color=st["color"],
                          markersize=st["markersize"], lw=style.LW.PRIMARY, label=st["label"])
        ax[i, j].set(xlabel="β", ylabel=ylab, title=title)
        if metric == "sinTheta":
            ax[i, j].set_yscale("log")
        ax[i, j].legend(fontsize=9)

    fig.suptitle("E5  pre-polar, post-polar, and polar-only vs β "
                 "(high-frequency and stochastic disturbances)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
