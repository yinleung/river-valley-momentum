"""Figure E5-Wedin — filter-first (Thm. filterfirst / theory-note T5) constants: buffer tail and subspace error vs guides.

Pure module: reads the cached matrix_muon run's overlay arrays -> out/fig_e5_wedin.{pdf,png}.
    cd figures && python fig_e5_wedin.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e5_wedin"
BETAS = [0.9, 0.99]


def build() -> None:
    style.apply_style()
    rec = _data.load_task("matrix_muon")
    A = rec["arrays"]

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    # (a) clean scenario: buffer tail vs the two-sided eps_t guides
    for b in BETAS:
        tag = f"highfreq_clean_b{str(b).replace('.', 'p')}"
        c = style.beta_color(min(b, 0.99))
        t = A[f"ov_{tag}_t"]
        ax[0].plot(t, A[f"ov_{tag}_tail_meas"], color=c, lw=style.LW.PRIMARY,
                   label=f"$\\sigma_{{r+1}}(\\mathbf{{M}}_t)$, β={b}")
        ax[0].plot(t, A[f"ov_{tag}_tail_hi"], color=c, ls="--", lw=style.LW.GUIDE)
        ax[0].plot(t, A[f"ov_{tag}_tail_lo"], color=c, ls=":", lw=1.4)
    ax[0].set(xlabel="$t$", ylabel="singular value", yscale="log",
              title="(a) clean: tail vs $|\\varepsilon_t|$ guides")
    ax[0].legend(fontsize=9)

    # (b) clean scenario: measured subspace error vs the Wedin guide
    for b in BETAS:
        tag = f"highfreq_clean_b{str(b).replace('.', 'p')}"
        c = style.beta_color(min(b, 0.99))
        t = A[f"ov_{tag}_t"]
        ax[1].plot(t, A[f"ov_{tag}_sin_meas"], color=c, lw=style.LW.PRIMARY,
                   label=f"measured, β={b}")
        ax[1].plot(t, A[f"ov_{tag}_wedin"], color=c, ls="--", lw=style.LW.GUIDE,
                   label=f"Wedin guide, β={b}")
    ax[1].set(xlabel="$t$", ylabel=r"$\Vert\sin\Theta\Vert_2$", yscale="log",
              title="(b) clean: subspace error vs Wedin")
    ax[1].legend(fontsize=9)

    # (c) noisy scenario: the stochastic floor takes over at large beta
    for b in BETAS:
        tag = f"highfreq_b{str(b).replace('.', 'p')}"
        c = style.beta_color(min(b, 0.99))
        t = A[f"ov_{tag}_t"]
        ax[2].plot(t, A[f"ov_{tag}_tail_meas"], color=c, lw=style.LW.PRIMARY,
                   label=f"$\\sigma_{{r+1}}(\\mathbf{{M}}_t)$, β={b}")
        ax[2].plot(t, A[f"ov_{tag}_tail_hi"] + A[f"ov_{tag}_noise_floor"], color=c,
                   ls="--", lw=style.LW.GUIDE)
        ax[2].plot(t, A[f"ov_{tag}_tail_hi"], color=c, ls=":", lw=1.4)
    ax[2].set(xlabel="$t$", ylabel="singular value", yscale="log",
              title="(c) noisy: tail vs guide + noise floor")
    ax[2].legend(fontsize=9)

    fig.suptitle("E5 overlay  filter-first constants on the matrix/Muon toy "
                 "(dashed: upper guide; dotted: deterministic-only part)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
