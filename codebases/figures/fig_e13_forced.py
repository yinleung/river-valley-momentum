"""Figure E13 — forced-disturbance controls: loss and forced amplitude on the FR guides.

Pure module: reads the cached forced_controls run -> out/fig_e13_forced.{pdf,png}.
Guide curves in panel (c) come from core.closedloop.forced_gain (the audited definition);
the per-cell guides in (a)/(b) are the run record's own arrays.
    cd figures && python fig_e13_forced.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402
from core.closedloop import forced_gain  # noqa: E402

LABEL = "fig_e13_forced"


def arm_labels(omegas: np.ndarray) -> list[str]:
    return [r"$\omega=0.02$ (passband)",
            rf"$\omega={omegas[1]:.3f}$ ($\theta_{{0.9}}$)",
            r"$\omega=0.6\pi$",
            r"$\omega=\pi$ (Nyquist)"]


def build() -> None:
    style.apply_style()
    rec = _data.load_task("forced_controls")
    A, cfg = rec["arrays"], rec["config"]
    omegas, betas = A["omegas"], list(A["betas"])
    eta, lam, amp_f = cfg["eta"], cfg["lam"], cfg["A"]
    labels = arm_labels(omegas)
    n_seeds = A["loss"].shape[-1]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # (a) tail hill loss vs beta per arm, with the FR + CL-2 guides
    for oi in range(len(omegas)):
        st = style.FORCING_STYLE[oi]
        mean = np.nanmean(A["loss"][oi], axis=-1)
        se = np.nanstd(A["loss"][oi], axis=-1) / np.sqrt(n_seeds)
        ax[0].errorbar(betas, mean, yerr=se, lw=style.LW.PRIMARY, label=labels[oi], **st)
        ax[0].plot(betas, A["guide"][oi], ls="--", color=st["color"], lw=style.LW.GUIDE)
    ax[0].set(xlabel=r"$\beta$", ylabel="tail hill loss", yscale="log",
              title="(a) loss on the forced-response guides")
    ax[0].legend(fontsize=9)

    # (b) lock-in forced amplitude vs beta per arm, guides A|G_beta(omega)|
    for oi in range(len(omegas)):
        st = style.FORCING_STYLE[oi]
        ax[1].plot(betas, np.nanmean(A["amp"][oi], axis=-1), lw=style.LW.PRIMARY,
                   label=labels[oi], **st)
        ax[1].plot(betas, A["amp_guide"][oi], ls="--", color=st["color"],
                   lw=style.LW.GUIDE)
    ax[1].set(xlabel=r"$\beta$", ylabel="forced amplitude", yscale="log",
              title=r"(b) amplitude on $A\,|G_\beta(\omega)|$")
    ax[1].legend(fontsize=9)

    # (c) the gain curve per beta, with the four arms marked
    om_grid = np.linspace(1e-3, np.pi, 600)
    for b in betas:
        ax[2].plot(om_grid / np.pi, forced_gain(om_grid, b, eta, lam),
                   color=style.beta_color(b), lw=style.LW.PRIMARY,
                   label=rf"$\beta={b}$")
    for oi, om in enumerate(omegas):
        ax[2].axvline(om / np.pi, color=style.FORCING_STYLE[oi]["color"], ls=":",
                      lw=1.4)
    ax[2].set(xlabel=r"$\omega/\pi$", ylabel=r"$|G_\beta(\omega)|$", yscale="log",
              title="(c) closed-loop gain, arms dotted")
    ax[2].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    style.save_figure(fig, LABEL)
    print(f"wrote out/{LABEL}.pdf/.png from {rec['run_id']}")


if __name__ == "__main__":
    build()
