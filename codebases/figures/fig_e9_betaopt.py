"""Figure E9 — does larger beta improve river-valley optimization? (straight noisy valley)

Pure module: reads the cached beta_opt run -> out/fig_e9_betaopt.{pdf,png}.
    cd figures && python fig_e9_betaopt.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402
from core import closedloop as Cl  # noqa: E402

LABEL = "fig_e9_betaopt"
_EL_CMAP = mpl.colormaps["plasma"]


def el_color(el: float, el_max: float = 2.5):
    """Color for an eta*lam value on a shared ramp (low bright, high dark)."""
    return _EL_CMAP(0.85 - 0.75 * (el / el_max))


def build() -> None:
    style.apply_style()
    rec = _data.load_task("beta_opt")
    A = rec["arrays"]
    betas = list(A["betas"])
    etalams = list(A["etalams"])
    conds = list(A["conds"])
    ci = conds.index(100.0)

    fig, ax = plt.subplots(2, 3, figsize=(16.6, 9.4))

    # (a) tail hill loss vs beta per eta*lam, CL-3 guides, faint per-seed dots
    for ei, el in enumerate(etalams):
        c = el_color(el)
        for bi, b in enumerate(betas):
            seeds = A["hill_seeds"][ci, ei, bi]
            seeds = seeds[np.isfinite(seeds)]
            ax[0, 0].plot([b] * len(seeds), seeds, ".", color=c, alpha=0.15, ms=4)
        ax[0, 0].plot(betas, A["hill"][ci, ei], marker="o", color=c,
                      lw=style.LW.PRIMARY, label=rf"$\eta\lambda={el}$")
        ax[0, 0].plot(betas, A["pred_hill"][ei], ls="--", color=c, lw=style.LW.GUIDE,
                      alpha=0.7)
    ax[0, 0].set(xlabel="β", ylabel="tail hill loss", yscale="log",
                 title="(a) hill loss vs β (dashed: CL-3 guide)")
    ax[0, 0].legend(fontsize=9, ncol=2)

    # (b) rms / CL-2 guide ratio heatmap at cond = 100
    ratio = A["ratio"][ci]
    im = ax[0, 1].imshow(ratio, origin="lower", aspect="auto", cmap="RdBu_r",
                         vmin=0.8, vmax=1.2)
    for ei in range(len(etalams)):
        for bi in range(len(betas)):
            if A["ndiv"][ci, ei, bi] == rec["config"]["n_seeds"]:
                ax[0, 1].text(bi, ei, "div", ha="center", va="center", fontsize=8)
            elif np.isfinite(ratio[ei, bi]):
                ax[0, 1].text(bi, ei, f"{ratio[ei, bi]:.2f}", ha="center", va="center",
                              fontsize=7)
    ax[0, 1].set(xticks=range(len(betas)), xticklabels=[str(b) for b in betas],
                 yticks=range(len(etalams)), yticklabels=[str(e) for e in etalams],
                 xlabel="β", ylabel=r"$\eta\lambda$", title="(b) rms / CL-2 guide")
    ax[0, 1].tick_params(axis="x", labelsize=8)
    fig.colorbar(im, ax=ax[0, 1], shrink=0.85)

    # (c) maximal relative hill-loss reduction vs eta*lam, with the eta*lam/2 prediction
    els = np.array(etalams)
    for cj, cond in enumerate(conds):
        ax[0, 2].plot(els, A["reduction"][cj], marker="o", ls="none", ms=7,
                      color=style.beta_color(0.5 + 0.15 * cj),
                      label=rf"$\lambda/\mu={int(cond)}$")
    el_fine = np.linspace(min(els), 2.0, 100)
    ax[0, 2].plot(el_fine, [Cl.cl3_relative_reduction(e) for e in el_fine], ls="--",
                  color=style.ROLE_COLORS["guide"], lw=style.LW.GUIDE,
                  label=r"CL-3: $\eta\lambda/2$")
    ax[0, 2].axvspan(2.0, max(els) + 0.1, color="0.9")
    ax[0, 2].text(2.25, 0.45, "β=0\ndiverges", ha="center", fontsize=10)
    ax[0, 2].set(xlabel=r"$\eta\lambda$", ylabel="max relative hill-loss reduction",
                 ylim=(0, 1.05), title="(c) improvement is regime-scoped")
    ax[0, 2].legend(fontsize=9, loc="upper left")

    # (d) escape frequency vs beta per eta*lam
    for ei, el in enumerate(etalams):
        ax[1, 0].plot(betas, A["esc"][ci, ei], marker="o", color=el_color(el),
                      lw=style.LW.PRIMARY, label=rf"$\eta\lambda={el}$")
    ax[1, 0].set(xlabel="β", ylabel="escape frequency",
                 title=f"(d) tube escape (R={rec['config']['tube_radius']})")
    ax[1, 0].legend(fontsize=9, ncol=2)

    # (e) stability map: diverged fraction with the CL-1 boundary
    div_frac = A["ndiv"][ci] / rec["config"]["n_seeds"]
    im = ax[1, 1].imshow(div_frac, origin="lower", aspect="auto", cmap="Reds",
                         vmin=0, vmax=1)
    b_fine = np.linspace(0, 0.995, 200)
    el_bound = np.array([Cl.stability_threshold(b) for b in b_fine])
    bx = np.interp(b_fine, betas, range(len(betas)))
    ey = np.interp(el_bound, etalams, range(len(etalams)))
    inside = el_bound <= max(etalams)
    ax[1, 1].plot(bx[inside], ey[inside], color="#1d4ed8", lw=style.LW.GUIDE,
                  label=r"CL-1: $\eta\lambda=2T_{\mathrm{eff}}$")
    ax[1, 1].set(xticks=range(len(betas)), xticklabels=[str(b) for b in betas],
                 yticks=range(len(etalams)), yticklabels=[str(e) for e in etalams],
                 xlabel="β", ylabel=r"$\eta\lambda$",
                 title="(e) diverged fraction vs CL-1")
    ax[1, 1].tick_params(axis="x", labelsize=8)
    ax[1, 1].legend(fontsize=9, loc="upper right")
    fig.colorbar(im, ax=ax[1, 1], shrink=0.85)

    # (f) normalization ablation: EMA fixed eta vs heavy-ball fixed eta_HB
    ax[1, 2].errorbar(betas, A["ema_row"], yerr=A["ema_row_se"], marker="o",
                      color="#1d4ed8", lw=style.LW.PRIMARY,
                      label=r"EMA, fixed $\eta$ ($\eta\lambda=0.6$)")
    ax[1, 2].plot(betas, A["ema_row_pred"], ls="--", color="#1d4ed8", lw=style.LW.GUIDE,
                  alpha=0.7)
    ax[1, 2].errorbar(betas, A["hb_hill"], yerr=A["hb_se"], marker="s",
                      color="#dc2626", lw=style.LW.PRIMARY,
                      label=r"heavy-ball, fixed $\eta_{\mathrm{HB}}$"
                            r" ($\eta_{\mathrm{HB}}\lambda=0.6$)")
    ax[1, 2].plot(betas, A["hb_pred"], ls="--", color="#dc2626", lw=style.LW.GUIDE,
                  alpha=0.7)
    ax[1, 2].set(xlabel="β", ylabel="tail hill loss", yscale="log",
                 title="(f) normalization decides the direction")
    ax[1, 2].legend(fontsize=9)

    fig.suptitle("E9  straight noisy valley: β improves optimization exactly where "
                 "CL-1/CL-3 predict", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
