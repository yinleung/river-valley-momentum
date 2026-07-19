"""Figure E8 — headline regime figure: momentum restores the tube, extends stability, lags late.

Pure module: reads the cached valley_regimes run (panels a, b) and the straight_valley run
(panel c) -> out/fig_e8_headline.{pdf,png}.
    cd figures && python fig_e8_headline.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import style  # noqa: E402

import _data  # noqa: E402

LABEL = "fig_e8_headline"
PANEL_BETAS = {"a": [0.0, 0.5, 0.9, 0.999], "b": [0.0, 0.5, 0.9]}


def _loss_grid(cfg, xs, ys):
    X, Y = np.meshgrid(xs, ys)
    f = cfg["a"] * np.sin(cfg["k"] * X)
    return 0.5 * cfg["mu"] * (X - cfg["x_star"]) ** 2 + 0.5 * cfg["lam"] * (Y - f) ** 2


def _traj_panel(ax, rec, panel, betas, title):
    cfg = rec["config"]
    A = rec["arrays"]
    xs = np.linspace(-7, 9, 321)
    ys = np.linspace(-5, 5, 201)
    Z = _loss_grid(cfg, xs, ys)
    ax.contourf(xs, ys, np.log10(Z + 0.05), levels=24, cmap="Greys", alpha=0.55)
    ax.plot(xs, cfg["a"] * np.sin(cfg["k"] * xs), color="#0f766e", lw=1.6, ls=":",
            label="river floor")
    for b in betas:
        W = A[f"{panel}_b{b}_W"]
        ok = np.all(np.isfinite(W), axis=1)
        ax.plot(W[ok, 0], W[ok, 1], lw=style.LW.PRIMARY, color=style.beta_color(min(b, 0.99)),
                label=f"β={b}")
        if not ok.all():  # diverged: mark the last finite point
            last = np.where(ok)[0][-1]
            ax.plot(W[last, 0], W[last, 1], "x", ms=11, mew=2.6,
                    color=style.beta_color(min(b, 0.99)))
    ax.plot(*W[0, :2], marker="o", ms=6, color="k")
    ax.set(xlim=(-7, 9), ylim=(-5, 5), xlabel="$x$ (river)", ylabel="$y$ (hill)", title=title)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    # loss-vs-iteration inset (seed-mean where finite)
    ins = ax.inset_axes([0.045, 0.62, 0.42, 0.33])
    for b in betas:
        L = A[f"{panel}_b{b}_losscurves"]
        cnt = np.sum(np.isfinite(L), axis=0)
        mean = np.full(L.shape[1], np.nan)
        if np.any(cnt > 0):  # all-NaN columns stay NaN (diverged seeds)
            mean[cnt > 0] = np.nanmean(L[:, cnt > 0], axis=0)
        ins.plot(mean, lw=1.2, color=style.beta_color(min(b, 0.99)))
    ins.set(yscale="log", xticks=[], ylabel="loss")
    ins.tick_params(labelsize=8)
    ins.yaxis.label.set_size(9)


def build() -> None:
    style.apply_style()
    rec = _data.load_task("valley_regimes")
    rec_e1 = _data.load_task("straight_valley")

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.9))
    m = rec["metrics"]
    _traj_panel(ax[0], rec, "a", PANEL_BETAS["a"],
                f"(a) noisy, $\\eta\\lambda=1.8$ ({rec['config']['n_seeds']} seeds)")
    _traj_panel(ax[1], rec, "b", PANEL_BETAS["b"],
                f"(b) $\\eta\\lambda=2.5>2$: β=0 diverges "
                f"{m['b_b0.0_n_div']}/{rec['config']['n_seeds']}")
    ax[0].annotate(f"β=0.9 tail rms {m['a_b0.9_rms']:.3f}\nCL-2 guide {m['a_b0.9_pred_rms']:.3f}",
                   xy=(0.03, 0.04), xycoords="axes fraction", fontsize=10,
                   bbox=dict(fc="white", alpha=0.85, ec="0.6"))
    ax[1].annotate(f"β=0.9 tail rms {m['b_b0.9_rms']:.3f}\nCL-2 guide {m['b_b0.9_pred_rms']:.3f}",
                   xy=(0.03, 0.04), xycoords="axes fraction", fontsize=10,
                   bbox=dict(fc="white", alpha=0.85, ec="0.6"))

    # (c) the deterministic straight-valley caveat: rms floor distance vs beta (E1)
    d_rms = rec_e1["metrics"]["dist_rms"]
    bs = [float(k) for k in d_rms]
    ax[2].plot(bs, [d_rms[k] for k in d_rms], marker="o", lw=style.LW.PRIMARY,
               color="#7c3aed")
    ax[2].set(xlabel="β", ylabel=r"$d_{\mathrm{rms}}$",
              title="(c) clean straight valley (E1)")
    for b, k in zip(bs, d_rms):
        ax[2].annotate(f"{d_rms[k]:.2f}", (b, d_rms[k]), textcoords="offset points",
                       xytext=(0, 7), fontsize=9, ha="center")

    fig.suptitle("E8  regimes of momentum in the river valley: tube restoration (a), "
                 "stability extension (b), deterministic caveat (c)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    build()
