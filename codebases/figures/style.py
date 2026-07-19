"""Shared figure style for the figures/ package.

One place for rcParams, sizes, line widths, colors, and markers. Every figure module imports
from here and never redefines style locally. See ../CODING.md Pillar 4.

The numeric defaults below are sensible for single- and multi-panel research figures; adapt the
color / marker registries to each project, but keep the typography and sizes uniform so figures
look like one set.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt

# --- rcParams --------------------------------------------------------------
RCPARAMS = {
    "font.size": 16,
    "axes.labelsize": 17,
    "axes.titlesize": 17,   # match labelsize so per-panel subtitles read at the same scale
    "legend.fontsize": 13,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_style() -> None:
    """Apply the shared rcParams. Call once at the top of every figure's build()."""
    plt.rcParams.update(RCPARAMS)


# --- figure sizes ----------------------------------------------------------
FIGSIZE_SINGLE = (6.5, 5.2)   # one panel
FIGSIZE_ROW = (13.0, 5.2)     # 2x1 row


def grid_figsize(nrows: int, ncols: int) -> Tuple[float, float]:
    """Figure size for an (nrows x ncols) panel grid. Share x/y axes when you use it."""
    return (3.5 * ncols, 3.0 * nrows)


# --- line widths -----------------------------------------------------------
class LW:
    PRIMARY = 1.8   # data curve
    GUIDE = 2.0     # dashed reference / theoretical guide
    RAW = 1.5       # raw / background overlay
    MEAN = 1.9      # mean / signal overlay


# --- colors & markers ------------------------------------------------------
# Reference-curve roles common to probe figures. Rename / extend per project.
ROLE_COLORS = {
    "raw": "#808080",     # grey background series
    "guide": "#111111",   # dashed theoretical guide
    "mean": "#ea580c",    # orange mean / signal overlay
}

# Extensible per-series style. Add one entry per named series (pipelines, layers, ...).
# Each value is a dict of matplotlib kwargs merged into plot calls.
SERIES_STYLE: Dict[str, dict] = {
    # "series-a": {"color": "#1d4ed8", "marker": "s", "markersize": 8},
    # "series-b": {"color": "#dc2626", "marker": "D", "markersize": 8},
    # "series-c": {"color": "#059669", "marker": "^", "markersize": 8},
}


def series_style(name: str) -> dict:
    """Plot kwargs for a named series, or an empty dict if unregistered."""
    return dict(SERIES_STYLE.get(name, {}))


# --- project series registries (momentum-filtering figures) ----------------
# Momentum coefficient beta -> color, on a single perceptual ramp (low beta light, high dark).
import matplotlib as _mpl  # noqa: E402

_BETA_CMAP = _mpl.colormaps["viridis"]


def beta_color(beta: float, beta_max: float = 0.99):
    """Color for a momentum coefficient on the shared viridis ramp (beta in [0, beta_max])."""
    return _BETA_CMAP(0.12 + 0.78 * (beta / beta_max))


# Muon-style pipelines compared in E5 — one fixed color/marker each.
PIPELINE_STYLE = {
    "pre_polar": {"color": "#1d4ed8", "marker": "o", "markersize": 7, "label": "pre-polar"},
    "post_polar": {"color": "#dc2626", "marker": "s", "markersize": 7, "label": "post-polar"},
    "polar_only": {"color": "#6b7280", "marker": "^", "markersize": 7, "label": "polar-only"},
}

# Gradient-noise models compared in E3.
NOISE_STYLE = {
    "gaussian": {"color": "#1d4ed8", "marker": "o", "label": "Gaussian"},
    "anisotropic": {"color": "#059669", "marker": "s", "label": "anisotropic hill"},
    "heavy_tailed": {"color": "#b45309", "marker": "D", "label": "heavy-tailed"},
}

# Forcing-frequency arms of the E13 controls — one fixed color/marker per arm, low to high.
FORCING_STYLE = [
    {"color": "#1d4ed8", "marker": "o", "markersize": 7},   # passband
    {"color": "#059669", "marker": "s", "markersize": 7},   # resonance (theta_0.9)
    {"color": "#b45309", "marker": "D", "markersize": 7},   # high band
    {"color": "#dc2626", "marker": "^", "markersize": 7},   # Nyquist
]

# E12 decomposition streams: raw mini-batch / state (large-batch) / sampling residual.
DECOMP_STYLE = {
    "mb": {"color": "#808080", "label": "raw $g^{\\mathrm{mb}}$"},
    "lb": {"color": "#ea580c", "label": "state $\\bar g^{\\mathrm{LB}}$"},
    "xi": {"color": "#1d4ed8", "label": "residual $\\xi^{\\mathrm{res}}$"},
}


# --- saving ----------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "out"


def save_figure(fig, label: str, out_dir: Optional[Path] = None) -> None:
    """Write <label>.pdf and <label>.png into the single out/ directory, then close the figure."""
    out = out_dir or OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{label}.{ext}")
    plt.close(fig)
