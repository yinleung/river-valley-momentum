"""Template figure module — copy to fig_<label>.py and fill in the TODOs.

One PURE module per figure: build() reads cached run records and writes out/<label>.{pdf,png}.
It never recomputes an experiment and never calls the live tracker. See ../CODING.md Pillar 4.

Run from inside figures/ (so `import style` resolves):
    cd figures && python fig_<label>.py
"""
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

import style  # shared figure style (sibling module in this package)

LABEL = "template"  # -> rename to the paper's \label key, e.g. "fig_spectral_gap"

CACHE = Path(__file__).resolve().parents[1] / "results" / "cache"


def load_run(run_id: str) -> dict:
    """Load one cached run's record from results/cache/<run-id>/.

    Adapt to your record format (json, npz, ...). Figures read the offline cache, never the
    live tracker. Discover run-ids in results/index/runs.csv.
    """
    with open(CACHE / run_id / "metrics.json") as f:
        return json.load(f)


def build(run_ids: List[str]) -> None:
    """Render the figure from cached records and save to out/."""
    style.apply_style()
    records = [load_run(r) for r in run_ids]

    fig, ax = plt.subplots(figsize=style.FIGSIZE_SINGLE)
    for rec in records:
        # TODO: replace with the real metric and axes for this figure.
        ax.plot(
            rec["x"],
            rec["y"],
            linewidth=style.LW.PRIMARY,
            label=rec.get("series", ""),
            **style.series_style(rec.get("series", "")),
        )
    ax.set_xlabel("x")  # TODO
    ax.set_ylabel("y")  # TODO
    ax.legend()

    style.save_figure(fig, LABEL)


if __name__ == "__main__":
    # Pass the feeding run-ids explicitly and record them in the figure's
    # results/figures/<label>/summary.md provenance.
    build(run_ids=[])
