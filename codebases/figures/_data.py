"""Offline cache loader for figure modules.

Figures read run records from results/cache/ — never the live tracker, never recomputing an
experiment (../CODING.md Pillar 4). This helper resolves a task's most recent run from the
offline index and loads its config / metrics / arrays.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "results" / "cache"
INDEX = _ROOT / "results" / "index" / "runs.csv"


def latest_run(task: str, probe: str | None = None) -> str:
    """Run id of the most recently created run for `task` (from results/index/runs.csv).

    Tasks can log several probe protocols (e.g. nanogpt closedloop vs decomp); pass `probe`
    to disambiguate. With probe=None all of the task's runs compete on recency.
    """
    with open(INDEX, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["task"] == task and (probe is None or r["probe"] == probe)]
    if not rows:
        raise FileNotFoundError(f"no run for task {task!r} (probe {probe!r}) in {INDEX}")
    return max(rows, key=lambda r: r["created"])["run_id"]


def load(run_id: str) -> dict:
    """Load a run's config, metrics, and arrays from results/cache/<run-id>/."""
    d = CACHE / run_id
    with open(d / "config.json") as f:
        config = json.load(f)
    with open(d / "metrics.json") as f:
        metrics = json.load(f)
    arrays = dict(np.load(d / "arrays.npz")) if (d / "arrays.npz").exists() else {}
    return {"config": config, "metrics": metrics, "arrays": arrays, "run_id": run_id}


def load_task(task: str, probe: str | None = None) -> dict:
    """Convenience: load the latest run for a task (optionally a specific probe)."""
    return load(latest_run(task, probe))
