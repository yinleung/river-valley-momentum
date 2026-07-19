"""Unified run logging: one self-describing record per run under results/cache/.

This is the lightweight local backend for the record-everything layer (../CODING.md Pillar 3).
The standard's default tracker is Weights & Biases; it is not installed here and these toy runs
are offline and deterministic, so we mirror the *intent* directly:

  - every run gets a unified, sortable id  <task>_<probe>_<key>_<sha8>;
  - the full config (plus a code SHA and environment versions) is written to config.json;
  - scalar metrics go to metrics.json, heavy arrays to arrays.npz;
  - a queryable index row is appended to results/index/runs.csv;
  - figures read results/cache/, never recompute.

Because the project is not a git repository, run identity uses a content hash of the core/
library (plus any caller-supplied source files) as the `code_sha` surrogate. Swap `code_sha`
for `git rev-parse --short HEAD` and this module for a wandb wrapper without touching callers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

__all__ = ["code_sha", "log_run", "CACHE_DIR", "INDEX_CSV"]

_CORE_DIR = Path(__file__).resolve().parent
_RESULTS = _CORE_DIR.parent / "results"
CACHE_DIR = _RESULTS / "cache"
INDEX_CSV = _RESULTS / "index" / "runs.csv"

_INDEX_FIELDS = ["run_id", "task", "probe", "group", "job_type", "key", "code_sha", "created"]


def code_sha(extra_files: Iterable[str | Path] = ()) -> str:
    """8-char content hash of the core/ library plus any caller source files.

    A git-SHA surrogate: hashes core/*.py (the written-once analysis layer) and any extra files
    (e.g. the experiment script's __file__) so each result ties to the exact code that produced
    it. Deterministic and independent of file order.
    """
    h = hashlib.sha256()
    paths = sorted(_CORE_DIR.glob("*.py")) + [Path(p) for p in extra_files]
    for p in sorted({Path(p).resolve() for p in paths}):
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


def _coerce_scalar(v):
    """JSON-friendly scalar coercion for metric values."""
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    return v


def log_run(
    *,
    task: str,
    probe: str,
    key: str,
    config: Mapping,
    metrics: Mapping,
    arrays: Mapping[str, np.ndarray] | None = None,
    sha: str | None = None,
) -> str:
    """Write one run's record to results/cache/<run-id>/ and append the index row.

    Args:
        task: experiment family, e.g. "straight_valley" (becomes the run `group`).
        probe: protocol/measurement, e.g. "trajectory" (becomes the run `job_type`).
        key: compact key-params string, e.g. "eta0p18_lam10".
        config: full parameter dict; augmented here with code_sha and environment versions.
        metrics: scalar summary metrics (must be JSON-serializable after coercion).
        arrays: optional heavy arrays saved to arrays.npz (figures read these).
        sha: code SHA; computed from core/ if omitted.

    Returns:
        run_id = "<task>_<probe>_<key>_<sha8>".
    """
    sha = sha or code_sha()
    run_id = f"{task}_{probe}_{key}_{sha}"
    run_dir = CACHE_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    full_config = {
        **dict(config),
        "task": task,
        "probe": probe,
        "code_sha": sha,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(run_dir / "config.json", "w") as f:
        json.dump(full_config, f, indent=2, default=str)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({k: _coerce_scalar(v) for k, v in metrics.items()}, f, indent=2, default=str)
    if arrays:
        np.savez_compressed(run_dir / "arrays.npz", **arrays)

    _append_index_row(
        {
            "run_id": run_id,
            "task": task,
            "probe": probe,
            "group": task,
            "job_type": probe,
            "key": key,
            "code_sha": sha,
            "created": created,
        }
    )
    return run_id


def _append_index_row(row: Mapping[str, str]) -> None:
    """Append (or replace, by run_id) one row in results/index/runs.csv."""
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if INDEX_CSV.exists():
        with open(INDEX_CSV, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("run_id") != row["run_id"]]
    rows.append(dict(row))
    rows.sort(key=lambda r: (r.get("task", ""), r.get("probe", ""), r.get("key", "")))
    with open(INDEX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
