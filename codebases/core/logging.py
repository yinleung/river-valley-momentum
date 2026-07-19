"""Unified run logging: one self-describing record per run under results/cache/.

This is the lightweight local backend for the record-everything layer (../CODING.md Pillar 3).
The standard's default tracker is Weights & Biases; it is not installed here and these toy runs
are offline and deterministic, so we mirror the *intent* directly:

  - every run gets a unified, sortable id  <task>_<probe>_<key>_<sha8>;
  - the full config (plus a code SHA and environment versions) is written to config.json;
  - scalar metrics go to metrics.json, heavy arrays to arrays.npz;
  - a queryable index row is appended to results/index/runs.csv;
  - figures read results/cache/, never recompute.

Run identity (plan_v5 §3.1 B3): with the repo under git, a clean tree stamps the run id with
`git rev-parse --short=8 HEAD`, tying every record to a commit; a dirty tree (or missing git)
falls back to the content hash of core/*.py + caller files, so uncommitted edits still churn
the cache key. The config records `git_sha`, `git_dirty`, and `content_sha` explicitly either
way. Legacy callers that pass `sha=` keep their ids unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

__all__ = ["code_sha", "git_sha", "log_run", "CACHE_DIR", "INDEX_CSV"]

_CORE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CORE_DIR.parent.parent
_RESULTS = _CORE_DIR.parent / "results"
CACHE_DIR = _RESULTS / "cache"
INDEX_CSV = _RESULTS / "index" / "runs.csv"

_INDEX_FIELDS = ["run_id", "task", "probe", "group", "job_type", "key", "code_sha",
                 "git_sha", "stage", "created"]


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


def git_sha() -> tuple[str | None, bool]:
    """(short-8 HEAD sha, dirty flag) of the repo, or (None, True) when unresolvable.

    Dirty = any modified tracked CODE file. Untracked files are ignored (data/ and caches
    are gitignored), and `codebases/results/` is excluded from the check: log_run itself
    appends to the tracked runs.csv, so run OUTPUTS must not flip later runs in the same
    job from the git sha to the content hash (Codex Phase-0 review finding). Falls back to
    reading .git/HEAD directly where the git binary is unavailable — the dirty flag is
    then unknowable and reported True, so the cache key stays on the content hash.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--",
             ".", ":(exclude)codebases/results"], cwd=_REPO_ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return sha, bool(status)
    except Exception:
        pass
    try:  # file fallback: HEAD -> ref -> sha (dirty unknowable without git)
        head = (_REPO_ROOT / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = _REPO_ROOT / ".git" / head[5:]
            head = ref.read_text().strip() if ref.exists() else ""
        return (head[:8] or None), True
    except Exception:
        return None, True


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
    extra_files: Iterable[str | Path] = (),
) -> str:
    """Write one run's record to results/cache/<run-id>/ and append the index row.

    Args:
        task: experiment family, e.g. "straight_valley" (becomes the run `group`).
        probe: protocol/measurement, e.g. "trajectory" (becomes the run `job_type`).
        key: compact key-params string, e.g. "eta0p18_lam10".
        config: full parameter dict; augmented here with sha/git/environment identity.
            Campaign runs must carry config["stage"] = "explore" | "confirm" (§3.0.2).
        metrics: scalar summary metrics (must be JSON-serializable after coercion).
        arrays: optional heavy arrays saved to arrays.npz (figures read these).
        sha: explicit run-id sha (legacy callers); if omitted, a clean git tree stamps the
            git sha8 and a dirty/git-less tree stamps the content hash (B3 policy).
        extra_files: caller source files folded into the content hash (e.g. __file__).

    Returns:
        run_id = "<task>_<probe>_<key>_<sha8>".
    """
    gsha, dirty = git_sha()
    content = code_sha(extra_files)
    if sha is None:
        sha = gsha if (gsha and not dirty) else content
    run_id = f"{task}_{probe}_{key}_{sha}"
    run_dir = CACHE_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    full_config = {
        **dict(config),
        "task": task,
        "probe": probe,
        "code_sha": sha,
        "git_sha": gsha or "",
        "git_dirty": dirty,
        "content_sha": content,
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
            "git_sha": gsha or "",
            "stage": str(dict(config).get("stage", "")),
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
        w = csv.DictWriter(f, fieldnames=_INDEX_FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
