"""Paired repository-cluster bootstrap (paper Sec. 3.4).

Tasks are clustered by repository. For each benchmark one matrix of cluster multiplicities is drawn and shared by
every method and by both executors of that benchmark, so differences between methods (and cells) are paired
resample by resample. A drawn repository carries all of its runs and windows.
"""
from __future__ import annotations

import hashlib

import numpy as np


def repo_of(task: str, cluster: str = "repo") -> str:
    """Repository of a task id: ``owner__repo-1234`` -> ``owner__repo``; SWE-bench Pro ids ``instance_<org>__<repo>-<hash>...``
    -> ``<org>``. ``cluster="task"`` makes every task its own cluster."""
    if cluster == "task":
        return task
    if task.startswith("instance_"):
        return task[len("instance_"):].split("-", 1)[0]
    return task.rsplit("-", 1)[0]


def bench_multiplicities(tasks, cluster: str, draws: int, seed: int, bench: str) -> tuple[dict, np.ndarray]:
    """({cluster: column}, multiplicities of shape (draws, n_clusters)) for one benchmark.

    The generator is seeded with (seed, hash(bench)), so the draws depend only on the benchmark's task ids, the
    seed and the number of draws, never on scores or on which methods are evaluated.
    """
    index: dict[str, int] = {}
    for t in sorted(tasks):
        index.setdefault(repo_of(t, cluster), len(index))
    rng = np.random.default_rng([seed, int(hashlib.sha256(bench.encode()).hexdigest()[:8], 16)])
    n = len(index)
    m = np.zeros((draws, n), int)
    for b in range(draws):
        np.add.at(m[b], rng.integers(0, n, n), 1)
    return index, m


def ci95(values) -> list[float]:
    """Percentile 95% interval, ignoring undefined (NaN / None) resamples."""
    v = np.array([x for x in values if x is not None and not np.isnan(x)], float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def simultaneous_intervals(points: dict, draws: dict, level: float = 0.95) -> tuple[dict, float]:
    """Max-|t| bootstrap band over a family of paired comparisons whose draws are aligned resample by resample.

    ``points`` {name: estimate}, ``draws`` {name: [one value per resample]} -> ({name: [lo, hi]}, critical value).
    """
    names = [n for n in points if n in draws and len(draws[n])]
    if not names:
        return {}, float("nan")
    d = np.array([draws[n] for n in names], float)
    se = np.nanstd(d, axis=1, ddof=1)
    ok = (se > 0) & np.isfinite(se)
    t = np.abs(d - np.array([points[n] for n in names])[:, None]) / np.where(ok, se, np.nan)[:, None]
    t_max = np.nanmax(t[ok], axis=0) if ok.any() else np.array([np.nan])
    c = float(np.nanquantile(t_max, level))
    out = {}
    for i, n in enumerate(names):
        if ok[i]:
            out[n] = [float(points[n] - c * se[i]), float(points[n] + c * se[i])]
        else:
            out[n] = [float(points[n])] * 2 if se[i] == 0 else [float("nan")] * 2
    return out, c
