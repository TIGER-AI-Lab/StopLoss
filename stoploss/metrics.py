"""Ranking metrics (paper Sec. 3.3, "Prediction").

* ``auc``: probability that a window of a run that eventually fails gets a higher failure score than a window of a
  run that succeeds (ties count one half).
* ``tier_auc``: failure windows tend to occur later and at higher spend. Tier AUC splits a cell's checkpoint windows
  at the quartiles of cumulative execution spend, computes the AUC inside every tier that contains both outcomes,
  and averages those tiers, so that windows are compared at similar accumulated cost.
"""
from __future__ import annotations

import numpy as np


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """P(score of a failed window > score of a solved window) + 0.5 * ties. ``y``: 1.0 solved, 0.0 failed."""
    n_fail = int((y == 0.0).sum())
    n_solved = int((y == 1.0).sum())
    if n_fail == 0 or n_solved == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    _, inverse, counts = np.unique(p[order], return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    mid_ranks = (ends - counts + 1 + ends) / 2.0
    ranks = np.empty(len(p))
    ranks[order] = mid_ranks[inverse]
    return float((ranks[y == 0.0].sum() - n_fail * (n_fail + 1) / 2.0) / (n_fail * n_solved))


def tier_auc(spend: np.ndarray, p: np.ndarray, y: np.ndarray) -> float:
    """Mean AUC over the four spend-quartile tiers (-inf, q1], (q1, q2], (q2, q3], (q3, inf) that contain both outcomes."""
    if len(spend) == 0:
        return float("nan")
    q = np.quantile(spend, [0.25, 0.5, 0.75])
    tiers = []
    for lo, hi in ((-1, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e9)):
        sel = (spend > lo) & (spend <= hi)
        value = auc(p[sel], y[sel])
        if not np.isnan(value):
            tiers.append(value)
    return float(np.mean(tiers)) if tiers else float("nan")
