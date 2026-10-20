"""Calibrated early stopping and its cost accounting (paper Sec. 2.3 and 3.3).

A *run row* is one completed execution: ``{"task", "reward" (1.0 / 0.0), "scores" {k: failure score},
"spend" [cumulative $ after tool step 1..n, full-run total], "n" (number of tool steps)}``. Only policy checkpoints
k = 5, 8, 11, ... are used. ``spend[k - 1]`` is what the run has cost when it is stopped at checkpoint k;
``spend[-1]`` is the full run including calls after the last tool step.

Calibration: for each successful calibration run that reaches a checkpoint take its maximum checkpoint score;
with n such runs and a nominal false-kill rate rho, tau is placed just above the (floor(rho * n) + 1)-th largest
maximum, so at most floor(rho * n) calibration successes would be stopped (tied maxima are never split).

Replay: a run is stopped at the first checkpoint whose score reaches tau, pays execution up to that checkpoint plus
every check it visited, and loses its success if it would have succeeded. Runs without checkpoints always survive.
CPS = (execution + monitoring) / retained successes; Delta CPS = 100 * (CPS / CPS_0 - 1), CPS_0 = full execution
without monitoring.
"""
from __future__ import annotations

import math

import numpy as np

FIRST_CHECK, CHECK_EVERY = 5, 3
RATES = (0.02, 0.05, 0.10)


def is_checkpoint(k: int) -> bool:
    return k >= FIRST_CHECK and (k - FIRST_CHECK) % CHECK_EVERY == 0


def checkpoints(row: dict) -> list[int]:
    return [k for k in sorted(row["scores"]) if is_checkpoint(k)]


def statistic(row: dict, k: int, policy: str = "score") -> float:
    """What the stopping rule thresholds: the failure score, or the step count itself (a free baseline)."""
    if policy == "steps":
        return float(k)
    return row["scores"][k]


def calibrate(cal_rows: list[dict], policy: str = "score", rates=RATES) -> tuple[dict, int]:
    """({rate: tau}, number of successful calibration runs with at least one checkpoint)."""
    maxima = sorted(v for v in (max((statistic(r, k, policy) for k in checkpoints(r)), default=float("nan"))
                                for r in cal_rows if r["reward"] == 1.0) if not math.isnan(v))
    taus = {}
    for rate in rates:
        m = int(rate * len(maxima))
        taus[rate] = (maxima[-(m + 1)] + 1e-9) if m < len(maxima) else 0.0
    return taus, len(maxima)


def calibration_report(cal_rows: list[dict], taus: dict, policy: str = "score") -> dict:
    """How many calibration successes each threshold actually stops (at most floor(rate * n), fewer with ties)."""
    solved = [r for r in cal_rows if r["reward"] == 1.0]
    with_check = [r for r in solved if checkpoints(r)]
    out = {"n_solved": len(solved), "n_solved_with_checkpoint": len(with_check), "rates": {}}
    for rate, tau in taus.items():
        kills = sum(1 for r in with_check if max(statistic(r, k, policy) for k in checkpoints(r)) >= tau)
        out["rates"][str(rate)] = {"tau": tau, "target_at_most": int(rate * len(with_check)), "actual_calibration_false_kills": kills}
    return out


def replay(rows: list[dict], tau: float, check_cost=None, policy: str = "score") -> dict:
    """Per-run arrays of the stopping replay. ``check_cost(row, k)`` -> $ of reading checkpoint k (None: free)."""
    n = len(rows)
    spend, killed, calls, monitor = np.zeros(n), np.zeros(n, bool), np.zeros(n, int), np.zeros(n)
    for i, r in enumerate(rows):
        stop = None
        for k in checkpoints(r):
            calls[i] += 1
            if check_cost is not None:
                monitor[i] += check_cost(r, k)
            if statistic(r, k, policy) >= tau:
                stop = k
                break
        spend[i] = r["spend"][-1] if stop is None else r["spend"][min(stop, r["n"]) - 1]
        killed[i] = stop is not None
    solved = np.array([r["reward"] == 1.0 for r in rows])
    full = np.array([r["spend"][-1] for r in rows])
    return {"spend": spend, "killed": killed, "false_kill": killed & solved, "kept": solved & ~killed,
            "calls": calls, "monitor": monitor, "solved": solved, "full": full}


def stopping_metrics(rep: dict, weights: np.ndarray | None = None) -> dict:
    """Aggregate a replay; ``weights`` are bootstrap multiplicities per run (None: every run once)."""
    w = np.ones(len(rep["spend"])) if weights is None else weights
    total = lambda a: float((np.asarray(a, float) * w).sum())
    n = float(w.sum())
    solved, kept, spend, full, monitor = total(rep["solved"]), total(rep["kept"]), total(rep["spend"]), total(rep["full"]), total(rep["monitor"])
    kills, false_kills = total(rep["killed"]), total(rep["false_kill"])
    nan = float("nan")
    base = full / solved if solved else nan
    cps = spend / kept if kept else nan
    net = (spend + monitor) / kept if kept else nan
    failed = n - solved
    return {"n": n, "solved": solved, "kept": kept, "kills": kills, "false_kills": false_kills, "spend": spend, "full": full,
            "monitor_usd": monitor, "calls": total(rep["calls"]), "base_cps": base, "cps": cps, "net_cps": net,
            "delta_pct": 100 * (cps / base - 1) if base and kept else nan,
            "net_delta_pct": 100 * (net / base - 1) if base and kept else nan,
            "fk_rate": false_kills / solved if solved else nan, "retention": kept / solved if solved else nan,
            "failure_capture": (kills - false_kills) / failed if failed else nan}
