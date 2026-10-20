"""Evaluation of one setting (benchmark x executor): ranking quality and calibrated stopping for every method
(paper Sec. 3.3-3.4; Tables 1-2, Figure 3).

Inputs are plain Python objects so the function can be driven by ``scripts/evaluate.py`` or by other code:

* ``runs``: the cohort of the setting, ``[{"task", "reward", "spend", "n"}, ...]`` (see ``stoploss.policy``);
* ``methods``: ``{name: {"scores": {task: {k: p}}, "seconds_per_check": s}}`` for a small monitor (it pays
  ``s * gpu_usd_per_hour / 3600`` per check; an ensemble pays the summed seconds of its members) or
  ``{name: {"scores": ..., "window_cost": {(task, k): $}}}`` for a judge (it pays the recorded cost of each window it
  reads). A method with neither is free. Every method must score the same checkpoint windows;
* ``calibration``: ``{name: [row, ...]}``, the successful calibration runs scored by that method (only their
  checkpoint scores are used);
* ``groups``: ``{group: [member, ...]}`` for ensembles; the ensemble is added as ``<group>_ens`` (mean failure score,
  summed check cost) and the mean single-model result as ``<group>_seedmean`` (ranking only);
* ``cluster_index`` / ``multiplicities``: the benchmark-level bootstrap draws from ``stoploss.bootstrap``.
"""
from __future__ import annotations

import collections
import time

import numpy as np

from . import bootstrap, metrics, policy

STOP_QUANTITIES = ("delta_pct", "net_delta_pct", "fk_rate", "retention", "failure_capture")


def make_rows(runs: list[dict], scores_by_task: dict) -> list[dict]:
    """One row per run of the cohort, with the method's checkpoint scores (a run may have none)."""
    rows = []
    for run in runs:
        scores = {int(k): float(p) for k, p in (scores_by_task.get(run["task"]) or {}).items() if policy.is_checkpoint(int(k))}
        if scores and max(scores) > run["n"]:
            raise ValueError(f"{run['task']}: scored window k={max(scores)} beyond the run's {run['n']} tool steps")
        rows.append({"task": run["task"], "reward": run["reward"], "scores": scores, "spend": run["spend"], "n": run["n"]})
    return rows


def ensemble_scores(members: list[dict]) -> dict:
    """{task: {k: mean member score}} over the windows every member scored."""
    out = {}
    tasks = set.union(*[set(m) for m in members]) if members else set()
    for t in sorted(tasks):
        if not all(t in m for m in members):
            continue
        ks = set.intersection(*[set(m[t]) for m in members])
        if ks:
            out[t] = {k: float(np.mean([m[t][k] for m in members])) for k in ks}
    return out


def ranking_arrays(rows: list[dict]):
    """Flat checkpoint-window arrays: spend at the window, score, run outcome, run index."""
    spend, p, y, run_index = [], [], [], []
    for i, r in enumerate(rows):
        for k in policy.checkpoints(r):
            spend.append(r["spend"][min(k, r["n"]) - 1])
            p.append(r["scores"][k])
            y.append(r["reward"])
            run_index.append(i)
    return np.array(spend, float), np.array(p, float), np.array(y, float), np.array(run_index, int)


def _cost_fn(method: dict, gpu_usd_per_hour: float):
    """Monitoring charge of reading checkpoint k of a run, or None for a free method."""
    if method.get("window_cost") is not None:
        return lambda r, k, c=method["window_cost"]: c[(r["task"], k)]     # a judged window without a cost record is an error
    seconds = method.get("seconds_per_check") or 0.0
    return (lambda r, k, c=seconds * (gpu_usd_per_hour / 3600.0): c) if seconds else None


def mean_check_cost(method: dict, gpu_usd_per_hour: float, windows) -> float:
    """Average cost of one prediction over the given (task, k) windows (App. C.2)."""
    if method.get("window_cost") is not None:
        return float(np.mean([method["window_cost"][w] for w in windows])) if windows else float("nan")
    return (method.get("seconds_per_check") or 0.0) * (gpu_usd_per_hour / 3600.0)


def evaluate_cell(runs, methods, calibration, cluster_index, multiplicities, groups=None, primary=None,
                  rates=policy.RATES, cluster="repo", gpu_usd_per_hour=0.8, bootstrap_draws=True,
                  fixed_steps=None, fixed_step_methods=(), timing_method=None, timing_rate=0.05, log=print) -> dict:
    """Ranking, thresholds, stopping replay and bootstrap intervals for every method of one setting.

    Optional extras: ``fixed_steps`` / ``fixed_step_methods`` add the fixed-checkpoint AUC of Figure 3 and
    ``timing_method`` the stopping-timing summary of Sec. 5.1 (both descriptive, on the given cohort).
    """
    groups = groups or {}
    methods = dict(methods)
    calibration = dict(calibration)
    for g, members in groups.items():
        methods[g + "_ens"] = {"scores": ensemble_scores([methods[m]["scores"] for m in members]),
                               "seconds_per_check": sum(methods[m].get("seconds_per_check") or 0.0 for m in members)}
        cal_members = [calibration[m] for m in members]
        calibration[g + "_ens"] = [dict(rs[0], scores=ensemble_scores([{0: r["scores"]} for r in rs]).get(0, {}))
                                   for rs in zip(*cal_members)]
    rows = {n: make_rows(runs, m["scores"]) for n, m in methods.items()}
    ref = next(iter(rows.values()))
    for n, rs in rows.items():
        if [sorted(r["scores"]) for r in rs] != [sorted(r["scores"]) for r in ref]:
            raise ValueError(f"method {n} does not score the same checkpoint windows as the other methods")
    ids = np.array([cluster_index[bootstrap.repo_of(r["task"], cluster)] for r in ref], int)

    # ---------------- ranking on the checkpoint windows (plus the step-count-only baseline s(k) = k)
    spend, _, y, run_index = ranking_arrays(ref)
    scores = {n: ranking_arrays(rs)[1] for n, rs in rows.items()}
    scores["stepcount"] = np.array([k for r in ref for k in policy.checkpoints(r)], float)

    def rank_all(idx=None):
        sel = slice(None) if idx is None else idx
        out = {n: (metrics.tier_auc(spend[sel], p[sel], y[sel]), metrics.auc(p[sel], y[sel])) for n, p in scores.items()}
        for g, members in groups.items():
            out[g + "_seedmean"] = tuple(float(np.mean([out[m][j] for m in members])) for j in (0, 1))
        return out

    point_rank = rank_all()

    # ---------------- calibrated stopping
    taus, cal_report, replays = {}, {}, {}
    for n in rows:
        taus[n], _ = policy.calibrate(calibration[n], "score", rates)
        cal_report[n] = policy.calibration_report(calibration[n], taus[n], "score")
        cost = _cost_fn(methods[n], gpu_usd_per_hour)
        replays[n] = {rate: policy.replay(rows[n], taus[n][rate], cost) for rate in rates}
    # free baseline: stop at a calibrated step count, without reading the trajectory
    cal_any = calibration[next(iter(rows))]
    taus["stepcount"], _ = policy.calibrate(cal_any, "steps", rates)
    cal_report["stepcount"] = policy.calibration_report(cal_any, taus["stepcount"], "steps")
    replays["stepcount"] = {rate: policy.replay(ref, taus["stepcount"][rate], None, "steps") for rate in rates}
    point_stop = {n: {rate: policy.stopping_metrics(replays[n][rate]) for rate in rates} for n in replays}

    # ---------------- bootstrap: one multiplicity vector per resample, shared by all methods
    draws = collections.defaultdict(list)
    primary_ens = primary + "_ens" if primary else None
    if bootstrap_draws:
        t0 = time.time()
        for m in multiplicities:
            per_run = m[ids]
            weights = per_run.astype(float)
            idx = np.repeat(np.arange(len(run_index)), per_run[run_index]) if len(run_index) else np.zeros(0, int)
            ranked = rank_all(idx) if len(idx) else {n: (float("nan"),) * 2 for n in point_rank}
            for n, v in ranked.items():
                draws[("rank", n)].append(v)
            stopped = {n: {rate: policy.stopping_metrics(replays[n][rate], weights) for rate in rates} for n in replays}
            for n in stopped:
                for rate in rates:
                    draws[("stop", n, rate)].append([stopped[n][rate][q] for q in STOP_QUANTITIES])
            if primary_ens:
                for n in ranked:
                    if n != primary_ens:
                        draws[("drank", n)].append((ranked[n][0] - ranked[primary_ens][0], ranked[n][1] - ranked[primary_ens][1]))
                for n in stopped:
                    if n != primary_ens:
                        for rate in rates:
                            draws[("dstop", n, rate)].append(stopped[n][rate]["net_delta_pct"] - stopped[primary_ens][rate]["net_delta_pct"])
        log(f"   bootstrap: {len(multiplicities)} resamples over {len(cluster_index)} clusters in {time.time() - t0:.0f}s")

    column = lambda values, j: bootstrap.ci95([v[j] for v in values])
    ranking = {"n_checkpoint_windows": int(len(y)), "n_fail_windows": int((y == 0).sum()), "methods": {}}
    for n, (tier, plain) in point_rank.items():
        ranking["methods"][n] = {"tier_auc": tier, "auc": plain,
                                 "tier_auc_ci95": column(draws[("rank", n)], 0) if bootstrap_draws else None,
                                 "auc_ci95": column(draws[("rank", n)], 1) if bootstrap_draws else None}
    for g, members in groups.items():
        ranking["methods"][g + "_seedmean"]["per_member"] = {m: {"tier_auc": point_rank[m][0], "auc": point_rank[m][1]} for m in members}
    stopping = {"thresholds": {n: {str(r): taus[n][r] for r in rates} for n in taus}, "calibration": cal_report, "methods": {}}
    for n in replays:
        stopping["methods"][n] = {}
        for rate in rates:
            entry = dict(point_stop[n][rate])
            if bootstrap_draws:
                entry["ci95"] = {q: column(draws[("stop", n, rate)], j) for j, q in enumerate(STOP_QUANTITIES)}
            stopping["methods"][n][str(rate)] = entry
    paired = {}
    if primary_ens and bootstrap_draws:
        for n in point_rank:
            if n == primary_ens:
                continue
            d = draws[("drank", n)]
            paired[n] = {"tier_auc": {"point": point_rank[n][0] - point_rank[primary_ens][0], "ci95": column(d, 0),
                                      "p_le0": float(np.mean([v[0] <= 0 for v in d])) if d else None, "draws": [v[0] for v in d]}}
            if n in replays:
                for rate in rates:
                    dd = draws[("dstop", n, rate)]
                    paired[n][f"net_delta_pct@{rate}"] = {"point": point_stop[n][rate]["net_delta_pct"] - point_stop[primary_ens][rate]["net_delta_pct"],
                                                          "ci95": bootstrap.ci95(dd), "draws": dd}
    windows = [(r["task"], k) for r in ref for k in policy.checkpoints(r)]
    result = {"n_runs": len(runs), "n_solved": int(sum(r["reward"] == 1.0 for r in runs)),
              "n_runs_without_checkpoint": sum(1 for r in ref if not policy.checkpoints(r)), "n_checkpoint_windows": int(len(y)),
              "clusters": {"kind": cluster, "n_in_benchmark": len(cluster_index), "n_in_cell": int(len(set(ids.tolist())))},
              "mean_check_cost_usd": {n: mean_check_cost(m, gpu_usd_per_hour, windows) for n, m in methods.items()},
              "ranking": ranking, "stopping": stopping, "paired_vs_primary": paired}
    if fixed_steps:
        result["fixed_step_auc"] = {n: fixed_step_auc(rows[n], fixed_steps, cluster_index, multiplicities, cluster) for n in fixed_step_methods}
    if timing_method:
        result["timing"] = dict(stopping_timing(rows[timing_method], taus[timing_method][timing_rate]),
                                method=timing_method, rate=timing_rate)
    return result


def fixed_step_auc(rows: list[dict], ks, cluster_index, multiplicities, cluster="repo") -> list[dict]:
    """Ordinary AUC at fixed checkpoints (Figure 3): one observation per run reaching step k, before any stopping."""
    spend, p, y, run_index = ranking_arrays(rows)
    k_of = np.array([k for r in rows for k in policy.checkpoints(r)])
    ids = np.array([cluster_index[bootstrap.repo_of(r["task"], cluster)] for r in rows], int)
    draws = collections.defaultdict(list)
    for m in multiplicities:
        idx = np.repeat(np.arange(len(run_index)), m[ids][run_index])
        for k in ks:
            sub = idx[k_of[idx] == k]
            draws[k].append(metrics.auc(p[sub], y[sub]))
    out = []
    for k in ks:
        sel = k_of == k
        out.append({"k": k, "n": int(sel.sum()), "failed": int((y[sel] == 0).sum()), "auc": metrics.auc(p[sel], y[sel]),
                    "ci95": bootstrap.ci95(draws[k])})
    return out


def stopping_timing(rows: list[dict], tau: float) -> dict:
    """When correct stops happen: execution cost they avoid and the fraction of each run's cost still ahead (Sec. 5.1)."""
    rep = policy.replay(rows, tau)
    correct = rep["killed"] & ~rep["solved"]
    avoided = rep["full"] - rep["spend"]
    savings = avoided[correct]
    remaining = savings / rep["full"][correct] if correct.any() else np.zeros(0)
    total = float(savings.sum())
    return {"correct_stops": int(correct.sum()), "failed_runs": int((~rep["solved"]).sum()),
            "remaining_fraction_median": float(np.median(remaining)) if len(remaining) else None,
            "avoided_usd": total, "top3_share_of_avoided": float(sum(sorted(savings, reverse=True)[:3]) / total) if total > 0 else None}
