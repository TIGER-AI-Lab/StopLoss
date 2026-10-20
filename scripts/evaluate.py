#!/usr/bin/env python3
"""Evaluate monitors and judges on one cohort: ranking (tier AUC), calibrated stopping (Delta CPS, monitoring cost,
success retention, failure capture) and repository-cluster bootstrap intervals (paper Sec. 3.3-3.4, Tables 1-2).

The evaluation is described by a JSON spec; see configs/evaluation_example.json. Run it once
on the stopping cohort (calibration-disjoint tasks, used for the cost and stopping columns of Table 1) and once on the
ranking cohort (every run with a verdict, used for tier AUC, Table 2 and Figure 3).

    python scripts/evaluate.py --spec my_eval.json --out results/stopping_cohort.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stoploss import bootstrap, costs, policy                       # noqa: E402
from stoploss.evaluate import evaluate_cell                         # noqa: E402


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_monitor_scores(path) -> dict:
    """{(bench, worker, task, k): p}; the ``set`` field is ignored (a window has one score per model)."""
    out = {}
    for row in read_jsonl(path):
        if row.get("p") is None:
            continue
        key = (row["bench"], row["worker"], row["task"], int(row["k"]))
        p = float(row["p"])
        if not (math.isfinite(p) and 0.0 <= p <= 1.0):
            raise ValueError(f"{path}: invalid score {p} at {key}")
        if key in out and out[key] != p:
            raise ValueError(f"{path}: two different scores for {key}")
        out[key] = p
    return out


def seconds_per_check(spec_entry: dict, score_path: Path) -> float:
    """GPU seconds per check of one model: given in the spec, or read from the scorer's ``.meta.json``."""
    if spec_entry.get("seconds_per_check") is not None:
        return float(spec_entry["seconds_per_check"])
    meta = Path(str(score_path)[: -len(".jsonl")] + ".meta.json") if str(score_path).endswith(".jsonl") else None
    if meta and meta.exists():
        value = json.loads(meta.read_text()).get("seconds_per_window")
        if value:
            return float(value)
    raise ValueError(f"{score_path}: no seconds_per_check in the spec and no seconds_per_window in the score metadata")


def package_config_sha(package: Path, entry: dict) -> str | None:
    """The configuration fingerprint a package was scored under: from the spec, else from its manifest."""
    if entry.get("config_sha"):
        return entry["config_sha"]
    manifest = package / "manifest.json"
    if not manifest.exists():
        return None
    m = json.loads(manifest.read_text())
    return m.get("config_sha") or (m.get("judge_config") or {}).get("config_sha")


def load_judge(name: str, entry: dict, judge_configs: dict, prices: dict, pricing: str):
    """Valid scores and per-window costs of one judge, merged over its work packages. A window may have a valid score
    in one package only; its cost is what that package's ledger recorded for it (failed attempts included)."""
    from stoploss.judges import runner
    judge = judge_configs[entry.get("judge_id", name)]
    scores, window_cost = {}, {}
    for package in entry["packages"]:
        package = Path(package)
        valid, bill = runner.load_package(package, judge, prices, pricing, config_sha=package_config_sha(package, entry))
        for (s, b, w, t, k), p in valid.items():
            key = (b, w, t, k)
            if key in scores:
                raise ValueError(f"judge {name}: window {key} has valid scores in two packages")
            scores[key] = p
            window_cost[key] = bill[(s, b, w, t, k)]
    return scores, window_cost


def by_run(scores: dict) -> dict:
    """{(bench, worker, task): {k: p}}"""
    out = collections.defaultdict(dict)
    for (b, w, t, k), p in scores.items():
        out[(b, w, t)][k] = p
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="evaluation spec (JSON); relative paths are resolved against its folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default=None, help="comma-separated subset of the spec's cells")
    ap.add_argument("--no-bootstrap", action="store_true", help="point estimates only")
    a = ap.parse_args()
    t0 = time.time()
    spec_path = Path(a.spec).resolve()
    spec = json.loads(spec_path.read_text())
    here = spec_path.parent
    resolve = lambda p: (here / p).resolve() if not Path(p).is_absolute() else Path(p)
    prices = costs.load_prices(resolve(spec["prices"]))
    pricing = spec.get("pricing", "list")
    rates = tuple(spec.get("rates", policy.RATES))
    boot = spec.get("bootstrap", {})
    draws, seed, cluster = int(boot.get("draws", 2000)), int(boot.get("seed", 0)), boot.get("cluster", "repo")
    gpu_usd_per_hour = float(spec.get("gpu_usd_per_hour", 0.8))
    groups = spec.get("groups", {})
    primary = spec.get("primary")
    cells = a.cells.split(",") if a.cells else list(spec["cells"])

    # ---- runs of the cohort, with per-step execution spend
    ledgers = {}
    runs_of = {}
    for cell in cells:
        runs = []
        for rec in read_jsonl(resolve(spec["cells"][cell]["runs"])):
            if rec.get("reward") not in (0.0, 1.0):
                continue                                             # runs without a verdict are excluded
            calls = None
            if rec.get("ledger"):
                path = str(resolve(rec["ledger"]))
                ledgers.setdefault(path, costs.load_ledger(path))
                calls = ledgers[path].get(rec["run_id"]) or []
            tc = costs.token_curve(resolve(rec["trajectory"]), calls, rec.get("model"),
                                   allow_atif_fallback=bool(spec["cells"][cell].get("allow_atif_fallback", False)), who=rec["task"])
            runs.append({"task": rec["task"], "bench": rec["bench"], "worker": rec["worker"], "reward": float(rec["reward"]),
                         "spend": costs.price_curve(tc, prices, pricing), "n": tc["n"]})
        runs_of[cell] = sorted(runs, key=lambda r: r["task"])
        print(f"[{cell}] {len(runs)} runs with a verdict", flush=True)

    # ---- scores: small monitors (one file per model) and judges (one or more work packages each)
    monitors = {}
    for name, entry in spec.get("monitors", {}).items():
        path = resolve(entry["scores"])
        monitors[name] = {"scores": by_run(load_monitor_scores(path)), "seconds_per_check": seconds_per_check(entry, path)}
    default_judges = Path(__file__).resolve().parents[1] / "configs/judges.json"
    judge_configs = {j["judge_id"]: j for j in json.loads(resolve(spec.get("judge_configs", default_judges)).read_text())["judges"]} \
        if spec.get("judges") else {}
    judges = {}
    for name, entry in spec.get("judges", {}).items():
        entry = dict(entry, packages=[resolve(p) for p in entry["packages"]])
        scores, window_cost = load_judge(name, entry, judge_configs, prices, pricing)
        judges[name] = {"scores": by_run(scores), "window_cost": window_cost}

    # ---- bootstrap draws: one matrix per benchmark, shared by its cells and by every method
    bench_tasks = collections.defaultdict(set)
    for cell in cells:
        for r in runs_of[cell]:
            bench_tasks[r["bench"]].add(r["task"])
    draws_of = {b: bootstrap.bench_multiplicities(tasks, cluster, draws, seed, b) for b, tasks in bench_tasks.items()}

    result = {"spec": str(spec_path), "pricing": pricing, "price_file": str(resolve(spec["prices"])), "rates": list(rates),
              "bootstrap": {"draws": draws, "seed": seed, "cluster": cluster}, "gpu_usd_per_hour": gpu_usd_per_hour,
              "groups": groups, "primary": primary + "_ens" if primary else None, "cells": {}}
    primary_draws = {}
    for cell in cells:
        cfg = spec["cells"][cell]
        runs = runs_of[cell]
        benches = {r["bench"] for r in runs}
        workers = {r["worker"] for r in runs}
        if len(benches) != 1 or len(workers) != 1:
            raise ValueError(f"{cell}: a cell must hold one benchmark and one executor, got {benches} x {workers}")
        bench, worker = benches.pop(), workers.pop()
        cal_runs = [r for r in read_jsonl(resolve(spec["calibration"][cfg["calibration"]])) if r.get("reward") == 1.0]
        names = list(monitors) + [j for j in cfg.get("judges", [])]
        methods, calibration, cal_keys = {}, {}, {}
        for n in names:
            src = monitors[n] if n in monitors else judges[n]
            scores = {t: ks for (b, w, t), ks in src["scores"].items() if (b, w) == (bench, worker)}
            method = {"scores": scores}
            if "window_cost" in src:
                method["window_cost"] = {(t, k): c for (b, w, t, k), c in src["window_cost"].items() if (b, w) == (bench, worker)}
            else:
                method["seconds_per_check"] = src["seconds_per_check"]
            key = n if n in monitors else f"judge:{n}"
            methods[key] = method
            calibration[key] = [{"task": r["task"], "reward": 1.0,
                                 "scores": {k: p for k, p in src["scores"].get((r["bench"], r["worker"], r["task"]), {}).items() if policy.is_checkpoint(k)}}
                                for r in cal_runs]
            cal_keys[key] = sorted((r["bench"], r["worker"], r["task"], k) for r, c in zip(cal_runs, calibration[key]) for k in c["scores"])
        reference = next(iter(cal_keys.values()))
        for key, ks in cal_keys.items():
            if ks != reference:
                raise ValueError(f"{cell}: method {key} is calibrated on different windows than the other methods")
        cluster_index, multiplicities = draws_of[bench]
        extras = cfg.get("extras", {})
        print(f"== {cell}: {len(runs)} runs, {len(methods)} methods, calibration set {cfg['calibration']} ({len(cal_runs)} successful runs)", flush=True)
        out = evaluate_cell(runs, methods, calibration, cluster_index, multiplicities, groups=groups, primary=primary, rates=rates,
                            cluster=cluster, gpu_usd_per_hour=gpu_usd_per_hour, bootstrap_draws=not a.no_bootstrap,
                            fixed_steps=extras.get("fixed_steps"), fixed_step_methods=extras.get("fixed_step_methods", ()),
                            timing_method=extras.get("timing_method"), timing_rate=extras.get("timing_rate", 0.05))
        out["benchmark"], out["worker"], out["calibration_set"] = bench, worker, cfg["calibration"]
        for n, d in out["paired_vs_primary"].items():
            if n.startswith("judge:"):
                primary_draws[f"{cell}|{n}"] = {q: (v["point"], v["draws"]) for q, v in d.items() if q in ("tier_auc", "net_delta_pct@0.05")}
            for v in d.values():
                v.pop("draws", None)                                 # keep the output small
        result["cells"][cell] = out
        shown = lambda n: n.endswith(("_ens", "_base")) or n.startswith("judge:") or n == "stepcount"
        print("   tier AUC:      " + "  ".join(f"{n} {v['tier_auc']:.3f}" for n, v in out["ranking"]["methods"].items() if shown(n)), flush=True)
        print("   Delta CPS @5%: " + "  ".join(f"{n} {v['0.05']['net_delta_pct']:+.1f}" for n, v in out["stopping"]["methods"].items() if shown(n)), flush=True)
    if primary_draws:
        # every judge vs the primary ensemble in every setting, as one family: simultaneous max-|t| intervals
        result["primary_comparisons"] = {}
        for q, meaning in (("tier_auc", "judge minus primary; negative = the primary ensemble ranks better"),
                           ("net_delta_pct@0.05", "judge minus primary at the nominal 5% rate; positive = the primary ensemble is cheaper per retained solve")):
            pts = {n: d[q][0] for n, d in primary_draws.items() if q in d}
            sim, crit = bootstrap.simultaneous_intervals(pts, {n: d[q][1] for n, d in primary_draws.items() if q in d})
            result["primary_comparisons"][q] = {"direction": meaning, "critical_value_max_t": crit,
                                                "comparisons": {n: {"point": pts[n], "ci95_simultaneous": sim.get(n)} for n in pts}}
    result["elapsed_s"] = round(time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, default=float))
    print(f"wrote {a.out} ({result['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
