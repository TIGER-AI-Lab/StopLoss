#!/usr/bin/env python3
"""Turn completed runs into labelled monitor windows (paper Sec. 2.1, App. A).

Input: run records (JSONL, one per run) with at least task, bench, worker, reward and the
path of the Harbor ATIF trajectory. Runs without a verdict and runs with fewer than two tool steps are skipped.
Output: one window per (run, k) with the fitted prompt and its outcome label.

    python scripts/build_windows.py --runs runs.jsonl --out windows.jsonl --ks all          # evaluation sets
    python scripts/build_windows.py --runs train_runs.jsonl --out train.jsonl --ks grid     # training corpus grid
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stoploss import policy, windows                                  # noqa: E402

_TOK = None


def _init(model: str) -> None:
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(model)


def _windows_of(job):
    record, trajectory, ks, max_len, set_label = job
    out = []
    for w in windows.run_windows(trajectory, float(record["reward"]), _TOK, max_len, ks):
        out.append({"set": set_label, "bench": record["bench"], "worker": record["worker"], "task": record["task"], **w})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="run records (JSONL); relative trajectory paths are resolved against this file's folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="tokenizer that defines the token budget")
    ap.add_argument("--max-len", type=int, default=1600)
    ap.add_argument("--ks", default="all", choices=["all", "grid", "checkpoints"],
                    help="all: every k >= 2; grid: the training-corpus grid; checkpoints: k = 5, 8, 11, ... only")
    ap.add_argument("--set", default="test", help="value of the 'set' field written to every window (e.g. train, val, test)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    runs_path = Path(a.runs).resolve()
    records = [json.loads(line) for line in open(runs_path) if line.strip()]
    records = [r for r in records if r.get("reward") in (0.0, 1.0)]
    ks = {"all": None, "grid": windows.K_GRID, "checkpoints": range(policy.FIRST_CHECK, 10_000, policy.CHECK_EVERY)}[a.ks]
    jobs = []
    for r in records:
        trajectory = Path(r["trajectory"])
        trajectory = trajectory if trajectory.is_absolute() else runs_path.parent / trajectory
        jobs.append((r, trajectory, ks, a.max_len, a.set))
    n_rows, n_runs, counts = 0, 0, {}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers, initializer=_init, initargs=(a.model,)) as pool, open(a.out, "w") as stream:
        for rows in pool.imap(_windows_of, jobs, chunksize=4):        # imap keeps the input order
            n_runs += bool(rows)
            for row in rows:
                stream.write(json.dumps(row) + "\n")
                n_rows += 1
                counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"{len(records)} runs with a verdict -> {n_runs} runs with >= 2 tool steps -> {n_rows} windows {counts} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
