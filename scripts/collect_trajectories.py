#!/usr/bin/env python3
"""Collect executor trajectories: one stock Harbor Terminus-2 trial per task through the metering gateway.

Start the gateway first (it holds the provider API keys), then run this script for one (benchmark, executor) arm:

    python -m stoploss.collect.gateway --ledger runs/ledger.jsonl &
    python scripts/collect_trajectories.py --worker luna --bench verified \
        --dataset-dir /path/to/harbor/datasets/swebench-verified --out runs --concurrency 8

Re-running the same command resumes the arm.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stoploss.collect import DEFAULT_EXECUTORS, load_executors  # noqa: E402
from stoploss.collect import harbor_runner  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", required=True, help="executor label from the executor configuration")
    parser.add_argument("--bench", required=True, help="benchmark label written into the run records")
    parser.add_argument("--dataset-dir", required=True,
                        help="Harbor dataset directory (one sub-directory with a task.toml per task)")
    parser.add_argument("--out", required=True, help="output root; records go to <out>/<bench>/<worker>/")
    parser.add_argument("--gateway", default="http://127.0.0.1:8765", help="URL of the metering gateway")
    parser.add_argument("--executors", default=str(DEFAULT_EXECUTORS), help="executor configuration (JSON)")
    parser.add_argument("--tasks-file", default=None,
                        help="collect only these tasks (JSON list or one name per line)")
    parser.add_argument("--n", type=int, default=None, help="deterministic subset of this many tasks")
    parser.add_argument("--seed", type=int, default=0, help="seed of the --n subset")
    parser.add_argument("--concurrency", type=int, default=4, help="trials running at the same time")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="override the executor's turn budget (written to a separate <worker>_t<N> arm)")
    parser.add_argument("--harbor-cmd", default="harbor",
                        help="command that runs the Harbor CLI, e.g. 'harbor' or 'uv run harbor'")
    parser.add_argument("--harbor-dir", default=None, help="working directory for the Harbor command")
    parser.add_argument("--jobs-dir", default=None, help="Harbor jobs directory (default: <out>/jobs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the Harbor job configuration of the first task and exit")
    args = parser.parse_args(argv)

    executors = load_executors(args.executors)
    if args.worker not in executors:
        parser.error(f"unknown worker {args.worker!r}; known: {sorted(executors)}")
    executor = executors[args.worker]
    tasks = harbor_runner.select_tasks(args.dataset_dir, args.bench, args.tasks_file, args.n, args.seed)
    if args.dry_run:
        import json

        config = harbor_runner.harbor_job_config(
            executor, task=tasks[0], dataset_dir=Path(args.dataset_dir).resolve(), job_name="<job>",
            jobs_dir=Path(args.jobs_dir or Path(args.out) / "jobs").resolve(),
            api_base=harbor_runner.GatewayClient(args.gateway).api_base,
            max_turns=args.max_turns or executor.max_turns)
        print(f"{len(tasks)} tasks; Harbor job configuration of the first one ({tasks[0]}), run as "
              f"'{args.harbor_cmd} run -c <config> -y' with OPENAI_API_KEY='<run_id>|<task>':")
        print(json.dumps(config, indent=2))
        return
    harbor_runner.collect(
        executor, bench=args.bench, dataset_dir=args.dataset_dir, tasks=tasks, out_dir=args.out,
        gateway=harbor_runner.GatewayClient(args.gateway), harbor_cmd=args.harbor_cmd, harbor_dir=args.harbor_dir,
        jobs_dir=args.jobs_dir, concurrency=args.concurrency, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
