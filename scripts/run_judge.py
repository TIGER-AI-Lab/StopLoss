#!/usr/bin/env python3
"""Score monitor windows with one paid LLM judge (paper Sec. 3.2 and App. A).

The judge reads the same windows as the small monitor, at the policy checkpoints k = 5, 8, 11, ... (every k with
--all-windows), and only windows of its own executor family. Results go to a package directory:
scores.jsonl (one row per window), calls.jsonl (one line per request: the bill) and manifest.json. Re-running the
same command resumes: windows that already have a valid score are skipped.

API keys come from ANTHROPIC_API_KEY (Claude judges) or OPENAI_API_KEY (GPT judges); --base-url, or
ANTHROPIC_BASE_URL / OPENAI_BASE_URL, points at another endpoint.

Exit codes: 0 complete, 1 some windows without a valid score (run again), 3 stopped (credentials, exhausted
account, repeated failures or an unexpected error), 4 stopped (the API answered with a model outside the judge's
allow-list).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))       # allows running from a checkout without installing the package

EXAMPLES = """examples:
  python scripts/run_judge.py --judge opus5 --windows data/windows/test.jsonl --out-dir out/judges/opus5 --dry-run \\
      --tokens-per-window 1800 1500
  python scripts/run_judge.py --judge opus5 --windows data/windows/test.jsonl --out-dir out/judges/opus5 --threads 4
  python scripts/run_judge.py --judge sol --windows data/windows/val_calibration.jsonl --set val --out-dir out/judges/sol
"""


def read_windows(path: str) -> list[dict]:
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=EXAMPLES,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", required=True,
                        help="judge_id in the judge config (sol, terra, luna, opus5, opus46, sonnet46)")
    parser.add_argument("--windows", required=True,
                        help="windows JSONL: bench, worker, task, k, prompt, label (+ optional set)")
    parser.add_argument("--out-dir", required=True, help="package directory of this judge; reuse it to resume")
    parser.add_argument("--set", dest="set_name", default="test",
                        help="set label for windows without a 'set' field (default: test)")
    parser.add_argument("--threads", type=int, default=2, help="concurrent requests (default: 2)")
    parser.add_argument("--all-windows", action="store_true", help="score every k instead of the checkpoints 5, 8, 11, ...")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N planned windows (e.g. to meter the cost)")
    parser.add_argument("--base-url", default=None, help="API base URL (default: $ANTHROPIC_BASE_URL / $OPENAI_BASE_URL, "
                                                         "else the official endpoint)")
    parser.add_argument("--timeout", type=float, default=600.0, help="socket timeout per request in seconds (default: 600)")
    parser.add_argument("--max-failed-streak", type=int, default=3,
                        help="stop after this many consecutive windows without a valid score (default: 3)")
    parser.add_argument("--config", default=str(REPO / "configs" / "judges.json"), help="judge configuration file")
    parser.add_argument("--prices", default=str(REPO / "configs" / "prices_20260918.json"), help="price file (list prices)")
    parser.add_argument("--dry-run", action="store_true",
                        help="make no request; print the plan (and an estimate with --tokens-per-window)")
    parser.add_argument("--tokens-per-window", nargs=2, type=int, metavar=("INPUT", "OUTPUT"),
                        help="dry run: assumed input and output (incl. reasoning) tokens per window, "
                             "for a rough cost estimate")
    args = parser.parse_args(argv)

    from stoploss.costs import judge_cost_usd, load_prices
    from stoploss.judges import load_judges, run_judge, window_costs
    from stoploss.judges.clients import messages_body, responses_body

    judges = load_judges(args.config)
    if args.judge not in judges:
        parser.error(f"unknown judge {args.judge!r}; known: {sorted(judges)}")
    judge = judges[args.judge]
    prices = load_prices(args.prices)
    summary = run_judge(read_windows(args.windows), judge, args.out_dir, base_url=args.base_url, set_name=args.set_name,
                        all_windows=args.all_windows, limit=args.limit or None, threads=args.threads,
                        timeout=args.timeout, max_failed_streak=args.max_failed_streak, dry_run=args.dry_run,
                        windows_source=args.windows, log=lambda *a: print(*a, flush=True))

    if args.dry_run:
        body = messages_body(judge, "<judge prompt>", judge.get("max_tokens")) if judge["api"] == "messages" \
            else responses_body(judge, "<judge prompt>")
        print(f"[{args.judge}] DRY RUN, no request made. Request body: {json.dumps(body)}")
        price = prices.get(judge["model"])
        if price:
            print(f"[{args.judge}] list prices: ${price['input']}/M input tokens, ${price['output']}/M output tokens "
                  "(every input token at the input price; reasoning / thinking is output)")
        if args.tokens_per_window and price:
            n_in, n_out = args.tokens_per_window
            per_window = judge_cost_usd({"prompt_tokens": n_in, "completion_tokens": n_out}, judge["model"], prices)
            print(f"[{args.judge}] rough estimate: {summary['n_to_score']} windows x ${per_window:.4f} = "
                  f"${summary['n_to_score'] * per_window:.2f} (first requests only; truncation retries cost more)")
        elif not args.tokens_per_window:
            print(f"[{args.judge}] no cost estimate: pass --tokens-per-window INPUT OUTPUT, "
                  "or meter a few windows with --limit")
        return 0

    calls_path = Path(args.out_dir) / "calls.jsonl"
    if calls_path.exists() and judge["model"] in prices:
        bill = window_costs(calls_path, judge, prices)
        summary.update(package_usd_list=round(bill.total_usd, 4), package_requests=bill.n_calls,
                       package_requests_unknown_usage=bill.n_unknown_usage)
    print("SUMMARY " + json.dumps(summary), flush=True)
    if summary.get("stop_message"):
        print(f"[{args.judge}] stopped ({summary['stopped']}): {summary['stop_message']}", flush=True)
    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
