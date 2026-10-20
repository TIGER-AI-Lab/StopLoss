#!/usr/bin/env python3
"""Score windows with a trained monitor (or an untuned base model) and record the GPU time per check.

    python scripts/score_windows.py --base Qwen/Qwen2.5-7B-Instruct --adapter runs/qwen2.5-7b-instruct_s0/best \
        --windows test.jsonl calibration.jsonl --out scores/qwen2.5-7b-instruct_s0.jsonl
    python scripts/score_windows.py --base Qwen/Qwen2.5-7B-Instruct --windows test.jsonl --out scores/qwen2.5-7b-instruct_base.jsonl

Writes one line per window ({"set", "bench", "worker", "task", "k", "p", "prompt_sha256"}) and <out>.meta.json with
seconds_per_window, which scripts/evaluate.py turns into the monitoring cost. Use --batch-size 1 (the default) when
the timing is used as the cost proxy; larger batches are faster and are checked against batch-1 scores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stoploss.score import load_monitor, score_windows, write_scores       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base model (hub id or local path)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory; omit for the untuned base")
    ap.add_argument("--windows", nargs="+", required=True, help="window JSONL files (from scripts/build_windows.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    rows = [json.loads(line) for path in a.windows for line in open(path) if line.strip()]
    keys = [(r.get("set", "test"), r["bench"], r["worker"], r["task"], r["k"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate windows in the input files")
    print(f"scoring {len(rows)} windows with {a.base}" + (f" + {a.adapter}" if a.adapter else " (untuned)"), flush=True)
    tok, model = load_monitor(a.base, a.adapter, a.device)
    p, meta = score_windows(tok, model, rows, a.max_len, a.batch_size)
    meta.update({"base": a.base, "adapter": a.adapter, "windows": a.windows})
    write_scores(a.out, rows, p, meta)
    print(f"wrote {len(rows)} scores -> {a.out}; {meta['seconds_per_window']:.4f} s per window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
