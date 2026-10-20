"""Append-only JSONL call ledger written by the metering gateway (paper Sec. 3.3, App. C).

One line per LLM call::

    {"type": "call", "call_id", "run_id", "task_id", "worker", "model", "requested_model", "backend_model",
     "prompt_tokens", "cache_read_tokens", "cache_write_tokens", "input_tokens_total", "completion_tokens",
     "reasoning_tokens", "finish_reason", "cost_usd", "latency_ms", "ts", "discarded": false}

Token fields are provider-neutral: ``prompt_tokens`` is the UNCACHED input only, cache reads and writes are
separate, ``input_tokens_total`` is their sum (the context the model actually processed) and
``completion_tokens`` includes reasoning / thinking tokens. ``stoploss.costs.load_ledger`` reads these lines
(it relies on the default ``json.dumps`` spacing, ``"type": "call"``). ``cost_usd`` is what the gateway booked
for its spend valve; reported costs are always recomputed from the token counts.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class Ledger:
    """Thread-safe append-only call ledger with per-run totals (rebuilt from the file on start-up)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._load_totals()

    def _load_totals(self) -> None:
        # Survive a gateway restart: the spend valve and the run totals continue from what is on disk.
        with self.path.open() as stream:
            for line in stream:
                if '"type": "call"' not in line:
                    continue
                try:
                    call = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if call.get("type") == "call" and not call.get("discarded"):
                    self._add_to_totals(call)

    def _add_to_totals(self, call: dict[str, Any]) -> None:
        run = self._runs.setdefault(call.get("run_id"), {
            "run_id": call.get("run_id"), "task_id": call.get("task_id"), "worker": call.get("worker"),
            "n_calls": 0, "cost_usd": 0.0})
        run["n_calls"] += 1
        run["cost_usd"] += float(call.get("cost_usd") or 0.0)

    def record_call(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Append one call; ``type``, ``ts`` and ``discarded`` are added here. Returns the written entry."""
        entry = dict(entry, type="call", ts=time.time(), discarded=False)
        line = json.dumps(entry) + "\n"
        with self._lock:
            with self.path.open("a") as stream:
                stream.write(line)
            self._add_to_totals(entry)
        return entry

    def run_totals(self, run_id: str) -> dict[str, Any] | None:
        """{"run_id", "task_id", "worker", "n_calls", "cost_usd"} for one run, or None if it made no call."""
        with self._lock:
            run = self._runs.get(run_id)
            return dict(run) if run is not None else None

    def spent_usd(self, run_id: str) -> float:
        with self._lock:
            run = self._runs.get(run_id)
            return run["cost_usd"] if run is not None else 0.0
