"""Execution and monitoring costs under one frozen list-price schedule (paper Sec. 3.3 and App. C).

Pricing rule used for every reported number ("list" pricing):
    every input token (uncached, cache read, cache write) is charged at the model's input price, and every
    output token, reasoning / thinking included, at its output price.
A cache-aware variant ("cache" pricing: cache reads and writes at their own listed prices) is available for
sensitivity analyses. Dollars are always recomputed from token counts, never taken from what a gateway booked.

Executor spend is needed per tool step: the stopping replay charges a run only up to the checkpoint where it
is stopped. Two sources of per-call token usage exist:
  * the call ledger written by the metering gateway (``stoploss.collect``): one JSON line per LLM call,
    including calls that are not tool steps (e.g. a summary after the last command);
  * the usage Harbor stores in each ATIF trajectory step (``metrics``).
The ledger is authoritative. Ledger calls are aligned to tool steps in order (a call matches the next tool
step when its output and input token counts agree with that step); calls between two tool steps are charged
to the following step, and calls after the last tool step are charged to the run total only. ATIF usage is a
fallback for GPT executors; Claude runs always need the ledger because ATIF omits cache writes.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

LONG_CONTEXT_GUARD = 128_000      # no call above this many input tokens is priced (a long-context tariff would apply)


# ---------------------------------------------------------------------------------------------------- prices
def load_prices(path: str | Path) -> dict:
    """{model: {"input", "cache_read", "cache_write", "output"}} in dollars per million tokens."""
    return json.loads(Path(path).read_text())["usd_per_mtok"]


def price_of(prices: dict, model: str) -> dict:
    if model not in prices:
        raise KeyError(f"no price for model {model!r}; add it to the price file instead of pricing it at $0")
    return prices[model]


# ---------------------------------------------------------------------------------------------------- judge calls
def usage_sum(attempts: list[dict]) -> dict:
    """Summed usage over every attempt for one window (failed and retried attempts are paid for too)."""
    total = collections.Counter()
    for attempt in attempts or []:
        usage = attempt.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            total[key] += int(usage.get(key) or 0)
        total["cached_tokens"] += int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    return dict(total)


def judge_cost_usd(usage: dict, model: str, prices: dict, pricing: str = "list") -> float:
    """Dollar cost of judge usage.

    ``prompt_tokens`` follows each provider's convention: for Claude it is the uncached input (cache reads and
    writes are separate fields); for GPT it is the total input, of which ``cached_tokens`` is a subset.
    """
    p = price_of(prices, model)
    get = lambda key: int(usage.get(key) or 0)
    prompt, write, read = get("prompt_tokens"), get("cache_creation_input_tokens"), get("cache_read_input_tokens")
    out, cached = get("completion_tokens"), get("cached_tokens")
    if pricing == "list":
        return (p["input"] * (prompt + write + read) + p["output"] * out) / 1e6
    if pricing == "cache":
        write_price = p["cache_write"] if p.get("cache_write") is not None else p["input"]
        return (p["input"] * (prompt - cached) + p["cache_read"] * (cached + read) + write_price * write + p["output"] * out) / 1e6
    raise ValueError(f"unknown pricing {pricing!r}")


# ---------------------------------------------------------------------------------------------------- executor runs
def tool_steps(trajectory: str | Path) -> list[dict]:
    """Agent steps of a Harbor ATIF trajectory that executed at least one tool call, oldest first."""
    data = json.loads(Path(trajectory).read_text())
    return [s for s in data.get("steps") or [] if s.get("source") == "agent" and (s.get("tool_calls") or [])]


_LEDGERS: dict[str, dict] = {}


def load_ledger(path: str | Path) -> dict[str, list[dict]]:
    """{run_id: [call, ...]} sorted by timestamp; discarded calls are ignored. Cached per path."""
    key = str(path)
    if key not in _LEDGERS:
        index: dict[str, list[dict]] = {}
        with open(path) as stream:
            for line in stream:
                if '"type": "call"' not in line:
                    continue
                try:
                    call = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if call.get("type") == "call" and not call.get("discarded"):
                    index.setdefault(call.get("run_id"), []).append(call)
        for calls in index.values():
            calls.sort(key=lambda c: c.get("ts", 0))
        _LEDGERS[key] = index
    return _LEDGERS[key]


def align_calls(calls: list[dict], steps: list[dict]):
    """Order-preserving alignment of ledger calls to tool steps.

    A call matches the next unmatched step iff its completion tokens agree with the step's and its input
    accounting agrees (GPT: ledger input total == ATIF prompt tokens; Claude: ledger uncached prompt == ATIF prompt
    tokens and ledger cache reads == ATIF cached tokens). Returns ``(calls_per_step, trailing_calls)``, where each
    step's group also holds the unmatched calls made since the previous step, or ``None`` if not every step matched.
    """
    j, per_step, pending = 0, [], []
    for call in calls:
        if j < len(steps):
            m = steps[j].get("metrics") or {}
            out_ok = int(call.get("completion_tokens") or 0) == int(m.get("completion_tokens") or -1)
            in_ok = (int(call.get("input_tokens_total") or 0) == int(m.get("prompt_tokens") or -1)) or \
                    (int(call.get("prompt_tokens") or 0) == int(m.get("prompt_tokens") or -1)
                     and int(call.get("cache_read_tokens") or 0) == int(m.get("cached_tokens") or 0))
            if out_ok and in_ok:
                per_step.append(pending + [call])
                pending = []
                j += 1
                continue
        pending.append(call)
    return (per_step, pending) if j == len(steps) else None


def _call_tokens(call: dict, who: str) -> np.ndarray:
    """[uncached input, cache read, cache write, output] of one ledger call."""
    u, r, w, o = (int(call.get(k) or 0) for k in ("prompt_tokens", "cache_read_tokens", "cache_write_tokens", "completion_tokens"))
    total = call.get("input_tokens_total")
    if total is not None and int(total) != u + r + w:
        raise ValueError(f"{who}: ledger call {call.get('call_id')}: input_tokens_total {total} != {u} + {r} + {w}")
    if u + r + w > LONG_CONTEXT_GUARD:
        raise ValueError(f"{who}: a call with {u + r + w} input tokens exceeds the short-context pricing guard")
    return np.array([u, r, w, o], float)


def token_curve(trajectory: str | Path, calls: list[dict] | None, model: str | None = None,
                allow_atif_fallback: bool = False, who: str = "") -> dict:
    """Cumulative token usage after every tool step, plus one final row for the whole run.

    Returns ``{"tok": array (n_steps + 1, 4), "n": n_steps, "model": model, "source": "ledger" | "atif"}`` with
    columns [uncached input, cache read, cache write, output]. ``tok[k - 1]`` is the usage up to and including
    tool step k; ``tok[-1]`` additionally includes calls after the last tool step.
    """
    steps = tool_steps(trajectory)
    calls = calls or []
    aligned = align_calls(calls, steps) if calls else None
    cum, rows = np.zeros(4), []
    if aligned is not None:
        per_step, trailing = aligned
        for group in per_step:
            for call in group:
                cum = cum + _call_tokens(call, who)
            rows.append(cum.copy())
        for call in trailing:
            cum = cum + _call_tokens(call, who)
        rows.append(cum.copy())
        model = model or calls[0].get("requested_model") or calls[0].get("model")
        source = "ledger"
    else:
        model = model or ((calls[0].get("requested_model") or calls[0].get("model")) if calls else None)
        if not allow_atif_fallback or str(model).startswith("claude"):
            raise ValueError(f"{who}: ledger calls do not align with the tool steps ({len(calls)} calls / {len(steps)} steps) "
                             "and ATIF usage is not accepted for this run")
        for step in steps:
            m = step.get("metrics") or {}
            prompt, cached = int(m.get("prompt_tokens") or 0), int(m.get("cached_tokens") or 0)
            if prompt > LONG_CONTEXT_GUARD:
                raise ValueError(f"{who}: a step with {prompt} input tokens exceeds the short-context pricing guard")
            cum = cum + np.array([prompt - cached, cached, 0, int(m.get("completion_tokens") or 0)], float)
            rows.append(cum.copy())
        rows.append(cum.copy())       # trailing non-tool calls are unknown without a ledger
        source = "atif"
    return {"tok": np.array(rows), "n": len(steps), "model": model, "source": source}


def price_curve(tc: dict, prices: dict, pricing: str = "list") -> list[float]:
    """Cumulative dollars after every tool step plus the full-run total (same layout as ``token_curve``)."""
    p = price_of(prices, tc["model"])
    t = tc["tok"]
    if pricing == "list":
        return list(((t[:, 0] + t[:, 1] + t[:, 2]) * p["input"] + t[:, 3] * p["output"]) / 1e6)
    if pricing == "cache":
        write_price = p["cache_write"] if p.get("cache_write") is not None else p["input"]
        return list((t[:, 0] * p["input"] + t[:, 1] * p["cache_read"] + t[:, 2] * write_price + t[:, 3] * p["output"]) / 1e6)
    raise ValueError(f"unknown pricing {pricing!r}")
