"""Scoring windows with one paid LLM judge, and reading the results back (paper Sec. 3.2, App. A and App. C).

A judge configuration is one entry of ``configs/judges.json``. ``run_judge`` scores a list of windows with it:
  * only the policy checkpoints k = 5, 8, 11, ... (unless every window is requested);
  * only windows whose executor (the window's ``worker``) is in the entry's ``workers``: the paper's same-family
    protocol (GPT judges read GPT-executor runs, Claude judges read Claude-executor runs);
  * the text sent is ``judge_prompt(window["prompt"])``.

Retry ladder of one window (``score_window``), at most six requests in total:
  * normal termination (finish = stop) with a parsed answer -> score p = answer / 100;
  * truncated answer (finish = length) -> never accepted, even with a number in it: Claude judges climb the
    max_tokens ladder 4000 -> 8000 -> 16000; GPT judges (no output cap) are retried up to ``length_retries`` times;
  * any other finish or an unparsed answer -> one retry, and only if it was the window's first request;
  * a failed request -> retried after max(4 s x request number, Retry-After);
  * rejected credentials, an exhausted account, or a backend model outside ``allowed_backend_models`` -> the run stops;
  * ``max_failed_streak`` windows in a row without a valid score -> the run stops instead of burning requests.

Package directory (one per judge configuration; re-running the same command resumes it):
  manifest.json  the judge configuration, its config_sha, one entry per invocation
  scores.jsonl   one row per scored window; failed rows stay as records and the window is scored again next time
  calls.jsonl    the call ledger: one line per request, written when the request returns or fails
The bill comes from calls.jsonl (every request is paid, failed and truncated ones included), never from the score rows.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

from ..costs import judge_cost_usd, usage_sum
from ..policy import is_checkpoint
from .clients import FatalAPIError, make_client
from .prompt import JUDGE_SUFFIX, judge_prompt, parse_answer, sha16

MAX_REQUESTS = 6                              # per window, all retry reasons together
ERROR_BACKOFF_S = 4.0                         # after the n-th request failed: wait n * 4 s (or Retry-After, if longer)
MAX_RETRY_AFTER_S = 900.0
RAW_CHARS = 4000                              # judge text kept in a score row
ERROR_CHARS = 300

# The part of a judge entry that shapes the request and the retry ladder. Its fingerprint (``judge_config_sha``,
# the ``config_sha`` field) is stored in every row and every ledger line. The acceptance lists
# (allowed_backend_models, workers) are applied when reading, so extending them never invalidates a package.
CONFIG_KEYS = ("judge_id", "model", "family", "api", "thinking", "output_config", "reasoning",
               "max_tokens", "max_tokens_ladder", "length_retries")
REQUIRED_KEYS = ("judge_id", "model", "family", "api", "allowed_backend_models", "workers")
LEDGER_FIELDS = ("request_sha256", "max_tokens", "finish", "usage", "usage_unknown", "backend_model", "latency_s",
                 "error", "parse", "answer")
STOP_ERRORS = ("stopped", "auth", "quota", "backend_model")     # failures that do not count towards the failure streak
EXIT_COMPLETE, EXIT_INCOMPLETE, EXIT_STOPPED, EXIT_BACKEND_MODEL = 0, 1, 3, 4

Key = tuple   # (set, bench, worker, task, k)


# ---------------------------------------------------------------------------------------------------- configuration
def load_judges(path: str | Path) -> dict[str, dict]:
    """{judge_id: judge entry} of a judge configuration file (configs/judges.json)."""
    judges = {}
    for entry in json.loads(Path(path).read_text())["judges"]:
        missing = [k for k in REQUIRED_KEYS if not entry.get(k)]
        if missing:
            raise ValueError(f"{path}: judge {entry.get('judge_id')!r} lacks {missing}")
        if entry["api"] not in ("messages", "responses"):
            raise ValueError(f"{path}: judge {entry['judge_id']!r}: unknown api {entry['api']!r}")
        if entry["api"] == "messages" and not (entry.get("max_tokens") or entry.get("max_tokens_ladder")):
            raise ValueError(f"{path}: judge {entry['judge_id']!r}: the Messages API needs max_tokens")
        if entry["judge_id"] in judges:
            raise ValueError(f"{path}: duplicate judge_id {entry['judge_id']!r}")
        judges[entry["judge_id"]] = entry
    return judges


def judge_config(judge: dict) -> dict:
    """The configuration fingerprinted by ``judge_config_sha`` (absent keys are None)."""
    config = {k: judge.get(k) for k in CONFIG_KEYS}
    config["judge_suffix_sha256"] = sha16(JUDGE_SUFFIX)
    return config


def judge_config_sha(judge: dict) -> str:
    """First 16 hex digits of sha256 over the sorted-key JSON of ``judge_config(judge)``."""
    return hashlib.sha256(json.dumps(judge_config(judge), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def window_key(row: dict) -> Key:
    return (row.get("set"), row.get("bench"), row.get("worker"), row.get("task"), int(row.get("k")))


# ---------------------------------------------------------------------------------------------------- validity rule
def row_problems(row: dict, judge: dict, config_sha: str | None = None, prompt_sha: str | None = None,
                 judge_prompt_sha: str | None = None) -> list[str]:
    """The one validity rule for a judge score row; [] means valid.

    A valid row belongs to this judge and model, reads a window of an allowed executor (same family), terminated
    normally (finish = stop), names an allowed backend model, has p in [0, 1] and has recorded usage. When given,
    the configuration and both prompt hashes must match as well.
    """
    problems = []
    p = row.get("p")
    usage = row.get("usage_total") or {}
    if row.get("judge_id") != judge["judge_id"]:
        problems.append(f"judge_id {row.get('judge_id')!r}")
    if row.get("model") != judge["model"]:
        problems.append(f"model {row.get('model')!r}")
    if row.get("worker") not in judge["workers"]:
        problems.append(f"cross-family worker {row.get('worker')!r}")
    if config_sha is not None and row.get("config_sha") != config_sha:
        problems.append(f"config_sha {row.get('config_sha')!r}")
    if prompt_sha is not None and row.get("prompt_sha256") != prompt_sha:
        problems.append("prompt_sha256")
    if judge_prompt_sha is not None and row.get("judge_prompt_sha256") != judge_prompt_sha:
        problems.append("judge_prompt_sha256")
    if row.get("finish") != "stop":
        problems.append(f"finish {row.get('finish')!r}")
    if row.get("backend_model") not in judge["allowed_backend_models"]:
        problems.append(f"backend_model {row.get('backend_model')!r}")
    if not (isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0.0 <= p <= 1.0):
        problems.append(f"p {p!r}")
    if not (int(usage.get("prompt_tokens") or 0) > 0 and int(usage.get("completion_tokens") or 0) > 0):
        problems.append("usage missing")
    return problems


# ---------------------------------------------------------------------------------------------------- one window
class StopSignal:
    """Shared stop flag of a run; the first reason wins."""

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self.kind: str | None = None
        self.message: str | None = None

    def set(self, kind: str, message: str = "") -> None:
        with self._lock:
            if self.kind is None:
                self.kind, self.message = kind, message
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


def score_window(window: dict, judge: dict, client, *, book=None, stop: StopSignal | None = None, sleep=time.sleep):
    """Ask ``judge`` about one window. Returns (final, attempts, error).

    ``client(prompt, max_tokens) -> result`` (see clients.py); ``book(attempt) -> attempt`` records every request in
    the ledger. ``final`` is (answer, parse mode, result) or None; ``attempts`` lists every request made, in order.
    """
    book = book or (lambda attempt: attempt)
    stop = stop or StopSignal()
    prompt = judge_prompt(window["prompt"])
    ladder = list(judge.get("max_tokens_ladder") or [judge.get("max_tokens")])
    attempts, step, n_length, error, final = [], 0, 0, None, None
    for attempt in range(MAX_REQUESTS):
        if stop.is_set():
            return None, attempts, "stopped"
        max_tokens = ladder[min(step, len(ladder) - 1)]
        try:
            result = client(prompt, max_tokens)
        except FatalAPIError as e:
            attempts.append(book({"max_tokens": max_tokens, "error": e.kind, "usage_unknown": True}))
            stop.set(e.kind, str(e)[:ERROR_CHARS])
            return None, attempts, e.kind
        except Exception as e:    # request, network or stream failure: usage unknown, retried
            error = str(e)[:ERROR_CHARS]
            attempts.append(book({"max_tokens": max_tokens, "error": error, "usage_unknown": True}))
            if attempt < MAX_REQUESTS - 1:
                wait = max(ERROR_BACKOFF_S * (attempt + 1), getattr(e, "retry_after", None) or 0.0)
                sleep(min(wait, MAX_RETRY_AFTER_S))
            continue
        answer, mode = parse_answer(result["content"])
        attempts.append(book({"max_tokens": max_tokens, "finish": result["finish"], "usage": result["usage"],
                              "latency_s": result["latency_s"], "parse": mode, "answer": answer,
                              "backend_model": result["backend_model"], "thinking_blocks": result["thinking_blocks"],
                              "reasoning_tokens": result["reasoning_tokens"], "backend_usage": result["backend_usage"],
                              "request_sha256": result["request_sha256"]}))
        if result["backend_model"] not in judge["allowed_backend_models"]:
            stop.set("backend_model", f"the API answered with model {result['backend_model']!r}, which is not in the "
                                      f"allowed_backend_models {judge['allowed_backend_models']} "
                                      f"of judge {judge['judge_id']!r}")
            return None, attempts, "backend_model"
        if result["finish"] == "stop" and answer is not None:     # normal termination first, then the parsed score
            final = (answer, mode, result)
            break
        if result["finish"] == "length":                          # truncated: never accepted
            n_length += 1
            error = "truncated (finish=length)"
            if step < len(ladder) - 1:
                step += 1
                continue
            if judge.get("max_tokens") is None and n_length <= int(judge.get("length_retries") or 0):
                continue
            break
        error = f"finish={result['finish']} parse={mode}"
        if attempt >= 1:
            break
    return final, attempts, error


def score_row(window: dict, judge: dict, cfg_sha: str, final, attempts: list[dict], error) -> dict:
    """The scores.jsonl row of one window. A row is written as scored only if it passes ``row_problems``."""
    row = {"set": window["set"], "bench": window["bench"], "worker": window["worker"], "task": window["task"],
           "k": int(window["k"]), "label": window.get("label"), "judge_id": judge["judge_id"], "model": judge["model"],
           "config_sha": cfg_sha, "attempts": attempts, "usage_total": usage_sum(attempts),
           "usage_unknown_attempts": sum(1 for a in attempts if a.get("usage_unknown")), "n_attempts": len(attempts),
           "prompt_sha256": sha16(window["prompt"]), "judge_prompt_sha256": sha16(judge_prompt(window["prompt"]))}
    if final is None:
        row.update(p=None, answer=None, finish=attempts[-1].get("finish") if attempts else None, error=error)
        return row
    answer, mode, result = final
    row.update(p=answer / 100.0, answer=answer, parse=mode, raw=(result["content"] or "")[:RAW_CHARS],
               finish=result["finish"], usage=result["usage"], latency_s=result["latency_s"],
               backend_model=result["backend_model"], thinking_blocks=result["thinking_blocks"],
               reasoning_tokens=result["reasoning_tokens"])
    problems = row_problems(row, judge)
    if problems:     # e.g. a response without usage: kept as a record, the window is scored again on the next run
        row.update(p=None, answer=None, error="not a valid result: " + "; ".join(problems))
    return row


# ---------------------------------------------------------------------------------------------------- reading packages
def _read_lines(path: Path) -> tuple[list[str], bool]:
    """(lines, last line lacks its newline)."""
    if not path.exists():
        return [], False
    text = path.read_bytes().decode("utf-8", errors="replace")
    return text.split("\n"), bool(text) and not text.endswith("\n")


def read_scores(scores_path: str | Path, judge: dict, *, config_sha: str | None = None,
                prompt_sha: dict | None = None, strict: bool = True) -> dict:
    """Read a scores.jsonl of one judge configuration.

    Returns {"valid": {key: p}, "rows": {key: valid row}, "failed_rows": n, "call_ids": set of ledger ids referenced,
    "problems": [(line number, reason)]}. A problem is an unparseable line, a row of another judge or configuration,
    or a scored row that breaks the validity rule. ``strict`` raises on any problem (and on a missing file).
    ``config_sha`` defaults to ``judge_config_sha(judge)``; ``prompt_sha`` ({key: prompt_sha256}) additionally checks
    which window was read.
    """
    if strict and not Path(scores_path).exists():
        raise FileNotFoundError(scores_path)
    expected_sha = config_sha or judge_config_sha(judge)
    out = {"valid": {}, "rows": {}, "failed_rows": 0, "call_ids": set(), "problems": []}
    lines, _ = _read_lines(Path(scores_path))
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = window_key(row)
        except (ValueError, TypeError, AttributeError):
            out["problems"].append((number, "unparseable line"))
            continue
        if row.get("judge_id") != judge["judge_id"] or row.get("config_sha") != expected_sha:
            out["problems"].append((number, f"row of judge {row.get('judge_id')!r} / config {row.get('config_sha')!r} "
                                            f"in a package of {judge['judge_id']!r} / {expected_sha!r}"))
            continue
        if row.get("p") is not None:
            problems = row_problems(row, judge, expected_sha, (prompt_sha or {}).get(key))
            if problems:
                out["problems"].append((number, "scored row is not a valid result: " + "; ".join(problems)))
                continue
        out["call_ids"].update(a["call_id"] for a in row.get("attempts") or [] if a.get("call_id"))
        if row.get("p") is None:
            out["failed_rows"] += 1
        else:
            out["valid"][key] = float(row["p"])
            out["rows"][key] = row
    if strict and out["problems"]:
        number, reason = out["problems"][0]
        raise ValueError(f"{scores_path}: {len(out['problems'])} line(s) cannot be used, e.g. line {number}: {reason}")
    return out


def valid_scores(scores_path: str | Path, judge: dict, *, config_sha: str | None = None,
                 prompt_sha: dict | None = None) -> dict[Key, float]:
    """{(set, bench, worker, task, k): p} of every valid score in a package; raises on rows of another judge or
    configuration, on unparseable lines and on scored rows that break the validity rule (see ``read_scores``)."""
    return read_scores(scores_path, judge, config_sha=config_sha, prompt_sha=prompt_sha)["valid"]


class WindowCosts(dict):
    """{(set, bench, worker, task, k): dollars} of one judge configuration, plus ledger bookkeeping.

    Attributes: ``calls`` {key: requests}, ``unknown_usage`` {key: requests whose usage never arrived; priced at $0
    here, but their real cost is unknown}, ``other_calls`` (readable lines of another judge or configuration, not part
    of this bill), ``unreadable_lines`` (torn lines), ``call_ids`` (every readable call id of the ledger).
    """

    def __init__(self):
        super().__init__()
        self.calls: dict[Key, int] = {}
        self.unknown_usage: dict[Key, int] = {}
        self.other_calls = 0
        self.unreadable_lines = 0
        self.call_ids: set[str] = set()

    @property
    def total_usd(self) -> float:
        return sum(self.values())

    @property
    def n_calls(self) -> int:
        return sum(self.calls.values())

    @property
    def n_unknown_usage(self) -> int:
        return sum(self.unknown_usage.values())


def window_costs(calls_path: str | Path, judge: dict, prices: dict, pricing: str = "list", *,
                 config_sha: str | None = None) -> WindowCosts:
    """Per-window judge cost from the call ledger: judge_cost_usd(usage_sum(all calls of this judge + configuration
    for that window)). Failed and truncated requests are included; a ledger line is counted once per call_id.
    ``config_sha`` defaults to ``judge_config_sha(judge)``."""
    if not Path(calls_path).exists():
        raise FileNotFoundError(calls_path)
    expected_sha = config_sha or judge_config_sha(judge)
    by_key = collections.defaultdict(list)
    out = WindowCosts()
    lines, _ = _read_lines(Path(calls_path))
    for line in lines:
        if not line.strip():
            continue
        try:
            call = json.loads(line)
            key = window_key(call)
        except (ValueError, TypeError, AttributeError):
            out.unreadable_lines += 1
            continue
        if call.get("call_id") in out.call_ids:
            continue
        out.call_ids.add(call.get("call_id"))
        if call.get("judge_id") == judge["judge_id"] and call.get("config_sha") == expected_sha:
            by_key[key].append(call)
        else:
            out.other_calls += 1
    for key, calls in by_key.items():
        out[key] = judge_cost_usd(usage_sum(calls), judge["model"], prices, pricing)
        out.calls[key] = len(calls)
        unknown = sum(1 for c in calls if c.get("usage_unknown"))
        if unknown:
            out.unknown_usage[key] = unknown
    return out


def load_package(package_dir: str | Path, judge: dict, prices: dict, pricing: str = "list", *,
                 config_sha: str | None = None, prompt_sha: dict | None = None) -> tuple[dict[Key, float], WindowCosts]:
    """(valid scores, window costs) of a package directory, refusing a ledger that would understate the bill: every
    request referenced by a score row must be in calls.jsonl, and every valid window must have a cost record."""
    package_dir = Path(package_dir)
    scores = read_scores(package_dir / "scores.jsonl", judge, config_sha=config_sha, prompt_sha=prompt_sha)
    costs = window_costs(package_dir / "calls.jsonl", judge, prices, pricing, config_sha=config_sha)
    missing = scores["call_ids"] - costs.call_ids
    if missing:
        raise ValueError(f"{package_dir}: {len(missing)} request(s) referenced by score rows are missing from calls.jsonl")
    uncosted = [key for key in scores["valid"] if key not in costs]
    if uncosted:
        raise ValueError(f"{package_dir}: {len(uncosted)} valid window(s) without a ledger record, e.g. {uncosted[0]}")
    return scores["valid"], costs


# ---------------------------------------------------------------------------------------------------- running a judge
def plan_windows(windows, judge: dict, *, set_name: str = "test", all_windows: bool = False,
                 limit: int | None = None) -> tuple[list[dict], collections.Counter]:
    """(windows to be judged, in input order, {worker: windows skipped as another executor family}).

    A window without a ``set`` field gets ``set_name``. Only checkpoints are kept unless ``all_windows``; ``limit``
    keeps the first N planned windows.
    """
    planned, other_family, seen = [], collections.Counter(), set()
    for window in windows:
        window = dict(window, set=window.get("set") or set_name, k=int(window["k"]))
        if not all_windows and not is_checkpoint(window["k"]):
            continue
        if window["worker"] not in judge["workers"]:
            other_family[window["worker"]] += 1
            continue
        key = window_key(window)
        if key in seen:
            raise ValueError(f"duplicate window {key}")
        seen.add(key)
        planned.append(window)
    return (planned[:limit] if limit else planned), other_family


def _group_runs(windows: list[dict]) -> list[list[dict]]:
    """Windows grouped by run (first appearance order), each run in increasing k; a thread scores a whole run."""
    runs = collections.OrderedDict()
    for window in windows:
        runs.setdefault(window_key(window)[:4], []).append(window)
    return [sorted(run, key=lambda w: w["k"]) for run in runs.values()]


def _repair_tail(path: Path, quarantine: Path | None) -> None:
    """Make sure the next append starts on a new line. A torn last line (an interrupted write) is moved to
    ``quarantine``; without a quarantine file it stays in place and is merely terminated."""
    lines, torn = _read_lines(path)
    if not torn:
        return
    try:
        json.loads(lines[-1])
        complete = True
    except ValueError:
        complete = False
    if complete or quarantine is None:
        with open(path, "a") as f:
            f.write("\n")
        return
    with open(quarantine, "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": "torn last line",
                            "raw": lines[-1]}, ensure_ascii=False) + "\n")
    data = path.read_bytes()
    with open(path, "r+b") as f:
        f.truncate(data.rfind(b"\n") + 1)


def _existing_rows(scores_path: Path, judge: dict, cfg_sha: str) -> dict[Key, dict]:
    """Valid rows already in the package; tolerates only a torn last line (repaired before anything is appended)."""
    lines, torn = _read_lines(scores_path)
    result = read_scores(scores_path, judge, config_sha=cfg_sha, strict=False)
    problems = [(n, why) for n, why in result["problems"] if not (torn and n == len(lines))]
    if problems:
        raise ValueError(f"{scores_path} holds {len(problems)} line(s) of another judge / configuration or "
                         f"unusable rows (e.g. line {problems[0][0]}: {problems[0][1]}); use a fresh --out-dir")
    return result["rows"]


class _Ledger:
    """Append-only calls.jsonl; ``book`` writes one line per request and stamps the attempt with its call_id."""

    def __init__(self, path: Path, judge: dict, cfg_sha: str):
        self.path, self.judge, self.cfg_sha = path, judge, cfg_sha
        self.lock = threading.Lock()
        self.n_calls = 0
        _repair_tail(path, None)

    def book(self, window: dict, attempt: dict) -> dict:
        attempt["call_id"] = uuid.uuid4().hex
        line = {"call_id": attempt["call_id"], "at": time.strftime("%Y-%m-%d %H:%M:%S"), "set": window["set"],
                "bench": window["bench"], "worker": window["worker"], "task": window["task"], "k": int(window["k"]),
                "judge_id": self.judge["judge_id"], "model": self.judge["model"], "config_sha": self.cfg_sha,
                **{k: attempt.get(k) for k in LEDGER_FIELDS}}
        with self.lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            self.n_calls += 1
        return attempt


def _check_manifest(path: Path, judge: dict, cfg_sha: str) -> dict | None:
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("judge_id") != judge["judge_id"] or manifest.get("config_sha") != cfg_sha:
        raise ValueError(f"{path.parent} belongs to judge {manifest.get('judge_id')!r} with config "
                         f"{manifest.get('config_sha')!r}, not {judge['judge_id']!r} / {cfg_sha!r}: a package is never "
                         "mixed, use a fresh --out-dir")
    return manifest


def run_judge(windows, judge: dict, out_dir: str | Path, *, client=None, base_url: str | None = None,
              set_name: str = "test", all_windows: bool = False, limit: int | None = None, threads: int = 2,
              timeout: float = 600.0, max_failed_streak: int = 3, dry_run: bool = False,
              windows_source: str | None = None, sleep=time.sleep, log=print) -> dict:
    """Score ``windows`` with ``judge`` into the package ``out_dir`` and return a summary with an ``exit_code``:
    0 every planned window has a valid score, 1 some do not (run again to retry them), 3 stopped (credentials,
    exhausted account, repeated failures, an unexpected error), 4 stopped because the API answered with a model
    outside the allow-list.
    ``client`` overrides the API client (tests); ``dry_run`` reads the package but makes no request and writes nothing.
    """
    out_dir = Path(out_dir)
    scores_path, calls_path, manifest_path = out_dir / "scores.jsonl", out_dir / "calls.jsonl", out_dir / "manifest.json"
    cfg_sha = judge_config_sha(judge)
    planned, other_family = plan_windows(windows, judge, set_name=set_name, all_windows=all_windows, limit=limit)
    if other_family:
        log(f"[{judge['judge_id']}] skipping windows of executors outside this judge's family "
            f"(workers {judge['workers']}): {dict(other_family)}")
    if not planned:
        raise ValueError(f"no window to judge for {judge['judge_id']!r}: check the workers list of the judge "
                         f"({judge['workers']}) and the windows' worker labels")
    for window in planned:          # refuse before the first request, not inside a worker thread
        judge_prompt(window["prompt"])
    _check_manifest(manifest_path, judge, cfg_sha)
    existing = _existing_rows(scores_path, judge, cfg_sha)
    for window in planned:
        row = existing.get(window_key(window))
        if row is not None and (row.get("prompt_sha256") != sha16(window["prompt"])
                                or row.get("judge_prompt_sha256") != sha16(judge_prompt(window["prompt"]))):
            raise ValueError(f"{scores_path}: the valid row of {window_key(window)} was scored on a different prompt; "
                             "the windows changed, use a fresh --out-dir")
    todo = [w for w in planned if window_key(w) not in existing]
    runs = _group_runs(todo)
    per_cell = collections.Counter(f"{w['bench']}/{w['worker']}" for w in planned)
    summary = {"judge_id": judge["judge_id"], "model": judge["model"], "config_sha": cfg_sha, "out_dir": str(out_dir),
               "n_planned": len(planned), "n_valid_before": len(planned) - len(todo), "n_to_score": len(todo),
               "n_runs_to_score": len(runs), "per_cell": dict(per_cell)}
    log(f"[{judge['judge_id']}] {judge['model']} ({judge['api']} API), config {cfg_sha}: {len(planned)} windows planned "
        f"{dict(per_cell)}, {summary['n_valid_before']} already valid, {len(todo)} to score in {len(runs)} runs")
    if dry_run or not todo:
        summary.update(exit_code=EXIT_COMPLETE, dry_run=dry_run)
        return summary

    client = client or make_client(judge, base_url=base_url, timeout=timeout)
    out_dir.mkdir(parents=True, exist_ok=True)
    _repair_tail(scores_path, out_dir / "scores.quarantine.jsonl")
    manifest = _check_manifest(manifest_path, judge, cfg_sha) or {
        "judge_id": judge["judge_id"], "config_sha": cfg_sha, "config": judge_config(judge),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "invocations": []}
    manifest["invocations"].append({
        "at": time.strftime("%Y-%m-%d %H:%M:%S"), "windows": windows_source, "set_default": set_name,
        "all_windows": all_windows, "limit": limit, "threads": threads, "endpoint": getattr(client, "url", None),
        "allowed_backend_models": judge["allowed_backend_models"], "workers": judge["workers"],
        "n_planned": len(planned), "n_valid_before": summary["n_valid_before"]})
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    os.replace(tmp, manifest_path)

    ledger = _Ledger(calls_path, judge, cfg_sha)
    work = queue.Queue()
    for run in runs:
        work.put(run)
    stop = StopSignal()
    lock = threading.Lock()
    counts = collections.Counter()
    started = time.time()

    with open(scores_path, "a") as scores_file:
        def score_run(run):
            for window in run:
                final, attempts, error = score_window(
                    window, judge, client, book=lambda attempt, w=window: ledger.book(w, attempt), stop=stop, sleep=sleep)
                with lock:
                    if attempts:        # a window stopped before its first request leaves no trace
                        row = score_row(window, judge, cfg_sha, final, attempts, error)
                        scores_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        scores_file.flush()
                        counts["windows"] += 1
                        if row["p"] is not None:
                            counts["valid"] += 1
                            counts["streak"] = 0
                        else:
                            counts["failed"] += 1
                            if row.get("error") not in STOP_ERRORS:
                                counts["streak"] += 1
                                if counts["streak"] >= max_failed_streak:
                                    stop.set("failures", f"{counts['streak']} consecutive windows without a valid "
                                                         f"score (last: {row.get('error')})")
                        if counts["windows"] % 25 == 0:
                            log(f"  {counts['windows']} windows | valid {counts['valid']} | failed {counts['failed']} "
                                f"| requests {ledger.n_calls} | {(time.time() - started) / 60:.1f} min")
                if stop.is_set():
                    return

        def worker():
            try:
                while not stop.is_set():
                    try:
                        run = work.get_nowait()
                    except queue.Empty:
                        return
                    score_run(run)
            except Exception as e:      # e.g. a failed disk write: stop the run instead of losing a thread silently
                stop.set("error", f"{type(e).__name__}: {e}")
                log(traceback.format_exc())

        pool = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, threads))]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()

    valid = read_scores(scores_path, judge, config_sha=cfg_sha)["valid"]
    missing = [window_key(w) for w in planned if window_key(w) not in valid]
    summary.update(windows_scored_now=counts["windows"], failed_now=counts["failed"], requests_now=ledger.n_calls,
                   n_valid=len(planned) - len(missing), n_missing=len(missing), stopped=stop.kind,
                   stop_message=stop.message, elapsed_min=round((time.time() - started) / 60, 1))
    if stop.kind == "backend_model":
        summary["exit_code"] = EXIT_BACKEND_MODEL
    elif stop.kind is not None:
        summary["exit_code"] = EXIT_STOPPED
    else:
        summary["exit_code"] = EXIT_INCOMPLETE if missing else EXIT_COMPLETE
    return summary
