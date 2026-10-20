"""Run stock Harbor Terminus-2 trials through the metering gateway, one trial per task (paper Sec. 3.1, App. A).

Every task gets a fresh run id, passed to the agent as its API key ``"<run_id>|<task>"`` so that each LLM call of
the trial is billed to that run in the gateway's ledger (one gateway run per task, so per-task cost is a ledger
fact, not an estimate). Limits come from the executor configuration: the Terminus-2 turn budget (the binding
limit), separate agent and verifier timeouts, and an outer wall-clock timeout on the whole Harbor process. The
gateway's spend valve sits above the cost of a full 100-turn run and should never fire: a trial stopped by it
never reaches the verifier, so it is recorded as censored (``end == "cap"``), not as a failure.

Collection is resumable. A task whose record on disk is final is skipped: it has a verdict, or it ended in a
final no-verdict state (valve, or an infrastructure failure seen on two attempts). Provider rate-limit walls and
Docker Hub throttling are not readings of the task, so those trials are not written and run again next time.
Runs without a verdict carry ``reward: null`` and must be excluded from analysis.

One record per task is written to ``<out>/<bench>/<worker>/<task>.json`` and every final record is listed in
``runs.jsonl`` in the same directory.
"""
from __future__ import annotations

import concurrent.futures as cf
import fcntl
import json
import os
import random
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from stoploss.collect import Executor

FINAL_ENDS = ("cap", "env-error-final", "harness-error-final")
MAX_ENV_RETRIES = 2
REGISTRY_BACKOFF_SEC = 300.0
KILL_GRACE_SEC = 60.0

# Provider rate limits only. A bare "429" would also match Docker Hub throttling, which has a different cure.
_RATE_LIMIT_RE = re.compile(r"rate.?limit|too many requests|upstream 429", re.I)
# Docker Hub throttling: retryable, but the fleet must slow down rather than hammer the registry.
_REGISTRY_RE = re.compile(r"registry-1\.docker\.io|docker\.io/v2/|toomanyrequests", re.I)
_VALVE_RE = re.compile(r"spend valve|budget exhausted|\b402\b", re.I)
_ENV_RE = re.compile(r"tmux|no server running|failed to send|docker|container|connection reset|environment", re.I)


# ---------------------------------------------------------------------------------------------------- gateway
class GatewayClient:
    """The two gateway endpoints the runner needs (standard library only)."""

    def __init__(self, url: str):
        self.url = url.rstrip("/").removesuffix("/v1")

    @property
    def api_base(self) -> str:
        return f"{self.url}/v1"

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.url}{path}", timeout=30) as response:
            return json.loads(response.read())

    def health(self) -> dict:
        return self._get("/health")

    def run_totals(self, run_id: str) -> dict:
        """{"n_calls", "cost_usd"} of one run; a run that made no call has zero of both."""
        try:
            return self._get(f"/runs/{run_id}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"n_calls": 0, "cost_usd": 0.0}
            raise


# ---------------------------------------------------------------------------------------------------- tasks
def read_task_list(path: str | Path) -> list[str]:
    """Task names from a JSON list, a JSON object with a "tasks" list, or a text file with one name per line."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    return list(data["tasks"] if isinstance(data, dict) else data)


def select_tasks(dataset_dir: str | Path, bench: str, tasks_file: str | Path | None = None,
                 n: int | None = None, seed: int = 0) -> list[str]:
    """Sorted task names to collect: every task of the Harbor dataset directory, or those listed in tasks_file.

    With ``n`` smaller than the list, a deterministic subset is taken (shuffle seeded by seed and bench, keep the
    first n, sort), so a smaller budget always selects a subset of a larger one with the same seed.
    """
    dataset_dir = Path(dataset_dir)
    available = sorted(p.name for p in dataset_dir.iterdir() if (p / "task.toml").is_file())
    tasks = available
    if tasks_file is not None:
        wanted = sorted(set(read_task_list(tasks_file)))
        missing = sorted(set(wanted) - set(available))
        if missing:
            raise SystemExit(f"{len(missing)} listed tasks are not in {dataset_dir}, e.g. {missing[:3]}")
        tasks = wanted
    if n and n < len(tasks):
        order = tasks[:]
        random.Random(f"{seed}:{bench}:budget").shuffle(order)
        tasks = sorted(order[:n])
    return tasks


# ---------------------------------------------------------------------------------------------------- outcomes
def valid(record: dict) -> bool:
    """A record is final when it carries a verdict or when retrying it has been given up; otherwise re-run it."""
    return record.get("reward") is not None or record.get("end") in FINAL_ENDS


def classify(trial_dir: Path) -> tuple[str, str]:
    """(kind, detail) of a finished trial, read from Harbor's result.json.

    Harbor's exit code is not informative (it is 0 even when the agent raised), so ``exception_info`` is the
    only reliable channel.
    """
    result_file = trial_dir / "result.json"
    if not result_file.exists():
        return "no-result", ""
    try:
        result = json.loads(result_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return "no-result", repr(exc)[:200]
    if result.get("verifier_result") is not None:
        return "verdict", ""
    info = result.get("exception_info") or {}
    if not info:
        return "no-verdict", "verifier_result null with no exception recorded"
    message = f"{info.get('exception_type', '')}: {info.get('exception_message', '')}"
    if _REGISTRY_RE.search(message) and re.search(r"429|too many requests", message, re.I):
        return "registry", message[:300]
    if _RATE_LIMIT_RE.search(message):
        return "rate-limit", message[:300]
    if _VALVE_RE.search(message):
        return "cap", message[:300]
    if "AgentTimeout" in message:
        return "timeout", message[:300]
    if _ENV_RE.search(message):
        return "env", message[:300]
    return "agent-error", message[:300]


# ---------------------------------------------------------------------------------------------------- one trial
_registry_gate = threading.Event()
_registry_gate.set()
_registry_lock = threading.Lock()


def _registry_backoff(seconds: float = REGISTRY_BACKOFF_SEC) -> None:
    """Hold new trials for a while after Docker Hub throttling; retrying at once only burns the next window."""
    with _registry_lock:
        if _registry_gate.is_set():
            print(f"[registry] Docker Hub rate limit: pausing new trials for {seconds:.0f}s", flush=True)
            _registry_gate.clear()
            timer = threading.Timer(seconds, _registry_gate.set)
            timer.daemon = True
            timer.start()


def harbor_job_config(executor: Executor, *, task: str, dataset_dir: Path, job_name: str, jobs_dir: Path,
                      api_base: str, max_turns: int) -> dict:
    """Harbor JobConfig for one Terminus-2 trial on one task."""
    agent = {
        "name": "terminus-2",
        "model_name": executor.agent_model_name,
        # parser_name "json" is Terminus-2's default action format, which every paper run used.
        "kwargs": {"api_base": api_base, "max_turns": max_turns, "parser_name": "json"},
    }
    if executor.agent_timeout_sec is not None:
        agent["override_timeout_sec"] = executor.agent_timeout_sec
    config = {
        "job_name": job_name,
        "jobs_dir": str(jobs_dir),
        "n_attempts": 1,
        "quiet": True,
        "agents": [agent],
        "datasets": [{"path": str(dataset_dir), "task_names": [task]}],
    }
    if executor.verifier_timeout_sec is not None:
        config["verifier"] = {"override_timeout_sec": executor.verifier_timeout_sec}
    return config


def _run_harbor(cmd: list[str], *, cwd: str | None, env: dict, timeout: float) -> tuple[int, str]:
    """(exit code, tail of the output); on the outer timeout the whole process group is stopped."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, (err or out or "")[-1500:]
    except subprocess.TimeoutExpired:
        for sig, grace in ((signal.SIGTERM, KILL_GRACE_SEC), (signal.SIGKILL, None)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.communicate(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        return -9, f"trial timeout {timeout:.0f}s"


def _trial_dir(job_dir: Path) -> Path | None:
    trials = sorted(p for p in job_dir.glob("*__*") if p.is_dir()) if job_dir.is_dir() else []
    return trials[0] if trials else None


def run_one(task: str, *, executor: Executor, bench: str, dataset_dir: Path, arm_dir: Path, jobs_dir: Path,
            gateway: GatewayClient, ledger_path: str | None, harbor_cmd: list[str], harbor_dir: str | None,
            max_turns: int) -> dict:
    """Run (or skip) one task and return its record; the record is written unless the trial must be re-run."""
    record_file = arm_dir / f"{task}.json"
    previous = None
    if record_file.exists():
        try:
            previous = json.loads(record_file.read_text())
        except json.JSONDecodeError:
            previous = None
        if previous is not None and valid(previous):
            return previous

    run_id = f"{bench[:16]}-{executor.label[:16]}-{uuid.uuid4().hex[:12]}"
    job_name = f"{bench[:16]}-{executor.label[:16]}-{task[:40]}-{uuid.uuid4().hex[:6]}"
    config = harbor_job_config(executor, task=task, dataset_dir=dataset_dir, job_name=job_name,
                               jobs_dir=jobs_dir, api_base=gateway.api_base, max_turns=max_turns)
    _registry_gate.wait()
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)             # the agent only ever talks to the gateway
    env.update({"OPENAI_API_KEY": f"{run_id}|{task}", "OPENAI_BASE_URL": gateway.api_base,
                "OPENAI_API_BASE": gateway.api_base})
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="stoploss-job-") as tmp:
        config_file = Path(tmp) / "job.json"
        config_file.write_text(json.dumps(config, indent=2))
        rc, err = _run_harbor([*harbor_cmd, "run", "-c", str(config_file), "-y"], cwd=harbor_dir, env=env,
                              timeout=executor.outer_timeout_sec)
    wall_sec = round(time.monotonic() - started, 1)
    try:
        totals = gateway.run_totals(run_id)
    except (OSError, ValueError) as exc:
        totals = {"n_calls": None, "cost_usd": None, "error": repr(exc)[:200]}

    record = {"task": task, "bench": bench, "worker": executor.label, "model": executor.model, "run_id": run_id,
              "job": job_name, "rc": rc, "wall_sec": wall_sec, "cost_usd": totals.get("cost_usd"),
              "n_calls": totals.get("n_calls"), "valve_usd": executor.spend_valve_usd, "ledger": ledger_path,
              "attempts": int((previous or {}).get("attempts", 0)) + 1}
    if totals.get("error"):
        record["gateway_error"] = totals["error"]

    trial_dir = _trial_dir(jobs_dir / job_name)
    if trial_dir is None:
        # Harbor never produced a trial directory: infrastructure, not the agent.
        record["reward"] = None
        record["end"] = "harness-error-final" if record["attempts"] >= MAX_ENV_RETRIES else "harness-error"
        record["error"] = err
        record_file.write_text(json.dumps(record, indent=2))
        return record

    reward_file = trial_dir / "verifier" / "reward.txt"
    record["reward"] = float(reward_file.read_text().strip()) if reward_file.exists() else None
    trajectory = trial_dir / "agent" / "trajectory.json"
    record["trajectory"] = str(trajectory.resolve()) if trajectory.exists() else None
    record["n_steps"] = record["n_tool_steps"] = None
    if trajectory.exists():
        try:
            from stoploss.costs import tool_steps

            record["n_steps"] = len(json.loads(trajectory.read_text()).get("steps") or [])
            record["n_tool_steps"] = len(tool_steps(trajectory))
        except (OSError, ValueError):
            pass
    record["trial_dir"] = str(trial_dir.resolve())
    kind, detail = classify(trial_dir)
    record["kind"] = kind
    if detail:
        record["detail"] = detail

    if record["reward"] is not None:
        record["end"] = "done"
    elif kind == "rate-limit":
        # Not a reading: the agent was starved of calls, not beaten by the task. Not written, re-run later.
        record["end"] = "rate-limited"
        print(f"[{executor.label}] {task}: provider rate limit during the trial -> will be re-run", flush=True)
        return record
    elif kind == "registry":
        record["end"] = "registry-429"
        _registry_backoff()
        print(f"[{executor.label}] {task}: Docker Hub rate limit -> will be re-run", flush=True)
        return record
    elif kind == "cap":
        # The spend valve fired. Final (a re-run would burn the valve again) but censored: no verifier ran.
        record["end"] = "cap"
    else:
        record["end"] = "env-error-final" if record["attempts"] >= MAX_ENV_RETRIES else "env-error"
    if record["reward"] is None and rc != 0:
        record["error"] = err
    record_file.write_text(json.dumps(record, indent=2))
    return record


# ---------------------------------------------------------------------------------------------------- one arm
def arm_directory(out_dir: str | Path, bench: str, executor: Executor, max_turns: int) -> Path:
    """<out>/<bench>/<worker>; a non-default turn budget is a different arm and gets its own directory."""
    suffix = "" if max_turns == executor.max_turns else f"_t{max_turns}"
    return Path(out_dir) / bench / f"{executor.label}{suffix}"


def write_index(arm_dir: Path) -> Path:
    """runs.jsonl: every final record of the arm, sorted by task."""
    records = []
    for path in sorted(arm_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and "task" in record and valid(record):
            records.append(record)
    index = arm_dir / "runs.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in records))
    return index


def collect(executor: Executor, *, bench: str, dataset_dir: str | Path, tasks: list[str], out_dir: str | Path,
            gateway: GatewayClient, harbor_cmd: str | list[str] = "harbor", harbor_dir: str | None = None,
            jobs_dir: str | Path | None = None, concurrency: int = 4, max_turns: int | None = None,
            ledger_path: str | None = None) -> list[dict]:
    """Collect one (bench, executor) arm; returns the records of this invocation."""
    max_turns = max_turns or executor.max_turns
    dataset_dir = Path(dataset_dir).resolve()
    arm_dir = arm_directory(out_dir, bench, executor, max_turns)
    arm_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = Path(jobs_dir or Path(out_dir) / "jobs").resolve()
    harbor_cmd = shlex.split(harbor_cmd) if isinstance(harbor_cmd, str) else list(harbor_cmd)

    # A gateway that is down would turn every trial into an infrastructure failure within minutes.
    try:
        health = gateway.health()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"gateway {gateway.url} is not reachable ({exc!r}); start it first") from exc
    if executor.label not in health.get("executors", []):
        raise SystemExit(f"gateway {gateway.url} does not serve executor {executor.label!r}")
    ledger_path = ledger_path or health.get("ledger")

    # One driver per arm: records are written when a trial ends, so a second driver would re-run every task
    # that is in flight or not yet started.
    lock = open(arm_dir / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit(f"another collection is already running for {arm_dir}")

    try:
        done = sum(1 for t in tasks if (arm_dir / f"{t}.json").exists())
        print(f"bench={bench} worker={executor.label} model={executor.model} tasks={len(tasks)} already={done} "
              f"max_turns={max_turns} valve=${executor.spend_valve_usd} concurrency={concurrency}", flush=True)
        records, n_solved, n_failed, spent = [], 0, 0, 0.0
        started = time.time()
        with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(run_one, task, executor=executor, bench=bench, dataset_dir=dataset_dir,
                                   arm_dir=arm_dir, jobs_dir=jobs_dir, gateway=gateway, ledger_path=ledger_path,
                                   harbor_cmd=harbor_cmd, harbor_dir=harbor_dir, max_turns=max_turns): task
                       for task in tasks}
            for i, future in enumerate(cf.as_completed(futures), 1):
                task = futures[future]
                try:
                    record = future.result()
                except Exception as exc:          # keep the fleet going; the task is re-run next time
                    record = {"task": task, "end": "driver-crash", "error": repr(exc)[:300]}
                    (arm_dir / f"{task}.json").write_text(json.dumps(record, indent=2))
                records.append(record)
                spent += record.get("cost_usd") or 0.0
                n_solved += record.get("reward") == 1.0
                n_failed += record.get("reward") == 0.0
                print(f"[{i}/{len(tasks)}] {task} reward={record.get('reward')} ${record.get('cost_usd')} "
                      f"steps={record.get('n_steps')} end={record.get('end')} | solved {n_solved} failed {n_failed} "
                      f"spent ${spent:.2f} {(time.time() - started) / 60:.0f} min", flush=True)
        index = write_index(arm_dir)
        print(f"DONE bench={bench} worker={executor.label} solved={n_solved} failed={n_failed} "
              f"spent=${spent:.2f} index={index}", flush=True)
        return records
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
