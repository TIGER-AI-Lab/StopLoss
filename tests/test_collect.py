"""Collection stage: call ledger, metering gateway (against fake provider upstreams) and Harbor runner (fake Harbor).

No network and no Harbor: upstream providers are ``httpx.MockTransport`` handlers and Harbor is a small script
that writes the files a real trial would leave behind.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")              # the gateway's dependencies: pip install -e ".[collect]"
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from stoploss import costs  # noqa: E402
from stoploss.collect import Executor, load_executors  # noqa: E402
from stoploss.collect import gateway as gw  # noqa: E402
from stoploss.collect import harbor_runner as hr  # noqa: E402
from stoploss.collect.ledger import Ledger  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICES = costs.load_prices(ROOT / "configs" / "prices_20260918.json")
EXECUTORS = load_executors(ROOT / "configs" / "executors.json")
AUTH = {"Authorization": "Bearer run-1|task-a"}


# ---------------------------------------------------------------------------------------------------- fakes
class FakeUpstream:
    """Records every upstream request and answers with the queued (status, json) responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append({"url": str(request.url), "headers": dict(request.headers),
                              "json": json.loads(request.content)})
        status, body = self.responses.pop(0)
        return httpx.Response(status, json=body)


def openai_reply(prompt, cached, completion, reasoning=0, text='{"analysis": "a", "commands": []}'):
    return {"id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-5.6-luna-backend",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
                      "prompt_tokens_details": {"cached_tokens": cached},
                      "completion_tokens_details": {"reasoning_tokens": reasoning}}}


def anthropic_reply(uncached, read, write, output, thinking_tokens=0, text="done"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
            "content": [{"type": "thinking", "thinking": "private", "signature": "sig"},
                        {"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": uncached, "cache_read_input_tokens": read,
                      "cache_creation_input_tokens": write, "output_tokens": output,
                      "output_tokens_details": {"thinking_tokens": thinking_tokens}}}


@pytest.fixture
def make_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")

    def make(responses, executors=EXECUTORS, ledger_path=None):
        upstream = FakeUpstream(responses)
        ledger = Ledger(ledger_path or tmp_path / "ledger.jsonl")
        app = gw.create_app(executors, ledger, PRICES, transport=httpx.MockTransport(upstream))
        return app, upstream, ledger

    return make


def ledger_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# ---------------------------------------------------------------------------------------------------- config
def test_executor_config_matches_the_paper():
    luna, terra, sonnet = EXECUTORS["luna"], EXECUTORS["terra"], EXECUTORS["sonnet46t"]
    for ex in (luna, terra, sonnet):
        assert (ex.max_turns, ex.agent_timeout_sec, ex.verifier_timeout_sec, ex.outer_timeout_sec) == \
               (100, 7200, 7200, 16000)
        costs.price_of(PRICES, ex.model)
    assert (luna.model, luna.spend_valve_usd, luna.reasoning_effort) == ("gpt-5.6-luna", 9.0, None)
    assert (terra.model, terra.spend_valve_usd) == ("gpt-5.6-terra", 22.0)
    assert (sonnet.provider, sonnet.model, sonnet.spend_valve_usd) == ("anthropic", "claude-sonnet-4-6", 26.0)
    assert (sonnet.thinking_budget_tokens, sonnet.max_output_tokens) == (8000, 16000)
    assert sonnet.agent_model_name == "openai/sonnet46t"
    with pytest.raises(ValueError):
        Executor(label="claude-sonnet", provider="anthropic", model="claude-sonnet-4-6")


# ---------------------------------------------------------------------------------------------------- ledger
def test_ledger_is_thread_safe_and_readable_by_costs(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")

    def write(worker_id):
        for i in range(50):
            ledger.record_call({"call_id": f"c{worker_id}-{i}", "run_id": f"r{worker_id % 2}", "cost_usd": 0.5,
                                "prompt_tokens": 1, "cache_read_tokens": 0, "cache_write_tokens": 0,
                                "input_tokens_total": 1, "completion_tokens": 1})

    threads = [threading.Thread(target=write, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = Path(ledger.path).read_text().splitlines()
    assert len(lines) == 400
    assert all('"type": "call"' in line for line in lines)      # the substring costs.load_ledger filters on
    calls = [json.loads(line) for line in lines]
    assert all(c["type"] == "call" and c["discarded"] is False and "ts" in c for c in calls)
    assert ledger.run_totals("r0")["n_calls"] == 200 and ledger.run_totals("r1")["cost_usd"] == pytest.approx(100.0)
    assert sum(len(v) for v in costs.load_ledger(ledger.path).values()) == 400
    assert Ledger(ledger.path).run_totals("r0")["n_calls"] == 200          # totals rebuilt from the file


# ---------------------------------------------------------------------------------------------------- gateway
def test_openai_call_is_forwarded_and_metered(make_gateway):
    app, upstream, ledger = make_gateway([(200, openai_reply(prompt=1000, cached=600, completion=250,
                                                             reasoning=100))])
    body = {"model": "luna", "messages": [{"role": "user", "content": "solve it"}], "max_tokens": 50,
            "temperature": 0.3, "top_p": 0.9, "stop": ["x"], "session_id": "s-1", "logprobs": True}
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens"] == 1000           # the agent sees the provider's usage

    sent = upstream.requests[0]
    assert sent["url"] == "https://api.openai.com/v1/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer sk-test"
    assert sent["json"] == {"messages": body["messages"], "model": "gpt-5.6-luna", "prompt_cache_key": "run-1"}

    [call] = ledger_lines(ledger.path)
    assert {k: call[k] for k in ("run_id", "task_id", "worker", "model", "requested_model", "backend_model")} == {
        "run_id": "run-1", "task_id": "task-a", "worker": "luna", "model": "gpt-5.6-luna",
        "requested_model": "gpt-5.6-luna", "backend_model": "gpt-5.6-luna-backend"}
    assert (call["prompt_tokens"], call["cache_read_tokens"], call["cache_write_tokens"]) == (400, 600, 0)
    assert (call["input_tokens_total"], call["completion_tokens"], call["reasoning_tokens"]) == (1000, 250, 100)
    assert call["cost_usd"] == pytest.approx((400 * 0.20 + 600 * 0.02 + 250 * 1.20) / 1e6)
    assert call["type"] == "call" and call["discarded"] is False


def test_anthropic_call_is_translated_and_metered(make_gateway):
    app, upstream, ledger = make_gateway([(200, anthropic_reply(12, 5000, 700, 300, thinking_tokens=120,
                                                                text="next command"))])
    body = {"model": "openai/sonnet46t", "temperature": 0.7, "messages": [
        {"role": "system", "content": "be careful"},
        {"role": "user", "content": "the task"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "observation 1"},
        {"role": "user", "content": [{"type": "text", "text": "observation 2"}]}]}
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body, headers={"x-api-key": "run-2|task-b"})
    assert response.status_code == 200
    reply = response.json()
    assert reply["choices"][0] == {"index": 0, "message": {"role": "assistant", "content": "next command"},
                                   "finish_reason": "stop"}
    assert reply["model"] == "claude-sonnet-4-6"
    assert reply["usage"] == {"prompt_tokens": 12, "completion_tokens": 300, "cache_read_input_tokens": 5000,
                              "cache_creation_input_tokens": 700, "total_tokens": 6012}

    sent = upstream.requests[0]
    assert sent["url"] == "https://api.anthropic.com/v1/messages"
    assert sent["headers"]["x-api-key"] == "ak-test" and sent["headers"]["anthropic-version"] == "2023-06-01"
    ephemeral = {"type": "ephemeral"}
    assert sent["json"] == {
        "model": "claude-sonnet-4-6",
        "max_tokens": 16000,
        "thinking": {"type": "enabled", "budget_tokens": 8000},
        "system": [{"type": "text", "text": "be careful", "cache_control": ephemeral}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "the task"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "observation 1"},
                                         {"type": "text", "text": "observation 2", "cache_control": ephemeral}]}]}

    [call] = ledger_lines(ledger.path)
    assert (call["run_id"], call["task_id"], call["worker"], call["model"]) == \
           ("run-2", "task-b", "sonnet46t", "claude-sonnet-4-6")
    assert (call["prompt_tokens"], call["cache_read_tokens"], call["cache_write_tokens"]) == (12, 5000, 700)
    assert (call["input_tokens_total"], call["completion_tokens"]) == (5712, 300)
    assert (call["reasoning_tokens"], call["thinking_blocks"], call["backend_model"]) == (120, 1, "claude-sonnet-4-6")
    assert call["cost_usd"] == pytest.approx((12 * 3.0 + 5000 * 0.30 + 700 * 3.75 + 300 * 15.0) / 1e6)


def test_anthropic_output_limit_and_thinking_budget():
    sonnet = EXECUTORS["sonnet46t"]
    request = gw.anthropic_request({"messages": [{"role": "user", "content": "x"}], "max_tokens": 4000}, sonnet)
    assert (request["max_tokens"], request["thinking"]) == (4000, {"type": "enabled", "budget_tokens": 3999})
    request = gw.anthropic_request({"messages": []}, sonnet)
    assert request["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "Continue.", "cache_control": {"type": "ephemeral"}}]}]
    assert "system" not in request and request["max_tokens"] == 16000
    with pytest.raises(gw.HTTPException):
        gw.anthropic_request({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]},
                             sonnet)


def test_run_identity_and_request_validation(make_gateway):
    app, upstream, ledger = make_gateway([(200, openai_reply(10, 0, 5))])
    message = {"model": "luna", "messages": [{"role": "user", "content": "x"}]}
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=message).status_code == 401
        assert client.post("/v1/chat/completions", json=dict(message, model="gpt-4o"), headers=AUTH).status_code == 400
        assert client.post("/v1/chat/completions", json=dict(message, stream=True), headers=AUTH).status_code == 400
        headers = {"Authorization": "Bearer ignored|ignored", "X-Run-Id": "run-h", "X-Task-Id": "task-h"}
        assert client.post("/chat/completions", json=message, headers=headers).status_code == 200
        health = client.get("/health").json()
    [call] = ledger_lines(ledger.path)
    assert (call["run_id"], call["task_id"]) == ("run-h", "task-h")
    assert health["executors"] == ["luna", "sonnet46t", "terra"] and health["ledger"].endswith("ledger.jsonl")


def test_spend_valve(make_gateway, tmp_path):
    executors = dict(EXECUTORS, luna=dataclasses.replace(EXECUTORS["luna"], spend_valve_usd=1e-4))
    reply = openai_reply(prompt=1000, cached=0, completion=100)                   # $0.00032 > valve
    app, upstream, ledger = make_gateway([(200, reply)], executors=executors)
    message = {"model": "luna", "messages": [{"role": "user", "content": "x"}]}
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=message, headers=AUTH).status_code == 200   # crosses
        blocked = client.post("/v1/chat/completions", json=message, headers=AUTH)
        other_run = client.get("/runs/run-2")
        totals = client.get("/runs/run-1").json()
    assert blocked.status_code == 402 and "spend valve" in blocked.json()["detail"]
    assert hr._VALVE_RE.search(blocked.json()["detail"])
    assert len(upstream.requests) == 1 and len(ledger_lines(ledger.path)) == 1
    assert other_run.status_code == 404
    assert totals["n_calls"] == 1 and totals["cost_usd"] == pytest.approx(3.2e-4)

    # the valve survives a gateway restart: totals are rebuilt from the ledger file
    app2, upstream2, _ = make_gateway([], executors=executors, ledger_path=ledger.path)
    with TestClient(app2) as client:
        assert client.post("/v1/chat/completions", json=message, headers=AUTH).status_code == 402
    assert upstream2.requests == []


def test_upstream_errors(make_gateway, monkeypatch):
    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(gw, "_backoff_sleep", no_sleep)
    message = {"model": "luna", "messages": [{"role": "user", "content": "x"}]}

    app, upstream, ledger = make_gateway([(400, {"error": {"message": "context_length_exceeded"}})])
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=message, headers=AUTH)
    assert response.status_code == 502 and "upstream 400" in response.json()["detail"]
    assert len(upstream.requests) == 1 and not ledger.path.exists()

    app, upstream, ledger = make_gateway([(503, {})] + [(200, openai_reply(10, 0, 5))])
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=message, headers=AUTH).status_code == 200
    assert len(upstream.requests) == 2

    app, upstream, ledger = make_gateway([(429, {"error": {"message": "Rate limit reached"}})] * 7)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=message, headers=AUTH)
    assert response.status_code == 429 and len(upstream.requests) == gw.MAX_RETRIES + 1


# ---------------------------------------------------------------------------------------------------- costs
def _trajectory(path, metrics):
    """A minimal ATIF trajectory: the task prompt, then one tool step per metrics dict, then a final message."""
    steps = [{"step_id": "1", "source": "user", "message": "task"}]
    for i, m in enumerate(metrics):
        steps.append({"step_id": str(i + 2), "source": "agent", "message": "m",
                      "tool_calls": [{"tool_call_id": f"call_{i}", "function_name": "bash_command",
                                      "arguments": {"keystrokes": "ls\n"}}], "metrics": m})
    steps.append({"step_id": str(len(steps) + 1), "source": "agent", "message": "done"})
    path.write_text(json.dumps({"schema_version": "ATIF-v1.7", "steps": steps}))
    return path


def test_costs_consumes_gateway_ledger_for_gpt(make_gateway, tmp_path):
    usages = [(1000, 0, 200), (1800, 900, 150), (2500, 1700, 90)]        # (prompt incl. cached, cached, output)
    trailing = (2600, 2400, 40)                                          # a final call that is not a tool step
    app, _, ledger = make_gateway([(200, openai_reply(p, c, o)) for p, c, o in usages + [trailing]])
    with TestClient(app) as client:
        for _ in range(4):
            client.post("/v1/chat/completions", headers=AUTH,
                        json={"model": "luna", "messages": [{"role": "user", "content": "x"}]})
    # what Harbor's LiteLLM records per step for an OpenAI-style usage block
    metrics = [{"prompt_tokens": p, "completion_tokens": o, **({"cached_tokens": c} if c else {})}
               for p, c, o in usages]
    trajectory = _trajectory(tmp_path / "trajectory.json", metrics)
    calls = costs.load_ledger(ledger.path)["run-1"]
    assert costs.align_calls(calls, costs.tool_steps(trajectory)) is not None
    curve = costs.token_curve(trajectory, calls)
    assert curve["source"] == "ledger" and curve["model"] == "gpt-5.6-luna" and curve["n"] == 3
    expected = [[1000, 0, 0, 200], [1900, 900, 0, 350], [2700, 2600, 0, 440], [2900, 5000, 0, 480]]
    assert curve["tok"].tolist() == expected
    dollars = costs.price_curve(curve, PRICES)
    assert dollars[-1] == pytest.approx(((2900 + 5000) * 0.20 + 480 * 1.20) / 1e6)


def test_costs_consumes_gateway_ledger_for_claude(make_gateway, tmp_path):
    usages = [(9, 0, 1085, 215), (9, 1085, 1320, 381), (10, 2405, 489, 214)]   # (uncached, read, write, output)
    app, _, ledger = make_gateway([(200, anthropic_reply(*u)) for u in usages])
    with TestClient(app) as client:
        for _ in usages:
            client.post("/v1/chat/completions", headers=AUTH,
                        json={"model": "sonnet46t", "messages": [{"role": "user", "content": "x"}]})
    # Harbor records Anthropic-style usage as uncached prompt tokens plus cache reads; cache writes are lost
    metrics = [{"prompt_tokens": u, "completion_tokens": o, **({"cached_tokens": r} if r else {})}
               for u, r, w, o in usages]
    trajectory = _trajectory(tmp_path / "trajectory.json", metrics)
    curve = costs.token_curve(trajectory, costs.load_ledger(ledger.path)["run-1"])
    assert curve["source"] == "ledger" and curve["model"] == "claude-sonnet-4-6"
    assert curve["tok"][-1].tolist() == [28, 3490, 2894, 810]


# ---------------------------------------------------------------------------------------------------- runner
FAKE_HARBOR = r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[0] == "run" and args[1] == "-c" and "-y" in args, args
config = json.loads(Path(args[2]).read_text())
task = config["datasets"][0]["task_names"][0]
run_id, _, key_task = os.environ["OPENAI_API_KEY"].partition("|")
assert key_task == task and run_id and "ANTHROPIC_API_KEY" not in os.environ
counter = Path(config["jobs_dir"]) / "invocations.txt"
counter.parent.mkdir(parents=True, exist_ok=True)
with counter.open("a") as stream:
    stream.write(task + "\n")
trial = Path(config["jobs_dir"]) / config["job_name"] / (task[:32].rstrip("_-") + "__AbCdEfG")
(trial / "agent").mkdir(parents=True)
(trial / "config.json").write_text(json.dumps(config))
steps = [{"source": "user", "message": "task"},
         {"source": "agent", "message": "m", "tool_calls": [{"function_name": "bash_command"}],
          "metrics": {"prompt_tokens": 10, "completion_tokens": 5}},
         {"source": "agent", "message": "done"}]
(trial / "agent" / "trajectory.json").write_text(json.dumps({"steps": steps}))
exception = {"task-ratelimit": "RateLimitError: litellm.RateLimitError: upstream 429: Rate limit reached",
             "task-valve": "APIError: Error code: 402 - budget exhausted: spend valve reached",
             "task-env": "RuntimeError: tmux server died: no server running"}.get(task)
result = {"verifier_result": None if exception else {"rewards": {"reward": 1.0 if task == "task-pass" else 0.0}},
          "exception_info": {"exception_type": "Error", "exception_message": exception} if exception else None}
(trial / "result.json").write_text(json.dumps(result))
if not exception:
    (trial / "verifier").mkdir()
    (trial / "verifier" / "reward.txt").write_text("1" if task == "task-pass" else "0")
'''


class FakeGatewayClient(hr.GatewayClient):
    def __init__(self):
        super().__init__("http://gateway.invalid:1")

    def health(self):
        return {"ok": True, "executors": ["luna"], "ledger": "/data/ledger.jsonl"}

    def run_totals(self, run_id):
        return {"n_calls": 3, "cost_usd": 0.25}


def test_runner_records_and_resume(tmp_path):
    dataset = tmp_path / "dataset"
    tasks = ["task-env", "task-fail", "task-pass", "task-ratelimit", "task-valve"]
    for task in tasks:
        (dataset / task).mkdir(parents=True)
        (dataset / task / "task.toml").write_text("")
    (dataset / "not-a-task").mkdir()
    fake = tmp_path / "fake_harbor.py"
    fake.write_text(FAKE_HARBOR)
    assert hr.select_tasks(dataset, "toy") == tasks

    def collect():
        records = hr.collect(EXECUTORS["luna"], bench="toy", dataset_dir=dataset, tasks=tasks,
                             out_dir=tmp_path / "out", gateway=FakeGatewayClient(),
                             harbor_cmd=[sys.executable, str(fake)], concurrency=2)
        return {r["task"]: r for r in records}

    first = collect()
    arm = tmp_path / "out" / "toy" / "luna"
    passed = json.loads((arm / "task-pass.json").read_text())
    assert passed["reward"] == 1.0 and passed["end"] == "done" and passed["kind"] == "verdict"
    assert (passed["bench"], passed["worker"], passed["model"]) == ("toy", "luna", "gpt-5.6-luna")
    assert (passed["n_steps"], passed["n_tool_steps"], passed["n_calls"], passed["cost_usd"]) == (3, 1, 3, 0.25)
    assert passed["ledger"] == "/data/ledger.jsonl" and Path(passed["trajectory"]).is_file()
    assert passed["run_id"].startswith("toy-luna-") and passed["attempts"] == 1
    assert first["task-fail"]["reward"] == 0.0
    assert first["task-valve"]["end"] == "cap" and first["task-valve"]["reward"] is None
    assert first["task-env"]["end"] == "env-error"
    assert first["task-ratelimit"]["end"] == "rate-limited" and not (arm / "task-ratelimit.json").exists()

    harbor_config = json.loads((Path(passed["trial_dir"]) / "config.json").read_text())
    assert harbor_config["agents"] == [{"name": "terminus-2", "model_name": "openai/luna", "override_timeout_sec": 7200,
                                        "kwargs": {"api_base": "http://gateway.invalid:1/v1", "max_turns": 100,
                                                   "parser_name": "json"}}]
    assert harbor_config["verifier"] == {"override_timeout_sec": 7200}
    assert harbor_config["datasets"] == [{"path": str(dataset.resolve()), "task_names": ["task-pass"]}]

    second = collect()                       # resume: only the env error and the rate-limited task run again
    assert second["task-env"]["end"] == "env-error-final" and second["task-env"]["attempts"] == 2
    invocations = (tmp_path / "out" / "jobs" / "invocations.txt").read_text().split()
    assert sorted(invocations) == sorted(tasks + ["task-env", "task-ratelimit"])
    index = [json.loads(line) for line in (arm / "runs.jsonl").read_text().splitlines()]
    assert [r["task"] for r in index] == ["task-env", "task-fail", "task-pass", "task-valve"]


def test_task_subsets_are_nested(tmp_path):
    for i in range(10):
        (tmp_path / f"t{i}").mkdir()
        (tmp_path / f"t{i}" / "task.toml").write_text("")
    small = hr.select_tasks(tmp_path, "bench", n=3, seed=1)
    large = hr.select_tasks(tmp_path, "bench", n=6, seed=1)
    assert len(small) == 3 and set(small) <= set(large) and small == sorted(small)
    listed = tmp_path / "tasks.txt"
    listed.write_text("t3\nt1\n")
    assert hr.select_tasks(tmp_path, "bench", tasks_file=listed) == ["t1", "t3"]


def test_classify_registry_throttling(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"verifier_result": None, "exception_info": {
        "exception_type": "RuntimeError",
        "exception_message": "unexpected status from HEAD request to https://registry-1.docker.io/v2/x: "
                             "429 Too Many Requests"}}))
    assert hr.classify(tmp_path)[0] == "registry"
    assert hr.classify(tmp_path / "missing") == ("no-result", "")


def test_cli_help_works_without_harbor():
    for cmd in ([sys.executable, str(ROOT / "scripts" / "collect_trajectories.py"), "--help"],
                [sys.executable, "-m", "stoploss.collect.gateway", "--help"]):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=120)
        assert result.returncode == 0, result.stderr
        assert "usage" in result.stdout
