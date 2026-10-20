"""Tests of the paid-judge baseline (stoploss.judges): synthetic data and fake clients only, no network."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from stoploss.judges import (JUDGE_SUFFIX, is_checkpoint, judge_config_sha, judge_prompt, load_judges, load_package,
                             parse_answer, row_problems, run_judge, score_window, sha16, valid_scores, window_costs)
from stoploss.judges.clients import (APIError, FatalAPIError, classify_error, endpoint, messages_body,
                                     read_messages_stream, read_responses_stream, responses_body, retry_after_seconds,
                                     sse_events)
from stoploss.judges.runner import MAX_REQUESTS, StopSignal, plan_windows, score_row
from stoploss.windows import QUESTION_SUFFIX, window_prompt

REPO = Path(__file__).resolve().parents[1]
JUDGES = load_judges(REPO / "configs" / "judges.json")
OPUS, SOL = JUDGES["opus5"], JUDGES["sol"]
PRICES = {"claude-opus-5": {"input": 5.0, "cache_read": 0.5, "cache_write": 6.25, "output": 25.0},
          "gpt-5.6-sol": {"input": 4.0, "cache_read": 0.4, "cache_write": None, "output": 20.0}}


# ---------------------------------------------------------------------------------------------------- helpers
def make_windows(worker, tasks, n_steps=8, bench="verified"):
    windows = []
    for task in tasks:
        pairs = [(f"{task}: command {i}", f"{task}: output {i}") for i in range(n_steps)]
        for k in range(2, n_steps + 1):
            windows.append({"bench": bench, "worker": worker, "task": task, "k": k, "label": "ABORT",
                            "prompt": window_prompt(pairs, k)})
    return windows


def answer(text="Looks fine.\nFAIL_PROBABILITY: 30", finish="stop", model="claude-opus-5", prompt_tokens=100,
           completion_tokens=50):
    return {"content": text, "finish": finish, "backend_model": model, "thinking_blocks": 1, "reasoning_tokens": None,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            "backend_usage": {}, "request_sha256": "0" * 16, "latency_s": 0.01}


class FakeClient:
    """Plays a script of results / exceptions per judge prompt; unscripted prompts get ``default()``."""

    def __init__(self, scripts=None, default=answer):
        self.scripts = {prompt: list(items) for prompt, items in (scripts or {}).items()}
        self.default = default
        self.requests = []
        self.lock = threading.Lock()

    def __call__(self, prompt, max_tokens):
        with self.lock:
            self.requests.append((prompt, max_tokens))
            script = self.scripts.get(prompt)
            item = script.pop(0) if script else self.default()
        if isinstance(item, Exception):
            raise item
        return item


def quiet(**kwargs):
    return dict(sleep=lambda seconds: None, log=lambda *args: None, threads=1, **kwargs)


# ---------------------------------------------------------------------------------------------------- prompt
def test_judge_prompt_replaces_exactly_the_monitor_question():
    prompt = window_prompt([("ls", "a.py"), ("cat a.py", "print(1)")], 2)
    sent = judge_prompt(prompt)
    assert sent == prompt[: -len(QUESTION_SUFFIX)] + JUDGE_SUFFIX
    assert sent.endswith("FAIL_PROBABILITY: <integer 0-100>") and "Answer with one word" not in sent
    assert sha16(JUDGE_SUFFIX) == "f7958e22063781b5"      # suffix fingerprint recorded with the paper's judge scores


@pytest.mark.parametrize("bad", ["no question at all", "QUESTION" + QUESTION_SUFFIX + " ", QUESTION_SUFFIX[:-1],
                                 QUESTION_SUFFIX + "\n", QUESTION_SUFFIX.strip()])
def test_judge_prompt_refuses_another_tail(bad):
    with pytest.raises(ValueError):
        judge_prompt(bad)


@pytest.mark.parametrize("text, expected", [
    ("Reasoning.\nFAIL_PROBABILITY: 35", (35, "tag")),
    ("FAIL_PROBABILITY: 10\nOn reflection:\nFAIL_PROBABILITY: 70", (70, "tag")),     # the last tag wins
    ("FAIL_PROBABILITY: **80**", (80, "tag")),
    ("**FAIL_PROBABILITY:** 12", (12, "tag")),
    ("FAIL_PROBABILITY\uff1a 7", (7, "tag")),                                        # fullwidth colon
    ("FAIL_PROBABILITY : 5", (5, "tag")),
    ("FAIL_PROBABILITY: 250", (100, "tag")),                                          # clamped
    ("FAIL_PROBABILITY: 1000", (100, "tag")),
    ("Maybe 30% at first, now 45 %.", (45, "percent_fallback")),
    ("risk 400%", (100, "percent_fallback")),
    ("**FAIL_PROBABILITY**: 12", (None, "unparsed")),                                 # bold tag name is not the tag
    ("no number here", (None, "unparsed")),
    ("", (None, "no_text")),
    (None, (None, "no_text")),
])
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


# ---------------------------------------------------------------------------------------------------- configuration
def test_config_holds_the_paper_judges():
    assert sorted(JUDGES) == ["luna", "opus46", "opus5", "sol", "sonnet46", "terra"]
    for judge in JUDGES.values():
        assert judge["allowed_backend_models"] == [judge["model"]]
        if judge["family"] == "gpt":
            assert judge["api"] == "responses" and judge["reasoning"] == {"effort": "high"}
            assert judge["max_tokens"] is None and judge["length_retries"] == 2 and judge["workers"] == ["luna", "terra"]
        else:
            assert judge["api"] == "messages" and judge["thinking"] == {"type": "adaptive"}
            assert judge["output_config"] == {"effort": "high"} and judge["max_tokens_ladder"] == [4000, 8000, 16000]
            assert judge["workers"] == ["sonnet46t"]


def test_config_sha_covers_the_request_not_the_acceptance_lists():
    base = judge_config_sha(OPUS)
    assert len(base) == 16
    assert judge_config_sha(dict(OPUS, allowed_backend_models=["claude-opus-5", "snapshot"], workers=["x"])) == base
    assert judge_config_sha(dict(OPUS, output_config={"effort": "low"})) != base
    assert judge_config_sha(dict(OPUS, max_tokens_ladder=[4000, 8000])) != base


def test_plan_windows_checkpoints_family_and_duplicates():
    windows = make_windows("sonnet46t", ["t1"], n_steps=12) + make_windows("luna", ["t2"], n_steps=12)
    assert [k for k in range(1, 30) if is_checkpoint(k)][:5] == [5, 8, 11, 14, 17]
    planned, other = plan_windows(windows, OPUS)
    assert [w["k"] for w in planned] == [5, 8, 11] and other == {"luna": 3}
    assert all(w["set"] == "test" for w in planned)
    assert len(plan_windows(windows, OPUS, all_windows=True)[0]) == 11
    assert len(plan_windows(windows, OPUS, limit=2)[0]) == 2
    assert plan_windows([dict(windows[3], set="cal")], OPUS)[0][0]["set"] == "cal"
    with pytest.raises(ValueError):
        plan_windows(windows + windows[3:4], OPUS)


# ---------------------------------------------------------------------------------------------------- requests, streams
def test_request_bodies_are_the_papers():
    assert messages_body(OPUS, "hi", 8000) == {
        "model": "claude-opus-5", "max_tokens": 8000, "messages": [{"role": "user", "content": "hi"}], "stream": True,
        "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
    body = responses_body(SOL, "hi")
    assert body == {
        "model": "gpt-5.6-sol", "instructions": "",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "tools": [], "tool_choice": "auto", "parallel_tool_calls": False, "store": False, "stream": True,
        "reasoning": {"effort": "high"}}
    assert list(body) == ["model", "instructions", "input", "tools", "tool_choice", "parallel_tool_calls", "store",
                          "stream", "reasoning"]          # key order fixes the bytes behind request_sha256


def test_endpoint_accepts_base_urls_with_or_without_v1():
    assert endpoint("https://api.anthropic.com", "/messages") == "https://api.anthropic.com/v1/messages"
    assert endpoint("https://api.openai.com/v1", "/responses") == "https://api.openai.com/v1/responses"
    assert endpoint("http://proxy/v1/", "/messages") == "http://proxy/v1/messages"
    assert endpoint("http://proxy/v1/responses", "/responses") == "http://proxy/v1/responses"


def test_sse_events_reads_data_lines_only():
    lines = [b"event: message_start\n", b'data: {"type": "a"}\n', b"\n", b": comment\n", b"data: not json\n",
             b"data: [DONE]\n", 'data: {"type": "b"}']
    assert [e["type"] for e in sse_events(lines)] == ["a", "b"]


def test_messages_stream_normalisation():
    events = [{"type": "message_start", "message": {"model": "claude-opus-5", "usage": {
                  "input_tokens": 12, "cache_creation_input_tokens": 1700, "cache_read_input_tokens": 0, "output_tokens": 1}}},
              {"type": "content_block_start", "content_block": {"type": "thinking"}},
              {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "secret"}},
              {"type": "content_block_start", "content_block": {"type": "text"}},
              {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "FAIL_PROBABILITY: "}},
              {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "40"}},
              {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
               "usage": {"input_tokens": None, "output_tokens": 400}}]
    result = read_messages_stream(events)
    assert result["content"] == "FAIL_PROBABILITY: 40" and result["finish"] == "stop"
    assert result["usage"] == {"prompt_tokens": 12, "completion_tokens": 400, "cache_creation_input_tokens": 1700,
                               "cache_read_input_tokens": 0}
    assert result["backend_model"] == "claude-opus-5" and result["thinking_blocks"] == 1 and result["reasoning_tokens"] is None
    events[-1] = {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 4000}}
    assert read_messages_stream(events)["finish"] == "length"
    events[-1] = {"type": "message_delta", "delta": {"stop_reason": "refusal"}, "usage": {"output_tokens": 5}}
    assert read_messages_stream(events)["finish"] == "refusal"
    with pytest.raises(APIError):                       # no terminal event
        read_messages_stream(events[:-1])
    with pytest.raises(APIError):
        read_messages_stream(events[:2] + [{"type": "error", "error": {"type": "overloaded_error"}}])
    with pytest.raises(FatalAPIError):
        read_messages_stream([{"type": "error", "error": {"type": "authentication_error"}}])


def test_responses_stream_normalisation():
    usage = {"input_tokens": 1243, "input_tokens_details": {"cached_tokens": 200}, "output_tokens": 1067,
             "output_tokens_details": {"reasoning_tokens": 977}}
    done = {"type": "response.completed", "response": {"model": "gpt-5.6-sol", "usage": usage, "output": []}}
    result = read_responses_stream([{"type": "response.output_text.delta", "delta": "FAIL_PROBABILITY: 2"},
                                    {"type": "response.output_text.delta", "delta": "5"}, done])
    assert result["content"] == "FAIL_PROBABILITY: 25" and result["finish"] == "stop" and result["reasoning_tokens"] == 977
    assert result["usage"] == {"prompt_tokens": 1243, "completion_tokens": 1067,
                               "prompt_tokens_details": {"cached_tokens": 200},
                               "completion_tokens_details": {"reasoning_tokens": 977}}
    with_output = {"type": "response.completed", "response": {"model": "m", "output": [
        {"type": "reasoning", "content": None},
        {"type": "message", "content": [{"type": "output_text", "text": "FAIL_PROBABILITY: 9"}]}]}}
    assert read_responses_stream([with_output])["content"] == "FAIL_PROBABILITY: 9"
    incomplete = {"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}
    assert read_responses_stream([incomplete])["finish"] == "length"
    incomplete["response"]["incomplete_details"]["reason"] = "content_filter"
    assert read_responses_stream([incomplete])["finish"] == "incomplete:content_filter"
    with pytest.raises(APIError):
        read_responses_stream([{"type": "response.output_text.delta", "delta": "x"}])
    with pytest.raises(APIError):
        read_responses_stream([{"type": "response.failed", "response": {"error": {"code": "server_error"}}}])


def test_error_classification_and_retry_after():
    assert isinstance(classify_error(401, "{}"), FatalAPIError) and classify_error(403, "{}").kind == "auth"
    quota = '{"error": {"type": "insufficient_quota", "code": "insufficient_quota"}}'
    assert classify_error(429, quota).kind == "quota"
    assert classify_error(400, '{"error": {"message": "Your credit balance is too low"}}').kind == "quota"
    limited = classify_error(429, '{"error": {"type": "rate_limit_error"}}', {"retry-after": "12"})
    assert isinstance(limited, APIError) and limited.retry_after == 12.0
    assert classify_error(529, '{"error": {"type": "overloaded_error"}}').retry_after is None
    assert retry_after_seconds({"retry-after-ms": "1500"}) == 1.5
    assert retry_after_seconds({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}) == 0.0   # a date in the past
    assert retry_after_seconds({"retry-after": "soon"}) is None and retry_after_seconds(None) is None


# ---------------------------------------------------------------------------------------------------- retry ladder
WINDOW = make_windows("sonnet46t", ["t1"])[3]          # k = 5


def test_truncation_climbs_the_max_tokens_ladder():
    client = FakeClient(default=iter([answer("FAIL_PROBABILITY: 9", finish="length"), answer("...", finish="length"),
                                      answer("FAIL_PROBABILITY: 61")]).__next__)
    final, attempts, error = score_window(WINDOW, OPUS, client)
    assert final[0] == 61 and [a["max_tokens"] for a in attempts] == [4000, 8000, 16000]
    assert [a["answer"] for a in attempts] == [9, None, 61]          # a truncated number is recorded, never accepted


def test_truncation_gives_up_at_the_top_of_the_ladder():
    client = FakeClient(default=lambda: answer("FAIL_PROBABILITY: 9", finish="length"))
    final, attempts, error = score_window(WINDOW, OPUS, client)
    assert final is None and len(attempts) == 3 and error == "truncated (finish=length)"


def test_gpt_truncation_is_retried_without_a_cap():
    gpt_window = make_windows("luna", ["t1"])[3]
    client = FakeClient(default=lambda: answer(finish="length", model="gpt-5.6-sol"))
    final, attempts, error = score_window(gpt_window, SOL, client)
    assert final is None and [a["max_tokens"] for a in attempts] == [None, None, None]    # 1 + length_retries
    client = FakeClient(default=iter([answer(finish="length", model="gpt-5.6-sol"), answer(model="gpt-5.6-sol")]).__next__)
    final, attempts, error = score_window(gpt_window, SOL, client)
    assert final[0] == 30 and len(attempts) == 2


def test_unparsed_or_abnormal_answer_is_retried_once():
    client = FakeClient(default=lambda: answer("I cannot tell."))
    final, attempts, error = score_window(WINDOW, OPUS, client)
    assert final is None and len(attempts) == 2 and error == "finish=stop parse=unparsed"
    client = FakeClient(default=iter([answer("I cannot tell."), answer("FAIL_PROBABILITY: 5")]).__next__)
    assert score_window(WINDOW, OPUS, client)[0][0] == 5
    client = FakeClient(default=iter([answer(finish="length"), answer(finish="refusal")]).__next__)
    final, attempts, error = score_window(WINDOW, OPUS, client)      # not the first request: no further retry
    assert final is None and len(attempts) == 2 and error == "finish=refusal parse=tag"


def test_request_errors_are_retried_with_backoff():
    sleeps = []
    client = FakeClient(default=iter([APIError("HTTP 500: boom"), APIError("HTTP 429: slow", 429, retry_after=30.0),
                                      answer()]).__next__)
    final, attempts, error = score_window(WINDOW, OPUS, client, sleep=sleeps.append)
    assert final[0] == 30 and len(attempts) == 3 and sleeps == [4.0, 30.0]
    assert attempts[0] == {"max_tokens": 4000, "error": "HTTP 500: boom", "usage_unknown": True}
    sleeps.clear()
    client = FakeClient(default=lambda: ConnectionResetError("reset"))
    final, attempts, error = score_window(WINDOW, OPUS, client, sleep=sleeps.append)
    assert final is None and len(attempts) == MAX_REQUESTS and len(sleeps) == MAX_REQUESTS - 1 and error == "reset"


def test_wrong_backend_model_and_fatal_errors_stop_the_run():
    stop = StopSignal()
    final, attempts, error = score_window(WINDOW, OPUS, FakeClient(default=lambda: answer(model="claude-other")), stop=stop)
    assert final is None and error == "backend_model" and len(attempts) == 1 and stop.kind == "backend_model"
    assert score_window(WINDOW, OPUS, FakeClient(), stop=stop) == (None, [], "stopped")
    stop = StopSignal()
    client = FakeClient(default=lambda: FatalAPIError("auth", "HTTP 401: bad key", 401))
    final, attempts, error = score_window(WINDOW, OPUS, client, stop=stop)
    assert error == "auth" and stop.kind == "auth"
    assert attempts == [{"max_tokens": 4000, "error": "auth", "usage_unknown": True}]


# ---------------------------------------------------------------------------------------------------- validity rule
def valid_row():
    final, attempts, error = score_window(WINDOW, OPUS, FakeClient())
    return score_row(dict(WINDOW, set="test"), OPUS, judge_config_sha(OPUS), final, attempts, error)


def test_row_problems():
    row = valid_row()
    assert row["p"] == 0.3 and row_problems(row, OPUS, judge_config_sha(OPUS), sha16(WINDOW["prompt"]),
                                            sha16(judge_prompt(WINDOW["prompt"]))) == []
    defects = {"finish": "length", "backend_model": "claude-other", "worker": "luna", "judge_id": "sol", "model": "x",
               "p": 1.5, "usage_total": {"prompt_tokens": 10, "completion_tokens": 0}}
    for field, value in defects.items():
        assert row_problems(dict(row, **{field: value}), OPUS), field
    assert row_problems(dict(row, p=True), OPUS) and row_problems(dict(row, p=float("nan")), OPUS)
    assert row_problems(row, OPUS, config_sha="0" * 16) == ["config_sha '" + row["config_sha"] + "'"]
    assert row_problems(row, OPUS, prompt_sha="0" * 16) == ["prompt_sha256"]


def test_a_response_without_usage_is_not_written_as_a_score():
    final, attempts, error = score_window(WINDOW, OPUS, FakeClient(default=lambda: answer(prompt_tokens=0)))
    row = score_row(dict(WINDOW, set="test"), OPUS, judge_config_sha(OPUS), final, attempts, error)
    assert row["p"] is None and "usage missing" in row["error"]


# ---------------------------------------------------------------------------------------------------- packages
def test_run_judge_resumes_and_skips_valid_windows(tmp_path):
    windows = make_windows("sonnet46t", ["t1", "t2"]) + make_windows("luna", ["t3"])
    t2_k5 = judge_prompt(windows[10]["prompt"])
    assert windows[10]["task"] == "t2" and windows[10]["k"] == 5
    first = FakeClient(scripts={t2_k5: [answer("no idea"), answer("still no idea")]})
    summary = run_judge(windows, OPUS, tmp_path, client=first, **quiet())
    assert summary["n_planned"] == 4 and summary["n_valid"] == 3 and summary["exit_code"] == 1
    assert len(first.requests) == 5
    second = FakeClient()
    summary = run_judge(windows, OPUS, tmp_path, client=second, **quiet())
    assert [prompt for prompt, _ in second.requests] == [t2_k5] and summary["exit_code"] == 0
    scores = valid_scores(tmp_path / "scores.jsonl", OPUS)
    assert len(scores) == 4 and set(scores.values()) == {0.3}
    assert len((tmp_path / "scores.jsonl").read_text().splitlines()) == 5           # the failed row stays as a record
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 6
    third = FakeClient()
    assert run_judge(windows, OPUS, tmp_path, client=third, **quiet())["exit_code"] == 0 and not third.requests


def test_run_judge_stops_and_reports(tmp_path):
    windows = make_windows("sonnet46t", ["t1", "t2"])
    summary = run_judge(windows, OPUS, tmp_path / "a", client=FakeClient(default=lambda: answer(model="other")), **quiet())
    assert summary["exit_code"] == 4 and summary["stopped"] == "backend_model" and summary["requests_now"] == 1
    client = FakeClient(default=lambda: FatalAPIError("quota", "HTTP 429: insufficient_quota", 429))
    assert run_judge(windows, OPUS, tmp_path / "b", client=client, **quiet())["exit_code"] == 3
    summary = run_judge(windows, OPUS, tmp_path / "c", client=FakeClient(default=lambda: answer("?")),
                        max_failed_streak=2, **quiet())
    assert summary["stopped"] == "failures" and summary["windows_scored_now"] == 2 and summary["exit_code"] == 3


def test_a_window_without_the_monitor_question_is_refused_before_any_request(tmp_path):
    windows = make_windows("sonnet46t", ["t1"])
    windows[3] = dict(windows[3], prompt=windows[3]["prompt"] + " ")
    client = FakeClient()
    with pytest.raises(ValueError):
        run_judge(windows, OPUS, tmp_path, client=client, **quiet())
    assert not client.requests and not (tmp_path / "scores.jsonl").exists()


def test_an_unexpected_error_stops_the_run(tmp_path):
    messages = []
    broken = FakeClient(default=lambda: dict(answer(), usage={"prompt_tokens": {1}}))     # not JSON-serialisable
    summary = run_judge(make_windows("sonnet46t", ["t1"]), OPUS, tmp_path, client=broken,
                        sleep=lambda s: None, log=messages.append, threads=1)
    assert summary["exit_code"] == 3 and summary["stopped"] == "error" and "TypeError" in summary["stop_message"]
    assert any("Traceback" in str(m) for m in messages)


def test_a_package_is_never_mixed(tmp_path):
    windows = make_windows("sonnet46t", ["t1"])
    run_judge(windows, OPUS, tmp_path, client=FakeClient(), **quiet())
    with pytest.raises(ValueError):
        run_judge(windows, dict(OPUS, output_config={"effort": "low"}), tmp_path, client=FakeClient(), **quiet())
    changed = [dict(w, prompt=w["prompt"].replace("command", "cmd")) for w in windows]
    with pytest.raises(ValueError):
        run_judge(changed, OPUS, tmp_path, client=FakeClient(), **quiet())
    with pytest.raises(ValueError):
        valid_scores(tmp_path / "scores.jsonl", JUDGES["opus46"])


def test_bill_comes_from_the_ledger_including_failed_attempts(tmp_path):
    windows = make_windows("sonnet46t", ["t1"])          # checkpoints k = 5, 8
    k5, k8 = (judge_prompt(w["prompt"]) for w in windows if w["k"] in (5, 8))
    client = FakeClient(scripts={
        k5: [answer(finish="length", prompt_tokens=1000, completion_tokens=4000),
             answer(finish="length", prompt_tokens=1000, completion_tokens=8000),
             answer(prompt_tokens=1000, completion_tokens=500)],
        k8: [APIError("HTTP 529: overloaded"), answer(prompt_tokens=800, completion_tokens=300)]})
    run_judge(windows, OPUS, tmp_path, client=client, **quiet())
    costs = window_costs(tmp_path / "calls.jsonl", OPUS, PRICES)
    key5, key8 = ("test", "verified", "sonnet46t", "t1", 5), ("test", "verified", "sonnet46t", "t1", 8)
    assert costs[key5] == pytest.approx((5 * 3000 + 25 * 12500) / 1e6)       # both truncated attempts are paid
    assert costs[key8] == pytest.approx((5 * 800 + 25 * 300) / 1e6)
    assert costs.calls == {key5: 3, key8: 2} and costs.unknown_usage == {key8: 1} and costs.n_unknown_usage == 1
    with open(tmp_path / "calls.jsonl", "a") as ledger:
        ledger.write(json.dumps({"call_id": "x1", "set": "test", "bench": "verified", "worker": "sonnet46t", "task": "t1",
                                 "k": 5, "judge_id": "opus5", "config_sha": "another", "usage": {"prompt_tokens": 9}}) + "\n")
        ledger.write('{"call_id": "torn')
    costs_after = window_costs(tmp_path / "calls.jsonl", OPUS, PRICES)
    assert dict(costs_after) == dict(costs) and costs_after.other_calls == 1 and costs_after.unreadable_lines == 1
    scores, package_costs = load_package(tmp_path, OPUS, PRICES)
    assert scores == {key5: 0.3, key8: 0.3} and package_costs.total_usd == pytest.approx(costs.total_usd)


def test_readers_are_strict(tmp_path):
    windows = make_windows("sonnet46t", ["t1"])
    run_judge(windows, OPUS, tmp_path, client=FakeClient(), **quiet())
    path = tmp_path / "scores.jsonl"
    good = path.read_text()
    for extra in ['{"not json\n', json.dumps(dict(json.loads(good.splitlines()[0]), finish="length")) + "\n"]:
        path.write_text(good + extra)
        with pytest.raises(ValueError):
            valid_scores(path, OPUS)
    path.write_text(good + json.dumps(dict(json.loads(good.splitlines()[0]), k=11, p=None, error="x")) + "\n")
    assert len(valid_scores(path, OPUS)) == 2                                   # failed rows are records, not scores
    with pytest.raises(FileNotFoundError):
        valid_scores(tmp_path / "missing.jsonl", OPUS)


def test_a_torn_last_row_is_quarantined_on_resume(tmp_path):
    windows = make_windows("sonnet46t", ["t1"])
    run_judge(windows, OPUS, tmp_path, client=FakeClient(), **quiet())
    path = tmp_path / "scores.jsonl"
    first_row = path.read_text().splitlines()[0]
    path.write_text(first_row + "\n" + '{"set": "test", "bench": "verif')            # interrupted write
    with pytest.raises(ValueError):
        valid_scores(path, OPUS)
    client = FakeClient()
    assert run_judge(windows, OPUS, tmp_path, client=client, **quiet())["exit_code"] == 0
    assert len(client.requests) == 1 and len(valid_scores(path, OPUS)) == 2
    assert "torn last line" in (tmp_path / "scores.quarantine.jsonl").read_text()


def test_threads_score_every_window_once(tmp_path):
    windows = make_windows("sonnet46t", [f"t{i}" for i in range(12)], n_steps=14)
    client = FakeClient()
    summary = run_judge(windows, OPUS, tmp_path, client=client, sleep=lambda s: None, log=lambda *a: None, threads=4)
    assert summary["n_planned"] == 12 * 4 and summary["exit_code"] == 0 and len(client.requests) == 48
    assert len(set(prompt for prompt, _ in client.requests)) == 48
