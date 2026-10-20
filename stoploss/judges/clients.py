"""Streaming clients for the two judge APIs (paper App. A, "Executor and judge configurations").

Claude judges use the Anthropic Messages API and GPT judges the OpenAI Responses API, both streamed. The request
bodies are the ones used for the paper's measurements:

    Messages:  {"model", "max_tokens", "messages": [one user message], "stream": true,
                "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
    Responses: {"model", "instructions": "", "input": [one user message], "tools": [], "tool_choice": "auto",
                "parallel_tool_calls": false, "store": false, "stream": true, "reasoning": {"effort": "high"}}

No output cap is sent to the Responses API: the paper's GPT judges had no per-call output limit. Both clients return
the same result shape:

    content          the answer text (thinking / reasoning text is not included)
    finish           "stop" for normal termination, "length" for a truncated answer; anything else is passed through
                     (Messages: other stop reasons verbatim; Responses: "incomplete:<reason>")
    usage            normalised token usage. Messages: prompt_tokens (uncached input), completion_tokens (incl.
                     thinking), cache_creation_input_tokens, cache_read_input_tokens. Responses: prompt_tokens (total
                     input), completion_tokens (incl. reasoning), prompt_tokens_details.cached_tokens,
                     completion_tokens_details.reasoning_tokens
    backend_model    the model id reported by the API
    thinking_blocks  number of thinking blocks (Messages; None for Responses)
    reasoning_tokens reasoning tokens (Responses; None for Messages)
    backend_usage    the usage object exactly as the API reported it
    request_sha256   first 16 hex digits of sha256 of the request body bytes
    latency_s        wall time of the request including the whole stream

Endpoints default to the official APIs; a base URL can be overridden (environment ANTHROPIC_BASE_URL /
OPENAI_BASE_URL, or the ``base_url`` argument). Keys come from ANTHROPIC_API_KEY / OPENAI_API_KEY.
Only the Python standard library (urllib) is used.

Errors: rejected credentials (HTTP 401/403) and an exhausted account (billing / insufficient quota) raise
``FatalAPIError``; everything else (rate limits, overload, server and network errors, a stream without its terminal
event, an error event inside the stream) raises ``APIError`` or a network exception and is retried by the runner,
which honours ``APIError.retry_after`` (from the Retry-After header).
"""
from __future__ import annotations

import email.utils
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
ERROR_TEXT_CHARS = 800

AUTH_ERROR_TYPES = ("authentication_error", "permission_error", "invalid_api_key")
QUOTA_ERROR_TYPES = ("billing_error", "insufficient_quota")
QUOTA_ERROR_TEXT = ("insufficient_quota", "credit balance is too low")


class APIError(RuntimeError):
    """A failed request that may succeed when retried."""

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class FatalAPIError(RuntimeError):
    """A failure that retrying cannot fix. ``kind``: "auth" (credentials rejected) or "quota" (account exhausted)."""

    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


# ---------------------------------------------------------------------------------------------------- request bodies
def messages_body(judge: dict, prompt: str, max_tokens: int) -> dict:
    body = {"model": judge["model"], "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}], "stream": True}
    if judge.get("thinking") is not None:
        body["thinking"] = judge["thinking"]
    if judge.get("output_config") is not None:
        body["output_config"] = judge["output_config"]
    return body


def responses_body(judge: dict, prompt: str) -> dict:
    return {"model": judge["model"], "instructions": "",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "tools": [], "tool_choice": "auto", "parallel_tool_calls": False, "store": False, "stream": True,
            "reasoning": dict(judge.get("reasoning") or {})}


def encode_body(body: dict) -> tuple[bytes, str]:
    """(request bytes, request_sha256). The byte encoding is the one the paper's ledgers were hashed with."""
    raw = json.dumps(body).encode()
    return raw, hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------- errors
def retry_after_seconds(headers) -> float | None:
    """Seconds requested by a Retry-After (or retry-after-ms) header; None if absent or unreadable."""
    if headers is None:
        return None
    millis = headers.get("retry-after-ms")
    if millis:
        try:
            return max(0.0, float(millis) / 1000.0)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _error_type(text: str) -> str:
    try:
        error = json.loads(text).get("error")
    except (ValueError, AttributeError):
        return ""
    if not isinstance(error, dict):
        return ""
    return str(error.get("type") or error.get("code") or "").lower()


def classify_error(status: int | None, text: str, headers=None) -> Exception:
    """The exception for a failed request (HTTP status, or None for an error event inside the stream)."""
    message = f"HTTP {status}: {text}" if status is not None else f"stream error: {text}"
    error_type = _error_type(text)
    if status in (401, 403) or error_type in AUTH_ERROR_TYPES:
        return FatalAPIError("auth", message, status)
    if status == 402 or error_type in QUOTA_ERROR_TYPES or any(marker in text.lower() for marker in QUOTA_ERROR_TEXT):
        return FatalAPIError("quota", message, status)
    return APIError(message, status, retry_after_seconds(headers))


def _stream_error(error) -> Exception:
    return classify_error(None, json.dumps({"error": error})[:ERROR_TEXT_CHARS])


# ---------------------------------------------------------------------------------------------------- transport
def endpoint(base_url: str, path: str) -> str:
    """Join a base URL with "/messages" or "/responses"; the base may end with or without "/v1"."""
    base = base_url.rstrip("/")
    if base.endswith("/v1" + path):
        return base
    if base.endswith("/v1"):
        return base + path
    return base + "/v1" + path


def post_stream(url: str, body: dict, headers: dict, timeout: float):
    """POST a JSON body; returns (open streaming response, request_sha256) or raises a classified error."""
    raw, request_sha = encode_body(body)
    request = urllib.request.Request(url, data=raw, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(request, timeout=timeout), request_sha
    except urllib.error.HTTPError as e:
        text = e.read()[:ERROR_TEXT_CHARS].decode(errors="replace")
        raise classify_error(e.code, text, e.headers) from None


def sse_events(lines):
    """JSON payloads of the "data:" lines of a server-sent-event stream (both APIs put the event type inside)."""
    for raw in lines:
        line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


# ---------------------------------------------------------------------------------------------------- stream readers
def read_messages_stream(events) -> dict:
    """Result of one Anthropic Messages stream (without request_sha256 / latency_s)."""
    usage, text, blocks, finish, model = {}, [], 0, None, None
    for event in events:
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message") or {}
            model = message.get("model")
            usage.update({k: v for k, v in (message.get("usage") or {}).items() if v is not None})
        elif kind == "content_block_start":
            blocks += (event.get("content_block") or {}).get("type") in ("thinking", "redacted_thinking")
        elif kind == "content_block_delta" and (event.get("delta") or {}).get("type") == "text_delta":
            text.append(event["delta"].get("text", ""))
        elif kind == "message_delta":
            # usage in message_delta is cumulative; a null field must not erase the count from message_start
            usage.update({k: v for k, v in (event.get("usage") or {}).items() if v is not None})
            finish = (event.get("delta") or {}).get("stop_reason", finish)
        elif kind == "error":
            raise _stream_error(event.get("error") or event)
    if finish is None:
        raise APIError("stream ended without a stop_reason (no terminal event)")
    normalised = {"prompt_tokens": int(usage.get("input_tokens") or 0),
                  "completion_tokens": int(usage.get("output_tokens") or 0),
                  "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
                  "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0)}
    return {"content": "".join(text), "finish": {"end_turn": "stop", "max_tokens": "length"}.get(finish, finish),
            "usage": normalised, "backend_model": model, "thinking_blocks": blocks, "reasoning_tokens": None,
            "backend_usage": usage}


def read_responses_stream(events) -> dict:
    """Result of one OpenAI Responses stream (without request_sha256 / latency_s)."""
    deltas, done, terminal, response = [], [], None, {}
    for event in events:
        kind = event.get("type", "")
        if kind == "response.output_text.delta":
            deltas.append(event.get("delta", ""))
        elif kind == "response.output_text.done":
            done.append(event.get("text", ""))
        elif kind in ("response.completed", "response.incomplete"):
            terminal, response = kind, event.get("response") or {}
        elif kind in ("response.failed", "error", "response.error"):
            raise _stream_error((event.get("response") or {}).get("error") or event.get("error") or event)
    if terminal is None:
        raise APIError("stream ended without response.completed / response.incomplete")
    # the completed event's output list can be empty; the streamed text events are preferred
    text = "".join(done) or "".join(deltas) or "".join(
        part.get("text", "") for item in response.get("output") or [] for part in item.get("content") or []
        if part.get("type") == "output_text")
    raw_usage = response.get("usage") or {}
    reason = (response.get("incomplete_details") or {}).get("reason")
    if terminal == "response.completed":
        finish = "stop"
    else:
        finish = "length" if reason == "max_output_tokens" else f"incomplete:{reason}"
    usage = {"prompt_tokens": int(raw_usage.get("input_tokens") or 0),
             "completion_tokens": int(raw_usage.get("output_tokens") or 0),
             "prompt_tokens_details": {
                 "cached_tokens": int((raw_usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)},
             "completion_tokens_details": {
                 "reasoning_tokens": int((raw_usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)}}
    return {"content": text, "finish": finish, "usage": usage, "backend_model": response.get("model"),
            "thinking_blocks": None, "reasoning_tokens": usage["completion_tokens_details"]["reasoning_tokens"],
            "backend_usage": raw_usage}


# ---------------------------------------------------------------------------------------------------- clients
class _StreamingClient:
    """Callable ``client(prompt, max_tokens) -> result``; one instance per judge, safe to share between threads."""

    base_env: str
    key_env: str
    default_base: str
    path: str

    def __init__(self, judge: dict, base_url: str | None = None, api_key: str | None = None, timeout: float = 600.0):
        self.judge = judge
        self.url = endpoint(base_url or os.environ.get(self.base_env) or self.default_base, self.path)
        self._api_key = api_key if api_key is not None else os.environ.get(self.key_env)
        if not self._api_key:
            raise ValueError(f"no API key: set {self.key_env} (judge {judge['judge_id']!r} uses {self.url})")
        self.timeout = timeout

    def _headers(self) -> dict:
        raise NotImplementedError

    def _body(self, prompt: str, max_tokens: int | None) -> dict:
        raise NotImplementedError

    def _read(self, events) -> dict:
        raise NotImplementedError

    def __call__(self, prompt: str, max_tokens: int | None) -> dict:
        start = time.time()
        response, request_sha = post_stream(self.url, self._body(prompt, max_tokens), self._headers(), self.timeout)
        with response:
            result = self._read(sse_events(response))
        result.update(request_sha256=request_sha, latency_s=round(time.time() - start, 2))
        return result


class MessagesClient(_StreamingClient):
    """Anthropic Messages API (Claude judges)."""

    base_env, key_env, default_base, path = "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", ANTHROPIC_BASE_URL, "/messages"

    def _headers(self) -> dict:
        return {"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION}

    def _body(self, prompt, max_tokens):
        return messages_body(self.judge, prompt, max_tokens)

    def _read(self, events):
        return read_messages_stream(events)


class ResponsesClient(_StreamingClient):
    """OpenAI Responses API (GPT judges). ``max_tokens`` is ignored: no output cap is sent."""

    base_env, key_env, default_base, path = "OPENAI_BASE_URL", "OPENAI_API_KEY", OPENAI_BASE_URL, "/responses"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _body(self, prompt, max_tokens):
        return responses_body(self.judge, prompt)

    def _read(self, events):
        return read_responses_stream(events)


CLIENTS = {"messages": MessagesClient, "responses": ResponsesClient}


def make_client(judge: dict, base_url: str | None = None, api_key: str | None = None, timeout: float = 600.0):
    """The streaming client for ``judge["api"]``."""
    if judge.get("api") not in CLIENTS:
        raise ValueError(f"judge {judge.get('judge_id')!r}: unknown api {judge.get('api')!r} "
                         f"(expected one of {sorted(CLIENTS)})")
    return CLIENTS[judge["api"]](judge, base_url=base_url, api_key=api_key, timeout=timeout)
