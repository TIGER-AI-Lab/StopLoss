"""Metering gateway between Terminus-2 and the executor's provider (paper Sec. 3.1 and App. A).

Terminus-2 talks to an OpenAI-compatible ``POST /v1/chat/completions``. The gateway
  * identifies the run from the API key ``"<run_id>|<task_id>"`` (stock Harbor agents can only be given a base URL
    and a key) or from ``X-Run-Id`` / ``X-Task-Id`` headers;
  * maps the requested model (an executor label from ``configs/executors.json``) to the upstream model and
    forwards the call: OpenAI executors to the Chat Completions API, Anthropic executors to the Messages API
    through a chat -> Messages translation;
  * writes one ledger line per call with provider-neutral token counts (see ``stoploss.collect.ledger``);
  * enforces the executor's per-run spend valve: once a run's booked spend reaches the valve, further calls get
    HTTP 402. The call that crosses the valve is allowed and recorded, so a small overshoot is possible.

Request shaping follows the provider proxies used during collection: output-token caps and sampling parameters
sent by the agent are not forwarded to OpenAI (provider defaults apply; reasoning effort stays at the upstream
default unless configured), every call of a run carries the same ``prompt_cache_key``; Anthropic calls get
extended thinking with the configured budget, the executor's default output limit when the agent sends no
``max_tokens``, and ephemeral cache breakpoints on the last system block and the last message.

Run with ``python -m stoploss.collect.gateway --ledger runs/ledger.jsonl``; keys come from ``OPENAI_API_KEY`` and
``ANTHROPIC_API_KEY`` in the gateway's environment. Bind it to localhost only: anyone who can reach it can
spend your API keys.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from stoploss.collect import DEFAULT_EXECUTORS, Executor, load_executors
from stoploss.collect.ledger import Ledger
from stoploss.costs import load_prices, price_of

log = logging.getLogger("stoploss.collect.gateway")

DEFAULT_PRICES = Path(__file__).resolve().parents[2] / "configs" / "prices_20260918.json"
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

# Upstream retry policy (same as the collection-time provider proxies): up to 6 retries on rate limits, 5xx and
# transport errors, exponential backoff 2, 4, ..., 60 s (a Retry-After header can lengthen a wait), at most
# 540 s per call including all retries so the agent's own request timeout (600 s) is not hit first.
RETRY_STATUSES = (429, 500, 502, 503, 529)
MAX_RETRIES = 6
CALL_BUDGET_SEC = 540.0
REQUEST_TIMEOUT_SEC = 600.0

# The only OpenAI request fields forwarded upstream; everything else (max_tokens, temperature, top_p, stop,
# logprobs, LiteLLM's session_id, ...) is dropped.
OPENAI_FORWARDED_FIELDS = ("messages", "tools", "tool_choice", "parallel_tool_calls", "response_format")

ANTHROPIC_STOP_TO_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                            "tool_use": "tool_calls"}
EMPTY_CONVERSATION_TEXT = "Continue."


# ---------------------------------------------------------------------------------------------------- identity
def run_identity(request: Request) -> tuple[str, str | None]:
    """(run_id, task_id) from X-Run-Id / X-Task-Id, else from the API key "<run_id>|<task_id>"."""
    run_id = request.headers.get("x-run-id")
    task_id = request.headers.get("x-task-id")
    if not run_id:
        auth = request.headers.get("authorization", "")
        key = auth.split(" ", 1)[1] if " " in auth else auth
        key = (key or request.headers.get("x-api-key", "")).strip()
        run_id, _, key_task = key.partition("|")
        task_id = task_id or key_task or None
    if not run_id:
        raise HTTPException(401, "set the API key to '<run_id>|<task_id>' (or send an X-Run-Id header)")
    return run_id, task_id


# ---------------------------------------------------------------------------------------------------- accounting
def normalize_usage(usage: dict) -> dict:
    """Ledger token fields from a chat-completions usage block, for either provider's convention.

    OpenAI counts cached input INSIDE ``prompt_tokens`` (``prompt_tokens_details.cached_tokens`` is a subset), so
    the uncached part is prompt - cached. Anthropic counts cache reads and writes OUTSIDE its input tokens; the
    translated response keeps them as ``cache_read_input_tokens`` / ``cache_creation_input_tokens``.
    ``completion_tokens`` already includes reasoning / thinking tokens for both providers.
    """
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    if cached:
        cached = min(cached, prompt)
        prompt -= cached
        cache_read += cached
    return {"prompt_tokens": prompt, "cache_read_tokens": cache_read, "cache_write_tokens": cache_write,
            "input_tokens_total": prompt + cache_read + cache_write, "completion_tokens": completion}


def booked_cost_usd(prices: dict, model: str, tokens: dict) -> float:
    """Dollars the gateway books for one call (spend valve and ledger ``cost_usd``).

    Cache reads and writes are charged at their own listed prices (a model without a cache-write price is
    charged its input price), as the collection-time gateway did. Reported costs do not use this number: they are
    recomputed from the ledger token counts by ``stoploss.costs``.
    """
    p = price_of(prices, model)
    write_price = p["cache_write"] if p.get("cache_write") is not None else p["input"]
    return (tokens["prompt_tokens"] * p["input"] + tokens["cache_read_tokens"] * p["cache_read"]
            + tokens["cache_write_tokens"] * write_price + tokens["completion_tokens"] * p["output"]) / 1e6


# ---------------------------------------------------------------------------------------------------- upstream calls
async def _backoff_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


async def post_with_retries(client: httpx.AsyncClient, url: str, payload: dict, headers: dict) -> httpx.Response:
    """POST under the retry policy above; returns the final response, which may carry an error status."""
    deadline = time.monotonic() + CALL_BUDGET_SEC
    backoff = 2.0
    for attempt in range(MAX_RETRIES + 1):
        timeout = httpx.Timeout(min(REQUEST_TIMEOUT_SEC, max(deadline - time.monotonic(), 1.0)), connect=30.0)
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=timeout)
        except httpx.TransportError as exc:
            if attempt == MAX_RETRIES or time.monotonic() + backoff > deadline:
                raise HTTPException(502, f"upstream connection error: {exc!r}"[:500]) from exc
            log.warning("upstream transport error %r; retry %d/%d in %.0fs", exc, attempt + 1, MAX_RETRIES, backoff)
        else:
            if response.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                return response
            retry_after = _retry_after(response)
            if retry_after is not None:
                backoff = min(max(backoff, retry_after), 60.0)
            if time.monotonic() + backoff > deadline:
                return response
            log.warning("upstream %s; retry %d/%d in %.0fs", response.status_code, attempt + 1, MAX_RETRIES, backoff)
        await _backoff_sleep(backoff)
        backoff = min(backoff * 2, 60.0)
    raise AssertionError("unreachable")


def _upstream_json(response: httpx.Response) -> dict:
    """Body of a successful upstream response; otherwise an HTTP error for the agent.

    A rate limit stays a 429 so the agent's client sees a rate limit; every other upstream failure becomes a 502
    whose message carries the upstream status and body.
    """
    if response.status_code != 200:
        log.error("upstream %s: %s", response.status_code, response.text[:500])
        status = 429 if response.status_code == 429 else 502
        raise HTTPException(status, f"upstream {response.status_code}: {response.text[:500]}")
    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(502, f"upstream returned invalid JSON: {response.text[:200]}") from exc


def _api_key(name: str) -> str:
    key = os.environ.get(name)
    if not key:
        raise HTTPException(500, f"{name} is not set in the gateway's environment")
    return key


def openai_request(body: dict, executor: Executor, run_id: str) -> dict:
    """Chat Completions request for an OpenAI executor."""
    payload = {key: body[key] for key in OPENAI_FORWARDED_FIELDS if key in body}
    payload["model"] = executor.model
    if executor.prompt_cache_key:
        # One cache key per run: every turn replays the growing conversation prefix, and a stable key keeps those
        # turns on the same prompt-cache shard.
        payload["prompt_cache_key"] = run_id
    if executor.reasoning_effort:
        payload["reasoning_effort"] = executor.reasoning_effort
    return payload


# ---------------------------------------------------------------------------------------------------- Anthropic
def _text_of(content: Any) -> str:
    """Plain text of a chat message content: a string or a list of text parts (joined by newlines)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def _user_blocks(content: Any) -> list[dict]:
    """Anthropic text blocks for a user message: one block per non-empty text part."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for part in content if isinstance(content, list) else []:
        if isinstance(part, str):
            text = part
        elif isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
            text = part.get("text") or ""
        else:
            raise HTTPException(400, "only text content is supported for Anthropic executors")
        if text:
            blocks.append({"type": "text", "text": text})
    return blocks


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _thinking(requested: Any, budget: int | None, max_tokens: int) -> dict | None:
    """Extended-thinking field: the configured budget, kept strictly below max_tokens.

    With no configured budget, the agent's own ``thinking`` field (if any) is passed through unchanged.
    """
    if budget is None:
        return dict(requested) if isinstance(requested, dict) else None
    cap = min(budget, max(max_tokens - 1, 0))
    if cap <= 0:
        return None
    requested_budget = _as_int(requested.get("budget_tokens")) if isinstance(requested, dict) else None
    thinking = dict(requested) if isinstance(requested, dict) else {"type": "enabled"}
    thinking["budget_tokens"] = min(requested_budget, cap) if requested_budget is not None else cap
    thinking.setdefault("type", "enabled")
    return thinking


def anthropic_request(body: dict, executor: Executor) -> dict:
    """Translate an OpenAI chat body into an Anthropic Messages request (text conversations only).

    System and developer messages become system blocks; consecutive user messages are merged into one turn;
    empty assistant turns are dropped; sampling parameters are not forwarded. A missing ``max_tokens`` is filled
    with the executor's default output limit (a value sent by the agent is kept), and thinking gets the
    configured budget. Ephemeral cache breakpoints on the last system block and on the last message make the
    next turn read the whole conversation prefix from cache (Anthropic does not cache without them).
    """
    if body.get("tools") or body.get("functions"):
        raise HTTPException(400, "tool calling is not supported for Anthropic executors (Terminus-2 does not use it)")
    system: list[dict] = []
    messages: list[dict] = []
    for message in body.get("messages") or []:
        role, content = message.get("role"), message.get("content")
        if role in ("system", "developer"):
            text = _text_of(content)
            if text:
                system.append({"type": "text", "text": text})
        elif role == "assistant":
            if message.get("tool_calls"):
                raise HTTPException(400, "tool calls are not supported for Anthropic executors")
            text = _text_of(content)
            if text:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        elif role == "tool":
            raise HTTPException(400, "tool results are not supported for Anthropic executors")
        else:
            blocks = _user_blocks(content)
            if not blocks:
                continue
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": "user", "content": blocks})
    if not messages:
        messages.append({"role": "user", "content": [{"type": "text", "text": EMPTY_CONVERSATION_TEXT}]})

    requested = _as_int(body.get("max_tokens"))
    if requested is None:
        requested = _as_int(body.get("max_completion_tokens"))
    max_tokens = requested or executor.max_output_tokens
    request: dict[str, Any] = {"model": executor.model, "max_tokens": max_tokens, "messages": messages}
    thinking = _thinking(body.get("thinking"), executor.thinking_budget_tokens, max_tokens)
    if thinking is not None:
        request["thinking"] = thinking
    if system:
        system[-1]["cache_control"] = {"type": "ephemeral"}
        request["system"] = system
    messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return request


def anthropic_response_to_chat(message: dict, model: str) -> tuple[dict, dict]:
    """(chat.completion response, ledger extras) for one Anthropic Messages response.

    Only text blocks are returned to the agent; thinking blocks are counted, not forwarded. The usage block keeps
    Anthropic's convention: ``prompt_tokens`` is the uncached input and cache reads / writes are separate fields
    (Harbor's LiteLLM records cache reads as the step's cached tokens).
    """
    content = message.get("content") or []
    text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
    thinking_blocks = sum(1 for block in content if block.get("type") in ("thinking", "redacted_thinking"))
    raw = message.get("usage") or {}
    usage = {"prompt_tokens": int(raw.get("input_tokens") or 0),
             "completion_tokens": int(raw.get("output_tokens") or 0)}
    for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        if key in raw:
            usage[key] = int(raw.get(key) or 0)
    usage["total_tokens"] = sum(usage.values())
    finish = ANTHROPIC_STOP_TO_FINISH.get(message.get("stop_reason") or "end_turn", "stop")
    chat = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text or None}, "finish_reason": finish}],
        "usage": usage,
    }
    extras = {"backend_model": message.get("model"), "thinking_blocks": thinking_blocks,
              "reasoning_tokens": (raw.get("output_tokens_details") or {}).get("thinking_tokens")}
    return chat, extras


# ---------------------------------------------------------------------------------------------------- app
def create_app(executors: dict[str, Executor], ledger: Ledger, prices: dict,
               openai_base_url: str = OPENAI_BASE_URL, anthropic_base_url: str = ANTHROPIC_BASE_URL,
               transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """The gateway app. ``transport`` replaces the network (tests use ``httpx.MockTransport``)."""
    for executor in executors.values():
        price_of(prices, executor.model)          # fail at start-up, not after a paid call
    openai_base_url = openai_base_url.rstrip("/")
    anthropic_base_url = anthropic_base_url.rstrip("/")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            transport=transport, limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))
        yield
        await app.state.client.aclose()

    app = FastAPI(title="stoploss-collection-gateway", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ok": True, "executors": sorted(executors), "ledger": str(ledger.path.resolve())}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": label, "object": "model", "owned_by": "stoploss"}
                                           for label in sorted(executors)]}

    @app.get("/runs/{run_id}")
    async def run_totals(run_id: str):
        totals = ledger.run_totals(run_id)
        if totals is None:
            raise HTTPException(404, f"run {run_id} has made no call")
        return totals

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        run_id, task_id = run_identity(request)
        body = await request.json()
        if body.get("stream"):
            raise HTTPException(400, "streaming is not supported")
        name = str(body.get("model") or "")
        executor = executors.get(name.split("/", 1)[1] if "/" in name else name)
        if executor is None:
            raise HTTPException(400, f"model must be an executor label, one of {sorted(executors)}")
        spent = ledger.spent_usd(run_id)
        if executor.spend_valve_usd is not None and spent >= executor.spend_valve_usd:
            raise HTTPException(402, f"budget exhausted: spend valve reached for run {run_id} "
                                     f"(spent ${spent:.6f} >= valve ${executor.spend_valve_usd:.2f})")

        client: httpx.AsyncClient = request.app.state.client
        started = time.monotonic()
        if executor.provider == "openai":
            headers = {"authorization": f"Bearer {_api_key('OPENAI_API_KEY')}"}
            response = await post_with_retries(client, f"{openai_base_url}/chat/completions",
                                               openai_request(body, executor, run_id), headers)
            data = _upstream_json(response)
            details = (data.get("usage") or {}).get("completion_tokens_details") or {}
            extras = {"backend_model": data.get("model"), "thinking_blocks": None,
                      "reasoning_tokens": details.get("reasoning_tokens")}
        else:
            headers = {"x-api-key": _api_key("ANTHROPIC_API_KEY"), "anthropic-version": ANTHROPIC_VERSION}
            response = await post_with_retries(client, f"{anthropic_base_url}/v1/messages",
                                               anthropic_request(body, executor), headers)
            data, extras = anthropic_response_to_chat(_upstream_json(response), executor.model)
        latency_ms = (time.monotonic() - started) * 1000.0

        tokens = normalize_usage(data.get("usage") or {})
        choice = (data.get("choices") or [{}])[0] or {}
        ledger.record_call({
            "call_id": f"call-{uuid.uuid4().hex[:12]}",
            "run_id": run_id,
            "task_id": task_id,
            "worker": executor.label,
            "model": executor.model,
            "requested_model": executor.model,
            "backend_model": extras["backend_model"],
            **tokens,
            "reasoning_tokens": extras["reasoning_tokens"],
            "thinking_blocks": extras["thinking_blocks"],
            "finish_reason": choice.get("finish_reason"),
            "cost_usd": booked_cost_usd(prices, executor.model, tokens),
            "latency_ms": latency_ms,
        })
        return JSONResponse(data)

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Metering gateway for trajectory collection (OpenAI-compatible).")
    parser.add_argument("--ledger", required=True, help="JSONL call ledger to append to")
    parser.add_argument("--executors", default=str(DEFAULT_EXECUTORS), help="executor configuration (JSON)")
    parser.add_argument("--prices", default=str(DEFAULT_PRICES), help="price file used for the spend valve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--openai-base-url", default=OPENAI_BASE_URL)
    parser.add_argument("--anthropic-base-url", default=ANTHROPIC_BASE_URL)
    args = parser.parse_args(argv)

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    executors = load_executors(args.executors)
    for executor in executors.values():
        key = "OPENAI_API_KEY" if executor.provider == "openai" else "ANTHROPIC_API_KEY"
        if not os.environ.get(key):
            log.warning("%s is not set: calls for executor %r will fail", key, executor.label)
    app = create_app(executors, Ledger(args.ledger), load_prices(args.prices),
                     openai_base_url=args.openai_base_url, anthropic_base_url=args.anthropic_base_url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
