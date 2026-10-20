"""Trajectory collection: stock Harbor Terminus-2 runs behind a metering gateway (paper Sec. 3.1 and App. A).

The collection stage has three parts:
  * ``stoploss.collect.gateway``: an OpenAI-compatible proxy that Terminus-2 talks to. It forwards each call to
    the executor's provider, writes one ledger line per call and enforces an optional per-run spend valve;
  * ``stoploss.collect.ledger``: the append-only call ledger that ``stoploss.costs`` reads;
  * ``stoploss.collect.harbor_runner``: runs one Harbor trial per task through the gateway and writes one run
    record per task.

This module holds the executor configuration shared by the gateway and the runner (``configs/executors.json``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_EXECUTORS = Path(__file__).resolve().parents[2] / "configs" / "executors.json"
PROVIDERS = ("openai", "anthropic")


@dataclass(frozen=True)
class Executor:
    """One executor: the upstream model plus the generation settings and run limits used to collect it."""

    label: str                                  # name used in run records and as the agent-facing model alias
    provider: str                               # "openai" (Chat Completions) or "anthropic" (Messages API)
    model: str                                  # upstream model id; also the key into the price file
    max_turns: int = 100                        # Terminus-2 turn budget (the binding limit)
    agent_timeout_sec: float | None = 7200.0    # None keeps the timeout from the task's task.toml
    verifier_timeout_sec: float | None = 7200.0
    outer_timeout_sec: float = 16000.0          # wall-clock limit of the whole Harbor process
    spend_valve_usd: float | None = None        # per-run safety valve (HTTP 402 once reached); None disables it
    reasoning_effort: str | None = None         # OpenAI only; None leaves the upstream default
    prompt_cache_key: bool = True               # OpenAI only; pin every call of a run to one prompt-cache key
    thinking_budget_tokens: int | None = None   # Anthropic only; extended thinking budget (None: no thinking)
    max_output_tokens: int = 16000              # Anthropic only; max_tokens used when the agent sends none

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"executor {self.label!r}: provider must be one of {PROVIDERS}, got {self.provider!r}")
        lowered = self.label.lower()
        # Harbor rewrites the messages of any model whose name mentions Claude/Anthropic (it adds its own
        # cache_control blocks), and LiteLLM would apply that model's context limits. The paper's runs used an
        # alias LiteLLM does not know, so the label must not look like a provider model name.
        if "claude" in lowered or "anthropic" in lowered or "/" in self.label or "|" in self.label:
            raise ValueError(f"executor label {self.label!r} must not contain 'claude', 'anthropic', '/' or '|'")

    @property
    def agent_model_name(self) -> str:
        """The model name Terminus-2 is given; LiteLLM sends the bare label to the gateway."""
        return f"openai/{self.label}"


def load_executors(path: str | Path = DEFAULT_EXECUTORS) -> dict[str, Executor]:
    """{label: Executor} from an executors JSON file (``{"executors": {label: {...}}}``)."""
    data = json.loads(Path(path).read_text())
    known = {f.name for f in fields(Executor)} - {"label"}
    executors = {}
    for label, spec in data["executors"].items():
        unknown = set(spec) - known - {"description"}
        if unknown:
            raise ValueError(f"executor {label!r}: unknown fields {sorted(unknown)}")
        executors[label] = Executor(label=label, **{k: v for k, v in spec.items() if k in known})
    return executors
