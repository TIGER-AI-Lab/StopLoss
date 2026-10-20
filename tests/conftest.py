import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class CharTokenizer:
    """A character-level stand-in for a Hugging Face tokenizer (enough for the window and encoding code paths)."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def convert_tokens_to_ids(self, token):
        return None


@pytest.fixture
def tok():
    return CharTokenizer()


def write_trajectory(path: Path, steps: list[tuple[str, str, str]], metrics: list[dict] | None = None) -> Path:
    """ATIF-like trajectory: a user message, then one agent step per (message, keystrokes, observation)."""
    out = [{"source": "user", "message": "Task Description:\nfix the bug"}]
    for i, (message, keys, obs) in enumerate(steps):
        step = {"source": "agent", "message": message, "tool_calls": [{"arguments": {"keystrokes": keys}}],
                "observation": {"results": [{"content": obs}]}}
        if metrics:
            step["metrics"] = metrics[i]
        out.append(step)
    out.append({"source": "agent", "message": "done, no tool call"})      # not a tool step
    path.write_text(json.dumps({"schema_version": "ATIF-v1", "steps": out}))
    return path
