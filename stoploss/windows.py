"""Trajectory windows and outcome labels (paper Sec. 2.1 and App. A).

A completed run is a Harbor ATIF trajectory plus a verifier verdict. Its tool steps become (action, observation)
pairs: the action is the agent's message (reasoning) followed by the commands it sent, truncated to 700 characters;
the observation is what the tools returned, truncated to 900 characters. The window at step k shows the step count
and the three most recent pairs, followed by the monitor question:

    An autonomous agent has run <k> steps on a task. Its last steps:
    <action> ... </action>
    <observation> ... </observation>
    ...
    Decide the agent's best move now: CONTINUE working, DELIVER (verify and declare done), or ABORT (hopeless).
    Answer with one word.

Labels come from the final outcome only: every window of a failed run is ABORT; on a successful run the windows at
the last two tool steps (k >= n - 1) are DELIVER and earlier ones CONTINUE. Runtime stopping uses only the ABORT score.

If the templated window exceeds the token budget (1,600 tokens in the paper), the middle of the window body is
removed: the step-count header and the question are kept, and the body keeps a quarter of its token budget from the
beginning and the rest from the end, marked with ``[... middle omitted ...]``.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import sequence

QUESTION = ("Decide the agent's best move now: CONTINUE working, DELIVER (verify and declare done), "
            "or ABORT (hopeless). Answer with one word.")
QUESTION_SUFFIX = "\n\n" + QUESTION         # how every window prompt ends (the judges replace exactly this part)
ACTION_CHARS, OBSERVATION_CHARS = 700, 900
RECENT_STEPS = 3
OMISSION_MARK = "\n[... middle omitted ...]\n"
# k values of the original training-corpus windows; the evaluation sets use every k
K_GRID = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100)


def step_pairs(trajectory: str | Path) -> list[tuple[str, str]]:
    """(action, observation) for every agent step of an ATIF trajectory that ran at least one tool call."""
    data = json.loads(Path(trajectory).read_text())
    pairs = []
    for step in data.get("steps") or []:
        if step.get("source") != "agent" or not step.get("tool_calls"):
            continue
        args = [tc.get("arguments") or {} for tc in step["tool_calls"]]
        if any(not isinstance(a, dict) for a in args):
            raise ValueError(f"{trajectory}: unsupported non-object tool arguments")
        # Terminus-2 calls the command `keystrokes`; other Harbor agents use `command`.
        commands = [str(a.get("keystrokes") or a.get("command") or a) for a in args]
        observation = "".join(str(r.get("content", "")) for r in ((step.get("observation") or {}).get("results") or []))
        action = (str(step.get("message") or "") + "\n" + "\n".join(commands)).strip()
        pairs.append((action[:ACTION_CHARS], observation[:OBSERVATION_CHARS]))
    return pairs


def label_for(k: int, n_steps: int, reward: float) -> str:
    if reward == 1.0:
        return "DELIVER" if k >= n_steps - 1 else "CONTINUE"
    return "ABORT"


def _header(k: int) -> str:
    return f"An autonomous agent has run {k} steps on a task. Its last steps:\n"


def _body(pairs: list[tuple[str, str]], k: int) -> str:
    return "".join(f"<action>\n{a}\n</action>\n<observation>\n{o}\n</observation>\n" for a, o in pairs[:k][-RECENT_STEPS:])


def window_prompt(pairs: list[tuple[str, str]], k: int) -> str:
    """The untruncated window at tool step k (1-based)."""
    return f"{_header(k)}{_body(pairs, k)}\n{QUESTION}"


def fit_to_budget(pairs: list[tuple[str, str]], k: int, tok, max_len: int) -> tuple[str, bool]:
    """(prompt, middle_was_omitted). The full chat-templated sequence, for every candidate answer, fits ``max_len``."""
    prompt = window_prompt(pairs, k)
    if sequence.fits(tok, prompt, max_len):
        return prompt, False
    head = _header(k)
    ids = tok(_body(pairs, k), add_special_tokens=False)["input_ids"]
    n_fixed = len(tok(head + QUESTION, add_special_tokens=False)["input_ids"])
    for budget in range(max_len - n_fixed - 40, 0, -16):
        n_head = budget // 4
        candidate = head + tok.decode(ids[:n_head]) + OMISSION_MARK + tok.decode(ids[-(budget - n_head):]) + "\n" + QUESTION
        if sequence.fits(tok, candidate, max_len):
            return candidate, True
    raise ValueError(f"window k={k} cannot fit the template and answers within {max_len} tokens")


def run_windows(trajectory: str | Path, reward: float, tok, max_len: int = 1600, ks=None) -> list[dict]:
    """Labelled windows of one completed run. ``ks``: iterable of tool steps to emit (default: every k >= 2)."""
    pairs = step_pairs(trajectory)
    n = len(pairs)
    if n < 2:
        return []
    wanted = range(2, n + 1) if ks is None else sorted(k for k in set(ks) if 2 <= k <= n)
    rows = []
    for k in wanted:
        prompt, omitted = fit_to_budget(pairs, k, tok, max_len)
        rows.append({"k": k, "prompt": prompt, "label": label_for(k, n, reward), "n_steps": n, "middle_omitted": omitted})
    return rows
