"""Judge prompt and answer parsing for the paid LLM-judge baseline (paper Sec. 3.2 and App. A).

A judge reads exactly the window that the small monitor reads; only the final question differs. The monitor question
(``stoploss.windows.QUESTION_SUFFIX``) is replaced by ``JUDGE_SUFFIX``, which asks for a final line
``FAIL_PROBABILITY: <integer 0-100>``. The judge's score is that integer divided by 100.
"""
from __future__ import annotations

import hashlib
import re

from ..windows import QUESTION_SUFFIX

JUDGE_SUFFIX = (
    "\n\nYou are an expert reviewer of autonomous coding-agent runs. The agent above is working on a software-engineering "
    "task in a repository; only its most recent steps are shown. Estimate the probability that this run will ULTIMATELY "
    "FAIL (its final patch will not pass the hidden tests) if the agent keeps going as it has been. Reply with at most "
    "three sentences of reasoning, then a final line exactly of the form:\nFAIL_PROBABILITY: <integer 0-100>"
)

# The tag accepts an ASCII or a fullwidth colon (U+FF1A) and markdown bold before the number; the last tag wins.
ANSWER_RE = re.compile(r"FAIL_PROBABILITY\s*[:\uff1a]\s*\**\s*(\d{1,3})")
# Fallback when no tag is present: the last "<number>%" in the text.
PERCENT_RE = re.compile(r"(\d{1,3})\s*%")


def sha16(text: str) -> str:
    """First 16 hex digits of sha256(text); used for prompt and configuration fingerprints."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def judge_prompt(window_prompt: str) -> str:
    """The text sent to a judge: the window with its monitor question replaced by ``JUDGE_SUFFIX``."""
    if not window_prompt.endswith(QUESTION_SUFFIX):
        raise ValueError("window prompt does not end with the monitor question (stoploss.windows.QUESTION_SUFFIX); "
                         "refusing to build a judge prompt from it")
    return window_prompt[: -len(QUESTION_SUFFIX)] + JUDGE_SUFFIX


def parse_answer(text: str | None) -> tuple[int | None, str]:
    """(answer in 0..100 or None, parse mode). Modes: "tag", "percent_fallback", "unparsed", "no_text".

    Numbers are clamped to 0..100. A parsed answer is only a valid score if the response also terminated normally;
    that rule is applied by the runner, not here.
    """
    if not text:
        return None, "no_text"
    tags = ANSWER_RE.findall(text)
    if tags:
        return max(0, min(100, int(tags[-1]))), "tag"
    percents = PERCENT_RE.findall(text)
    if percents:
        return max(0, min(100, int(percents[-1]))), "percent_fallback"
    return None, "unparsed"
