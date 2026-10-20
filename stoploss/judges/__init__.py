"""Paid LLM-judge baseline (paper Sec. 3.2 and App. A).

Six general-purpose judges read the same windows as the small monitor, at the policy checkpoints k = 5, 8, 11, ...,
with the monitor question replaced by a request for ``FAIL_PROBABILITY: <0-100>``. GPT-5.6 Sol / Terra / Luna read
GPT-executor runs through the OpenAI Responses API; Claude Opus 5 / Opus 4.6 / Sonnet 4.6 read Claude-executor runs
through the Anthropic Messages API (``configs/judges.json``).
"""
from .clients import APIError, FatalAPIError, make_client
from .prompt import JUDGE_SUFFIX, judge_prompt, parse_answer, sha16
from .runner import (WindowCosts, is_checkpoint, judge_config, judge_config_sha, load_judges, load_package,
                     read_scores, row_problems, run_judge, score_window, valid_scores, window_costs)

__all__ = ["APIError", "FatalAPIError", "JUDGE_SUFFIX", "WindowCosts", "is_checkpoint", "judge_config",
           "judge_config_sha", "judge_prompt", "load_judges", "load_package", "make_client", "parse_answer",
           "read_scores", "row_problems", "run_judge", "score_window", "sha16", "valid_scores", "window_costs"]
