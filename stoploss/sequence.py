"""Chat-template encoding shared by the window builder, the trainer and the scorer (paper App. A).

Every component encodes a window the same way, so the token ids seen in training and in scoring are identical:
  * the window prompt is rendered as a single user message with the model's chat template and generation header
    (for Qwen3, thinking is disabled so the model answers directly, like an Instruct model);
  * a window is admissible only if the templated prompt plus *every* candidate answer fits in ``max_len`` tokens;
  * encoders raise instead of truncating.
"""
from __future__ import annotations

LABELS = ("CONTINUE", "DELIVER", "ABORT")
ABORT_INDEX = LABELS.index("ABORT")


def chat_template_kwargs(tok) -> dict:
    """Qwen3 tokenizers (those with a ``<think>`` token) render the empty think block, i.e. direct-answer mode."""
    return {"enable_thinking": False} if tok.convert_tokens_to_ids("<think>") is not None else {}


def render(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, **chat_template_kwargs(tok))


def prompt_ids(tok, prompt: str) -> list[int]:
    return tok(render(tok, prompt), add_special_tokens=False)["input_ids"]


def label_ids(tok, label: str) -> list[int]:
    """Token ids of an answer word appended right after the generation header."""
    return tok(label, add_special_tokens=False)["input_ids"]


def seq_len_max(tok, prompt: str) -> int:
    """Length of the longest full sequence (templated prompt + answer) over the three candidate answers."""
    pre = render(tok, prompt)
    return max(len(tok(pre + y, add_special_tokens=False)["input_ids"]) for y in LABELS)


def fits(tok, prompt: str, max_len: int) -> bool:
    return seq_len_max(tok, prompt) <= max_len


def encode_train(tok, prompt: str, label: str, max_len: int) -> dict:
    """``input_ids`` / ``labels`` for one training row: prompt tokens are masked with -100 and every answer token is
    supervised. No EOS is appended. Raises if the window is over budget or the prompt/answer boundary does not align."""
    if label not in LABELS:
        raise ValueError(f"bad label {label!r}")
    pre = render(tok, prompt)
    ids = tok(pre + label, add_special_tokens=False)["input_ids"]
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    n_pre = len(pre_ids)
    if ids[:n_pre] != pre_ids:
        raise ValueError("prompt/label tokenization boundary mismatch")
    if len(ids) - n_pre < 1:
        raise ValueError("label produced no tokens")
    if seq_len_max(tok, prompt) > max_len:
        raise ValueError(f"sequence exceeds max_len {max_len}")
    return {"input_ids": ids, "labels": [-100] * n_pre + ids[n_pre:]}


def encode_infer(tok, prompt: str, max_len: int) -> list[int]:
    """Prompt ids for scoring; the logits at the last position predict the first answer token."""
    if seq_len_max(tok, prompt) > max_len:
        raise ValueError(f"sequence exceeds max_len {max_len}")
    return prompt_ids(tok, prompt)


def first_token_ids(tok) -> list[int]:
    """First token of CONTINUE, DELIVER and ABORT: the three logits of the failure-score readout."""
    return [label_ids(tok, y)[0] for y in LABELS]
