"""Failure scores of a monitor (paper Sec. 2.2): one forward pass per window, no generation.

s(k) = softmax over the next-token logits of the first tokens of CONTINUE, DELIVER and ABORT, read at ABORT.
Scoring runs the model under BF16 autocast (as in training) and takes the softmax in FP32. With batch size 1 the
average wall time per window is the monitoring-cost proxy of App. C.2 (GPU seconds x price per GPU-hour).
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import time
from pathlib import Path

from . import sequence


def load_monitor(base: str, adapter: str | None = None, device: str = "cuda:0"):
    """(tokenizer, model) for a base model, optionally with a trained LoRA adapter; ``adapter=None`` = untuned base."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": device})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = False
    return tok, model


def score_windows(tok, model, rows: list[dict], max_len: int = 1600, batch_size: int = 1, consistency: int = 64,
                  tolerance: float = 0.02, log=print) -> tuple[list[float], dict]:
    """P(ABORT) for every window and timing / consistency metadata.

    Windows of equal token length are batched together (no padding). With ``batch_size > 1`` a random subset is
    re-scored one by one and the run fails if any score moves by more than ``tolerance``.
    """
    import torch
    readout = sequence.first_token_ids(tok)
    ids = [sequence.encode_infer(tok, r["prompt"], max_len) for r in rows]        # raises on over-budget windows
    by_length = collections.defaultdict(list)
    for i, x in enumerate(ids):
        by_length[len(x)].append(i)
    batches = [g[j:j + batch_size] for g in by_length.values() for j in range(0, len(g), batch_size)]

    def logits_of(batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=batch).logits[:, -1, :]
        return logits.float()

    p = [None] * len(rows)
    t0 = time.time()
    with torch.no_grad():
        for bi, idx in enumerate(batches):
            batch = torch.tensor([ids[i] for i in idx], device=model.device)
            probs = torch.softmax(logits_of(batch)[:, readout], dim=-1)[:, sequence.ABORT_INDEX].tolist()
            for i, value in zip(idx, probs):
                p[i] = value
            if bi % 300 == 0:
                log(f"  scored {sum(len(b) for b in batches[:bi + 1])}/{len(rows)} windows, {time.time() - t0:.0f}s")
        loop_seconds = time.time() - t0
        diffs = []
        for i in random.Random(0).sample(range(len(rows)), min(consistency, len(rows))) if batch_size > 1 else []:
            q = torch.softmax(logits_of(torch.tensor([ids[i]], device=model.device))[0][readout], dim=0)[sequence.ABORT_INDEX].item()
            diffs.append(abs(q - p[i]))
    if diffs and max(diffs) > tolerance:
        raise RuntimeError(f"batch size {batch_size} changes scores by up to {max(diffs):.4f} > {tolerance}; use a smaller batch")
    if any(v is None or not (math.isfinite(v) and 0.0 <= v <= 1.0) for v in p):
        raise RuntimeError("some windows did not get a valid score")
    meta = {"n": len(rows), "batch_size": batch_size, "loop_seconds": loop_seconds,
            "seconds_per_window": loop_seconds / len(rows) if rows else None,
            "max_batch1_abs_diff": max(diffs) if diffs else None, "max_len": max_len,
            "precision": "bf16 autocast forward, fp32 softmax over the three answer tokens",
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    return p, meta


def write_scores(path: str | Path, rows: list[dict], p: list[float], meta: dict) -> None:
    """``<path>`` (one JSON line per window) and ``<path minus .jsonl>.meta.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        for r, value in zip(rows, p):
            stream.write(json.dumps({"set": r.get("set", "test"), "bench": r["bench"], "worker": r["worker"], "task": r["task"],
                                     "k": r["k"], "p": value, "prompt_sha256": hashlib.sha256(r["prompt"].encode()).hexdigest()[:16]}) + "\n")
    Path(str(path)[: -len(".jsonl")] + ".meta.json" if str(path).endswith(".jsonl") else str(path) + ".meta.json").write_text(json.dumps(meta, indent=1))
