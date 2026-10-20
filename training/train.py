#!/usr/bin/env python3
"""Train one StopLoss monitor (one backbone, one seed) with LoRA (paper Sec. 2.2 and App. A).

The three recipes of the paper are in ``training/recipes/``. They share one recipe: one epoch, LoRA rank 16 /
alpha 32 / dropout 0.05 on q, k, v, o projections, AdamW (lr 1e-4, weight decay 0.05), cosine schedule with 3%
warm-up, BF16, gradient checkpointing, sequences of at most 1,600 tokens, effective batch 16. They differ only in the
backbone and in how the batch is split over GPUs:

    qwen2.5-7b-instruct         1 GPU,  micro-batch 2 x accumulation 8, seeds 0-4 (the primary StopLoss ensemble)
    qwen3-8b                    2 GPUs, micro-batch 2 x accumulation 4, seeds 0-3 (Qwen3 thinking disabled)
    qwen2.5-coder-7b-instruct   2 GPUs, micro-batch 2 x accumulation 4, seeds 0-3

    python training/train.py --recipe training/recipes/qwen2.5-7b-instruct.json \
        --data data/train.jsonl --val-data data/validation.jsonl --out runs/qwen2.5-7b-instruct_s0 --seed 0
    torchrun --nproc_per_node 2 training/train.py --recipe training/recipes/qwen3-8b.json \
        --data data/train.jsonl --val-data data/validation.jsonl --out runs/qwen3-8b_s0 --seed 0
    bash training/run_seeds.sh training/recipes/qwen3-8b.json runs --data data/train.jsonl --val-data data/validation.jsonl

The loss is the ordinary causal-LM loss on the answer tokens only (``stoploss.sequence.encode_train``). Windows of
held-out tasks are never trained on. The held-out tasks are those of ``--val-data`` (the validation split of the
released training data), or those listed in ``--val-tasks`` (for windows built from your own runs), or by default 10%
of the repositories. From the held-out runs, 150 trajectories are drawn with a fixed seed and their policy-checkpoint
windows (k = 5, 8, 11, ...) form the selection set; the released validation split is exactly this set. Every 300
optimizer steps, and at the end, rank 0 scores it and keeps the adapter with the best ABORT AUC in ``<out>/best``.
``<out>/adapter`` is the last step.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stoploss import sequence                                       # noqa: E402
from stoploss.policy import is_checkpoint                           # noqa: E402


@dataclass
class TrainConfig:
    data: str                               # windows JSONL with "task", "k", "prompt", "label" (+ "bench", "worker")
    out: str
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    gpus: int = 1                           # number of processes the recipe expects (torchrun --nproc_per_node)
    val_tasks: str | None = None            # JSON list of held-out task ids; default: 10% of repositories
    val_data: str | None = None             # or: held-out windows in their own JSONL (the released validation split)
    max_len: int = 1600
    seed: int = 0
    lr: float = 1e-4
    weight_decay: float = 0.05
    epochs: float = 1.0
    batch_size: int = 2                     # per GPU
    grad_accum: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    warmup_ratio: float = 0.03
    eval_every: int = 300
    val_trajectories: int = 150
    max_steps: int = -1
    gradient_checkpointing: bool = True
    final_val_windows: int = 0              # >0: also report ABORT AUC on up to this many held-out windows at the end


def _auc(pos: list[float], neg: list[float]) -> float:
    neg = sorted(neg)
    if not pos or not neg:
        return float("nan")
    return sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos) / (len(pos) * len(neg))


def split_rows(rows: list[dict], val_tasks_file: str | None, seed: int, val_rows: list[dict] | None = None):
    """(train rows shuffled with ``seed``, held-out rows, held-out task set).

    ``val_rows`` are held-out windows from a separate file; windows of their tasks in ``rows`` are not trained on either.
    """
    tasks = sorted({r["task"] for r in rows})
    if val_rows is not None:
        val_tasks = {r["task"] for r in val_rows}
    elif val_tasks_file:
        val_tasks = set(json.loads(Path(val_tasks_file).read_text())) & set(tasks)
    else:                                   # hold out whole repositories, never single tasks of a repository seen in training
        repo = lambda t: t.rsplit("-", 1)[0]
        repos = sorted({repo(t) for t in tasks})
        random.Random(0).shuffle(repos)
        held = set(repos[: max(2, len(repos) // 10)])
        val_tasks = {t for t in tasks if repo(t) in held}
    train = [r for r in rows if r["task"] not in val_tasks]
    val = val_rows if val_rows is not None else [r for r in rows if r["task"] in val_tasks]
    random.Random(seed).shuffle(train)
    return train, val, val_tasks


def selection_windows(val: list[dict], n_trajectories: int) -> list[dict]:
    """Policy-checkpoint windows of a fixed random subset of held-out trajectories (the checkpoint-selection set)."""
    by_run: dict[tuple, list[dict]] = {}
    for r in val:
        by_run.setdefault((r.get("bench"), r.get("worker"), r["task"]), []).append(r)
    keys = sorted(by_run)
    random.Random(1).shuffle(keys)
    return [r for key in keys[:n_trajectories] for r in by_run[key] if is_checkpoint(r["k"])]


def train(cfg: TrainConfig) -> None:
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments,
                              set_seed)

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != cfg.gpus:
        raise SystemExit(f"the recipe expects {cfg.gpus} GPU process(es) but {world} were started; the effective batch "
                         f"(batch_size x grad_accum x GPUs) would change. Use torchrun --nproc_per_node {cfg.gpus} or --gpus.")
    if cfg.val_data and cfg.val_tasks:
        raise SystemExit("give either --val-data or --val-tasks, not both")
    rows = [json.loads(line) for line in open(cfg.data)]
    held_out = [json.loads(line) for line in open(cfg.val_data)] if cfg.val_data else None
    train_rows, val_rows, val_tasks = split_rows(rows, cfg.val_tasks, cfg.seed, held_out)
    select_rows = selection_windows(val_rows, cfg.val_trajectories)
    if rank == 0:
        json.dump(sorted({r["task"] for r in rows} | val_tasks), open(out / "train_tasks.json", "w"))
        json.dump(sorted(val_tasks), open(out / "val_tasks.json", "w"))
        print(f"[train] {len(train_rows)} training windows, {len(val_rows)} held-out windows ({len(val_tasks)} tasks), "
              f"{len(select_rows)} selection windows; model {cfg.model}; GPUs {world}; "
              f"effective batch {cfg.batch_size * cfg.grad_accum * world}", flush=True)

    set_seed(cfg.seed)                       # seed before model and LoRA initialisation
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.model)

    lengths = []
    if rank == 0:                            # pre-flight: every row must encode within budget before the model loads
        bad = []
        for name, part in (("train", train_rows), ("selection", select_rows), ("held-out", val_rows)):
            for r in part:
                try:
                    lengths.append(len(sequence.encode_train(tok, r["prompt"], r["label"], cfg.max_len)["input_ids"]))
                except ValueError as exc:
                    bad.append((name, r.get("task"), r.get("k"), str(exc)[:80]))
        if bad:
            raise SystemExit(f"{len(bad)} windows exceed max_len={cfg.max_len} or do not encode: {bad[:5]}")
        lengths.sort()
        print(f"[preflight] {len(lengths)} windows fit max_len={cfg.max_len} (p50 {lengths[len(lengths) // 2]}, max {lengths[-1]})", flush=True)

    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, device_map={"": rank})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                             task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    init_hash = hashlib.sha256()                # identical across runs with the same seed and backbone
    for name, param in sorted(model.named_parameters()):
        if "lora_" in name:
            init_hash.update(param.detach().float().cpu().numpy().tobytes())
    if rank == 0:
        print(f"[lora] adapter init sha256 {init_hash.hexdigest()[:16]} (seed {cfg.seed})", flush=True)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    class Windows(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            r = self.items[i]
            return sequence.encode_train(tok, r["prompt"], r["label"], cfg.max_len)

    def collate(batch):
        width = max(len(x["input_ids"]) for x in batch)
        pad = tok.pad_token_id or tok.eos_token_id
        return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (width - len(x["input_ids"])) for x in batch]),
                "labels": torch.tensor([x["labels"] + [-100] * (width - len(x["labels"])) for x in batch]),
                "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (width - len(x["input_ids"])) for x in batch])}

    readout_ids = sequence.first_token_ids(tok)

    def abort_score(r) -> float:
        ids = torch.tensor([sequence.encode_infer(tok, r["prompt"], cfg.max_len)], device=model.device)
        return torch.softmax(model(input_ids=ids).logits[0, -1][readout_ids].float(), dim=0)[sequence.ABORT_INDEX].item()

    def selection_auc(items) -> float:
        model.eval()
        with torch.no_grad():
            scored = [(abort_score(r), r["label"] == "ABORT") for r in items]
        model.train()
        return _auc([p for p, y in scored if y], [p for p, y in scored if not y])

    if rank == 0:
        import peft
        import transformers
        config = asdict(cfg) | {"rows_train": len(train_rows), "rows_val": len(val_rows), "selection_rows": len(select_rows),
                                "data_sha256": hashlib.sha256(open(cfg.data, "rb").read()).hexdigest(),
                                "val_data_sha256": hashlib.sha256(open(cfg.val_data, "rb").read()).hexdigest() if cfg.val_data else None,
                                "world_size": world,
                                "seq_len_p50": lengths[len(lengths) // 2], "seq_len_max": lengths[-1],
                                "adapter_init_sha256": init_hash.hexdigest(),
                                "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                             "peft": peft.__version__, "python": platform.python_version()}}
        json.dump(config, open(out / "run_config.json", "w"), indent=1)

    history: list[tuple[int, float]] = []
    best = [float("-inf"), 0]

    def evaluate_and_keep(step: int) -> None:
        value = selection_auc(select_rows)
        history.append((step, value))
        improved = value > best[0]
        if improved:
            best[:] = [value, step]
            model.save_pretrained(out / "best", safe_serialization=True)
            tok.save_pretrained(out / "best")
            json.dump({"step": step, "val_ckpt_auc": value, "history": history}, open(out / "best.json", "w"), indent=1)
        json.dump({"best": best, "history": history}, open(out / "eval_history.json", "w"), indent=1)
        print(f"[eval] step {step} selection AUC {value:.4f} (best {best[0]:.4f} @ {best[1]})", flush=True)

    class Selection(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if cfg.eval_every and state.global_step % cfg.eval_every == 0 and rank == 0:
                evaluate_and_keep(state.global_step)
            return control

    args = TrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs, max_steps=cfg.max_steps, learning_rate=cfg.lr, bf16=True, logging_steps=25,
        save_strategy="no", report_to=[], optim="adamw_torch", warmup_ratio=cfg.warmup_ratio, weight_decay=cfg.weight_decay,
        label_smoothing_factor=0.0, lr_scheduler_type="cosine", seed=cfg.seed, dataloader_num_workers=2,
        remove_unused_columns=False, ddp_find_unused_parameters=False, save_only_model=True)
    trainer = Trainer(model=model, args=args, train_dataset=Windows(train_rows), data_collator=collate, callbacks=[Selection()])
    t0 = time.time()
    trainer.train()
    if rank == 0:
        if cfg.eval_every:
            evaluate_and_keep(trainer.state.global_step)
        model.save_pretrained(out / "adapter", safe_serialization=True)       # the last step, kept for reference
        tok.save_pretrained(out / "adapter")
        summary = {"best": best, "steps": trainer.state.global_step, "minutes": round((time.time() - t0) / 60, 1),
                   "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
        if cfg.final_val_windows:
            held = val_rows[:]
            random.Random(1).shuffle(held)
            summary["held_out_abort_auc"] = selection_auc(held[: cfg.final_val_windows])
        json.dump(summary, open(out / "train_summary.json", "w"), indent=1)
        print(f"[train] done: best selection AUC {best[0]:.4f} at step {best[1]} -> {out / 'best'}", flush=True)


def main() -> None:
    names = {f.name for f in fields(TrainConfig)} - {"data", "out"}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True, help="one of training/recipes/*.json")
    ap.add_argument("--data", required=True, help="training windows (train.jsonl of the released training data, or "
                                                  "JSONL from scripts/build_windows.py)")
    ap.add_argument("--out", required=True, help="output folder of this run")
    held = ap.add_mutually_exclusive_group()
    held.add_argument("--val-data", default=None, help="held-out windows (validation.jsonl of the released training data)")
    held.add_argument("--val-tasks", default=None, help="JSON list of held-out task ids of --data (default: 10%% of repositories)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early, e.g. for a smoke test")
    ap.add_argument("--eval-every", type=int, default=None, help="override the recipe's selection interval")
    ap.add_argument("--val-trajectories", type=int, default=None, help="override the size of the selection set")
    ap.add_argument("--final-val-windows", type=int, default=None)
    a = ap.parse_args()
    recipe = {k: v for k, v in json.loads(Path(a.recipe).read_text()).items() if k in names}
    overrides = {"val_tasks": a.val_tasks, "val_data": a.val_data, "seed": a.seed, "max_steps": a.max_steps, "eval_every": a.eval_every,
                 "val_trajectories": a.val_trajectories, "final_val_windows": a.final_val_windows}
    train(TrainConfig(data=a.data, out=a.out, **(recipe | {k: v for k, v in overrides.items() if v is not None})))


if __name__ == "__main__":
    main()
