import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_trainer():
    spec = importlib.util.spec_from_file_location("stoploss_train", ROOT / "training" / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module                      # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_released_layout_matches_held_out_task_list(tmp_path):
    """Training windows plus a separate selection file (the released data) give the trainer the same training order and
    selection set as the full window file with a held-out task list (the paper's run)."""
    trainer = load_trainer()
    rows = [{"bench": f"batch{t % 3}", "worker": "w", "task": f"owner__repo{t % 8}-{t}", "k": k,
             "prompt": f"task {t} step {k}", "label": "ABORT" if t % 2 else "CONTINUE"}
            for t in range(40) for k in (2, 5, 6, 8, 11)]
    held = sorted({r["task"] for r in rows if r["task"].startswith("owner__repo3-")})
    (tmp_path / "val_tasks.json").write_text(json.dumps(held))
    key = lambda r: (r["task"], r["k"], r["prompt"], r["label"])
    strip = lambda r: {f: r[f] for f in ("task", "k", "prompt", "label")}

    train_a, val_a, _ = trainer.split_rows(rows, str(tmp_path / "val_tasks.json"), seed=3)
    select_a = trainer.selection_windows(val_a, n_trajectories=3)
    assert {r["k"] for r in select_a} == {5, 8, 11} and len({r["task"] for r in select_a}) == 3

    released_train = [strip(r) for r in rows if r["task"] not in held]
    released_val = [strip(r) for r in sorted(select_a, key=key)]
    train_b, val_b, tasks_b = trainer.split_rows(released_train, None, seed=3, val_rows=released_val)
    assert [key(r) for r in train_b] == [key(r) for r in train_a]
    assert sorted(map(key, trainer.selection_windows(val_b, n_trajectories=3))) == sorted(map(key, select_a))
    assert not tasks_b & {r["task"] for r in train_b}
