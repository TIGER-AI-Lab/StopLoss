import json

import numpy as np
import pytest
from conftest import write_trajectory

from stoploss import bootstrap, costs, policy
from stoploss.evaluate import evaluate_cell

PRICES = {"gpt-x": {"input": 1.0, "cache_read": 0.1, "cache_write": None, "output": 10.0},
          "claude-x": {"input": 2.0, "cache_read": 0.2, "cache_write": 2.5, "output": 20.0}}


def call(run, prompt, completion, read=0, write=0, ts=0.0, model="gpt-x"):
    return {"type": "call", "call_id": f"c{ts}", "run_id": run, "model": model, "requested_model": model, "prompt_tokens": prompt,
            "cache_read_tokens": read, "cache_write_tokens": write, "input_tokens_total": prompt + read + write,
            "completion_tokens": completion, "ts": ts, "discarded": False}


def test_ledger_alignment_and_spend_curve(tmp_path):
    traj = write_trajectory(tmp_path / "t.json", [("a", "ls\n", "x"), ("b", "cat\n", "y")],
                            metrics=[{"prompt_tokens": 100, "completion_tokens": 10}, {"prompt_tokens": 200, "completion_tokens": 20}])
    ledger = tmp_path / "ledger.jsonl"
    lines = [call("r1", 100, 10, ts=1), call("r1", 50, 5, ts=2),        # an extra call before step 2 (charged to step 2)
             call("r1", 200, 20, ts=3), call("r1", 300, 30, ts=4)]      # trailing call after the last tool step
    ledger.write_text("".join(json.dumps(x) + "\n" for x in lines))
    calls = costs.load_ledger(ledger)["r1"]
    tc = costs.token_curve(traj, calls)
    assert tc["n"] == 2 and tc["source"] == "ledger"
    assert tc["tok"].tolist() == [[100, 0, 0, 10], [350, 0, 0, 35], [650, 0, 0, 65]]
    curve = costs.price_curve(tc, PRICES)
    assert curve == pytest.approx([(100 + 100) / 1e6, (350 + 350) / 1e6, (650 + 650) / 1e6])
    with pytest.raises(ValueError):                         # a Claude run without an aligned ledger is refused
        costs.token_curve(traj, [], model="claude-x", allow_atif_fallback=True)
    atif = costs.token_curve(traj, [], model="gpt-x", allow_atif_fallback=True)
    assert atif["source"] == "atif" and atif["tok"][-1].tolist() == [300, 0, 0, 30]


def test_judge_cost_conventions():
    gpt = {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 400}           # GPT: cached is a subset of prompt
    assert costs.judge_cost_usd(gpt, "gpt-x", PRICES) == pytest.approx((1000 * 1.0 + 100 * 10.0) / 1e6)
    assert costs.judge_cost_usd(gpt, "gpt-x", PRICES, "cache") == pytest.approx((600 * 1.0 + 400 * 0.1 + 100 * 10.0) / 1e6)
    claude = {"prompt_tokens": 10, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 1000, "completion_tokens": 50}
    assert costs.judge_cost_usd(claude, "claude-x", PRICES) == pytest.approx((1110 * 2.0 + 50 * 20.0) / 1e6)
    attempts = [{"usage": {"prompt_tokens": 5, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 2}}}, {"usage": None}]
    assert costs.usage_sum(attempts)["cached_tokens"] == 2


def synthetic_cell(seed=0, n_repos=12, per_repo=4):
    rng = np.random.default_rng(seed)
    runs, member_scores, judge_scores, judge_cost = [], [{}, {}], {}, {}
    for r in range(n_repos):
        for i in range(per_repo):
            task = f"org__repo{r}-{i}"
            reward = float(rng.random() < 0.6)
            n = int(rng.integers(6, 30))
            runs.append({"task": task, "reward": reward, "spend": list(np.cumsum(rng.random(n))) + [0.0], "n": n})
            runs[-1]["spend"][-1] = runs[-1]["spend"][-2] + 0.1
            for m in member_scores:
                m[task] = {k: float(np.clip(0.5 + (0.2 if reward == 0 else -0.2) + rng.normal(0, 0.25), 0, 1)) for k in range(2, n + 1)}
            judge_scores[task] = {k: float(rng.random()) for k in range(2, n + 1)}
            judge_cost.update({(task, k): 0.01 for k in range(2, n + 1)})
    cal = [{"task": f"cal{i}", "reward": 1.0, "scores": {5: float(rng.random()), 8: float(rng.random())}} for i in range(40)]
    return runs, member_scores, judge_scores, judge_cost, cal


def test_evaluate_cell_end_to_end():
    runs, (s0, s1), js, jc, cal = synthetic_cell()
    methods = {"m_s0": {"scores": s0, "seconds_per_check": 0.2}, "m_s1": {"scores": s1, "seconds_per_check": 0.3},
               "judge:j": {"scores": js, "window_cost": jc}}
    calibration = {"m_s0": cal, "m_s1": cal, "judge:j": cal}
    index, mult = bootstrap.bench_multiplicities([r["task"] for r in runs], "repo", 200, 0, "bench")
    out = evaluate_cell(runs, methods, calibration, index, mult, groups={"m": ["m_s0", "m_s1"]}, primary="m",
                        gpu_usd_per_hour=3.6, fixed_steps=[5, 8], fixed_step_methods=["m_ens"], timing_method="m_ens", log=lambda *a: None)
    ranking, stopping = out["ranking"]["methods"], out["stopping"]["methods"]
    assert set(ranking) >= {"m_s0", "m_s1", "m_ens", "m_seedmean", "judge:j", "stepcount"}
    assert ranking["m_seedmean"]["tier_auc"] == pytest.approx((ranking["m_s0"]["tier_auc"] + ranking["m_s1"]["tier_auc"]) / 2)
    ens = stopping["m_ens"]["0.05"]
    assert ens["monitor_usd"] == pytest.approx(ens["calls"] * 0.5 * 3.6 / 3600)      # an ensemble pays both members
    judge = stopping["judge:j"]["0.05"]
    assert judge["monitor_usd"] == pytest.approx(0.01 * judge["calls"])
    lo, hi = ens["ci95"]["net_delta_pct"]
    assert lo <= ens["net_delta_pct"] <= hi
    assert "judge:j" in out["paired_vs_primary"]
    assert [x["k"] for x in out["fixed_step_auc"]["m_ens"]] == [5, 8]
    assert out["timing"]["correct_stops"] == int(stopping["m_ens"]["0.05"]["kills"] - stopping["m_ens"]["0.05"]["false_kills"])
    assert out["n_checkpoint_windows"] == sum(len([k for k in range(2, r["n"] + 1) if policy.is_checkpoint(k)]) for r in runs)
