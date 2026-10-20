import itertools
import random

import numpy as np
import pytest

from stoploss import bootstrap, metrics, policy


def row(task, reward, scores, n=20, step_cost=1.0, tail=0.5):
    spend = [step_cost * (k + 1) for k in range(n)] + [step_cost * n + tail]
    return {"task": task, "reward": reward, "scores": scores, "spend": spend, "n": n}


def test_checkpoints():
    assert [k for k in range(1, 21) if policy.is_checkpoint(k)] == [5, 8, 11, 14, 17, 20]


def test_calibration_kills_at_most_floor_rate_n():
    cal = [row(f"t{i}", 1.0, {5: i / 10, 8: 0.0}) for i in range(10)] + [row("fail", 0.0, {5: 0.99})]
    taus, n = policy.calibrate(cal, rates=(0.1, 0.25))
    assert n == 10
    assert taus[0.1] == pytest.approx(0.8 + 1e-9)          # 1 of 10 successes (score 0.9) at or above tau
    assert taus[0.25] == pytest.approx(0.7 + 1e-9)         # floor(2.5) = 2
    report = policy.calibration_report(cal, taus)
    assert report["rates"]["0.1"]["actual_calibration_false_kills"] == 1


def test_replay_and_cost_per_retained_solve():
    rows = [row("a", 0.0, {5: 0.9, 8: 0.9}),               # stopped at k=5: pays 5
            row("b", 1.0, {5: 0.1, 8: 0.95}),              # false kill at k=8: pays 8
            row("c", 1.0, {5: 0.1, 8: 0.1}),               # survives: pays 20.5
            row("d", 0.0, {})]                             # no checkpoint: survives, pays 20.5, no checks
    rep = policy.replay(rows, 0.9, check_cost=lambda r, k: 0.01)
    m = policy.stopping_metrics(rep)
    assert m["kills"] == 2 and m["false_kills"] == 1 and m["kept"] == 1
    assert m["spend"] == pytest.approx(5 + 8 + 20.5 + 20.5)
    assert m["monitor_usd"] == pytest.approx(0.01 * (1 + 2 + 2))
    base = (4 * 20.5) / 2
    assert m["net_delta_pct"] == pytest.approx(100 * ((54.0 + 0.05) / 1 / base - 1))
    assert m["failure_capture"] == pytest.approx(0.5) and m["retention"] == pytest.approx(0.5)
    w = policy.stopping_metrics(rep, np.array([2.0, 0.0, 1.0, 1.0]))     # bootstrap weights
    assert w["n"] == 4 and w["kills"] == 2 and w["false_kills"] == 0


def brute_auc(p, y):
    pos = [a for a, t in zip(p, y) if t == 0.0]
    neg = [b for b, t in zip(p, y) if t == 1.0]
    return sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))


def test_auc_with_ties_and_tier_auc():
    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(2, 40)
        p = np.array([rng.choice([0.1, 0.5, rng.random()]) for _ in range(n)])
        y = np.array([float(rng.random() < 0.5) for _ in range(n)])
        if 0 < y.sum() < n:
            assert metrics.auc(p, y) == pytest.approx(brute_auc(p, y))
    spend = np.arange(1, 9, dtype=float)
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.7, 0.4, 0.6])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    # quartiles of 1..8 split the windows in pairs; every pair holds one failure
    expected = np.mean([brute_auc(p[i:i + 2], y[i:i + 2]) for i in range(0, 8, 2)])
    assert metrics.tier_auc(spend, p, y) == pytest.approx(expected)


def test_repo_clusters_and_shared_draws():
    assert bootstrap.repo_of("django__django-11099") == "django__django"
    assert bootstrap.repo_of("instance_ansible__ansible-1234abcd-v1") == "ansible__ansible"
    tasks = ["a__x-1", "a__x-2", "b__y-1", "c__z-9"]
    i1, m1 = bootstrap.bench_multiplicities(tasks, "repo", 100, 0, "verified")
    i2, m2 = bootstrap.bench_multiplicities(list(reversed(tasks)), "repo", 100, 0, "verified")
    assert i1 == i2 and (m1 == m2).all() and (m1.sum(axis=1) == 3).all()
    _, m3 = bootstrap.bench_multiplicities(tasks, "repo", 100, 0, "pro")
    assert not (m1 == m3).all()                             # each benchmark has its own stream
