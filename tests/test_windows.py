from conftest import write_trajectory

from stoploss import sequence, windows


def test_pairs_labels_and_prompt(tmp_path, tok):
    traj = write_trajectory(tmp_path / "t.json", [("look", "ls\n", "a.py"), ("read", "cat a.py\n", "x" * 2000),
                                                   ("edit", "sed -i s/a/b/ a.py\n", "ok"), ("test", "pytest\n", "1 passed")])
    pairs = windows.step_pairs(traj)
    assert len(pairs) == 4                                   # the final agent message without a tool call is not a step
    assert pairs[0] == ("look\nls", "a.py")
    assert len(pairs[1][1]) == windows.OBSERVATION_CHARS     # observations are cut at 900 characters
    solved = windows.run_windows(traj, 1.0, tok, max_len=100_000)
    assert [(w["k"], w["label"]) for w in solved] == [(2, "CONTINUE"), (3, "DELIVER"), (4, "DELIVER")]
    assert {w["label"] for w in windows.run_windows(traj, 0.0, tok, max_len=100_000)} == {"ABORT"}
    prompt = solved[-1]["prompt"]
    assert prompt.startswith("An autonomous agent has run 4 steps on a task. Its last steps:\n<action>\nread\n")
    assert prompt.endswith(windows.QUESTION_SUFFIX)
    assert prompt.count("<action>") == windows.RECENT_STEPS


def test_middle_omission_fits_budget(tmp_path, tok):
    traj = write_trajectory(tmp_path / "t.json", [(f"step {i} " + "r" * 600, "cmd\n", "o" * 900) for i in range(6)])
    pairs = windows.step_pairs(traj)
    budget = 2000
    prompt, omitted = windows.fit_to_budget(pairs, 6, tok, budget)
    assert omitted and windows.OMISSION_MARK in prompt
    assert prompt.startswith("An autonomous agent has run 6 steps on a task. Its last steps:\n")
    assert prompt.endswith("\n" + windows.QUESTION)
    assert sequence.seq_len_max(tok, prompt) <= budget


def test_encoding_masks_prompt(tok):
    enc = sequence.encode_train(tok, "hello", "ABORT", 1000)
    n_prompt = len(sequence.render(tok, "hello"))
    assert enc["labels"][:n_prompt] == [-100] * n_prompt
    assert enc["labels"][n_prompt:] == [ord(c) for c in "ABORT"]
    assert sequence.first_token_ids(tok) == [ord("C"), ord("D"), ord("A")]
