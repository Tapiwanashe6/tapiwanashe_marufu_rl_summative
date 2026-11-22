"""Smoke tests: every TorchRL agent trains briefly, logs, checkpoints and reloads.

These run each algorithm for a few thousand frames into a temporary directory;
they check plumbing and finiteness, not learning quality.
"""
import csv
import json
import math

import numpy as np
import pytest
import torch

import training.common as common
from environment import MediFlightEnv
from training import dqn_training, pg_training
from training.pg_training import discounted_returns


@pytest.fixture()
def tmp_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(common, "MODEL_ROOT", tmp_path / "models")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    cfg = tmp_path / "smoke.toml"
    cfg.write_text('name = "smoke"\n[hyperparameters]\neval_interval_frames = 1000\neval_episodes = 2\n')
    return tmp_path, cfg


def _check_run(root, algo, summary, loss_key, score_key):
    run_dir = root / "logs" / algo / "smoke" / "seed_0"
    rows = list(csv.DictReader(open(run_dir / "train.csv")))
    assert rows, "no training rows logged"
    losses = [float(r[loss_key]) for r in rows if r.get(loss_key) not in (None, "")]
    assert losses and all(math.isfinite(v) for v in losses), f"{loss_key} missing or non-finite"
    assert list(csv.DictReader(open(run_dir / "eval.csv")))
    assert json.loads((run_dir / "config.json").read_text())["seed"] == 0
    assert summary["frames"] >= 2000
    net = common.load_score_net(root / summary["model_path"], score_key)
    env = MediFlightEnv()
    obs, _ = env.reset(seed=0)
    a = common.greedy_act_fn(net)(env, obs)
    assert obs["action_mask"][a] == 1, "greedy action must respect the mask"


def test_dqn_smoke(tmp_roots):
    root, cfg = tmp_roots
    s = dqn_training.train(str(cfg), 0, {"total_frames": 3000, "init_random_frames": 500,
                                         "buffer_size": 5000, "batch_size": 32})
    _check_run(root, "dqn", s, "loss", "q_net")
    rows = list(csv.DictReader(open(root / "logs/dqn/smoke/seed_0/train.csv")))
    assert any(r["mean_q"] not in ("", None) for r in rows)


@pytest.mark.parametrize("algo,over", [
    ("ppo", {"total_frames": 4096, "frames_per_batch": 1024, "minibatch_size": 256}),
    ("a2c", {"total_frames": 2048}),
    ("reinforce", {"total_frames": 2000, "episodes_per_batch": 4}),
    ("reinforce", {"total_frames": 2000, "episodes_per_batch": 4, "use_baseline": True}),
])
def test_policy_gradient_smoke(tmp_roots, algo, over):
    root, cfg = tmp_roots
    s = pg_training.train(str(cfg), algo, 0, over)
    _check_run(root, algo, s, "loss_objective", "actor")
    rows = list(csv.DictReader(open(root / f"logs/{algo}/smoke/seed_0/train.csv")))
    ent = [float(r["entropy"]) for r in rows]
    assert all(0.0 <= e <= math.log(25) + 1e-6 for e in ent), "entropy outside [0, log|A|]"


def test_training_is_reproducible(tmp_roots):
    root, cfg = tmp_roots
    over = {"total_frames": 2048, "frames_per_batch": 512, "minibatch_size": 128}
    pg_training.train(str(cfg), "ppo", 0, over)
    a = (root / "logs/ppo/smoke/seed_0/train.csv").read_text().splitlines()
    pg_training.train(str(cfg), "ppo", 0, over)
    b = (root / "logs/ppo/smoke/seed_0/train.csv").read_text().splitlines()
    strip = lambda lines: [",".join(l.split(",")[1:]) for l in lines]  # drop wall-clock column
    assert strip(a) == strip(b)


def test_discounted_returns():
    r = torch.tensor([[1.0], [0.0], [2.0]])
    g = discounted_returns(r, 0.5)
    np.testing.assert_allclose(g.squeeze(-1).numpy(), [1 + 0.25 * 2, 0.5 * 2, 2.0])


def test_unknown_hyperparameter_is_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "bad"\n[hyperparameters]\nlearning_rate = 0.1\n')
    with pytest.raises(ValueError, match="unknown hyperparameters"):
        common.load_config(bad, pg_training.DEFAULTS["ppo"])


def test_all_experiment_configs_are_valid():
    from pathlib import Path
    files = sorted(Path(common.ROOT, "configs").glob("*/*.toml"))
    per_algo = {}
    for f in files:
        algo = f.parent.name
        d = dqn_training.DEFAULTS if algo == "dqn" else pg_training.DEFAULTS[algo]
        cfg = common.load_config(f, d)
        common.env_config(cfg["env"])
        per_algo[algo] = per_algo.get(algo, 0) + 1
    assert set(per_algo) == {"dqn", "ppo", "a2c", "reinforce"}
    assert all(n >= 10 for n in per_algo.values()), per_algo
