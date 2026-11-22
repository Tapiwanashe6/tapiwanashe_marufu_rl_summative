"""Shared training infrastructure for all four TorchRL agents.

* Config loading (TOML) with per-algorithm defaults, resolved config saved per run.
* TorchRL environment factory (GymWrapper + transforms) used by every algorithm.
* Networks, seeding, CSV logging, periodic validation and checkpointing.

Every algorithm trains on the same MediFlightEnv, observes the same 108-D
vector and action mask, uses the same network sizes by default, and is
evaluated by the same deterministic protocol (evaluation/protocol.py).
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
import tomllib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from environment import EnvConfig, MediFlightEnv, N_ACTIONS
from environment.observation_spec import OBS_DIM
from evaluation.protocol import VALIDATION_SEEDS, run_episodes, summarize

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs"
MODEL_ROOT = ROOT / "models"
ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}

COMMON_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "total_frames": 500_000,     # identical environment-frame budget for every algorithm
    "eval_interval_frames": 20_000,
    "eval_episodes": 20,          # validation seeds used for periodic checkpoint selection
    "hidden": [128, 128],
    "activation": "tanh",
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "lr": 3e-4,
}


# ------------------------------------------------------------------ config
def load_config(path: str | Path, defaults: dict[str, Any]) -> dict[str, Any]:
    """Merge COMMON_DEFAULTS <- algorithm defaults <- TOML file."""
    raw = tomllib.loads(Path(path).read_text())
    hp = raw.get("hyperparameters", {})
    unknown = set(hp) - set(COMMON_DEFAULTS) - set(defaults)
    if unknown:  # a typo must never silently fall back to a default
        raise ValueError(f"{path}: unknown hyperparameters {sorted(unknown)}")
    cfg = {**COMMON_DEFAULTS, **defaults, **hp}
    cfg["name"] = raw.get("name", Path(path).stem)
    cfg["description"] = raw.get("description", "")
    cfg["env"] = raw.get("env", {})
    cfg["config_path"] = str(Path(path).resolve().relative_to(ROOT)) if Path(path).resolve().is_relative_to(ROOT) else str(path)
    return cfg


def env_config(overrides: dict[str, Any]) -> EnvConfig:
    valid = {f.name for f in fields(EnvConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown env overrides: {unknown}")
    return replace(EnvConfig(), **overrides)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # deterministic CPU kernels; sweeps parallelise across processes


# --------------------------------------------------------------- TorchRL env
def make_torchrl_env(env_cfg: EnvConfig, seed: int, num_envs: int = 1):
    """GymWrapper(MediFlightEnv) with bool action mask, episode reward and step count."""
    from torchrl.envs import (ActionMask, Compose, DTypeCastTransform, GymWrapper, RewardSum,
                              SerialEnv, StepCounter, TransformedEnv)

    def _one():
        return GymWrapper(MediFlightEnv(env_cfg), device="cpu", categorical_action_encoding=True)

    base = _one() if num_envs == 1 else SerialEnv(num_envs, _one)
    env = TransformedEnv(base, Compose(
        DTypeCastTransform(torch.int8, torch.bool, in_keys=["action_mask"], in_keys_inv=[]),
        ActionMask(),
        RewardSum(),
        StepCounter(),
    ))
    env.set_seed(seed)
    return env


def mlp(out_features: int, hidden: list[int], activation: str) -> nn.Module:
    from torchrl.modules import MLP
    return MLP(in_features=OBS_DIM, out_features=out_features, num_cells=list(hidden),
               activation_class=ACTIVATIONS[activation])


# ---------------------------------------------------- greedy evaluation
def greedy_act_fn(score_net: nn.Module):
    """Deterministic policy: argmax of masked Q-values (DQN) or logits (PG)."""
    @torch.no_grad()
    def act(env: MediFlightEnv, obs: dict) -> int:
        x = torch.as_tensor(obs["observation"]).unsqueeze(0)
        scores = score_net(x).squeeze(0)
        mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
        return int(torch.where(mask, scores, torch.finfo(scores.dtype).min).argmax())
    return act


def validate(score_net: nn.Module, env_cfg: EnvConfig, n_episodes: int) -> dict:
    was_training = score_net.training
    score_net.eval()
    rows = run_episodes(lambda s: greedy_act_fn(score_net), VALIDATION_SEEDS[:n_episodes], env_cfg)
    score_net.train(was_training)
    return summarize(rows)


# ------------------------------------------------------------- run logging
class RunLogger:
    """Writes train.csv, eval.csv, config.json and summary.json for one (config, seed) run."""

    def __init__(self, cfg: dict[str, Any], algo: str, model_dir: str):
        self.algo, self.cfg = algo, cfg
        self.dir = LOG_ROOT / algo / cfg["name"] / f"seed_{cfg['seed']}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model_path = MODEL_ROOT / model_dir / f"{algo}_{cfg['name']}_seed{cfg['seed']}.pt"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
        self._rows: dict[str, list[dict]] = {}
        self._fields: dict[str, list[str]] = {}
        self.t0 = time.perf_counter()
        self.best_val = -float("inf")
        self.best_frames = 0
        self.eval_history: list[dict] = []

    def log(self, stream: str, row: dict[str, Any]) -> None:
        """Append a row; if it introduces new columns the file is rewritten so no
        value is ever silently dropped (e.g. losses that only exist after warm-up)."""
        row = {"wall_s": round(time.perf_counter() - self.t0, 2), **row}
        rows = self._rows.setdefault(stream, [])
        fields_ = self._fields.setdefault(stream, [])
        rows.append(row)
        new = [k for k in row if k not in fields_]
        path = self.dir / f"{stream}.csv"
        if new or len(rows) == 1:
            fields_.extend(new)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields_, restval="")
                w.writeheader()
                w.writerows(rows)
        else:
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields_, restval="").writerow(row)

    def evaluate_and_checkpoint(self, frames: int, episodes: int, score_net: nn.Module,
                                modules: dict[str, nn.Module], env_cfg: EnvConfig) -> dict:
        s = validate(score_net, env_cfg, self.cfg["eval_episodes"])
        row = {"frames": frames, "episodes": episodes,
               **{k: round(v, 4) for k, v in s.items() if k.endswith(("_mean", "_ci95"))}}
        self.log("eval", row)
        self.eval_history.append(row)
        if s["return_mean"] > self.best_val:
            self.best_val, self.best_frames = s["return_mean"], frames
            torch.save({"algo": self.algo, "config": self.cfg, "frames": frames,
                        "val_return": s["return_mean"], "obs_dim": OBS_DIM, "n_actions": N_ACTIONS,
                        "hidden": self.cfg["hidden"], "activation": self.cfg["activation"],
                        "state_dicts": {k: m.state_dict() for k, m in modules.items()}},
                       self.model_path)
        return s

    def finish(self, frames: int, episodes: int, extra: dict | None = None) -> dict:
        summary = {"algo": self.algo, "name": self.cfg["name"], "seed": self.cfg["seed"],
                   "frames": frames, "episodes": episodes,
                   "wall_s": round(time.perf_counter() - self.t0, 1),
                   "best_val_return": self.best_val, "best_frames": self.best_frames,
                   "final_val_return": self.eval_history[-1]["return_mean"] if self.eval_history else None,
                   "model_path": str(self.model_path.relative_to(ROOT)),
                   "torch": torch.__version__, "pid": os.getpid(), **(extra or {})}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def episode_stats_from_batch(batch) -> dict[str, float]:
    """Mean episode reward / length for episodes that finished inside this batch."""
    done = batch["next", "done"].squeeze(-1)
    if not done.any():
        return {}
    ep_r = batch["next", "episode_reward"].squeeze(-1)[done]
    ep_len = batch["next", "step_count"].squeeze(-1)[done].float()
    return {"train_episodes": int(done.sum()), "train_return": float(ep_r.mean()),
            "train_ep_len": float(ep_len.mean())}


def load_score_net(path: str | Path, key: str) -> nn.Module:
    """Rebuild the network that scores actions (Q-net for DQN, logits net for PG)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = mlp(ckpt["n_actions"], ckpt["hidden"], ckpt["activation"])
    net.load_state_dict(ckpt["state_dicts"][key])
    net.eval()
    return net
