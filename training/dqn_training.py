"""Deep Q-Network on MediFlightEnv with TorchRL.

    uv run python -m training.dqn_training --config configs/dqn/exp01_baseline.toml

Components (all TorchRL):
* QValueActor over the 108-D observation with the action mask applied before the
  argmax, so neither acting nor the bootstrap target ever selects an invalid action.
* EGreedyModule with linearly annealed epsilon; random actions are drawn from the
  masked action spec.
* Collector + TensorDictReplayBuffer (uniform sampling, LazyTensorStorage).
* DQNLoss (optionally Double DQN) with a target network updated by HardUpdate
  (every N gradient steps) or SoftUpdate (Polyak averaging, tau).
"""
from __future__ import annotations

import argparse
from typing import Any

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential

from environment import N_ACTIONS
from training.common import (RunLogger, env_config, episode_stats_from_batch, load_config,
                             make_torchrl_env, mlp, set_seed)

DEFAULTS: dict[str, Any] = {
    "lr": 5e-4,
    "gamma": 0.99,
    "buffer_size": 100_000,
    "batch_size": 128,
    "frames_per_batch": 16,         # env frames collected between update rounds
    "updates_per_batch": 4,         # gradient steps per round (replay ratio 1:4)
    "init_random_frames": 5_000,    # uniform random (masked) actions to warm up the buffer
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_frames": 100_000,
    "target_update": "hard",        # "hard" | "soft"
    "target_update_interval": 1_000,  # gradient steps between hard updates
    "tau": 0.005,                   # soft update rate
    "double_dqn": True,
    "loss_function": "smooth_l1",
    "max_grad_norm": 10.0,
}


def train(config_path: str, seed: int | None = None, overrides: dict | None = None) -> dict:
    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.modules import EGreedyModule, QValueActor
    from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

    cfg = load_config(config_path, DEFAULTS)
    if seed is not None:
        cfg["seed"] = seed
    cfg.update(overrides or {})
    set_seed(cfg["seed"])
    ecfg = env_config(cfg["env"])
    env = make_torchrl_env(ecfg, cfg["seed"], 1)

    q_net = mlp(N_ACTIONS, cfg["hidden"], cfg["activation"])
    qvalue = QValueActor(TensorDictModule(q_net, in_keys=["observation"], out_keys=["action_value"]),
                         in_keys=["observation"], spec=env.action_spec,
                         action_space="categorical", action_mask_key="action_mask")
    egreedy = EGreedyModule(env.action_spec, eps_init=cfg["eps_start"], eps_end=cfg["eps_end"],
                            annealing_num_steps=cfg["eps_decay_frames"],
                            action_mask_key="action_mask")
    explore_policy = TensorDictSequential(qvalue, egreedy)

    loss_fn = DQNLoss(qvalue, loss_function=cfg["loss_function"], delay_value=True,
                      double_dqn=cfg["double_dqn"], action_space=env.action_spec)
    loss_fn.make_value_estimator(gamma=cfg["gamma"])
    if cfg["target_update"] == "hard":
        target = HardUpdate(loss_fn, value_network_update_interval=cfg["target_update_interval"])
    else:
        target = SoftUpdate(loss_fn, tau=cfg["tau"])
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]),
                                    batch_size=cfg["batch_size"])
    collector = Collector(env, explore_policy, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=cfg["total_frames"],
                          init_random_frames=cfg["init_random_frames"])

    log = RunLogger(cfg, "dqn", "dqn")
    modules = {"q_net": q_net}
    frames = episodes = grad_steps = 0
    next_eval = 0
    window: dict[str, list[float]] = {"loss": [], "q": [], "td": [], "gn": []}
    ep_window: list[dict] = []
    log_every = min(2_000, cfg["eval_interval_frames"])
    next_log = log_every
    for batch in collector:
        frames += batch.numel()
        stats = episode_stats_from_batch(batch)
        if stats:
            ep_window.append(stats)
            episodes += stats["train_episodes"]
        buffer.extend(batch.reshape(-1))
        if frames > cfg["init_random_frames"]:
            # Epsilon is annealed per collected frame after the warm-up period.
            egreedy.step(batch.numel())
            for _ in range(cfg["updates_per_batch"]):
                sample = buffer.sample()
                out = loss_fn(sample)
                optim.zero_grad()
                out["loss"].backward()
                gn = torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["max_grad_norm"])
                optim.step()
                target.step()
                grad_steps += 1
                window["loss"].append(float(out["loss"].detach()))
                window["gn"].append(float(gn))
                with torch.no_grad():
                    # Q of the taken action under the *current* network (the value stored
                    # in the buffer was computed at collection time and is stale).
                    q_now = q_net(sample["observation"]).gather(-1, sample["action"].long().unsqueeze(-1))
                    window["q"].append(float(q_now.mean()))
                    window["td"].append(float(sample["td_error"].mean()))
        collector.update_policy_weights_()
        if frames >= next_log:
            n_ep = sum(s["train_episodes"] for s in ep_window)
            row = {"frames": frames, "episodes": episodes, "grad_steps": grad_steps,
                   "epsilon": round(float(egreedy.eps), 4)}
            if window["loss"]:
                row.update(loss=sum(window["loss"]) / len(window["loss"]),
                           mean_q=sum(window["q"]) / len(window["q"]),
                           td_error_sq=sum(window["td"]) / len(window["td"]),
                           grad_norm=sum(window["gn"]) / len(window["gn"]))
            if n_ep:
                row.update(train_episodes=n_ep,
                           train_return=sum(s["train_return"] * s["train_episodes"] for s in ep_window) / n_ep,
                           train_ep_len=sum(s["train_ep_len"] * s["train_episodes"] for s in ep_window) / n_ep)
            log.log("train", row)
            window = {k: [] for k in window}
            ep_window = []
            next_log += log_every
        if frames >= next_eval:
            log.evaluate_and_checkpoint(frames, episodes, q_net, modules, ecfg)
            next_eval += cfg["eval_interval_frames"]
    collector.shutdown()
    log.evaluate_and_checkpoint(frames, episodes, q_net, modules, ecfg)
    return log.finish(frames, episodes, {"grad_steps": grad_steps})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--total-frames", type=int, help="override (smoke tests)")
    a = p.parse_args()
    s = train(a.config, a.seed, {"total_frames": a.total_frames} if a.total_frames else None)
    print(f"[dqn] {s['name']} seed={s['seed']} frames={s['frames']} grad_steps={s['grad_steps']} "
          f"best_val_return={s['best_val_return']:.2f} @ {s['best_frames']} | {s['wall_s']}s")


if __name__ == "__main__":
    main()
