"""Generate the report's §4 hyperparameter tables (template column format) from results.

    uv run python -m evaluation.report_tables     # writes results/hyperparameter_tables.md

"Mean Reward" is the mean validation return over 50 held-out validation
episodes, averaged across training seeds (± std across seeds). It is never a
training-reward number.
"""
from __future__ import annotations

import json

from evaluation.compare import RESULTS
from training import dqn_training, pg_training
from training.common import COMMON_DEFAULTS, ROOT, load_config


def _cfg(algo: str, name: str) -> dict:
    defaults = dqn_training.DEFAULTS if algo == "dqn" else pg_training.DEFAULTS[algo]
    return load_config(ROOT / "configs" / algo / f"{name}.toml", defaults)


def _g(x: float) -> str:
    return f"{x:g}"


def _explore(c: dict) -> str:
    return f"ε-greedy {c['eps_start']:g}→{c['eps_end']:g} over {c['eps_decay_frames'] // 1000}k frames"


def _target(c: dict) -> str:
    return (f"hard / {c['target_update_interval']} steps" if c["target_update"] == "hard"
            else f"soft τ={c['tau']:g}") + ("" if c["double_dqn"] else ", vanilla")


COLUMNS = {
    "dqn": [("Learning Rate", lambda c: _g(c["lr"])), ("Gamma", lambda c: _g(c["gamma"])),
            ("Replay Buffer Size", lambda c: f"{c['buffer_size']:,}"),
            ("Batch Size", lambda c: str(c["batch_size"])),
            ("Exploration Strategy", _explore), ("Target Update", _target)],
    "reinforce": [("Learning Rate", lambda c: _g(c["lr"])), ("Gamma", lambda c: _g(c["gamma"])),
                  ("Episodes / Update", lambda c: str(c["episodes_per_batch"])),
                  ("Baseline", lambda c: "V(s)" if c["use_baseline"] else "none"),
                  ("Return Norm.", lambda c: "yes" if c["normalize_returns"] else "no"),
                  ("Entropy Coef.", lambda c: _g(c["entropy_coeff"]))],
    "ppo": [("Learning Rate", lambda c: _g(c["lr"])), ("Clip ε", lambda c: _g(c["clip_epsilon"])),
            ("Epochs", lambda c: str(c["ppo_epochs"])), ("GAE λ", lambda c: _g(c["gae_lambda"])),
            ("Entropy Coef.", lambda c: _g(c["entropy_coeff"])),
            ("Rollout / Minibatch", lambda c: f"{c['frames_per_batch']} / {c['minibatch_size']}")],
    "a2c": [("Learning Rate", lambda c: _g(c["lr"])),
            ("Rollout (envs × steps)", lambda c: f"{c['num_envs']} × {c['frames_per_batch'] // c['num_envs']}"),
            ("GAE λ", lambda c: _g(c["gae_lambda"])), ("Entropy Coef.", lambda c: _g(c["entropy_coeff"])),
            ("Critic Coef.", lambda c: _g(c["critic_coeff"])),
            ("Adv. Norm.", lambda c: "yes" if c["normalize_advantage"] else "no")],
}
TITLES = {"dqn": "4.1 DQN", "reinforce": "4.2 REINFORCE", "ppo": "4.3 PPO", "a2c": "4.4 A2C (Actor-Critic)"}


def build() -> str:
    tables = json.loads((RESULTS / "config_tables.json").read_text())
    best = json.loads((RESULTS / "best_configs.json").read_text())
    out = [f"Budget: {COMMON_DEFAULTS['total_frames']:,} environment frames per run, "
           "3 training seeds per configuration. Mean Reward = mean validation return over 50 "
           "held-out validation episodes (± std across training seeds). All runs share γ=0.99 "
           "and a 2×128 tanh MLP unless stated. Bold = selected configuration.\n"]
    for algo in ("dqn", "reinforce", "ppo", "a2c"):
        cols = COLUMNS[algo]
        out.append(f"### {TITLES[algo]}\n")
        out.append("| # | " + " | ".join(h for h, _ in cols) +
                   " | Mean Reward | On-time | Critical svc. | Crash % | Notes |")
        out.append("|" + "---|" * (len(cols) + 6))
        for row in tables[algo]:
            c = _cfg(algo, row["name"])
            num = row["name"][3:5]
            mark = "**" if row["name"] == best[algo]["name"] else ""
            note = "no action mask" if "no_mask" in row["name"] else row["description"].split(":")[0]
            if "no_mask" in row["name"]:
                note += f"; invalid actions/ep {row.get('invalid_actions_mean', 0):.1f}"
            out.append(f"| {mark}{num}{mark} | " + " | ".join(f(c) for _, f in cols) +
                       f" | {mark}{row['return_mean']:.2f} ± {row['return_std']:.2f}{mark}"
                       f" | {row['delivered_on_time_mean']:.2f} | {row['critical_service_rate_mean']:.3f}"
                       f" | {100 * row['crashed_mean']:.1f} | {note} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    text = build()
    (RESULTS / "hyperparameter_tables.md").write_text(text)
    print(text)
