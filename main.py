"""MediFlight AI entry point: `uv run main.py`.

Currently runs one verbose episode of a reference policy so the environment can
be inspected end to end. Once training is complete this becomes the launcher for
the best trained agent (see play.py).
"""
from __future__ import annotations

import argparse

import numpy as np

from environment import MediFlightEnv
from environment.baselines import BASELINES


def main() -> None:
    p = argparse.ArgumentParser(description="Run one verbose MediFlight AI episode.")
    p.add_argument("--policy", choices=sorted(BASELINES), default="edf")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    env = MediFlightEnv(render_mode="ansi")
    env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed + 10_000)
    print(f"MediFlight AI | policy={args.policy} (reference baseline, not a trained agent) "
          f"| seed={args.seed}")
    print(env.render())
    total, done, info = 0.0, False, {}
    while not done:
        _, reward, terminated, truncated, info = env.step(BASELINES[args.policy](env, rng))
        total += reward
        done = terminated or truncated
        print(f"step {env.steps:03d} | t={info['time_min']:6.1f} | {info['action_name']:<26} "
              f"| r={reward:+7.2f} | battery={info['battery']:5.1f}% | {info['weather']:5s} "
              f"| on-time={info['delivered_on_time']:.0f} expired={info['expired']:.0f}")
    print(f"\nEpisode return {total:.2f} | on-time {info['delivered_on_time']:.0f} / "
          f"{info['orders_created']:.0f} orders | late {info['delivered_late']:.0f} | "
          f"expired {info['expired']:.0f} | crashed {bool(info['crashes'])}")
    env.close()


if __name__ == "__main__":
    main()
