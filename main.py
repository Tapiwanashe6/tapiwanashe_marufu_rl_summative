"""MediFlight AI entry point: `uv run main.py`.

By default this runs the best trained agent (selected on validation data, see
play.py): every decision is printed to the terminal and the episode opens in the
3D mission-control viewer.

    uv run main.py                         # best trained agent + 3D viewer
    uv run main.py --algo dqn --seed 7     # any play.py option is passed through
    uv run main.py --policy edf            # reference baseline, terminal only
"""
from __future__ import annotations

import argparse

import numpy as np

from environment import MediFlightEnv
from environment.baselines import BASELINES


def run_baseline(policy: str, seed: int) -> None:
    env = MediFlightEnv(render_mode="ansi")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    print(f"MediFlight AI | policy={policy} (reference baseline, not a trained agent) | seed={seed}")
    print(env.render())
    total, done, info = 0.0, False, {}
    while not done:
        _, reward, terminated, truncated, info = env.step(BASELINES[policy](env, rng))
        total += reward
        done = terminated or truncated
        print(f"step {env.steps:03d} | t={info['time_min']:6.1f} | {info['action_name']:<26} "
              f"| r={reward:+7.2f} | battery={info['battery']:5.1f}% | {info['weather']:5s} "
              f"| on-time={info['delivered_on_time']:.0f} expired={info['expired']:.0f}")
    print(f"\nEpisode return {total:.2f} | on-time {info['delivered_on_time']:.0f} / "
          f"{info['orders_created']:.0f} orders | late {info['delivered_late']:.0f} | "
          f"expired {info['expired']:.0f} | crashed {bool(info['crashes'])}")
    env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["agent", *sorted(BASELINES)], default="agent",
                   help="'agent' (default) runs the best trained agent via play.py")
    p.add_argument("--seed", type=int, default=42)
    args, rest = p.parse_known_args()
    if args.policy != "agent":
        if rest:
            p.error(f"unrecognised arguments for a baseline: {' '.join(rest)}")
        run_baseline(args.policy, args.seed)
        return
    import play
    play.main(["--seed", str(args.seed), *rest])


if __name__ == "__main__":
    main()
