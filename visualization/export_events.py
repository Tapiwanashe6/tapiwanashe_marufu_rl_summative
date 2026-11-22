"""Export a replay of real MediFlightEnv transitions for the Three.js viewer.

Run:  uv run python -m visualization.export_events --policy edf --seed 42

Trained-agent export is added with the training code (play.py); the baselines
here use exactly the same recorder and schema (environment/rendering.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from environment import MediFlightEnv
from environment.baselines import BASELINES
from environment.rendering import EpisodeRecorder, FileSink

DEFAULT_OUT = Path(__file__).resolve().parent / "events.json"


def export(seed: int = 42, policy: str = "edf", output: str | Path | None = None) -> dict:
    env = MediFlightEnv(record_events=True)
    sink = FileSink(output or DEFAULT_OUT)
    rec = EpisodeRecorder(env, [sink], metadata={"policy": policy, "kind": "baseline"})
    rec.reset(seed)
    rng = np.random.default_rng(seed + 10_000)
    done = False
    while not done:
        _, _, term, trunc, _ = rec.step(BASELINES[policy](env, rng))
        done = term or trunc
    rec.close()
    doc = sink.document()
    s = doc["summary"]
    print(f"Exported {len(doc['steps'])} decisions to {sink.path} | return={s['return']:.1f} "
          f"on-time={s['delivered_on_time']:.0f} expired={s['expired']:.0f} crashed={s['crashed']}")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policy", choices=sorted(BASELINES), default="edf")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    export(a.seed, a.policy, a.output)
