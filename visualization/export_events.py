"""Export replays of real MediFlightEnv episodes for the Three.js viewer.

    uv run python -m visualization.export_events                    # EDF, nearest-first, random on seed 42
    uv run python -m visualization.export_events --policy edf --seed 7

Each replay is written to visualization/episodes/<policy>_seed<k>.json and listed
in visualization/episodes/index.json, which the viewer reads to populate its
episode selector. Trained agents are exported through the same function
(`export_episode`) by play.py, so every replay uses one recorder and one schema
(environment/rendering.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from environment import EnvConfig, MediFlightEnv
from environment.baselines import BASELINES
from environment.rendering import EpisodeRecorder, FileSink

EPISODE_DIR = Path(__file__).resolve().parent / "episodes"
LABELS = {"edf": "EDF dispatcher (baseline)", "nearest_first": "Nearest-first (baseline)",
          "random": "Random (baseline)"}

# act(env, obs) -> (action, policy_info dict for the replay, e.g. action probabilities)
ActFn = Callable[[MediFlightEnv, dict], tuple[int, dict]]


def export_episode(act: ActFn, seed: int, policy_id: str, label: str, kind: str,
                   env_cfg: EnvConfig | None = None, out_dir: Path = EPISODE_DIR,
                   on_step: Callable[[dict], None] | None = None) -> dict:
    """Run one episode, write its replay file and register it in index.json."""
    env = MediFlightEnv(env_cfg or EnvConfig(), record_events=True)
    path = out_dir / f"{policy_id}_seed{seed}.json"
    sink = FileSink(path)
    rec = EpisodeRecorder(env, [sink], metadata={"policy": policy_id, "label": label, "kind": kind})
    obs, _ = rec.reset(seed)
    done = False
    while not done:
        action, pinfo = act(env, obs)
        obs, _, term, trunc, _ = rec.step(action, pinfo)
        if on_step:
            on_step(sink.steps[-1])
        done = term or trunc
    rec.close()
    doc = sink.document()
    _register(out_dir, path.name, policy_id, label, kind, seed, doc["summary"])
    return doc


def _register(out_dir: Path, file: str, policy_id: str, label: str, kind: str, seed: int,
              summary: dict) -> None:
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"episodes": []}
    index["episodes"] = [e for e in index["episodes"] if e["file"] != file]
    index["episodes"].append({
        "file": file, "policy": policy_id, "label": label, "kind": kind, "seed": seed,
        "return": round(summary["return"], 2), "delivered_on_time": summary["delivered_on_time"],
        "critical_on_time": summary["critical_on_time"], "critical_requests": summary["critical_requests"],
        "crashed": summary["crashed"]})
    # Trained agents first, then baselines; stable within each group.
    index["episodes"].sort(key=lambda e: (e["kind"] != "agent", e["policy"], e["seed"]))
    index_path.write_text(json.dumps(index, indent=2))


def baseline_act(name: str, seed: int) -> ActFn:
    rng = np.random.default_rng(seed + 10_000)
    return lambda env, obs: (BASELINES[name](env, rng), {})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=sorted(BASELINES), nargs="+",
                   default=["edf", "nearest_first", "random"])
    p.add_argument("--seed", type=int, nargs="+", default=[42])
    a = p.parse_args()
    for seed in a.seed:
        for name in a.policy:
            doc = export_episode(baseline_act(name, seed), seed, name, LABELS[name], "baseline")
            s = doc["summary"]
            print(f"{name:14s} seed={seed}: {len(doc['steps'])} decisions, return {s['return']:.1f}, "
                  f"on-time {s['delivered_on_time']:.0f}, crashed={s['crashed']}")
    print(f"Replays in {EPISODE_DIR}")


if __name__ == "__main__":
    main()
