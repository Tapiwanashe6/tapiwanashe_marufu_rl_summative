"""Event serialisation for the Three.js mission viewer.

The environment itself emits fine-grained, timestamped domain events
(take-off, arrival, delivery, expiry, weather change, ...) when constructed with
`record_events=True`.  `EpisodeRecorder` wraps a stepping loop and turns each
decision into one JSON-serialisable *record*:

    {"type": "episode_start", "schema_version", "metadata", "map", "config", "state"}
    {"type": "step", "step", "action", "action_name", "reward", "cumulative_reward",
     "t_start", "t_end", "events": [...], "state": {...}, "policy": {...}}
    {"type": "episode_end", "summary": {...}}

Records go to one or more *sinks*.  `FileSink` writes a replay file the viewer
loads; a live sink (e.g. Server-Sent Events) can stream the very same records,
so the browser never needs to know which transport produced them.  The viewer
only animates what is in these records: it has no simulation logic of its own.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from . import entities as E
from .custom_env import MediFlightEnv, action_name

SCHEMA_VERSION = "2.0"


class EventSink(Protocol):
    def emit(self, record: dict[str, Any]) -> None: ...
    def close(self) -> None: ...


class FileSink:
    """Collects records and writes a single replay JSON file on close()."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.header: dict | None = None
        self.steps: list[dict] = []
        self.summary: dict | None = None

    def emit(self, record: dict[str, Any]) -> None:
        kind = record["type"]
        if kind == "episode_start":
            self.header, self.steps, self.summary = record, [], None
        elif kind == "step":
            self.steps.append(record)
        elif kind == "episode_end":
            self.summary = record["summary"]

    def document(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "header": self.header,
                "steps": self.steps, "summary": self.summary}

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.document(), separators=(",", ":")), encoding="utf-8")


class MemorySink(FileSink):
    """Like FileSink but never touches disk (tests, notebooks)."""

    def __init__(self):
        super().__init__(path="")

    def close(self) -> None:
        pass


def map_description() -> dict[str, Any]:
    return {"units": {"distance": "km", "time": "min", "energy": "% battery", "mass": "kg"},
            "nodes": [{"id": i, "name": n.name, "kind": n.kind, "x": n.x, "y": n.y,
                       "charger": i == 0} for i, n in enumerate(E.NODES)],
            "package_types": [{"name": p.name, "urgency": p.urgency, "value": p.value,
                               "cold_chain_min": p.cold_chain_min} for p in E.PACKAGE_TYPES],
            "weather": list(E.WEATHER_NAMES)}


class EpisodeRecorder:
    """Drives an environment and publishes one record per decision to sinks."""

    def __init__(self, env: MediFlightEnv, sinks: list[EventSink], metadata: dict | None = None):
        if not env.record_events:
            raise ValueError("EpisodeRecorder requires MediFlightEnv(record_events=True)")
        self.env, self.sinks = env, sinks
        self.metadata = metadata or {}
        self.cumulative = 0.0

    def _publish(self, record: dict) -> None:
        for s in self.sinks:
            s.emit(record)

    def reset(self, seed: int):
        obs, info = self.env.reset(seed=seed)
        self.cumulative = 0.0
        self._publish({"type": "episode_start", "schema_version": SCHEMA_VERSION,
                       "metadata": {"seed": seed, **self.metadata}, "map": map_description(),
                       "config": self.env.config_dict(), "events": info.get("events", []),
                       "state": self.env.snapshot()})
        return obs, info

    def step(self, action: int, policy_info: dict | None = None):
        t_start = self.env.time
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.cumulative += reward
        self._publish({"type": "step", "step": self.env.steps - 1, "action": int(action),
                       "action_name": action_name(action), "valid": info["valid_action"],
                       "reward": round(reward, 4), "cumulative_reward": round(self.cumulative, 4),
                       "reward_breakdown": {k: round(v, 4) for k, v in info["reward_breakdown"].items()},
                       "t_start": round(t_start, 3), "t_end": round(self.env.time, 3),
                       "events": info["events"], "state": self.env.snapshot(),
                       "policy": policy_info or {}})
        if terminated or truncated:
            self._publish({"type": "episode_end", "summary": {
                "return": round(self.cumulative, 4), "steps": self.env.steps,
                "terminated": terminated, "truncated": truncated,
                "crashed": bool(self.env.stats["crashes"]), **self.env.snapshot()["stats"]}})
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        for s in self.sinks:
            s.close()
