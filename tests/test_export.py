"""The replay file must be a faithful record of the real environment."""
import json

import numpy as np

from environment import MediFlightEnv
from environment.baselines import edf_policy
from environment.rendering import SCHEMA_VERSION, EpisodeRecorder, MemorySink
from visualization.export_events import export


def test_export_file_roundtrip(tmp_path):
    path = tmp_path / "events.json"
    doc = export(seed=42, policy="edf", output=path)
    on_disk = json.loads(path.read_text())
    assert on_disk["schema_version"] == SCHEMA_VERSION
    assert len(on_disk["header"]["map"]["nodes"]) == 12
    assert len(on_disk["steps"]) == doc["summary"]["steps"] > 0


def test_recorded_states_match_environment():
    env = MediFlightEnv(record_events=True)
    sink = MemorySink()
    rec = EpisodeRecorder(env, [sink])
    rec.reset(seed=5)
    rng = np.random.default_rng(0)
    done, total = False, 0.0
    while not done:
        _, r, te, tr, _ = rec.step(edf_policy(env, rng))
        total += r
        s = sink.steps[-1]
        assert s["state"] == env.snapshot()
        assert s["t_end"] >= s["t_start"]
        assert abs(s["reward"] - sum(s["reward_breakdown"].values())) < 1e-3
        done = te or tr
    assert abs(sink.summary["return"] - total) < 1e-3
    # Every delivery event in the log corresponds to a counted delivery.
    delivers = [e for st in sink.steps for e in st["events"] if e["type"] == "deliver"]
    assert len(delivers) == sink.summary["delivered_on_time"] + sink.summary["delivered_late"]
