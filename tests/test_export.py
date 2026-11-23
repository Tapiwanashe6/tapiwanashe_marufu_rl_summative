"""The replay file must be a faithful record of the real environment."""
import json

import numpy as np

from environment import MediFlightEnv
from environment.baselines import edf_policy
from environment.rendering import SCHEMA_VERSION, EpisodeRecorder, MemorySink
from visualization.export_events import baseline_act, export_episode


def test_export_file_roundtrip(tmp_path):
    doc = export_episode(baseline_act("edf", 42), 42, "edf", "EDF", "baseline", out_dir=tmp_path)
    on_disk = json.loads((tmp_path / "edf_seed42.json").read_text())
    assert on_disk["schema_version"] == SCHEMA_VERSION
    assert len(on_disk["header"]["map"]["nodes"]) == 12
    assert len(on_disk["steps"]) == doc["summary"]["steps"] > 0
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["episodes"][0]["file"] == "edf_seed42.json"
    # Every flight in the log has a matching arrival or crash (the viewer relies on this).
    for st in on_disk["steps"]:
        types = [e["type"] for e in st["events"]]
        if "takeoff" in types:
            assert ("arrive" in types) != ("crash" in types)


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
