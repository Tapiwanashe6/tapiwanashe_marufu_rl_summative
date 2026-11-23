"""End-to-end check of the analysis pipeline on tiny runs in a temporary directory."""
import json

import pytest

import evaluation.compare as compare
import evaluation.plots as plots
import evaluation.report_tables as report_tables
import training.common as common
from training import dqn_training, pg_training


@pytest.fixture()
def mini_sweep(tmp_path, monkeypatch):
    logs, models, results = tmp_path / "logs", tmp_path / "models", tmp_path / "results"
    for mod, attr, val in [(common, "LOG_ROOT", logs), (common, "MODEL_ROOT", models),
                           (common, "ROOT", tmp_path), (compare, "LOGS", logs),
                           (compare, "RESULTS", results), (compare, "ROOT", tmp_path),
                           (plots, "LOGS", logs), (plots, "RESULTS", results),
                           (plots, "OUT", tmp_path / "figs"), (report_tables, "RESULTS", results),
                           (report_tables, "ROOT", tmp_path)]:
        monkeypatch.setattr(mod, attr, val)
    small = tuple(range(10_000, 10_003))
    monkeypatch.setattr(compare, "VALIDATION_SEEDS", small)
    monkeypatch.setattr(compare, "TEST_SEEDS", tuple(range(20_000, 20_003)))
    monkeypatch.setattr(compare, "SCENARIOS", {"nominal": compare.SCENARIOS["nominal"],
                                               "stormy": compare.SCENARIOS["stormy"]})
    cfg_dir = tmp_path / "configs"
    tiny = {"dqn": {"total_frames": 1200, "init_random_frames": 400, "batch_size": 32},
            "ppo": {"total_frames": 1024, "frames_per_batch": 512, "minibatch_size": 128},
            "a2c": {"total_frames": 1024}, "reinforce": {"total_frames": 800, "episodes_per_batch": 2}}
    for algo in compare.ALGOS:
        for name in ("exp01_baseline", "exp02_variant"):
            d = cfg_dir / algo
            d.mkdir(parents=True, exist_ok=True)
            hp = "lr = 0.001\n" if name == "exp02_variant" else ""
            (d / f"{name}.toml").write_text(
                f'name = "{name}"\ndescription = "test {name}"\n[hyperparameters]\n'
                f"eval_interval_frames = 400\neval_episodes = 2\n{hp}")
            for seed in (0, 1):
                over = {**tiny[algo], "eval_interval_frames": 400, "eval_episodes": 2}
                if algo == "dqn":
                    dqn_training.train(str(d / f"{name}.toml"), seed, over)
                else:
                    pg_training.train(str(d / f"{name}.toml"), algo, seed, over)
    return tmp_path


def test_pipeline_end_to_end(mini_sweep):
    root = mini_sweep
    rows = compare.validate()
    assert len(rows) == 4 * 2 * 2
    best = compare.select(rows)
    assert set(best) == set(compare.ALGOS)
    t = compare.test()
    assert {"edf", "nearest_first", "random", *compare.ALGOS} <= set(t["policies"])
    assert set(t["paired_vs_edf"]) == set(compare.ALGOS)
    compare.generalise(n_seeds=2)
    (root / "results" / "baselines_validation.json").write_text(json.dumps({"edf": 1.0, "nearest_first": 0.0}))
    best = plots.best_configs()
    base = plots.baseline_val()
    plots.fig_cumulative_rewards(best, base)
    plots.fig_learning_comparison(best, base)
    plots.fig_dqn_objective(best)
    plots.fig_pg_entropy(best)
    plots.fig_convergence(best)
    plots.fig_generalisation()
    plots.fig_test_comparison()
    plots.fig_sweep_sensitivity(base)
    assert len(list((root / "figs").glob("*.png"))) == 8
    md = report_tables.build()
    assert "### 4.4 A2C" in md
    assert md.count("| **") == 2 * 4   # selected row bolds its # and Mean Reward cells, per algorithm
