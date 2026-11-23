"""Post-sweep analysis: validation re-scoring, model selection, test and generalisation.

    uv run python -m evaluation.compare validate      # score every run on 50 validation seeds
    uv run python -m evaluation.compare select        # best config per algorithm (by validation)
    uv run python -m evaluation.compare test          # held-out test + baselines + paired stats
    uv run python -m evaluation.compare generalise    # shifted scenarios
    uv run python -m evaluation.compare all

Selection never looks at test seeds. Outputs go to results/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.protocol import (SCENARIOS, TEST_SEEDS, VALIDATION_SEEDS, baseline_act,
                                 paired_difference, run_episodes, summarize)
from training.common import ROOT, greedy_act_fn, load_score_net

LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
ALGOS = ("dqn", "reinforce", "ppo", "a2c")
SCORE_KEY = {"dqn": "q_net", "reinforce": "actor", "ppo": "actor", "a2c": "actor"}
HEADLINE = ("return", "delivered_on_time", "service_rate", "critical_service_rate", "value_rate",
            "crashed", "expired", "spoiled", "invalid_actions", "avg_delivery_time_min",
            "energy_per_delivery")
BASELINE_NAMES = ("random", "nearest_first", "edf")


def _runs():
    """Main-study runs only. Follow-up runs (followup_*) are analysed by evaluation/followup.py."""
    for summary in sorted(LOGS.glob("*/*/seed_*/summary.json")):
        if summary.parent.parent.name.startswith("followup_"):
            continue
        s = json.loads(summary.read_text())
        s["config"] = json.loads((summary.parent / "config.json").read_text())
        s["run_dir"] = summary.parent
        yield s


def _agent_act(model_path: Path, algo: str):
    net = load_score_net(model_path, SCORE_KEY[algo])
    act = greedy_act_fn(net)
    return lambda seed: act


def _write_json(name: str, obj) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=float))


def _env_cfg(run_cfg: dict):
    from training.common import env_config
    return env_config(run_cfg.get("env", {}))


# -------------------------------------------------------------------- stages
def validate() -> list[dict]:
    """Re-score every run's selected checkpoint on all validation seeds."""
    rows = []
    for s in _runs():
        model = ROOT / s["model_path"]
        ep = run_episodes(_agent_act(model, s["algo"]), VALIDATION_SEEDS, _env_cfg(s["config"]))
        summ = summarize(ep)
        rows.append({"algo": s["algo"], "name": s["name"], "seed": s["seed"],
                     "description": s["config"].get("description", ""),
                     "best_frames": s["best_frames"], "episodes_trained": s["episodes"],
                     "wall_s": s["wall_s"], **{k: summ[f"{k}_mean"] for k in HEADLINE
                                               if f"{k}_mean" in summ}})
        print(f"validated {s['algo']}/{s['name']} seed={s['seed']}: return={summ['return_mean']:.2f}")
    _write_json("validation_runs.json", rows)
    return rows


def config_table(rows: list[dict]) -> dict[str, list[dict]]:
    """Aggregate runs to one row per (algo, config): mean and std across seeds."""
    tables: dict[str, list[dict]] = {}
    for algo in ALGOS:
        names = sorted({r["name"] for r in rows if r["algo"] == algo})
        out = []
        for n in names:
            rs = [r for r in rows if r["algo"] == algo and r["name"] == n]
            row = {"name": n, "description": rs[0]["description"], "seeds": len(rs)}
            for k in HEADLINE + ("best_frames", "episodes_trained"):
                v = np.asarray([r[k] for r in rs if r.get(k) is not None], dtype=float)
                v = v[~np.isnan(v)]
                if len(v):
                    row[f"{k}_mean"] = float(v.mean())
                    row[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            out.append(row)
        tables[algo] = out
    return tables


def select(rows: list[dict] | None = None) -> dict:
    rows = rows or json.loads((RESULTS / "validation_runs.json").read_text())
    tables = config_table(rows)
    _write_json("config_tables.json", tables)
    best = {}
    for algo, table in tables.items():
        # Main comparison uses masked agents; the no-mask ablation is reported separately.
        eligible = [r for r in table if "no_mask" not in r["name"]]
        b = max(eligible, key=lambda r: r["return_mean"])
        best[algo] = {"name": b["name"], "val_return_mean": b["return_mean"],
                      "val_return_std": b["return_std"], "seeds": b["seeds"]}
        print(f"best {algo}: {b['name']} val return {b['return_mean']:.2f} ± {b['return_std']:.2f}")
    _write_json("best_configs.json", best)
    return best


def _best_runs(algo: str, name: str) -> list[dict]:
    return sorted((s for s in _runs() if s["algo"] == algo and s["name"] == name),
                  key=lambda s: s["seed"])


def test() -> dict:
    """Held-out test: best config per algorithm (all its seeds) vs baselines, same seeds."""
    best = json.loads((RESULTS / "best_configs.json").read_text())
    per_episode: dict[str, list[dict]] = {}
    for b in BASELINE_NAMES:
        per_episode[b] = run_episodes(lambda seed, b=b: baseline_act(b, seed), TEST_SEEDS)
    out = {"test_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]], "policies": {}}
    for algo, info in best.items():
        runs = _best_runs(algo, info["name"])
        seed_rows = []
        for s in runs:
            ep = run_episodes(_agent_act(ROOT / s["model_path"], algo), TEST_SEEDS)
            per_episode[f"{algo}_seed{s['seed']}"] = ep
            seed_rows.append(summarize(ep))
        # Average each test episode across training seeds -> one paired row per test seed.
        merged = []
        for i, seed in enumerate(TEST_SEEDS):
            vals = [per_episode[f"{algo}_seed{s['seed']}"][i] for s in runs]
            merged.append({"seed": seed, **{k: float(np.nanmean([v[k] for v in vals]))
                                            for k in vals[0] if k != "seed"}})
        per_episode[algo] = merged
        across = np.asarray([r["return_mean"] for r in seed_rows])
        out["policies"][algo] = {"config": info["name"], "training_seeds": [s["seed"] for s in runs],
                                 "return_mean_across_seeds": float(across.mean()),
                                 "return_std_across_seeds": float(across.std(ddof=1)) if len(across) > 1 else 0.0,
                                 "per_seed_return": across.tolist(), **summarize(merged)}
    for b in BASELINE_NAMES:
        out["policies"][b] = summarize(per_episode[b])
    out["paired_vs_edf"] = {a: paired_difference(per_episode[a], per_episode["edf"]) for a in best}
    out["paired_vs_nearest"] = {a: paired_difference(per_episode[a], per_episode["nearest_first"]) for a in best}
    out["paired_critical_vs_edf"] = {a: paired_difference(per_episode[a], per_episode["edf"],
                                                          key="critical_service_rate") for a in best}
    _write_json("test_results.json", out)
    _write_json("test_episodes.json", per_episode)
    for name, p in out["policies"].items():
        print(f"TEST {name:14s} return {p['return_mean']:7.2f} ± {p['return_ci95']:.2f} | "
              f"on-time {p['delivered_on_time_mean']:.2f} | critical {p['critical_service_rate_mean']:.3f} | "
              f"crash {p['crashed_mean']:.3f}")
    for a, d in out["paired_vs_edf"].items():
        print(f"  {a} - edf: {d['mean_diff']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]")
    return out


def generalise(n_seeds: int = 100) -> dict:
    best = json.loads((RESULTS / "best_configs.json").read_text())
    seeds = TEST_SEEDS[:n_seeds]
    out: dict[str, dict] = {}
    for scen, cfg in SCENARIOS.items():
        out[scen] = {}
        for b in ("nearest_first", "edf"):
            out[scen][b] = summarize(run_episodes(lambda s, b=b: baseline_act(b, s), seeds, cfg))
        for algo, info in best.items():
            per_seed = [summarize(run_episodes(_agent_act(ROOT / s["model_path"], algo), seeds, cfg))
                        for s in _best_runs(algo, info["name"])]
            r = np.asarray([p["return_mean"] for p in per_seed])
            out[scen][algo] = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}
            out[scen][algo]["return_std_across_seeds"] = float(r.std(ddof=1)) if len(r) > 1 else 0.0
        print(scen, {k: round(v["return_mean"], 1) for k, v in out[scen].items()})
    _write_json("generalisation.json", out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["validate", "select", "test", "generalise", "all"])
    a = p.parse_args()
    if a.stage in ("validate", "all"):
        rows = validate()
    if a.stage in ("select", "all"):
        select()
    if a.stage in ("test", "all"):
        test()
    if a.stage in ("generalise", "all"):
        generalise()


if __name__ == "__main__":
    main()
