"""Follow-up experiment: does a larger training budget improve the selected agents?

This runs AFTER the main study, whose results (results/test_results.json) are
final and unchanged. Because the held-out test seeds had already been used once
in the main study, the follow-up is kept strictly separate:

  * candidates are compared with the original selected configs on the
    VALIDATION seeds only (stage `select`);
  * only the validation winner per algorithm is evaluated on the test seeds,
    once (stage `test`), and reported as an extension, not a replacement.

    uv run python -m evaluation.followup train --workers 7   # 3 configs x 5 seeds (resumable)
    uv run python -m evaluation.followup validate
    uv run python -m evaluation.followup select
    uv run python -m evaluation.followup test

Configs: configs/followup/*.toml. Runs: logs/<algo>/followup_*/seed_<k>/.
Outputs: results/followup_*.json.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

from evaluation.compare import RESULTS, ROOT, _agent_act, _env_cfg, _write_json
from evaluation.protocol import TEST_SEEDS, VALIDATION_SEEDS, paired_difference, run_episodes, summarize
from training.sweep import _git_commit, run_job

CONFIG_DIR = ROOT / "configs" / "followup"
SEEDS = (0, 1, 2, 3, 4)
BASE = {"dqn": "exp10_vanilla_dqn", "ppo": "exp06_epochs_10"}     # main-study selections


def _algo_of(name: str) -> str:
    return name.removeprefix("followup_").split("_")[0]


def jobs() -> list[dict]:
    out = []
    for cfg in sorted(CONFIG_DIR.glob("*.toml")):
        for seed in SEEDS:
            out.append({"algo": _algo_of(cfg.stem), "config": str(cfg.relative_to(ROOT)), "name": cfg.stem, "seed": seed})
    out.sort(key=lambda j: j["algo"] != "dqn")          # slowest first
    return out


def train(workers: int) -> None:
    todo = [j for j in jobs() if not (ROOT / "logs" / j["algo"] / j["name"] / f"seed_{j['seed']}" / "summary.json").exists()]
    manifest_path = ROOT / "logs" / "followup_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
    manifest.update(git_commit=_git_commit(), started=datetime.now(timezone.utc).isoformat())
    print(f"{len(jobs())} follow-up jobs, {len(todo)} to run with {workers} workers", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(run_job, j, None) for j in todo]
        for k, f in enumerate(as_completed(futures), 1):
            r = f.result()
            manifest["runs"] = [x for x in manifest["runs"] if not (x["name"] == r["name"] and x["seed"] == r["seed"])] + [r]
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"[{k}/{len(todo)}] {r['algo']}/{r['name']} seed={r['seed']} "
                  f"{'ok' if r['returncode'] == 0 else 'FAILED rc=' + str(r['returncode'])} ({r['wall_s']}s)", flush=True)
    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _runs(name: str, algo: str) -> list[dict]:
    out = []
    for seed in SEEDS:
        d = ROOT / "logs" / algo / name / f"seed_{seed}"
        if (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text())
            s["config"] = json.loads((d / "config.json").read_text())
            out.append(s)
    return out


def validate() -> list[dict]:
    rows = []
    for cfg in sorted(CONFIG_DIR.glob("*.toml")):
        name, algo = cfg.stem, _algo_of(cfg.stem)
        for s in _runs(name, algo):
            summ = summarize(run_episodes(_agent_act(ROOT / s["model_path"], algo), VALIDATION_SEEDS, _env_cfg(s["config"])))
            rows.append({"algo": algo, "name": name, "seed": s["seed"], "return": summ["return_mean"],
                         "critical_service_rate": summ["critical_service_rate_mean"],
                         "best_frames": s["best_frames"], "wall_s": s["wall_s"]})
            print(f"validated {algo}/{name} seed={s['seed']}: return={summ['return_mean']:.2f} (best at {s['best_frames']} frames)")
    _write_json("followup_validation.json", rows)
    return rows


def select() -> dict:
    """Per algorithm: original selection vs follow-up variants, all on validation seeds, 5 training seeds each."""
    follow = json.loads((RESULTS / "followup_validation.json").read_text())
    main = json.loads((RESULTS / "validation_runs.json").read_text())
    out = {}
    for algo, base in BASE.items():
        cands = {base: [r["return"] for r in main if r["algo"] == algo and r["name"] == base]}
        for r in follow:
            if r["algo"] == algo:
                cands.setdefault(r["name"], []).append(r["return"])
        table = {n: {"val_return_mean": float(np.mean(v)), "val_return_std": float(np.std(v, ddof=1)), "seeds": len(v)}
                 for n, v in cands.items()}
        winner = max(table, key=lambda n: table[n]["val_return_mean"])
        out[algo] = {"candidates": table, "winner": winner, "winner_is_original": winner == base}
        print(f"{algo}: " + " | ".join(f"{n} {t['val_return_mean']:.2f}±{t['val_return_std']:.2f}" for n, t in table.items())
              + f"  -> {winner}")
    _write_json("followup_selection.json", out)
    return out


def test() -> dict:
    """Second (and final) use of the test seeds: only for follow-up winners."""
    sel = json.loads((RESULTS / "followup_selection.json").read_text())
    episodes = json.loads((RESULTS / "test_episodes.json").read_text())
    out = {"note": "Follow-up after the main study; the main results are unchanged. "
                   "Winners were chosen on validation seeds only.", "algos": {}}
    for algo, info in sel.items():
        if info["winner_is_original"]:
            out["algos"][algo] = {"winner": info["winner"], "tested": False,
                                  "reason": "original configuration still best on validation"}
            print(f"{algo}: original config still best on validation - no new test run")
            continue
        runs = _runs(info["winner"], algo)
        per_seed = [run_episodes(_agent_act(ROOT / s["model_path"], algo), TEST_SEEDS, _env_cfg(s["config"])) for s in runs]
        merged = [{"seed": seed, **{k: float(np.nanmean([ps[i][k] for ps in per_seed])) for k in per_seed[0][i] if k != "seed"}}
                  for i, seed in enumerate(TEST_SEEDS)]
        across = [summarize(ps)["return_mean"] for ps in per_seed]
        res = {"winner": info["winner"], "tested": True, "training_seeds": [s["seed"] for s in runs],
               "per_seed_return": across, **summarize(merged),
               "paired_vs_edf": paired_difference(merged, episodes["edf"]),
               "paired_critical_vs_edf": paired_difference(merged, episodes["edf"], key="critical_service_rate"),
               "paired_vs_original_500k": paired_difference(merged, episodes[algo])}
        out["algos"][algo] = res
        d, o = res["paired_vs_edf"], res["paired_vs_original_500k"]
        print(f"TEST {algo}/{info['winner']}: return {res['return_mean']:.2f} ± {res['return_ci95']:.2f} | "
              f"vs EDF {d['mean_diff']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}] | "
              f"vs 500k {o['mean_diff']:+.2f} [{o['ci_low']:+.2f}, {o['ci_high']:+.2f}]")
    _write_json("followup_test.json", out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["train", "validate", "select", "test"])
    p.add_argument("--workers", type=int, default=7)
    a = p.parse_args()
    {"train": lambda: train(a.workers), "validate": validate, "select": select, "test": test}[a.stage]()


if __name__ == "__main__":
    main()
