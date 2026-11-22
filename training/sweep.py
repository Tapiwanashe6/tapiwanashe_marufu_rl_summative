"""Run the hyperparameter sweep: every config in configs/<algo>/ x every seed.

    uv run python -m training.sweep --seeds 0 1 2 --workers 6
    uv run python -m training.sweep --algos dqn --seeds 0 --dry-run

Each (config, seed) runs in its own process (single-threaded torch), writes to
logs/<algo>/<exp>/seed_<k>/ and is skipped on re-run if summary.json exists,
so an interrupted sweep resumes where it stopped.  A manifest with the git
commit, budget and per-job status is written to logs/sweep_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALGOS = ("dqn", "ppo", "a2c", "reinforce")
# Rough relative cost, used only to start the slowest jobs first (load balancing).
COST = {"dqn": 3.0, "reinforce": 1.5, "a2c": 1.2, "ppo": 1.0}


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def jobs_for(algos, seeds):
    jobs = []
    for algo in algos:
        for cfg in sorted((ROOT / "configs" / algo).glob("*.toml")):
            for seed in seeds:
                jobs.append({"algo": algo, "config": str(cfg.relative_to(ROOT)), "name": cfg.stem,
                             "seed": seed})
    jobs.sort(key=lambda j: -COST[j["algo"]])
    return jobs


def command(job, total_frames: int | None) -> list[str]:
    if job["algo"] == "dqn":
        cmd = [sys.executable, "-m", "training.dqn_training"]
    else:
        cmd = [sys.executable, "-m", "training.pg_training", "--algo", job["algo"]]
    cmd += ["--config", job["config"], "--seed", str(job["seed"])]
    if total_frames:
        cmd += ["--total-frames", str(total_frames)]
    return cmd


def run_job(job, total_frames):
    run_dir = ROOT / "logs" / job["algo"] / job["name"] / f"seed_{job['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(run_dir / "stdout.log", "w") as out:
        rc = subprocess.run(command(job, total_frames), cwd=ROOT, stdout=out,
                            stderr=subprocess.STDOUT, env={**__import__("os").environ,
                                                           "RL_WARNINGS": "False"}).returncode
    return {**job, "returncode": rc, "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algos", nargs="+", choices=ALGOS, default=list(ALGOS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--total-frames", type=int, help="override budget (default: config/common default)")
    p.add_argument("--force", action="store_true", help="re-run jobs that already finished")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    jobs = jobs_for(a.algos, a.seeds)
    todo = [j for j in jobs if a.force or not
            (ROOT / "logs" / j["algo"] / j["name"] / f"seed_{j['seed']}" / "summary.json").exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run "
          f"with {a.workers} workers")
    if a.dry_run:
        for j in todo:
            print(" ", " ".join(command(j, a.total_frames)))
        return
    manifest_path = ROOT / "logs" / "sweep_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
    manifest.update(git_commit=_git_commit(), started=datetime.now(timezone.utc).isoformat(),
                    seeds=a.seeds, total_frames_override=a.total_frames, python=sys.version)
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run_job, j, a.total_frames) for j in todo]
        for f in as_completed(futures):
            r = f.result()
            done += 1
            manifest["runs"] = [x for x in manifest["runs"]
                                if (x["algo"], x["name"], x["seed"]) != (r["algo"], r["name"], r["seed"])]
            manifest["runs"].append(r)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            status = "ok" if r["returncode"] == 0 else f"FAILED rc={r['returncode']}"
            print(f"[{done}/{len(todo)}] {r['algo']}/{r['name']} seed={r['seed']} {status} "
                  f"({r['wall_s']}s)", flush=True)
    failed = [r for r in manifest["runs"] if r["returncode"] != 0]
    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Sweep finished: {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
