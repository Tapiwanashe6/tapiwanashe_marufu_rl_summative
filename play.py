"""Run a trained MediFlight AI agent: verbose terminal log + 3D mission-control replay.

    uv run play.py                          # best agent (chosen on validation data), seed 42
    uv run play.py --algo ppo --seed 7      # best PPO checkpoint on another scenario
    uv run play.py --model models/dqn/dqn_exp10_vanilla_dqn_seed1.pt
    uv run play.py --no-viewer              # terminal only

How the agent is chosen (no test data is used):
  1. results/best_configs.json  - best hyperparameter config per algorithm, ranked by
     mean return on the 50 validation seeds (evaluation/compare.py select).
  2. If the follow-up experiment ran (results/followup_selection.json) and a longer-trained
     variant beat that config on the same validation seeds, the variant is used instead
     (evaluation/followup.py select). Pass --main-study to ignore the follow-up.
  3. Without --algo, the algorithm whose chosen config has the highest validation return.
  4. Among that config's training seeds, the checkpoint with the highest validation return
     (results/validation_runs.json or results/followup_validation.json).

The episode is run in the real Gymnasium environment with the greedy (masked argmax)
policy. Every decision is printed to the terminal and recorded through the same
EpisodeRecorder used for the baselines; the viewer then replays that recording, so
what you see in 3D is exactly what the environment computed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from environment import MediFlightEnv, action_name
from training.common import load_score_net
from visualization.export_events import export_episode

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SCORE_KEY = {"dqn": "q_net", "reinforce": "actor", "ppo": "actor", "a2c": "actor"}
NOTABLE = {"order_created": "new order", "deliver": "DELIVERED", "order_late": "now late",
           "order_expired": "EXPIRED", "spoiled": "SPOILED", "crash": "CRASH", "weather": "weather change",
           "charge": "charged"}
ALGO_LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}


# ---------------------------------------------------------------- agent choice
def choose_checkpoint(algo: str | None, use_followup: bool = True) -> tuple[str, str, Path, float]:
    """(algo, config name, checkpoint path, validation return) using validation data only."""
    best_path, val_path = RESULTS / "best_configs.json", RESULTS / "validation_runs.json"
    if not best_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "results/best_configs.json or results/validation_runs.json is missing. Run\n"
            "  uv run python -m evaluation.compare validate && uv run python -m evaluation.compare select\n"
            "or pass --model PATH.")
    # candidate per algorithm: (config name, validation mean, per-seed validation rows)
    main_runs = json.loads(val_path.read_text())
    cands = {a: (b["name"], b["val_return_mean"], [r for r in main_runs if r["algo"] == a and r["name"] == b["name"]])
             for a, b in json.loads(best_path.read_text()).items()}
    sel_path, fval_path = RESULTS / "followup_selection.json", RESULTS / "followup_validation.json"
    if use_followup and sel_path.exists() and fval_path.exists():
        fruns = json.loads(fval_path.read_text())
        for a, info in json.loads(sel_path.read_text()).items():
            if not info["winner_is_original"]:
                w = info["winner"]
                cands[a] = (w, info["candidates"][w]["val_return_mean"],
                            [r for r in fruns if r["algo"] == a and r["name"] == w])
    if algo is None:
        algo = max(cands, key=lambda a: cands[a][1])
    name, _, runs = cands[algo]
    run = max(runs, key=lambda r: r["return"])
    folder = "dqn" if algo == "dqn" else "pg"
    return algo, name, ROOT / "models" / folder / f"{algo}_{name}_seed{run['seed']}.pt", run["return"]


def algo_from_checkpoint(path: Path) -> str:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    algo = ckpt.get("algo") or path.name.split("_")[0]
    if algo not in SCORE_KEY:
        raise ValueError(f"cannot tell which algorithm produced {path}")
    return algo


def agent_act(net: torch.nn.Module, algo: str):
    """Greedy masked policy that also reports how confident the decision was."""
    @torch.no_grad()
    def act(env: MediFlightEnv, obs: dict) -> tuple[int, dict]:
        scores = net(torch.as_tensor(obs["observation"]).unsqueeze(0)).squeeze(0)
        mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
        masked = torch.where(mask, scores, torch.finfo(scores.dtype).min)
        a = int(masked.argmax())
        top = masked.topk(min(3, int(mask.sum()))).indices.tolist()
        if algo == "dqn":   # Q-values: report the chosen value and the margin to the runner-up
            info = {"kind": "q_values", "value": float(scores[a]),
                    "margin": float(scores[a] - scores[top[1]]) if len(top) > 1 else None}
        else:               # policy logits -> probabilities over valid actions
            probs = torch.softmax(masked, dim=0)
            info = {"kind": "probability", "value": float(probs[a])}
        info["top"] = [{"action": action_name(i), "score": round(float(scores[i]), 3)} for i in top]
        return a, info
    return act


# ---------------------------------------------------------------- terminal log
def print_step(step: dict) -> None:
    p = step["policy"]
    conf = (f"p={p['value']:.2f}" if p.get("kind") == "probability"
            else f"Q={p['value']:+.1f}" if p.get("kind") == "q_values" else "")
    s = step["state"]
    notes = [e["type"] for e in step["events"]
             if e["type"] in NOTABLE]
    print(f"step {step['step']:03d} | t={step['t_end']:6.1f} min | {step['action_name']:<24} {conf:>8} "
          f"| r={step['reward']:+7.2f} | total={step['cumulative_reward']:+8.2f} "
          f"| battery={s['battery']:5.1f}% | {s['weather']:<5} "
          f"| {', '.join(dict.fromkeys(NOTABLE[n] for n in notes))}")


# ---------------------------------------------------------------- viewer
class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:   # keep the terminal for the agent's decisions
        pass


def serve_viewer(replay_file: str, port: int, open_browser: bool) -> None:
    handler = partial(_QuietHandler, directory=str(ROOT / "visualization"))
    server = None
    for p in range(port, port + 20):           # fall back to the next free port
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), handler)
            port = p
            break
        except OSError:
            continue
    if server is None:
        print(f"No free port in {port}-{port + 19}; pass --port to choose one.")
        return
    url = f"http://127.0.0.1:{port}/index.html?file={replay_file}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"\n3D mission control: {url}\nPress Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=sorted(SCORE_KEY), help="algorithm (default: best on validation)")
    p.add_argument("--model", type=Path, help="explicit checkpoint path (overrides --algo)")
    p.add_argument("--seed", type=int, default=42, help="scenario seed (default 42)")
    p.add_argument("--delay", type=float, default=0.0, help="seconds to pause between printed steps")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--no-viewer", action="store_true", help="terminal output only")
    p.add_argument("--no-browser", action="store_true", help="serve the viewer but do not open a browser")
    p.add_argument("--main-study", action="store_true",
                   help="ignore the follow-up experiment; use the main-study (500k-frame) selection")
    a = p.parse_args(argv)

    if a.model:
        algo, path, val = algo_from_checkpoint(a.model), a.model, None
        name = re.sub(rf"^{algo}_|_seed\d+$", "", a.model.stem)
    else:
        algo, name, path, val = choose_checkpoint(a.algo, use_followup=not a.main_study)
    if not path.exists():
        sys.exit(f"checkpoint not found: {path}")
    net = load_score_net(path, SCORE_KEY[algo])
    print(f"MediFlight AI | trained {ALGO_LABEL[algo]} agent | config {name} | checkpoint {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    if val is not None:
        print(f"Selected on validation data (seeds 10000-10049): mean return {val:.2f}. Scenario seed {a.seed}.")
    print("Policy: greedy argmax over valid actions (action mask applied).\n")

    def on_step(step: dict) -> None:
        print_step(step)
        if a.delay:
            time.sleep(a.delay)

    policy_id = f"{algo}_{name}"
    doc = export_episode(agent_act(net, algo), a.seed, policy_id,
                         f"{ALGO_LABEL[algo]} agent ({name})", "agent", on_step=on_step)
    s = doc["summary"]
    print(f"\nEpisode return {s['return']:.2f} | on time {s['delivered_on_time']:.0f} of {s['requests_total']:.0f} requests "
          f"| late {s['delivered_late']:.0f} | expired {s['expired']:.0f} | spoiled {s['spoiled']:.0f} "
          f"| critical on time {s['critical_on_time']:.0f}/{s['critical_requests']:.0f} | crashed {s['crashed']}")
    replay = f"{policy_id}_seed{a.seed}.json"
    print(f"Replay saved to visualization/episodes/{replay}")
    if not a.no_viewer:
        serve_viewer(replay, a.port, not a.no_browser)


if __name__ == "__main__":
    main()
