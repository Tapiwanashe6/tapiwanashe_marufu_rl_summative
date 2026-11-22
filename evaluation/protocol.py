"""Evaluation protocol shared by every policy (baselines and trained agents).

* Seed sets are disjoint: training environments are seeded from the run seed
  (0..99) and continue their own RNG chain; model selection uses VALIDATION
  seeds; reported numbers use held-out TEST seeds.
* Common random numbers: on a given seed every policy faces identical order
  requests and weather (see MediFlightEnv._generate_world), so policies can be
  compared with paired statistics.
* Rates use *exogenous* denominators (all requests, including those rejected
  because a site was busy), which are identical across policies.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from environment import EnvConfig, MediFlightEnv
from environment.baselines import BASELINES

VALIDATION_SEEDS: tuple[int, ...] = tuple(range(10_000, 10_050))
TEST_SEEDS: tuple[int, ...] = tuple(range(20_000, 20_200))

# Shifted conditions for the generalisation analysis (never used in training).
SCENARIOS: dict[str, EnvConfig] = {
    "nominal": EnvConfig(),
    "high_demand": EnvConfig(arrival_scale=1.5),
    "low_demand": EnvConfig(arrival_scale=0.6),
    "stormy": EnvConfig(storm_bias=0.15),
    "busy_start": EnvConfig(initial_orders=(8, 10)),
}

# act(env, obs) -> action.  Trained agents use obs; baselines read env state.
ActFn = Callable[[MediFlightEnv, dict], int]


def baseline_act(name: str, seed: int) -> ActFn:
    rng = np.random.default_rng(seed + 10_000)
    policy = BASELINES[name]
    return lambda env, obs: policy(env, rng)


def episode_metrics(info: dict, ret: float, steps: int) -> dict:
    delivered = info["delivered_on_time"] + info["delivered_late"]
    return {
        "return": ret, "steps": steps,
        "requests": info["requests_total"], "delivered_on_time": info["delivered_on_time"],
        "delivered_late": info["delivered_late"], "expired": info["expired"],
        "spoiled": info["spoiled"], "crashed": float(info["crashes"] > 0),
        "invalid_actions": info["invalid_actions"],
        "critical_on_time": info["critical_on_time"], "critical_requests": info["critical_requests"],
        "service_rate": info["delivered_on_time"] / max(1.0, info["requests_total"]),
        "critical_service_rate": info["critical_on_time"] / max(1.0, info["critical_requests"]),
        "value_rate": info["value_delivered"] / max(1e-9, info["value_requested"]),
        "energy_used": info["energy_used"], "distance_km": info["distance_km"],
        "energy_per_delivery": info["energy_used"] / max(1.0, delivered),
        "avg_delivery_time_min": (info["delivery_lead_time_total"] / delivered) if delivered else np.nan,
        "charge_minutes": info["charge_minutes"],
    }


def run_episodes(make_act: Callable[[int], ActFn], seeds, env_cfg: EnvConfig | None = None) -> list[dict]:
    """Roll out one episode per seed. `make_act(seed)` builds a fresh policy."""
    env = MediFlightEnv(env_cfg or EnvConfig())
    rows = []
    for seed in seeds:
        act = make_act(seed)
        obs, info = env.reset(seed=int(seed))
        ret, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(act(env, obs))
            ret += r
            done = term or trunc
        rows.append({"seed": int(seed), **episode_metrics(info, ret, env.steps)})
    return rows


def summarize(rows: list[dict]) -> dict:
    """Mean, std and normal-approximation 95% CI half-width for each metric."""
    out: dict[str, float] = {"episodes": len(rows)}
    for k in rows[0]:
        if k == "seed":
            continue
        v = np.asarray([r[k] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out[f"{k}_ci95"] = float(1.96 * out[f"{k}_std"] / np.sqrt(len(v)))
    return out


def paired_difference(a: list[dict], b: list[dict], key: str = "return",
                      n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired bootstrap of mean(a - b) over common seeds (CRN makes pairing valid)."""
    da = {r["seed"]: r[key] for r in a}
    db = {r["seed"]: r[key] for r in b}
    common = sorted(set(da) & set(db))
    diff = np.asarray([da[s] - db[s] for s in common])
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_a_better": float((diff > 0).mean()), "n": len(common)}
