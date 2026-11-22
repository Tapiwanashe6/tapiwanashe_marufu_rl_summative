"""Non-learning reference policies.

They read the environment state directly (the same information the agent's
observation encodes) and are used to (1) sanity-check the environment dynamics
and (2) give the RL agents an honest bar to beat.

* random_policy        uniform over currently valid actions
* nearest_first_policy load nearest orders, fly nearest-first, charge when low
* edf_policy           urgency/earliest-deadline-first with battery-feasible
                       sortie planning (a strong hand-written dispatcher)
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import entities as E
from .custom_env import (A_CHARGE, A_FLY, A_HOLD, A_LOAD, A_RETURN, MediFlightEnv)

Policy = Callable[[MediFlightEnv, np.random.Generator], int]
SAFETY = 1.15     # plan with 15% margin over expected energy (gusts, weather change)
RESERVE = 5.0     # % battery kept in reserve


def _route_cost(env: MediFlightEnv, start: int, sites: list[int], weights: dict[int, float],
                weather: int) -> tuple[float, list[float]]:
    """Expected energy of visiting `sites` in order and returning home, and arrival offsets."""
    d = env.spec_drone
    node, payload = start, sum(weights[s] for s in sites)
    energy, t, arrivals = 0.0, 0.0, []
    for s in sites:
        e, m = E.leg_expectation(d, E.DISTANCE_KM[node, s + 1], payload, weather)
        energy += e
        t += m
        arrivals.append(t)
        t += d.drop_min
        payload -= weights[s]
        node = s + 1
    e, _ = E.leg_expectation(d, E.DISTANCE_KM[node, 0], payload, weather)
    return energy + e, arrivals


def random_policy(env: MediFlightEnv, rng: np.random.Generator) -> int:
    return int(rng.choice(np.flatnonzero(env.valid_actions())))


def nearest_first_policy(env: MediFlightEnv, rng: np.random.Generator) -> int:
    d, valid = env.spec_drone, env.valid_actions()
    loc = env.location
    if loc == 0:
        pending = sorted((E.DISTANCE_KM[0, s + 1], s) for s in range(E.N_SITES)
                         if valid[A_LOAD + s])
        # Load the nearest pending order while the sortie stays battery-feasible.
        for _, s in pending:
            sites = sorted(env.payload + [s], key=lambda x: E.DISTANCE_KM[0, x + 1])
            w = {x: env.orders[x].weight for x in sites}
            need, _ = _route_cost(env, 0, sites, w, env.weather)
            if need * SAFETY + RESERVE <= env.battery:
                return A_LOAD + s
        if env.payload:
            return A_FLY + min(env.payload, key=lambda x: E.DISTANCE_KM[0, x + 1])
        if valid[A_CHARGE] and env.battery < 0.9 * d.battery_capacity:
            return A_CHARGE
        return A_HOLD
    if env.payload:
        s = min(env.payload, key=lambda x: E.DISTANCE_KM[loc, x + 1])
        w = {x: env.orders[x].weight for x in env.payload}
        need, _ = _route_cost(env, loc, [s], w | {}, env.weather)
        rest = sum(w.values()) - w[s]
        e_leg, _ = E.leg_expectation(d, E.DISTANCE_KM[loc, s + 1], sum(w.values()), env.weather)
        e_home, _ = E.leg_expectation(d, E.DISTANCE_KM[s + 1, 0], rest, env.weather)
        if (e_leg + e_home) * SAFETY <= env.battery:
            return A_FLY + s
    return A_RETURN


def edf_policy(env: MediFlightEnv, rng: np.random.Generator) -> int:
    """Urgency-aware earliest-deadline-first dispatcher with feasibility checks."""
    d, valid = env.spec_drone, env.valid_actions()
    loc, now = env.location, env.time

    def order_key(s: int):
        o = env.orders[s]
        return (o.deadline, -o.ptype.urgency)

    if loc == 0:
        candidates = [s for s in range(E.N_SITES) if valid[A_LOAD + s]]
        # Prefer orders that can still arrive on time, then by urgency and deadline.
        def feasible_alone(s: int) -> bool:
            _, m = E.leg_expectation(d, E.DISTANCE_KM[0, s + 1], env.orders[s].weight, env.weather)
            return now + d.load_min + m <= env.orders[s].deadline
        candidates.sort(key=lambda s: (not feasible_alone(s), -env.orders[s].ptype.urgency,
                                       env.orders[s].deadline))
        for s in candidates:
            if not feasible_alone(s):
                break
            sites = sorted(env.payload + [s], key=order_key)
            w = {x: env.orders[x].weight for x in sites}
            need, arrivals = _route_cost(env, 0, sites, w, env.weather)
            t0 = now + d.load_min * (1 + len(env.payload))
            on_time = all(t0 + a <= env.orders[x].deadline for x, a in zip(sites, arrivals))
            if need * SAFETY + RESERVE <= env.battery and on_time:
                return A_LOAD + s
        if env.payload:
            sites = sorted(env.payload, key=order_key)
            w = {x: env.orders[x].weight for x in sites}
            need, _ = _route_cost(env, 0, sites, w, env.weather)
            if need * SAFETY + RESERVE <= env.battery:
                return A_FLY + sites[0]
            # Not enough energy for the planned sortie: top up first.
            if valid[A_CHARGE]:
                return A_CHARGE
            return A_FLY + sites[0]
        if valid[A_CHARGE] and (env.battery < 0.95 * d.battery_capacity):
            return A_CHARGE
        return A_HOLD
    if env.payload:
        sites = sorted(env.payload, key=order_key)
        w = {x: env.orders[x].weight for x in sites}
        need, _ = _route_cost(env, loc, sites[:1], w | {}, env.weather)
        total = sum(w.values())
        e_leg, _ = E.leg_expectation(d, E.DISTANCE_KM[loc, sites[0] + 1], total, env.weather)
        e_home, _ = E.leg_expectation(d, E.DISTANCE_KM[sites[0] + 1, 0], total - w[sites[0]], env.weather)
        if (e_leg + e_home) * SAFETY <= env.battery:
            return A_FLY + sites[0]
    return A_RETURN


BASELINES: dict[str, Policy] = {
    "random": random_policy,
    "nearest_first": nearest_first_policy,
    "edf": edf_policy,
}


def run_episode(env: MediFlightEnv, policy: Policy, seed: int) -> dict:
    """Roll out one episode; returns the final info dict plus the episode return."""
    rng = np.random.default_rng(seed + 10_000)
    env.reset(seed=seed)
    total, done, info = 0.0, False, {}
    while not done:
        _, r, term, trunc, info = env.step(policy(env, rng))
        total += r
        done = term or trunc
    return {"return": total, "steps": env.steps, **info}
