"""Behavioural tests for MediFlightEnv v2.

Mechanics are tested on hand-constructed states (no arrivals, no gust noise) so
expected values can be computed exactly; invariants are tested over many
randomised episodes.
"""
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment import (A_CHARGE, A_FLY, A_HOLD, A_LOAD, A_RETURN, N_ACTIONS, EnvConfig,
                         MediFlightEnv)
from environment import entities as E
from environment.baselines import BASELINES, random_policy, run_episode
from environment.custom_env import Order
from environment.observation_spec import OBS_DIM, feature_names, markdown_table


# ----------------------------------------------------------------- helpers
def quiet_env(**overrides) -> MediFlightEnv:
    """Deterministic env: no initial orders, no arrivals, no gusts, calm weather forever."""
    drone = replace(E.DroneSpec(), energy_noise_sd=0.0)
    cfg = EnvConfig(initial_orders=(0, 0), arrival_scale=0.0, initial_weather=0, drone=drone,
                    **overrides)
    env = MediFlightEnv(cfg)
    env.reset(seed=0)
    env._weather_path[:] = 0    # calm for the whole episode
    return env


def put_order(env, site, ptype=E.MEDICINE, weight=1.0, deadline=200.0):
    env._create_order(site, env.time, (ptype, weight, deadline - env.time))
    return env.orders[site]


def accounting_balances(info) -> bool:
    out = (info["delivered_on_time"] + info["delivered_late"] + info["expired"] + info["spoiled"]
           + info["lost_in_crash"] + info["undelivered_onboard_at_end"] + info["pending_at_end"])
    return out == info["orders_created"]


# ------------------------------------------------------------- API contract
@pytest.mark.parametrize("mask", [True, False])
def test_gymnasium_contract(mask):
    check_env(MediFlightEnv(EnvConfig(use_action_mask=mask)), skip_render_check=True)


def test_observation_spec_matches_vector():
    env = MediFlightEnv()
    obs, _ = env.reset(seed=1)
    assert obs["observation"].shape == (OBS_DIM,)
    assert len(feature_names()) == OBS_DIM
    assert "Source (Sensor/Camera/API/Dataset)" in markdown_table()


def test_seed_reproducibility_full_episode():
    def trajectory(seed):
        env, rng = MediFlightEnv(), np.random.default_rng(7)
        obs, _ = env.reset(seed=seed)
        out = [obs["observation"]]
        done = False
        while not done:
            obs, r, te, tr, info = env.step(random_policy(env, rng))
            out.append(np.append(obs["observation"], [r, info["time_min"]]))
            done = te or tr
        return np.concatenate(out)
    np.testing.assert_array_equal(trajectory(3), trajectory(3))
    assert not np.array_equal(trajectory(3), trajectory(4))


# --------------------------------------------------------------- invariants
@pytest.mark.parametrize("policy", sorted(BASELINES))
def test_invariants_over_many_episodes(policy):
    env = MediFlightEnv()
    space = env.observation_space
    for seed in range(40):
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        done, last_t = False, 0.0
        while not done:
            assert space.contains(obs)
            np.testing.assert_array_equal(obs["action_mask"], env.valid_actions().astype(np.int8))
            obs, r, te, tr, info = env.step(BASELINES[policy](env, rng))
            assert 0.0 <= env.battery <= env.spec_drone.battery_capacity + 1e-9
            assert env.time > last_t, "every action must consume time"
            assert len(env.payload) <= env.spec_drone.max_packages
            assert env.payload_weight <= env.spec_drone.payload_capacity_kg + 1e-9
            assert r == pytest.approx(sum(info["reward_breakdown"].values()))
            last_t = env.time
            done = te or tr
        assert te and not tr, "episodes end by termination (horizon or crash), not the safety cap"
        assert accounting_balances(info), info


def test_no_positive_reward_without_delivery():
    env = MediFlightEnv()
    rng = np.random.default_rng(0)
    for seed in range(30):
        env.reset(seed=seed)
        done = False
        while not done:
            _, r, te, tr, info = env.step(random_policy(env, rng))
            positive = {k: v for k, v in info["reward_breakdown"].items() if v > 0}
            assert set(positive) <= {"delivery_on_time", "delivery_late"}
            done = te or tr


# ------------------------------------------------------------------ masking
def test_mask_at_depot_and_capacity_limits():
    env = quiet_env()
    put_order(env, 0, weight=1.2)
    put_order(env, 1, weight=1.2)
    put_order(env, 2, weight=0.5)
    m = env.valid_actions()
    assert m[A_LOAD + 0] and m[A_LOAD + 1] and m[A_LOAD + 2]
    assert not m[A_RETURN] and not m[A_CHARGE]           # at depot, battery full
    assert not m[A_FLY:A_RETURN].any()                   # nothing on board yet
    env.step(A_LOAD + 0)
    env.step(A_LOAD + 1)                                 # 2.4 kg on board
    m = env.valid_actions()
    assert not m[A_LOAD + 2], "0.5 kg more would exceed the 2.5 kg capacity"
    assert m[A_FLY + 0] and m[A_FLY + 1] and not m[A_FLY + 2]


def test_package_count_limit():
    env = quiet_env()
    for s in range(4):
        put_order(env, s, weight=0.3)
    for s in range(3):
        env.step(A_LOAD + s)
    assert not env.valid_actions()[A_LOAD + 3]


def test_invalid_action_without_mask_is_penalised_not_executed():
    env = quiet_env(use_action_mask=False)
    obs, _ = env.reset(seed=0)
    assert obs["action_mask"].all()
    _, r, te, _, info = env.step(A_RETURN)               # already at depot
    assert not info["valid_action"] and info["invalid_actions"] == 1
    assert r == pytest.approx(-env.cfg.reward.invalid_action_penalty)
    assert env.time == pytest.approx(1.0) and env.location == 0 and not te


# ------------------------------------------------------------------ physics
def test_flight_leg_consumes_expected_energy_and_time():
    env = quiet_env()
    put_order(env, 0, weight=1.0)                        # Hospital North
    env.step(A_LOAD + 0)
    t0, b0 = env.time, env.battery
    dist = E.DISTANCE_KM[0, 1]
    e, m = E.leg_expectation(env.spec_drone, dist, 1.0, 0)
    _, r, *_ , info = env.step(A_FLY + 0)
    assert env.location == 1
    assert env.battery == pytest.approx(b0 - e)
    assert env.time == pytest.approx(t0 + m + env.spec_drone.drop_min)
    assert info["delivered_on_time"] == 1
    assert r == pytest.approx(E.MEDICINE.value - env.cfg.reward.energy_cost_per_pct * e)


def test_storm_costs_more_energy_than_calm():
    d = E.DroneSpec()
    calm, _ = E.leg_expectation(d, 5.0, 1.0, E.Weather.CALM)
    storm, _ = E.leg_expectation(d, 5.0, 1.0, E.Weather.STORM)
    assert storm > calm * 1.5


def test_payload_increases_leg_energy():
    d = E.DroneSpec()
    assert E.leg_expectation(d, 5.0, 2.0, 0)[0] > E.leg_expectation(d, 5.0, 0.0, 0)[0]


def test_multi_stop_route_order_changes_outcome():
    """The same two packages delivered in different orders give different results."""
    def run(order):
        env = quiet_env()
        put_order(env, 1, E.BLOOD, weight=1.0, deadline=14.0)   # Hospital East, tight
        put_order(env, 8, E.MEDICINE, weight=1.0, deadline=200.0)  # Home C, relaxed
        env.step(A_LOAD + 1)
        env.step(A_LOAD + 8)
        total = 0.0
        for s in order:
            total += env.step(A_FLY + s)[1]
        total += env.step(A_RETURN)[1]
        return total, env.stats["delivered_on_time"]
    urgent_first, n1 = run([1, 8])
    relaxed_first, n2 = run([8, 1])
    assert n1 == 2 and n2 == 1
    assert urgent_first > relaxed_first


def test_charging():
    env = quiet_env()
    env.battery = 50.0
    env.step(A_CHARGE)
    d = env.spec_drone
    assert env.battery == pytest.approx(50.0 + d.charge_rate_per_min * d.charge_block_min)
    assert env.time == pytest.approx(d.charge_block_min)
    env.battery = 98.0
    env.step(A_CHARGE)
    assert env.battery == pytest.approx(100.0)          # capped, partial block
    env.battery = 99.5
    assert not env.valid_actions()[A_CHARGE]             # <1% gain is not offered


def test_idle_drain_away_from_depot_only():
    env = quiet_env()
    env.step(A_HOLD)
    assert env.battery == pytest.approx(100.0)
    put_order(env, 3, weight=0.5)
    env.step(A_LOAD + 3)
    env.step(A_FLY + 3)
    b = env.battery
    env.step(A_HOLD)
    assert env.battery == pytest.approx(b - env.spec_drone.idle_drain_per_min * env.spec_drone.hold_min)


# ---------------------------------------------------- deadlines, cold chain
def test_pending_order_expires_with_penalty():
    env = quiet_env()
    put_order(env, 0, E.BLOOD, deadline=3.0)
    _, r, *_ , info = env.step(A_HOLD)                   # 5 minutes pass
    assert env.orders[0] is None and info["expired"] == 1
    assert r == pytest.approx(-env.cfg.reward.deadline_miss_fraction * E.BLOOD.value)


def test_late_delivery_earns_partial_value_once():
    env = quiet_env()
    put_order(env, 1, E.MEDICINE, weight=1.0, deadline=5.0)
    env.step(A_LOAD + 1)
    _, r, *_ , info = env.step(A_FLY + 1)                # ~11 min leg: arrives late
    rc = env.cfg.reward
    assert info["delivered_late"] == 1 and info["delivered_on_time"] == 0
    bd = info["reward_breakdown"]
    assert bd["deadline_missed"] == pytest.approx(-rc.deadline_miss_fraction * E.MEDICINE.value)
    assert bd["delivery_late"] == pytest.approx(rc.late_delivery_fraction * E.MEDICINE.value)
    # Late is better than never: net late outcome beats the miss penalty alone.
    assert bd["deadline_missed"] + bd["delivery_late"] > bd["deadline_missed"]


def test_cold_chain_spoilage():
    env = quiet_env()
    put_order(env, 0, E.BLOOD, weight=1.0, deadline=500.0)
    env.step(A_LOAD + 0)
    total = 0.0
    while env.orders[0] is not None:
        total += env.step(A_HOLD)[1]
    assert env.stats["spoiled"] == 1 and 0 not in env.payload
    # Cold-chain clock starts when loading starts (t=0); it is checked at the end of each
    # 5-minute hold, so the package is discarded at the first check past 60 minutes.
    assert E.BLOOD.cold_chain_min < env.time <= E.BLOOD.cold_chain_min + env.spec_drone.hold_min
    assert total < 0


# -------------------------------------------------------------- termination
def test_crash_terminates_with_failure_penalties():
    env = quiet_env()
    put_order(env, 1, E.BLOOD, weight=1.0, deadline=100.0)
    put_order(env, 5, E.MEDICINE, weight=1.0, deadline=100.0)  # stays pending
    env.step(A_LOAD + 1)
    env.battery = 5.0
    _, r, te, tr, info = env.step(A_FLY + 1)
    assert te and not tr and info["crashes"] == 1 and env.battery == 0.0
    bd = info["reward_breakdown"]
    rc = env.cfg.reward
    assert bd["crash"] == -rc.crash_penalty
    assert bd["crash_cargo_lost"] == pytest.approx(-E.BLOOD.value)
    assert bd["crash_orders_unserved"] == pytest.approx(-rc.deadline_miss_fraction * E.MEDICINE.value)
    assert accounting_balances(info)
    with pytest.raises(RuntimeError):
        env.step(A_HOLD)


def test_horizon_is_termination_and_drone_must_be_home():
    env = quiet_env(horizon_min=12.0)
    put_order(env, 3, weight=0.5, deadline=500.0)
    env.step(A_LOAD + 3)
    env.step(A_FLY + 3)                                   # parked at Pharmacy Central
    done, info = False, {}
    while not done:
        _, r, te, tr, info = env.step(A_HOLD)
        done = te or tr
    assert te and not tr
    assert info["reward_breakdown"]["not_at_depot"] == -env.cfg.reward.not_at_depot_penalty


def test_truncation_only_from_step_cap():
    env = quiet_env(max_steps=3)
    for _ in range(2):
        assert not env.step(A_HOLD)[3]
    _, _, te, tr, _ = env.step(A_HOLD)
    assert tr and not te


# ------------------------------------------------------------- sanity check
def test_baseline_ordering_is_sensible():
    env = MediFlightEnv()
    mean = {name: np.mean([run_episode(env, pol, s)["return"] for s in range(60)])
            for name, pol in BASELINES.items()}
    assert mean["edf"] > mean["nearest_first"] > mean["random"]


def test_stranded_at_site_is_mission_failure():
    env = quiet_env()
    put_order(env, 3, weight=0.5, deadline=500.0)
    env.step(A_LOAD + 3)
    env.step(A_FLY + 3)
    env.battery = 0.1                                    # less than one hold of idle drain
    t0 = env.time
    _, r, te, tr, info = env.step(A_HOLD)
    assert te and not tr and info["crashes"] == 1 and env.time > t0
    assert info["reward_breakdown"]["crash"] == -env.cfg.reward.crash_penalty
    assert accounting_balances(info)


# ------------------------------------------------ comparability & metrics
def _exogenous_trace(policy_name, seed):
    """All order requests (incl. rejected) and the weather seen at 10-min ticks."""
    env = MediFlightEnv(record_events=True)
    rng = np.random.default_rng(seed + 10_000)
    _, info = env.reset(seed=seed)
    requests = [(0.0, e["site"]) for e in info["events"] if e["type"] == "order_created"]
    done = False
    while not done:
        _, _, te, tr, info = env.step(BASELINES[policy_name](env, rng))
        requests += [(e["t"], e["site"]) for e in info["events"]
                     if e["type"] in ("order_created", "order_rejected")]
        done = te or tr
    return sorted(requests), env._weather_path.copy(), env.time


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_policies_face_identical_demand_and_weather(seed):
    """Common random numbers: exogenous demand/weather do not depend on actions."""
    edf_req, edf_w, edf_t = _exogenous_trace("edf", seed)
    near_req, near_w, near_t = _exogenous_trace("nearest_first", seed)
    np.testing.assert_array_equal(edf_w, near_w)
    horizon = min(edf_t, near_t)               # compare up to the shorter episode
    assert [r for r in edf_req if r[0] <= horizon] == [r for r in near_req if r[0] <= horizon]


def test_request_accounting_and_critical_metrics_from_event_log():
    """Recount deliveries/critical metrics independently from the event log."""
    critical = {p.name for p in E.PACKAGE_TYPES if p.urgency == 3}
    for policy in ("edf", "nearest_first", "random"):
        for seed in range(15):
            env = MediFlightEnv(record_events=True)
            rng = np.random.default_rng(seed)
            _, info = env.reset(seed=seed)
            events = list(info["events"])
            done = False
            while not done:
                _, _, te, tr, info = env.step(BASELINES[policy](env, rng))
                events += info["events"]
                done = te or tr
            created = [e for e in events if e["type"] == "order_created"]
            rejected = [e for e in events if e["type"] == "order_rejected"]
            on_time = [e for e in events if e["type"] == "deliver" and e["on_time"]]
            assert info["requests_total"] == len(created) + len(rejected)
            assert info["orders_created"] + info["rejected_busy"] == info["requests_total"]
            assert info["critical_created"] == sum(e["order"]["type"] in critical for e in created)
            assert info["critical_on_time"] == sum(e["package"] in critical for e in on_time)
            assert info["critical_on_time"] <= info["critical_created"] <= info["critical_requests"]
            assert accounting_balances(info)
