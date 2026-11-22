"""MediFlight AI v2: multi-stop medical drone delivery as a sequential decision problem.

One drone operates from a medical distribution center (node 0, with a charging
pad) serving 3 hospitals, 3 pharmacies and 5 homes.  Orders arrive over time
(Poisson), each with a package type that fixes its urgency, clinical value,
deadline and cold-chain limit.  The agent controls the drone one operation at a
time: load packages at the depot (subject to mass and count capacity), fly
legs between nodes in any order, return, charge, or hold.  Every action takes
simulated time, during which new orders arrive, weather evolves and deadlines
expire, so decisions have delayed, persistent consequences.

This is a semi-Markov decision process: actions have variable duration and the
discount is applied per decision (a documented simplification).

Action layout, Discrete(25):
    0            hold 5 minutes
    1  .. 11     load the active order for site i (depot only)
    12 .. 22     fly to site i (delivers its package on arrival)
    23           return to depot
    24           charge for up to 10 minutes (depot only)

Observation: Dict{"observation": Box(OBS_DIM), "action_mask": MultiBinary(25)}.
See observation_spec.py for every element.

Termination (`terminated=True`):
    * mission horizon reached (time is part of the state, so this is a true
      terminal, not a time-limit truncation);
    * battery exhausted in flight (crash, mission failure, cargo lost).
Truncation (`truncated=True`): only a safety cap on the number of decisions.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import entities as E
from .observation_spec import OBS_DIM

# ------------------------------------------------------------------- actions
A_HOLD = 0
A_LOAD = 1                      # 1..11
A_FLY = A_LOAD + E.N_SITES      # 12..22
A_RETURN = A_FLY + E.N_SITES    # 23
A_CHARGE = A_RETURN + 1         # 24
N_ACTIONS = A_CHARGE + 1        # 25


def action_name(a: int) -> str:
    a = int(a)
    if a == A_HOLD:
        return "hold"
    if A_LOAD <= a < A_FLY:
        return f"load:{E.SITES[a - A_LOAD].name}"
    if A_FLY <= a < A_RETURN:
        return f"fly:{E.SITES[a - A_FLY].name}"
    if a == A_RETURN:
        return "return_to_depot"
    if a == A_CHARGE:
        return "charge"
    raise ValueError(a)


# -------------------------------------------------------------------- config
@dataclass(frozen=True)
class RewardConfig:
    late_delivery_fraction: float = 0.25   # value earned for a late delivery
    deadline_miss_fraction: float = 0.5    # penalty (x value) when a deadline passes
    spoilage_fraction: float = 1.0         # penalty (x value) when cold chain is broken
    crash_penalty: float = 50.0            # plus all active orders counted as failed
    energy_cost_per_pct: float = 0.02
    not_at_depot_penalty: float = 5.0      # drone not recovered at end of shift
    invalid_action_penalty: float = 1.0    # only reachable when masking is off


@dataclass(frozen=True)
class EnvConfig:
    horizon_min: float = 240.0
    initial_orders: tuple[int, int] = (4, 6)   # inclusive range
    arrival_scale: float = 1.0                 # multiplies all arrival rates
    initial_weather: int | None = None         # None: sample stationary distribution
    storm_bias: float = 0.0                    # >0 shifts weather towards storms (stress test)
    use_action_mask: bool = True
    max_steps: int = 500
    drone: E.DroneSpec = field(default_factory=E.DroneSpec)
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class Order:
    order_id: int
    site: int                 # 0..10 (index into SITES)
    ptype: E.PackageType
    weight: float
    created: float
    deadline: float
    status: str = "pending"   # pending | onboard
    loaded_at: float | None = None
    deadline_missed: bool = False

    def to_dict(self, now: float) -> dict[str, Any]:
        return {"id": self.order_id, "site": self.site, "type": self.ptype.name,
                "urgency": self.ptype.urgency, "value": self.ptype.value,
                "weight": round(self.weight, 3), "created": round(self.created, 2),
                "deadline": round(self.deadline, 2), "slack": round(self.deadline - now, 2),
                "status": self.status, "deadline_missed": self.deadline_missed,
                "cold_chain_left": None if self.ptype.cold_chain_min is None else round(
                    self.ptype.cold_chain_min - (0.0 if self.loaded_at is None else now - self.loaded_at), 2)}


STAT_KEYS = ("orders_created", "delivered_on_time", "delivered_late", "expired",
             "spoiled", "lost_in_crash", "undelivered_onboard_at_end", "pending_at_end",
             "invalid_actions", "crashes", "energy_used", "distance_km", "flight_minutes",
             "charge_minutes", "value_delivered", "value_available",
             "critical_created", "critical_on_time")


class MediFlightEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None,
                 record_events: bool = False):
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi' (3D view: see rendering.py)")
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.record_events = record_events
        self.spec_drone = self.cfg.drone
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(N_ACTIONS),
        })
        self._weather_P = self._weather_matrix(self.cfg.storm_bias)
        self._started = False

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _weather_matrix(storm_bias: float) -> np.ndarray:
        P = E.WEATHER_TRANSITION.copy()
        if storm_bias > 0:
            P[:, E.Weather.STORM] += storm_bias
            P /= P.sum(axis=1, keepdims=True)
        return P

    @property
    def payload_weight(self) -> float:
        return sum(self.orders[s].weight for s in self.payload)

    def _emit(self, kind: str, **data: Any) -> None:
        if self.record_events:
            self._events.append({"t": round(self.time, 3), "type": kind, **data})

    def _reward(self, amount: float, component: str) -> None:
        self._step_reward += amount
        self._breakdown[component] += amount

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        d = self.spec_drone
        self.time = 0.0
        self.steps = 0
        self.battery = d.battery_capacity
        self.location = 0
        self.orders: list[Order | None] = [None] * E.N_SITES
        self.payload: list[int] = []       # site indices whose package is on board
        self.stats = {k: 0.0 for k in STAT_KEYS}
        self._next_order_id = 0
        self._events: list[dict] = []
        self._done = False
        self._step_reward, self._breakdown = 0.0, defaultdict(float)
        if self.cfg.initial_weather is None:
            self.weather = int(self.np_random.choice(3, p=E.stationary_weather_distribution()))
        else:
            self.weather = int(self.cfg.initial_weather)
        lo, hi = self.cfg.initial_orders
        n0 = int(self.np_random.integers(lo, hi + 1))
        for s in self.np_random.choice(E.N_SITES, size=n0, replace=False):
            self._create_order(int(s), 0.0)
        self._started = True
        return self._observation(), self._info(action=None, valid=True)

    def _create_order(self, site: int, t: float) -> None:
        kind = E.SITES[site].kind
        types, probs = zip(*E.SITE_DEMAND[kind])
        ptype = types[int(self.np_random.choice(len(types), p=np.asarray(probs)))]
        weight = float(self.np_random.uniform(*ptype.weight_kg))
        deadline = t + float(self.np_random.uniform(*ptype.deadline_min))
        order = Order(self._next_order_id, site, ptype, weight, t, deadline)
        self._next_order_id += 1
        self.orders[site] = order
        self.stats["orders_created"] += 1
        self.stats["value_available"] += ptype.value
        if ptype.urgency == 3:
            self.stats["critical_created"] += 1
        self._emit("order_created", site=site, order=order.to_dict(t))

    # ------------------------------------------------------------- time model
    def _advance(self, minutes: float) -> None:
        """Advance the world clock: weather, order arrivals, deadlines, cold chain."""
        if minutes <= 0:
            return
        t0, t1 = self.time, self.time + minutes
        # Weather Markov chain ticks at every 10-minute boundary crossed.
        for _ in range(int(t0 // E.WEATHER_STEP_MIN) + 1, int(t1 // E.WEATHER_STEP_MIN) + 1):
            new = int(self.np_random.choice(3, p=self._weather_P[self.weather]))
            if new != self.weather:
                self.weather = new
                self._emit("weather", weather=E.WEATHER_NAMES[new])
        # Poisson arrivals at sites without an active order, only within the shift.
        span = max(0.0, min(t1, self.cfg.horizon_min) - t0)
        if span > 0:
            for s, site in enumerate(E.SITES):
                if self.orders[s] is not None:
                    continue
                rate = E.ARRIVAL_RATE_PER_MIN[site.kind] * self.cfg.arrival_scale
                if self.np_random.random() < 1.0 - math.exp(-rate * span):
                    self._create_order(s, t0 + float(self.np_random.random()) * span)
        self.time = t1
        # Deadlines: pending orders are reassigned (expired); on-board orders become late.
        for s, o in enumerate(self.orders):
            if o is None or o.deadline_missed or o.deadline >= self.time:
                continue
            o.deadline_missed = True
            self._reward(-self.cfg.reward.deadline_miss_fraction * o.ptype.value, "deadline_missed")
            if o.status == "pending":
                self.orders[s] = None
                self.stats["expired"] += 1
                self._emit("order_expired", site=s, order_id=o.order_id)
            else:
                self._emit("order_late", site=s, order_id=o.order_id)
        # Cold chain: blood/vaccines spoil if kept on board too long.
        for s in list(self.payload):
            o = self.orders[s]
            lim = o.ptype.cold_chain_min
            if lim is not None and self.time - o.loaded_at > lim:
                self.payload.remove(s)
                self.orders[s] = None
                self.stats["spoiled"] += 1
                self._reward(-self.cfg.reward.spoilage_fraction * o.ptype.value, "spoilage")
                self._emit("spoiled", site=s, order_id=o.order_id)

    # ----------------------------------------------------------------- masks
    def valid_actions(self) -> np.ndarray:
        """True physical/operational validity of each action in the current state."""
        d = self.spec_drone
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[A_HOLD] = True
        at_depot = self.location == 0
        if at_depot:
            w, n = self.payload_weight, len(self.payload)
            for s, o in enumerate(self.orders):
                if (o is not None and o.status == "pending" and n < d.max_packages
                        and w + o.weight <= d.payload_capacity_kg + 1e-9):
                    m[A_LOAD + s] = True
            # Charging is only offered when it adds at least 1% (no zero-length actions).
            m[A_CHARGE] = self.battery <= d.battery_capacity - 1.0
        else:
            m[A_RETURN] = True
        for s in self.payload:
            if self.location != s + 1:
                m[A_FLY + s] = True
        return m

    def action_masks(self) -> np.ndarray:
        """Mask exposed to the agent (all ones when masking is disabled)."""
        if self.cfg.use_action_mask:
            return self.valid_actions().astype(np.int8)
        return np.ones(N_ACTIONS, dtype=np.int8)

    # ------------------------------------------------------------------ step
    def step(self, action):
        if not self._started:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode has ended; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} outside Discrete({N_ACTIONS})")
        a = int(action)
        self._events = []
        self._step_reward, self._breakdown = 0.0, defaultdict(float)
        t_start = self.time
        crashed = False
        valid = bool(self.valid_actions()[a])

        if not valid:
            # Only reachable with masking disabled: the command is rejected by the
            # flight controller, wasting a minute.
            self.stats["invalid_actions"] += 1
            self._reward(-self.cfg.reward.invalid_action_penalty, "invalid_action")
            self._emit("invalid_action", action=a, name=action_name(a))
            crashed = self._idle(1.0)
        elif a == A_HOLD:
            self._emit("hold", minutes=self.spec_drone.hold_min)
            crashed = self._idle(self.spec_drone.hold_min)
        elif A_LOAD <= a < A_FLY:
            s = a - A_LOAD
            o = self.orders[s]
            o.status, o.loaded_at = "onboard", self.time
            self.payload.append(s)
            self._emit("load", site=s, order_id=o.order_id, package=o.ptype.name)
            self._advance(self.spec_drone.load_min)
        elif A_FLY <= a < A_RETURN:
            crashed = self._fly(a - A_FLY + 1)
        elif a == A_RETURN:
            crashed = self._fly(0)
        else:  # A_CHARGE
            self._charge()

        self.steps += 1
        terminated = crashed or self.time >= self.cfg.horizon_min
        if terminated and not crashed:
            self._end_of_mission()
        truncated = (not terminated) and self.steps >= self.cfg.max_steps
        self._done = terminated or truncated
        info = self._info(action=a, valid=valid, t_start=t_start)
        return self._observation(), float(self._step_reward), terminated, truncated, info

    def _idle(self, minutes: float) -> bool:
        """Wait in place. Returns True if the drone is stranded (battery flat at a site)."""
        if self.location != 0:  # parked at a site pad: avionics/heater drain
            drain = min(self.battery, self.spec_drone.idle_drain_per_min * minutes)
            self.battery -= drain
            self.stats["energy_used"] += drain
            if self.battery <= 0.0:
                self._advance(minutes)
                self._mission_failure("stranded", origin=self.location, destination=self.location,
                                      fraction=0.0)
                return True
        self._advance(minutes)
        return False

    def _fly(self, dest: int) -> bool:
        """Fly one leg to node `dest`. Returns True if the drone crashed."""
        d = self.spec_drone
        dist = float(E.DISTANCE_KM[self.location, dest])
        exp_energy, minutes = E.leg_expectation(d, dist, self.payload_weight, self.weather)
        gust = float(np.clip(self.np_random.normal(1.0, d.energy_noise_sd), 0.8, 1.25))
        energy = exp_energy * gust
        origin = self.location
        self._emit("takeoff", origin=origin, destination=dest,
                   payload=list(self.payload), weather=E.WEATHER_NAMES[self.weather],
                   expected_energy=round(exp_energy, 2), minutes=round(minutes, 2))
        if energy > self.battery:
            frac = self.battery / energy
            self.stats["energy_used"] += self.battery
            self.stats["distance_km"] += dist * frac
            self.stats["flight_minutes"] += minutes * frac
            self._reward(-self.cfg.reward.energy_cost_per_pct * self.battery, "energy")
            self.battery = 0.0
            self._advance(minutes * frac)
            self._mission_failure("crash", origin=origin, destination=dest, fraction=frac)
            return True
        self.battery -= energy
        self.stats["energy_used"] += energy
        self.stats["distance_km"] += dist
        self.stats["flight_minutes"] += minutes
        self._reward(-self.cfg.reward.energy_cost_per_pct * energy, "energy")
        self._advance(minutes)
        self.location = dest
        self._emit("arrive", node=dest, battery=round(self.battery, 2))
        if dest > 0 and (dest - 1) in self.payload:
            self._deliver(dest - 1)
        return False

    def _mission_failure(self, reason: str, origin: int, destination: int, fraction: float) -> None:
        """Battery exhausted (in flight or parked): the mission ends in failure.

        Cargo is destroyed (full value lost) and every other active order goes
        unserved. Without this, ending the episode early could look attractive
        because it stops further deadline penalties accruing.
        """
        r = self.cfg.reward
        lost = len(self.payload)
        self.battery = 0.0
        self.stats["lost_in_crash"] += lost
        self.stats["crashes"] += 1
        for s in self.payload:
            self._reward(-r.spoilage_fraction * self.orders[s].ptype.value, "crash_cargo_lost")
            self.orders[s] = None
        self.payload = []
        for o in self.orders:
            if o is not None and not o.deadline_missed:
                self._reward(-r.deadline_miss_fraction * o.ptype.value, "crash_orders_unserved")
        self.stats["pending_at_end"] = sum(o is not None for o in self.orders)
        self._reward(-r.crash_penalty, "crash")
        self._emit("crash", reason=reason, origin=origin, destination=destination,
                   fraction=round(fraction, 3), packages_lost=lost)

    def _deliver(self, s: int) -> None:
        o = self.orders[s]
        r = self.cfg.reward
        on_time = self.time <= o.deadline and not o.deadline_missed
        self.payload.remove(s)
        self.orders[s] = None
        if on_time:
            self.stats["delivered_on_time"] += 1
            self.stats["value_delivered"] += o.ptype.value
            if o.ptype.urgency == 3:
                self.stats["critical_on_time"] += 1
            self._reward(o.ptype.value, "delivery_on_time")
        else:
            self.stats["delivered_late"] += 1
            self._reward(r.late_delivery_fraction * o.ptype.value, "delivery_late")
        self._emit("deliver", site=s, order_id=o.order_id, package=o.ptype.name,
                   on_time=on_time, lateness=round(max(0.0, self.time - o.deadline), 2))
        self._advance(self.spec_drone.drop_min)

    def _charge(self) -> None:
        d = self.spec_drone
        minutes = min(d.charge_block_min, (d.battery_capacity - self.battery) / d.charge_rate_per_min)
        before = self.battery
        self._advance(minutes)
        self.battery = min(d.battery_capacity, self.battery + d.charge_rate_per_min * minutes)
        self.stats["charge_minutes"] += minutes
        self._emit("charge", minutes=round(minutes, 2), battery_before=round(before, 2),
                   battery_after=round(self.battery, 2))

    def _end_of_mission(self) -> None:
        r = self.cfg.reward
        for s in list(self.payload):
            o = self.orders[s]
            if not o.deadline_missed:
                self._reward(-r.deadline_miss_fraction * o.ptype.value, "undelivered_at_end")
            self.stats["undelivered_onboard_at_end"] += 1
        self.stats["pending_at_end"] = sum(
            1 for o in self.orders if o is not None and o.status == "pending")
        if self.location != 0:
            self._reward(-r.not_at_depot_penalty, "not_at_depot")
        self._emit("mission_end", at_depot=self.location == 0)

    # ------------------------------------------------------------ observation
    def _observation(self) -> dict[str, np.ndarray]:
        d = self.spec_drone
        cap = d.battery_capacity
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = self.battery / cap
        obs[1] = max(0.0, 1.0 - self.time / self.cfg.horizon_min)
        obs[2 + self.weather] = 1.0
        obs[5 + self.location] = 1.0
        k = 5 + E.N_NODES
        w = self.payload_weight
        obs[k] = w / d.payload_capacity_kg
        obs[k + 1] = len(self.payload) / d.max_packages
        e_home, _ = E.leg_expectation(d, E.DISTANCE_KM[self.location, 0], w, self.weather)
        obs[k + 2] = min(1.0, e_home / cap)
        k += 3
        for s, o in enumerate(self.orders):
            e_leg, _ = E.leg_expectation(d, E.DISTANCE_KM[self.location, s + 1], w, self.weather)
            if o is not None:
                obs[k] = float(o.status == "pending")
                obs[k + 1] = float(o.status == "onboard")
                obs[k + 2] = o.ptype.urgency / 3
                obs[k + 3] = o.ptype.value / E.MAX_VALUE
                obs[k + 4] = np.clip((o.deadline - self.time) / 120.0, -1.0, 1.0)
                obs[k + 5] = o.weight / d.payload_capacity_kg
                if o.ptype.cold_chain_min is not None:
                    left = o.ptype.cold_chain_min - (0.0 if o.loaded_at is None else self.time - o.loaded_at)
                    obs[k + 6] = np.clip(left / E.MAX_COLD_CHAIN_MIN, 0.0, 1.0)
            obs[k + 7] = min(1.0, e_leg / cap)
            k += 8
        return {"observation": obs, "action_mask": self.action_masks()}

    def _info(self, action: int | None, valid: bool, t_start: float | None = None) -> dict:
        info = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.stats.items()}
        info.update(time_min=round(self.time, 3), battery=round(self.battery, 3),
                    location=self.location, weather=E.WEATHER_NAMES[self.weather],
                    payload=list(self.payload),
                    pending=sum(o is not None and o.status == "pending" for o in self.orders),
                    action=action, action_name=None if action is None else action_name(action),
                    valid_action=valid, reward_breakdown=dict(self._breakdown))
        if t_start is not None:
            info["duration_min"] = round(self.time - t_start, 3)
        if self.record_events:
            info["events"] = list(self._events)
        return info

    # ------------------------------------------------------------- snapshots
    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable full state (used by rendering.py and the 3D viewer)."""
        return {"time": round(self.time, 3), "battery": round(self.battery, 3),
                "location": self.location, "weather": E.WEATHER_NAMES[self.weather],
                "payload": list(self.payload),
                "payload_kg": round(self.payload_weight, 3),
                "orders": [None if o is None else o.to_dict(self.time) for o in self.orders],
                "stats": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.stats.items()}}

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def render(self):
        lines = [f"t={self.time:6.1f} min  battery={self.battery:5.1f}%  weather={E.WEATHER_NAMES[self.weather]:5s}  "
                 f"at={E.NODES[self.location].name}  payload={[E.SITES[s].name for s in self.payload]}  "
                 f"on-time={int(self.stats['delivered_on_time'])} late={int(self.stats['delivered_late'])} "
                 f"expired={int(self.stats['expired'])}"]
        for s, o in enumerate(self.orders):
            if o is not None:
                lines.append(f"   [{o.status:7s}] {E.SITES[s].name:16s} {o.ptype.name:13s} "
                             f"P{o.ptype.urgency} {o.weight:.2f}kg slack={o.deadline - self.time:6.1f}")
        text = "\n".join(lines)
        if self.render_mode == "ansi":
            return text
        print(text)
        return None
