"""MediFlight AI: sequential medical logistics environment.

Actions: 0 wait, 1..11 deliver to location, 12 return to base, 13 recharge.
All travel is round-trip to base: the drone carries one package per sortie.
This simplifying assumption is explicit and can be relaxed in later versions.
"""
from dataclasses import dataclass
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass(frozen=True)
class Site:
    name: str
    kind: str
    x: float
    y: float


SITES = (
    Site("Hospital North", "hospital", 4, 7),
    Site("Hospital East", "hospital", 9, 4),
    Site("Hospital South", "hospital", 3, -7),
    Site("Pharmacy Central", "pharmacy", 2, 2),
    Site("Pharmacy West", "pharmacy", -6, 3),
    Site("Pharmacy South", "pharmacy", -4, -5),
    Site("Home A", "home", 1, 5),
    Site("Home B", "home", -3, 6),
    Site("Home C", "home", 7, -2),
    Site("Home D", "home", -7, -4),
    Site("Home E", "home", 5, -6),
)


class MediFlightEnv(gym.Env):
    """Single-drone delivery dispatch environment with persistent order queues.

    Observation: [battery fraction, time fraction, weather index / 2,
    location index / 11, payload capacity / 2, then per site:
    pending flag, urgency / 3, remaining deadline / 120 clipped to [-1,1],
    package weight / 2, distance / 15].

    The simulator advances in minutes. A delivery action represents a complete
    round trip (dispatch, handoff, return); only the dispatch decision is learned.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, horizon=240, max_battery=100.0,
                 capacity_kg=2.0, arrival_rate=0.12):
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("Only None and 'ansi' render modes are supported")
        self.render_mode = render_mode
        self.horizon = float(horizon)
        self.max_battery = float(max_battery)
        self.capacity_kg = float(capacity_kg)
        self.arrival_rate = float(arrival_rate)
        self.n = len(SITES)
        self.action_space = spaces.Discrete(self.n + 3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                             shape=(5 + self.n * 5,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0.0
        self.battery = self.max_battery
        self.location = 0  # base, always 0 at decision boundaries
        self.weather = int(self.np_random.integers(0, 3))  # calm, breezy, windy
        self.orders = [None] * self.n
        self.completed = 0
        self.missed = 0
        self.total_energy = 0.0
        self.total_distance = 0.0
        self.terminated = False
        self.truncated = False
        self._spawn_orders(initial=True)
        return self._obs(), self._info()

    def _distance(self, idx):
        site = SITES[idx]
        return math.hypot(site.x, site.y)

    def _spawn_orders(self, initial=False):
        for idx, site in enumerate(SITES):
            if self.orders[idx] is not None:
                continue
            if self.np_random.random() < (0.5 if initial else self.arrival_rate):
                urgency = int(self.np_random.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
                if site.kind == "hospital":
                    urgency = max(urgency, 2)
                weight = float(self.np_random.uniform(0.2, 1.8))
                allowance = {1: 110, 2: 75, 3: 45}[urgency]
                self.orders[idx] = {"urgency": urgency, "weight": weight,
                                    "deadline": self.time + allowance,
                                    "created": self.time}

    def _expire_orders(self):
        # Orders become missed once their deadline passes; this is a real cost
        # even if the drone is busy with another delivery.
        lost = 0
        for i, order in enumerate(self.orders):
            if order is not None and self.time > order["deadline"]:
                self.orders[i] = None
                lost += 1
        self.missed += lost
        return lost

    def _advance(self, minutes):
        old = self.time
        self.time = min(self.horizon, self.time + minutes)
        # Generate arrivals at each crossed ten-minute boundary.
        for tick in range(int(old // 10) + 1, int(self.time // 10) + 1):
            self.weather = int(self.np_random.integers(0, 3))
            self._spawn_orders()
        return self._expire_orders()

    def _obs(self):
        data = [self.battery / self.max_battery,
                min(1.0, self.time / self.horizon), self.weather / 2,
                self.location / self.n, self.capacity_kg / 2]
        for i, order in enumerate(self.orders):
            if order is None:
                data.extend([0, 0, 0, 0, self._distance(i) / 15])
            else:
                data.extend([1, order["urgency"] / 3,
                             np.clip((order["deadline"] - self.time) / 120, -1, 1),
                             order["weight"] / 2, self._distance(i) / 15])
        return np.asarray(data, dtype=np.float32)

    def _info(self):
        return {"time_min": round(self.time, 2),
                "battery": round(self.battery, 2), "completed": self.completed,
                "missed": self.missed, "pending": sum(x is not None for x in self.orders),
                "weather": ("calm", "breezy", "windy")[self.weather],
                "distance_km": round(self.total_distance, 2)}

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("Episode ended; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"Action outside Discrete({self.n + 3}): {action}")
        action = int(action)
        reward = 0.0
        event = ""
        if action == 0:  # wait
            lost = self._advance(5.0)
            reward = -0.5 - 40 * lost
            event = "wait"
        elif 1 <= action <= self.n:
            idx = action - 1
            order = self.orders[idx]
            if order is None or order["weight"] > self.capacity_kg:
                lost = self._advance(2.0)
                reward = -10.0 - 40 * lost
                event = "invalid_delivery"
            else:
                distance = 2 * self._distance(idx)
                wind = [1.0, 1.2, 1.5][self.weather]
                travel_minutes = distance / 0.75 * wind + 3.0
                energy = distance * 2.0 * wind + 3.0
                if self.battery < energy + 5.0:
                    lost = self._advance(2.0)
                    reward = -12.0 - 40 * lost
                    event = "unsafe_battery_dispatch"
                else:
                    # Delivery occurs halfway through the trip, before return.
                    arrival = self.time + (travel_minutes - 3.0) / 2
                    on_time = arrival <= order["deadline"] and arrival <= self.horizon
                    self.orders[idx] = None
                    self.battery -= energy
                    self.total_energy += energy
                    self.total_distance += distance
                    self.completed += 1 if on_time else 0
                    self.missed += 0 if on_time else 1
                    lost = self._advance(travel_minutes)
                    reward = ({1: 30, 2: 60, 3: 100}[order["urgency"]] if on_time else -40)
                    reward -= 0.15 * energy + 0.05 * travel_minutes + 40 * lost
                    event = "delivered" if on_time else "late_delivery"
        elif action == self.n + 1:  # return: already at base after every sortie
            lost = self._advance(2.0)
            reward = -3.0 - 40 * lost
            event = "already_at_base"
        else:  # recharge at base
            if self.battery >= self.max_battery - 0.01:
                lost = self._advance(2.0)
                reward = -3.0 - 40 * lost
                event = "battery_full"
            else:
                duration = (self.max_battery - self.battery) / 2.0
                lost = self._advance(duration)
                self.battery = self.max_battery
                reward = -0.1 * duration - 40 * lost
                event = "recharged"
        self.truncated = self.time >= self.horizon
        info = self._info()
        info["event"] = event
        return self._obs(), float(reward), self.terminated, self.truncated, info

    def render(self):
        orders = [f"{i+1}: {SITES[i].name} (priority {o['urgency']}, "
                  f"deadline {o['deadline']:.0f})" for i, o in enumerate(self.orders) if o]
        result = (f"MediFlight | t={self.time:.1f} min | battery={self.battery:.1f}% "
                  f"| weather={self._info()['weather']} | delivered={self.completed} "
                  f"| missed={self.missed}\n" + "\n".join(orders))
        if self.render_mode == "ansi":
            return result
        print(result)

    def close(self):
        pass
