"""Domain model for MediFlight AI: map, package types, weather and drone physics.

All physical constants live here so the environment logic, the baselines, the
report and the visualisation share one source of truth.

Units: distance in km, time in simulated minutes, energy in % of battery,
mass in kg.  The numbers are plausible for a small fixed-wing/VTOL medical
drone (order of magnitude of published Zipline/Wingcopter figures) but are
*not* calibrated to a specific aircraft; they are chosen so that battery,
payload and deadlines all genuinely constrain the decision problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np


# --------------------------------------------------------------------------- map
@dataclass(frozen=True)
class Site:
    name: str
    kind: str  # "depot" | "hospital" | "pharmacy" | "home"
    x: float
    y: float


DEPOT = Site("Medical Distribution Center", "depot", 0.0, 0.0)

SITES: tuple[Site, ...] = (
    Site("Hospital North", "hospital", 4.0, 7.0),
    Site("Hospital East", "hospital", 9.0, 4.0),
    Site("Hospital South", "hospital", 3.0, -7.0),
    Site("Pharmacy Central", "pharmacy", 2.0, 2.0),
    Site("Pharmacy West", "pharmacy", -6.0, 3.0),
    Site("Pharmacy South", "pharmacy", -4.0, -5.0),
    Site("Home A", "home", 1.0, 5.0),
    Site("Home B", "home", -3.0, 6.0),
    Site("Home C", "home", 7.0, -2.0),
    Site("Home D", "home", -7.0, -4.0),
    Site("Home E", "home", 5.0, -6.0),
)

# Node 0 is the depot (with charging pad); node i (1..11) is SITES[i-1].
NODES: tuple[Site, ...] = (DEPOT,) + SITES
N_SITES = len(SITES)
N_NODES = len(NODES)


def _distance_matrix() -> np.ndarray:
    d = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    for i, a in enumerate(NODES):
        for j, b in enumerate(NODES):
            d[i, j] = math.hypot(a.x - b.x, a.y - b.y)
    return d


DISTANCE_KM: np.ndarray = _distance_matrix()
MAX_DISTANCE_KM: float = float(DISTANCE_KM.max())


# ---------------------------------------------------------------- package types
@dataclass(frozen=True)
class PackageType:
    name: str
    urgency: int                       # 1 routine, 2 priority, 3 critical
    value: float                       # reward for an on-time delivery
    weight_kg: tuple[float, float]     # uniform range
    deadline_min: tuple[float, float]  # uniform range, from order creation
    cold_chain_min: float | None       # max minutes on board before spoiling


BLOOD = PackageType("blood", 3, 10.0, (0.5, 1.2), (40.0, 60.0), 60.0)
EMERGENCY_KIT = PackageType("emergency_kit", 3, 9.0, (0.8, 1.6), (35.0, 50.0), None)
VACCINE = PackageType("vaccine", 2, 6.0, (0.6, 1.5), (90.0, 120.0), 90.0)
PRESCRIPTION = PackageType("prescription", 2, 4.0, (0.2, 0.8), (60.0, 120.0), None)
MEDICINE = PackageType("medicine", 1, 3.0, (0.8, 2.0), (120.0, 180.0), None)

PACKAGE_TYPES: tuple[PackageType, ...] = (BLOOD, EMERGENCY_KIT, VACCINE, PRESCRIPTION, MEDICINE)
MAX_VALUE = max(p.value for p in PACKAGE_TYPES)
MAX_COLD_CHAIN_MIN = max(p.cold_chain_min or 0.0 for p in PACKAGE_TYPES)

# Which package types each destination kind requests, with probabilities.
SITE_DEMAND: dict[str, tuple[tuple[PackageType, float], ...]] = {
    "hospital": ((BLOOD, 0.45), (EMERGENCY_KIT, 0.30), (VACCINE, 0.25)),
    "pharmacy": ((VACCINE, 0.35), (MEDICINE, 0.65)),
    "home": ((PRESCRIPTION, 0.75), (EMERGENCY_KIT, 0.25)),
}

# Poisson order-arrival rate per site (orders / minute) at arrival_scale = 1.
ARRIVAL_RATE_PER_MIN: dict[str, float] = {
    "hospital": 1 / 60,
    "pharmacy": 1 / 100,
    "home": 1 / 90,
}


# --------------------------------------------------------------------- weather
class Weather(IntEnum):
    CALM = 0
    WINDY = 1
    STORM = 2


WEATHER_NAMES = ("calm", "windy", "storm")
WEATHER_STEP_MIN = 10.0  # the weather Markov chain advances every 10 minutes
# Rows: current state, columns: next state.  Persistent, storms are rare but sticky.
WEATHER_TRANSITION = np.array(
    [
        [0.90, 0.09, 0.01],
        [0.10, 0.82, 0.08],
        [0.05, 0.25, 0.70],
    ]
)
# Multipliers applied to a flight leg, fixed at take-off.
WEATHER_ENERGY_FACTOR = (1.0, 1.3, 1.8)
WEATHER_SPEED_FACTOR = (1.0, 0.85, 0.7)


def stationary_weather_distribution() -> np.ndarray:
    vals, vecs = np.linalg.eig(WEATHER_TRANSITION.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return v / v.sum()


# ----------------------------------------------------------------------- drone
@dataclass(frozen=True)
class DroneSpec:
    battery_capacity: float = 100.0         # %
    payload_capacity_kg: float = 2.5
    max_packages: int = 3
    cruise_speed_km_per_min: float = 1.0    # 60 km/h in calm air
    energy_per_km: float = 1.6              # % per km, empty, calm
    energy_per_kg_km: float = 0.5           # extra % per km per kg of payload
    takeoff_landing_energy: float = 1.0     # % per leg
    takeoff_landing_min: float = 1.0        # min per leg
    energy_noise_sd: float = 0.06           # multiplicative gust noise on a leg
    charge_rate_per_min: float = 2.5        # % per minute on the depot pad
    charge_block_min: float = 10.0          # one charge action lasts up to 10 min
    idle_drain_per_min: float = 0.05        # % per minute when parked away from depot
    load_min: float = 1.0                   # minutes to load one package
    drop_min: float = 1.0                   # minutes to lower a package at a site
    hold_min: float = 5.0                   # minutes for a hold action


def leg_expectation(spec: DroneSpec, distance_km: float, payload_kg: float,
                    weather: int) -> tuple[float, float]:
    """Expected (energy %, minutes) of one flight leg, excluding gust noise."""
    if distance_km <= 0.0:
        return 0.0, 0.0
    energy = (distance_km * (spec.energy_per_km + spec.energy_per_kg_km * payload_kg)
              * WEATHER_ENERGY_FACTOR[weather] + spec.takeoff_landing_energy)
    minutes = (distance_km / (spec.cruise_speed_km_per_min * WEATHER_SPEED_FACTOR[weather])
               + spec.takeoff_landing_min)
    return energy, minutes
