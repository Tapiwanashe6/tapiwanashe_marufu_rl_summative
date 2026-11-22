"""Single source of truth for the observation vector.

The environment builds its observation in exactly this order, a unit test checks
that the dimensions match, and `markdown_table()` generates the report's
observation-provenance table (rubric: meaning, encoding/range, source,
update cadence, agent-input path).

"Source" names the real-world system that would supply the value on a deployed
drone; in this project every value is produced by the simulator rule given in
`rule`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .entities import N_NODES, N_SITES, SITES, WEATHER_NAMES


@dataclass(frozen=True)
class ObsFeature:
    name: str
    description: str
    source: str        # real-world provenance (sensor / API / system)
    rule: str          # how the simulator produces it
    encoding: str
    value_range: str
    cadence: str
    width: int = 1


GLOBAL_FEATURES: tuple[ObsFeature, ...] = (
    ObsFeature("battery", "State of charge of the drone battery",
               "Sensor: battery management system", "100% at start; minus leg energy, plus charging",
               "float32, battery / capacity", "[0, 1]", "every decision step"),
    ObsFeature("time_remaining", "Fraction of the mission shift remaining",
               "System: mission clock", "1 - t / horizon",
               "float32", "[0, 1]", "every decision step"),
    ObsFeature("weather", f"Current weather ({', '.join(WEATHER_NAMES)})",
               "API: weather service / onboard anemometer",
               "3-state Markov chain advanced every 10 simulated minutes",
               "one-hot (3)", "{0, 1}", "every 10 sim-minutes", width=3),
    ObsFeature("location", "Current node: depot or one of 11 sites",
               "Sensor: GNSS position snapped to node", "set on arrival of each flight leg",
               f"one-hot ({N_NODES})", "{0, 1}", "every decision step", width=N_NODES),
    ObsFeature("payload_weight", "Mass currently carried",
               "Sensor: payload load cell", "sum of weights of loaded packages",
               "float32, kg / payload capacity", "[0, 1]", "every decision step"),
    ObsFeature("payload_count", "Number of packages on board",
               "System: cargo manifest", "count of loaded packages",
               "float32, count / max packages", "[0, 1]", "every decision step"),
    ObsFeature("energy_to_depot", "Expected energy to fly home now",
               "Derived: map + weather + payload", "leg energy model, current weather and payload",
               "float32, % / capacity, clipped", "[0, 1]", "every decision step"),
)

PER_SITE_FEATURES: tuple[ObsFeature, ...] = (
    ObsFeature("pending", "An order for this site is waiting at the depot",
               "API: hospital/pharmacy/patient order system", "Poisson arrivals, max 1 active order per site",
               "binary", "{0, 1}", "every decision step"),
    ObsFeature("on_board", "This site's package is loaded on the drone",
               "System: cargo manifest", "set by a load action, cleared on delivery/spoilage",
               "binary", "{0, 1}", "every decision step"),
    ObsFeature("urgency", "Clinical urgency of the active order",
               "API: order system (triage level)", "fixed by package type",
               "float32, level / 3 (0 if none)", "[0, 1]", "on order arrival"),
    ObsFeature("value", "Clinical value of delivering the order on time",
               "API: order system", "fixed by package type",
               "float32, value / max value (0 if none)", "[0, 1]", "on order arrival"),
    ObsFeature("deadline_slack", "Minutes left until the deadline",
               "API: order system (requested delivery time)", "deadline - t",
               "float32, minutes / 120, clipped", "[-1, 1]", "every decision step"),
    ObsFeature("weight", "Package mass",
               "Dataset: dispatch manifest", "uniform within package-type range",
               "float32, kg / payload capacity (0 if none)", "[0, 1]", "on order arrival"),
    ObsFeature("cold_chain_left", "Minutes of cold-chain life left (blood, vaccine)",
               "Sensor: cold-box temperature logger", "limit - minutes on board; 1.0 if not yet loaded",
               "float32, minutes / 90 (0 if not cold-chain)", "[0, 1]", "every decision step"),
    ObsFeature("leg_energy", "Expected energy to fly from current node to this site",
               "Derived: map + weather + payload", "leg energy model, current weather and payload",
               "float32, % / capacity, clipped", "[0, 1]", "every decision step"),
)

GLOBAL_DIM = sum(f.width for f in GLOBAL_FEATURES)
PER_SITE_DIM = sum(f.width for f in PER_SITE_FEATURES)
OBS_DIM = GLOBAL_DIM + N_SITES * PER_SITE_DIM


def feature_names() -> list[str]:
    """Flat list of names, one per observation element (for debugging/plots)."""
    names: list[str] = []
    for f in GLOBAL_FEATURES:
        names += [f.name] if f.width == 1 else [f"{f.name}[{i}]" for i in range(f.width)]
    for site in SITES:
        names += [f"{site.name}.{f.name}" for f in PER_SITE_FEATURES]
    return names


def markdown_table() -> str:
    """Observation table in the report template's column format."""
    rows = ["| Observation | Description | Source (Sensor/Camera/API/Dataset) | "
            "Encoding / Data Type | Range | Update cadence | Simulator rule |",
            "|---|---|---|---|---|---|---|"]
    for f in GLOBAL_FEATURES:
        rows.append(f"| {f.name} | {f.description} | {f.source} | {f.encoding} | "
                    f"{f.value_range} | {f.cadence} | {f.rule} |")
    for f in PER_SITE_FEATURES:
        rows.append(f"| site.{f.name} (x{N_SITES}) | {f.description} | {f.source} | "
                    f"{f.encoding} | {f.value_range} | {f.cadence} | {f.rule} |")
    rows.append(f"\nTotal: {OBS_DIM} float32 values = {GLOBAL_DIM} global + "
                f"{N_SITES} sites x {PER_SITE_DIM}. Agent-input path: Gymnasium Dict "
                "observation key `observation` -> TorchRL GymWrapper -> MLP input; "
                "key `action_mask` -> masked Q-values / MaskedCategorical.")
    return "\n".join(rows)
