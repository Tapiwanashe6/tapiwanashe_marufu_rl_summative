# MediFlight AI

A mission-based Gymnasium reinforcement learning environment for emergency medical drone logistics.

## Quick start

```bash
uv sync
uv run main.py
uv run --extra dev pytest -q
```

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

## Environment

One drone serves 3 hospitals, 3 pharmacies and 5 homes. Each dispatch is a **round trip** from a medical depot; flight is simulated in time and energy rather than physically controlled. At decision boundaries, the drone is at base. Requests persist, new orders arrive, and deadlines may expire while the drone is busy. The agent selects the next order or chooses to wait/recharge.

- Actions: `0` wait 5 min, `1..11` deliver to a site, `12` return to base (redundant in this prototype), `13` recharge.
- Observation: 60-element float vector with normalized battery/time/weather/location/capacity plus 5 values for each site (pending, urgency, remaining deadline, weight, distance).
- Reward: on-time delivery +30/+60/+100 by urgency; expired/late order −40; invalid action −10; unsafe dispatch −12; travel energy/time cost; charging and waiting costs.
- Termination: 240 simulated minutes (Gymnasium `truncated=True`).
- Weather changes every ten simulated minutes and affects energy/travel time at dispatch.

**Limitations:** This is a dispatch model, not a low-level flight controller. Each site holds at most one pending order; the drone transports one package per trip. Arrival generation is a lightweight simulation and not calibrated to real demand. Battery percentage is an abstract energy budget. The 'return' action is currently redundant, retained for compatibility with the initial action specification; it should be removed or redesigned before training. Weather is sampled at ten-minute boundaries but a trip's wind factor is fixed at dispatch. Late deliveries are counted as missed; no clinical outcomes are modeled.

## Next milestones

1. Improve request-generation event timing and add action masks or remove redundant actions.
2. Add a deterministic heuristic baseline and evaluate over many seeds.
3. Integrate DQN, REINFORCE, PPO and A2C using TorchRL.
4. Log 10 meaningful hyperparameter runs per algorithm with identical evaluation seeds.
5. Build the Three.js renderer using logged trip events and create the report/video.

This release is the **first environment prototype**, not a trained RL system.

## Phase 2 — 3D mission-control replay

Generate a deterministic event log from the **real Python Gymnasium environment**:

```bash
uv run python -m visualization.export_events --seed 42 --policy heuristic
```

Then launch the browser viewer from the project root:

```bash
uv run python -m http.server 8001
```

Open **http://localhost:8001/visualization/** in Chrome or Safari. Use Play/Pause, Next/Previous, Reset, timeline seek and speed controls. Drag to orbit and scroll to zoom. Stop the server with Ctrl+C.

The browser must be online on the first load: Three.js and OrbitControls are imported from jsDelivr CDN. The Python exporter and local JSON file work offline. To replay a random baseline, regenerate with `--policy random` and refresh the browser. The heuristic is a rule-based comparison, **not a trained RL agent**.

### Accuracy of the visual replay

`visualization/events.json` is produced by stepping the **same `MediFlightEnv`** used in training. Each event records pre/post state, action, reward, weather, outstanding orders and delivery outcome. Completed delivery actions are animated as a **depot → destination → depot** flight, matching the environment's round-trip assumption. Time and battery values are snapshots at action boundaries, not continuous telemetry; intermediate drone positions are interpolated for presentation. The simulation does not implement flight physics, collision avoidance or real navigation. The 3D town is a schematic illustration, not a geographic map.

The original 14-action space is preserved. Action 12 (`return to base`) is currently redundant because all deliveries are round trips; this should be reconsidered before RL training. For credible training, also refine order-arrival timing and reward accounting, and expand environment tests.
