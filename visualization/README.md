# 3D Mission Control

From project root:

```sh
uv sync --extra dev
uv run python -m visualization.export_events --seed 42 --policy heuristic
uv run python -m http.server 8001
```

Then open http://localhost:8001/visualization/ . Requires internet access for Three.js CDN. This is a replay of Python environment transitions, not real-time bidirectional control.
