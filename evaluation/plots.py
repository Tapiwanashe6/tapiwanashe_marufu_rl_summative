"""Report figures (PNG, 200 dpi) from sweep logs and results/.

    uv run python -m evaluation.plots          # writes assets/figures/*.png

Figure -> template section:
  fig1_cumulative_rewards      5.1  training + validation return, best config per algorithm
  fig2_learning_comparison     5.1  validation return of all four best models on one axis
  fig3_dqn_objective           5.1  DQN loss and mean Q (training stability)
  fig4_pg_entropy              5.1  policy entropy for REINFORCE / PPO / A2C
  fig5_convergence             5.2  episodes to converge (definition in convergence_point)
  fig6_generalisation          5.3  return vs EDF under shifted scenarios
  fig7_test_comparison         5.x  held-out test return and critical service vs baselines
  fig8_sweep_sensitivity       4.x  validation return of every configuration
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluation.compare import LOGS, RESULTS  # noqa: E402
from training.common import ROOT  # noqa: E402

OUT = ROOT / "assets" / "figures"
ALGOS = ("dqn", "reinforce", "ppo", "a2c")
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C",
         "edf": "EDF dispatcher", "nearest_first": "Nearest-first", "random": "Random"}
# Reference categorical palette, fixed order (identity follows the algorithm, never rank).
COLOR = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BASE_STYLE = {"edf": dict(color="#52514e", ls="--"), "nearest_first": dict(color="#898781", ls=":")}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2.0,
})


# ------------------------------------------------------------------ data
def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def col(rows, key):
    return np.asarray([float(r[key]) if r.get(key) not in (None, "") else np.nan for r in rows])


def best_configs() -> dict:
    return json.loads((RESULTS / "best_configs.json").read_text())


def seed_dirs(algo: str, name: str) -> list[Path]:
    return sorted((LOGS / algo / name).glob("seed_*"))


def on_grid(xs: list[np.ndarray], ys: list[np.ndarray], n: int = 120):
    """Interpolate several seeds' curves onto a shared x grid -> (grid, mean, lo, hi)."""
    hi_x = min(x[-1] for x in xs)
    grid = np.linspace(min(x[0] for x in xs), hi_x, n)
    m = np.vstack([np.interp(grid, x[~np.isnan(y)], y[~np.isnan(y)]) for x, y in zip(xs, ys)])
    return grid, m.mean(0), m.min(0), m.max(0)


def smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(y) < k:
        return y
    yy = np.where(np.isnan(y), np.nanmean(y), y)
    # centred moving average; divide by the number of real samples in each window so the
    # first and last points are not pulled towards zero by implicit zero padding
    w = np.ones(k)
    return np.convolve(yy, w, mode="same") / np.convolve(np.ones_like(yy), w, mode="same")


def baseline_val() -> dict[str, float]:
    """Validation-seed baseline returns (same seeds as the agents' validation curves)."""
    path = RESULTS / "baselines_validation.json"
    if not path.exists():
        from evaluation.protocol import VALIDATION_SEEDS, baseline_act, run_episodes, summarize
        n = 20  # the agents' periodic validation uses the first 20 validation seeds
        data = {b: summarize(run_episodes(lambda s, b=b: baseline_act(b, s), VALIDATION_SEEDS[:n]))["return_mean"]
                for b in ("edf", "nearest_first")}
        path.write_text(json.dumps(data, indent=2))
    return json.loads(path.read_text())


def ref_lines(ax, values: dict[str, float], x_label_pos: float) -> None:
    for b, v in values.items():
        ax.axhline(v, lw=1.2, zorder=1, **BASE_STYLE[b])
        ax.text(x_label_pos, v, f" {LABEL[b]} {v:.1f}", va="bottom", ha="left", fontsize=7.5, color=INK2)


def frames_k(ax) -> None:
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# --------------------------------------------------------------- figures
def fig_cumulative_rewards(best, base) -> None:
    """Template 5.1: cumulative (per-episode) reward over training episodes, one subplot per method."""
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4), sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        dirs = seed_dirs(algo, best[algo]["name"])
        tr = [read_csv(d / "train.csv") for d in dirs]
        ev = [read_csv(d / "eval.csv") for d in dirs]
        for rows in tr:
            x, y = col(rows, "episodes"), col(rows, "train_return")
            ok = ~np.isnan(y)
            ax.plot(x[ok], smooth(y[ok], 5), color=COLOR[algo], lw=0.8, alpha=0.35)
        g, m, lo, hi = on_grid([col(r, "episodes") for r in ev], [col(r, "return_mean") for r in ev])
        ax.fill_between(g, lo, hi, color=COLOR[algo], alpha=0.18, lw=0)
        ax.plot(g, m, color=COLOR[algo], lw=2.0)
        ref_lines(ax, base, g[0])
        ax.set_title(f"{LABEL[algo]} ({best[algo]['name']})")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("training episodes")
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative reward per episode")
    fig.tight_layout()
    save(fig, "fig1_cumulative_rewards")


def fig_learning_comparison(best, base) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for algo in ALGOS:
        ev = [read_csv(d / "eval.csv") for d in seed_dirs(algo, best[algo]["name"])]
        g, m, lo, hi = on_grid([col(r, "frames") for r in ev], [col(r, "return_mean") for r in ev])
        ax.fill_between(g, lo, hi, color=COLOR[algo], alpha=0.12, lw=0)
        ax.plot(g, m, color=COLOR[algo], label=LABEL[algo])
        ax.text(g[-1], m[-1], f" {LABEL[algo]}", color=INK, fontsize=8, va="center")
    ref_lines(ax, base, 0)
    frames_k(ax)
    ax.set_xlabel("environment frames")
    ax.set_ylabel("validation return (20 seeds)")
    ax.set_title("Validation return of each algorithm's selected configuration")
    ax.legend(loc="lower right")
    save(fig, "fig2_learning_comparison")


def fig_dqn_objective(best) -> None:
    dirs = seed_dirs("dqn", best["dqn"]["name"])
    rows = [read_csv(d / "train.csv") for d in dirs]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.4))
    for ax, key, title in ((axes[0], "loss", "TD loss (smooth-L1)"), (axes[1], "mean_q", "Mean Q of taken action")):
        for i, r in enumerate(rows):
            x, y = col(r, "frames"), col(r, key)
            ok = ~np.isnan(y)
            ax.plot(x[ok], y[ok], color=COLOR["dqn"], lw=1.2, alpha=0.45 + 0.2 * (i == 0),
                    label=f"seed {dirs[i].name[-1]}")
        ax.set_title(title)
        ax.set_xlabel("environment frames")
        frames_k(ax)
    axes[0].set_ylabel("loss")
    axes[1].set_ylabel("Q-value")
    axes[0].legend()
    fig.tight_layout()
    save(fig, "fig3_dqn_objective")


def fig_pg_entropy(best) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for algo in ("reinforce", "ppo", "a2c"):
        rows = [read_csv(d / "train.csv") for d in seed_dirs(algo, best[algo]["name"])]
        g, m, lo, hi = on_grid([col(r, "frames") for r in rows], [col(r, "entropy") for r in rows])
        ax.fill_between(g, lo, hi, color=COLOR[algo], alpha=0.15, lw=0)
        ax.plot(g, m, color=COLOR[algo], label=LABEL[algo])
        ax.text(g[-1], m[-1], f" {LABEL[algo]}", color=INK, fontsize=8, va="center")
    frames_k(ax)
    ax.set_xlabel("environment frames")
    ax.set_ylabel("policy entropy (nats)")
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig, "fig4_pg_entropy")


def convergence_point(frames, episodes, ret) -> tuple[float, float]:
    """First evaluation where the 3-point moving average reaches 90% of the way from the
    initial return to the final plateau (mean of the last 5 evaluations)."""
    ma = np.convolve(ret, np.ones(3) / 3, mode="valid")
    start, plateau = ret[0], ret[-5:].mean()
    target = start + 0.9 * (plateau - start)
    idx = int(np.argmax(ma >= target)) + 2 if (ma >= target).any() else len(ret) - 1
    return float(episodes[idx]), float(frames[idx])


def convergence_table(best) -> dict:
    out = {}
    for algo in ALGOS:
        pts = []
        for d in seed_dirs(algo, best[algo]["name"]):
            ev = read_csv(d / "eval.csv")
            pts.append(convergence_point(col(ev, "frames"), col(ev, "episodes"), col(ev, "return_mean")))
        e = np.asarray([p[0] for p in pts])
        f = np.asarray([p[1] for p in pts])
        out[algo] = {"episodes_mean": float(e.mean()), "episodes_std": float(e.std(ddof=1)) if len(e) > 1 else 0.0,
                     "frames_mean": float(f.mean()), "per_seed_episodes": e.tolist()}
    (RESULTS / "convergence.json").write_text(json.dumps(out, indent=2))
    return out


def fig_convergence(best) -> None:
    conv = convergence_table(best)
    fig, ax = plt.subplots(figsize=(4.6, 2.1))
    y = np.arange(len(ALGOS))
    for i, algo in enumerate(ALGOS):
        c = conv[algo]
        ax.barh(i, c["episodes_mean"], height=0.55, color=COLOR[algo], zorder=2)
        ax.errorbar(c["episodes_mean"], i, xerr=c["episodes_std"], color=INK2, capsize=3, lw=1, zorder=3)
        ax.scatter(c["per_seed_episodes"], [i] * len(c["per_seed_episodes"]), s=14, color=INK, zorder=4)
        ax.text(c["episodes_mean"] + c["episodes_std"], i, f"  {c['episodes_mean']:,.0f}", va="center",
                fontsize=8, color=INK)
    ax.set_yticks(y, [LABEL[a] for a in ALGOS])
    ax.invert_yaxis()
    ax.set_xlabel("training episodes to converge (dots: individual seeds)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig5_convergence")


def fig_generalisation() -> None:
    gen = json.loads((RESULTS / "generalisation.json").read_text())
    scen = list(gen)
    policies = [*ALGOS, "nearest_first"]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    w = 0.16
    x = np.arange(len(scen))
    for i, p in enumerate(policies):
        diff = [gen[s][p]["return_mean"] - gen[s]["edf"]["return_mean"] for s in scen]
        color = COLOR.get(p, "#b4b2ab")
        ax.bar(x + (i - 2) * w, diff, width=w - 0.02, color=color, label=LABEL[p], zorder=2)
    ax.axhline(0, color=INK2, lw=1.2, ls="--")
    ax.text(x[-1] + 0.45, 0, " EDF", va="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [s.replace("_", " ") for s in scen])
    ax.set_ylabel("return minus EDF return")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7, handlelength=1.0, columnspacing=0.8)
    ax.grid(axis="x", visible=False)
    save(fig, "fig6_generalisation")


def fig_test_comparison() -> None:
    t = json.loads((RESULTS / "test_results.json").read_text())["policies"]
    order = [*ALGOS, "edf", "nearest_first", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, key, title in ((axes[0], "return", "Episode return"),
                           (axes[1], "critical_service_rate", "Critical orders delivered on time")):
        for i, p in enumerate(order):
            m, ci = t[p][f"{key}_mean"], t[p][f"{key}_ci95"]
            ax.barh(i, m, height=0.6, color=COLOR.get(p, "#b4b2ab"), zorder=2)
            ax.errorbar(m, i, xerr=ci, color=INK2, capsize=3, lw=1, zorder=3)
            ax.text(max(m, 0) + ci, i, f"  {m:.2f}" if key == "return" else f"  {100 * m:.1f}%",
                    va="center", fontsize=8, color=INK)
        ax.set_title(title)
        ax.axvline(0, color=AXIS, lw=1)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [LABEL[p] for p in order])
    axes[0].invert_yaxis()
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("Held-out test performance (200 seeds; bars: mean, whiskers: 95% CI)",
                 fontsize=9, color=INK2, y=1.02)
    fig.tight_layout()
    save(fig, "fig7_test_comparison")


def fig_sweep_sensitivity(base) -> None:
    tables = json.loads((RESULTS / "config_tables.json").read_text())
    best = best_configs()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for ax, algo in zip(axes.flat, ALGOS):
        rows = tables[algo]
        for i, r in enumerate(rows):
            is_best = r["name"] == best[algo]["name"]
            ax.barh(i, r["return_mean"], height=0.62, color=COLOR[algo],
                    alpha=1.0 if is_best else 0.55, zorder=2)
            ax.errorbar(r["return_mean"], i, xerr=r["return_std"], color=INK2, capsize=2, lw=0.9, zorder=3)
        ax.set_yticks(range(len(rows)), [r["name"].replace("_", " ") for r in rows], fontsize=7.5)
        ax.invert_yaxis()
        for b, v in base.items():
            ax.axvline(v, lw=1.1, **BASE_STYLE[b])
        ax.set_title(LABEL[algo])
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("validation return (50 seeds; ± std over 3 training seeds)")
    fig.suptitle("Hyperparameter sweep: every configuration (dark bar = selected; dashed = EDF, "
                 "dotted = nearest-first)", fontsize=9, color=INK2, y=1.0)
    fig.tight_layout()
    save(fig, "fig8_sweep_sensitivity")


def main() -> None:
    best = best_configs()
    base = baseline_val()
    fig_cumulative_rewards(best, base)
    fig_learning_comparison(best, base)
    fig_dqn_objective(best)
    fig_pg_entropy(best)
    fig_convergence(best)
    if (RESULTS / "generalisation.json").exists():
        fig_generalisation()
    if (RESULTS / "test_results.json").exists():
        fig_test_comparison()
    from evaluation.protocol import VALIDATION_SEEDS, baseline_act, run_episodes, summarize
    full = {b: summarize(run_episodes(lambda s, b=b: baseline_act(b, s), VALIDATION_SEEDS))["return_mean"]
            for b in ("edf", "nearest_first")}
    fig_sweep_sensitivity(full)


if __name__ == "__main__":
    main()
