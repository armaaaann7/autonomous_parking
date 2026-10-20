"""Reads the CSVs from run_experiments.py, builds the Part 19 comparison table, and produces the
bar charts / box plots for the report and demo (blueprint Part 11 / Part 26 step 6).

Usage:
    python -m evaluation.compare_controllers --dir experiments/results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import mcnemar_exact, summarize

_HERE = Path(__file__).resolve().parent.parent
LABELS = {"fuzzy_only": "Fuzzy-only", "ppo_only": "RL-only (PPO)",
          "ppo_fuzzy_bolted": "RL + Fuzzy",
          "fuzzy_only_env": "Fuzzy-only + envelope shield", "ppo_only_env": "RL-only + envelope shield",
          "ppo_fuzzy_bolted_env": "RL + Fuzzy + envelope shield"}
ORDER = list(LABELS)


def build_table(result_dir: Path) -> pd.DataFrame:
    rows = {}
    frames = {}
    for key in ORDER:
        path = result_dir / f"{key}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        frames[key] = df
        rows[LABELS[key]] = summarize(df)
    table = pd.DataFrame(rows).T
    table.index.name = "Model"
    return table, frames


def plot_bars(table: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    err = [[(r.success_rate - r.success_ci_low) for r in table.itertuples()],
           [(r.success_ci_high - r.success_rate) for r in table.itertuples()]]
    table["success_rate"].plot(kind="bar", ax=axes[0], color="#4CAF50", yerr=err, capsize=4,
                               title="Success rate (95% Wilson CI)")
    axes[0].set_ylim(0, 1)
    table["collision_rate"].plot(kind="bar", ax=axes[1], color="#E53935", title="Collision rate")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_boxes(frames: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # SUCCESSFUL episodes only, matching the table (failed episodes would mix in 1-3 m timeout errors).
    ok = {k: frames[k][frames[k]["success"].astype(bool)] for k in ORDER if k in frames}
    data_pos = [ok[k]["pos_error_m"] for k in ok]
    data_head = [ok[k]["heading_error_deg"] for k in ok]
    labels = [f"{LABELS[k]}\n(n={len(ok[k])})" for k in ok]
    axes[0].boxplot(data_pos, tick_labels=labels)
    axes[0].set_title("Final position error (m), successful episodes only")
    axes[1].boxplot(data_head, tick_labels=labels)
    axes[1].set_title("Final heading error (deg), successful episodes only")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectories(result_dir: Path, out_path: Path):
    """Overlay trajectory-length distributions (blueprint Part 11's "trajectory overlay plots";
    per-step (x,y) traces aren't logged by run_experiments.py, so this overlays the length
    distribution instead -- swap in real traces here if you add per-step logging)."""
    _, frames = build_table(result_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ORDER:
        if key in frames:
            ax.hist(frames[key]["trajectory_length_m"], bins=20, alpha=0.5, label=LABELS[key])
    ax.set_xlabel("Trajectory length (m)")
    ax.set_ylabel("Episodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_paired_tests(frames: dict, out_path: Path):
    """Exact McNemar tests on paired success and collision outcomes (same seed => same scenario)."""
    keys = [k for k in ORDER if k in frames]
    lines = ["Paired exact McNemar tests (discordant episodes only; same seed = same scenario).",
             "columns: A, B, outcome, A-only, B-only, p"]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if len(frames[a]) != len(frames[b]) or not (frames[a]["episode"].values == frames[b]["episode"].values).all():
                continue
            for outcome in ("success", "collision"):
                na, nb, p = mcnemar_exact(frames[a][outcome].astype(bool).values, frames[b][outcome].astype(bool).values)
                lines.append(f"{LABELS[a]} | {LABELS[b]} | {outcome} | {na} | {nb} | p={p:.4g}")
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(result_dir: str):
    result_dir = Path(result_dir) if Path(result_dir).is_absolute() else _HERE / result_dir
    table, frames = build_table(result_dir)
    if table.empty:
        print(f"No result CSVs found in {result_dir}. Run evaluation.run_experiments first.")
        return
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    cols = ["n_episodes", "success_rate", "success_ci_low", "success_ci_high", "collision_rate", "timeout_rate",
            "strict_in_bay_rate", "avg_time_s", "pos_error_m", "heading_error_deg", "smoothness", "mean_reward"]
    print(table[[c for c in cols if c in table.columns]])
    table.to_csv(result_dir / "comparison_table.csv")

    plot_bars(table, result_dir / "bars_success_collision.png")
    plot_boxes(frames, result_dir / "boxplots_errors.png")
    plot_trajectories(result_dir, result_dir / "trajectory_length_hist.png")
    print(f"\nSaved comparison_table.csv and 3 plots to {result_dir}")
    write_paired_tests(frames, result_dir / "paired_tests.txt")

    print("Compare success, collisions and timeouts together. A single run does not establish statistical significance.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="experiments/results")
    args = p.parse_args()
    main(args.dir)
