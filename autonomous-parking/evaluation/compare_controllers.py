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

from .metrics import summarize, two_proportion_z_test

_HERE = Path(__file__).resolve().parent.parent
LABELS = {"rule_based": "Rule-Based", "fuzzy_only": "Fuzzy-only", "ppo_only": "PPO",
          "ppo_fuzzy_bolted": "PPO + shield (bolted on)", "ppo_shielded": "PPO (trained through shield)",
          "parallel_geometric": "Geometric parallel (reverse)"}
ORDER = ["rule_based", "fuzzy_only", "ppo_only", "ppo_fuzzy_bolted", "ppo_shielded", "parallel_geometric"]


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
    table["success_rate"].plot(kind="bar", ax=axes[0], color="#4CAF50", title="Success rate")
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
    data_pos = [frames[k]["pos_error_m"].dropna() for k in ORDER if k in frames]
    data_head = [frames[k]["heading_error_deg"].dropna() for k in ORDER if k in frames]
    labels = [LABELS[k] for k in ORDER if k in frames]
    axes[0].boxplot(data_pos, tick_labels=labels)
    axes[0].set_title("Final position error (m)")
    axes[1].boxplot(data_head, tick_labels=labels)
    axes[1].set_title("Final heading error (deg)")
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


def main(result_dir: str):
    result_dir = _HERE / result_dir
    table, frames = build_table(result_dir)
    if table.empty:
        print(f"No result CSVs found in {result_dir}. Run evaluation.run_experiments first.")
        return
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(table[["n_episodes", "success_rate", "collision_rate", "timeout_rate", "avg_time_s",
                 "pos_error_m", "heading_error_deg", "smoothness", "mean_reward"]])
    table.to_csv(result_dir / "comparison_table.csv")

    plot_bars(table, result_dir / "bars_success_collision.png")
    plot_boxes(frames, result_dir / "boxplots_errors.png")
    plot_trajectories(result_dir, result_dir / "trajectory_length_hist.png")
    print(f"\nSaved comparison_table.csv and 3 plots to {result_dir}")

    print("\nCore ablation (does the shield help, and does training through it matter?):")
    for a_, b_ in [("ppo_only", "ppo_fuzzy_bolted"), ("ppo_only", "ppo_shielded"),
                   ("ppo_fuzzy_bolted", "ppo_shielded")]:
        if a_ in frames and b_ in frames:
            for metric in ("success", "collision"):
                z, p = two_proportion_z_test(frames[a_][metric].values, frames[b_][metric].values)
                print(f"  {LABELS[a_]} vs {LABELS[b_]} [{metric}]: z={z:.2f}, p={p:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="experiments/results")
    args = p.parse_args()
    main(args.dir)
