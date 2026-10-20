"""Metric definitions (blueprint Part 18). Kept as pure functions so both the online evaluation
loop (evaluation/run_experiments.py) and any offline analysis can share them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "success_rate", "collision_rate", "timeout_rate",
    "avg_time_s", "pos_error_m", "heading_error_deg",
    "trajectory_length_m", "smoothness", "mean_abs_accel", "mean_brake",
    "mean_reward",
]


def summarize(df: pd.DataFrame) -> dict:
    """One controller's episode-level CSV -> the aggregate row used in Part 19's table."""
    succ = df["success"].astype(bool)
    out = {
        "n_episodes": len(df),
        "success_rate": succ.mean(),
        "collision_rate": df["collision"].astype(bool).mean(),
        "timeout_rate": df["timeout"].astype(bool).mean(),
        # timing/error stats are reported over SUCCESSFUL episodes only, per blueprint Part 18
        "avg_time_s": df.loc[succ, "time_s"].mean() if succ.any() else float("nan"),
        "pos_error_m": df.loc[succ, "pos_error_m"].mean() if succ.any() else float("nan"),
        "pos_error_m_std": df.loc[succ, "pos_error_m"].std() if succ.any() else float("nan"),
        "heading_error_deg": df.loc[succ, "heading_error_deg"].mean() if succ.any() else float("nan"),
        "trajectory_length_m": df["trajectory_length_m"].mean(),
        "trajectory_length_m_std": df["trajectory_length_m"].std(),
        "smoothness": df["smoothness"].mean(),
        "mean_abs_accel": df["mean_abs_accel"].mean(),
        "mean_brake": df["mean_brake"].mean(),
        "mean_reward": df["reward"].mean(),
        "mean_reward_std": df["reward"].std(),
    }
    if "shield_active_frac" in df.columns:
        out["shield_active_frac"] = df["shield_active_frac"].mean()
    return out


def two_proportion_z_test(success_a: np.ndarray, success_b: np.ndarray) -> tuple[float, float]:
    """Two-proportion z-test (blueprint Part 18's "optional polish"), e.g. comparing RL-only vs
    RL+Fuzzy success or collision rates. Returns (z, two-sided p-value); uses a normal
    approximation via ``scipy.stats.norm`` so we don't need statsmodels as a dependency."""
    from scipy.stats import norm

    n1, n2 = len(success_a), len(success_b)
    p1, p2 = np.mean(success_a), np.mean(success_b)
    p_pool = (np.sum(success_a) + np.sum(success_b)) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p_value)
