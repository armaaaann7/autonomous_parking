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
    lo, hi = wilson_interval(int(succ.sum()), len(df))
    out["success_ci_low"], out["success_ci_high"] = lo, hi
    if "in_bay" in df.columns:
        inbay = df["in_bay"].astype(bool)
        out["strict_in_bay_rate"] = float((succ & inbay).mean())      # success AND car fully inside the bay lines
        out["in_bay_given_success"] = float(inbay[succ].mean()) if succ.any() else float("nan")
    if "bay_protrusion_m" in df.columns:
        out["bay_protrusion_median_m"] = float(df.loc[succ, "bay_protrusion_m"].median()) if succ.any() else float("nan")
    out["completion_rate"] = float(1.0 - df["collision"].astype(bool).mean() - df["timeout"].astype(bool).mean())
    if "success_mode" in df.columns:
        out["success_mode"] = str(df["success_mode"].iloc[0])
    if "long_error_m" in df.columns:
        # final signed offsets from the slot centre in the slot frame (all episodes that did not collide)
        ok = ~df["collision"].astype(bool)
        out["final_long_error_abs_m"] = float(df.loc[ok, "long_error_m"].abs().mean()) if ok.any() else float("nan")
        out["final_lat_error_abs_m"] = float(df.loc[ok, "lat_error_m"].abs().mean()) if ok.any() else float("nan")
    if "min_clearance_m" in df.columns:
        out["min_clearance_m_worst"] = float(df["min_clearance_m"].min())
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


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n, as fractions in [0, 1]."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test for PAIRED binary outcomes (same seeds for both controllers).

    Returns (n_a_only, n_b_only, two-sided p). Only discordant episodes carry information; under
    H0 each discordant pair is a fair coin flip. Unlike the unpaired two-proportion z-test this
    respects that every controller faced the same scenario for each seed.
    """
    from scipy.stats import binomtest

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired outcomes must have equal length")
    n_a = int((a & ~b).sum())
    n_b = int((~a & b).sum())
    if n_a + n_b == 0:
        return n_a, n_b, 1.0
    return n_a, n_b, float(binomtest(n_a, n_a + n_b, 0.5).pvalue)
