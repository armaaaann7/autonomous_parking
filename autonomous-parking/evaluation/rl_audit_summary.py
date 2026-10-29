"""Per-arm / per-seed summary of RL-audit evaluations (experiments/rl_audit/*), strict + pose metrics.

    python -m evaluation.rl_audit_summary --dir experiments/rl_audit/tuning_eval --set T [--out summary.csv]

Arm name = checkpoint name without the trailing _s<seed>[_best]. Pooled rows use all seeds of an arm;
the 'seed_sd' columns are the standard deviation of the per-training-seed rates (variability across
training seeds, n = number of seeds, so read it as indicative only).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import wilson_interval


def split_name(stem: str):
    base = stem.replace("ppo-models__", "").replace("_best", "")
    env = base.endswith("_env")
    if env:
        base = base[:-4]
    sfx = "+shield" if env else ""
    m = re.match(r"(.*?)_s(\d+)$", base)
    if base.startswith("exp_ppo_cur") and m:
        return "BASE(exp_ppo_cur)" + sfx, int(m.group(2))
    if base == "ppo_only":
        return "ORIG(ppo_only_100k)" + sfx, 0
    if m:
        return m.group(1).replace("rl_audit_", "") + sfx, int(m.group(2))
    return base + sfx, 0


def stats(d: pd.DataFrame) -> dict:
    n = len(d)
    ps = (d.legacy_outcome == "success").mean()
    pc = (d.legacy_outcome == "collision").mean()
    ss = d.success.astype(bool).mean()
    sc = d.collision.astype(bool).mean()
    lo, hi = wilson_interval(int(d.success.sum()), n)
    ok = ~d.collision.astype(bool)
    return {"n": n, "pose_succ": ps, "pose_coll": pc, "pose_timeout": (d.legacy_outcome == "timeout").mean(),
            "strict_succ": ss, "strict_ci": f"[{lo*100:.0f}-{hi*100:.0f}]", "strict_coll": sc, "strict_timeout": d.timeout.astype(bool).mean(),
            "steps_succ": d.loc[d.success.astype(bool), "steps"].mean(), "traj_m_succ": d.loc[d.success.astype(bool), "trajectory_length_m"].mean(),
            "pos_err_m": d.loc[ok, "pos_error_m"].mean(), "head_err_deg": d.loc[ok, "heading_error_deg"].mean(),
            "min_clear_m": d.min_clearance_m.min(), "min_clear_p5": d.min_clearance_m.quantile(.05),
            "shield_rate": d.shield_active_frac.mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--set", default="T")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, per_seed = [], []
    for f in sorted((Path(a.dir) / f"set_{a.set}").glob("*.csv")):
        arm, seed = split_name(f.stem)
        d = pd.read_csv(f)
        per_seed.append({"arm": arm, "seed": seed, "file": f.stem, **stats(d)})
    ps = pd.DataFrame(per_seed)
    for arm, g in ps.groupby("arm"):
        agg = {"arm": arm, "seeds": len(g)}
        for c in ["pose_succ", "pose_coll", "strict_succ", "strict_coll", "strict_timeout", "steps_succ", "traj_m_succ", "pos_err_m", "head_err_deg", "shield_rate"]:
            agg[c] = g[c].mean()
        agg["strict_seed_sd"] = g.strict_succ.std(ddof=1) if len(g) > 1 else float("nan")
        agg["strict_min"] = g.strict_succ.min(); agg["strict_max"] = g.strict_succ.max()
        agg["min_clear_m"] = g.min_clear_m.min()
        rows.append(agg)
    tab = pd.DataFrame(rows).sort_values("strict_succ", ascending=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(tab.round(3).to_string(index=False))
    print("\nper seed:"); print(ps[["arm", "seed", "pose_succ", "pose_coll", "strict_succ", "strict_ci", "strict_coll", "strict_timeout"]].round(3).to_string(index=False))
    if a.out:
        tab.to_csv(a.out, index=False); ps.to_csv(a.out.replace(".csv", "_per_seed.csv"), index=False)


if __name__ == "__main__":
    main()
