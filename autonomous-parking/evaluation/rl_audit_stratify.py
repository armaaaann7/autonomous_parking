"""Outcome by start distance for RL-audit arms (pooled over training seeds).

    python -m evaluation.rl_audit_stratify --dir experiments/rl_audit/tuning_eval --set T --seed0 7100
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from evaluation.rl_audit_summary import split_name

BINS = [(0.0, 2.5, "near<2.5"), (2.5, 4.5, "mid2.5-4.5"), (4.5, 99.0, "far>=4.5")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--set", default="T"); ap.add_argument("--seed0", type=int, required=True)
    a = ap.parse_args()
    files = sorted((Path(a.dir) / f"set_{a.set}").glob("*.csv"))
    n = len(pd.read_csv(files[0]))
    env = ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION, "scenario_kinds": [Kind.PERPENDICULAR]})
    dist = []
    for i in range(n):
        env.reset(seed=a.seed0 + i)
        dist.append(float(np.hypot(*(env.scenario.start_pose[:2] - env.scenario.slot_pose[:2]))))
    dist = np.array(dist)
    rows = []
    for f in files:
        arm, seed = split_name(f.stem)
        d = pd.read_csv(f)
        for lo, hi, nm in BINS:
            m = (dist >= lo) & (dist < hi)
            g = d[m]
            rows.append({"arm": arm, "bin": nm, "n": len(g), "succ": g.success.astype(bool).sum(), "coll": g.collision.astype(bool).sum(),
                         "timeout": g.timeout.astype(bool).sum()})
    t = pd.DataFrame(rows).groupby(["arm", "bin"], as_index=False).sum()
    for c in ["succ", "coll", "timeout"]:
        t[c + "_%"] = (100 * t[c] / t.n).round(1)
    print(t.drop(columns=["succ", "coll", "timeout"]).to_string(index=False))


if __name__ == "__main__":
    main()
