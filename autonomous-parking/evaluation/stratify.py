"""Stratify the corrected evaluation by how hard the start is (audit finding: the scenario generator draws the
start distance uniformly in [0.05, 1] x 6 m, so many episodes start with the car already beside / partly inside
the bay mouth, within centimetres of a neighbour).

    python -m evaluation.stratify --dir experiments/corrected_evaluation
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from evaluation.summarize_corrected import label, md

SEEDS = {"set_A": 3000, "set_B": 5000, "set_C": 9000}
BINS = [(0.0, 2.5, "near (<2.5 m)"), (2.5, 4.5, "mid (2.5-4.5 m)"), (4.5, 99.0, "far (>=4.5 m)")]


def start_features():
    env = ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION, "scenario_kinds": [Kind.PERPENDICULAR]})
    rows = []
    for blk, s0 in SEEDS.items():
        n = len(pd.read_csv(next((Path("experiments/corrected_evaluation") / blk).glob("*.csv"))))
        for i in range(n):
            env.reset(seed=s0 + i)
            sp, tp = env.scenario.start_pose, env.scenario.slot_pose
            rows.append({"set": blk, "episode": i, "start_dist": float(np.hypot(*(sp[:2] - tp[:2]))),
                         "start_clearance": env.body_clearance(),
                         "start_overlaps_bay_rows": bool(abs(sp[1] - tp[1]) < 2.1 + 2.1)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/corrected_evaluation")
    a = ap.parse_args()
    root = Path(a.dir)
    feat = start_features()
    feat.to_csv(root / "start_features.csv", index=False)
    lines = ["# Results stratified by start distance (pooled seed blocks A+B+C, n=500)\n",
             f"Start-distance bins: " + ", ".join(f"{b[2]}: n={int(((feat.start_dist >= b[0]) & (feat.start_dist < b[1])).sum())}" for b in BINS),
             f"\nStarts with body clearance < 0.30 m to a neighbour: {(feat.start_clearance < 0.30).mean()*100:.1f} %; < 0.05 m: {(feat.start_clearance < 0.05).mean()*100:.1f} %\n"]
    ctrls = sorted(f.stem for f in (root / "set_A").glob("*.csv"))
    order = ["fuzzy_only", "fuzzy_precise", "ppo_only", "ppo_v2", "ppo_fuzzy_bolted", "ppo_v2_fuzzy"]
    ctrls = sorted(ctrls, key=lambda c: (order.index(c[:-4] if c.endswith("_env") else c), c.endswith("_env")))
    rows = []
    for c in ctrls:
        d = pd.concat([pd.read_csv(root / s / f"{c}.csv").assign(set=s) for s in SEEDS], ignore_index=True)
        d = d.merge(feat, on=["set", "episode"])
        for lo, hi, nm in BINS + [(0, 99, "ALL")]:
            x = d[(d.start_dist >= lo) & (d.start_dist < hi)]
            n = len(x)
            rows.append({"controller": label(c), "start": nm, "n": n,
                         "pose_success_%": round(100 * (x.legacy_outcome == "success").mean(), 1),
                         "pose_collision_%": round(100 * (x.legacy_outcome == "collision").mean(), 1),
                         "pose_timeout_%": round(100 * (x.legacy_outcome == "timeout").mean(), 1),
                         "contained_success_%": round(100 * x.success.mean(), 1),
                         "contained_collision_%": round(100 * x.collision.mean(), 1)})
    tab = pd.DataFrame(rows)
    tab.to_csv(root / "stratified.csv", index=False)
    for nm in [b[2] for b in BINS] + ["ALL"]:
        lines.append(f"\n### {nm}\n\n" + md(tab[tab.start == nm].drop(columns=["start"])))
    (root / "stratified.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
