"""Failure taxonomy for held-out RL-audit runs (pooled A+B+C): every episode falls in exactly one class.

 strict_success      whole car inside the bay, heading and speed within tolerance (the headline metric)
 collision           hit an obstacle / left the map (episode ends)
 pose_only_timeout   centre-pose box reached (legacy criterion) but the car never ended up fully inside the lines
 never_reached       timed out without ever satisfying the pose box (stalled / dithering far or near the bay)
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
names = sys.argv[2:]
rows = []
for n in names:
    d = pd.concat([pd.read_csv(root / s / f"ppo-models__{n}.csv") for s in ("set_A", "set_B", "set_C")], ignore_index=True)
    succ = d.success.astype(bool)
    col = d.collision.astype(bool)
    pose = d.legacy_outcome == "success"
    cls = pd.Series("never_reached", index=d.index)
    cls[pose & ~succ & ~col] = "pose_only_timeout"
    cls[col] = "collision"
    cls[succ] = "strict_success"
    vc = (cls.value_counts() / len(d) * 100).round(1)
    t = d[(cls == "never_reached")]
    rows.append({"ckpt": n, **{k: vc.get(k, 0.0) for k in ["strict_success", "collision", "pose_only_timeout", "never_reached"]},
                 "never_reached_median_pos_err_m": round(float(t.pos_error_m.median()), 2) if len(t) else None,
                 "pose_only_median_protrusion_m": round(float(d.loc[cls == "pose_only_timeout", "bay_protrusion_m"].median()), 3) if (cls == "pose_only_timeout").any() else None})
print(pd.DataFrame(rows).to_string(index=False))
