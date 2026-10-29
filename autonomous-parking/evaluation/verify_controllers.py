"""Presentation smoke check: every controller loads, steps the real env, emits finite in-bounds actions, and
reaches a valid terminal state. Same seeds for every controller; strict (whole-car-in-bay) success.

    python -m evaluation.verify_controllers --seeds 20000:20 --out experiments/presentation_smoke
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from evaluation.run_experiments import make_adapter

CTRLS = ["ppo_v2", "ppo_v3", "fuzzy_only", "fuzzy_precise", "ppo_v2_fuzzy", "ppo_fuzzy_bolted",
         "ppo_v2+env", "ppo_v3+env", "fuzzy_precise+env", "ppo_v2_fuzzy+env"]


def check(name, seeds):
    ad = make_adapter(name, "models/ppo_only_best")
    env = ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION, "scenario_kinds": [Kind.PERPENDICULAR],
                      "success_mode": "contained", **getattr(ad, "env_extra", {})})
    rows, bad = [], 0
    for sd in seeds:
        ad.reset()
        obs, _ = env.reset(seed=sd)
        assert obs.shape == env.observation_space.shape
        steps = shield = 0
        while True:
            a, flag = ad.act(obs, env)
            a = np.asarray(a, dtype=float)
            if a.shape != (2,) or not np.isfinite(a).all() or np.abs(a).max() > 1.0 + 1e-6:
                bad += 1
            obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
            steps += 1; shield += int(bool(flag))
            if term or trunc:
                break
        rows.append({"controller": name, "seed": sd, "success": bool(info["success"]), "collision": bool(info["collision"]),
                     "timeout": bool(info["timeout"]), "steps": steps, "pos_error_m": info["pos_error"],
                     "heading_error_deg": float(info["heading_error_deg"]), "shield_steps": shield})
    return rows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="20000:20"); ap.add_argument("--out", default="experiments/presentation_smoke")
    ap.add_argument("--controllers", default=",".join(CTRLS))
    a = ap.parse_args()
    s0, n = (int(x) for x in a.seeds.split(":"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allrows, report = [], {}
    for c in a.controllers.split(","):
        try:
            rows, bad = check(c, list(range(s0, s0 + n)))
            allrows += rows
            d = pd.DataFrame(rows)
            report[c] = {"loads": True, "runs": True, "invalid_actions": bad, "episodes": n, "strict_success": int(d.success.sum()),
                         "collisions": int(d.collision.sum()), "timeouts": int(d.timeout.sum()),
                         "shield_step_frac": round(d.shield_steps.sum() / d.steps.sum(), 3)}
        except Exception as e:  # isolate a failing controller; never let it hide the others
            report[c] = {"loads": False, "error": repr(e)[:200]}
        print(c, report[c], flush=True)
    pd.DataFrame(allrows).to_csv(out / "episodes.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
