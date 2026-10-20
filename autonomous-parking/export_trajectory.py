"""Export one real episode (pose + action + collision per step, plus scene geometry) to JSON for
the three.js viewer.

    python export_trajectory.py --controller fuzzy_only --out ../viewer/assets/data/trajectory.json
    python export_trajectory.py --controller ppo_shielded --model models/ppo_shielded --seed 1003
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from evaluation.run_experiments import (FuzzyOnlyAdapter, ParallelGeometricAdapter, PPOOnlyAdapter,
                                         PPOPlusShieldAdapter, RuleBasedAdapter)


def build_controller(name: str, model: str):
    return {
        "rule_based": lambda: RuleBasedAdapter(),
        "fuzzy_only": lambda: FuzzyOnlyAdapter(),
        "ppo_only": lambda: PPOOnlyAdapter(model),
        "ppo_fuzzy_bolted": lambda: PPOPlusShieldAdapter(model, "ppo_fuzzy_bolted"),
        "ppo_shielded": lambda: PPOPlusShieldAdapter(model, "ppo_shielded"),
        "parallel_geometric": lambda: ParallelGeometricAdapter(),
    }[name]()


def run_episode(adapter, seed: int, kind, controller: str = "", reverse: bool = False):
    env = ParkingEnv({"scenario_kinds": [kind], "randomization": RandomizationLevel.SLOT_POSITION,
                      "allow_reverse": reverse, "min_start_distance_scale": 1.0 if reverse else 0.05})
    adapter.reset()
    obs, _ = env.reset(seed=seed)
    v = env.vehicle
    frames = [{"x": v.x, "y": v.y, "theta": v.theta, "delta": v.delta, "v": v.v,
               "action": [0.0, 0.0], "shield": False, "collision": False}]
    info = {}
    done = False
    while not done:
        action, shield = adapter.act(obs, env)
        obs, _, term, trunc, info = env.step(action)
        v = env.vehicle
        frames.append({"x": float(v.x), "y": float(v.y), "theta": float(v.theta), "delta": float(v.delta),
                       "v": float(v.v), "action": [float(action[0]), float(action[1])],
                       "shield": bool(shield), "collision": bool(info["collision"])})
        done = term or trunc
    sc = env.scenario
    data = {
        "controller": controller, "reverse": reverse, "kind": sc.kind.value, "dt": env.dt, "seed": seed,
        "car": {"length": env.vehicle.length, "width": env.vehicle.width},
        "slot": [float(x) for x in sc.slot_pose],
        "obstacles": [[float(c) for c in o] for o in np.asarray(sc.obstacles)],
        "bounds": [float(b) for b in sc.bounds],
        "frames": frames,
        "result": {k: (float(info[k]) if isinstance(info[k], (float, np.floating)) else bool(info[k]))
                   for k in ("success", "collision", "timeout", "pos_error", "heading_error_deg")},
    }
    env.close()
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default="fuzzy_only")
    p.add_argument("--model", default="models/ppo_shielded")
    p.add_argument("--kind", choices=["parallel", "perpendicular"], default="perpendicular")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reverse", action="store_true", help="enable reverse gear (automatic for parallel_geometric)")
    p.add_argument("--min-frames", type=int, default=40, help="skip trivially short successes")
    p.add_argument("--max-attempts", type=int, default=150)
    p.add_argument("--out", default="../viewer/assets/data/trajectory.json")
    a = p.parse_args()
    kind = Kind.PARALLEL if a.kind == "parallel" else Kind.PERPENDICULAR
    adapter = build_controller(a.controller, a.model)
    reverse = a.reverse or a.controller == "parallel_geometric"
    if a.controller == "parallel_geometric":
        kind = Kind.PARALLEL
    seeds = [a.seed] if a.seed is not None else range(1000, 1000 + a.max_attempts)
    best = None
    for s in seeds:
        d = run_episode(adapter, s, kind, a.controller, reverse)
        if best is None:
            best = d
        if a.seed is not None or (d["result"]["success"] and len(d["frames"]) >= a.min_frames):
            best = d
            break
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best))
    print(f"wrote {len(best['frames'])} frames ({a.controller}, seed {best['seed']}, {best['kind']}) -> {out.resolve()}")
    print("result:", best["result"])


if __name__ == "__main__":
    main()
