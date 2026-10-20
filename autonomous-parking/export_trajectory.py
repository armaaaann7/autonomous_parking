from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from evaluation.run_experiments import make_adapter


def build_controller(name: str, model: str):
    """Kept for backwards compatibility; delegates to the shared factory."""
    return make_adapter(name, model)


def run_episode(adapter, seed: int, kind, controller: str = "", reverse: bool = False,
                success_mode: str = "contained", tail: int = 40):
    env = ParkingEnv({"scenario_kinds": [kind], "randomization": RandomizationLevel.SLOT_POSITION,
                      "allow_reverse": reverse, "min_start_distance_scale": 1.0 if reverse else 0.05,
                      "success_mode": success_mode})
    adapter.reset()
    obs, _ = env.reset(seed=seed)
    v = env.vehicle
    frames = [{"x": v.x, "y": v.y, "theta": v.theta, "delta": v.delta, "v": v.v,
               "action": [0.0, 0.0], "shield": False, "collision": False,
               "clearance": float(env.body_clearance()), "protrusion": float(env.bay_protrusion())}]
    info = {}
    done = False
    pose_streak, pose_step, steps = 0, None, 0
    while not done:
        action, shield = adapter.act(obs, env)
        obs, _, term, trunc, info = env.step(action)
        v = env.vehicle
        frames.append({"x": float(v.x), "y": float(v.y), "theta": float(v.theta), "delta": float(v.delta),
                       "v": float(v.v), "action": [float(action[0]), float(action[1])],
                       "shield": bool(shield), "mode": getattr(adapter, "last_mode", "Fuzzy navigation" if controller == "fuzzy_only" else "RL driving"),
                       "collision": bool(info["collision"]), "clearance": float(env.body_clearance()),
                       "protrusion": float(info["bay_protrusion"])})
        steps += 1
        pose_streak = pose_streak + 1 if info["pose_ok"] else 0
        if pose_step is None and not info["collision"] and pose_streak >= env.hold_steps:
            pose_step = steps
        # strict mode keeps driving after the pose criterion is met; show 'tail' more steps, then stop recording
        done = term or trunc or (pose_step is not None and steps >= pose_step + tail)
    sc = env.scenario
    data = {
        "controller": controller, "reverse": reverse, "source": "recorded simulation episode (not live inference)", "kind": sc.kind.value, "dt": env.dt, "seed": seed,
        "car": {"length": env.vehicle.length, "width": env.vehicle.width},
        "slot": [float(x) for x in sc.slot_pose],
        "obstacles": [[float(c) for c in o] for o in np.asarray(sc.obstacles)],
        "bounds": [float(b) for b in sc.bounds],
        "frames": frames,
        "result": {k: (float(info[k]) if isinstance(info[k], (float, np.floating)) else bool(info[k]))
                   for k in ("success", "collision", "timeout", "pos_error", "heading_error_deg")},
        "in_bay": bool(env.inside_bay()),
        "success_mode": success_mode, "pose_success_step": pose_step, "recorded_steps": steps,
        "outcome": ("collision" if info["collision"] else "parked inside the bay" if info["success"] else
                    "stopped within pose tolerance but NOT inside the bay" if pose_step is not None else
                    "timeout / not parked"),
    }
    env.close()
    return data


CONTROLLERS = ('fuzzy_only', 'fuzzy_precise', 'ppo_only', 'ppo_v2', 'ppo_fuzzy_bolted',
               'fuzzy_only+env', 'fuzzy_precise+env', 'ppo_only+env', 'ppo_v2+env', 'ppo_fuzzy_bolted+env')

def main():
    p = argparse.ArgumentParser(description='Export one fixed-seed perpendicular episode; never search for a favourable result.')
    p.add_argument('--controller', choices=CONTROLLERS, default='ppo_only')
    p.add_argument('--model', default='models/ppo_only_best')
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    try:
        adapter = make_adapter(a.controller, a.model)
    except FileNotFoundError as err:
        p.error(str(err))
    d = run_episode(adapter, a.seed, Kind.PERPENDICULAR, a.controller)
    out = Path(a.out or f'../viewer/assets/data/{a.controller.replace("+", "_")}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d))
    print(f'Exported {a.controller}, seed {a.seed}: {d["result"]} -> {out.resolve()}')

if __name__ == '__main__':
    main()
