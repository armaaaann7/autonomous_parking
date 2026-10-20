"""Independent stress verification of the braking-envelope shield (audit P4).

Four experiments, all under the simulator's own dynamics (so they test the shield's logic and the
simulator-level invariant, NOT real-world safety):
  1. random_states   - random start states near the neighbours (all headings, speeds 0..2 m/s, corners close to
                       the obstacles) driven by an ADVERSARIAL policy (full throttle, random steering) for 150 steps.
  2. substep_audit   - shielded episodes of every controller re-checked with 10x finer interpolated poses between
                       steps (the environment, like the shield, only tests collisions at step instants).
  3. brake_mismatch  - the REAL vehicle brakes weaker than the shield's model (model mismatch): the guarantee is
                       expected to break; this quantifies how.
  4. timeout_anatomy - where do shielded episodes that time out end up (stationary? close to a neighbour? far?).

    python -m evaluation.shield_stress --out experiments/shield_verification
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from environment.geometry import obb_aabb_overlap, obb_corners
from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from evaluation.run_experiments import make_adapter
from integration.envelope_shield import BrakingEnvelopeShield

_HERE = Path(__file__).resolve().parent.parent


def new_env(**kw):
    return ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION, "scenario_kinds": [Kind.PERPENDICULAR], **kw})


def random_states(n=400, steps=150, seed=0, brake_factor=1.0):
    rng = np.random.default_rng(seed)
    env = new_env()
    env.vehicle.max_brake = env.vehicle.max_brake * brake_factor     # REAL brakes; the shield still models the nominal ones
    shield = BrakingEnvelopeShield()
    res = {"episodes": 0, "skipped_unsafe_start": 0, "collisions": 0, "interventions": 0, "steps": 0,
           "min_clearance_all": 9.0, "episodes_start_clearance_ge_margin": 0, "min_clearance_when_start_ge_margin": 9.0,
           "episodes_start_clearance_lt_margin": 0, "min_clearance_when_start_lt_margin": 9.0}
    for ep in range(n):
        env.reset(seed=10_000 + ep)
        sx, sy, st = env.scenario.slot_pose
        # random pose in a box around the bay mouth / neighbours, any heading, any forward speed
        for _ in range(50):
            x, y = sx + rng.uniform(-5, 5), sy + rng.uniform(-6, 3)
            env.vehicle.reset(x, y, rng.uniform(-np.pi, np.pi))
            env.vehicle.v = float(rng.choice([0.0, 0.5, 1.0, 2.0]))
            if env.body_clearance() > 0.0 and not env._check_collision():
                break
        else:
            continue
        # skip states from which even hold-steer+brake cannot avoid a collision (shield has no safe fallback there)
        if not env.braking_envelope_safe(np.array([0.0, -1.0]), margin=0.0)[0]:
            res["skipped_unsafe_start"] += 1
            continue
        shield.reset()
        res["episodes"] += 1
        start_ok = env.body_clearance() >= shield.margin
        key = "ge" if start_ok else "lt"
        res[f"episodes_start_clearance_{key}_margin"] += 1
        ep_min = 9.0
        for _ in range(steps):
            a = np.array([rng.uniform(-1, 1), 1.0], dtype=np.float32)       # adversary: floor it, steer randomly
            safe_a = shield.filter(env, a)
            _, _, term, trunc, info = env.step(safe_a)
            res["steps"] += 1
            ep_min = min(ep_min, env.body_clearance())
            if info["collision"]:
                res["collisions"] += 1
                break
        res["min_clearance_all"] = min(res["min_clearance_all"], ep_min)
        res[f"min_clearance_when_start_{key}_margin"] = min(res[f"min_clearance_when_start_{key}_margin"], ep_min)
        res["interventions"] += shield.n_interventions
    return res


def substep_audit(n=60, seed=9000, k=10):
    env = new_env()
    out = {}
    for name in ["fuzzy_only+env", "fuzzy_precise+env", "ppo_only+env", "ppo_v2+env", "ppo_fuzzy_bolted+env", "ppo_v2_fuzzy+env"]:
        ad = make_adapter(name, "models/ppo_only_best")
        step_coll = sub_coll = 0
        for s in range(seed, seed + n):
            ad.reset()
            obs, _ = env.reset(seed=s)
            done = False
            prev = (env.vehicle.x, env.vehicle.y, env.vehicle.theta)
            hit_sub = False
            while not done:
                a, _ = ad.act(obs, env)
                obs, _, te, tr, info = env.step(a)
                cur = (env.vehicle.x, env.vehicle.y, env.vehicle.theta)
                for f in np.linspace(0, 1, k + 1)[1:-1]:
                    x = prev[0] + f * (cur[0] - prev[0])
                    y = prev[1] + f * (cur[1] - prev[1])
                    th = prev[2] + f * np.arctan2(np.sin(cur[2] - prev[2]), np.cos(cur[2] - prev[2]))
                    if obb_aabb_overlap(obb_corners(x, y, th, env.vehicle.length, env.vehicle.width), env.scenario.obstacles).any():
                        hit_sub = True
                prev = cur
                done = te or tr
            step_coll += int(info["collision"])
            sub_coll += int(hit_sub or info["collision"])
        out[name] = {"episodes": n, "step_instant_collisions": step_coll, "episodes_with_any_interpolated_overlap": sub_coll}
        print("substep", name, out[name], flush=True)
    return out


def brake_mismatch(n=200):
    """Model mismatch: the real car brakes weaker than the shield assumes. (a) adversarial random states (up to
    2 m/s, floor-it policy) and (b) the actual fuzzy_precise+env controller on the evaluation scenarios."""
    out = {}
    for factor in (1.0, 0.85, 0.7, 0.5):
        r = random_states(n=n, steps=100, seed=1, brake_factor=factor)
        env = new_env()
        env.vehicle.max_brake = env.vehicle.max_brake * factor
        ad = make_adapter("fuzzy_precise+env")
        coll = 0
        for sd in range(9000, 9100):
            ad.reset()
            obs, _ = env.reset(seed=sd)
            done = False
            while not done:
                a, _ = ad.act(obs, env)
                obs, _, te, tr, info = env.step(a)
                done = te or tr
            coll += int(info["collision"])
        out[f"real_brake_x{factor}"] = {"adversarial_episodes": r["episodes"], "adversarial_collisions": r["collisions"],
                                        "fuzzy_precise_env_episodes": 100, "fuzzy_precise_env_collisions": coll}
        print("mismatch", factor, out[f"real_brake_x{factor}"], flush=True)
    return out


def timeout_anatomy(root="experiments/corrected_evaluation"):
    out = {}
    for c in ["fuzzy_only_env", "fuzzy_precise_env", "ppo_v2_env", "ppo_only_env"]:
        d = pd.concat([pd.read_csv(_HERE / root / s / f"{c}.csv") for s in ("set_A", "set_B", "set_C")], ignore_index=True)
        base = pd.concat([pd.read_csv(_HERE / root / s / f"{c[:-4]}.csv") for s in ("set_A", "set_B", "set_C")], ignore_index=True)
        legacy_t = d[d.legacy_outcome == "timeout"]
        out[c] = {
            "n": len(d), "pose_timeouts": len(legacy_t), "pose_timeouts_pct": round(100 * len(legacy_t) / len(d), 1),
            "of_which_unshielded_run_collided": int(((base.legacy_outcome == "collision") & (d.legacy_outcome == "timeout")).sum()),
            "of_which_unshielded_run_succeeded": int(((base.legacy_outcome == "success") & (d.legacy_outcome == "timeout")).sum()),
            "of_which_unshielded_run_timed_out": int(((base.legacy_outcome == "timeout") & (d.legacy_outcome == "timeout")).sum()),
            "shield_active_frac_mean_in_pose_timeouts": round(float(legacy_t.shield_active_frac.mean()), 3),
            "median_final_dist_to_slot_m_in_pose_timeouts": round(float(np.hypot(legacy_t.long_error_m, legacy_t.lat_error_m).median()), 2),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/shield_verification")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = _HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in [("random_states", random_states), ("substep_audit", substep_audit), ("brake_mismatch", brake_mismatch),
                     ("timeout_anatomy", timeout_anatomy)]:
        if a.only and a.only != name:
            continue
        r = fn()
        (out / f"{name}.json").write_text(json.dumps(r, indent=2))
        print(name, json.dumps(r, indent=1)[:1500], flush=True)
