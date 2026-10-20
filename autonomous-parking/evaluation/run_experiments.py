"""Runs the 5-way comparison (blueprint Part 17/19): rule-based, fuzzy-only, PPO-only,
PPO+fuzzy, each for N episodes, logging one CSV per controller with every metric from Part 18.

Differences from the blueprint's Step-8 sketch: real per-episode metrics are recorded (position
error, heading error, trajectory length, smoothness, control effort) rather than hard-coded
zeros, and a fixed evaluation-seed sequence is shared across all 4 controllers so a given episode
index means the same start pose / slot / obstacles for everyone -- otherwise the four rows in
Part 19's table are not a fair comparison.

Usage:
    python -m evaluation.run_experiments --episodes 100 --ppo-model models/ppo_parking_v1
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from baselines.rule_based_controller import RuleBasedController
from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from baselines.parallel_geometric import ParallelGeometricController
from fuzzy.fuzzy_only_controller import FuzzyOnlyController
from fuzzy.fuzzy_system import FuzzySystem
from integration.hybrid_controller import HybridController

_HERE = Path(__file__).resolve().parent.parent

FIELDS = [
    "episode", "scenario", "success", "collision", "timeout", "steps", "time_s",
    "pos_error_m", "heading_error_deg", "trajectory_length_m", "smoothness",
    "mean_abs_accel", "mean_brake", "reward", "shield_active_frac",
]


class RuleBasedAdapter:
    name = "rule_based"

    def __init__(self):
        self.controller = RuleBasedController()

    def reset(self):
        pass

    def act(self, obs, env):
        return self.controller.act(obs), 0.0


class FuzzyOnlyAdapter:
    name = "fuzzy_only"

    def __init__(self):
        self.controller = FuzzyOnlyController()

    def reset(self):
        pass

    def act(self, obs, env):
        return self.controller.act(obs, env.fuzzy_inputs(), env.slot_bearing_deg()), 0.0


class ParallelGeometricAdapter:
    """Scripted reverse S-curve (needs the env's reverse gear; parallel slots only)."""
    name = "parallel_geometric"
    kinds = [Kind.PARALLEL]

    def __init__(self):
        self.controller = ParallelGeometricController()

    def reset(self):
        self.controller.reset()

    def act(self, obs, env):
        return self.controller.act(env), 0.0


class PPOOnlyAdapter:
    name = "ppo_only"

    def __init__(self, model_path: str):
        from rl.policy_wrapper import PPOPolicy
        self.policy = PPOPolicy(model_path)

    def reset(self):
        pass

    def act(self, obs, env):
        return self.policy.act(obs), 0.0


class PPOPlusShieldAdapter:
    """PPO checkpoint + fuzzy shield at inference. name='ppo_fuzzy_bolted' for a model trained
    WITHOUT the shield, 'ppo_shielded' for one trained through it."""

    def __init__(self, model_path: str, name: str = "ppo_fuzzy_bolted", safe_threshold: float = 1.0):
        self.name = name
        from rl.policy_wrapper import PPOPolicy
        self.policy = PPOPolicy(model_path)
        self.hybrid = HybridController(FuzzySystem(), safe_threshold=safe_threshold)

    def reset(self):
        pass

    def act(self, obs, env):
        a_rl = self.policy.act(obs)
        d_obs, angle_err, speed = env.fuzzy_inputs()
        a_final = self.hybrid.act(a_rl, d_obs, angle_err, speed)
        return a_final, float(self.hybrid.last_shield_active)


def run_episodes(adapter, n_episodes: int, seeds: list[int], scenario_kinds, out_path: Path,
                  randomization=RandomizationLevel.SLOT_POSITION, env_extra: dict | None = None):
    scenario_kinds = getattr(adapter, "kinds", None) or scenario_kinds
    env = ParkingEnv({"randomization": randomization, "scenario_kinds": scenario_kinds, **(env_extra or {})})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for ep in range(n_episodes):
            adapter.reset()
            obs, info = env.reset(seed=seeds[ep])
            prev_pos = env.vehicle.centre.copy()
            prev_steer = 0.0
            traj_len, smooth_sum, accel_sum, brake_sum, shield_sum = 0.0, 0.0, 0.0, 0.0, 0.0
            ep_reward, steps = 0.0, 0
            t0 = time.time()
            done = False
            while not done:
                action, shield_active = adapter.act(obs, env)
                obs, reward, term, trunc, step_info = env.step(action)
                steps += 1
                ep_reward += reward
                pos = env.vehicle.centre
                traj_len += float(np.linalg.norm(pos - prev_pos))
                prev_pos = pos.copy()
                steer_now = action[0]
                smooth_sum += abs(float(steer_now) - prev_steer)
                prev_steer = float(steer_now)
                _, accel, brake = step_info["action_final"]
                accel_sum += abs(accel)
                brake_sum += brake
                shield_sum += shield_active
                done = term or trunc
            row = {
                "episode": ep, "scenario": info.get("scenario", ""),
                "success": step_info["success"], "collision": step_info["collision"],
                "timeout": step_info["timeout"], "steps": steps, "time_s": steps * env.dt,
                "pos_error_m": step_info["pos_error"], "heading_error_deg": step_info["heading_error_deg"],
                "trajectory_length_m": traj_len, "smoothness": smooth_sum / max(steps, 1),
                "mean_abs_accel": accel_sum / max(steps, 1), "mean_brake": brake_sum / max(steps, 1),
                "reward": ep_reward, "shield_active_frac": shield_sum / max(steps, 1),
            }
            writer.writerow(row)
    env.close()
    print(f"[{adapter.name}] wrote {n_episodes} episodes -> {out_path}")


def main(episodes: int, ppo_only_model: str, ppo_shielded_model: str, out_dir: str, seed0: int = 1000,
         safe_threshold: float = 1.0, reverse: bool = False, kinds_name: str = "both",
         full_approach: bool = False):
    seeds = list(range(seed0, seed0 + episodes))
    kinds = {"both": [Kind.PARALLEL, Kind.PERPENDICULAR], "parallel": [Kind.PARALLEL],
             "perpendicular": [Kind.PERPENDICULAR]}[kinds_name]
    env_extra = {"allow_reverse": True} if reverse else {}
    if full_approach:   # always start at the standard ~6 m approach (default draws start distance at random,
        env_extra.update({"min_start_distance_scale": 1.0, "start_distance_scale": 1.0})  # incl. starts beside/in the row)
    out_dir = _HERE / out_dir

    adapters = [RuleBasedAdapter(), FuzzyOnlyAdapter()]
    if reverse:
        adapters.append(ParallelGeometricAdapter())
    p_only, p_shield = _HERE / ppo_only_model, _HERE / ppo_shielded_model
    if p_only.with_suffix(".zip").exists():
        adapters += [PPOOnlyAdapter(str(p_only)),
                     PPOPlusShieldAdapter(str(p_only), "ppo_fuzzy_bolted", safe_threshold)]
    else:
        print(f"[warn] no unshielded PPO at {p_only}.zip -- skipping ppo_only / ppo_fuzzy_bolted")
    if p_shield.with_suffix(".zip").exists():
        adapters.append(PPOPlusShieldAdapter(str(p_shield), "ppo_shielded", safe_threshold))
    else:
        print(f"[warn] no shield-trained PPO at {p_shield}.zip -- skipping ppo_shielded")

    for adapter in adapters:
        run_episodes(adapter, episodes, seeds, kinds, out_dir / f"{adapter.name}.csv", env_extra=env_extra)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--ppo-only-model", type=str, default="models/ppo_only")
    p.add_argument("--ppo-shielded-model", type=str, default="models/ppo_shielded")
    p.add_argument("--safe-threshold", type=float, default=1.0)
    p.add_argument("--reverse", action="store_true", help="enable the reverse gear (needed for parallel parking)")
    p.add_argument("--full-approach", action="store_true", help="fix the start distance at the standard 6 m approach")
    p.add_argument("--kinds", choices=["both", "parallel", "perpendicular"], default="both")
    p.add_argument("--out-dir", type=str, default="experiments/results")
    args = p.parse_args()
    main(args.episodes, args.ppo_only_model, args.ppo_shielded_model, args.out_dir,
         safe_threshold=args.safe_threshold, reverse=args.reverse, kinds_name=args.kinds,
         full_approach=args.full_approach)
