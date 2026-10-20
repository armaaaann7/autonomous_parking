from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from fuzzy.fuzzy_only_controller import FuzzyOnlyController
from fuzzy.fuzzy_system import FuzzySystem
from integration.envelope_shield import BrakingEnvelopeShield
from integration.hybrid_controller import HybridController

_HERE = Path(__file__).resolve().parent.parent

FIELDS = [
    "episode", "scenario", "success", "collision", "timeout", "steps", "time_s",
    "pos_error_m", "heading_error_deg", "trajectory_length_m", "smoothness",
    "mean_abs_accel", "mean_brake", "reward", "shield_active_frac",
    "recovery_frac", "min_clearance_m", "in_bay", "bay_protrusion_m",
    "long_error_m", "lat_error_m", "success_mode",
    "legacy_outcome", "pose_success_step", "ps_pos_error_m", "ps_heading_error_deg", "ps_protrusion_m", "ps_in_bay",
]


# Stop radius of the containment-aware fuzzy variant ("fuzzy_precise"). Chosen on TUNING seeds only;
# see experiments/containment_tuning/ and PRESENTATION_READINESS_AUDIT.md. "fuzzy_only" keeps .32.
PRECISE_STOP_RADIUS = 0.10
PRECISE_HEADING_GAIN = 0.6


class FuzzyOnlyAdapter:
    name = "fuzzy_only"

    def __init__(self, stop_radius: float = .32, name: str = "fuzzy_only", heading_gain: float = .6):
        self.name = name
        self.controller = FuzzyOnlyController(stop_radius=stop_radius, heading_gain=heading_gain)

    def reset(self):
        pass

    def act(self, obs, env):
        return self.controller.act(obs, env.fuzzy_inputs(), env.slot_bearing_deg()), 0.0

    last_mode = "fuzzy navigation"


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
    """Weighted RL + fuzzy navigation with obstacle shielding and stall recovery.

    Normal action: 25% original PPO + 75% Sugeno fuzzy navigation, with fuzzy
    braking taking priority. A persistent stall triggers latched fuzzy recovery.
    The raw PPO checkpoint is unchanged. This is not a trained joint policy.
    """

    shield_input = "clearance"
    last_recovery = False
    rl_weight = .25
    stall_steps = 12
    trace = None

    def __init__(self, model_path: str, name: str = "ppo_fuzzy_bolted", safe_threshold: float = 1.0,
                 shield_input: str = "clearance", rl_weight: float = .25, stall_steps: int = 12):
        # rl_weight / stall_steps default to the historical values (25 % PPO, latch after 12 stalled
        # steps). They exist so evaluation/diagnose_hybrid.py can run counterfactuals; the reported
        # "RL + Fuzzy" controller always uses the defaults.
        self.rl_weight, self.stall_steps = float(rl_weight), int(stall_steps)
        self.trace = None            # optional list: per-step dicts of every intermediate action
        # shield_input: "clearance" (default) feeds the Mamdani shield the TRUE body clearance
        # (ParkingEnv.shield_inputs); "centre_ray" reproduces the original, flawed input (distance
        # of the nearest of 8 rays cast from the body centre), kept only so the old results can be
        # reproduced. See PRESENTATION_READINESS_AUDIT.md, defect D1.
        if shield_input not in ("clearance", "centre_ray"):
            raise ValueError("shield_input must be 'clearance' or 'centre_ray'")
        self.shield_input = shield_input
        self.name = name
        from rl.policy_wrapper import PPOPolicy
        self.policy = PPOPolicy(model_path)
        self.hybrid = HybridController(FuzzySystem(), safe_threshold=safe_threshold)
        self.recovery = FuzzyOnlyController()
        self.reset()

    def reset(self):
        self.stalled_steps = 0
        self.recovery_active = False
        self.last_recovery = False
        self.last_mode = 'RL + obstacle shield'

    def act(self, obs, env):
        # Explicit supervisory hybrid: fuse RL and fuzzy actions normally; use
        # fuzzy recovery after 1.2 seconds stationary outside parking tolerance.
        dist = float(np.hypot(obs[0], obs[1]))
        self.stalled_steps = self.stalled_steps + 1 if abs(obs[4]) < .03 and dist >= env.pos_tol else 0
        if self.stalled_steps >= self.stall_steps:
            self.recovery_active = True
        if self.recovery_active:
            action = self.recovery.act(obs, env.fuzzy_inputs(), env.slot_bearing_deg())
            self.last_mode = 'Fuzzy recovery (PPO stalled)'
            self.last_recovery = True
            if self.trace is not None:
                self.trace.append({"recovery": True, "a_final": np.array(action, float)})
            return action, 0.0
        a_rl = self.policy.act(obs)
        inputs = env.fuzzy_inputs()
        a_fuzzy = self.recovery.act(obs, inputs, env.slot_bearing_deg())
        # Explicit weighted fusion, not a hidden replacement of the RL baseline.
        # Fuzzy navigation supplies conservative docking guidance throughout.
        action = self.rl_weight * a_rl + (1.0 - self.rl_weight) * a_fuzzy
        a_blend = action.copy()
        if a_fuzzy[1] < 0:
            action[1] = min(action[1], a_fuzzy[1])
        d_obs, angle_err, speed = inputs if self.shield_input == "centre_ray" else env.shield_inputs()
        a_final = self.hybrid.act(action, d_obs, angle_err, speed)
        self.last_recovery = False
        if self.trace is not None:
            self.trace.append({"a_rl": np.array(a_rl, float), "a_fuzzy": np.array(a_fuzzy, float), "a_blend": a_blend,
                               "a_braked": np.array(action, float), "a_final": np.array(a_final, float),
                               "shield": bool(self.hybrid.last_shield_active), "recovery": False})
        self.last_mode = 'RL 25% + Fuzzy 75%'
        if self.hybrid.last_shield_active:
            self.last_mode += ' + obstacle shield'
        # Second value = did a SAFETY layer (Mamdani obstacle shield) modify this step's action?
        # (It used to be a constant 1.0, which made shield_active_frac meaningless.)
        return a_final, float(self.hybrid.last_shield_active)


class EnvelopeAdapter:
    """Wrap any adapter with the model-based braking-envelope shield (integration/envelope_shield.py)."""

    def __init__(self, inner, shield: BrakingEnvelopeShield | None = None):
        self.inner = inner
        self.shield = shield or BrakingEnvelopeShield()
        self.name = inner.name + "_env"
        self.kinds = getattr(inner, "kinds", None)
        self.last_mode = getattr(inner, "last_mode", "")
        self.last_recovery = False

    def reset(self):
        self.inner.reset()
        self.shield.reset()

    def act(self, obs, env):
        action, flag = self.inner.act(obs, env)
        safe_action = self.shield.filter(env, action)
        self.last_recovery = bool(getattr(self.inner, "last_recovery", False))
        base_mode = getattr(self.inner, "last_mode", "")
        self.last_mode = (base_mode + " + " if base_mode else "") + ("ENVELOPE BRAKE" if self.shield.last_intervened else "envelope ok")
        return safe_action, float(bool(flag) or self.shield.last_intervened)


# ppo_v2 = the retrained PPO (audit P3: curriculum recipe, 3 seeds, selected on validation seeds 7000-7099);
# ppo_v2_fuzzy = 25 % ppo_v2 + 75 % Sugeno fuzzy with the Mamdani shield and stall latch DISABLED (weight chosen on
# tuning seeds 8000-8099 as the best non-zero weight; see experiments/hybrid_diagnosis/).
PPO_V2_MODEL = "models/ppo_v2_curriculum_selected"
ADAPTER_NAMES = ("fuzzy_only", "ppo_only", "ppo_fuzzy_bolted", "fuzzy_precise", "ppo_v2", "ppo_v2_fuzzy")


def _named(adapter, name):
    adapter.name = name
    return adapter


def make_adapter(name: str, model_path: str | None = None):
    """Single factory used by evaluation, demo, trajectory export and the viewer scripts.

    ``name`` is one of ADAPTER_NAMES, optionally suffixed with ``+env`` (or ``_env``) to wrap it
    in the braking-envelope shield. Raises a clear error if a PPO checkpoint is needed but missing.
    """
    envelope = name.endswith("+env") or name.endswith("_env")
    base = name[:-4] if name.endswith("+env") else name[:-4] if name.endswith("_env") else name
    stop_override, hg_override = None, None
    if "@" in base:                      # e.g. fuzzy_precise@0.15 or fuzzy_precise@0.06x1.2 (stop radius x heading gain; tuning only)
        base, val = base.split("@", 1)
        parts = [float(v) for v in val.split("x")]          # "r" or "rxheading_gain"
        stop_override, hg_override = parts[0], (parts[1] if len(parts) > 1 else None)
    if base not in ADAPTER_NAMES:
        raise ValueError(f"unknown controller {name!r}; choose from {ADAPTER_NAMES} (optionally + '+env')")
    if base in ("ppo_v2", "ppo_v2_fuzzy"):
        model_path = str(_HERE / PPO_V2_MODEL)
    if base not in ("fuzzy_only", "fuzzy_precise"):
        if not model_path or not Path(model_path).with_suffix(".zip").is_file():
            raise FileNotFoundError(f"PPO checkpoint not found: {model_path}.zip - controller {base!r} needs it")
    adapter = {"fuzzy_only": lambda: FuzzyOnlyAdapter(),
               "fuzzy_precise": lambda: FuzzyOnlyAdapter(
                   stop_override if stop_override is not None else PRECISE_STOP_RADIUS,
                   "fuzzy_precise" if stop_override is None else
                   f"fuzzy_precise@{stop_override:g}" + ("" if hg_override is None else f"x{hg_override:g}"),
                   hg_override if hg_override is not None else PRECISE_HEADING_GAIN),
               "ppo_only": lambda: PPOOnlyAdapter(str(model_path)),
               "ppo_fuzzy_bolted": lambda: PPOPlusShieldAdapter(str(model_path), "ppo_fuzzy_bolted"),
               "ppo_v2": lambda: _named(PPOOnlyAdapter(str(model_path)), "ppo_v2"),
               "ppo_v2_fuzzy": lambda: PPOPlusShieldAdapter(str(model_path), "ppo_v2_fuzzy", safe_threshold=0.0,
                                                           rl_weight=.25, stall_steps=10 ** 9)}[base]()
    return EnvelopeAdapter(adapter) if envelope else adapter



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
            recovery_sum, min_clear = 0.0, float(env.body_clearance())
            ep_reward, steps = 0.0, 0
            # Legacy (pose-criterion) outcome reconstructed from the same trajectory: the first of
            # {collision, pose success held hold_steps steps, timeout}. In 'pose' mode the episode ends
            # there; in 'contained' mode it continues, so one run yields BOTH outcomes without
            # redefining either (verified against the stored legacy CSVs by tests/test_containment_metric).
            pose_streak, legacy_outcome, ps = 0, "", {}
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
                recovery_sum += float(getattr(adapter, 'last_recovery', False))
                min_clear = min(min_clear, env.body_clearance())
                if not legacy_outcome:
                    pose_streak = pose_streak + 1 if step_info["pose_ok"] else 0
                    if step_info["collision"]:
                        legacy_outcome = "collision"
                    elif pose_streak >= env.hold_steps:
                        legacy_outcome = "success"
                        ps = {"pose_success_step": steps, "ps_pos_error_m": step_info["pos_error"],
                              "ps_heading_error_deg": step_info["heading_error_deg"],
                              "ps_protrusion_m": step_info["bay_protrusion"], "ps_in_bay": step_info["contained"]}
                done = term or trunc
            row = {
                "episode": ep, "scenario": info.get("scenario", ""),
                "success": step_info["success"], "collision": step_info["collision"],
                "timeout": step_info["timeout"], "steps": steps, "time_s": steps * env.dt,
                "pos_error_m": step_info["pos_error"], "heading_error_deg": step_info["heading_error_deg"],
                "trajectory_length_m": traj_len, "smoothness": smooth_sum / max(steps, 1),
                "mean_abs_accel": accel_sum / max(steps, 1), "mean_brake": brake_sum / max(steps, 1),
                "reward": ep_reward, "shield_active_frac": shield_sum / max(steps, 1),
                "recovery_frac": recovery_sum / max(steps, 1), "min_clearance_m": min_clear,
                "in_bay": env.inside_bay(), "bay_protrusion_m": env.bay_protrusion(),
                "long_error_m": step_info["slot_frame_error"][0], "lat_error_m": step_info["slot_frame_error"][1],
                "success_mode": env.success_mode,
                "legacy_outcome": legacy_outcome or "timeout",
                "pose_success_step": ps.get("pose_success_step", ""), "ps_pos_error_m": ps.get("ps_pos_error_m", ""),
                "ps_heading_error_deg": ps.get("ps_heading_error_deg", ""), "ps_protrusion_m": ps.get("ps_protrusion_m", ""),
                "ps_in_bay": ps.get("ps_in_bay", ""),
            }
            writer.writerow(row)
    env.close()
    print(f"[{adapter.name}] wrote {n_episodes} episodes -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare controllers on identical perpendicular scenarios (shared seeds).")
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--model', default='models/ppo_only_best')
    parser.add_argument('--seed', type=int, default=1000, help='first seed; episode i uses seed+i for EVERY controller')
    parser.add_argument('--out-dir', default='experiments/results')
    parser.add_argument('--success-mode', default='pose', choices=['pose', 'contained'],
                        help="'pose' = legacy centre-tolerance success (default, reproduces old results); "
                             "'contained' = whole footprint inside the bay")
    parser.add_argument('--controllers', default=','.join(ADAPTER_NAMES),
                        help="comma list from %s, each optionally suffixed '+env' (envelope shield)" % (ADAPTER_NAMES,))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    model = _HERE / args.model
    names = [n.strip() for n in args.controllers.split(',') if n.strip()]
    try:
        adapters = [make_adapter(n, str(model)) for n in names]
    except (FileNotFoundError, ValueError) as err:
        parser.error(str(err))
    seeds = list(range(args.seed, args.seed + args.episodes))
    for adapter in adapters:
        run_episodes(adapter, args.episodes, seeds, [Kind.PERPENDICULAR],
                     _HERE / args.out_dir / f'{adapter.name}.csv', env_extra={"success_mode": args.success_mode})


if __name__ == '__main__':
    main()
