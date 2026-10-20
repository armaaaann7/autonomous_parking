"""RL + fuzzy safety shield (blueprint Part 5), adapted to the env's 2-D
``[steer, throttle_brake]`` action (see environment/parking_env.py docstring, point 1).

Safety invariant preserved from the blueprint: the shield can only ever *add* braking
(`max(rl_brake, fuzzy_brake)`), never remove RL-commanded braking. It is a pure function of
its inputs and one config flag (``enabled``) away from being a no-op, so it can be switched off
for the RL-only baseline (Part 17, Experiment 2) and for the "shield at train time vs. test time
only" ablation (Part 20).
"""
from __future__ import annotations

import numpy as np

from fuzzy.fuzzy_system import FuzzySystem


class HybridController:
    def __init__(self, fuzzy: FuzzySystem | None = None, safe_threshold: float = 1.0, enabled: bool = True):
        # NOTE: safe_threshold is deliberately tight (1.0m, not the fuzzy system's own ~3m
        # "medium distance" range). PPO is trained with NO shield in the loop (rl/train_ppo.py),
        # so every shield intervention at test time is out-of-distribution for the policy. With
        # neighbour cars parked close to the slot, a generous threshold makes the shield active
        # on >90% of steps, constantly perturbing an otherwise-competent approach and dragging
        # success down well below PPO-alone (see docs/REVIEW.md). The methodologically correct
        # fix is to train a policy WITH the shield wrapped into the environment from the start
        # (true "shielded RL", not shield-bolted-on-at-test-time); this tight threshold is the
        # pragmatic stand-in given this session's compute budget.
        self.fuzzy = fuzzy or FuzzySystem()
        self.safe_threshold = safe_threshold
        self.enabled = enabled
        self.last_shield_active = False
        self.last_fuzzy_out = None

    def act(self, a_rl, d_obs: float, angle_err_deg: float, speed: float):
        steer, thb = float(a_rl[0]), float(a_rl[1])
        if not self.enabled or d_obs >= self.safe_threshold:
            self.last_shield_active = False
            self.last_fuzzy_out = None
            return np.array([steer, thb], dtype=np.float32)

        out = self.fuzzy.compute(d_obs, angle_err_deg, speed)
        self.last_shield_active = True
        self.last_fuzzy_out = out

        steer_final = float(np.clip(steer + out["steer_corr"], -1.0, 1.0))

        rl_accel = max(thb, 0.0)
        rl_brake = max(-thb, 0.0)
        brake_final = max(rl_brake, out["brake_cmd"])          # can only add braking, never remove it
        throttle_cap = max(1.0 + min(out["speed_adj"], 0.0), 0.0)  # negative speed_adj caps throttle
        accel_final = min(rl_accel, throttle_cap)
        thb_final = float(np.clip(accel_final - brake_final, -1.0, 1.0))

        return np.array([steer_final, thb_final], dtype=np.float32)
