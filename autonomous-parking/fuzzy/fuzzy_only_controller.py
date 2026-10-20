"""Fuzzy system driving the car directly (blueprint Part 17, Experiment 3 / Part 5's "Approach
A"). This isolates what hand-designed expert knowledge alone achieves, with no RL involved.

The fuzzy rule base (fuzzy_system.py) only ever modelled obstacle AVOIDANCE (brake near
obstacles, steer away from them) -- it was never a general-purpose "steer toward any target"
controller, and its angle_err input uses a clockwise-positive convention specifically for that
(see ParkingEnv.fuzzy_inputs's docstring). An earlier version of this file fed it the
bearing-to-slot instead and used its output as the only steering signal, which both pointed the
car the wrong way (navigation needs the opposite sign convention) and had no distinct avoidance
behaviour layered on top. This version mirrors integration/hybrid_controller.py's architecture
instead: a plain proportional-navigation steer toward the slot, with the SAME fuzzy avoidance
correction the RL+Fuzzy hybrid uses added on top. See docs/REVIEW.md.
"""
from __future__ import annotations

import numpy as np

from .fuzzy_system import FuzzySystem


class FuzzyOnlyController:
    def __init__(self, fuzzy: FuzzySystem | None = None, slow_dist: float = 3.0, safe_threshold: float = 3.0):
        self.fuzzy = fuzzy or FuzzySystem()
        self.slow_dist = slow_dist
        self.safe_threshold = safe_threshold

    def act(self, obs: np.ndarray, fuzzy_inputs, slot_bearing_deg: float) -> np.ndarray:
        d_obs, obstacle_bearing_deg, speed = fuzzy_inputs
        dx, dy = obs[0], obs[1]
        heading_err = np.arctan2(obs[2], obs[3])
        dist = float(np.hypot(dx, dy))

        # navigation: same proportional law as the rule-based baseline
        steer = float(np.clip(1.5 * np.deg2rad(slot_bearing_deg) - 0.5 * heading_err, -1.0, 1.0))
        target_throttle = float(np.clip(dist / self.slow_dist, 0.15, 1.0))
        thb = target_throttle

        if d_obs < self.safe_threshold:
            out = self.fuzzy.compute(d_obs, obstacle_bearing_deg, speed)
            steer = float(np.clip(steer + out["steer_corr"], -1.0, 1.0))
            throttle_cap = max(1.0 + min(out["speed_adj"], 0.0), 0.0)
            thb = min(thb, throttle_cap) - out["brake_cmd"]

        return np.array([steer, float(np.clip(thb, -1.0, 1.0))], dtype=np.float32)
