"""Weak, hand-coded baseline (blueprint Part 17, #1): proportional steering toward the slot,
throttle proportional to distance, hard braking near obstacles. Deliberately simple -- its job
is to establish the floor, not to be competitive.
"""
from __future__ import annotations

import numpy as np


class RuleBasedController:
    def __init__(self, brake_dist: float = 1.5, slow_dist: float = 3.0):
        self.brake_dist = brake_dist
        self.slow_dist = slow_dist

    def act(self, obs: np.ndarray) -> np.ndarray:
        dx, dy, sin_e, cos_e, v, delta = obs[:6]
        rays = obs[6:]
        d_obs = float(np.min(rays))
        heading_err = np.arctan2(sin_e, cos_e)
        dist = float(np.hypot(dx, dy))
        bearing = np.arctan2(dy, dx)

        # steer to reduce bearing error, damped by heading error so the car doesn't overshoot
        steer = float(np.clip(1.5 * bearing - 0.5 * heading_err, -1.0, 1.0))

        if d_obs < self.brake_dist:
            thb = -1.0
        elif dist < 0.5:
            thb = -0.3  # gentle brake on final approach
        else:
            target = np.clip(dist / self.slow_dist, 0.2, 1.0)
            thb = float(np.clip(target - 0.3 * abs(v), -1.0, 1.0))
        return np.array([steer, thb], dtype=np.float32)
