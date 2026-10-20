"""ShieldedParkingEnv: applies the fuzzy safety shield INSIDE the environment, before the
physics step, so a policy trained on it learns against the same dynamics it is deployed under
("train through the shield"). Training unshielded and bolting the shield on at test time is a
distribution shift; this wrapper removes it.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from fuzzy.fuzzy_system import FuzzySystem
from integration.hybrid_controller import HybridController


class ShieldedParkingEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, safe_threshold: float = 1.0, fuzzy: FuzzySystem | None = None):
        super().__init__(env)
        self.hybrid = HybridController(fuzzy or FuzzySystem(), safe_threshold=safe_threshold)
        self._shield_steps = 0
        self._steps = 0

    def reset(self, **kwargs):
        self._shield_steps = 0
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        base = self.env.unwrapped
        d_obs, bearing, speed = base.fuzzy_inputs()
        shielded = self.hybrid.act(np.asarray(action, dtype=np.float32), d_obs, bearing, speed)
        obs, reward, term, trunc, info = self.env.step(shielded)
        self._steps += 1
        self._shield_steps += int(self.hybrid.last_shield_active)
        info["shield_active"] = bool(self.hybrid.last_shield_active)
        info["shield_frac"] = self._shield_steps / self._steps
        info["action_requested"] = [float(a) for a in np.asarray(action).ravel()]
        return obs, reward, term, trunc, info
