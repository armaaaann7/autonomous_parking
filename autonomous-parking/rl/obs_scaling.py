"""Fixed-constant observation scaling for the PPO policy (RL audit, experiment E1).

The raw 14-dim observation mixes metres (dx, dy up to ~8), unit-bounded sin/cos, m/s, radians
and 0-10 m ray ranges. SB3's MlpPolicy does not normalise, so the first layer sees inputs whose
scales differ by an order of magnitude. A *fixed* divisor vector (no running statistics, nothing to
save or to drift between training and deployment) puts every channel near [-1, 1].

Usage: wrap the env with ``ScaledObs`` for training/eval, and ``PPOPolicy`` applies ``scale_obs``
automatically when the checkpoint's ``.meta.json`` says ``obs_scale: true``.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

# [dx, dy, sin, cos, v, delta] + 8 rays.  dx,dy ~ +-8 m ; v max 2 m/s ; delta max 35 deg = 0.611 rad ; rays 0-10 m
OBS_SCALE = np.array([8.0, 8.0, 1.0, 1.0, 2.0, 0.611] + [10.0] * 8, dtype=np.float32)


def scale_obs(obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float32) / OBS_SCALE).astype(np.float32)


class ScaledObs(gym.ObservationWrapper):
    def observation(self, obs):
        return scale_obs(obs)
