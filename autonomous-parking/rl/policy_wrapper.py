"""Loads a trained PPO checkpoint and exposes a plain ``.act(obs) -> action`` interface, so
evaluation code doesn't need to know it's talking to SB3 (blueprint Part 14's stated purpose for
this file).
"""
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO


class PPOPolicy:
    def __init__(self, model_path: str):
        self.model = PPO.load(model_path)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)
