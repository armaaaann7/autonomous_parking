"""Loads a trained PPO checkpoint and exposes a plain ``.act(obs) -> action`` interface, so
evaluation code doesn't need to know it's talking to SB3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

OBS_DIM = 14      # [dx, dy, sin, cos, v, delta] + 8 rays   (environment/parking_env.py)
ACT_DIM = 2       # [steer, throttle_brake]


class PPOPolicy:
    def __init__(self, model_path: str):
        zip_path = Path(model_path).with_suffix(".zip") if Path(model_path).suffix != ".zip" else Path(model_path)
        if not zip_path.is_file():
            raise FileNotFoundError(f"PPO checkpoint not found: {zip_path}")
        try:
            self.model = PPO.load(str(zip_path))
        except Exception as err:  # corrupt zip, SB3/torch version mismatch, wrong file type...
            raise RuntimeError(f"Could not load PPO checkpoint {zip_path}: {err}") from err
        obs_shape = tuple(self.model.observation_space.shape)
        act_shape = tuple(self.model.action_space.shape)
        if obs_shape != (OBS_DIM,) or act_shape != (ACT_DIM,):
            raise ValueError(
                f"Checkpoint {zip_path.name} expects observation {obs_shape} / action {act_shape}, but "
                f"ParkingEnv provides ({OBS_DIM},) / ({ACT_DIM},). It was trained for a different environment.")

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (OBS_DIM,) or not np.isfinite(obs).all():
            raise ValueError(f"invalid observation {obs.shape}; expected finite ({OBS_DIM},)")
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)
