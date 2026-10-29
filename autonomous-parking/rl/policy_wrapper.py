"""Loads a trained PPO checkpoint and exposes a plain ``.act(obs) -> action`` interface, so
evaluation code doesn't need to know it's talking to SB3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from environment.observations import SLOT_OBS_DIM
from rl.obs_scaling import scale_obs

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
        # Optional sidecar written by rl.train_ppo: <name>.meta.json. Absent for the original
        # checkpoints, which were trained on raw observations (obs_scale False).
        meta_path = zip_path.with_suffix(".meta.json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        self.obs_scale = bool(self.meta.get("obs_scale", False))
        # Observation convention and gear mode the checkpoint was trained with (old checkpoints: legacy, forward-only).
        env_extra = self.meta.get("env_extra", {})
        self.obs_mode = env_extra.get("obs_mode", "legacy")
        self.obs_dim = SLOT_OBS_DIM if self.obs_mode == "slot" else OBS_DIM
        # keys the EVALUATION environment must be built with for this policy to be meaningful
        self.env_extra = {k: env_extra[k] for k in ("obs_mode", "allow_reverse") if k in env_extra}
        obs_shape = tuple(self.model.observation_space.shape)
        act_shape = tuple(self.model.action_space.shape)
        if obs_shape != (self.obs_dim,) or act_shape != (ACT_DIM,):
            raise ValueError(
                f"Checkpoint {zip_path.name} expects observation {obs_shape} / action {act_shape}, but "
                f"ParkingEnv({self.obs_mode}) provides ({self.obs_dim},) / ({ACT_DIM},). It was trained for a different environment.")

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.obs_dim,) or not np.isfinite(obs).all():
            raise ValueError(f"invalid observation {obs.shape}; expected finite ({self.obs_dim},) for obs_mode={self.obs_mode}")
        if self.obs_scale:
            obs = scale_obs(obs)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)
