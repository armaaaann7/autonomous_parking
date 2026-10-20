import numpy as np

from environment.parking_env import ParkingEnv
from environment.scenarios import RandomizationLevel
from environment.shielded_env import ShieldedParkingEnv


def _env(threshold=1.0):
    return ShieldedParkingEnv(ParkingEnv({"randomization": RandomizationLevel.NONE, "seed": 0}), safe_threshold=threshold)


def test_wrapper_runs_and_reports_shield_info():
    env = _env()
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert "shield_active" in info and "shield_frac" in info


def test_shield_never_removes_braking():
    env = _env(threshold=100.0)  # always active
    env.reset(seed=2)
    _, _, _, _, info = env.step(np.array([0.0, -1.0], dtype=np.float32))
    _, accel, brake = info["action_final"]
    assert brake >= 1.0 - 1e-6 and accel == 0.0


def test_huge_threshold_matches_inner_env_space():
    env = _env()
    assert env.action_space == env.unwrapped.action_space
    assert env.observation_space == env.unwrapped.observation_space
