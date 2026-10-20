import numpy as np

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel


def _env(reverse):
    return ParkingEnv({"scenario_kinds": [Kind.PARALLEL], "randomization": RandomizationLevel.SLOT_POSITION,
                       "allow_reverse": reverse, "min_start_distance_scale": 1.0, "start_distance_scale": 1.0})


def test_default_env_cannot_reverse():
    env = _env(False)
    env.reset(seed=0)
    for _ in range(30):
        env.step(np.array([0.0, 1.0], dtype=np.float32))
    for _ in range(150):
        env.step(np.array([0.0, -1.0], dtype=np.float32))
        assert env.vehicle.v >= 0.0


def test_reverse_env_can_reverse_and_brakes_first():
    env = _env(True)
    env.reset(seed=0)
    for _ in range(20):
        env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert env.vehicle.v > 0.5
    env.step(np.array([0.0, -1.0], dtype=np.float32))          # while rolling forward: this is braking
    assert env.vehicle.v >= 0.0 and env.vehicle.v < 2.0
    for _ in range(80):
        env.step(np.array([0.0, -1.0], dtype=np.float32))
    assert env.vehicle.v < -0.5                                  # at standstill it reverses


def test_reverse_env_default_gap_is_feasible_size():
    env = _env(True)
    env.reset(seed=1)
    ob = env.scenario.obstacles
    gap = ob[1, 0] - ob[0, 2]
    assert abs(gap - 8.0) < 1e-6

