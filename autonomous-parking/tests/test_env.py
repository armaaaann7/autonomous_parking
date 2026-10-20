import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel


def test_check_env_passes():
    env = ParkingEnv({"seed": 0})
    check_env(env, skip_render_check=True)


def test_obstacle_free_mode_has_no_obstacles():
    env = ParkingEnv({"include_obstacles": False})
    env.reset(seed=0)
    assert env.scenario.obstacles.shape == (0, 4)


@pytest.mark.parametrize("kind", [Kind.PARALLEL, Kind.PERPENDICULAR])
def test_reset_is_deterministic_given_seed(kind):
    env = ParkingEnv({"scenario_kinds": [kind]})
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs1, obs2)


def test_collision_terminates_episode():
    env = ParkingEnv({"randomization": RandomizationLevel.NONE, "scenario_kinds": [Kind.PARALLEL]})
    env.reset(seed=0)
    # drive hard left into the neighbouring obstacle
    terminated = False
    for _ in range(100):
        obs, r, term, trunc, info = env.step(np.array([1.0, 1.0]))
        if term:
            terminated = True
            assert info["collision"]
            break
    assert terminated, "expected the car to collide within 100 steps of driving at max steer+throttle"


def test_success_requires_holding_tolerance_for_consecutive_steps():
    env = ParkingEnv({"include_obstacles": False, "scenario_kinds": [Kind.PARALLEL]})
    env.reset(seed=0)  # PARALLEL slot has theta=0, so the manual pose override below is exact
    env.vehicle.rx = env.scenario.slot_pose[0] - env.vehicle.centre_offset
    env.vehicle.ry = env.scenario.slot_pose[1]
    env.vehicle.theta = env.scenario.slot_pose[2]
    env.vehicle.v = 0.0
    assert env._success_streak == 0
    _, _, term1, _, info1 = env.step(np.array([0.0, 0.0]))
    assert not info1["success"], "should not succeed on the very first qualifying step"
    _, _, term2, _, info2 = env.step(np.array([0.0, 0.0]))
    assert info2["success"]  # hold_steps=2 in configs/env_config.yaml


def test_reward_is_always_finite():
    env = ParkingEnv({"seed": 1})
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=2))
        assert np.isfinite(r)
        if term or trunc:
            env.reset()
