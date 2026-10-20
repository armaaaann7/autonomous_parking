import numpy as np
from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from fuzzy.fuzzy_only_controller import FuzzyOnlyController
from evaluation.run_experiments import PPOPlusShieldAdapter


def test_stationary_near_goal_cannot_collect_positive_docking_reward():
    env = ParkingEnv()
    env.reset(seed=1)
    env.vehicle.v = 0.
    env._prev_dist = .6
    env._prev_heading_err = 0.
    assert env._reward(.6, 0., 0., False, False, False) <= 0.


def test_fuzzy_docking_finishes_reported_stall_scenario():
    env = ParkingEnv({'scenario_kinds': [Kind.PERPENDICULAR], 'randomization': RandomizationLevel.SLOT_POSITION})
    obs, _ = env.reset(seed=1000)
    ctrl = FuzzyOnlyController()
    for _ in range(500):
        obs, _, term, trunc, info = env.step(ctrl.act(obs, env.fuzzy_inputs(), env.slot_bearing_deg()))
        if term or trunc:
            break
    assert info['success'] and not info['collision']
    assert info['pos_error'] < env.pos_tol


def test_fuzzy_navigation_does_not_brake_for_safe_side_neighbour():
    ctrl = FuzzyOnlyController()
    action = ctrl.act(np.array([0., 3., 0., 1., 0., 0.]), (1.6, 90., 0.), 0.)
    assert action[1] > .25  # overcomes rolling resistance at standstill


def test_hybrid_recovery_resets_between_episodes():
    # Isolate the supervisor from model loading: actions here are deliberately stalled.
    class Policy:
        def act(self, obs): return np.array([0., .1])
    from integration.hybrid_controller import HybridController
    controller = PPOPlusShieldAdapter.__new__(PPOPlusShieldAdapter)
    controller.policy = Policy()
    controller.hybrid = HybridController()
    controller.recovery = FuzzyOnlyController()
    controller.reset()
    env = ParkingEnv({'scenario_kinds': [Kind.PERPENDICULAR]})
    obs, _ = env.reset(seed=1000)
    obs[0:2] = [0., 2.]
    obs[4] = 0.
    for _ in range(12): controller.act(obs, env)
    assert controller.recovery_active
    controller.reset()
    assert not controller.recovery_active and controller.stalled_steps == 0
