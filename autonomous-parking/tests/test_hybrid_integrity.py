"""Tests for the action path of the 'RL + Fuzzy' hybrid (audit P2).

Path under test: PPO policy -> weighted blend with Sugeno fuzzy -> fuzzy braking priority -> Mamdani
obstacle shield -> env; plus the stall latch that hands control to pure Fuzzy-only.
"""
import numpy as np
import pytest

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from evaluation.run_experiments import FuzzyOnlyAdapter, PPOPlusShieldAdapter
from fuzzy.fuzzy_only_controller import FuzzyOnlyController
from fuzzy.fuzzy_system import FuzzySystem
from integration.hybrid_controller import HybridController


class StubPolicy:
    def __init__(self, action):
        self.action, self.calls = np.array(action, dtype=np.float32), 0

    def act(self, obs):
        self.calls += 1
        return self.action.copy()


def make_adapter(policy, **kw):
    ad = PPOPlusShieldAdapter.__new__(PPOPlusShieldAdapter)
    ad.rl_weight = kw.get("rl_weight", .25)
    ad.stall_steps = kw.get("stall_steps", 12)
    ad.shield_input = "clearance"
    ad.policy = policy
    ad.hybrid = HybridController(FuzzySystem(), safe_threshold=kw.get("safe_threshold", 1.0))
    ad.recovery = FuzzyOnlyController()
    ad.trace = []
    ad.name = "t"
    ad.reset()
    return ad


def free_env():
    env = ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION, "scenario_kinds": [Kind.PERPENDICULAR]})
    obs, _ = env.reset(seed=3500)
    env.scenario.obstacles = np.zeros((0, 4))          # nothing near: Mamdani shield stays idle
    env.vehicle.reset(env.scenario.slot_pose[0] + 0.2, env.scenario.slot_pose[1] - 5.0, np.pi / 2)
    return env, env._get_obs()


def test_policy_is_queried_every_live_step():
    pol = StubPolicy([0.3, 0.2])
    ad = make_adapter(pol)
    env, obs = free_env()
    for _ in range(5):
        ad.act(obs, env)
    assert pol.calls == 5


def test_blend_is_the_documented_weighted_sum_when_shield_idle():
    pol = StubPolicy([-0.4, 0.2])
    ad = make_adapter(pol, rl_weight=.25)
    env, obs = free_env()
    a, flag = ad.act(obs, env)
    a_f = FuzzyOnlyController().act(obs, env.fuzzy_inputs(), env.slot_bearing_deg())
    expected = .25 * np.array([-0.4, 0.2]) + .75 * a_f
    assert flag == 0.0
    np.testing.assert_allclose(a, np.clip(expected, -1, 1), atol=1e-6)
    t = ad.trace[-1]
    np.testing.assert_allclose(t["a_rl"], [-0.4, 0.2], atol=1e-6)
    np.testing.assert_allclose(t["a_fuzzy"], a_f, atol=1e-6)


def test_zero_rl_weight_reproduces_fuzzy_only_action():
    ad = make_adapter(StubPolicy([0.9, -0.9]), rl_weight=0.0)
    env, obs = free_env()
    a, _ = ad.act(obs, env)
    a_f = FuzzyOnlyController().act(obs, env.fuzzy_inputs(), env.slot_bearing_deg())
    np.testing.assert_allclose(a, a_f, atol=1e-6)


def test_nonzero_rl_weight_changes_the_action():
    env, obs = free_env()
    a0, _ = make_adapter(StubPolicy([0.9, 0.0]), rl_weight=0.0).act(obs, env)
    a1, _ = make_adapter(StubPolicy([0.9, 0.0]), rl_weight=0.25).act(obs, env)
    assert abs(a1[0] - a0[0]) > 0.1


def test_stall_latch_hands_over_to_pure_fuzzy_and_stops_querying_ppo():
    """The root cause of 'RL + Fuzzy == Fuzzy-only': a stationary car triggers the latch after
    stall_steps steps; from then on PPO is never queried and the action is exactly the Fuzzy-only one."""
    pol = StubPolicy([0.0, 0.0])
    ad = make_adapter(pol, stall_steps=12)
    env, obs = free_env()
    assert abs(obs[4]) < .03                              # at rest: counts as stalled every step
    for _ in range(11):
        ad.act(obs, env)
    assert not ad.last_recovery and pol.calls == 11
    ad.act(obs, env)                                      # 12th stalled step -> latch
    assert ad.last_recovery
    calls = pol.calls
    a, _ = ad.act(obs, env)
    a_f = FuzzyOnlyController().act(obs, env.fuzzy_inputs(), env.slot_bearing_deg())
    np.testing.assert_allclose(a, a_f, atol=1e-6)
    assert pol.calls == calls                             # PPO no longer consulted
    assert ad.trace[-1]["recovery"] is True


def test_shield_next_to_a_neighbour_holds_the_car_still_at_spawn():
    """Seed 3002 spawns with ~2 cm body clearance. The Mamdani shield (threshold 1.0 m on true
    clearance) overrides the blend with a brake, so the car cannot move until the latch fires."""
    env = ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION, "scenario_kinds": [Kind.PERPENDICULAR]})
    obs, _ = env.reset(seed=3002)
    assert env.body_clearance() < 0.1
    ad = make_adapter(StubPolicy([-0.2, 0.3]))
    for _ in range(5):
        a, flag = ad.act(obs, env)
        assert flag == 1.0 and a[1] < 0
        obs, *_ = env.step(a)
    assert abs(env.vehicle.v) < 0.03


def test_fuzzy_only_and_hybrid_adapters_are_distinct_code_paths():
    assert FuzzyOnlyAdapter().name == "fuzzy_only"
    from evaluation.run_experiments import make_adapter as mk
    assert type(mk("ppo_fuzzy_bolted", str(__import__("pathlib").Path(__file__).resolve().parent.parent / "models" / "ppo_only_best"))).__name__ == "PPOPlusShieldAdapter"
    assert type(mk("fuzzy_only")).__name__ == "FuzzyOnlyAdapter"


def test_real_checkpoint_gives_nonconstant_state_dependent_actions():
    from pathlib import Path
    from rl.policy_wrapper import PPOPolicy
    ckpt = Path(__file__).resolve().parent.parent / "models" / "ppo_only_best.zip"
    if not ckpt.is_file():
        pytest.skip("checkpoint missing")
    pol = PPOPolicy(str(ckpt))
    env = ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION, "scenario_kinds": [Kind.PERPENDICULAR]})
    outs = []
    for seed in range(3000, 3010):
        obs, _ = env.reset(seed=seed)
        outs.append(pol.act(obs))
    outs = np.array(outs)
    assert np.abs(outs).max() > 0.05 and outs.std(0).max() > 0.01
