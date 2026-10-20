import numpy as np
import pytest

from fuzzy.fuzzy_system import FuzzySystem
from integration.hybrid_controller import HybridController


@pytest.fixture(scope="module")
def fuzzy():
    return FuzzySystem()


def test_no_rule_gap_across_full_grid(fuzzy):
    """The blueprint's original rule base had a (d_obs=medium, speed=medium) gap that raised in
    skfuzzy; sweep the full input grid to make sure no combination is unhandled."""
    for d in np.linspace(0, 10, 21):
        for s in np.linspace(0, 3, 13):
            for a in np.linspace(-90, 90, 13):
                out = fuzzy.compute(d, a, s)
                assert np.isfinite(out["brake_cmd"])
                assert np.isfinite(out["steer_corr"])
                assert np.isfinite(out["speed_adj"])


def test_brake_increases_as_obstacle_closes(fuzzy):
    far = fuzzy.compute(d_obs=8.0, angle_err_deg=0.0, speed=1.0)["brake_cmd"]
    near = fuzzy.compute(d_obs=0.3, angle_err_deg=0.0, speed=1.0)["brake_cmd"]
    assert near > far


def test_steer_corr_points_away_from_obstacle(fuzzy):
    """Obstacle to the left (negative bearing, shield's CW-positive convention) -> correction
    steers right (positive), and vice versa."""
    left = fuzzy.compute(d_obs=1.0, angle_err_deg=-40.0, speed=0.5)["steer_corr"]
    right = fuzzy.compute(d_obs=1.0, angle_err_deg=40.0, speed=0.5)["steer_corr"]
    assert left > 0
    assert right < 0


def test_shield_only_adds_braking_never_removes_it():
    shield = HybridController(safe_threshold=5.0)  # always active for this test
    rl_action = np.array([0.0, -0.8])  # RL already braking hard
    out = shield.act(rl_action, d_obs=0.3, angle_err_deg=0.0, speed=1.0)
    rl_brake = 0.8
    final_brake = max(0.0, -out[1])
    assert final_brake >= rl_brake - 1e-6


def test_shield_disabled_is_identity():
    shield = HybridController(enabled=False)
    rl_action = np.array([0.3, -0.2])
    out = shield.act(rl_action, d_obs=0.1, angle_err_deg=30.0, speed=1.0)
    assert np.allclose(out, rl_action)


def test_shield_inactive_outside_safe_threshold():
    shield = HybridController(safe_threshold=1.0)
    rl_action = np.array([0.3, 0.5])
    out = shield.act(rl_action, d_obs=5.0, angle_err_deg=30.0, speed=1.0)
    assert np.allclose(out, rl_action)
    assert not shield.last_shield_active
