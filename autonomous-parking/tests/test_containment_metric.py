"""Tests for the strict full-containment success metric (audit P1).

Geometry under test (configs/env_config.yaml + environment/scenarios.py): car 4.2 x 1.8 m, painted
bay = car + 0.6 m = 2.4 x 4.8 m (0.30 m clearance on every side), legacy pose success = centre within
0.40 m, heading within 15 deg, speed under 0.20 m/s for 2 steps.
"""
import numpy as np
import pandas as pd
import pytest

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel


def make_env(mode="pose"):
    env = ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION,
                      "scenario_kinds": [Kind.PERPENDICULAR], "success_mode": mode})
    env.reset(seed=11)
    return env


def place(env, long=0.0, lat=0.0, hdg_deg=0.0):
    """Put the (stationary) car at a slot-frame offset / heading error from the slot centre."""
    sx, sy, st = env.scenario.slot_pose
    x = sx + long * np.cos(st) - lat * np.sin(st)
    y = sy + long * np.sin(st) + lat * np.cos(st)
    env.vehicle.reset(x, y, st + np.deg2rad(hdg_deg))


def test_default_mode_is_legacy_pose():
    assert ParkingEnv({}).success_mode == "pose"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        ParkingEnv({"success_mode": "loose"})


def test_perfectly_parked_is_contained():
    env = make_env()
    place(env)
    assert env.bay_protrusion() == pytest.approx(0.0, abs=1e-9) and env.inside_bay()


@pytest.mark.parametrize("long,expect_in", [(0.29, True), (0.30, True), (0.31, False), (-0.31, False), (-0.30, True)])
def test_longitudinal_boundary(long, expect_in):
    # bay end clearance is exactly 0.30 m; flush is contained (numerical tolerance 1e-9), 1 cm over is not
    env = make_env()
    place(env, long=long)
    assert env.inside_bay() is expect_in
    if not expect_in:
        assert env.bay_protrusion() == pytest.approx(abs(long) - 0.30, abs=1e-6)


@pytest.mark.parametrize("lat,expect_in", [(0.29, True), (-0.29, True), (0.31, False), (-0.31, False)])
def test_lateral_boundary(lat, expect_in):
    env = make_env()
    place(env, lat=lat)
    assert env.inside_bay() is expect_in


def test_heading_error_swings_the_corners_out():
    # With zero offset the footprint half-width is 0.9cos(a)+2.1sin(a): 8 deg -> 1.183 (< 1.2, inside),
    # 9 deg -> 1.2175 (> 1.2, protrudes ~1.7 cm), 15 deg (the legacy heading tolerance) -> ~1.41 (0.21 m out).
    env = make_env()
    place(env, hdg_deg=8.0)
    assert env.inside_bay()
    place(env, hdg_deg=9.0)
    assert env.bay_protrusion() == pytest.approx(0.9 * np.cos(np.deg2rad(9)) + 2.1 * np.sin(np.deg2rad(9)) - 1.2, abs=1e-6)
    place(env, hdg_deg=-15.0)
    assert 0.20 < env.bay_protrusion() < 0.22


def test_corner_protrusion_with_pose_inside_legacy_tolerance():
    # centre 0.25 m off and 5 deg rotated: the legacy criterion is satisfied, but a corner is ~13 cm outside
    env = make_env()
    place(env, lat=0.25, hdg_deg=5.0)
    assert np.hypot(*env.slot_frame_error()) < env.pos_tol
    assert env.bay_protrusion() > 0.10


def _hold(env, n=None):
    info = None
    for _ in range(n or env.hold_steps + 1):
        _, _, term, trunc, info = env.step(np.array([0.0, -1.0], dtype=np.float32))
        if term or trunc:
            break
    return info


def test_pose_mode_accepts_what_contained_mode_rejects():
    legacy, strict = make_env("pose"), make_env("contained")
    for env in (legacy, strict):
        place(env, long=-0.35, lat=0.05)           # 35 cm short: pose-OK (0.354 m < 0.40), rear overhangs 5 cm
    assert not legacy.inside_bay()
    i_legacy, i_strict = _hold(legacy), _hold(strict)
    assert i_legacy["success"] is True and i_legacy["pose_ok"] is True
    assert i_strict["success"] is False and i_strict["pose_ok"] is True and i_strict["contained"] is False


def test_contained_mode_succeeds_when_inside():
    env = make_env("contained")
    place(env, long=-0.05, lat=0.03, hdg_deg=2.0)
    info = _hold(env)
    assert info["success"] is True and info["contained"] is True and info["bay_protrusion"] == 0.0


def test_contained_mode_needs_the_hold_steps():
    env = make_env("contained")
    place(env)
    _, _, term, _, info = env.step(np.array([0.0, -1.0], dtype=np.float32))
    assert info["success"] is False and term is False       # hold_steps == 2


@pytest.mark.parametrize("mode", ["pose", "contained"])
def test_collision_on_terminal_step_is_not_success(mode):
    # centre 0.39 m off (< 0.40 m) and 12 deg: pose criterion satisfied, but a corner is inside a neighbour
    env = make_env(mode)
    place(env, lat=0.39, hdg_deg=12.0)
    assert env._check_collision()
    info = _hold(env)
    assert info["pose_ok"] is True and info["collision"] is True and info["success"] is False


def test_slot_frame_error_signs():
    env = make_env()
    place(env, long=0.2, lat=-0.1)
    long, lat = env.slot_frame_error()
    assert long == pytest.approx(0.2, abs=1e-9) and lat == pytest.approx(-0.1, abs=1e-9)


def test_summary_completion_and_error_fields():
    from evaluation.metrics import summarize
    df = pd.DataFrame({
        "success": [True, False, False, True], "collision": [False, True, False, False],
        "timeout": [False, False, True, False], "time_s": [5, 2, 50, 6], "pos_error_m": [.1, .5, .3, .2],
        "heading_error_deg": [1, 2, 3, 4], "trajectory_length_m": [3, 3, 3, 3], "smoothness": [0.] * 4,
        "mean_abs_accel": [0.] * 4, "mean_brake": [0.] * 4, "reward": [1.] * 4,
        "in_bay": [True, False, False, False], "bay_protrusion_m": [0, .2, .3, .05],
        "long_error_m": [-.1, .2, -.3, .1], "lat_error_m": [.05, .1, .0, -.05], "success_mode": ["pose"] * 4})
    m = summarize(df)
    assert m["completion_rate"] == pytest.approx(0.5)
    assert m["strict_in_bay_rate"] == pytest.approx(0.25)
    assert m["final_long_error_abs_m"] == pytest.approx((.1 + .3 + .1) / 3)      # collisions excluded
    assert m["success_mode"] == "pose"


def test_legacy_results_still_reproduce():
    """The pose-mode outcome of the original controller must not be silently redefined: the first
    episodes of the stored audit_evaluation CSV (seeds 3000..) are re-run and compared."""
    from pathlib import Path
    from evaluation.run_experiments import FuzzyOnlyAdapter
    csv = Path(__file__).resolve().parent.parent / "experiments" / "audit_evaluation" / "fuzzy_only.csv"
    if not csv.is_file():
        pytest.skip("stored evaluation CSV not present")
    ref = pd.read_csv(csv).head(8)
    env = ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION, "scenario_kinds": [Kind.PERPENDICULAR]})
    ad = FuzzyOnlyAdapter()
    for i, row in ref.iterrows():
        ad.reset()
        obs, _ = env.reset(seed=3000 + i)
        done = False
        while not done:
            a, _ = ad.act(obs, env)
            obs, _, te, tr, info = env.step(a)
            done = te or tr
        assert (info["success"], info["collision"], info["timeout"]) == (bool(row.success), bool(row.collision), bool(row.timeout))


def test_precise_variant_only_changes_stop_radius():
    from evaluation.run_experiments import PRECISE_STOP_RADIUS, make_adapter
    assert make_adapter("fuzzy_only").controller.stop_radius == pytest.approx(0.32)
    assert make_adapter("fuzzy_precise").controller.stop_radius == pytest.approx(PRECISE_STOP_RADIUS)
    assert PRECISE_STOP_RADIUS < 0.30      # must be inside the 0.30 m end clearance of the bay
