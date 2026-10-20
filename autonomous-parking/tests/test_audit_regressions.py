"""Regression and invariant tests added by the presentation-readiness audit.

Expected values are derived independently of the implementation wherever possible (shapely
geometry, closed-form kinematics, hand-computed statistics) instead of re-stating the code.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from environment.geometry import obb_aabb_distance, obb_corners, wrap_angle
from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel, slot_rectangle
from environment.vehicle import Vehicle
from integration.envelope_shield import BrakingEnvelopeShield
from integration.hybrid_controller import HybridController

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "ppo_only_best"


def _env(**kw):
    cfg = {"scenario_kinds": [Kind.PERPENDICULAR], "randomization": RandomizationLevel.SLOT_POSITION}
    cfg.update(kw)
    return ParkingEnv(cfg)


def _shapely_overlap(env):
    """Independent collision oracle: positive-area intersection using shapely."""
    car = Polygon(env.vehicle.corners())
    return any(car.intersection(box(*o)).area > 1e-9 for o in env.scenario.obstacles)


# ------------------------------------------------------------------ geometry / clearance
def test_obb_aabb_distance_matches_shapely_on_random_poses():
    rng = np.random.default_rng(7)
    for _ in range(300):
        c = obb_corners(*rng.uniform(-6, 6, 2), rng.uniform(-np.pi, np.pi), 4.2, 1.8)
        boxes = np.array([[-1.5 + dx, -2.1, -0.6 + dx, 2.1] for dx in rng.uniform(-3, 3, 3)])
        got = obb_aabb_distance(c, boxes)
        want = [Polygon(c).distance(box(*b)) for b in boxes]
        assert np.allclose(got, want, atol=1e-9)


@pytest.mark.parametrize("a", [-7.0, -np.pi, -0.1, 0.0, np.pi - 1e-9, np.pi, 3.5 * np.pi, 100.0])
def test_wrap_angle_range_and_equivalence(a):
    w = float(wrap_angle(a))
    assert -np.pi <= w < np.pi + 1e-12
    assert np.cos(w) == pytest.approx(np.cos(a)) and np.sin(w) == pytest.approx(np.sin(a), abs=1e-9)


def test_vehicle_positive_steer_turns_counter_clockwise_with_closed_form_rate():
    v = Vehicle(dict(length=4.2, width=1.8, wheelbase=2.5, rear_overhang=0.85, max_speed=2.0,
                     max_steer_deg=35.0, steer_rate=100.0, max_accel=2.0, max_brake=4.0, drag=0.0), dt=0.1)
    v.reset(0, 0, 0)
    v.v = 1.0
    v.step(1.0, 0.0)                    # steering instantly at +35 deg, speed stays 1.0 (no drag, no accel)
    expected = 1.0 / 2.5 * np.tan(np.deg2rad(35.0)) * 0.1
    assert v.theta == pytest.approx(expected, rel=1e-9) and v.theta > 0


# ---------------------------------------------------------------- success / reward semantics
def test_success_is_never_reported_on_a_colliding_step():
    env = _env(randomization=RandomizationLevel.NONE)
    env.reset(seed=0)
    sx, sy, sth = env.scenario.slot_pose
    # a deliberately wrong obstacle sitting on the slot centre: the pose is inside the success
    # tolerance, but the car overlaps the obstacle on every step
    env.scenario.obstacles = np.array([[sx - 0.5, sy - 0.5, sx + 0.5, sy + 0.5]])
    env.vehicle.reset(sx, sy, sth)
    infos = [env.step(np.zeros(2, dtype=np.float32))[4] for _ in range(env.hold_steps + 1)]   # long enough for the streak
    assert all(i["collision"] for i in infos) and not any(i["success"] for i in infos)


def test_timeout_penalty_not_applied_to_a_successful_final_step():
    env = _env()
    env.reset(seed=1)
    env._prev_dist, env._prev_heading_err = 0.0, 0.0
    env.vehicle.v = 0.0
    with_to = env._reward(0.0, 0.0, 0.0, False, True, True)
    without = env._reward(0.0, 0.0, 0.0, False, True, False)
    assert with_to == pytest.approx(without)
    assert env._reward(1.0, 0.0, 0.0, False, False, True) < env._reward(1.0, 0.0, 0.0, False, False, False)


def test_bay_protrusion_is_zero_when_centred_and_matches_closed_form_when_shifted():
    env = _env(randomization=RandomizationLevel.NONE)
    env.reset(seed=0)
    sx, sy, sth = env.scenario.slot_pose
    env.vehicle.reset(sx, sy, sth)
    assert env.bay_protrusion() == 0.0 and env.inside_bay()
    # bay is car length + 0.6 = 4.8 m long, so 0.3 m of slack each end; shifting 0.5 m leaves 0.2 m out
    env.vehicle.reset(sx, sy + 0.5, sth)
    assert env.bay_protrusion() == pytest.approx(0.2, abs=1e-9)
    r = slot_rectangle(env.scenario)
    assert (r[3] - r[1]) == pytest.approx(4.8) and (r[2] - r[0]) == pytest.approx(2.4)


# ------------------------------------------------------------------------------- safety shield
def test_old_centre_ray_shield_input_is_blind_to_a_car_that_is_about_to_clip_a_neighbour():
    """Documents defect D1: with an obstacle 0.05 m from the car's flank the centre-ray distance
    is ~0.95 m (>= the 1.0 m threshold margin) whereas true clearance is 0.05 m."""
    env = _env(randomization=RandomizationLevel.NONE)
    env.reset(seed=0)
    sx, sy, sth = env.scenario.slot_pose
    env.vehicle.reset(sx, sy, sth)
    env.scenario.obstacles = np.array([[sx + 0.95, sy - 3, sx + 2.0, sy + 3]])   # 0.05 m from the flank
    d_ray, _, _ = env.fuzzy_inputs()
    d_true, _, _ = env.shield_inputs()
    assert d_true == pytest.approx(0.05, abs=1e-9)
    assert d_ray == pytest.approx(0.95, abs=1e-9)
    # body-front ray: obstacle ahead 0.1 m from the bumper -> centre ray says 2.2 m
    env.scenario.obstacles = np.array([[sx - 2, sy + 2.2, sx + 2, sy + 4]])
    d_ray, _, _ = env.fuzzy_inputs()
    d_true, _, _ = env.shield_inputs()
    assert d_ray == pytest.approx(2.2) and d_true == pytest.approx(0.1, abs=1e-9)


def test_mamdani_shield_with_true_clearance_activates_before_contact():
    env = _env(randomization=RandomizationLevel.NONE)
    env.reset(seed=0)
    sx, sy, sth = env.scenario.slot_pose
    env.vehicle.reset(sx, sy, sth)
    env.scenario.obstacles = np.array([[sx - 2, sy + 2.2, sx + 2, sy + 4]])      # 0.1 m ahead of bumper
    shield = HybridController(safe_threshold=1.0)
    d, b, v = env.shield_inputs()
    out = shield.act(np.array([0.0, 1.0]), d, b, v)
    assert shield.last_shield_active and out[1] < 1.0


def test_shield_fails_safe_on_non_finite_inputs():
    shield = HybridController()
    out = shield.act(np.array([0.2, 0.9]), float("nan"), 0.0, 1.0)
    assert out[1] == -1.0 and shield.last_shield_active and np.isfinite(out).all()
    from fuzzy.fuzzy_system import FuzzySystem
    with pytest.raises(ValueError):
        FuzzySystem().compute(float("nan"), 0.0, 1.0)


def test_envelope_shield_brakes_when_full_throttle_would_hit_obstacle_and_passes_safe_actions():
    env = _env(randomization=RandomizationLevel.NONE)
    env.reset(seed=0)
    sx, sy, sth = env.scenario.slot_pose
    env.vehicle.reset(sx, sy - 2.5, sth)       # bumper 0.4 m from the wall; stopping from 2 m/s needs ~0.64 m
    env.scenario.obstacles = np.array([[sx - 2, sy + 0.0, sx + 2, sy + 1.0]])    # wall dead ahead
    env.vehicle.v = 2.0
    shield = BrakingEnvelopeShield()
    a = shield.filter(env, np.array([0.0, 1.0]))
    assert shield.last_intervened and a[1] == -1.0
    # an already-stopped car far from the wall may do anything
    env.vehicle.reset(sx, sy - 5.0, sth)
    env.vehicle.v = 0.0
    b = shield.filter(env, np.array([0.3, 0.5]))
    assert not shield.last_intervened and np.allclose(b, [0.3, 0.5])


def test_envelope_shield_disabled_is_identity():
    env = _env()
    env.reset(seed=3)
    s = BrakingEnvelopeShield(enabled=False)
    assert np.allclose(s.filter(env, np.array([1.0, 1.0])), [1.0, 1.0]) and not s.last_intervened


def test_envelope_invariant_no_collision_under_adversarial_random_and_full_throttle_policies():
    """Safety property of the MODEL: whatever the policy asks, the shielded car never touches an
    obstacle (checked with an independent shapely oracle, not only the env's own flag)."""
    rng = np.random.default_rng(0)
    shield = BrakingEnvelopeShield()
    n_int = 0
    for ep in range(40):
        env = _env(randomization=RandomizationLevel.OBSTACLES)
        env.reset(seed=ep)
        shield.reset()
        policy = ["random", "throttle_left", "throttle_right", "throttle_straight"][ep % 4]
        for _ in range(300):
            a = {"random": rng.uniform(-1, 1, 2), "throttle_left": [1.0, 1.0],
                 "throttle_right": [-1.0, 1.0], "throttle_straight": [0.0, 1.0]}[policy]
            _, _, term, trunc, info = env.step(shield.filter(env, np.asarray(a, dtype=np.float32)))
            assert not info["collision"], f"collision in episode {ep} ({policy})"
            assert not _shapely_overlap(env)
            if term or trunc:
                break
        n_int += shield.n_interventions
    assert n_int > 0, "the adversarial policies must actually have triggered the shield"


def test_envelope_shield_does_not_exploit_hidden_state():
    """The scratch simulation must not mutate the real vehicle."""
    env = _env()
    env.reset(seed=5)
    before = (env.vehicle.x, env.vehicle.y, env.vehicle.theta, env.vehicle.v, env.vehicle.delta)
    env.braking_envelope_safe(np.array([1.0, 1.0]))
    after = (env.vehicle.x, env.vehicle.y, env.vehicle.theta, env.vehicle.v, env.vehicle.delta)
    assert before == after


def test_adapter_reports_real_shield_activity_not_a_constant():
    from evaluation.run_experiments import PPOPlusShieldAdapter
    from fuzzy.fuzzy_only_controller import FuzzyOnlyController

    class Policy:
        def act(self, obs):
            return np.array([0.0, 0.5])

    ad = PPOPlusShieldAdapter.__new__(PPOPlusShieldAdapter)
    ad.policy, ad.hybrid, ad.recovery = Policy(), HybridController(), FuzzyOnlyController()
    ad.reset()
    env = _env(randomization=RandomizationLevel.NONE)
    obs, _ = env.reset(seed=0)
    env.scenario.obstacles = np.zeros((0, 4))     # nothing nearby -> shield must stay idle
    _, flag_far = ad.act(obs, env)
    sx, sy, sth = env.scenario.slot_pose
    env.vehicle.reset(sx, sy, sth)
    env.scenario.obstacles = np.array([[sx + 0.95, sy - 3, sx + 2.0, sy + 3]])
    obs = env._get_obs()
    _, flag_near = ad.act(obs, env)
    assert flag_far == 0.0 and flag_near == 1.0


# ------------------------------------------------------------------------------------ metrics
def test_wilson_interval_known_values():
    from evaluation.metrics import wilson_interval
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and hi == pytest.approx(0.03699, abs=2e-4)
    lo, hi = wilson_interval(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=2e-4)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_mcnemar_exact_hand_computed():
    from evaluation.metrics import mcnemar_exact
    a = np.array([1] * 8 + [0] * 2 + [1] * 5 + [0] * 5, dtype=bool)
    b = np.array([0] * 8 + [1] * 2 + [1] * 5 + [0] * 5, dtype=bool)
    na, nb, p = mcnemar_exact(a, b)
    assert (na, nb) == (8, 2)
    assert p == pytest.approx(2 * (1 + 10 + 45) / 1024)       # 0.109375 by hand
    assert mcnemar_exact(a, a)[2] == 1.0


def test_summarize_reports_success_only_means_and_failures_are_kept():
    import pandas as pd
    from evaluation.metrics import summarize
    df = pd.DataFrame({
        "success": [True, True, False, False], "collision": [False, False, True, False],
        "timeout": [False, False, False, True], "time_s": [10., 20., 5., 50.],
        "pos_error_m": [0.1, 0.3, 2.0, 3.0], "heading_error_deg": [1., 3., 50., 60.],
        "trajectory_length_m": [1., 1., 1., 1.], "smoothness": [0.] * 4, "mean_abs_accel": [0.] * 4,
        "mean_brake": [0.] * 4, "reward": [100., 100., -20., -10.], "in_bay": [True, False, False, False],
        "bay_protrusion_m": [0., 0.1, 1., 2.], "min_clearance_m": [0.5, 0.2, 0., 1.]})
    s = summarize(df)
    assert s["success_rate"] == 0.5 and s["collision_rate"] == 0.25 and s["timeout_rate"] == 0.25
    assert s["avg_time_s"] == pytest.approx(15.0) and s["pos_error_m"] == pytest.approx(0.2)
    assert s["strict_in_bay_rate"] == 0.25 and s["in_bay_given_success"] == 0.5
    assert s["mean_reward"] == pytest.approx(42.5)


# ---------------------------------------------------------------------- evaluation integration
def test_run_episodes_csv_schema_determinism_and_no_success_with_collision(tmp_path):
    import csv
    from evaluation.run_experiments import FIELDS, FuzzyOnlyAdapter, run_episodes
    paths = []
    for k in range(2):
        out = tmp_path / f"r{k}.csv"
        run_episodes(FuzzyOnlyAdapter(), 4, [1000, 1001, 1002, 1003], [Kind.PERPENDICULAR], out)
        paths.append(out)
    rows = [list(csv.DictReader(open(p))) for p in paths]
    assert list(rows[0][0].keys()) == FIELDS
    assert rows[0] == rows[1]                                  # same seeds -> identical results
    for r in rows[0]:
        assert not (r["success"] == "True" and r["collision"] == "True")
        assert sum(r[k] == "True" for k in ("success", "collision", "timeout")) <= 1


def test_trajectory_export_is_consistent_with_the_episode_it_recorded():
    from evaluation.run_experiments import FuzzyOnlyAdapter
    from export_trajectory import run_episode
    d = run_episode(FuzzyOnlyAdapter(), 1000, Kind.PERPENDICULAR, "fuzzy_only")
    d2 = json.loads(json.dumps(d))
    assert d2["seed"] == 1000 and "recorded" in d2["source"] and len(d2["slot"]) == 3
    assert len(d2["frames"]) >= 2 and all(np.isfinite(list(f[k] for k in ("x", "y", "theta", "v"))).all() for f in d2["frames"])
    last, slot = d2["frames"][-1], d2["slot"]
    assert d2["result"]["pos_error"] == pytest.approx(np.hypot(last["x"] - slot[0], last["y"] - slot[1]), abs=1e-6)
    assert [f["collision"] for f in d2["frames"]].count(True) <= 1
    # Strict ("contained") mode: success needs the pose criterion AND the whole car inside the bay lines.
    # A car that stops within pose tolerance but outside the bay is therefore neither success, collision nor timeout.
    r = d2["result"]
    assert d2["success_mode"] == "contained"
    assert r["success"] == (d2["outcome"] == "parked inside the bay")
    assert not (r["success"] and (r["collision"] or r["timeout"]))
    if r["success"]:
        assert d2["in_bay"] is True
    if not (r["success"] or r["collision"] or r["timeout"]):
        assert d2["pose_success_step"] is not None and d2["in_bay"] is False


# ------------------------------------------------------------------------------- PPO checkpoint
@pytest.mark.skipif(not MODEL.with_suffix(".zip").is_file(), reason="PPO checkpoint not present")
def test_ppo_checkpoint_loads_and_acts_within_bounds_deterministically():
    from rl.policy_wrapper import PPOPolicy
    pol = PPOPolicy(str(MODEL))
    env = _env()
    obs, _ = env.reset(seed=1)
    a1, a2 = pol.act(obs), pol.act(obs)
    assert a1.shape == (2,) and np.allclose(a1, a2) and (np.abs(a1) <= 1.0 + 1e-6).all()


def test_ppo_policy_missing_or_incompatible_checkpoint_fails_clearly(tmp_path):
    from rl.policy_wrapper import PPOPolicy
    with pytest.raises(FileNotFoundError):
        PPOPolicy(str(tmp_path / "nope"))
    from evaluation.run_experiments import make_adapter
    with pytest.raises(FileNotFoundError):
        make_adapter("ppo_only", str(tmp_path / "nope"))
    with pytest.raises(ValueError):
        make_adapter("not_a_controller")
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class Tiny(gym.Env):
        observation_space = spaces.Box(-1, 1, (3,), np.float32)
        action_space = spaces.Box(-1, 1, (2,), np.float32)
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed); return np.zeros(3, np.float32), {}
        def step(self, a):
            return np.zeros(3, np.float32), 0.0, False, True, {}
    PPO("MlpPolicy", Tiny(), n_steps=8, batch_size=8, device="cpu").save(str(tmp_path / "wrong"))
    with pytest.raises(ValueError, match="different environment"):
        PPOPolicy(str(tmp_path / "wrong"))


# ------------------------------------------------------------------- reproducibility & fuzzy
def test_same_seed_gives_identical_fuzzy_episode_outcome():
    from evaluation.run_experiments import FuzzyOnlyAdapter
    from export_trajectory import run_episode
    a = run_episode(FuzzyOnlyAdapter(), 1007, Kind.PERPENDICULAR, "fuzzy_only")
    b = run_episode(FuzzyOnlyAdapter(), 1007, Kind.PERPENDICULAR, "fuzzy_only")
    assert a["frames"] == b["frames"] and a["result"] == b["result"]


def test_fuzzy_outputs_stay_in_universe_and_have_no_large_jumps():
    from fuzzy.fuzzy_system import FuzzySystem
    f = FuzzySystem()
    prev = None
    for d in np.arange(0, 10.01, 0.1):
        o = f.compute(d, 10.0, 1.0)
        assert 0.0 <= o["brake_cmd"] <= 1.0 and -1.0 <= o["steer_corr"] <= 1.0 and -1.0 <= o["speed_adj"] <= 1.0
        if prev is not None:
            assert abs(o["brake_cmd"] - prev["brake_cmd"]) < 0.1 and abs(o["speed_adj"] - prev["speed_adj"]) < 0.25
        prev = o
    # out-of-range inputs are clipped, not crashing
    assert np.isfinite(list(f.compute(-5, 500, 99).values())).all()


def test_envelope_margin_keeps_requested_clearance_and_is_ignored_when_already_inside_margin():
    env = _env(randomization=RandomizationLevel.NONE)
    env.reset(seed=0)
    sx, sy, sth = env.scenario.slot_pose
    env.vehicle.reset(sx, sy, sth)
    env.vehicle.v = 0.0
    env.scenario.obstacles = np.array([[sx + 0.93, sy - 3, sx + 2.0, sy + 3]])   # 0.03 m from the flank
    # already inside the 0.05 m margin: margin is ignored, a stationary car may stay put
    assert env.braking_envelope_safe(np.array([0.0, 0.0]), margin=0.05)[0]
    env.scenario.obstacles = np.array([[sx + 0.97, sy - 3, sx + 2.0, sy + 3]])   # 0.07 m clear
    env.vehicle.v = 1.0
    # driving straight keeps 0.07 m clearance (>= margin) -> safe; steering into the wall violates it
    assert env.braking_envelope_safe(np.array([0.0, 0.2]), margin=0.05)[0]
    assert not env.braking_envelope_safe(np.array([-1.0, 0.2]), margin=0.05)[0]
