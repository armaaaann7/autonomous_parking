"""Slot-relative observation (obs_mode='slot'): frames, angle wrapping, leakage, legacy compatibility."""
import numpy as np
import pytest

from environment.observations import CLEAR_CAP, SLOT_OBS_DIM, slot_observation, slot_relative_features
from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel


def _env(mode, **kw):
    return ParkingEnv({"obs_mode": mode, "scenario_kinds": [Kind.PERPENDICULAR],
                       "randomization": RandomizationLevel.SLOT_POSITION, **kw})


def test_legacy_default_unchanged_and_slot_dimensions():
    e = ParkingEnv({"scenario_kinds": [Kind.PERPENDICULAR]})
    o, _ = e.reset(seed=4)
    assert e.obs_mode == "legacy" and o.shape == (14,) and e.observation_space.shape == (14,)
    s = _env("slot")
    o2, _ = s.reset(seed=4)
    assert o2.shape == (SLOT_OBS_DIM,) == s.observation_space.shape
    assert s.observation_space.contains(o2)
    with pytest.raises(ValueError):
        ParkingEnv({"obs_mode": "nonsense"})


def test_legacy_obs_is_identical_in_both_modes_for_the_same_seed_and_actions():
    a, b = _env("legacy"), _env("slot")
    oa, _ = a.reset(seed=11); b.reset(seed=11)
    np.testing.assert_array_equal(oa, b.legacy_obs())
    rng = np.random.default_rng(0)
    for _ in range(40):
        act = rng.uniform(-1, 1, 2).astype(np.float32)
        oa, ra, ta, tra, _ = a.step(act)
        ob, rb, tb, trb, _ = b.step(act)
        np.testing.assert_array_equal(oa, b.legacy_obs())       # same RNG stream, same sensors, same dynamics
        assert ra == rb and ta == tb and tra == trb
        if ta or tra:
            break


@pytest.mark.parametrize("slot_theta", [0.0, np.pi / 2, 0.7, -2.5])
def test_pose_features_match_independent_geometry(slot_theta):
    rng = np.random.default_rng(1)
    for _ in range(50):
        slot = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), slot_theta])
        xy, th = rng.uniform(-8, 8, 2), rng.uniform(-np.pi, np.pi)
        f = slot_relative_features(xy, th, slot)
        # independent derivation with complex numbers: rotate the offset into each frame
        off = complex(*(xy - slot[:2]))
        in_slot = off * np.exp(-1j * slot_theta)                  # slot frame (x_s along the slot axis)
        in_car = complex(*(slot[:2] - xy)) * np.exp(-1j * th)     # car frame (x forward, y left)
        assert f[0] == pytest.approx(in_slot.real, abs=1e-5) and f[1] == pytest.approx(in_slot.imag, abs=1e-5)
        assert f[2] == pytest.approx(in_car.real, abs=1e-5) and f[3] == pytest.approx(in_car.imag, abs=1e-5)
        assert f[4] == pytest.approx(abs(off), abs=1e-5)
        assert (f[5], f[6]) == pytest.approx((np.sin(th - slot_theta), np.cos(th - slot_theta)), abs=1e-6)


def test_pose_features_are_invariant_to_rigid_motion_of_the_whole_world():
    rng = np.random.default_rng(2)
    for _ in range(50):
        slot = np.array([*rng.uniform(-3, 3, 2), rng.uniform(-3, 3)])
        xy, th = rng.uniform(-8, 8, 2), rng.uniform(-3, 3)
        phi, t = rng.uniform(-3, 3), rng.uniform(-5, 5, 2)
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        slot2 = np.array([*(R @ slot[:2] + t), slot[2] + phi])
        f1 = slot_relative_features(xy, th, slot)
        f2 = slot_relative_features(R @ xy + t, th + phi, slot2)
        np.testing.assert_allclose(f1, f2, atol=1e-5)


def test_goal_pose_and_angle_wrapping():
    slot = np.array([1.0, -2.0, np.pi / 2])
    g = slot_relative_features(slot[:2], np.pi / 2, slot)
    np.testing.assert_allclose(g, [0, 0, 0, 0, 0, 0, 1], atol=1e-6)
    eps = 1e-3
    a = slot_relative_features([0, 0], slot[2] + np.pi - eps, slot)     # just under +pi
    b = slot_relative_features([0, 0], slot[2] - np.pi + eps, slot)     # just over -pi: physically the same heading
    np.testing.assert_allclose(a[5:7], b[5:7], atol=3 * eps)            # no +-pi jump (sin/cos are continuous)
    c = slot_relative_features([0, 0], slot[2] + 0.3 + 2 * np.pi, slot)
    d = slot_relative_features([0, 0], slot[2] + 0.3, slot)
    np.testing.assert_allclose(c, d, atol=1e-5)                         # full-turn invariance


def test_no_leakage_of_success_state_or_distant_obstacles():
    e = _env("slot")
    o1, _ = e.reset(seed=7)
    # success tolerances / mode / curriculum counters are not observable
    e.pos_tol, e.heading_tol, e.hold_steps = 9.0, 3.0, 99
    e.success_mode = "contained"
    assert np.array_equal(o1, e._get_obs())
    # obstacles beyond the sensor range and clearance cap do not change the observation
    e.scenario.obstacles = e.scenario.obstacles + np.array([300.0, 300.0, 300.0, 300.0])
    far = e._get_obs()
    np.testing.assert_allclose(far[9:17], e.sensor_range)               # rays saturate
    assert far[17] == pytest.approx(CLEAR_CAP)
    assert np.array_equal(far[:9], o1[:9])


def test_clearance_feature_matches_body_clearance_and_is_capped():
    e = _env("slot")
    e.reset(seed=3)
    for _ in range(25):
        o, *_ = e.step(np.array([0.2, 0.6], dtype=np.float32))
        assert o[17] == pytest.approx(min(max(e.body_clearance(), 0.0), CLEAR_CAP), abs=1e-5)
    assert slot_observation([0, 0], 0.0, [0, 0, 0], 0, 0, np.zeros(8), 99.0)[17] == CLEAR_CAP
    assert slot_observation([0, 0], 0.0, [0, 0, 0], 0, 0, np.zeros(8), -1.0)[17] == 0.0


def test_slot_policy_dimension_guard(tmp_path):
    from rl import train_ppo
    from rl.policy_wrapper import PPOPolicy
    out = str(tmp_path / "slot_smoke")
    try:
        train_ppo.ENV_EXTRA.clear(); train_ppo.VAL_EXTRA.clear()
        train_ppo.ENV_EXTRA.update({"scenario_kinds": [Kind.PERPENDICULAR], "obs_mode": "slot"})
        train_ppo.VAL_EXTRA.update({"success_mode": "contained"})
        train_ppo.train(1024, out, True, seed=0, chunk=1024, val_seed=7000, val_episodes=2,
                        overrides={"net_arch": [16, 16], "n_envs": 1, "n_steps": 512, "batch_size": 128}, select="safe")
        pol = PPOPolicy(out)
        assert pol.obs_mode == "slot" and pol.env_extra["obs_mode"] == "slot"
        assert pol.act(np.zeros(18, dtype=np.float32)).shape == (2,)
        with pytest.raises(ValueError):
            pol.act(np.zeros(14, dtype=np.float32))               # a legacy vector must be rejected
        import json
        assert json.loads(open(out + ".meta.json").read())["val_extra"] == {"success_mode": "contained"}
        from evaluation.run_experiments import make_adapter
        ad = make_adapter("ppo:" + out)
        assert ad.env_extra["obs_mode"] == "slot"
    finally:
        train_ppo.ENV_EXTRA.clear(); train_ppo.VAL_EXTRA.clear()
        train_ppo.RUN_CFG.clear(); train_ppo.RUN_CFG.update(obs_scale=False)


def test_legacy_checkpoint_still_loads_and_runs_through_the_new_evaluator(tmp_path):
    from evaluation.run_experiments import make_adapter, run_episodes
    from pathlib import Path
    ad = make_adapter("ppo_v3+env")
    assert ad.env_extra == {}                                   # legacy, forward-only: nothing forced on the env
    out = tmp_path / "x.csv"
    run_episodes(ad, 2, [9000, 9001], [Kind.PERPENDICULAR], out, env_extra={"success_mode": "contained"})
    assert out.read_text().count("\n") == 3
