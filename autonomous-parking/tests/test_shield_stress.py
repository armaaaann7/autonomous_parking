"""Independent stress tests of the braking-envelope shield (audit P4).

They verify the shield's logic against the simulator's own dynamics. They do NOT, and cannot, show
real-world safety: see test_guarantee_breaks_under_brake_mismatch for the documented limit.
"""
import numpy as np
import pytest

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from integration.envelope_shield import BrakingEnvelopeShield


def new_env(**kw):
    return ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION, "scenario_kinds": [Kind.PERPENDICULAR], **kw})


def adversarial_run(n, steps, brake_factor=1.0, seed=0, speeds=(0.0, 0.5, 1.0, 2.0)):
    rng = np.random.default_rng(seed)
    env = new_env()
    env.vehicle.max_brake *= brake_factor
    sh = BrakingEnvelopeShield()
    collisions, ran, worst_ge_margin = 0, 0, 9.0
    for ep in range(n):
        env.reset(seed=20_000 + ep)
        sx, sy, _ = env.scenario.slot_pose
        ok = False
        for _ in range(50):
            env.vehicle.reset(sx + rng.uniform(-5, 5), sy + rng.uniform(-6, 3), rng.uniform(-np.pi, np.pi))
            env.vehicle.v = float(rng.choice(speeds))
            if not env._check_collision() and env.body_clearance() > 0:
                ok = True
                break
        if not ok or not env.braking_envelope_safe(np.array([0.0, -1.0]), margin=0.0)[0]:
            continue
        sh.reset()
        ran += 1
        start_ge = env.body_clearance() >= sh.margin
        ep_min = 9.0
        for _ in range(steps):
            _, _, _, _, info = env.step(sh.filter(env, np.array([rng.uniform(-1, 1), 1.0], dtype=np.float32)))
            ep_min = min(ep_min, env.body_clearance())
            if info["collision"]:
                collisions += 1
                break
        if start_ge:
            worst_ge_margin = min(worst_ge_margin, ep_min)
    return ran, collisions, worst_ge_margin


def test_adversarial_floor_it_policy_never_collides_from_safe_starts():
    ran, coll, _ = adversarial_run(60, 100)
    assert ran >= 40 and coll == 0


def test_margin_is_kept_when_the_start_respects_it():
    ran, coll, worst = adversarial_run(60, 100, seed=3)
    assert coll == 0 and worst >= 0.05 - 1e-6


def test_guarantee_breaks_under_brake_mismatch():
    """Documented LIMIT: if the real car brakes at 50 % of the modelled strength the shield no longer
    prevents collisions. This test pins the limitation so nobody claims universal safety."""
    ran, coll, _ = adversarial_run(60, 100, brake_factor=0.5, seed=1)
    assert coll > 0


@pytest.mark.parametrize("bad", [[np.nan, 0.5], [0.0, np.nan], [np.inf, np.inf], [-np.inf, 0.3]])
def test_non_finite_action_is_never_executed(bad):
    env = new_env()
    env.reset(seed=5)
    sh = BrakingEnvelopeShield()
    out = sh.filter(env, np.array(bad, dtype=np.float32))
    assert np.isfinite(out).all() and out[1] == -1.0 and sh.last_intervened


@pytest.mark.parametrize("shape", [(1,), (3,), (2, 1)])
def test_wrong_shape_raises(shape):
    env = new_env()
    env.reset(seed=5)
    with pytest.raises(ValueError):
        BrakingEnvelopeShield().filter(env, np.zeros(shape))


def test_out_of_range_finite_actions_saturate():
    env = new_env()
    env.reset(seed=5)
    env.scenario.obstacles = np.zeros((0, 4))
    out = BrakingEnvelopeShield().filter(env, np.array([50.0, -50.0]))
    assert out.tolist() == [1.0, -1.0]


def test_reverse_mode_is_rejected_not_silently_unsafe():
    env = ParkingEnv({"allow_reverse": True, "scenario_kinds": [Kind.PARALLEL]})
    env.reset(seed=1)
    with pytest.raises(NotImplementedError):
        BrakingEnvelopeShield().filter(env, np.array([0.0, 0.5]))


def test_forward_only_env_never_reverses():
    rng = np.random.default_rng(0)
    env = new_env()
    env.reset(seed=2)
    vmin = 0.0
    for _ in range(300):
        env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        vmin = min(vmin, env.vehicle.v)
    assert vmin >= 0.0


@pytest.mark.parametrize("gap", [0.8, 1.5, 3.0])
def test_high_speed_head_on_approach_is_stopped_short(gap):
    """Car at 2 m/s heading straight at the front face of a neighbour; the full-throttle command must be
    overridden early enough that the car stops with >= margin to spare (stopping distance ~0.45 m)."""
    env = new_env()
    env.reset(seed=7)
    env.scenario.obstacles = np.array([[-0.9, 5.0, 0.9, 7.0]])
    env.vehicle.reset(0.0, 5.0 - gap - 2.1, np.pi / 2)         # nose 'gap' metres from the obstacle face
    env.vehicle.v = 2.0
    sh = BrakingEnvelopeShield()
    safe_start = env.braking_envelope_safe(np.array([0.0, -1.0]), margin=0.0)[0]
    for _ in range(60):
        _, _, _, _, info = env.step(sh.filter(env, np.array([0.0, 1.0], dtype=np.float32)))
        assert not info["collision"] or not safe_start
    if safe_start:
        assert env.body_clearance() >= 0.0 and env.vehicle.v == 0.0


def test_action_safe_now_but_unsafe_next_step_is_blocked():
    # clearance 0.2 m, speed 1 m/s toward the wall: full throttle moves the car ~0.11 m (collision-free this very
    # step), but the braking tail afterwards (~0.1 m more) eats the 0.05 m margin -> the shield must refuse.
    env = new_env()
    env.reset(seed=7)
    env.scenario.obstacles = np.array([[-0.9, 5.0, 0.9, 7.0]])
    env.vehicle.reset(0.0, 5.0 - 0.2 - 2.1, np.pi / 2)
    env.vehicle.v = 1.0
    assert env.body_clearance() == pytest.approx(0.2, abs=1e-6)
    probe = env.braking_envelope_safe(np.array([0.0, 1.0]), margin=0.0)
    assert probe[0] is False and probe[1] > 1          # step 1 itself is collision-free; the stopping tail is not
    sh = BrakingEnvelopeShield()
    out = sh.filter(env, np.array([0.0, 1.0], dtype=np.float32))
    assert sh.last_intervened and out[1] == -1.0


def test_shield_is_a_noop_when_far_from_everything():
    env = new_env()
    env.reset(seed=7)
    env.scenario.obstacles = np.zeros((0, 4))
    sh = BrakingEnvelopeShield()
    a = np.array([0.3, 0.7], dtype=np.float32)
    np.testing.assert_allclose(sh.filter(env, a), a)
    assert not sh.last_intervened


def test_interpolated_poses_between_steps_do_not_overlap_obstacles():
    from environment.geometry import obb_aabb_overlap, obb_corners
    from evaluation.run_experiments import make_adapter
    env = new_env()
    ad = make_adapter("fuzzy_precise+env")
    hits = 0
    for sd in range(9000, 9012):
        ad.reset()
        obs, _ = env.reset(seed=sd)
        prev, done = (env.vehicle.x, env.vehicle.y, env.vehicle.theta), False
        while not done:
            a, _ = ad.act(obs, env)
            obs, _, te, tr, info = env.step(a)
            cur = (env.vehicle.x, env.vehicle.y, env.vehicle.theta)
            for f in np.linspace(0, 1, 11)[1:-1]:
                x, y = prev[0] + f * (cur[0] - prev[0]), prev[1] + f * (cur[1] - prev[1])
                th = prev[2] + f * np.arctan2(np.sin(cur[2] - prev[2]), np.cos(cur[2] - prev[2]))
                hits += int(obb_aabb_overlap(obb_corners(x, y, th, env.vehicle.length, env.vehicle.width), env.scenario.obstacles).any())
            prev, done = cur, te or tr
    assert hits == 0
