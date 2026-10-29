"""Regression tests for the RL architecture / hyper-parameter / reward audit (docs/RL_AUDIT.md).

They pin: (1) the default reward is unchanged, (2) the optional containment shaping is potential-based
(cannot be farmed by dithering) and points the right way, (3) the observation-scaling convention is
applied consistently at training and deployment, (4) the collision-aware checkpoint-selection rule,
(5) the chunk-safe LR schedule, (6) the trainer's override/metadata plumbing, and (7) that the
promoted PPO v3 checkpoint loads, has the right dimensions and runs with the safety shield without
collisions on a small fixed set.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from rl.obs_scaling import OBS_SCALE, ScaledObs, scale_obs
from rl.train_ppo import LinearLR, checkpoint_score, is_better

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "models" / "ppo_v3_rlaudit"


def _env(**kw):
    cfg = {"scenario_kinds": [Kind.PERPENDICULAR], "randomization": RandomizationLevel.SLOT_POSITION}
    cfg.update(kw)
    return ParkingEnv(cfg)


# ---------------------------------------------------------------- reward
def test_default_reward_has_no_containment_term():
    env = _env()
    env.reset(seed=1)
    assert env.reward_cfg.get("w_contain", 0.0) == 0.0
    base = env._reward(2.0, 0.1, 0.0, False, False, False)
    env._prev_protrusion = 5.0
    # with w_contain == 0 the protrusion argument must not change anything
    assert env._reward(2.0, 0.1, 0.0, False, False, False, protrusion=0.0) == base


def test_containment_shaping_is_potential_based_and_signed():
    env = _env(reward={"w_contain": 1.0})
    env.reset(seed=1)
    env._prev_dist = 2.0; env._prev_heading_err = 0.0
    env._prev_protrusion = 1.0
    gain = env._reward(2.0, 0.0, 0.0, False, False, False, protrusion=0.4) - env._reward(2.0, 0.0, 0.0, False, False, False, protrusion=1.0)
    assert gain == pytest.approx(0.6)                       # +w * (prev - now)
    loss = env._reward(2.0, 0.0, 0.0, False, False, False, protrusion=1.5) - env._reward(2.0, 0.0, 0.0, False, False, False, protrusion=1.0)
    assert loss == pytest.approx(-0.5)
    # dithering: going out 0.5 m and back in 0.5 m nets exactly zero shaping (no loop to farm)
    total = 0.0
    p = 1.0
    for q in (1.5, 1.0, 1.5, 1.0):
        env._prev_protrusion = p
        total += env._reward(2.0, 0.0, 0.0, False, False, False, protrusion=q) - env._reward(2.0, 0.0, 0.0, False, False, False, protrusion=p)
        p = q
    assert total == pytest.approx(0.0, abs=1e-9)


def test_containment_shaping_is_capped():
    env = _env(reward={"w_contain": 1.0})
    env.reset(seed=1)
    env._prev_protrusion = 50.0
    a = env._reward(2.0, 0.0, 0.0, False, False, False, protrusion=40.0)
    b = env._reward(2.0, 0.0, 0.0, False, False, False, protrusion=3.0)
    assert a == pytest.approx(b)                            # both capped at 3 m: no huge far-field gradients


def test_shaping_keeps_penalties_and_bonuses():
    env = _env(reward={"w_contain": 1.0})
    env.reset(seed=1)
    env._prev_protrusion = 0.0
    crash = env._reward(1.0, 0.0, 0.0, True, False, False, protrusion=0.0)
    assert crash < -10                                      # the collision penalty is untouched
    assert env._reward(0.0, 0.0, 0.0, False, True, False, protrusion=0.0) > 90


def test_step_wires_protrusion_into_reward():
    env = _env(reward={"w_contain": 1.0})
    env.reset(seed=3)
    assert hasattr(env, "_prev_protrusion")
    env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert env._prev_protrusion == pytest.approx(env.bay_protrusion())


# ---------------------------------------------------------------- observation scaling
def test_obs_scale_vector_matches_observation_layout():
    assert OBS_SCALE.shape == (14,) and np.all(OBS_SCALE > 0)
    env = _env()
    obs, _ = env.reset(seed=2)
    s = scale_obs(obs)
    assert s.dtype == np.float32 and s.shape == obs.shape and np.isfinite(s).all()
    assert np.abs(s).max() <= 1.5                           # every channel is O(1)
    wrapped = ScaledObs(_env())
    o2, _ = wrapped.reset(seed=2)
    np.testing.assert_allclose(o2, s)


def test_policy_wrapper_applies_scaling_only_when_meta_says_so():
    from rl.policy_wrapper import PPOPolicy
    legacy = PPOPolicy(str(ROOT / "models" / "ppo_only_best"))
    assert legacy.obs_scale is False                        # original checkpoints are raw-observation models
    if V3.with_suffix(".zip").is_file():
        assert PPOPolicy(str(V3)).obs_scale is False       # v3 was trained without scaling (E1 was rejected)


# ---------------------------------------------------------------- checkpoint selection
def test_checkpoint_rules():
    risky, safe = {"success": 0.70, "collision": 0.29}, {"success": 0.69, "collision": 0.02}
    assert is_better(risky, safe, "success_first")          # the original rule prefers the risky one ...
    assert is_better(safe, risky, "safe")                   # ... the collision-aware rule does not
    assert checkpoint_score({"success": 0.5, "collision": 0.0}, "safe") > checkpoint_score({"success": 0.5, "collision": 0.1}, "safe")
    assert is_better({"success": 0.0, "collision": 0.0}, {"success": -1.0, "collision": 1.0}, "safe")   # initial sentinel is beatable


# ---------------------------------------------------------------- LR schedule
def test_linear_lr_is_driven_by_absolute_steps_and_picklable():
    lr = LinearLR(3e-4, horizon=1000, floor=0.1)
    lr.t = 0
    assert lr(1.0) == pytest.approx(3e-4)
    lr.t = 500
    assert lr(0.123) == pytest.approx(1.5e-4)               # ignores SB3's per-learn() progress argument
    lr.t = 5000
    assert lr(0.0) == pytest.approx(3e-5)                   # floor
    lr2 = pickle.loads(pickle.dumps(lr))
    assert lr2.t == 5000 and lr2.horizon == 1000


# ---------------------------------------------------------------- trainer plumbing (tiny run)
def test_trainer_overrides_and_metadata(tmp_path):
    from rl import train_ppo
    out = str(tmp_path / "smoke")
    try:
        train_ppo.ENV_EXTRA.clear()
        train_ppo.ENV_EXTRA.update({"scenario_kinds": [Kind.PERPENDICULAR], "reward": {"w_contain": 0.5}})
        train_ppo.train(1024, out, True, seed=0, chunk=1024, val_seed=7000, val_episodes=2,
                        overrides={"obs_scale": True, "net_arch": [32, 32], "n_envs": 1, "n_steps": 512, "batch_size": 128,
                                   "lr_schedule": "linear", "activation": "relu", "vf_coef": 1.0, "max_grad_norm": 0.3}, select="safe")
        meta = json.loads(Path(out + ".meta.json").read_text())
        assert meta["obs_scale"] is True and meta["ppo"]["net_arch"] == [32, 32] and meta["select"] == "safe"
        assert meta["env_extra"]["reward"] == {"w_contain": 0.5}
        from rl.policy_wrapper import PPOPolicy
        pol = PPOPolicy(out)
        assert pol.obs_scale is True
        a = pol.act(np.zeros(14, dtype=np.float32))
        assert a.shape == (2,) and np.isfinite(a).all()
        assert pol.model.policy.optimizer.defaults["lr"] > 0
    finally:
        train_ppo.ENV_EXTRA.clear()
        train_ppo.RUN_CFG.clear(); train_ppo.RUN_CFG.update(obs_scale=False)


# ---------------------------------------------------------------- promoted checkpoint
@pytest.mark.skipif(not V3.with_suffix(".zip").is_file(), reason="PPO v3 checkpoint not present")
def test_ppo_v3_loads_dimensions_and_architecture():
    from rl.policy_wrapper import ACT_DIM, OBS_DIM, PPOPolicy
    pol = PPOPolicy(str(V3))
    assert pol.model.observation_space.shape == (OBS_DIM,) and pol.model.action_space.shape == (ACT_DIM,)
    assert pol.model.policy_kwargs["net_arch"] == [256, 256]
    meta = json.loads(V3.with_suffix(".meta.json").read_text())
    assert meta["env_extra"]["reward"]["w_contain"] == 1.0 and meta["seed"] == 1


@pytest.mark.skipif(not V3.with_suffix(".zip").is_file(), reason="PPO v3 checkpoint not present")
def test_ppo_v3_with_envelope_shield_runs_without_collision_on_fixed_seeds():
    from evaluation.run_experiments import make_adapter
    ad = make_adapter("ppo_v3+env")
    env = _env(success_mode="contained")
    collisions = 0
    for seed in range(9000, 9006):                           # held-out block C seeds, short smoke only
        ad.reset()
        obs, _ = env.reset(seed=seed)
        for _ in range(150):
            action, _ = ad.act(obs, env)
            obs, _, term, trunc, info = env.step(action)
            assert np.isfinite(obs).all()
            if term or trunc:
                break
        collisions += int(info["collision"])
    assert collisions == 0


def test_make_adapter_accepts_adhoc_checkpoint_and_rejects_missing():
    from evaluation.run_experiments import make_adapter
    with pytest.raises(FileNotFoundError):
        make_adapter("ppo:models/does_not_exist")
    ad = make_adapter("ppo:models/ppo_only_best")
    assert ad.name == "ppo:models/ppo_only_best"
