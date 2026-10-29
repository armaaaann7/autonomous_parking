"""Train PPO on ParkingEnv (blueprint Part 15 Phase 6 / Part 16 Step 5), with the curriculum
from Part 9: start at ``RandomizationLevel.NONE`` and step up once success rate clears 80%.

Usage:
    python -m rl.train_ppo --timesteps 300000 --out models/ppo_only
    python -m rl.train_ppo --timesteps 300000 --curriculum   # auto-advance randomization

Differences from the blueprint's Step-5 sketch: config is loaded from
``rl/configs/ppo_config.yaml`` instead of hard-coded, ``n_envs`` uses ``DummyVecEnv`` (this
container has a single CPU core, so subprocess workers would only add overhead), and a
``VecMonitor`` is attached so episode success/collision rates are visible in TensorBoard, not
just the raw reward.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from environment.parking_env import ParkingEnv
from environment.scenarios import RandomizationLevel

_HERE = Path(__file__).resolve().parent.parent

# Extra ParkingEnv config applied to every training / evaluation env (set from the CLI: --reverse, --kinds).
ENV_EXTRA: dict = {}
# Resolved PPO hyper-parameters for the run in progress (set by train()); read by make_env/evaluate_policy.
RUN_CFG: dict = {"obs_scale": False}
# Extra env config used ONLY by the validation episodes that pick the best checkpoint (e.g. success_mode='contained'),
# so training/curriculum can use the pose box while checkpoint selection uses the strict metric.
VAL_EXTRA: dict = {}


def _maybe_scale(env):
    if RUN_CFG.get("obs_scale"):
        from rl.obs_scaling import ScaledObs
        env = ScaledObs(env)
    return env


# Curriculum stages: (include_obstacles, start_distance_scale, RandomizationLevel), applied in
# order. This layers THREE axes from the blueprint: Part 8's scenario complexity (no obstacles ->
# obstacles), a distance bootstrap not in the original blueprint at all (see below), and Part 9's
# domain randomization (start pose -> slot -> obstacle position -> sensor noise).
#
# Two earlier versions of this curriculum each converged to a "brake hard and never move" policy:
#   1. The very first version never varied obstacles at all, so even the "no randomization"
#      stage had two parked cars right next to the slot.
#   2. After removing the obstacles, PPO *still* froze: at the nominal ~6-7m approach distance a
#      randomly-exploring policy essentially never reaches the success bonus by chance, so it
#      never gets a positive reward to learn from, and "sit still" is a low-variance, easy-to-find
#      local optimum with no gradient pushing the agent away from it.
# Stages 0-6 fix this by starting the car very close to the slot (scale 0.05, ~0.3m away -- just
# outside the success tolerance) so even a near-random policy succeeds often enough to bootstrap
# the value function with real +100 experiences, then walking the distance back out to the full
# ~6-7m approach in small enough steps that each stage is solvable by what the *previous* stage
# already learned (unlike pure random exploration, which empirically solves scale<=0.05 ~97% of
# the time but scale 0.15, a ~1m start distance, 0% of the time -- random search can't bridge that
# gap, but a policy that already learned the 0.3-0.5m case can, so the ramp is intentionally
# finer early and coarser later). Obstacles and Part-9 domain randomization are only switched on
# once the full-distance, obstacle-free case is solved. See docs/REVIEW.md for the full
# debugging trace, including the random-policy success-rate-vs-distance numbers above.
def _stage(obstacles, scale, level, pos_tol=None, heading_tol_deg=None):
    return {"obstacles": obstacles, "scale": scale, "level": level,
            "pos_tol": pos_tol, "heading_tol_deg": heading_tol_deg}


_N, _SP, _SL, _OB, _SN = (RandomizationLevel.NONE, RandomizationLevel.START_POSE,
                          RandomizationLevel.SLOT_POSITION, RandomizationLevel.OBSTACLES,
                          RandomizationLevel.SENSOR_NOISE)
CURRICULUM_STAGES = [
    _stage(False, 0.05, _N), _stage(False, 0.10, _N), _stage(False, 0.20, _N),
    _stage(False, 0.35, _N), _stage(False, 0.55, _N), _stage(False, 0.80, _N),
    _stage(False, 1.00, _N), _stage(True, 1.00, _N), _stage(True, 1.00, _SP),
    _stage(True, 1.00, _SL), _stage(True, 1.00, _OB), _stage(True, 1.00, _SN),
    # precision stages: tighten the success box once everything else is solved
    _stage(True, 1.00, _SN, pos_tol=0.35, heading_tol_deg=6.0),
    _stage(True, 1.00, _SN, pos_tol=0.25, heading_tol_deg=4.0),
]


class LinearLR:
    """Linear learning-rate decay lr0 -> lr0*floor over ``horizon`` TOTAL timesteps.

    SB3's built-in schedule is relative to each ``learn()`` call; this trainer calls learn() once per
    checkpoint chunk (and again on --resume), so a naive ``progress_remaining`` schedule would restart
    / collapse every chunk. This one is driven by the absolute step counter, kept in ``self.t`` by
    ``SuccessRateCallback``. Picklable (module-level class), so --resume works."""

    def __init__(self, lr0: float, horizon: int, floor: float = 0.1):
        self.lr0, self.horizon, self.floor, self.t = float(lr0), int(horizon), float(floor), 0

    def __call__(self, progress_remaining: float) -> float:
        return self.lr0 * max(1.0 - self.t / max(self.horizon, 1), self.floor)


class SuccessRateCallback(BaseCallback):
    """Tracks a rolling success rate and (optionally) advances ``CURRICULUM_STAGES`` once it
    crosses ``threshold`` -- avoids the "full randomization/full difficulty from step 1" failure
    mode the blueprint calls out in Part 9."""

    def __init__(self, curriculum: bool, threshold: float = 0.6, window: int = 60, verbose: int = 0,
                 init_stage: int = 0):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.threshold = threshold
        self.window = window
        self._buf: list[bool] = []
        self.stage = (len(CURRICULUM_STAGES) - 1) if not curriculum else init_stage

    def _apply_stage(self):
        st = CURRICULUM_STAGES[self.stage]
        for e in self.training_env.envs:
            u = e.unwrapped
            u.include_obstacles = st["obstacles"]
            u.start_distance_scale = st["scale"]
            u.randomization = st["level"]
            if st["pos_tol"] is not None:
                u.pos_tol = st["pos_tol"]
            if st["heading_tol_deg"] is not None:
                u.heading_tol = np.deg2rad(st["heading_tol_deg"])

    def _on_training_start(self) -> None:
        self._apply_stage()

    def _on_step(self) -> bool:
        lr = getattr(self.model, "learning_rate", None)
        if isinstance(lr, LinearLR):
            lr.t = self.num_timesteps
        for info in self.locals.get("infos", []):
            if "episode" in info:  # Monitor adds this key exactly once, at episode end
                self._buf.append(bool(info.get("success", False)))
        if len(self._buf) > self.window:
            self._buf = self._buf[-self.window:]
        if (self.curriculum and len(self._buf) >= self.window
                and np.mean(self._buf) >= self.threshold
                and self.stage < len(CURRICULUM_STAGES) - 1):
            self.stage += 1
            self._apply_stage()
            self._buf.clear()
            if self.verbose:
                st = CURRICULUM_STAGES[self.stage]
                print(f"[curriculum] advanced to stage {self.stage} "
                      f"(obstacles={st['obstacles']}, dist_scale={st['scale']}, level={st['level'].name}, "
                      f"pos_tol={st['pos_tol']}, heading_tol_deg={st['heading_tol_deg']}) "
                      f"at step {self.num_timesteps}")
        self.logger.record("rollout/success_rate", float(np.mean(self._buf)) if self._buf else 0.0)
        self.logger.record("rollout/curriculum_stage", self.stage)
        return True


def make_env(include_obstacles: bool, start_distance_scale: float, randomization: RandomizationLevel, seed: int,
             shielded: bool = False, safe_threshold: float = 1.0):
    def _init():
        env = ParkingEnv({"seed": seed, "randomization": randomization, "include_obstacles": include_obstacles,
                           "start_distance_scale": start_distance_scale, **ENV_EXTRA})
        if shielded:
            from environment.shielded_env import ShieldedParkingEnv
            env = ShieldedParkingEnv(env, safe_threshold=safe_threshold)
        return Monitor(_maybe_scale(env))
    return _init


def evaluate_policy(model, n_episodes: int = 40, shielded: bool = False, safe_threshold: float = 1.0,
                    seed0: int = 5000) -> dict:
    """Held-out evaluation under the SAME protocol as evaluation/run_experiments.py (both slot
    kinds, SLOT_POSITION randomization, default tolerances, deterministic policy). Training
    success is measured on the current curriculum stage, which can look healthy while the policy
    is actually collapsing on the real task -- this is the number that matters."""
    from environment.scenarios import Kind
    env = ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION,
                      **{"scenario_kinds": [Kind.PARALLEL, Kind.PERPENDICULAR], **ENV_EXTRA, **VAL_EXTRA}})
    if shielded:
        from environment.shielded_env import ShieldedParkingEnv
        env = ShieldedParkingEnv(env, safe_threshold=safe_threshold)
    env = _maybe_scale(env)
    succ = coll = 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + i)
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        succ += int(info["success"])
        coll += int(info["collision"])
    env.close()
    return {"success": succ / n_episodes, "collision": coll / n_episodes}


def _tb_dir():
    """TensorBoard log dir, or None when tensorboard is not installed (training still works)."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return None
    return str(_HERE / "experiments" / "tb_logs")


def checkpoint_score(ev: dict, rule: str = "success_first") -> tuple:
    """Sort key for best-checkpoint selection (higher is better).

    'success_first' (the ORIGINAL rule): max success, ties broken by lower collision. It can pick a
    checkpoint with high collision when success is equal-or-slightly-higher (seen in the RL audit:
    0.70 success / 0.29 collision beat 0.69 / 0.02).
    'safe': max (success - 2 * collision), ties by success. A collision costs twice a success, so a
    checkpoint trading safety for a few points of success is not selected."""
    if rule == "safe":
        return (ev["success"] - 2.0 * ev["collision"], ev["success"])
    return (ev["success"], -ev["collision"])


def is_better(ev: dict, best: dict, rule: str = "success_first") -> bool:
    return checkpoint_score(ev, rule) > checkpoint_score(best, rule)


def _stage_sidecar(out_path: Path) -> Path:
    return out_path.with_suffix(".stage.json")


def train(timesteps: int, out_path: str, curriculum: bool, seed: int = 0,
          resume: bool = False, init_stage: int = 0, shielded: bool = False, safe_threshold: float = 1.0, chunk: int = 100_000,
          val_seed: int = 5000, val_episodes: int = 40, overrides: dict | None = None, select: str = "success_first"):
    import json
    with open(_HERE / "rl" / "configs" / "ppo_config.yaml") as f:
        cfg = yaml.safe_load(f)["ppo"]
    cfg.update(overrides or {})
    RUN_CFG.clear()
    RUN_CFG.update(obs_scale=bool(cfg.get("obs_scale", False)))

    out = _HERE / out_path
    sidecar = _stage_sidecar(out)
    meta = {"ppo": cfg, "env_extra": {k: (str(v) if k == "scenario_kinds" else v) for k, v in ENV_EXTRA.items()},
            "obs_scale": RUN_CFG["obs_scale"], "seed": seed, "curriculum": curriculum, "shielded": shielded,
            "val_extra": dict(VAL_EXTRA), "val_seed": val_seed, "val_episodes": val_episodes, "timesteps_requested": timesteps, "select": select}
    meta_path = out.with_suffix(".meta.json")
    if resume and meta_path.exists():                    # a resumed run keeps the observation convention it was built with
        RUN_CFG["obs_scale"] = bool(json.loads(meta_path.read_text()).get("obs_scale", False))
        meta["obs_scale"] = RUN_CFG["obs_scale"]
    if resume and sidecar.exists():
        init_stage = json.loads(sidecar.read_text())["stage"]
        print(f"[resume] continuing from stage {init_stage}")

    if curriculum:
        st = CURRICULUM_STAGES[min(init_stage, len(CURRICULUM_STAGES) - 1)]
        start_obstacles, start_scale, start_level = st["obstacles"], st["scale"], st["level"]
    else:
        start_obstacles, start_scale, start_level = True, 1.0, RandomizationLevel.SLOT_POSITION
    venv = DummyVecEnv([make_env(start_obstacles, start_scale, start_level, seed + i, shielded, safe_threshold)
                        for i in range(cfg["n_envs"])])
    venv = VecMonitor(venv)

    if resume and out.with_suffix(".zip").exists():
        # tensorboard_log is stored in the checkpoint as an absolute path from the machine that
        # trained it (e.g. /home/claude/...); override it or resuming elsewhere crashes.
        tb_log = _tb_dir()
        co = {"tensorboard_log": tb_log}
        if cfg.get("lr_schedule", "constant") == "linear":
            co["learning_rate"] = co["lr_schedule"] = LinearLR(cfg["learning_rate"], cfg.get("lr_horizon", 420_000))
        model = PPO.load(str(out), env=venv, custom_objects=co)
        print(f"[resume] loaded weights from {out}.zip")
    else:
        import torch.nn as nn
        lr = cfg["learning_rate"]
        if cfg.get("lr_schedule", "constant") == "linear":
            lr = LinearLR(lr, cfg.get("lr_horizon", 420_000))
        pk = {"net_arch": cfg["net_arch"]}
        if cfg.get("activation", "tanh") == "relu":
            pk["activation_fn"] = nn.ReLU
        model = PPO(
            cfg["policy"], venv, verbose=1,
            n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
            gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"], learning_rate=lr,
            ent_coef=cfg["ent_coef"], clip_range=cfg["clip_range"],
            vf_coef=cfg.get("vf_coef", 0.5), max_grad_norm=cfg.get("max_grad_norm", 0.5),
            policy_kwargs=pk,
            tensorboard_log=_tb_dir(),
            seed=seed,
        )
    callback = SuccessRateCallback(curriculum=curriculum, verbose=1, init_stage=init_stage)
    # Train in chunks and checkpoint after each one, so a killed process (container reset, laptop
    # sleep) loses at most one chunk and can be continued with --resume.
    out.parent.mkdir(parents=True, exist_ok=True)
    best_path = out.parent / (out.name + "_best")
    best_sidecar = best_path.with_suffix(".stage.json")
    best = {"success": -1.0, "collision": 1.0}
    if resume and best_sidecar.exists():          # keep the best-so-far across --resume invocations
        _b = json.loads(best_sidecar.read_text())
        best = {"success": _b.get("success", -1.0), "collision": _b.get("collision", 1.0)}
    done, first = 0, True
    while done < timesteps:
        n = min(chunk, timesteps - done)
        model.learn(total_timesteps=n, callback=callback, progress_bar=False,
                    reset_num_timesteps=(not resume) and first)
        first = False
        done += n
        model.save(str(out))
        meta_path.write_text(json.dumps(meta, indent=1, default=str))
        sidecar.write_text(json.dumps({"stage": callback.stage}))
        ev = evaluate_policy(model, n_episodes=val_episodes, shielded=shielded, safe_threshold=safe_threshold, seed0=val_seed)
        flag = ""
        if is_better(ev, best, select):
            best = ev
            model.save(str(best_path))
            best_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str))
            best_sidecar.write_text(json.dumps({"stage": callback.stage, **ev}))
            flag = "  <- new best, saved " + best_path.name + ".zip"
        print(f"[checkpoint] {done}/{timesteps} steps (stage {callback.stage}) | held-out eval: "
              f"success={ev['success']:.2f} collision={ev['collision']:.2f} | best={best['success']:.2f}{flag}", flush=True)
        if best["success"] >= 0.15 and ev["success"] <= 0.3 * best["success"]:
            print("[warn] held-out success fell below 30% of its best -- the policy may be collapsing. "
                  f"Use {best_path.name}.zip (the best checkpoint), not {out.name}.zip.", flush=True)
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--out", type=str, default="models/ppo_only")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from --out's saved weights + curriculum stage")
    p.add_argument("--init-stage", type=int, default=0)
    p.add_argument("--shielded", action="store_true", help="train THROUGH the fuzzy safety shield")
    p.add_argument("--safe-threshold", type=float, default=1.0)
    p.add_argument("--chunk", type=int, default=100_000, help="checkpoint every N steps")
    p.add_argument("--val-seed", type=int, default=5000,
                   help="first seed of the checkpoint-selection (validation) episodes. The ORIGINAL training used 5000, which "
                        "overlaps the audit's evaluation set B (5000-5199); new experiments use 7000+ (see AUDIT P3)")
    p.add_argument("--val-episodes", type=int, default=40)
    p.add_argument("--reverse", action="store_true", help="enable the reverse gear (needed to learn parallel parking)")
    p.add_argument("--kinds", choices=["perpendicular"], default="perpendicular")
    p.add_argument("--select", choices=["success_first", "safe"], default="success_first",
                   help="best-checkpoint rule: 'success_first' (original) or 'safe' = success - 2*collision")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="override a ppo_config.yaml key (e.g. --set learning_rate=1e-4 --set net_arch=[256,256] "
                        "--set obs_scale=true --set lr_schedule=linear --set vf_coef=1.0). Recorded in <out>.meta.json")
    p.add_argument("--reward", action="append", default=[], metavar="KEY=VAL",
                   help="override a reward_config.yaml key for TRAINING, e.g. --reward w_contain=1.0")
    p.add_argument("--obs-mode", choices=["legacy", "slot"], default=None, help="observation representation (environment/observations.py)")
    p.add_argument("--val-success-mode", choices=["pose", "contained"], default=None,
                   help="success criterion for the checkpoint-selection episodes only (default: same as training)")
    p.add_argument("--success-mode", choices=["pose", "contained"], default=None,
                   help="success criterion used for training/validation (default: the env default, 'pose')")
    args = p.parse_args()
    ov = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        ov[k] = yaml.safe_load(v)
    rw = {}
    for kv in args.reward:
        k, v = kv.split("=", 1)
        rw[k] = yaml.safe_load(v)
    if rw:
        ENV_EXTRA["reward"] = rw
    if args.obs_mode:
        ENV_EXTRA["obs_mode"] = args.obs_mode
    if args.val_success_mode:
        VAL_EXTRA["success_mode"] = args.val_success_mode
    if args.success_mode:
        ENV_EXTRA["success_mode"] = args.success_mode
    if args.reverse:
        ENV_EXTRA["allow_reverse"] = True
    if args.kinds != "both":
        from environment.scenarios import Kind as _K
        ENV_EXTRA["scenario_kinds"] = [_K.PARALLEL if args.kinds == "parallel" else _K.PERPENDICULAR]
    train(args.timesteps, args.out, args.curriculum, args.seed, args.resume, args.init_stage,
          args.shielded, args.safe_threshold, args.chunk, args.val_seed, args.val_episodes, ov, args.select)
