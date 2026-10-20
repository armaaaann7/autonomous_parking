"""ParkingEnv: the Gymnasium environment described in blueprint Part 2/3/10.

Fixes relative to the blueprint's skeleton (Part 16, Step 4) -- see docs/REVIEW.md for the full
list:

1. **Action space ambiguity resolved.** Part 2 used ``[steering, throttle, brake]`` (3-dim);
   Part 32's summary used ``[steer, accel]`` (2-dim). This env uses the blueprint's own Part 3
   *recommendation*: a 2-dim continuous action ``[steer, throttle_brake]`` where the sign of the
   second value selects throttle (+) vs brake (-). That single scalar is split into the vehicle's
   separate throttle/brake channels internally.
2. ``delta`` (steering angle) is a real piece of state, not a hard-coded ``0.0``.
3. Collision detection is implemented (SAT vs the obstacle AABBs), not a ``return False`` stub.
4. Success requires the position/heading/speed criteria to hold for ``hold_steps`` consecutive
   steps (Part 10 says "held for a few consecutive steps"; the skeleton checked only one step,
   which lets a car that merely clips through the target register a false success).
5. The reward's collision penalty is bigger than its timeout penalty (20 vs 10, see
   configs/reward_config.yaml). In the blueprint's own numbers, collision was only -2 vs -20 for
   timeout, which pays a fast-approaching agent to crash rather than time out -- exactly the
   "reward hacking" failure mode the blueprint's own Part 10 pitfalls paragraph warns about.
6. Reward is computed from the SIGNED change (`prev - curr`, i.e. "improvement"), matching the
   blueprint's prose ("reward getting closer") -- the blueprint's Step-3 code computed
   `curr - prev` without the outer minus sign it *also* applies, which double-negates and
   rewards moving away. This file's sign convention is: shaping terms add a positive reward when
   distance/heading error decreases.
7. Reward and observation are asserted to be finite; a NaN anywhere raises instead of silently
   propagating (PRD Part 24, item 19).
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from .geometry import obb_aabb_overlap, wrap_angle
from .scenarios import Kind, RandomizationLevel, make_scenario, slot_rectangle
from .sensors import cast_rays
from .vehicle import Vehicle

_HERE = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: dict | None = None):
        super().__init__()
        env_cfg = _load_yaml(_HERE / "configs" / "env_config.yaml")
        reward_cfg = _load_yaml(_HERE / "configs" / "reward_config.yaml")
        config = config or {}
        env_cfg.update(config.get("env", {}))
        reward_cfg.update(config.get("reward", {}))
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg

        self.dt = env_cfg["dt"]
        self.max_steps = env_cfg["max_steps"]
        self.n_rays = env_cfg["sensors"]["n_rays"]
        self.sensor_range = env_cfg["sensors"]["max_range"]
        self.pos_tol = env_cfg["success"]["pos_tol"]
        self.heading_tol = np.deg2rad(env_cfg["success"]["heading_tol_deg"])
        self.speed_tol = env_cfg["success"]["speed_tol"]
        self.hold_steps = env_cfg["success"]["hold_steps"]

        self.scenario_kinds = config.get("scenario_kinds", [Kind.PARALLEL, Kind.PERPENDICULAR])
        self.randomization = RandomizationLevel(config.get("randomization", RandomizationLevel.NONE))
        self.include_obstacles = config.get("include_obstacles", True)
        # A single fixed scale (see train_ppo.py's earlier curriculum) causes catastrophic
        # forgetting: once training moves past a stage, easy start poses vanish from the data
        # entirely and the policy's competence on them decays. Sampling uniformly on
        # [min_start_distance_scale, start_distance_scale] every episode keeps easy cases mixed
        # into every later stage. See docs/REVIEW.md.
        self.start_distance_scale = config.get("start_distance_scale", 1.0)
        self.min_start_distance_scale = config.get("min_start_distance_scale", 0.05)
        self.render_mode = config.get("render_mode")
        # Reverse gear. OFF by default: the original action semantics (second value < 0 = brake only)
        # make reversing impossible, which makes parallel parking unreachable (see docs/REVIEW.md).
        # With allow_reverse=True the second value is a signed throttle: it brakes while the car is
        # moving the opposite way and, once (nearly) stopped, drives it in that direction.
        self.allow_reverse = bool(config.get("allow_reverse", False))
        self.parallel_gap = config.get("parallel_gap")
        if self.allow_reverse and self.parallel_gap is None:
            # A 3.6 m-radius car cannot enter a gap much smaller than ~7.4 m in one reverse S-curve;
            # 8.0 m (~1.9 car lengths) is solved 100% of the time by baselines/parallel_geometric.py.
            self.parallel_gap = 8.0

        self.vehicle = Vehicle(env_cfg["vehicle"], dt=self.dt)
        obs_dim = 6 + self.n_rays
        # [dx, dy, sin, cos, v, delta, rays...]; bounded (not +-inf) so check_env doesn't warn
        # and so SB3's default observation normalisation has something sane to work with.
        bound = np.concatenate([[40.0, 40.0, 1.0, 1.0, self.vehicle.max_speed, self.vehicle.max_steer],
                                 np.full(self.n_rays, self.sensor_range)]).astype(np.float32)
        self.observation_space = spaces.Box(low=-bound, high=bound, dtype=np.float32)
        # action = [steer, throttle_brake] ; see module docstring point 1.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._rng = np.random.default_rng(config.get("seed"))
        self.scenario = None
        self.steps = 0
        self._success_streak = 0
        self._renderer = None

    # ---------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        kind = options.get("kind") or self.scenario_kinds[self._rng.integers(len(self.scenario_kinds))]
        lo = min(self.min_start_distance_scale, self.start_distance_scale)
        scale = float(self._rng.uniform(lo, self.start_distance_scale)) if self.start_distance_scale > lo else lo

        # Rejection-sample: stacking slot-position + start-pose + heading jitter can occasionally
        # place the car's initial footprint already overlapping an obstacle (confirmed via
        # demo.py -- an episode that "collided" on literal step 1 before any action could matter).
        # That's a scenario-generation bug, not something a controller could ever avoid, so retry
        # a few times rather than handing out an unwinnable episode. Falls back to the last draw
        # (uncommon) if still colliding after the cap, rather than looping forever.
        for _ in range(20):
            self.scenario = make_scenario(kind, self.randomization, self._rng,
                                           self.vehicle.length, self.vehicle.width,
                                           include_obstacles=self.include_obstacles,
                                           start_distance_scale=scale, parallel_gap=self.parallel_gap)
            self.vehicle.reset(*self.scenario.start_pose)
            if not obb_aabb_overlap(self.vehicle.corners(), self.scenario.obstacles).any():
                break
        self.steps = 0
        self._success_streak = 0
        self._prev_dist = self._dist_to_slot()
        self._prev_heading_err = self._heading_err()
        return self._get_obs(), {"scenario": self.scenario.kind.value}

    def step(self, action: np.ndarray):
        steer_cmd, thb = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        v_now = self.vehicle.v
        if not self.allow_reverse:
            accel_cmd, brake_cmd = max(thb, 0.0), max(-thb, 0.0)
        elif thb < 0 and v_now > 0.05:
            accel_cmd, brake_cmd = 0.0, -thb          # braking while rolling forward
        elif thb > 0 and v_now < -0.05:
            accel_cmd, brake_cmd = 0.0, thb           # braking while rolling backward
        else:
            accel_cmd, brake_cmd = thb, 0.0           # signed throttle (negative = reverse)
        prev_steer = self.vehicle.delta / max(self.vehicle.max_steer, 1e-9)

        self.vehicle.step(steer_cmd, accel_cmd, brake_cmd)
        self.steps += 1

        dist = self._dist_to_slot()
        heading_err = self._heading_err()
        collided = self._check_collision() or self._out_of_bounds()
        ok_now = (dist < self.pos_tol and abs(heading_err) < self.heading_tol
                  and abs(self.vehicle.v) < self.speed_tol)
        self._success_streak = self._success_streak + 1 if ok_now else 0
        succeeded = self._success_streak >= self.hold_steps
        timed_out = self.steps >= self.max_steps

        reward = self._reward(dist, heading_err, steer_cmd - prev_steer, collided, succeeded, timed_out)
        self._prev_dist, self._prev_heading_err = dist, heading_err

        terminated = bool(collided or succeeded)
        truncated = bool(timed_out and not terminated)
        obs = self._get_obs()
        assert np.isfinite(obs).all(), f"non-finite observation: {obs}"
        assert np.isfinite(reward), f"non-finite reward at step {self.steps}"
        info = {
            "success": succeeded, "collision": collided, "timeout": truncated,
            "pos_error": dist, "heading_error_deg": np.rad2deg(abs(heading_err)),
            "action_final": [float(steer_cmd), float(accel_cmd), float(brake_cmd)],
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        from .renderer import Renderer  # local import: keeps pygame optional for headless training
        if self._renderer is None:
            self._renderer = Renderer(env=self, mode=self.render_mode)
        return self._renderer.draw()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --------------------------------------------------------------- helpers
    def _dist_to_slot(self) -> float:
        sx, sy, _ = self.scenario.slot_pose
        return float(np.hypot(self.vehicle.x - sx, self.vehicle.y - sy))

    def _heading_err(self) -> float:
        return float(wrap_angle(self.vehicle.theta - self.scenario.slot_pose[2]))

    def _check_collision(self) -> bool:
        corners = self.vehicle.corners()
        return bool(obb_aabb_overlap(corners, self.scenario.obstacles).any())

    def _out_of_bounds(self) -> bool:
        xmin, xmax, ymin, ymax = self.scenario.bounds
        return not (xmin <= self.vehicle.x <= xmax and ymin <= self.vehicle.y <= ymax)

    def _get_obs(self) -> np.ndarray:
        rays = cast_rays(self.vehicle.x, self.vehicle.y, self.vehicle.theta, self.scenario.obstacles,
                          n_rays=self.n_rays, max_range=self.sensor_range,
                          noise_std_frac=self.scenario.noise_std_frac, rng=self._rng)
        sx, sy, _ = self.scenario.slot_pose
        dx, dy = sx - self.vehicle.x, sy - self.vehicle.y
        heading_err = self._heading_err()
        obs = np.concatenate([
            [dx, dy, np.sin(heading_err), np.cos(heading_err), self.vehicle.v, self.vehicle.delta],
            rays,
        ]).astype(np.float32)
        return obs

    def _reward(self, dist, heading_err, steer_delta, collided, succeeded, timed_out) -> float:
        c = self.reward_cfg
        r = -c["time_penalty"]
        r += c["w_dist"] * (self._prev_dist - dist)                       # positive = got closer
        r += c["w_heading"] * (abs(self._prev_heading_err) - abs(heading_err))  # positive = better aligned
        r -= c["w_smooth"] * abs(steer_delta)
        # Dense "docking" bonus: reward being simultaneously close AND aligned AND slow, scaled
        # up smoothly as the car nears the success box, rather than only an all-or-nothing bonus
        # at the end. Without this, shaping credit for "closer" and "more aligned" can be earned
        # independently while drifting past the target, with no gradient pushing the agent to
        # actually settle inside the (tight) combined tolerance box -- see docs/REVIEW.md.
        near = max(0.0, 1.0 - dist / self.pos_tol / 3.0)
        aligned = max(0.0, 1.0 - abs(heading_err) / self.heading_tol / 3.0)
        slow = max(0.0, 1.0 - abs(self.vehicle.v) / self.speed_tol / 3.0)
        r += 0.3 * near * aligned * slow
        if collided:
            r -= c["collision_penalty"]
        if succeeded:
            r += c["success_bonus"]
        if timed_out:
            r -= c["timeout_penalty"]
        return float(r)

    # convenience for the fuzzy shield (integration/hybrid_controller.py)
    def fuzzy_inputs(self):
        """(d_obs, obstacle_bearing_deg, speed) for the SAFETY SHIELD's rule base.

        ``obstacle_bearing_deg`` is the bearing to the nearest ray hit, in the shield's own
        clockwise-positive convention (matching fuzzy/membership_configs.py's "large_left" being
        negative): if the nearest obstacle is off to the left, this is negative, and the rule
        base's ``large_left -> steer large_right`` fires, i.e. steer AWAY from the obstacle.

        An earlier version of this method returned the bearing to the SLOT instead, in the
        opposite (CCW-positive) sign convention. That made the "safety shield" issue large
        navigation-toward-slot steering commands whenever it was near an obstacle -- exactly when
        the RL policy's own fine maneuvering matters most -- rather than a small avoidance nudge,
        which is why an earlier evaluation run showed PPO+Fuzzy doing far *worse* than PPO alone.
        See docs/REVIEW.md.
        """
        rays = cast_rays(self.vehicle.x, self.vehicle.y, self.vehicle.theta, self.scenario.obstacles,
                          n_rays=self.n_rays, max_range=self.sensor_range)
        i = int(np.argmin(rays))
        d_obs = float(rays[i])
        ray_angle = 2.0 * np.pi * i / self.n_rays  # relative to heading, ray 0 = straight ahead, CCW
        obstacle_bearing_deg = float(np.rad2deg(wrap_angle(-ray_angle)))  # negate: CW-positive convention
        return d_obs, obstacle_bearing_deg, float(self.vehicle.v)

    def slot_bearing_deg(self) -> float:
        """Bearing to the slot centre, CCW-positive (i.e. positive = steer_cmd should be positive
        to turn toward it, matching baselines/rule_based_controller.py's convention). Used by
        fuzzy/fuzzy_only_controller.py for its navigation term -- kept separate from
        ``fuzzy_inputs()`` because the two need opposite sign conventions for different purposes.
        """
        sx, sy, _ = self.scenario.slot_pose
        bearing = np.arctan2(sy - self.vehicle.y, sx - self.vehicle.x) - self.vehicle.theta
        return float(np.rad2deg(wrap_angle(bearing)))
