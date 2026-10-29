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

from .geometry import obb_aabb_distance, obb_aabb_overlap, wrap_angle
from .observations import CLEAR_CAP, OBS_MODES, slot_observation
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
        # Success definition (see VALIDATION/AUDIT docs):
        #   "pose"      (default, LEGACY): centre within pos_tol, heading within heading_tol, speed
        #                within speed_tol, held hold_steps steps. All historical results use this.
        #   "contained" (STRICT): the whole 4.2 x 1.8 m footprint lies inside the painted
        #                2.4 x 4.8 m bay rectangle (zero protrusion), heading within heading_tol,
        #                speed within speed_tol, held hold_steps steps. pos_tol is not used.
        # Observation representation: 'legacy' (14-d, default, all old checkpoints) or 'slot' (18-d, environment/observations.py).
        self.obs_mode = config.get("obs_mode", "legacy")
        if self.obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}")
        self.success_mode = config.get("success_mode", "pose")
        if self.success_mode not in ("pose", "contained"):
            raise ValueError("success_mode must be 'pose' or 'contained'")

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
            # Legacy reverse-mode gap retained for environment compatibility.
            self.parallel_gap = 8.0

        self.vehicle = Vehicle(env_cfg["vehicle"], dt=self.dt)
        # [dx, dy, sin, cos, v, delta, rays...]; bounded (not +-inf) so check_env doesn't warn
        # and so SB3's default observation normalisation has something sane to work with.
        bound = np.concatenate([[40.0, 40.0, 1.0, 1.0, self.vehicle.max_speed, self.vehicle.max_steer],
                                 np.full(self.n_rays, self.sensor_range)]).astype(np.float32)
        if self.obs_mode == "slot":
            bound = np.concatenate([[40.0] * 5, [1.0, 1.0, self.vehicle.max_speed, self.vehicle.max_steer],
                                    np.full(self.n_rays, self.sensor_range), [CLEAR_CAP]]).astype(np.float32)
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
        self._prev_protrusion = self.bay_protrusion()
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
        protrusion = self.bay_protrusion()
        contained = protrusion <= 1e-9
        pose_ok = (dist < self.pos_tol and abs(heading_err) < self.heading_tol
                   and abs(self.vehicle.v) < self.speed_tol)
        if self.success_mode == "contained":
            ok_now = contained and abs(heading_err) < self.heading_tol and abs(self.vehicle.v) < self.speed_tol
        else:
            ok_now = pose_ok
        self._success_streak = self._success_streak + 1 if ok_now else 0
        # A step that collides can never count as a successful park, even if the centre pose is
        # inside the tolerance box (the car's corners can clip a neighbour while its centre is
        # within 0.4 m / 15 deg of the slot centre).
        succeeded = self._success_streak >= self.hold_steps and not collided
        timed_out = self.steps >= self.max_steps

        reward = self._reward(dist, heading_err, steer_cmd - prev_steer, collided, succeeded, timed_out,
                              protrusion=protrusion)
        self._prev_dist, self._prev_heading_err, self._prev_protrusion = dist, heading_err, protrusion

        terminated = bool(collided or succeeded)
        truncated = bool(timed_out and not terminated)
        obs = self._get_obs()
        assert np.isfinite(obs).all(), f"non-finite observation: {obs}"
        assert np.isfinite(reward), f"non-finite reward at step {self.steps}"
        info = {
            "success": succeeded, "collision": collided, "timeout": truncated,
            "pos_error": dist, "heading_error_deg": np.rad2deg(abs(heading_err)),
            "action_final": [float(steer_cmd), float(accel_cmd), float(brake_cmd)],
            "contained": bool(contained), "bay_protrusion": float(protrusion),
            "pose_ok": bool(pose_ok), "slot_frame_error": self.slot_frame_error(),
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
        legacy = np.concatenate([
            [dx, dy, np.sin(heading_err), np.cos(heading_err), self.vehicle.v, self.vehicle.delta],
            rays,
        ]).astype(np.float32)
        self._legacy_obs = legacy                 # cached so fuzzy controllers can read the legacy layout in any obs_mode
        if self.obs_mode == "slot":
            return slot_observation((self.vehicle.x, self.vehicle.y), self.vehicle.theta, self.scenario.slot_pose, self.vehicle.v,
                                    self.vehicle.delta, rays, self.body_clearance())
        return legacy

    def legacy_obs(self) -> np.ndarray:
        """The 14-d legacy vector of the CURRENT state (for fuzzy controllers / shields), whatever obs_mode is."""
        return self._legacy_obs

    def _reward(self, dist, heading_err, steer_delta, collided, succeeded, timed_out, protrusion=None) -> float:
        c = self.reward_cfg
        r = -c["time_penalty"]
        r += c["w_dist"] * (self._prev_dist - dist)                       # positive = got closer
        r += c["w_heading"] * (abs(self._prev_heading_err) - abs(heading_err))  # positive = better aligned
        r -= c["w_smooth"] * abs(steer_delta)
        # Optional containment shaping (w_contain, default 0 = the original reward). Potential-based
        # on the bay protrusion (metres any corner lies outside the painted bay, capped), so it adds
        # no net reward for dithering and cannot be farmed; it makes the training signal agree with
        # the strict 'whole car inside the bay' metric instead of only the centre-pose box.
        w_contain = c.get("w_contain", 0.0)
        if w_contain and protrusion is not None:
            r += w_contain * (min(self._prev_protrusion, 3.0) - min(protrusion, 3.0))
        # Dense "docking" bonus: reward being simultaneously close AND aligned AND slow, scaled
        # up smoothly as the car nears the success box, rather than only an all-or-nothing bonus
        # at the end. Without this, shaping credit for "closer" and "more aligned" can be earned
        # independently while drifting past the target, with no gradient pushing the agent to
        # actually settle inside the (tight) combined tolerance box -- see docs/REVIEW.md.
        near = max(0.0, 1.0 - dist / self.pos_tol / 3.0)
        aligned = max(0.0, 1.0 - abs(heading_err) / self.heading_tol / 3.0)
        slow = max(0.0, 1.0 - abs(self.vehicle.v) / self.speed_tol / 3.0)
        # Do not pay the docking bonus repeatedly for remaining stationary outside
        # the success region. Bonus requires actual improvement or a valid park.
        progressing = self._prev_dist - dist > 1e-4 or abs(self._prev_heading_err) - abs(heading_err) > 1e-4
        if progressing or succeeded:
            r += 0.3 * near * aligned * slow
        if collided:
            r -= c["collision_penalty"]
        if succeeded:
            r += c["success_bonus"]
        if timed_out and not (collided or succeeded):
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

    def bay_protrusion(self) -> float:
        """Largest distance (m) by which any car corner lies outside the painted bay rectangle
        (bay = car + 0.6 m in each dimension), 0.0 if the whole car is inside. Diagnostic only: the
        success criterion is centre pose + heading + speed (configs/env_config.yaml), which is
        looser than 'fully inside the lines' (0.4 m centre tolerance vs 0.3 m bay clearance)."""
        r = slot_rectangle(self.scenario, self.vehicle.length, self.vehicle.width)
        c = self.vehicle.corners()
        return float(max(0.0, r[0] - c[:, 0].min(), c[:, 0].max() - r[2],
                         r[1] - c[:, 1].min(), c[:, 1].max() - r[3]))

    def slot_frame_error(self):
        """(longitudinal, lateral) offset in metres of the car centre from the slot centre, in the
        slot frame: longitudinal = along the slot's long axis (the direction the car should face),
        positive = beyond the slot centre; lateral = across it."""
        sx, sy, st = self.scenario.slot_pose
        dx, dy = self.vehicle.x - sx, self.vehicle.y - sy
        return (float(np.cos(st) * dx + np.sin(st) * dy), float(-np.sin(st) * dx + np.cos(st) * dy))

    def inside_bay(self) -> bool:
        return self.bay_protrusion() <= 1e-9

    # ------------------------------------------------------------ true-geometry safety queries
    def body_clearance(self) -> float:
        """Exact minimum distance (m) from the car's 4.2 x 1.8 m footprint to any obstacle, capped
        at the sensor range. 0 means touching/overlapping. Independent of the 8 centre rays."""
        obs = self.scenario.obstacles
        if len(obs) == 0:
            return float(self.sensor_range)
        return float(min(obb_aabb_distance(self.vehicle.corners(), obs).min(), self.sensor_range))

    def shield_inputs(self):
        """(clearance, obstacle_bearing_deg, speed) for the Mamdani shield, using TRUE body
        clearance instead of the centre-ray distance used by ``fuzzy_inputs``. The bearing points
        (clockwise-positive, like ``fuzzy_inputs``) at the closest point of the nearest obstacle."""
        obs = self.scenario.obstacles
        if len(obs) == 0:
            return float(self.sensor_range), 0.0, float(abs(self.vehicle.v))
        dist = obb_aabb_distance(self.vehicle.corners(), obs)
        i = int(np.argmin(dist))
        box = obs[i]
        px = float(np.clip(self.vehicle.x, box[0], box[2]))
        py = float(np.clip(self.vehicle.y, box[1], box[3]))
        ang = np.arctan2(py - self.vehicle.y, px - self.vehicle.x) - self.vehicle.theta
        return (float(min(dist[i], self.sensor_range)), float(np.rad2deg(wrap_angle(-ang))),
                float(abs(self.vehicle.v)))

    def braking_envelope_safe(self, action, hold_steer_cmd=None, max_brake_steps: int = 80, margin: float = 0.0):
        """Model-based check used by the envelope shield (integration/envelope_shield.py).

        Simulates, on a scratch copy of the vehicle, ONE step of ``action`` followed by full braking
        with the steering command held until the car stops, and checks the car's true OBB against
        every obstacle and the bounds at each step. Returns (safe, first_unsafe_step or None).
        The scratch vehicle uses exactly the same dynamics as the real one (``Vehicle.step``).
        """
        steer_cmd, thb = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.allow_reverse:
            raise NotImplementedError("envelope shield is implemented for forward-only action semantics")
        sim = Vehicle(self.env_cfg["vehicle"], dt=self.dt)
        sim.reset(self.vehicle.x, self.vehicle.y, self.vehicle.theta)
        sim.v, sim.delta = self.vehicle.v, self.vehicle.delta
        xmin, xmax, ymin, ymax = self.scenario.bounds
        obstacles = self.scenario.obstacles
        # Optional safety margin: obstacles are inflated by `margin` metres on every side (a
        # conservative superset of the true margin neighbourhood). If the car is already closer than
        # the margin, it is not applied for this call, otherwise no action could ever be "safe".
        if margin > 0.0 and len(obstacles) and self.body_clearance() >= margin:
            obstacles = obstacles + np.array([-margin, -margin, margin, margin])

        def unsafe():
            if not (xmin <= sim.x <= xmax and ymin <= sim.y <= ymax):
                return True
            return bool(len(obstacles) and obb_aabb_overlap(sim.corners(), obstacles).any())

        sim.step(steer_cmd, max(thb, 0.0), max(-thb, 0.0))
        if unsafe():
            return False, 1
        hold = steer_cmd if hold_steer_cmd is None else float(hold_steer_cmd)
        for k in range(2, max_brake_steps + 2):
            if sim.v <= 0.0:
                break
            sim.step(hold, 0.0, 1.0)
            if unsafe():
                return False, k
        return True, None

    def slot_bearing_deg(self) -> float:
        """Bearing to the slot centre, CCW-positive (i.e. positive = steer_cmd should be positive
        to turn toward it, using the navigation steering convention). Used by
        fuzzy/fuzzy_only_controller.py for its navigation term -- kept separate from
        ``fuzzy_inputs()`` because the two need opposite sign conventions for different purposes.
        """
        sx, sy, _ = self.scenario.slot_pose
        bearing = np.arctan2(sy - self.vehicle.y, sx - self.vehicle.x) - self.vehicle.theta
        return float(np.rad2deg(wrap_angle(bearing)))
