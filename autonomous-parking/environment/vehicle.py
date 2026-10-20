"""Kinematic bicycle model (rear-axle reference).

Differences from the blueprint's Part 7 / Part 16 sketch (see docs/REVIEW.md):

* The blueprint says "front-axle reference" but writes the *rear-axle* equations; this file is
  consistently rear-axle. The public ``x, y`` are the **body centre** (used for goal errors,
  rendering and collision); the rear axle is kept internally.
* Steering is a rate-limited actuator, so the steering angle ``delta`` is a genuine state
  variable (the blueprint's observation had a hard-coded 0.0 in that slot).
* There is a separate ``brake`` input that decelerates the car *towards zero speed but never
  through it*. The blueprint's shield code (``accel - brake``) could push a braking car into
  reverse. Rolling resistance lets the car coast to a stop when the throttle is released.
* Pose is integrated with mid-point heading and the *updated* speed (semi-implicit Euler), so a
  command affects the pose in the same step.
"""
from __future__ import annotations

import numpy as np

from .geometry import obb_corners, wrap_angle


class Vehicle:
    def __init__(self, cfg: dict | None = None, dt: float = 0.1):
        cfg = cfg or {}
        self.length = cfg.get("length", 4.2)
        self.width = cfg.get("width", 1.8)
        self.wheelbase = cfg.get("wheelbase", 2.5)
        self.rear_overhang = cfg.get("rear_overhang", 0.85)
        self.max_speed = cfg.get("max_speed", 2.0)
        self.max_steer = np.deg2rad(cfg.get("max_steer_deg", 35.0))
        self.steer_rate = cfg.get("steer_rate", 2.0)
        self.max_accel = cfg.get("max_accel", 2.0)
        self.max_brake = cfg.get("max_brake", 4.0)
        self.drag = cfg.get("drag", 0.5)
        self.dt = dt
        # distance from rear axle to body centre along the heading
        self.centre_offset = self.length / 2.0 - self.rear_overhang
        self.reset(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ state
    def reset(self, cx: float, cy: float, theta: float) -> None:
        """Place the car by its **body centre** pose, at rest, wheels straight."""
        self.theta = float(theta)
        self.rx = cx - self.centre_offset * np.cos(theta)
        self.ry = cy - self.centre_offset * np.sin(theta)
        self.v = 0.0
        self.delta = 0.0

    @property
    def x(self) -> float:
        return self.rx + self.centre_offset * np.cos(self.theta)

    @property
    def y(self) -> float:
        return self.ry + self.centre_offset * np.sin(self.theta)

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def corners(self) -> np.ndarray:
        return obb_corners(self.x, self.y, self.theta, self.length, self.width)

    # ------------------------------------------------------------------- step
    def step(self, steer_cmd: float, accel_cmd: float, brake_cmd: float = 0.0):
        """Advance one ``dt``. All commands in [-1, 1] (brake in [0, 1])."""
        dt = self.dt
        steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))
        accel_cmd = float(np.clip(accel_cmd, -1.0, 1.0))
        brake_cmd = float(np.clip(brake_cmd, 0.0, 1.0))

        # steering actuator with slew-rate limit
        target = steer_cmd * self.max_steer
        max_d = self.steer_rate * dt
        self.delta += float(np.clip(target - self.delta, -max_d, max_d))

        # longitudinal dynamics: throttle, rolling resistance, brake
        v = self.v + accel_cmd * self.max_accel * dt
        v -= np.sign(v) * min(abs(v), self.drag * dt)
        v -= np.sign(v) * min(abs(v), brake_cmd * self.max_brake * dt)
        if abs(v) < 1e-3:
            v = 0.0
        v = float(np.clip(v, -self.max_speed, self.max_speed))

        # kinematics (rear axle)
        dtheta = v / self.wheelbase * np.tan(self.delta) * dt
        th_mid = self.theta + 0.5 * dtheta
        self.rx += v * np.cos(th_mid) * dt
        self.ry += v * np.sin(th_mid) * dt
        self.theta = float(wrap_angle(self.theta + dtheta))
        self.v = v
        return self.x, self.y, self.theta, self.v
