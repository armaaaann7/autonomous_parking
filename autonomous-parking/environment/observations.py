"""Observation representations for ParkingEnv.

``legacy`` (default, 14-d) is the original vector: [dx, dy, sin, cos, v, delta] + 8 rays, where (dx, dy)
= slot centre - car centre in the WORLD frame. Every checkpoint trained before the observation audit
expects it, so it stays the default.

``slot`` (18-d, experimental) expresses the goal geometry in frames a policy can generalise over:

 idx  feature                              frame / unit
  0   s_long  car centre along slot axis   slot frame, m   (0 at the goal; the car should face +axis)
  1   s_lat   car centre across slot axis  slot frame, m
  2   g_fwd   slot centre ahead of the car car frame, m     (what the steering needs)
  3   g_left  slot centre left of the car  car frame, m
  4   dist    |slot centre - car centre|    m
  5   sin(he) heading error, he = theta_car - theta_slot (wrapped)
  6   cos(he)
  7   v       signed speed, m/s
  8   delta   steering angle, rad
 9-16 rays    8 range readings, m (sensor noise as in legacy)
 17   clear   true body clearance to the nearest obstacle, clipped to [0, 5] m  (NOISE-FREE: privileged
              information, the same quantity the envelope shield uses; documented as such in docs)

No feature uses the scenario's future, the bay-protrusion/success flags, or the obstacle list directly.
Heading enters only through sin/cos, so there is no +-pi discontinuity.
"""
from __future__ import annotations

import numpy as np

SLOT_OBS_DIM = 18
LEGACY_OBS_DIM = 14
OBS_MODES = ("legacy", "slot")
CLEAR_CAP = 5.0


def slot_relative_features(car_xy, car_theta: float, slot_pose) -> np.ndarray:
    """The 7 pose features (idx 0-6): pure function of the car pose and the slot pose."""
    sx, sy, st = (float(v) for v in slot_pose)
    d = np.asarray(car_xy, dtype=float) - np.array([sx, sy])
    u, v = np.array([np.cos(st), np.sin(st)]), np.array([-np.sin(st), np.cos(st)])
    s_long, s_lat = float(d @ u), float(d @ v)
    g = -d                                                   # car -> slot, world frame
    c, s = np.cos(car_theta), np.sin(car_theta)
    g_fwd, g_left = c * g[0] + s * g[1], -s * g[0] + c * g[1]
    he = car_theta - st
    return np.array([s_long, s_lat, g_fwd, g_left, float(np.hypot(*g)), np.sin(he), np.cos(he)], dtype=np.float32)


def slot_observation(car_xy, car_theta, slot_pose, v, delta, rays, clearance) -> np.ndarray:
    return np.concatenate([slot_relative_features(car_xy, car_theta, slot_pose), [v, delta], np.asarray(rays, dtype=float),
                           [float(np.clip(clearance, 0.0, CLEAR_CAP))]]).astype(np.float32)
