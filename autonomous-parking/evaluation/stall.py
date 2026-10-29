"""Measurable stall / no-progress criterion (RL audit round 2).

A step t (t >= WINDOW) is a *no-progress step* when, over the last WINDOW steps (3.0 s at dt = 0.1):
  * distance to the slot centre fell by less than PROGRESS_M (0.10 m), AND
  * |heading error| fell by less than PROGRESS_DEG (2 deg), AND
  * the car is not currently inside the success region (pose box or whole car in bay with heading/speed ok).
The episode is *stalled* from the first no-progress step (onset). Subtype at onset, from the window:
  * 'stationary'  mean |v| < 0.05 m/s                              (sitting still: braking / zero throttle)
  * 'oscillating' >= 2 sign changes of v, or >= 4 sign changes of the steering command with |steer| > 0.2
  * 'wandering'   moving but not getting closer (circling / creeping sideways)
Recorded context at onset: distance, slot-frame position (long, lat), heading error, body clearance, speed,
fraction of window steps with brake > 0 (final action), fraction shield-active, mean throttle command.
Thresholds are fixed constants chosen before looking at any outcome; tests/test_stall.py pins them.
"""
from __future__ import annotations

from collections import deque

import numpy as np

WINDOW = 30
PROGRESS_M = 0.10
PROGRESS_DEG = 2.0
STATIONARY_V = 0.05

FIELDS = ["stall_onset_step", "stall_steps", "stall_type", "stall_dist_m", "stall_long_m", "stall_lat_m",
          "stall_heading_deg", "stall_clear_m", "stall_speed", "stall_brake_frac", "stall_shield_frac", "stall_throttle_mean"]


class StallDetector:
    def __init__(self, window: int = WINDOW, progress_m: float = PROGRESS_M, progress_deg: float = PROGRESS_DEG):
        self.window, self.progress_m, self.progress_deg = window, progress_m, progress_deg
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=self.window + 1)
        self.t = 0
        self.onset = None
        self.n_stalled = 0
        self.context = {}
        self.currently_stalled = False

    def update(self, env, requested_action, step_info, shield_active: bool) -> bool:
        """Call once per env step AFTER env.step. Returns True if this step is a no-progress step."""
        self.t += 1
        dist = float(step_info["pos_error"])
        head = float(step_info["heading_error_deg"])
        _, accel, brake = step_info["action_final"]
        in_goal = bool(step_info.get("pose_ok", False)) or (bool(step_info.get("contained", False)) and head < np.rad2deg(env.heading_tol)
                                                               and abs(env.vehicle.v) < env.speed_tol)
        self.hist.append({"dist": dist, "head": head, "v": float(env.vehicle.v), "steer": float(requested_action[0]),
                          "thr": float(requested_action[1]), "brake": float(brake) > 1e-6, "shield": bool(shield_active)})
        stalled = False
        if len(self.hist) == self.window + 1 and not in_goal:
            h0, h1 = self.hist[0], self.hist[-1]
            stalled = (h0["dist"] - h1["dist"] < self.progress_m) and (h0["head"] - h1["head"] < self.progress_deg)
        self.currently_stalled = stalled
        if stalled:
            self.n_stalled += 1
            if self.onset is None:
                self.onset = self.t
                w = list(self.hist)[1:]
                vs = np.array([x["v"] for x in w])
                st = np.array([x["steer"] for x in w])
                v_flips = int(np.sum(np.diff(np.sign(vs[np.abs(vs) > 0.02])) != 0)) if np.sum(np.abs(vs) > 0.02) > 1 else 0
                big = st[np.abs(st) > 0.2]
                s_flips = int(np.sum(np.diff(np.sign(big)) != 0)) if len(big) > 1 else 0
                if np.mean(np.abs(vs)) < STATIONARY_V:
                    kind = "stationary"
                elif v_flips >= 2 or s_flips >= 4:
                    kind = "oscillating"
                else:
                    kind = "wandering"
                lng, lat = env.slot_frame_error()
                self.context = {"stall_type": kind, "stall_dist_m": dist, "stall_long_m": lng, "stall_lat_m": lat,
                                "stall_heading_deg": head, "stall_clear_m": float(env.body_clearance()),
                                "stall_speed": float(env.vehicle.v),
                                "stall_brake_frac": float(np.mean([x["brake"] for x in w])),
                                "stall_shield_frac": float(np.mean([x["shield"] for x in w])),
                                "stall_throttle_mean": float(np.mean([x["thr"] for x in w]))}
        return stalled

    def summary(self) -> dict:
        out = {k: "" for k in FIELDS}
        out["stall_onset_step"] = self.onset if self.onset is not None else ""
        out["stall_steps"] = self.n_stalled
        out["stall_type"] = self.context.get("stall_type", "none") if self.onset is not None else "none"
        out.update({k: v for k, v in self.context.items() if k != "stall_type"})
        return out
