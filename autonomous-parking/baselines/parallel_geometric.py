"""Geometric parallel-parking controller (reverse S-curve), used as (a) a feasibility proof for the
parallel scenario and (b) a strong, interpretable baseline.

With full steering lock the rear axle follows a circle of radius R = wheelbase / tan(max_steer).
Two equal reverse arcs (lock one way until the heading has turned by alpha, then lock the other way
until it is back to 0) shift the car sideways by d = 2R(1 - cos alpha) while moving 2R sin(alpha)
backwards, so alpha = arccos(1 - d / 2R) and the stopping spot is fixed by pure geometry.

Needs ``ParkingEnv({"allow_reverse": True})`` -- without a reverse gear this cannot work.
"""
from __future__ import annotations

import numpy as np


class ParallelGeometricController:
    name = "parallel_geometric"

    def __init__(self, creep_speed: float = 0.6, lane_offset: float = 2.3):
        self.creep = creep_speed
        # rear-axle distance from the slot line to drive along before reversing in. Smaller = smaller
        # sideways shift = less swing of the car's nose into the parked car ahead (0.9 m car half-width
        # + 0.9 m parked-car half-width + margin keeps this >= ~2.1).
        self.lane_offset = lane_offset
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.alpha = None
        self.x_stop = None

    @staticmethod
    def _speed_cmd(v, v_des):
        if v_des == 0.0 and abs(v) < 0.05:
            return 0.0
        return float(np.clip(2.5 * (v_des - v), -1.0, 1.0))

    def act(self, env):
        veh, scn = env.vehicle, env.scenario
        R = veh.wheelbase / np.tan(veh.max_steer)
        off = veh.centre_offset
        # target = rear axle position that puts the body centre on the slot pose
        tx = scn.slot_pose[0] - off * np.cos(scn.slot_pose[2])
        ty = scn.slot_pose[1] - off * np.sin(scn.slot_pose[2])
        th, v, delta = veh.theta, veh.v, veh.delta
        rx, ry = veh.rx, veh.ry
        steer, v_des = 0.0, 0.0

        if self.phase == "approach":
            d = max(ry - ty, 0.5)                      # actual offset once the lane change has settled
            self.alpha = float(np.arccos(np.clip(1.0 - d / (2 * R), -1, 1)))
            self.x_stop = tx + 2 * R * np.sin(self.alpha)
            dist = self.x_stop - rx
            steer = float(np.clip(-2.0 * th - 0.6 * (ry - (ty + self.lane_offset)), -1, 1))
            if dist <= 0.03:
                if abs(v) < 0.03:
                    self.phase = "lock1"
                v_des = 0.0
            else:
                v_des = max(0.3, min(1.8, np.sqrt(2 * 1.5 * dist)))
        elif self.phase == "lock1":
            steer = -1.0
            if abs(delta + veh.max_steer) < 1e-3:
                self.phase = "arc1"
        elif self.phase == "arc1":
            steer, v_des = -1.0, -self.creep
            if th >= self.alpha - 0.01:
                self.phase = "flip"
        elif self.phase == "flip":
            steer = 1.0
            if abs(v) < 0.03 and abs(delta - veh.max_steer) < 1e-3:
                self.phase = "arc2"
        elif self.phase == "arc2":
            steer, v_des = 1.0, -self.creep
            if th <= 0.01:
                self.phase = "straighten"
        elif self.phase == "straighten":
            steer = 0.0
            if abs(v) < 0.03 and abs(delta) < 1e-3:
                self.phase = "adjust"
        elif self.phase == "adjust":
            ex = tx - rx                                  # +ve = need to roll forward
            steer = float(np.clip(-3.0 * (th - scn.slot_pose[2]), -0.3, 0.3)) * (1 if ex >= 0 else -1)
            v_des = float(np.clip(1.0 * ex, -0.3, 0.3))
            if abs(ex) < 0.08 and abs(v) < 0.05:
                self.phase = "done"
                v_des = 0.0
        # blend the pre-clip phases that must hold still while wheels move
        if self.phase in ("lock1", "flip", "straighten"):
            v_des = 0.0
        u = self._speed_cmd(v, v_des)
        return np.array([steer, u], dtype=np.float32)
