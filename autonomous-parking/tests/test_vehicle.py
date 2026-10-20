import numpy as np
import pytest

from environment.vehicle import Vehicle

CFG = dict(length=4.2, width=1.8, wheelbase=2.5, rear_overhang=0.85, max_speed=2.0,
           max_steer_deg=35.0, steer_rate=2.0, max_accel=2.0, max_brake=4.0, drag=0.5)


def test_straight_line_no_steer():
    v = Vehicle(CFG, dt=0.1)
    v.reset(0.0, 0.0, 0.0)
    for _ in range(20):
        v.step(steer_cmd=0.0, accel_cmd=1.0)
    assert v.y == pytest.approx(0.0, abs=1e-9)
    assert v.x > 0.0


def test_reverse_moves_backward():
    v = Vehicle(CFG, dt=0.1)
    v.reset(0.0, 0.0, 0.0)
    for _ in range(20):
        v.step(steer_cmd=0.0, accel_cmd=-1.0)
    assert v.x < 0.0
    assert v.v < 0.0


def test_brake_never_reverses_without_throttle():
    """Braking alone should never push the car through zero into the opposite direction."""
    v = Vehicle(CFG, dt=0.1)
    v.reset(0.0, 0.0, 0.0)
    for _ in range(5):
        v.step(steer_cmd=0.0, accel_cmd=1.0)
    assert v.v > 0.0
    for _ in range(50):
        v.step(steer_cmd=0.0, accel_cmd=0.0, brake_cmd=1.0)
        assert v.v >= 0.0, "brake pushed speed negative"
    assert v.v == 0.0


def test_steering_is_rate_limited():
    v = Vehicle(CFG, dt=0.1)
    v.reset(0.0, 0.0, 0.0)
    v.step(steer_cmd=1.0, accel_cmd=0.0)
    max_delta_change = v.steer_rate * v.dt
    assert abs(v.delta) <= max_delta_change + 1e-9


def test_left_steer_turns_left_ccw_positive_convention():
    v = Vehicle(CFG, dt=0.1)
    v.reset(0.0, 0.0, 0.0)
    for _ in range(30):
        v.step(steer_cmd=1.0, accel_cmd=0.5)
    assert v.theta > 0.0, "positive steer should increase heading (CCW) when moving forward"


def test_speed_never_exceeds_max():
    v = Vehicle(CFG, dt=0.1)
    v.reset(0.0, 0.0, 0.0)
    for _ in range(200):
        v.step(steer_cmd=0.0, accel_cmd=1.0)
    assert abs(v.v) <= v.max_speed + 1e-9
