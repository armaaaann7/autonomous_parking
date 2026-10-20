"""Scenario generation (blueprint Part 8 + Part 9).

A ``Scenario`` bundles: the target slot pose, the car's start pose, and the obstacle rectangles.
``RandomizationLevel`` controls which axes are randomised, matching the curriculum in Part 9:
start pose -> slot position -> obstacles -> sensor noise, each layered on top of the last.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np


class Kind(enum.Enum):
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"


class RandomizationLevel(enum.IntEnum):
    """Ordered so ``level >= X`` composes; matches Part 9's priority order."""
    NONE = 0
    START_POSE = 1
    SLOT_POSITION = 2
    OBSTACLES = 3
    SENSOR_NOISE = 4  # sensor noise itself is applied in sensors.py, not here


@dataclass
class Scenario:
    kind: Kind
    slot_pose: np.ndarray          # [x, y, theta]
    start_pose: np.ndarray         # [x, y, theta]
    obstacles: np.ndarray          # (M, 4) [xmin, ymin, xmax, ymax]
    noise_std_frac: float = 0.0
    bounds: tuple = (-15.0, 15.0, -15.0, 15.0)  # xmin, xmax, ymin, ymax, for out-of-bounds check


def _slot_rect(pose: np.ndarray, length: float, width: float) -> np.ndarray:
    x, y, theta = pose
    s = np.sin(theta)
    # Slot footprint aligned with theta; for our fixed 0/90 deg slots this stays axis-aligned.
    if abs(s) < 0.5:
        return np.array([x - length / 2, y - width / 2, x + length / 2, y + width / 2])
    return np.array([x - width / 2, y - length / 2, x + width / 2, y + length / 2])


def make_scenario(kind: Kind, level: RandomizationLevel, rng: np.random.Generator,
                   car_length: float = 4.2, car_width: float = 1.8,
                   include_obstacles: bool = True, start_distance_scale: float = 1.0,
                   parallel_gap: float | None = None) -> Scenario:
    """Build one scenario instance. Geometry loosely mirrors real parking-bay dimensions.

    ``include_obstacles=False`` gives Part 8's Level-1 "empty environment" scenario: this is a
    *separate* axis from ``level`` (domain-randomisation amount) because the blueprint's own
    training plan (Part 15, Phase 6) calls for proving the reward/environment plumbing works with
    no obstacles at all before obstacles are introduced -- conflating "no obstacles" with
    "level == NONE" (as an earlier version of this function did) meant PPO only ever saw the
    hardest variant (obstacles present, nothing randomised) and converged to a "never move, never
    risk the -collision penalty" degenerate policy. See docs/REVIEW.md.
    """
    slot_margin = 0.6  # extra length/width so a well-aligned car fits with clearance
    bay_len = car_length + slot_margin
    bay_wid = car_width + slot_margin

    if kind is Kind.PARALLEL:
        slot_xy = np.array([0.0, 0.0])
        slot_theta = 0.0
        approach_offset = np.array([-6.0, 3.0])   # start in the lane, offset from the slot
        start_theta_nom = 0.0
        # Free length between the two parked cars. Default (None) keeps the original 5.4 m, which is
        # too tight for a car with a 3.6 m turning radius to enter in one reverse manoeuvre; pass
        # a larger value (e.g. 6.6) for a realistic ~1.6x-car-length bay. See docs/REVIEW.md.
        gap = (2 * bay_len - car_length) if parallel_gap is None else float(parallel_gap)
        off = (gap + car_length) / 2.0
        neighbor_positions = [np.array([-off, 0.0]), np.array([off, 0.0])]
        neighbor_dims = (car_length, car_width)
    else:
        slot_xy = np.array([0.0, 0.0])
        slot_theta = np.pi / 2
        approach_offset = np.array([0.0, -6.0])
        start_theta_nom = np.pi / 2
        neighbor_positions = [np.array([-bay_wid, 0.0]), np.array([bay_wid, 0.0])]
        neighbor_dims = (car_width, car_length)  # perpendicular neighbours are rotated 90 deg

    if level >= RandomizationLevel.SLOT_POSITION:
        slot_xy = slot_xy + rng.uniform(-1.0, 1.0, size=2)

    slot_pose = np.array([slot_xy[0], slot_xy[1], slot_theta])

    # start_distance_scale shrinks the approach distance (curriculum bootstrap, see
    # docs/REVIEW.md): at 1.0 this is the nominal 6m approach; near 0 the car starts almost in
    # the slot already, so an untrained/random policy succeeds often enough to get a first
    # positive-reward signal to learn from.
    start_xy = slot_xy + approach_offset * float(np.clip(start_distance_scale, 0.0, 1.0))
    start_theta = start_theta_nom
    if level >= RandomizationLevel.START_POSE:
        start_xy = start_xy + rng.uniform(-1.0, 1.0, size=2)
        start_theta = start_theta_nom + rng.uniform(-np.deg2rad(15), np.deg2rad(15))

    if include_obstacles:
        obstacles = []
        for off, (nl, nw) in zip(neighbor_positions, [neighbor_dims, neighbor_dims]):
            pos = slot_xy + off
            if level >= RandomizationLevel.OBSTACLES:
                pos = pos + rng.uniform(-0.3, 0.3, size=2)
            obstacles.append(np.array([pos[0] - nl / 2, pos[1] - nw / 2, pos[0] + nl / 2, pos[1] + nw / 2]))
        obstacles = np.array(obstacles)
    else:
        obstacles = np.zeros((0, 4))

    noise = 0.03 if level >= RandomizationLevel.SENSOR_NOISE else 0.0

    return Scenario(
        kind=kind,
        slot_pose=slot_pose,
        start_pose=np.array([start_xy[0], start_xy[1], start_theta]),
        obstacles=obstacles,
        noise_std_frac=noise,
    )


def slot_rectangle(scn: Scenario, car_length: float = 4.2, car_width: float = 1.8) -> np.ndarray:
    """Outline rectangle for rendering the target slot."""
    return _slot_rect(scn.slot_pose, car_length + 0.6, car_width + 0.6)
