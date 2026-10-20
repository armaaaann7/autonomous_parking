"""8-ray distance sensor array, cast from the vehicle's body centre.

Uses environment.geometry.raycast_aabb (vectorised NumPy) rather than shapely per the note in
geometry.py. tests/test_geometry.py cross-checks this against shapely on random scenes.
"""
from __future__ import annotations

import numpy as np

from .geometry import raycast_aabb


def cast_rays(x: float, y: float, theta: float, obstacles: np.ndarray,
              n_rays: int = 8, max_range: float = 10.0,
              noise_std_frac: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return an ``(n_rays,)`` array of distances, evenly spaced around the car, ray 0 pointing
    straight ahead (along ``theta``), going counter-clockwise.

    ``obstacles``: ``(M, 4)`` array of ``[xmin, ymin, xmax, ymax]`` (empty array is fine).
    ``noise_std_frac``: optional multiplicative Gaussian noise (domain-randomisation, Part 9).
    """
    angles = theta + np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    origins = np.tile([x, y], (n_rays, 1))
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 4)
    dist = raycast_aabb(origins, dirs, obstacles, max_range)
    if noise_std_frac > 0.0:
        rng = rng or np.random.default_rng()
        dist = np.clip(dist * (1.0 + rng.normal(0.0, noise_std_frac, size=dist.shape)), 0.0, max_range)
    return dist
