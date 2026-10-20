"""Cross-checks environment/geometry.py's fast NumPy routines against shapely, which we trust
but didn't want in the RL inner loop (see geometry.py's module docstring)."""
import numpy as np
import pytest
from shapely.geometry import box as shapely_box
from shapely.affinity import rotate as shapely_rotate

from environment.geometry import obb_aabb_overlap, obb_corners, raycast_aabb


def _shapely_obb(cx, cy, theta, length, width):
    poly = shapely_box(cx - length / 2, cy - width / 2, cx + length / 2, cy + width / 2)
    return shapely_rotate(poly, np.degrees(theta), origin=(cx, cy))


@pytest.mark.parametrize("seed", range(20))
def test_obb_aabb_overlap_matches_shapely(seed):
    rng = np.random.default_rng(seed)
    cx, cy, theta = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi)
    length, width = 4.2, 1.8
    corners = obb_corners(cx, cy, theta, length, width)
    car_poly = _shapely_obb(cx, cy, theta, length, width)

    aabbs = rng.uniform(-6, 6, size=(5, 2))
    aabbs = np.concatenate([aabbs, aabbs + rng.uniform(0.5, 3, size=(5, 2))], axis=1)
    aabbs = np.stack([np.minimum(aabbs[:, 0], aabbs[:, 2]), np.minimum(aabbs[:, 1], aabbs[:, 3]),
                       np.maximum(aabbs[:, 0], aabbs[:, 2]), np.maximum(aabbs[:, 1], aabbs[:, 3])], axis=1)

    got = obb_aabb_overlap(corners, aabbs)
    for i, aabb in enumerate(aabbs):
        box_poly = shapely_box(*aabb)
        expected = car_poly.intersects(box_poly) and car_poly.intersection(box_poly).area > 1e-9
        assert bool(got[i]) == expected, f"seed={seed} box={i} aabb={aabb}"


def test_raycast_matches_brute_force_slab():
    """Sanity-checks the vectorised raycast against a simple independent per-ray loop."""
    rng = np.random.default_rng(0)
    aabbs = np.array([[1.0, -1.0, 3.0, 1.0], [-4.0, 2.0, -2.0, 4.0]])
    origins = rng.uniform(-5, 5, size=(30, 2))
    dirs = rng.normal(size=(30, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    max_range = 10.0

    got = raycast_aabb(origins, dirs, aabbs, max_range)

    for i in range(len(origins)):
        best = max_range
        for box in aabbs:
            inv = 1.0 / np.where(np.abs(dirs[i]) < 1e-12, 1e-12, dirs[i])
            t1 = (box[[0, 1]] - origins[i]) * inv
            t2 = (box[[2, 3]] - origins[i]) * inv
            tmin = np.maximum(np.minimum(t1, t2).min(), 0.0) if np.minimum(t1, t2).max() <= np.maximum(t1, t2).min() else None
            tnear = np.minimum(t1, t2).max()
            tfar = np.maximum(t1, t2).min()
            if tfar >= 0 and tnear <= tfar:
                best = min(best, max(tnear, 0.0))
        assert abs(got[i] - best) < 1e-6, f"ray {i}: got {got[i]} expected {best}"
