"""Fast 2D geometry helpers (pure NumPy).

The blueprint suggests shapely for ray casting and collision checks. Shapely is convenient
but roughly 50-100x too slow for an RL inner loop (8 rays x ~25 obstacles per step), so the
hot path uses vectorised NumPy instead. shapely is still used in ``tests/`` as an independent
oracle to verify these routines.

Conventions
-----------
* World frame: x to the right, y up, angles counter-clockwise, radians.
* Obstacles are axis-aligned rectangles stored as rows ``[xmin, ymin, xmax, ymax]``.
* The car is an oriented rectangle (OBB).
"""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def obb_corners(cx: float, cy: float, theta: float, length: float, width: float) -> np.ndarray:
    """Corners (4, 2) of an oriented rectangle centred at (cx, cy), long axis along ``theta``."""
    c, s = np.cos(theta), np.sin(theta)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def obb_aabb_overlap(corners: np.ndarray, aabbs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Separating-axis test between one OBB and many AABBs.

    Parameters
    ----------
    corners : (4, 2) OBB corners.
    aabbs   : (M, 4) rows [xmin, ymin, xmax, ymax].

    Returns
    -------
    (M,) boolean array, True where the OBB overlaps the AABB (touching does not count).
    """
    if len(aabbs) == 0:
        return np.zeros(0, dtype=bool)

    # Axes 1-2: world x / y.
    cmin = corners.min(axis=0)
    cmax = corners.max(axis=0)
    hit = (
        (cmax[0] > aabbs[:, 0] + eps) & (cmin[0] < aabbs[:, 2] - eps)
        & (cmax[1] > aabbs[:, 1] + eps) & (cmin[1] < aabbs[:, 3] - eps)
    )
    if not hit.any():
        return hit

    # Axes 3-4: the OBB's own edge normals.
    e1 = corners[0] - corners[3]          # along the long axis
    e2 = corners[0] - corners[1]          # along the short axis
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)
    boxes = np.stack(
        [aabbs[:, [0, 1]], aabbs[:, [2, 1]], aabbs[:, [2, 3]], aabbs[:, [0, 3]]], axis=1
    )  # (M, 4, 2)
    for ax in (e1, e2):
        cproj = corners @ ax
        bproj = boxes @ ax                 # (M, 4)
        sep = (cproj.max() < bproj.min(axis=1) + eps) | (cproj.min() > bproj.max(axis=1) - eps)
        hit &= ~sep
    return hit


def raycast_aabb(origins: np.ndarray, dirs: np.ndarray, aabbs: np.ndarray, max_range: float) -> np.ndarray:
    """Distance from each ray origin to the first AABB hit (slab method), capped at ``max_range``.

    origins : (N, 2), dirs : (N, 2) unit vectors, aabbs : (M, 4).
    A ray starting inside a box returns 0.
    """
    n = len(origins)
    if len(aabbs) == 0:
        return np.full(n, max_range)
    d = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    inv = 1.0 / d                                             # (N, 2)
    ox = origins[:, 0:1]
    oy = origins[:, 1:2]
    tx1 = (aabbs[None, :, 0] - ox) * inv[:, 0:1]
    tx2 = (aabbs[None, :, 2] - ox) * inv[:, 0:1]
    ty1 = (aabbs[None, :, 1] - oy) * inv[:, 1:2]
    ty2 = (aabbs[None, :, 3] - oy) * inv[:, 1:2]
    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = (tmax >= 0.0) & (tmin <= tmax)
    t = np.where(hit, np.maximum(tmin, 0.0), np.inf)
    return np.minimum(t.min(axis=1), max_range)


def _point_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point in ``pts`` (N, 2) to the segment a-b (each (2,))."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-18:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def obb_aabb_distance(corners: np.ndarray, aabbs: np.ndarray) -> np.ndarray:
    """Exact minimum Euclidean distance between one OBB and each AABB (0 when they overlap or touch).

    For two disjoint convex polygons the minimum distance is attained at a vertex of one polygon
    and a point on an edge of the other, so it is the minimum over all vertex-to-edge distances in
    both directions. This is the car's *true body clearance* (unlike a ray cast from the body
    centre, which is blind to the car's 4.2 x 1.8 m footprint and to obstacles between rays).

    corners : (4, 2) OBB corners; aabbs : (M, 4) rows [xmin, ymin, xmax, ymax]. Returns (M,).
    """
    aabbs = np.asarray(aabbs, dtype=float).reshape(-1, 4)
    out = np.empty(len(aabbs))
    overlap = obb_aabb_overlap(corners, aabbs)
    for i, bx in enumerate(aabbs):
        if overlap[i]:
            out[i] = 0.0
            continue
        box = np.array([[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]])
        best = np.inf
        for k in range(4):
            best = min(best,
                       _point_segment_distance(corners, box[k], box[(k + 1) % 4]).min(),
                       _point_segment_distance(box, corners[k], corners[(k + 1) % 4]).min())
        out[i] = best
    return out
