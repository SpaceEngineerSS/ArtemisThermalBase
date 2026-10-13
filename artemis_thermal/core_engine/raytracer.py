"""BVH-accelerated raytracer with Möller-Trumbore intersection.

Implements a flattened (linear) Bounding Volume Hierarchy for efficient
shadow ray queries on DEM-derived triangle meshes. All inner-loop
functions are compiled with Numba ``@njit(cache=True)`` for performance.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

Design Notes
------------
- **Flattened BVH**: Nodes are stored in contiguous 1D float64 arrays
  (no Python objects, no recursion in traversal) for Numba compatibility
  and cache locality.
- **Node layout** (8 doubles per node):
  ``[bbox_min_x, min_y, min_z, bbox_max_x, max_y, max_z, child_or_start, count_or_right]``
  - If ``count_or_right < 0``: leaf node → ``child_or_start`` = first triangle index,
    ``|count_or_right|`` = number of triangles.
  - If ``count_or_right >= 0``: internal node → ``child_or_start`` = left child node index,
    ``count_or_right`` = right child node index.
- **Precision**: float64 throughout; ε = 1e-10 for zero-tests.

References
----------
- Möller, T. & Trumbore, B. (1997). "Fast, Minimum Storage Ray-Triangle
  Intersection." J. Graphics Tools, 2(1), 21-28.
- Wald, I. (2007). "On fast Construction of SAH-based Bounding Volume
  Hierarchies." Proc. IEEE Symp. Interactive Ray Tracing, pp. 33-40.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    prange = range
else:
    from numba import prange

from artemis_thermal.core_engine.mesh import TriangleMesh

logger = logging.getLogger(__name__)

# ===================================================================
# Constants (loaded from config at runtime, but these are compile-time
# defaults for Numba. The build_bvh function injects the actual values.)
# ===================================================================

_DEFAULT_EPSILON: float = 1e-10
_DEFAULT_MAX_LEAF: int = 4
_INF: float = 1e30

# ===================================================================
# NODE LAYOUT — indices into the flat node array
# ===================================================================
_BBOX_MIN_X = 0
_BBOX_MIN_Y = 1
_BBOX_MIN_Z = 2
_BBOX_MAX_X = 3
_BBOX_MAX_Y = 4
_BBOX_MAX_Z = 5
_CHILD_OR_START = 6
_COUNT_OR_RIGHT = 7
_NODE_SIZE = 8  # floats per node


# ===================================================================
# MÖLLER-TRUMBORE RAY-TRIANGLE INTERSECTION — Numba JIT
# ===================================================================


@njit(cache=True, fastmath=False)
def moller_trumbore(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    epsilon: float,
) -> float:
    """Möller-Trumbore ray-triangle intersection test.

    Tests if a ray R(t) = origin + t * dir intersects the triangle
    defined by vertices v0, v1, v2. Returns the parametric distance t
    if hit, or -1.0 if miss.

    Parameters
    ----------
    ray_origin : np.ndarray
        Ray origin point [x, y, z]. Shape: (3,).
    ray_dir : np.ndarray
        Ray direction vector [dx, dy, dz]. Shape: (3,). Need not be normalized.
    v0, v1, v2 : np.ndarray
        Triangle vertex positions. Shape: (3,) each.
    epsilon : float
        Zero-test tolerance to prevent edge leakage.

    Returns
    -------
    float
        Parametric distance t > 0 if hit, -1.0 if no intersection.

    Notes
    -----
    ``fastmath=False`` is CRITICAL to prevent the compiler from reordering
    floating-point operations, which would break the epsilon comparisons
    and cause ray leakage at triangle edges — fatal for PSR accuracy.

    """
    # Edge vectors
    e1_x = v1[0] - v0[0]
    e1_y = v1[1] - v0[1]
    e1_z = v1[2] - v0[2]

    e2_x = v2[0] - v0[0]
    e2_y = v2[1] - v0[1]
    e2_z = v2[2] - v0[2]

    # P = ray_dir × e2
    p_x = ray_dir[1] * e2_z - ray_dir[2] * e2_y
    p_y = ray_dir[2] * e2_x - ray_dir[0] * e2_z
    p_z = ray_dir[0] * e2_y - ray_dir[1] * e2_x

    # Determinant = e1 · P
    det = e1_x * p_x + e1_y * p_y + e1_z * p_z

    # If determinant is near zero, ray is parallel to triangle plane
    if det > -epsilon and det < epsilon:
        return -1.0

    inv_det = 1.0 / det

    # T = ray_origin - v0
    t_x = ray_origin[0] - v0[0]
    t_y = ray_origin[1] - v0[1]
    t_z = ray_origin[2] - v0[2]

    # u = (T · P) * inv_det — first barycentric coordinate
    u = (t_x * p_x + t_y * p_y + t_z * p_z) * inv_det

    if u < -epsilon or u > 1.0 + epsilon:
        return -1.0

    # Q = T × e1
    q_x = t_y * e1_z - t_z * e1_y
    q_y = t_z * e1_x - t_x * e1_z
    q_z = t_x * e1_y - t_y * e1_x

    # v = (ray_dir · Q) * inv_det — second barycentric coordinate
    v = (ray_dir[0] * q_x + ray_dir[1] * q_y + ray_dir[2] * q_z) * inv_det

    if v < -epsilon or u + v > 1.0 + epsilon:
        return -1.0

    # t = (e2 · Q) * inv_det — parametric distance along ray
    t_dist = (e2_x * q_x + e2_y * q_y + e2_z * q_z) * inv_det

    if t_dist > epsilon:
        return t_dist

    return -1.0


# ===================================================================
# RAY-AABB INTERSECTION — Slab Method (Numba JIT)
# ===================================================================


@njit(cache=True, fastmath=False)
def ray_aabb_intersect(
    ray_origin: np.ndarray,
    inv_dir: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    t_max_limit: float,
) -> bool:
    """Test if a ray intersects an axis-aligned bounding box.

    Uses the slab method with precomputed inverse direction to avoid
    division. Handles rays parallel to slab planes via IEEE 754 infinity.

    Parameters
    ----------
    ray_origin : np.ndarray
        Ray origin [x, y, z]. Shape: (3,).
    inv_dir : np.ndarray
        Precomputed 1.0 / ray_dir for each axis. Shape: (3,).
    bbox_min : np.ndarray
        AABB minimum corner. Shape: (3,).
    bbox_max : np.ndarray
        AABB maximum corner. Shape: (3,).
    t_max_limit : float
        Maximum parametric distance (for early culling).

    Returns
    -------
    bool
        True if the ray intersects the AABB within [0, t_max_limit].

    """
    t_min = 0.0
    t_max = t_max_limit

    for axis in range(3):
        t1 = (bbox_min[axis] - ray_origin[axis]) * inv_dir[axis]
        t2 = (bbox_max[axis] - ray_origin[axis]) * inv_dir[axis]

        # Swap so t1 <= t2
        if t1 > t2:
            t1, t2 = t2, t1

        if t1 > t_min:
            t_min = t1
        if t2 < t_max:
            t_max = t2

        if t_min > t_max:
            return False

    return True


# ===================================================================
# BVH TRAVERSAL — iterative, allocation-free (Numba JIT)
# ===================================================================
#
# The hot kernels below run inside ``prange`` loops. Everything they need
# per ray lives in scalars, and one traversal stack is shared by a chunk
# of faces, so no heap allocation happens per ray. The build caps
# tree depth at ``_MAX_DEPTH``, which bounds the stack.

_MAX_DEPTH = 96
_STACK_SIZE = _MAX_DEPTH + 2
_SAH_DEPTH_LIMIT = 64  # below this depth, fall back to median splits
_CHUNK = 256  # faces per parallel work item; one traversal stack each


@njit(cache=True, fastmath=False, inline="always")
def _slab_hit(
    nodes: np.ndarray,
    base: int,
    ox: float, oy: float, oz: float,
    ix: float, iy: float, iz: float,
    t_limit: float,
) -> bool:
    t_min = 0.0
    t_max = t_limit
    t1 = (nodes[base + _BBOX_MIN_X] - ox) * ix
    t2 = (nodes[base + _BBOX_MAX_X] - ox) * ix
    if t1 > t2:
        t1, t2 = t2, t1
    t_min = max(t_min, t1)
    t_max = min(t_max, t2)
    if t_min > t_max:
        return False
    t1 = (nodes[base + _BBOX_MIN_Y] - oy) * iy
    t2 = (nodes[base + _BBOX_MAX_Y] - oy) * iy
    if t1 > t2:
        t1, t2 = t2, t1
    t_min = max(t_min, t1)
    t_max = min(t_max, t2)
    if t_min > t_max:
        return False
    t1 = (nodes[base + _BBOX_MIN_Z] - oz) * iz
    t2 = (nodes[base + _BBOX_MAX_Z] - oz) * iz
    if t1 > t2:
        t1, t2 = t2, t1
    t_min = max(t_min, t1)
    t_max = min(t_max, t2)
    return t_min <= t_max


@njit(cache=True, fastmath=False, inline="always")
def _tri_hit(
    tri_verts: np.ndarray,
    tri: int,
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    epsilon: float,
) -> float:
    """Scalar Möller-Trumbore against triangle ``tri``; same tests as above."""
    v0x = tri_verts[tri, 0, 0]
    v0y = tri_verts[tri, 0, 1]
    v0z = tri_verts[tri, 0, 2]
    e1x = tri_verts[tri, 1, 0] - v0x
    e1y = tri_verts[tri, 1, 1] - v0y
    e1z = tri_verts[tri, 1, 2] - v0z
    e2x = tri_verts[tri, 2, 0] - v0x
    e2y = tri_verts[tri, 2, 1] - v0y
    e2z = tri_verts[tri, 2, 2] - v0z
    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if det > -epsilon and det < epsilon:
        return -1.0
    inv_det = 1.0 / det
    tx = ox - v0x
    ty = oy - v0y
    tz = oz - v0z
    u = (tx * px + ty * py + tz * pz) * inv_det
    if u < -epsilon or u > 1.0 + epsilon:
        return -1.0
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv_det
    if v < -epsilon or u + v > 1.0 + epsilon:
        return -1.0
    t_dist = (e2x * qx + e2y * qy + e2z * qz) * inv_det
    if t_dist > epsilon:
        return t_dist
    return -1.0


@njit(cache=True, fastmath=False, inline="always")
def _inverse(d: float) -> float:
    return _INF if d == 0.0 else 1.0 / d


@njit(cache=True, fastmath=False)
def _occluded(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered: np.ndarray,
    epsilon: float,
    stack: np.ndarray,
) -> bool:
    """True if the ray hits any triangle (early exit)."""
    ix = _inverse(dx)
    iy = _inverse(dy)
    iz = _inverse(dz)
    sp = 1
    stack[0] = 0
    while sp > 0:
        sp -= 1
        base = stack[sp] * _NODE_SIZE
        if not _slab_hit(nodes, base, ox, oy, oz, ix, iy, iz, _INF):
            continue
        count_or_right = nodes[base + _COUNT_OR_RIGHT]
        if count_or_right < 0:
            start = int(nodes[base + _CHILD_OR_START])
            for k in range(start, start - int(count_or_right)):
                if _tri_hit(tri_verts, ordered[k], ox, oy, oz, dx, dy, dz, epsilon) > epsilon:
                    return True
        else:
            stack[sp] = int(nodes[base + _CHILD_OR_START])
            stack[sp + 1] = int(count_or_right)
            sp += 2
    return False


@njit(cache=True, fastmath=False)
def _closest(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered: np.ndarray,
    epsilon: float,
    stack: np.ndarray,
) -> tuple[int, float]:
    """Closest triangle hit by the ray, or (-1, INF)."""
    ix = _inverse(dx)
    iy = _inverse(dy)
    iz = _inverse(dz)
    best_t = _INF
    best_tri = -1
    sp = 1
    stack[0] = 0
    while sp > 0:
        sp -= 1
        base = stack[sp] * _NODE_SIZE
        if not _slab_hit(nodes, base, ox, oy, oz, ix, iy, iz, best_t):
            continue
        count_or_right = nodes[base + _COUNT_OR_RIGHT]
        if count_or_right < 0:
            start = int(nodes[base + _CHILD_OR_START])
            for k in range(start, start - int(count_or_right)):
                tri = ordered[k]
                t_hit = _tri_hit(tri_verts, tri, ox, oy, oz, dx, dy, dz, epsilon)
                if t_hit > epsilon and t_hit < best_t:
                    best_t = t_hit
                    best_tri = int(tri)
        else:
            stack[sp] = int(nodes[base + _CHILD_OR_START])
            stack[sp + 1] = int(count_or_right)
            sp += 2
    return best_tri, best_t


@njit(cache=True, fastmath=False)
def _shadow_ray_bvh(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    bvh_nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered_tri_indices: np.ndarray,
    epsilon: float,
) -> bool:
    """Array-argument wrapper of :func:`_occluded` for tests and tools."""
    stack = np.empty(_STACK_SIZE, dtype=np.int64)
    return _occluded(
        ray_origin[0], ray_origin[1], ray_origin[2],
        ray_dir[0], ray_dir[1], ray_dir[2],
        bvh_nodes, tri_verts, ordered_tri_indices, epsilon, stack,
    )


@njit(cache=True, fastmath=False)
def _closest_hit_bvh(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    bvh_nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered_tri_indices: np.ndarray,
    epsilon: float,
) -> tuple[int, float]:
    """Array-argument wrapper of :func:`_closest` used by view factors."""
    stack = np.empty(_STACK_SIZE, dtype=np.int64)
    return _closest(
        ray_origin[0], ray_origin[1], ray_origin[2],
        ray_dir[0], ray_dir[1], ray_dir[2],
        bvh_nodes, tri_verts, ordered_tri_indices, epsilon, stack,
    )


# ===================================================================
# SHADOW KERNELS
# ===================================================================


@njit(cache=True, parallel=True, fastmath=False)
def compute_shadow_map_point_source(
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
    sun_dir: np.ndarray,
    bvh_nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered_tri_indices: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Binary shadow map (1 lit, 0 shadowed) for a point-source Sun.

    Each face casts one ray from its centroid, nudged along the normal,
    toward the Sun. Faces turned away from the Sun are dark without a ray.
    """
    num_faces = face_centroids.shape[0]
    illumination = np.zeros(num_faces, dtype=np.float64)
    dx, dy, dz = sun_dir[0], sun_dir[1], sun_dir[2]
    offset = epsilon * 100.0
    num_chunks = (num_faces + _CHUNK - 1) // _CHUNK
    for chunk in prange(num_chunks):
        stack = np.empty(_STACK_SIZE, dtype=np.int64)
        for i in range(chunk * _CHUNK, min(num_faces, (chunk + 1) * _CHUNK)):
            nx, ny, nz = face_normals[i, 0], face_normals[i, 1], face_normals[i, 2]
            if nx * dx + ny * dy + nz * dz <= 0.0:
                continue
            hit = _occluded(
                face_centroids[i, 0] + nx * offset,
                face_centroids[i, 1] + ny * offset,
                face_centroids[i, 2] + nz * offset,
                dx, dy, dz,
                bvh_nodes, tri_verts, ordered_tri_indices, epsilon, stack,
            )
            if not hit:
                illumination[i] = 1.0
    return illumination


@njit(cache=True, parallel=True, fastmath=False)
def compute_extended_source_flux_factors(
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
    sun_samples: np.ndarray,
    bvh_nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered_tri_indices: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Visible fraction of the solar disk and disk-integrated cosine per face.

    The projected factor is ``mean(max(0, n·s) · visible(s))`` over
    equal-solid-angle samples of the disk. That is the quantity that
    multiplies irradiance; centre cosine × visible fraction is biased near
    the horizon and on steep facets.
    """
    num_faces = face_centroids.shape[0]
    num_samples = sun_samples.shape[0]
    inv_samples = 1.0 / float(num_samples)
    illumination = np.zeros(num_faces, dtype=np.float64)
    projected_factor = np.zeros(num_faces, dtype=np.float64)
    offset = epsilon * 100.0
    num_chunks = (num_faces + _CHUNK - 1) // _CHUNK
    for chunk in prange(num_chunks):
        stack = np.empty(_STACK_SIZE, dtype=np.int64)
        for i in range(chunk * _CHUNK, min(num_faces, (chunk + 1) * _CHUNK)):
            nx, ny, nz = face_normals[i, 0], face_normals[i, 1], face_normals[i, 2]
            ox = face_centroids[i, 0] + nx * offset
            oy = face_centroids[i, 1] + ny * offset
            oz = face_centroids[i, 2] + nz * offset
            visible = 0
            projected = 0.0
            for s in range(num_samples):
                sx, sy, sz = sun_samples[s, 0], sun_samples[s, 1], sun_samples[s, 2]
                if sz <= 0.0:
                    continue
                cos_theta = nx * sx + ny * sy + nz * sz
                if cos_theta <= 0.0:
                    continue
                if not _occluded(ox, oy, oz, sx, sy, sz, bvh_nodes, tri_verts,
                                 ordered_tri_indices, epsilon, stack):
                    visible += 1
                    projected += cos_theta
            illumination[i] = visible * inv_samples
            projected_factor[i] = projected * inv_samples
    return illumination, projected_factor


# ===================================================================
# BVH CONSTRUCTION — binned SAH, iterative (Numba JIT)
# ===================================================================


@njit(cache=True)
def _build_bvh_kernel(
    bmin: np.ndarray,
    bmax: np.ndarray,
    centroids: np.ndarray,
    max_leaf: int,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build the flat node array. Returns (nodes, order, node_count, depth)."""
    n = centroids.shape[0]
    order = np.arange(n).astype(np.int64)
    nodes = np.zeros(max(2 * n, 1) * _NODE_SIZE, dtype=np.float64)

    # Work stack of (node index, start, end, depth).
    work = np.empty((_MAX_DEPTH * 2 + 4, 4), dtype=np.int64)
    work[0, 0] = 0
    work[0, 1] = 0
    work[0, 2] = n
    work[0, 3] = 0
    wp = 1
    node_count = 1
    max_depth = 0

    bin_count = np.zeros(num_bins, dtype=np.int64)
    bin_min = np.empty((num_bins, 3))
    bin_max = np.empty((num_bins, 3))
    right_area = np.empty(num_bins)
    right_count = np.empty(num_bins, dtype=np.int64)

    while wp > 0:
        wp -= 1
        node = work[wp, 0]
        start = work[wp, 1]
        end = work[wp, 2]
        depth = work[wp, 3]
        max_depth = max(max_depth, depth)
        base = node * _NODE_SIZE
        count = end - start

        lo0 = lo1 = lo2 = _INF
        hi0 = hi1 = hi2 = -_INF
        c_lo0 = c_lo1 = c_lo2 = _INF
        c_hi0 = c_hi1 = c_hi2 = -_INF
        for k in range(start, end):
            t = order[k]
            lo0 = min(lo0, bmin[t, 0])
            lo1 = min(lo1, bmin[t, 1])
            lo2 = min(lo2, bmin[t, 2])
            hi0 = max(hi0, bmax[t, 0])
            hi1 = max(hi1, bmax[t, 1])
            hi2 = max(hi2, bmax[t, 2])
            c_lo0 = min(c_lo0, centroids[t, 0])
            c_lo1 = min(c_lo1, centroids[t, 1])
            c_lo2 = min(c_lo2, centroids[t, 2])
            c_hi0 = max(c_hi0, centroids[t, 0])
            c_hi1 = max(c_hi1, centroids[t, 1])
            c_hi2 = max(c_hi2, centroids[t, 2])
        nodes[base + _BBOX_MIN_X] = lo0
        nodes[base + _BBOX_MIN_Y] = lo1
        nodes[base + _BBOX_MIN_Z] = lo2
        nodes[base + _BBOX_MAX_X] = hi0
        nodes[base + _BBOX_MAX_Y] = hi1
        nodes[base + _BBOX_MAX_Z] = hi2

        make_leaf = count <= max_leaf or depth >= _MAX_DEPTH - 1
        split_axis = -1
        split_pos = 0.0
        if not make_leaf:
            ex0 = c_hi0 - c_lo0
            ex1 = c_hi1 - c_lo1
            ex2 = c_hi2 - c_lo2
            if depth < _SAH_DEPTH_LIMIT:
                d0, d1, d2 = hi0 - lo0, hi1 - lo1, hi2 - lo2
                parent_area = 2.0 * (d0 * d1 + d1 * d2 + d2 * d0)
                best_cost = float(count)
                if parent_area > 1e-30:
                    for axis in range(3):
                        c_lo = c_lo0 if axis == 0 else (c_lo1 if axis == 1 else c_lo2)
                        extent = ex0 if axis == 0 else (ex1 if axis == 1 else ex2)
                        if extent < 1e-10:
                            continue
                        scale = num_bins / extent
                        bin_count[:] = 0
                        bin_min[:, :] = _INF
                        bin_max[:, :] = -_INF
                        for k in range(start, end):
                            t = order[k]
                            b = int((centroids[t, axis] - c_lo) * scale)
                            if b >= num_bins:
                                b = num_bins - 1
                            bin_count[b] += 1
                            for a in range(3):
                                bin_min[b, a] = min(bin_min[b, a], bmin[t, a])
                                bin_max[b, a] = max(bin_max[b, a], bmax[t, a])
                        # Sweep right-to-left, then left-to-right.
                        r0 = r1 = r2 = _INF
                        s0 = s1 = s2 = -_INF
                        acc = 0
                        for b in range(num_bins - 1, 0, -1):
                            if bin_count[b] > 0:
                                r0 = min(r0, bin_min[b, 0])
                                r1 = min(r1, bin_min[b, 1])
                                r2 = min(r2, bin_min[b, 2])
                                s0 = max(s0, bin_max[b, 0])
                                s1 = max(s1, bin_max[b, 1])
                                s2 = max(s2, bin_max[b, 2])
                            acc += bin_count[b]
                            right_count[b] = acc
                            if acc > 0:
                                q0, q1, q2 = s0 - r0, s1 - r1, s2 - r2
                                right_area[b] = 2.0 * (q0 * q1 + q1 * q2 + q2 * q0)
                            else:
                                right_area[b] = 0.0
                        l0 = l1 = l2 = _INF
                        m0 = m1 = m2 = -_INF
                        left = 0
                        for b in range(0, num_bins - 1):
                            if bin_count[b] > 0:
                                l0 = min(l0, bin_min[b, 0])
                                l1 = min(l1, bin_min[b, 1])
                                l2 = min(l2, bin_min[b, 2])
                                m0 = max(m0, bin_max[b, 0])
                                m1 = max(m1, bin_max[b, 1])
                                m2 = max(m2, bin_max[b, 2])
                            left += bin_count[b]
                            right = right_count[b + 1]
                            if left == 0 or right == 0:
                                continue
                            q0, q1, q2 = m0 - l0, m1 - l1, m2 - l2
                            left_area = 2.0 * (q0 * q1 + q1 * q2 + q2 * q0)
                            cost = 1.0 + (
                                left_area * left + right_area[b + 1] * right
                            ) / parent_area
                            if cost < best_cost:
                                best_cost = cost
                                split_axis = axis
                                split_pos = c_lo + (b + 1) / scale
                if split_axis < 0 and count > 4 * max_leaf:
                    # SAH found nothing cheaper than a leaf, but the leaf
                    # would be large: split at the median instead.
                    split_axis = -2
            else:
                split_axis = -2
            if split_axis == -1:
                make_leaf = True

        if make_leaf:
            nodes[base + _CHILD_OR_START] = float(start)
            nodes[base + _COUNT_OR_RIGHT] = float(-count)
            continue

        mid = start
        if split_axis >= 0:
            i = start
            j = end - 1
            while i <= j:
                if centroids[order[i], split_axis] < split_pos:
                    i += 1
                else:
                    tmp = order[i]
                    order[i] = order[j]
                    order[j] = tmp
                    j -= 1
            mid = i
        if mid == start or mid == end:
            # Median split along the widest centroid extent.
            ex0 = c_hi0 - c_lo0
            ex1 = c_hi1 - c_lo1
            ex2 = c_hi2 - c_lo2
            axis = 0 if (ex0 >= ex1 and ex0 >= ex2) else (1 if ex1 >= ex2 else 2)
            keys = np.empty(count)
            for k in range(count):
                keys[k] = centroids[order[start + k], axis]
            perm = np.argsort(keys, kind="mergesort")
            segment = order[start:end].copy()
            for k in range(count):
                order[start + k] = segment[perm[k]]
            mid = start + count // 2

        left_node = node_count
        right_node = node_count + 1
        node_count += 2
        nodes[base + _CHILD_OR_START] = float(left_node)
        nodes[base + _COUNT_OR_RIGHT] = float(right_node)
        # Push right first so the left subtree is built first (better locality).
        work[wp, 0] = right_node
        work[wp, 1] = mid
        work[wp, 2] = end
        work[wp, 3] = depth + 1
        work[wp + 1, 0] = left_node
        work[wp + 1, 1] = start
        work[wp + 1, 2] = mid
        work[wp + 1, 3] = depth + 1
        wp += 2

    return nodes[: node_count * _NODE_SIZE].copy(), order, node_count, max_depth


def build_bvh(
    mesh: TriangleMesh,
    max_leaf_triangles: int = 4,
    sah_num_bins: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a flattened BVH over the mesh with binned SAH.

    Returns
    -------
    bvh_nodes : np.ndarray
        Flat node array, 8 float64 per node (see module docstring).
    tri_verts : np.ndarray
        Triangle vertices indexed by original triangle id. Shape (N, 3, 3).
    ordered_indices : np.ndarray
        Triangle ids in leaf order; leaves reference ranges of this array.

    """
    tri_verts = np.ascontiguousarray(mesh.vertices[mesh.triangles], dtype=np.float64)
    bmin = tri_verts.min(axis=1)
    bmax = tri_verts.max(axis=1)
    centroids = tri_verts.mean(axis=1)
    nodes, order, node_count, depth = _build_bvh_kernel(
        bmin, bmax, centroids, int(max_leaf_triangles), int(sah_num_bins)
    )
    logger.info(
        "BVH: %d triangles, %d nodes, depth %d, %.1f MB",
        tri_verts.shape[0], node_count, depth, (nodes.nbytes + tri_verts.nbytes) / 1e6,
    )
    return nodes, tri_verts, order
