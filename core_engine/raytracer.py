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
from typing import Any

import numpy as np
import numba as nb
from numba import njit, prange, float64, int64, boolean

from core_engine.mesh import TriangleMesh

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
) -> boolean:
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
# BVH TRAVERSAL — Stack-based, No Recursion (Numba JIT)
# ===================================================================


@njit(cache=True, fastmath=False)
def _shadow_ray_bvh(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    bvh_nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered_tri_indices: np.ndarray,
    epsilon: float,
) -> boolean:
    """Test if a shadow ray is occluded by ANY triangle in the BVH.

    Uses stack-based iterative traversal. Returns True on the FIRST hit
    (early exit) since we only care about occlusion, not closest hit.

    Parameters
    ----------
    ray_origin : np.ndarray
        Shadow ray origin. Shape: (3,).
    ray_dir : np.ndarray
        Shadow ray direction (toward sun). Shape: (3,).
    bvh_nodes : np.ndarray
        Flattened BVH node array. Shape: (num_nodes * 8,).
    tri_verts : np.ndarray
        Triangle vertices. Shape: (num_triangles, 3, 3).
    ordered_tri_indices : np.ndarray
        Triangle indices ordered by BVH leaf assignment. Shape: (num_triangles,).
    epsilon : float
        Intersection epsilon.

    Returns
    -------
    bool
        True if the ray is occluded (hits any triangle).
    """
    # Precompute inverse direction (handles zeros via IEEE 754 inf)
    inv_dir = np.empty(3, dtype=np.float64)
    for axis in range(3):
        if ray_dir[axis] == 0.0:
            inv_dir[axis] = _INF
        else:
            inv_dir[axis] = 1.0 / ray_dir[axis]

    # Stack for iterative traversal (max depth ~64 for any realistic BVH)
    stack = np.empty(64, dtype=np.int64)
    stack_ptr = 0
    stack[stack_ptr] = 0  # Push root node index
    stack_ptr += 1

    bbox_min_tmp = np.empty(3, dtype=np.float64)
    bbox_max_tmp = np.empty(3, dtype=np.float64)

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]
        base = node_idx * _NODE_SIZE

        # Extract AABB
        bbox_min_tmp[0] = bvh_nodes[base + _BBOX_MIN_X]
        bbox_min_tmp[1] = bvh_nodes[base + _BBOX_MIN_Y]
        bbox_min_tmp[2] = bvh_nodes[base + _BBOX_MIN_Z]
        bbox_max_tmp[0] = bvh_nodes[base + _BBOX_MAX_X]
        bbox_max_tmp[1] = bvh_nodes[base + _BBOX_MAX_Y]
        bbox_max_tmp[2] = bvh_nodes[base + _BBOX_MAX_Z]

        # Test ray-AABB intersection
        if not ray_aabb_intersect(ray_origin, inv_dir, bbox_min_tmp, bbox_max_tmp, _INF):
            continue

        count_or_right = bvh_nodes[base + _COUNT_OR_RIGHT]

        if count_or_right < 0:
            # LEAF NODE: test triangles
            start = int(bvh_nodes[base + _CHILD_OR_START])
            count = int(-count_or_right)
            for i in range(start, start + count):
                tri_idx = ordered_tri_indices[i]
                v0 = tri_verts[tri_idx, 0]
                v1 = tri_verts[tri_idx, 1]
                v2 = tri_verts[tri_idx, 2]
                t_hit = moller_trumbore(ray_origin, ray_dir, v0, v1, v2, epsilon)
                if t_hit > epsilon:
                    return True  # Early exit — shadow confirmed
        else:
            # INTERNAL NODE: push children
            left = int(bvh_nodes[base + _CHILD_OR_START])
            right = int(count_or_right)
            stack[stack_ptr] = left
            stack_ptr += 1
            stack[stack_ptr] = right
            stack_ptr += 1

    return False  # No occlusion


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
    """Compute a binary shadow map for a point-source sun.

    For each mesh face, casts a shadow ray from its centroid toward the
    sun and tests for occlusion using the BVH.

    Parameters
    ----------
    face_centroids : np.ndarray
        Centroid positions. Shape: (num_faces, 3).
    face_normals : np.ndarray
        Unit outward normals. Shape: (num_faces, 3).
    sun_dir : np.ndarray
        Unit vector pointing toward the sun. Shape: (3,).
    bvh_nodes : np.ndarray
        Flattened BVH node array.
    tri_verts : np.ndarray
        Triangle vertices. Shape: (num_triangles, 3, 3).
    ordered_tri_indices : np.ndarray
        Ordered triangle indices for BVH leaves.
    epsilon : float
        Intersection epsilon.

    Returns
    -------
    illumination : np.ndarray
        Illumination fraction per face: 1.0 = lit, 0.0 = shadowed.
        Shape: (num_faces,).
    """
    num_faces = face_centroids.shape[0]
    illumination = np.zeros(num_faces, dtype=np.float64)

    for i in prange(num_faces):
        # Check if face is oriented toward the sun
        cos_theta = (
            face_normals[i, 0] * sun_dir[0]
            + face_normals[i, 1] * sun_dir[1]
            + face_normals[i, 2] * sun_dir[2]
        )

        if cos_theta <= 0.0:
            # Face points away from sun — self-shadowed
            illumination[i] = 0.0
            continue

        # Offset ray origin slightly along normal to avoid self-intersection
        origin = np.empty(3, dtype=np.float64)
        origin[0] = face_centroids[i, 0] + face_normals[i, 0] * epsilon * 100.0
        origin[1] = face_centroids[i, 1] + face_normals[i, 1] * epsilon * 100.0
        origin[2] = face_centroids[i, 2] + face_normals[i, 2] * epsilon * 100.0

        # Cast shadow ray toward sun
        occluded = _shadow_ray_bvh(
            origin, sun_dir, bvh_nodes, tri_verts, ordered_tri_indices, epsilon
        )

        illumination[i] = 0.0 if occluded else 1.0

    return illumination


@njit(cache=True, parallel=True, fastmath=False)
def compute_shadow_map_extended_source(
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
    sun_samples: np.ndarray,
    bvh_nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered_tri_indices: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Compute illumination fraction for an extended (disk) sun.

    For each mesh face, casts multiple shadow rays toward sample points
    on the solar disk. The illumination fraction is the ratio of
    unoccluded rays to total rays (penumbra support).

    Parameters
    ----------
    face_centroids : np.ndarray
        Centroid positions. Shape: (num_faces, 3).
    face_normals : np.ndarray
        Unit outward normals. Shape: (num_faces, 3).
    sun_samples : np.ndarray
        Unit direction vectors toward solar disk sample points.
        Shape: (num_samples, 3).
    bvh_nodes : np.ndarray
        Flattened BVH node array.
    tri_verts : np.ndarray
        Triangle vertices. Shape: (num_triangles, 3, 3).
    ordered_tri_indices : np.ndarray
        Ordered triangle indices for BVH leaves.
    epsilon : float
        Intersection epsilon.

    Returns
    -------
    illumination : np.ndarray
        Illumination fraction per face, in [0, 1]. Shape: (num_faces,).
    """
    num_faces = face_centroids.shape[0]
    num_samples = sun_samples.shape[0]
    inv_samples = 1.0 / float(num_samples)

    illumination = np.zeros(num_faces, dtype=np.float64)

    for i in prange(num_faces):
        visible_count = 0

        for s in range(num_samples):
            # Check if face is oriented toward this sun sample
            cos_theta = (
                face_normals[i, 0] * sun_samples[s, 0]
                + face_normals[i, 1] * sun_samples[s, 1]
                + face_normals[i, 2] * sun_samples[s, 2]
            )

            if cos_theta <= 0.0:
                continue

            # Offset ray origin
            origin = np.empty(3, dtype=np.float64)
            origin[0] = face_centroids[i, 0] + face_normals[i, 0] * epsilon * 100.0
            origin[1] = face_centroids[i, 1] + face_normals[i, 1] * epsilon * 100.0
            origin[2] = face_centroids[i, 2] + face_normals[i, 2] * epsilon * 100.0

            sun_dir = sun_samples[s]
            occluded = _shadow_ray_bvh(
                origin, sun_dir, bvh_nodes, tri_verts, ordered_tri_indices, epsilon
            )

            if not occluded:
                visible_count += 1

        illumination[i] = float(visible_count) * inv_samples

    return illumination


# ===================================================================
# BVH CONSTRUCTION — Python (one-time cost, not JIT-compiled)
# ===================================================================


def build_bvh(
    mesh: TriangleMesh,
    max_leaf_triangles: int = 4,
    sah_num_bins: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a flattened BVH from a triangle mesh using binned SAH.

    Parameters
    ----------
    mesh : TriangleMesh
        Triangle mesh with vertices, triangle indices, and centroids.
    max_leaf_triangles : int
        Maximum number of triangles per leaf node. Default: 4.
    sah_num_bins : int
        Number of bins for SAH cost evaluation. Default: 16.

    Returns
    -------
    bvh_nodes : np.ndarray
        Flattened node array. Shape: (num_nodes * 8,), dtype: float64.
    tri_verts : np.ndarray
        Triangle vertex positions reordered for BVH locality.
        Shape: (num_triangles, 3, 3), dtype: float64.
    ordered_indices : np.ndarray
        Triangle indices in BVH leaf order. Shape: (num_triangles,), dtype: int64.
    """
    num_triangles = mesh.triangles.shape[0]
    logger.info(
        "Building BVH for %d triangles (max_leaf=%d, sah_bins=%d)...",
        num_triangles,
        max_leaf_triangles,
        sah_num_bins,
    )

    # Pre-extract all triangle vertex positions: (N, 3, 3)
    tri_verts = np.empty((num_triangles, 3, 3), dtype=np.float64)
    for k in range(3):
        tri_verts[:, k, :] = mesh.vertices[mesh.triangles[:, k]]

    # Compute per-triangle AABBs and centroids
    tri_bboxes_min = np.minimum(
        np.minimum(tri_verts[:, 0, :], tri_verts[:, 1, :]), tri_verts[:, 2, :]
    )
    tri_bboxes_max = np.maximum(
        np.maximum(tri_verts[:, 0, :], tri_verts[:, 1, :]), tri_verts[:, 2, :]
    )
    tri_centroids = (tri_verts[:, 0, :] + tri_verts[:, 1, :] + tri_verts[:, 2, :]) / 3.0

    # Working index array (reordered during construction)
    indices = np.arange(num_triangles, dtype=np.int64)

    # Pre-allocate node storage (worst case: 2*N - 1 nodes for N triangles)
    max_nodes = 2 * num_triangles
    nodes_flat = np.zeros(max_nodes * _NODE_SIZE, dtype=np.float64)

    node_count = [0]  # Mutable counter (list for closure access)

    def _allocate_node() -> int:
        idx = node_count[0]
        node_count[0] += 1
        return idx

    def _build_recursive(start: int, end: int) -> int:
        """Recursively build BVH. Returns node index."""
        node_idx = _allocate_node()
        base = node_idx * _NODE_SIZE
        count = end - start

        # Compute AABB for this subset
        bbox_min = tri_bboxes_min[indices[start:end]].min(axis=0)
        bbox_max = tri_bboxes_max[indices[start:end]].max(axis=0)

        nodes_flat[base + _BBOX_MIN_X] = bbox_min[0]
        nodes_flat[base + _BBOX_MIN_Y] = bbox_min[1]
        nodes_flat[base + _BBOX_MIN_Z] = bbox_min[2]
        nodes_flat[base + _BBOX_MAX_X] = bbox_max[0]
        nodes_flat[base + _BBOX_MAX_Y] = bbox_max[1]
        nodes_flat[base + _BBOX_MAX_Z] = bbox_max[2]

        if count <= max_leaf_triangles:
            # Leaf node
            nodes_flat[base + _CHILD_OR_START] = float(start)
            nodes_flat[base + _COUNT_OR_RIGHT] = float(-count)
            return node_idx

        # Find best split using binned SAH
        best_axis, best_split = _find_best_split_sah(
            indices, start, end, tri_centroids, tri_bboxes_min, tri_bboxes_max,
            bbox_min, bbox_max, sah_num_bins,
        )

        if best_axis < 0:
            # No beneficial split found — make leaf
            nodes_flat[base + _CHILD_OR_START] = float(start)
            nodes_flat[base + _COUNT_OR_RIGHT] = float(-count)
            return node_idx

        # Partition indices around split plane
        mid = _partition_indices(
            indices, start, end, best_axis, best_split, tri_centroids
        )

        # Fallback: if partition didn't split, use median
        if mid == start or mid == end:
            mid = (start + end) // 2
            # Sort by centroid along best_axis
            sub_indices = indices[start:end].copy()
            order = np.argsort(tri_centroids[sub_indices, best_axis])
            indices[start:end] = sub_indices[order]
            mid = (start + end) // 2

        # Recurse
        left_idx = _build_recursive(start, mid)
        right_idx = _build_recursive(mid, end)

        nodes_flat[base + _CHILD_OR_START] = float(left_idx)
        nodes_flat[base + _COUNT_OR_RIGHT] = float(right_idx)

        return node_idx

    _build_recursive(0, num_triangles)

    actual_nodes = node_count[0]
    bvh_nodes = nodes_flat[: actual_nodes * _NODE_SIZE].copy()

    logger.info(
        "BVH built: %d nodes (%d internal, %d est. leaves), "
        "%.1f MB node memory",
        actual_nodes,
        actual_nodes,  # includes both
        actual_nodes - (actual_nodes // 2),
        bvh_nodes.nbytes / 1e6,
    )

    return bvh_nodes, tri_verts, indices.copy()


def _find_best_split_sah(
    indices: np.ndarray,
    start: int,
    end: int,
    centroids: np.ndarray,
    bboxes_min: np.ndarray,
    bboxes_max: np.ndarray,
    parent_bbox_min: np.ndarray,
    parent_bbox_max: np.ndarray,
    num_bins: int,
) -> tuple[int, float]:
    """Find the best split axis and position using binned SAH.

    Parameters
    ----------
    indices : np.ndarray
        Working index array.
    start, end : int
        Range of indices to consider.
    centroids : np.ndarray
        Triangle centroids. Shape: (N, 3).
    bboxes_min, bboxes_max : np.ndarray
        Per-triangle AABBs. Shape: (N, 3).
    parent_bbox_min, parent_bbox_max : np.ndarray
        Parent node AABB. Shape: (3,).
    num_bins : int
        Number of bins for SAH sweep.

    Returns
    -------
    best_axis : int
        Best split axis (0, 1, 2) or -1 if no beneficial split.
    best_split : float
        Best split position along the axis.
    """
    count = end - start
    sub_idx = indices[start:end]

    # SAH costs: C_trav + (SA_L/SA_P)*N_L*C_isect + (SA_R/SA_P)*N_R*C_isect
    # We minimize: SA_L*N_L + SA_R*N_R (ignoring constants)
    c_trav = 1.0
    c_isect = 1.0
    parent_sa = _surface_area(parent_bbox_min, parent_bbox_max)
    if parent_sa < 1e-30:
        return -1, 0.0

    leaf_cost = count * c_isect
    best_cost = leaf_cost
    best_axis = -1
    best_split = 0.0

    for axis in range(3):
        axis_min = parent_bbox_min[axis]
        axis_max = parent_bbox_max[axis]
        axis_range = axis_max - axis_min

        if axis_range < 1e-10:
            continue

        # Bin centroids
        bin_width = axis_range / num_bins
        bin_counts = np.zeros(num_bins, dtype=np.int64)
        bin_bbox_min = np.full((num_bins, 3), _INF, dtype=np.float64)
        bin_bbox_max = np.full((num_bins, 3), -_INF, dtype=np.float64)

        for idx in sub_idx:
            b = int((centroids[idx, axis] - axis_min) / bin_width)
            b = min(b, num_bins - 1)
            bin_counts[b] += 1
            for d in range(3):
                if bboxes_min[idx, d] < bin_bbox_min[b, d]:
                    bin_bbox_min[b, d] = bboxes_min[idx, d]
                if bboxes_max[idx, d] > bin_bbox_max[b, d]:
                    bin_bbox_max[b, d] = bboxes_max[idx, d]

        # Sweep from left to right to evaluate split costs
        for split_bin in range(1, num_bins):
            left_count = int(bin_counts[:split_bin].sum())
            right_count = count - left_count

            if left_count == 0 or right_count == 0:
                continue

            # Compute left and right AABBs
            left_bins_valid = bin_counts[:split_bin] > 0
            right_bins_valid = bin_counts[split_bin:] > 0

            if not left_bins_valid.any() or not right_bins_valid.any():
                continue

            left_min = bin_bbox_min[:split_bin][left_bins_valid].min(axis=0)
            left_max = bin_bbox_max[:split_bin][left_bins_valid].max(axis=0)
            right_min = bin_bbox_min[split_bin:][right_bins_valid].min(axis=0)
            right_max = bin_bbox_max[split_bin:][right_bins_valid].max(axis=0)

            sa_left = _surface_area(left_min, left_max)
            sa_right = _surface_area(right_min, right_max)

            cost = c_trav + (sa_left * left_count + sa_right * right_count) * c_isect / parent_sa

            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                best_split = axis_min + split_bin * bin_width

    return best_axis, best_split


def _surface_area(bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    """Compute the surface area of an AABB.

    Parameters
    ----------
    bbox_min, bbox_max : np.ndarray
        AABB corners. Shape: (3,).

    Returns
    -------
    float
        Surface area.
    """
    d = bbox_max - bbox_min
    return 2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])


def _partition_indices(
    indices: np.ndarray,
    start: int,
    end: int,
    axis: int,
    split: float,
    centroids: np.ndarray,
) -> int:
    """Partition indices in-place around a split plane.

    Indices with centroid[axis] < split go to the left partition.

    Parameters
    ----------
    indices : np.ndarray
        Working index array (modified in-place).
    start, end : int
        Range to partition.
    axis : int
        Split axis.
    split : float
        Split position.
    centroids : np.ndarray
        Triangle centroids.

    Returns
    -------
    int
        Partition point (first index of right partition).
    """
    left = start
    right = end - 1

    while left <= right:
        if centroids[indices[left], axis] < split:
            left += 1
        else:
            indices[left], indices[right] = indices[right], indices[left]
            right -= 1

    return left


# ===================================================================
# HIGH-LEVEL API
# ===================================================================


def compute_illumination(
    mesh: TriangleMesh,
    sun_dir: np.ndarray,
    bvh_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    sun_samples: np.ndarray | None = None,
    epsilon: float = _DEFAULT_EPSILON,
    max_leaf_triangles: int = _DEFAULT_MAX_LEAF,
) -> np.ndarray:
    """Compute the illumination fraction for every mesh face.

    Parameters
    ----------
    mesh : TriangleMesh
        The terrain mesh.
    sun_dir : np.ndarray
        Unit vector pointing toward the sun center. Shape: (3,).
    bvh_data : tuple, optional
        Pre-built BVH (bvh_nodes, tri_verts, ordered_indices). If None,
        the BVH is built on the fly.
    sun_samples : np.ndarray, optional
        Solar disk sample directions. Shape: (N, 3). If None, point-source
        mode is used.
    epsilon : float
        Intersection epsilon.
    max_leaf_triangles : int
        BVH leaf capacity.

    Returns
    -------
    illumination : np.ndarray
        Per-face illumination fraction [0, 1]. Shape: (num_faces,).
    """
    # Build BVH if not provided
    if bvh_data is None:
        bvh_nodes, tri_verts, ordered_indices = build_bvh(
            mesh, max_leaf_triangles=max_leaf_triangles
        )
    else:
        bvh_nodes, tri_verts, ordered_indices = bvh_data

    if sun_samples is None:
        # Point-source mode
        logger.info("Computing illumination (point-source mode)...")
        return compute_shadow_map_point_source(
            mesh.face_centroids,
            mesh.face_normals,
            sun_dir,
            bvh_nodes,
            tri_verts,
            ordered_indices,
            epsilon,
        )
    else:
        # Extended-source mode
        logger.info(
            "Computing illumination (extended source, %d samples)...",
            sun_samples.shape[0],
        )
        return compute_shadow_map_extended_source(
            mesh.face_centroids,
            mesh.face_normals,
            sun_samples,
            bvh_nodes,
            tri_verts,
            ordered_indices,
            epsilon,
        )
