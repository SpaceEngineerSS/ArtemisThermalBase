"""Monte Carlo view factor computation with sparse CSR storage.

Computes the radiative view factor (form factor) matrix F_ij between
all mesh faces using cosine-weighted hemisphere sampling and BVH
closest-hit raytracing. Results are stored in CSR sparse format for
memory efficiency and Numba-compatible IR flux computation.

Physics
-------
The view factor F_ij is the fraction of diffuse radiation leaving
face i that directly reaches face j:

    F_ij = (1/A_i) ∫∫ [cos(θ_i)·cos(θ_j)] / (π·r²) · V(i,j) dA_j dA_i

Monte Carlo estimation with cosine-weighted hemisphere sampling
(Malley's method) gives an unbiased estimator:

    F_ij ≈ (hits on face j) / N_rays

Reciprocity: A_i · F_ij = A_j · F_ji

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

References
----------
- Schröder, P. & Hanrahan, P. (1993). "On the Form Factor between Two
  Polygons." SIGGRAPH 93.
- Cohen, M.F. & Wallace, J.R. (1993). "Radiosity and Realistic Image
  Synthesis." Academic Press.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange, int64, float64

logger = logging.getLogger(__name__)


# ===================================================================
# VIEW FACTOR SPARSE MATRIX — CSR Format
# ===================================================================


@dataclass
class ViewFactorMatrix:
    """Sparse view factor matrix in CSR (Compressed Sparse Row) format.

    Attributes
    ----------
    row_ptr : np.ndarray
        Row pointer array. Shape: (num_faces + 1,).
        row_ptr[i] to row_ptr[i+1] gives the range of entries for face i.
    col_idx : np.ndarray
        Column indices (target face indices). Shape: (nnz,).
    values : np.ndarray
        View factor values F_ij. Shape: (nnz,).
    num_faces : int
        Number of mesh faces.
    """

    row_ptr: np.ndarray
    col_idx: np.ndarray
    values: np.ndarray
    num_faces: int

    @property
    def nnz(self) -> int:
        """Number of non-zero entries."""
        return len(self.values)

    @property
    def memory_mb(self) -> float:
        """Approximate memory usage in MB."""
        return (
            self.row_ptr.nbytes + self.col_idx.nbytes + self.values.nbytes
        ) / 1e6


# ===================================================================
# COSINE-WEIGHTED HEMISPHERE SAMPLING — Numba JIT
# ===================================================================


@njit(cache=True)
def _cosine_weighted_hemisphere_sample(
    u1: float, u2: float,
) -> tuple[float, float, float]:
    """Generate a cosine-weighted random direction in the local hemisphere.

    Uses Malley's method: uniform disk sampling projected onto hemisphere.
    The PDF is p(ω) = cos(θ)/π, which importance-samples the cosine term
    in the rendering equation, eliminating the need for explicit cos(θ)
    weighting in the Monte Carlo estimator.

    Parameters
    ----------
    u1, u2 : float
        Uniform random numbers in [0, 1).

    Returns
    -------
    dx, dy, dz : float
        Direction in local frame (z = surface normal). dz > 0.
    """
    phi = 2.0 * np.pi * u1
    r = np.sqrt(u2)
    dx = r * np.cos(phi)
    dy = r * np.sin(phi)
    dz = np.sqrt(max(1.0 - u2, 0.0))
    return dx, dy, dz


@njit(cache=True)
def _local_to_world(
    dx: float, dy: float, dz: float,
    normal: np.ndarray,
) -> np.ndarray:
    """Transform a direction from local (normal-aligned) to world frame.

    Constructs a tangent-space basis from the surface normal using the
    method that avoids singularities at the poles.

    Parameters
    ----------
    dx, dy, dz : float
        Direction in local frame (z = normal).
    normal : np.ndarray
        Unit surface normal. Shape: (3,).

    Returns
    -------
    np.ndarray
        Direction in world frame. Shape: (3,).
    """
    # Build orthonormal basis (tangent, bitangent, normal)
    # Choose reference axis that is least aligned with normal
    if abs(normal[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    # tangent = normalize(ref × normal)
    t0 = ref[1] * normal[2] - ref[2] * normal[1]
    t1 = ref[2] * normal[0] - ref[0] * normal[2]
    t2 = ref[0] * normal[1] - ref[1] * normal[0]
    t_len = np.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
    if t_len < 1e-30:
        t_len = 1.0
    t0 /= t_len
    t1 /= t_len
    t2 /= t_len

    # bitangent = normal × tangent
    b0 = normal[1] * t2 - normal[2] * t1
    b1 = normal[2] * t0 - normal[0] * t2
    b2 = normal[0] * t1 - normal[1] * t0

    # Transform: world_dir = dx*tangent + dy*bitangent + dz*normal
    result = np.empty(3, dtype=np.float64)
    result[0] = dx * t0 + dy * b0 + dz * normal[0]
    result[1] = dx * t1 + dy * b1 + dz * normal[1]
    result[2] = dx * t2 + dy * b2 + dz * normal[2]
    return result


# ===================================================================
# VIEW FACTOR COMPUTATION — Numba JIT (Parallel)
# ===================================================================


@njit(cache=True, parallel=True)
def _compute_view_factors_mc(
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    bvh_nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered_tri_indices: np.ndarray,
    tri_to_face: np.ndarray,
    num_rays: int,
    epsilon: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sparse view factors using Monte Carlo hemisphere sampling.

    For each face i, shoots num_rays cosine-weighted random rays into
    the hemisphere above the face. Rays that hit another face j contribute
    to F_ij = (hits_on_j) / num_rays.

    Parameters
    ----------
    face_centroids : np.ndarray
        Face centroid positions. Shape: (N, 3).
    face_normals : np.ndarray
        Unit face normals. Shape: (N, 3).
    face_areas : np.ndarray
        Face areas [m²]. Shape: (N,).
    bvh_nodes : np.ndarray
        Flattened BVH node array.
    tri_verts : np.ndarray
        Triangle vertices. Shape: (num_tris, 3, 3).
    ordered_tri_indices : np.ndarray
        Triangle indices in BVH leaf order.
    tri_to_face : np.ndarray
        Mapping from triangle index to face index. Shape: (num_tris,).
    num_rays : int
        Number of rays per face (512 for publication quality).
    epsilon : float
        Ray intersection epsilon.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    all_col_idx : np.ndarray
        Target face indices for non-zero entries. Shape: (total_nnz,).
    all_values : np.ndarray
        View factor values. Shape: (total_nnz,).
    row_counts : np.ndarray
        Number of non-zero entries per face. Shape: (N,).
    """
    from core_engine.raytracer import _closest_hit_bvh

    num_faces = face_centroids.shape[0]

    # Pre-allocate per-face hit buffers
    # Max unique faces hit per source face is bounded by num_rays
    # Use a flat array: each face gets a slot of max_neighbors entries
    max_neighbors = min(num_rays, 256)

    # Output arrays — flat, will be compacted later
    # Temporary per-thread storage using parallel-safe indexing
    all_neighbor_idx = np.full(
        (num_faces, max_neighbors), -1, dtype=np.int64
    )
    all_neighbor_val = np.zeros(
        (num_faces, max_neighbors), dtype=np.float64
    )
    row_counts = np.zeros(num_faces, dtype=np.int64)

    inv_num_rays = 1.0 / float(num_rays)

    for i in prange(num_faces):
        # Per-face RNG (deterministic from seed + face index)
        rng_state = np.uint64(seed * 2654435761 + i * 1103515245 + 12345)

        normal = face_normals[i]
        origin = np.empty(3, dtype=np.float64)
        origin[0] = face_centroids[i, 0] + normal[0] * epsilon * 100.0
        origin[1] = face_centroids[i, 1] + normal[1] * epsilon * 100.0
        origin[2] = face_centroids[i, 2] + normal[2] * epsilon * 100.0

        # Local hit counter — accumulate hits per neighbor face
        # Use a simple hash map approximation with linear probing
        local_idx = np.full(max_neighbors, -1, dtype=np.int64)
        local_count = np.zeros(max_neighbors, dtype=np.int64)
        n_unique = 0

        for r in range(num_rays):
            # LCG pseudo-random number generator
            rng_state = (rng_state * np.uint64(6364136223846793005)
                         + np.uint64(1442695040888963407))
            u1 = float(rng_state >> np.uint64(33)) / float(np.uint64(1) << np.uint64(31))

            rng_state = (rng_state * np.uint64(6364136223846793005)
                         + np.uint64(1442695040888963407))
            u2 = float(rng_state >> np.uint64(33)) / float(np.uint64(1) << np.uint64(31))

            # Clamp to (0, 1)
            u1 = min(max(u1, 1e-10), 1.0 - 1e-10)
            u2 = min(max(u2, 1e-10), 1.0 - 1e-10)

            # Cosine-weighted hemisphere sample
            dx, dy, dz = _cosine_weighted_hemisphere_sample(u1, u2)

            # Transform to world frame
            ray_dir = _local_to_world(dx, dy, dz, normal)

            # Closest-hit ray query
            hit_tri, hit_t = _closest_hit_bvh(
                origin, ray_dir, bvh_nodes, tri_verts,
                ordered_tri_indices, epsilon,
            )

            if hit_tri < 0:
                continue

            hit_face = tri_to_face[hit_tri]
            if hit_face == i:
                continue  # Self-hit (shouldn't happen with offset, but guard)

            # Linear scan to find or insert this face
            found = False
            for k in range(n_unique):
                if local_idx[k] == hit_face:
                    local_count[k] += 1
                    found = True
                    break

            if not found and n_unique < max_neighbors:
                local_idx[n_unique] = hit_face
                local_count[n_unique] = 1
                n_unique += 1

        # Write results for this face
        row_counts[i] = n_unique
        for k in range(n_unique):
            all_neighbor_idx[i, k] = local_idx[k]
            all_neighbor_val[i, k] = float(local_count[k]) * inv_num_rays

    return all_neighbor_idx, all_neighbor_val, row_counts


# ===================================================================
# IR FLUX COMPUTATION — Numba JIT
# ===================================================================


@njit(cache=True, parallel=True)
def compute_ir_flux(
    surface_temps: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    vf_values: np.ndarray,
    emissivity: float,
    sigma: float,
) -> np.ndarray:
    """Compute absorbed IR flux for each face from thermal radiation exchange.

    Q_IR,i = ε · σ · Σ_j [F_ij · T_j⁴]

    Parameters
    ----------
    surface_temps : np.ndarray
        Surface temperature per face [K]. Shape: (N,).
    row_ptr : np.ndarray
        CSR row pointer. Shape: (N+1,).
    col_idx : np.ndarray
        CSR column indices. Shape: (nnz,).
    vf_values : np.ndarray
        View factor values. Shape: (nnz,).
    emissivity : float
        Thermal emissivity.
    sigma : float
        Stefan-Boltzmann constant [W/m²/K⁴].

    Returns
    -------
    Q_ir : np.ndarray
        Absorbed IR flux per face [W/m²]. Shape: (N,).
    """
    num_faces = len(surface_temps)
    Q_ir = np.zeros(num_faces, dtype=np.float64)

    for i in prange(num_faces):
        q = 0.0
        for k in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[k]
            T_j = surface_temps[j]
            q += vf_values[k] * T_j * T_j * T_j * T_j
        Q_ir[i] = emissivity * sigma * q

    return Q_ir


# ===================================================================
# HIGH-LEVEL API
# ===================================================================


def compute_view_factor_matrix(
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    bvh_nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered_tri_indices: np.ndarray,
    mesh_triangles: np.ndarray,
    num_rays: int = 512,
    epsilon: float = 1e-10,
    seed: int = 42,
    reciprocity_tol: float = 1e-5,
) -> ViewFactorMatrix:
    """Compute the sparse view factor matrix for a triangle mesh.

    Parameters
    ----------
    face_centroids : np.ndarray
        Face centroid positions. Shape: (N, 3).
    face_normals : np.ndarray
        Unit face normals. Shape: (N, 3).
    face_areas : np.ndarray
        Face areas [m²]. Shape: (N,).
    bvh_nodes : np.ndarray
        Flattened BVH node array.
    tri_verts : np.ndarray
        Triangle vertices. Shape: (num_tris, 3, 3).
    ordered_tri_indices : np.ndarray
        BVH-ordered triangle indices.
    mesh_triangles : np.ndarray
        Original mesh triangle indices. Shape: (num_tris, 3).
        Used to build tri_to_face mapping.
    num_rays : int
        Number of Monte Carlo rays per face. Default: 512.
    epsilon : float
        Ray intersection epsilon.
    seed : int
        Random seed for reproducibility.
    reciprocity_tol : float
        Tolerance for reciprocity validation.

    Returns
    -------
    ViewFactorMatrix
        Sparse view factor matrix in CSR format.
    """
    num_faces = face_centroids.shape[0]
    num_tris = tri_verts.shape[0]

    # Memory safety check
    estimated_nnz = num_faces * 64  # Conservative estimate
    estimated_mb = estimated_nnz * 16 / 1e6
    logger.info(
        "View factor computation: %d faces, %d rays/face, "
        "estimated memory: %.1f MB",
        num_faces, num_rays, estimated_mb,
    )

    if estimated_mb > 2000:
        logger.warning(
            "Estimated VF memory %.0f MB exceeds 2 GB safety limit. "
            "Consider reducing mesh resolution.",
            estimated_mb,
        )

    # Build triangle-to-face mapping
    # Each triangle in tri_verts corresponds to a face.
    # For DEM meshes, face_index == triangle_index (1:1 mapping)
    tri_to_face = np.arange(num_tris, dtype=np.int64)

    wall_start = time.perf_counter()

    # Monte Carlo view factor computation
    logger.info("Computing view factors (Monte Carlo, %d rays/face)...", num_rays)
    neighbor_idx, neighbor_val, row_counts = _compute_view_factors_mc(
        face_centroids=face_centroids,
        face_normals=face_normals,
        face_areas=face_areas,
        bvh_nodes=bvh_nodes,
        tri_verts=tri_verts,
        ordered_tri_indices=ordered_tri_indices,
        tri_to_face=tri_to_face,
        num_rays=num_rays,
        epsilon=epsilon,
        seed=seed,
    )

    # Compact into CSR format
    total_nnz = int(row_counts.sum())
    row_ptr = np.zeros(num_faces + 1, dtype=np.int64)
    col_idx_csr = np.empty(total_nnz, dtype=np.int64)
    values_csr = np.empty(total_nnz, dtype=np.float64)

    ptr = 0
    for i in range(num_faces):
        row_ptr[i] = ptr
        n = int(row_counts[i])
        for k in range(n):
            col_idx_csr[ptr] = neighbor_idx[i, k]
            values_csr[ptr] = neighbor_val[i, k]
            ptr += 1
    row_ptr[num_faces] = ptr

    wall_elapsed = time.perf_counter() - wall_start

    vf = ViewFactorMatrix(
        row_ptr=row_ptr,
        col_idx=col_idx_csr,
        values=values_csr,
        num_faces=num_faces,
    )

    logger.info(
        "View factors computed: %d nnz entries (%.1f MB), %.1f seconds",
        vf.nnz, vf.memory_mb, wall_elapsed,
    )

    # Validate row sums (should be <= 1.0 for energy conservation)
    row_sums = np.zeros(num_faces, dtype=np.float64)
    for i in range(num_faces):
        for k in range(row_ptr[i], row_ptr[i + 1]):
            row_sums[i] += values_csr[k]

    max_row_sum = row_sums.max()
    mean_row_sum = row_sums.mean()
    logger.info(
        "Row sum stats: mean=%.4f, max=%.4f (should be <= 1.0)",
        mean_row_sum, max_row_sum,
    )

    if max_row_sum > 1.0 + 0.05:
        logger.warning(
            "Max row sum %.4f exceeds 1.05 — possible Monte Carlo bias "
            "or geometry issue.",
            max_row_sum,
        )

    # Validate reciprocity: A_i * F_ij ≈ A_j * F_ji
    reciprocity_errors = _check_reciprocity(
        row_ptr, col_idx_csr, values_csr, face_areas, reciprocity_tol,
    )
    if reciprocity_errors > 0:
        pct = 100.0 * reciprocity_errors / max(total_nnz, 1)
        logger.warning(
            "Reciprocity violations: %d / %d entries (%.1f%%) exceed "
            "tolerance %.1e",
            reciprocity_errors, total_nnz, pct, reciprocity_tol,
        )
    else:
        logger.info("Reciprocity validation passed (tol=%.1e).", reciprocity_tol)

    return vf


def _check_reciprocity(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    values: np.ndarray,
    face_areas: np.ndarray,
    tol: float,
) -> int:
    """Check A_i * F_ij ≈ A_j * F_ji for all non-zero entries.

    Parameters
    ----------
    row_ptr, col_idx, values : np.ndarray
        CSR matrix data.
    face_areas : np.ndarray
        Face areas [m²].
    tol : float
        Relative tolerance for reciprocity check.

    Returns
    -------
    int
        Number of entries violating reciprocity.
    """
    num_faces = len(face_areas)
    violations = 0

    for i in range(num_faces):
        A_i = face_areas[i]
        for k in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[k]
            F_ij = values[k]
            A_j = face_areas[j]

            # Find F_ji in row j
            F_ji = 0.0
            for m in range(row_ptr[j], row_ptr[j + 1]):
                if col_idx[m] == i:
                    F_ji = values[m]
                    break

            # Check: |A_i * F_ij - A_j * F_ji| / max(...) < tol
            lhs = A_i * F_ij
            rhs = A_j * F_ji
            denom = max(abs(lhs), abs(rhs), 1e-30)
            if abs(lhs - rhs) / denom > tol:
                violations += 1

    return violations
