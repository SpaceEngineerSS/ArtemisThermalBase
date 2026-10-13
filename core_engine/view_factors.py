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
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from core_engine.raytracer import _STACK_SIZE, _closest

if TYPE_CHECKING:
    prange = range
else:
    from numba import prange

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


@njit(cache=True, inline="always")
def _local_to_world(
    dx: float, dy: float, dz: float,
    nx: float, ny: float, nz: float,
) -> tuple[float, float, float]:
    """Rotate a direction from the normal-aligned frame into the world frame."""
    if abs(nx) < 0.9:
        rx, ry, rz = 1.0, 0.0, 0.0
    else:
        rx, ry, rz = 0.0, 1.0, 0.0
    t0 = ry * nz - rz * ny
    t1 = rz * nx - rx * nz
    t2 = rx * ny - ry * nx
    t_len = np.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
    if t_len < 1e-30:
        t_len = 1.0
    t0 /= t_len
    t1 /= t_len
    t2 /= t_len
    b0 = ny * t2 - nz * t1
    b1 = nz * t0 - nx * t2
    b2 = nx * t1 - ny * t0
    return (
        dx * t0 + dy * b0 + dz * nx,
        dx * t1 + dy * b1 + dz * ny,
        dx * t2 + dy * b2 + dz * nz,
    )


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
    max_neighbors: int,
    epsilon: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    dropped : np.ndarray
        Hits per face that did not fit in ``max_neighbors``. Shape: (N,).

    """
    num_faces = face_centroids.shape[0]

    # Pre-allocate per-face hit buffers
    # Max unique faces hit per source face is bounded by num_rays
    # Use a flat array: each face gets a slot of max_neighbors entries
    max_neighbors = min(num_rays, max_neighbors)

    # Output arrays — flat, will be compacted later
    # Temporary per-thread storage using parallel-safe indexing
    all_neighbor_idx = np.full(
        (num_faces, max_neighbors), -1, dtype=np.int64
    )
    all_neighbor_val = np.zeros(
        (num_faces, max_neighbors), dtype=np.float64
    )
    row_counts = np.zeros(num_faces, dtype=np.int64)
    dropped = np.zeros(num_faces, dtype=np.int64)

    inv_num_rays = 1.0 / float(num_rays)

    for i in prange(num_faces):
        # Per-face RNG (deterministic from seed + face index)
        rng_state = np.uint64(seed * 2654435761 + i * 1103515245 + 12345)

        nx, ny, nz = face_normals[i, 0], face_normals[i, 1], face_normals[i, 2]
        ox = face_centroids[i, 0] + nx * epsilon * 100.0
        oy = face_centroids[i, 1] + ny * epsilon * 100.0
        oz = face_centroids[i, 2] + nz * epsilon * 100.0
        stack = np.empty(_STACK_SIZE, dtype=np.int64)

        # Local hit counter — accumulate hits per neighbor face
        # Use a simple hash map approximation with linear probing
        local_idx = np.full(max_neighbors, -1, dtype=np.int64)
        local_count = np.zeros(max_neighbors, dtype=np.int64)
        n_unique = 0

        for _ray_index in range(num_rays):
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

            wx, wy, wz = _local_to_world(dx, dy, dz, nx, ny, nz)
            hit_tri, _hit_t = _closest(
                ox, oy, oz, wx, wy, wz, bvh_nodes, tri_verts,
                ordered_tri_indices, epsilon, stack,
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

            if not found:
                if n_unique < max_neighbors:
                    local_idx[n_unique] = hit_face
                    local_count[n_unique] = 1
                    n_unique += 1
                else:
                    dropped[i] += 1

        # Write results for this face
        row_counts[i] = n_unique
        for k in range(n_unique):
            all_neighbor_idx[i, k] = local_idx[k]
            all_neighbor_val[i, k] = float(local_count[k]) * inv_num_rays

    return all_neighbor_idx, all_neighbor_val, row_counts, dropped


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


@njit(cache=True, parallel=True)
def compute_ir_flux_gray(
    surface_temps: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    vf_values: np.ndarray,
    emissivities: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Compute one-bounce gray-surface IR exchange with Kirchhoff absorption.

    Source face ``j`` emits ``epsilon_j * sigma * T_j**4`` and receiver
    ``i`` absorbs the fraction ``epsilon_i``. Reflected-IR radiosity is not
    included; the function deliberately models one terrain interaction.
    """
    num_faces = len(surface_temps)
    q_ir = np.zeros(num_faces, dtype=np.float64)
    for i in prange(num_faces):
        irradiation = 0.0
        for k in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[k]
            temp_j = surface_temps[j]
            irradiation += (
                vf_values[k]
                * emissivities[j]
                * sigma
                * temp_j
                * temp_j
                * temp_j
                * temp_j
            )
        q_ir[i] = emissivities[i] * irradiation
    return q_ir


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
    max_neighbors: int = 256,
    max_memory_mb: float = 2048.0,
    epsilon: float = 1e-10,
    seed: int = 42,
    reciprocity_tol: float = 0.05,
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
        Allowed area-weighted reciprocity asymmetry before warning.

    Returns
    -------
    ViewFactorMatrix
        Sparse view factor matrix in CSR format.

    """
    num_faces = face_centroids.shape[0]
    num_tris = tri_verts.shape[0]

    # Peak allocation includes the dense temporary neighbor buffers and the
    # worst-case compact CSR arrays that coexist during compaction.
    effective_neighbors = min(num_rays, max_neighbors)
    estimated_mb = (
        num_faces * effective_neighbors * 16 * 2
        + (num_faces + 1) * 8
    ) / 1e6
    logger.info(
        "View factor computation: %d faces, %d rays/face, "
        "estimated memory: %.1f MB",
        num_faces, num_rays, estimated_mb,
    )

    if estimated_mb > max_memory_mb:
        raise MemoryError(
            "View-factor preflight requires approximately "
            f"{estimated_mb:.0f} MB, above the configured "
            f"{max_memory_mb:.0f} MB limit. Reduce mesh/rays/neighbors "
            "or explicitly raise view_factors.max_memory_mb."
        )

    # Build triangle-to-face mapping
    # Each triangle in tri_verts corresponds to a face.
    # For DEM meshes, face_index == triangle_index (1:1 mapping)
    tri_to_face = np.arange(num_tris, dtype=np.int64)

    wall_start = time.perf_counter()

    logger.info("Computing view factors (Monte Carlo, %d rays/face)...", num_rays)
    neighbor_idx, neighbor_val, row_counts, dropped = _compute_view_factors_mc(
        face_centroids=face_centroids,
        face_normals=face_normals,
        face_areas=face_areas,
        bvh_nodes=bvh_nodes,
        tri_verts=tri_verts,
        ordered_tri_indices=ordered_tri_indices,
        tri_to_face=tri_to_face,
        num_rays=num_rays,
        max_neighbors=max_neighbors,
        epsilon=epsilon,
        seed=seed,
    )
    row_ptr, col_idx_csr, values_csr = _compact_csr(neighbor_idx, neighbor_val, row_counts)
    del neighbor_idx, neighbor_val
    vf = ViewFactorMatrix(
        row_ptr=row_ptr,
        col_idx=col_idx_csr,
        values=values_csr,
        num_faces=num_faces,
    )
    logger.info(
        "View factors: %d non-zeros (%.1f MB) in %.1f s",
        vf.nnz, vf.memory_mb, time.perf_counter() - wall_start,
    )

    total_dropped = int(dropped.sum())
    if total_dropped:
        lost = total_dropped / (num_faces * num_rays)
        logger.warning(
            "%d faces hit more than max_neighbors=%d distinct faces; %.2f %% of "
            "terrain-IR rays were dropped. Raise view_factors.max_neighbors if "
            "this matters for your case.",
            int(np.count_nonzero(dropped)), max_neighbors, 100.0 * lost,
        )

    row_sums = np.add.reduceat(values_csr, row_ptr[:-1]) if vf.nnz else np.zeros(1)
    row_sums[np.diff(row_ptr) == 0] = 0.0
    if row_sums.max() > 1.05:
        logger.warning("Largest view-factor row sum is %.3f (> 1.05).", row_sums.max())

    asymmetry = _reciprocity_asymmetry(
        row_ptr, col_idx_csr, values_csr, face_areas, max(1, num_faces // 2000),
    )
    if asymmetry > reciprocity_tol:
        logger.warning(
            "View-factor reciprocity: area-weighted asymmetry %.3f exceeds %.3f. "
            "More rays per face will reduce it.", asymmetry, reciprocity_tol,
        )
    else:
        logger.info("View-factor reciprocity asymmetry %.3f (tolerance %.3f).",
                    asymmetry, reciprocity_tol)
    return vf


@njit(cache=True)
def _compact_csr(
    neighbor_idx: np.ndarray,
    neighbor_val: np.ndarray,
    row_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack the padded per-face neighbour buffers into CSR arrays."""
    num_faces = row_counts.shape[0]
    row_ptr = np.zeros(num_faces + 1, dtype=np.int64)
    for i in range(num_faces):
        row_ptr[i + 1] = row_ptr[i] + row_counts[i]
    nnz = row_ptr[num_faces]
    col_idx = np.empty(nnz, dtype=np.int64)
    values = np.empty(nnz, dtype=np.float64)
    for i in range(num_faces):
        base = row_ptr[i]
        for k in range(row_counts[i]):
            col_idx[base + k] = neighbor_idx[i, k]
            values[base + k] = neighbor_val[i, k]
    return row_ptr, col_idx, values


@njit(cache=True)
def _reciprocity_asymmetry(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    values: np.ndarray,
    face_areas: np.ndarray,
    row_stride: int,
) -> float:
    """Area-weighted reciprocity error Σ|A_i F_ij − A_j F_ji| / Σ A_i F_ij.

    Individual Monte-Carlo entries are too noisy to test one by one, so we
    measure the aggregate over every ``row_stride``-th row.
    """
    num_faces = face_areas.shape[0]
    diff = 0.0
    total = 0.0
    for i in range(0, num_faces, row_stride):
        for k in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[k]
            f_ji = 0.0
            for m in range(row_ptr[j], row_ptr[j + 1]):
                if col_idx[m] == i:
                    f_ji = values[m]
                    break
            lhs = face_areas[i] * values[k]
            diff += abs(lhs - face_areas[j] * f_ji)
            total += lhs
    if total <= 0.0:
        return 0.0
    return diff / total
