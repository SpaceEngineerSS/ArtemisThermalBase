"""DEM to triangle mesh converter.

Converts a 2D elevation grid into a triangle mesh with computed face normals,
face areas, and face centroids. This mesh is consumed by the raytracer for
shadow/illumination computation.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

Notes
-----
The DEM grid of M rows × N columns produces 2*(M-1)*(N-1) triangles.
Each grid cell is split into two triangles along the diagonal:

    (i,j)-------(i,j+1)
      |  \\  T1  |
      |   \\     |
      | T0  \\   |
      |       \\  |
    (i+1,j)---(i+1,j+1)

T0: (i,j), (i+1,j), (i+1,j+1)   — lower-left triangle
T1: (i,j), (i+1,j+1), (i,j+1)   — upper-right triangle
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from data_ingestion.synthetic_dem import DEMData

logger = logging.getLogger(__name__)


@dataclass
class TriangleMesh:
    """Triangle mesh generated from a DEM.

    Attributes
    ----------
    vertices : np.ndarray
        Vertex positions [m]. Shape: (num_vertices, 3), dtype: float64.
        Columns are (x, y, z).
    triangles : np.ndarray
        Triangle vertex indices. Shape: (num_triangles, 3), dtype: int64.
        Each row contains three indices into the vertices array.
    face_normals : np.ndarray
        Unit outward normal vectors for each triangle face.
        Shape: (num_triangles, 3), dtype: float64.
    face_areas : np.ndarray
        Area of each triangle face [m²]. Shape: (num_triangles,), dtype: float64.
    face_centroids : np.ndarray
        Centroid (center of mass) of each triangle face [m].
        Shape: (num_triangles, 3), dtype: float64.
    metadata : dict
        Mesh statistics and provenance information.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    face_centroids: np.ndarray
    metadata: dict


def dem_to_mesh(dem: DEMData) -> TriangleMesh:
    """Convert a DEM elevation grid to a triangle mesh.

    Parameters
    ----------
    dem : DEMData
        Digital Elevation Model with x/y coordinate arrays and 2D elevation grid.

    Returns
    -------
    TriangleMesh
        Triangle mesh with vertices, indices, normals, areas, and centroids.

    Notes
    -----
    Memory estimate for a 1000×1000 DEM:
    - Vertices: 1M × 3 × 8 bytes = 24 MB
    - Triangles: 2M × 3 × 8 bytes = 48 MB
    - Normals: 2M × 3 × 8 bytes = 48 MB
    - Total: ~144 MB
    """
    elev = dem.elevation  # (ny, nx)
    ny, nx = elev.shape

    logger.info("Converting DEM (%d × %d) to triangle mesh...", nx, ny)

    # --- Build vertex array ---
    # Create 3D coordinates for every DEM pixel
    xx, yy = np.meshgrid(dem.x_coords, dem.y_coords, indexing="xy")
    vertices = np.column_stack(
        [xx.ravel(), yy.ravel(), elev.ravel()]
    ).astype(np.float64)
    num_vertices = vertices.shape[0]

    logger.debug("  Vertices: %d", num_vertices)

    # --- Build triangle index array ---
    # For each grid cell (i, j), create two triangles
    num_cells_y = ny - 1
    num_cells_x = nx - 1
    num_triangles = 2 * num_cells_y * num_cells_x

    # Vertex index for pixel (row, col) = row * nx + col
    row_idx, col_idx = np.meshgrid(
        np.arange(num_cells_y, dtype=np.int64),
        np.arange(num_cells_x, dtype=np.int64),
        indexing="ij",
    )
    row_flat = row_idx.ravel()
    col_flat = col_idx.ravel()

    # Four corners of each cell
    v00 = row_flat * nx + col_flat           # (i, j)
    v10 = (row_flat + 1) * nx + col_flat     # (i+1, j)
    v11 = (row_flat + 1) * nx + (col_flat + 1)  # (i+1, j+1)
    v01 = row_flat * nx + (col_flat + 1)     # (i, j+1)

    # Lower-left triangle: v00, v10, v11
    tri_lower = np.column_stack([v00, v10, v11])
    # Upper-right triangle: v00, v11, v01
    tri_upper = np.column_stack([v00, v11, v01])

    # Interleave: [lower0, upper0, lower1, upper1, ...]
    triangles = np.empty((num_triangles, 3), dtype=np.int64)
    triangles[0::2] = tri_lower
    triangles[1::2] = tri_upper

    logger.debug("  Triangles: %d", num_triangles)

    # --- Compute face normals, areas, and centroids ---
    face_normals, face_areas, face_centroids = _compute_face_properties(
        vertices, triangles
    )

    # Check for degenerate triangles (zero area)
    degenerate_count = np.sum(face_areas < 1e-20)
    if degenerate_count > 0:
        logger.warning(
            "  %d degenerate triangles detected (area < 1e-20 m²)", degenerate_count
        )

    metadata = {
        "source_dem_nx": nx,
        "source_dem_ny": ny,
        "num_vertices": num_vertices,
        "num_triangles": num_triangles,
        "resolution_m": dem.resolution_m,
        "degenerate_triangles": int(degenerate_count),
        "total_surface_area_km2": float(face_areas.sum()) / 1e6,
        "z_range_m": float(vertices[:, 2].max() - vertices[:, 2].min()),
        "memory_estimate_MB": (
            vertices.nbytes + triangles.nbytes + face_normals.nbytes
            + face_areas.nbytes + face_centroids.nbytes
        ) / 1e6,
    }

    logger.info(
        "Mesh created: %d vertices, %d triangles, %.1f km² surface area, "
        "~%.0f MB memory",
        num_vertices,
        num_triangles,
        metadata["total_surface_area_km2"],
        metadata["memory_estimate_MB"],
    )

    return TriangleMesh(
        vertices=vertices,
        triangles=triangles,
        face_normals=face_normals,
        face_areas=face_areas,
        face_centroids=face_centroids,
        metadata=metadata,
    )


def _compute_face_properties(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute face normals, areas, and centroids for all triangles.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions, shape (num_vertices, 3).
    triangles : np.ndarray
        Triangle vertex indices, shape (num_triangles, 3).

    Returns
    -------
    normals : np.ndarray
        Unit outward normals, shape (num_triangles, 3).
    areas : np.ndarray
        Triangle areas [m²], shape (num_triangles,).
    centroids : np.ndarray
        Triangle centroids [m], shape (num_triangles, 3).
    """
    v0 = vertices[triangles[:, 0]]  # (N, 3)
    v1 = vertices[triangles[:, 1]]  # (N, 3)
    v2 = vertices[triangles[:, 2]]  # (N, 3)

    # Edge vectors
    e1 = v1 - v0  # (N, 3)
    e2 = v2 - v0  # (N, 3)

    # Cross product gives normal direction with magnitude = 2 * area
    cross = np.cross(e1, e2)  # (N, 3)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)  # (N, 1)

    # Avoid division by zero for degenerate triangles
    safe_norms = np.where(norms > 1e-30, norms, 1.0)
    normals = cross / safe_norms

    # For degenerate triangles, set normal to (0, 0, 1)
    degenerate_mask = norms.ravel() < 1e-30
    normals[degenerate_mask] = np.array([0.0, 0.0, 1.0])

    # Ensure normals point "outward" (z-component should be positive for terrain)
    # For a terrain DEM, the upward normal should have positive z
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] *= -1.0

    # Areas = 0.5 * |cross product|
    areas = 0.5 * norms.ravel()

    # Centroids = average of three vertices
    centroids = (v0 + v1 + v2) / 3.0

    return normals, areas, centroids
