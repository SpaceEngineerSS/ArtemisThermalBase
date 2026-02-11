"""Illumination engine — orchestrates solar disk + raytracer for penumbra.

This module is the "conductor" that connects the solar disk model,
ephemeris, and BVH raytracer to produce per-face illumination fractions
(0.0–1.0) for the terrain mesh. It supports both point-source (fast)
and extended-source (accurate penumbra) modes.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

Pipeline
--------
1. Receive sun direction vector (from ephemeris or direct input).
2. Generate N solar disk sample directions around the sun center.
3. Dispatch to raytracer:
   - Point source: ``compute_shadow_map_point_source()``
   - Extended source: ``compute_shadow_map_extended_source()``
4. Return illumination fraction array [0, 1] per mesh face.

Notes
-----
The extended-source mode fires N shadow rays per face, where N is the
number of solar disk samples. For 39K faces × 64 samples = ~2.5M ray
queries per timestep. The BVH acceleration keeps this tractable (~1-2
minutes on modern CPUs with Numba parallel).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from core_engine.mesh import TriangleMesh
from core_engine.raytracer import (
    build_bvh,
    compute_shadow_map_extended_source,
    compute_shadow_map_point_source,
)
from core_engine.solar_disk import generate_solar_disk_samples

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Container
# ---------------------------------------------------------------------------


@dataclass
class IlluminationResult:
    """Result of an illumination computation.

    Attributes
    ----------
    illumination : np.ndarray
        Per-face illumination fraction [0.0, 1.0]. Shape: (num_faces,).
    sun_dir : np.ndarray
        Sun center direction vector (unit). Shape: (3,).
    sun_elevation_deg : float
        Sun elevation above the local horizon in degrees.
    num_samples : int
        Number of solar disk samples used (1 = point source).
    mode : str
        'point_source' or 'extended_source'.
    stats : dict[str, float]
        Summary statistics: mean illumination, shadow fraction, etc.
    """

    illumination: np.ndarray
    sun_dir: np.ndarray
    sun_elevation_deg: float
    num_samples: int
    mode: str
    stats: dict[str, float]


# ---------------------------------------------------------------------------
# Illumination Engine
# ---------------------------------------------------------------------------


class IlluminationEngine:
    """Orchestrates solar illumination computation for terrain meshes.

    Combines the solar disk model with the BVH raytracer to produce
    per-face illumination fractions that capture penumbra effects.

    Parameters
    ----------
    mesh : TriangleMesh
        Terrain mesh (from DEM).
    bvh_data : tuple, optional
        Pre-built BVH (bvh_nodes, tri_verts, ordered_indices). If None,
        built automatically on first use.
    solar_angular_radius_rad : float
        Angular radius of the solar disk in radians.
    num_samples : int
        Number of solar disk samples for extended-source mode.
    point_source_mode : bool
        If True, ignore solar disk and use point-source shadow.
    epsilon : float
        Raytracer intersection epsilon.
    max_leaf_triangles : int
        BVH leaf capacity.
    """

    def __init__(
        self,
        mesh: TriangleMesh,
        bvh_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        solar_angular_radius_rad: float = np.radians(0.533 / 2.0),
        num_samples: int = 64,
        point_source_mode: bool = False,
        epsilon: float = 1e-10,
        max_leaf_triangles: int = 4,
    ) -> None:
        self._mesh = mesh
        self._solar_angular_radius_rad = solar_angular_radius_rad
        self._num_samples = num_samples
        self._point_source_mode = point_source_mode
        self._epsilon = epsilon

        # Build or cache BVH
        if bvh_data is not None:
            self._bvh_nodes, self._tri_verts, self._ordered_indices = bvh_data
        else:
            logger.info("Building BVH for illumination engine...")
            self._bvh_nodes, self._tri_verts, self._ordered_indices = build_bvh(
                mesh, max_leaf_triangles=max_leaf_triangles
            )

        logger.info(
            "IlluminationEngine initialized: %d faces, mode=%s, "
            "N_samples=%d, angular_radius=%.4f°",
            mesh.face_centroids.shape[0],
            "point_source" if point_source_mode else "extended_source",
            1 if point_source_mode else num_samples,
            np.degrees(solar_angular_radius_rad),
        )

    def compute(
        self,
        sun_dir: np.ndarray,
        point_source_override: bool | None = None,
    ) -> IlluminationResult:
        """Compute illumination for a given sun direction.

        Parameters
        ----------
        sun_dir : np.ndarray
            Unit vector pointing toward the sun center. Shape: (3,).
        point_source_override : bool, optional
            If provided, overrides the engine's default mode for this
            single call. Useful for quick previews.

        Returns
        -------
        IlluminationResult
            Per-face illumination fractions and metadata.
        """
        # Normalize sun direction
        sun_dir = np.asarray(sun_dir, dtype=np.float64)
        sun_dir = sun_dir / np.linalg.norm(sun_dir)

        # Determine mode
        use_point_source = (
            point_source_override
            if point_source_override is not None
            else self._point_source_mode
        )

        # Compute sun elevation (angle above local horizon)
        # For the mesh, the "up" direction is approximately z
        sun_elevation_deg = float(np.degrees(np.arcsin(
            np.clip(sun_dir[2], -1.0, 1.0)
        )))

        # Sun below horizon → everything is in shadow
        if sun_elevation_deg <= 0.0:
            num_faces = self._mesh.face_centroids.shape[0]
            logger.info(
                "Sun below horizon (elevation=%.2f°). All faces shadowed.",
                sun_elevation_deg,
            )
            return IlluminationResult(
                illumination=np.zeros(num_faces, dtype=np.float64),
                sun_dir=sun_dir,
                sun_elevation_deg=sun_elevation_deg,
                num_samples=0,
                mode="sun_below_horizon",
                stats={
                    "mean_illumination": 0.0,
                    "shadow_fraction": 1.0,
                    "penumbra_fraction": 0.0,
                    "full_light_fraction": 0.0,
                },
            )

        if use_point_source:
            illumination = self._compute_point_source(sun_dir)
            mode_str = "point_source"
            n_samples = 1
        else:
            illumination = self._compute_extended_source(sun_dir)
            mode_str = "extended_source"
            n_samples = self._num_samples

        # Compute statistics
        stats = self._compute_stats(illumination)

        logger.info(
            "Illumination computed: mode=%s, elevation=%.2f°, "
            "mean=%.3f, shadow=%.1f%%, penumbra=%.1f%%",
            mode_str,
            sun_elevation_deg,
            stats["mean_illumination"],
            stats["shadow_fraction"] * 100.0,
            stats["penumbra_fraction"] * 100.0,
        )

        return IlluminationResult(
            illumination=illumination,
            sun_dir=sun_dir,
            sun_elevation_deg=sun_elevation_deg,
            num_samples=n_samples,
            mode=mode_str,
            stats=stats,
        )

    def _compute_point_source(self, sun_dir: np.ndarray) -> np.ndarray:
        """Point-source shadow computation (binary 0/1)."""
        return compute_shadow_map_point_source(
            self._mesh.face_centroids,
            self._mesh.face_normals,
            sun_dir,
            self._bvh_nodes,
            self._tri_verts,
            self._ordered_indices,
            self._epsilon,
        )

    def _compute_extended_source(self, sun_dir: np.ndarray) -> np.ndarray:
        """Extended-source illumination with solar disk sampling."""
        # Generate solar disk sample directions
        sun_samples = generate_solar_disk_samples(
            sun_dir,
            self._solar_angular_radius_rad,
            self._num_samples,
        )

        return compute_shadow_map_extended_source(
            self._mesh.face_centroids,
            self._mesh.face_normals,
            sun_samples,
            self._bvh_nodes,
            self._tri_verts,
            self._ordered_indices,
            self._epsilon,
        )

    @staticmethod
    def _compute_stats(illumination: np.ndarray) -> dict[str, float]:
        """Compute summary statistics for an illumination map."""
        n = len(illumination)
        if n == 0:
            return {
                "mean_illumination": 0.0,
                "shadow_fraction": 1.0,
                "penumbra_fraction": 0.0,
                "full_light_fraction": 0.0,
            }

        full_shadow = float(np.sum(illumination == 0.0)) / n
        full_light = float(np.sum(illumination == 1.0)) / n
        penumbra = 1.0 - full_shadow - full_light

        return {
            "mean_illumination": float(illumination.mean()),
            "shadow_fraction": full_shadow,
            "penumbra_fraction": max(0.0, penumbra),
            "full_light_fraction": full_light,
        }
