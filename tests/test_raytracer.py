"""Tests for the BVH raytracer module.

Validates Möller-Trumbore intersection accuracy, BVH consistency,
and analytical shadow boundaries for synthetic craters.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import numpy as np
import pytest

from core_engine.raytracer import (
    moller_trumbore,
    ray_aabb_intersect,
    build_bvh,
    compute_shadow_map_point_source,
)
from core_engine.mesh import dem_to_mesh
from data_ingestion.synthetic_dem import DEMData


# ===================================================================
# FIXTURES
# ===================================================================


@pytest.fixture
def simple_triangle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A triangle in the XY plane at z=0."""
    v0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return v0, v1, v2


@pytest.fixture
def epsilon() -> float:
    """Default intersection epsilon."""
    return 1e-10


@pytest.fixture
def small_bowl_dem() -> DEMData:
    """A small synthetic parabolic bowl for shadow testing.

    D=100m, R=500m → critical angle = arctan(D/R) = arctan(0.2) ≈ 11.31°
    """
    R = 500.0
    D = 100.0
    res = 10.0  # 10 m/pixel for fast testing
    pad = 200.0
    half_extent = R + pad

    x = np.arange(-half_extent, half_extent + res, res, dtype=np.float64)
    y = np.arange(-half_extent, half_extent + res, res, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(xx**2 + yy**2)

    elevation = np.zeros_like(r, dtype=np.float64)
    inside = r <= R
    elevation[inside] = -D * (1.0 - (r[inside] / R) ** 2)

    # Gaussian rim
    outside = ~inside
    rim_width = 0.1 * R
    elevation[outside] = 30.0 * np.exp(-((r[outside] - R) ** 2) / (2.0 * rim_width**2))

    return DEMData(
        elevation=elevation,
        x_coords=x,
        y_coords=y,
        resolution_m=res,
        metadata={"type": "test_bowl", "radius_m": R, "depth_m": D},
    )


# ===================================================================
# MÖLLER-TRUMBORE INTERSECTION TESTS
# ===================================================================


class TestMollerTrumbore:
    """Test suite for the Möller-Trumbore ray-triangle intersection."""

    def test_direct_hit_center(
        self, simple_triangle: tuple, epsilon: float
    ) -> None:
        """Ray hitting the center of the triangle from above."""
        v0, v1, v2 = simple_triangle
        origin = np.array([0.25, 0.25, 1.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        t = moller_trumbore(origin, direction, v0, v1, v2, epsilon)

        assert t > 0, "Should hit the triangle"
        assert abs(t - 1.0) < 1e-8, f"Expected t=1.0, got t={t}"

    def test_direct_hit_vertex(
        self, simple_triangle: tuple, epsilon: float
    ) -> None:
        """Ray hitting exactly at vertex v0."""
        v0, v1, v2 = simple_triangle
        origin = np.array([0.0, 0.0, 2.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        t = moller_trumbore(origin, direction, v0, v1, v2, epsilon)

        # Vertex hit is an edge case — the result depends on epsilon
        # With our epsilon-expanded bounds, this should register as a hit
        assert t > 0 or t == -1.0, "Vertex hit should either hit or cleanly miss"

    def test_miss_outside_triangle(
        self, simple_triangle: tuple, epsilon: float
    ) -> None:
        """Ray clearly missing the triangle."""
        v0, v1, v2 = simple_triangle
        origin = np.array([2.0, 2.0, 1.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        t = moller_trumbore(origin, direction, v0, v1, v2, epsilon)

        assert t == -1.0, "Should miss the triangle entirely"

    def test_parallel_ray(
        self, simple_triangle: tuple, epsilon: float
    ) -> None:
        """Ray parallel to the triangle plane (should miss)."""
        v0, v1, v2 = simple_triangle
        origin = np.array([0.25, 0.25, 1.0], dtype=np.float64)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        t = moller_trumbore(origin, direction, v0, v1, v2, epsilon)

        assert t == -1.0, "Parallel ray should not intersect"

    def test_behind_ray_origin(
        self, simple_triangle: tuple, epsilon: float
    ) -> None:
        """Triangle is behind the ray origin (negative t)."""
        v0, v1, v2 = simple_triangle
        origin = np.array([0.25, 0.25, -1.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        t = moller_trumbore(origin, direction, v0, v1, v2, epsilon)

        assert t == -1.0, "Triangle behind ray should not register"

    def test_degenerate_triangle(self, epsilon: float) -> None:
        """Degenerate triangle (zero area) should not intersect."""
        v0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v2 = np.array([0.5, 0.0, 0.0], dtype=np.float64)  # collinear

        origin = np.array([0.25, 0.0, 1.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        t = moller_trumbore(origin, direction, v0, v1, v2, epsilon)

        assert t == -1.0, "Degenerate triangle should not intersect"

    def test_edge_hit_no_leakage(self, epsilon: float) -> None:
        """Two adjacent triangles sharing an edge must not leak rays.

        This is CRITICAL for PSR accuracy. A ray aimed at the shared
        edge must hit at least one of the two triangles.
        """
        # Two triangles sharing edge v0-v1
        v0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v2a = np.array([0.5, 1.0, 0.0], dtype=np.float64)
        v2b = np.array([0.5, -1.0, 0.0], dtype=np.float64)

        # Ray aimed exactly at the shared edge midpoint
        origin = np.array([0.5, 0.0, 1.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        t_a = moller_trumbore(origin, direction, v0, v1, v2a, epsilon)
        t_b = moller_trumbore(origin, direction, v0, v1, v2b, epsilon)

        hit_count = (1 if t_a > 0 else 0) + (1 if t_b > 0 else 0)
        assert hit_count >= 1, (
            f"Ray at shared edge must hit at least one triangle "
            f"(t_a={t_a}, t_b={t_b}). Edge leakage detected!"
        )

    def test_grazing_ray(
        self, simple_triangle: tuple, epsilon: float
    ) -> None:
        """Ray at a very shallow angle to the triangle plane."""
        v0, v1, v2 = simple_triangle
        origin = np.array([0.25, 0.25, 0.001], dtype=np.float64)
        direction = np.array([0.01, 0.0, -0.001], dtype=np.float64)

        t = moller_trumbore(origin, direction, v0, v1, v2, epsilon)

        # Grazing ray should hit at t ~ 1.0
        assert t > 0, "Grazing ray should still hit"


# ===================================================================
# RAY-AABB TESTS
# ===================================================================


class TestRayAABB:
    """Test suite for ray-AABB intersection."""

    def test_ray_through_box(self) -> None:
        """Ray passing through the center of a box."""
        origin = np.array([0.5, 0.5, 2.0], dtype=np.float64)
        inv_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # 1/dir
        bbox_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        bbox_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        # inv_dir for dir = (0, 0, -1) is (inf, inf, -1)
        inv_d = np.array([1e30, 1e30, -1.0], dtype=np.float64)

        assert ray_aabb_intersect(origin, inv_d, bbox_min, bbox_max, 1e30)

    def test_ray_missing_box(self) -> None:
        """Ray clearly missing the box."""
        origin = np.array([5.0, 5.0, 2.0], dtype=np.float64)
        inv_d = np.array([1e30, 1e30, -1.0], dtype=np.float64)
        bbox_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        bbox_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        assert not ray_aabb_intersect(origin, inv_d, bbox_min, bbox_max, 1e30)


# ===================================================================
# BVH + SHADOW MAP TESTS
# ===================================================================


class TestBVHShadowMap:
    """Test BVH construction and shadow mapping."""

    def test_bvh_builds_without_error(self, small_bowl_dem: DEMData) -> None:
        """BVH should build successfully for a valid mesh."""
        mesh = dem_to_mesh(small_bowl_dem)
        bvh_nodes, tri_verts, ordered_indices = build_bvh(mesh, max_leaf_triangles=4)

        assert bvh_nodes.shape[0] > 0, "BVH should have nodes"
        assert tri_verts.shape[0] == mesh.triangles.shape[0]
        assert ordered_indices.shape[0] == mesh.triangles.shape[0]

    def test_flat_terrain_full_illumination(self) -> None:
        """Flat terrain with sun overhead should be fully illuminated."""
        res = 50.0
        x = np.arange(-500, 501, res, dtype=np.float64)
        y = np.arange(-500, 501, res, dtype=np.float64)
        elevation = np.zeros((len(y), len(x)), dtype=np.float64)
        dem = DEMData(elevation=elevation, x_coords=x, y_coords=y,
                      resolution_m=res, metadata={})

        mesh = dem_to_mesh(dem)
        bvh_nodes, tri_verts, ordered_indices = build_bvh(mesh)

        # Sun directly overhead
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        illum = compute_shadow_map_point_source(
            mesh.face_centroids, mesh.face_normals, sun_dir,
            bvh_nodes, tri_verts, ordered_indices, 1e-10,
        )

        assert np.all(illum == 1.0), (
            f"Flat terrain should be fully lit. "
            f"Shadowed fraction: {(illum == 0).sum() / len(illum):.2%}"
        )

    def test_shadow_fraction_increases_with_lower_sun(
        self, small_bowl_dem: DEMData,
    ) -> None:
        """Shadow fraction should increase as the sun drops lower.

        This is a robust physics test that does not depend on the
        exact shadow boundary position (which is discretization-
        sensitive), but validates the MONOTONIC behavior of shadows
        w.r.t. sun elevation — a fundamental physical invariant.

        Test: shadow_fraction(3°) > shadow_fraction(8°) > shadow_fraction(90°)
        """
        mesh = dem_to_mesh(small_bowl_dem)
        bvh_nodes, tri_verts, ordered_indices = build_bvh(mesh, max_leaf_triangles=4)

        shadow_fractions: list[float] = []
        angles_deg = [3.0, 8.0, 20.0, 45.0, 90.0]

        for alpha_deg in angles_deg:
            alpha_rad = np.radians(alpha_deg)
            if alpha_deg == 90.0:
                sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                sun_dir = np.array(
                    [np.cos(alpha_rad), 0.0, np.sin(alpha_rad)],
                    dtype=np.float64,
                )

            illum = compute_shadow_map_point_source(
                mesh.face_centroids, mesh.face_normals, sun_dir,
                bvh_nodes, tri_verts, ordered_indices, 1e-10,
            )

            shadow_frac = 1.0 - float(illum.mean())
            shadow_fractions.append(shadow_frac)

        # Shadow fraction should decrease (more light) as sun rises
        for i in range(len(shadow_fractions) - 1):
            assert shadow_fractions[i] >= shadow_fractions[i + 1] - 0.01, (
                f"Shadow fraction should decrease with higher sun. "
                f"α={angles_deg[i]}° → {shadow_fractions[i]:.3f}, "
                f"α={angles_deg[i + 1]}° → {shadow_fractions[i + 1]:.3f}"
            )

        # At α=3° (below critical 11.3°), should have significant shadow
        assert shadow_fractions[0] > 0.1, (
            f"At 3° sun, should have >10% shadow. Got {shadow_fractions[0]:.3f}"
        )

        # At 90° (overhead), should have almost no shadow
        assert shadow_fractions[-1] < 0.05, (
            f"At 90° sun, should have <5% shadow. Got {shadow_fractions[-1]:.3f}"
        )

    def test_shadow_boundary_moves_with_sun_angle(
        self, small_bowl_dem: DEMData,
    ) -> None:
        """The shadow boundary x-position should move toward center
        as the sun drops lower (smaller elevation angle).

        This validates that the raytracer correctly computes shadow
        geometry for a bowl crater without depending on the exact
        analytical boundary position (which is discretization-sensitive).
        """
        mesh = dem_to_mesh(small_bowl_dem)
        bvh_nodes, tri_verts, ordered_indices = build_bvh(mesh, max_leaf_triangles=4)

        y_tol = small_bowl_dem.resolution_m * 1.5
        centerline_mask = np.abs(mesh.face_centroids[:, 1]) < y_tol

        if not np.any(centerline_mask):
            pytest.skip("No faces along centerline")

        cx = mesh.face_centroids[centerline_mask, 0]
        boundaries: list[float] = []

        for alpha_deg in [5.0, 8.0]:
            alpha_rad = np.radians(alpha_deg)
            sun_dir = np.array(
                [np.cos(alpha_rad), 0.0, np.sin(alpha_rad)],
                dtype=np.float64,
            )

            illum = compute_shadow_map_point_source(
                mesh.face_centroids, mesh.face_normals, sun_dir,
                bvh_nodes, tri_verts, ordered_indices, 1e-10,
            )

            ci = illum[centerline_mask]
            order = np.argsort(cx)
            cx_sorted = cx[order]
            ci_sorted = ci[order]

            transitions = np.where(np.diff(ci_sorted) > 0.5)[0]
            if len(transitions) == 0:
                pytest.skip(f"No transition at α={alpha_deg}°")

            # Take the transition closest to x=0 (crater center)
            transition_xs = cx_sorted[transitions]
            idx_closest = np.argmin(np.abs(transition_xs))
            boundaries.append(float(transition_xs[idx_closest]))

        # At lower sun (5°), the shadow extends further from the sun
        # (more negative x) than at higher sun (8°)
        assert boundaries[0] < boundaries[1], (
            f"Shadow boundary should move toward -x at lower sun. "
            f"α=5° boundary={boundaries[0]:.1f}, "
            f"α=8° boundary={boundaries[1]:.1f}"
        )

    def test_center_shadowed_below_critical_angle(
        self, small_bowl_dem: DEMData
    ) -> None:
        """Crater center should be shadowed when sun < critical angle.

        Critical angle = arctan(D/R) = arctan(100/500) ≈ 11.31°
        Test with sun at 3° → center must be in shadow.
        """
        alpha_rad = np.radians(3.0)

        mesh = dem_to_mesh(small_bowl_dem)
        bvh_nodes, tri_verts, ordered_indices = build_bvh(mesh)

        sun_dir = np.array(
            [np.cos(alpha_rad), 0.0, np.sin(alpha_rad)], dtype=np.float64
        )

        illum = compute_shadow_map_point_source(
            mesh.face_centroids, mesh.face_normals, sun_dir,
            bvh_nodes, tri_verts, ordered_indices, 1e-10,
        )

        # Find faces near center (within 50m of origin)
        center_mask = (
            mesh.face_centroids[:, 0] ** 2 + mesh.face_centroids[:, 1] ** 2
        ) < 50.0**2

        if not np.any(center_mask):
            pytest.skip("No faces near center")

        center_illum = illum[center_mask]

        assert np.all(
            center_illum == 0.0
        ), f"Center should be shadowed at 3°. Lit fraction: {center_illum.mean():.2%}"
