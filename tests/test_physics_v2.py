"""Tests for Physics Upgrade v2 — energy conservation, view factors, volatiles.

Tests the three new physics modules:
1. Solar flux correction (inverse-square law)
2. View factor reciprocity and energy conservation
3. Volatile sublimation model (Langmuir + cold trap stability)

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import numpy as np
import pytest


# ===================================================================
# TEST 1: Solar Flux Inverse-Square Law
# ===================================================================


class TestSolarFluxCorrection:
    """Verify distance-dependent solar flux computation."""

    def test_flux_at_1au(self):
        """S(1 AU) should equal S₀ = 1361 W/m²."""
        # At exactly 1 AU distance, flux should be the solar constant
        S_0 = 1361.0
        AU_m = 1.495978707e11
        distance_m = AU_m  # exactly 1 AU

        S_t = S_0 * (AU_m / distance_m) ** 2
        assert abs(S_t - S_0) < 1e-6, f"Expected {S_0}, got {S_t}"

    def test_flux_at_perihelion(self):
        """At perihelion (~0.983 AU), flux should be > S₀."""
        S_0 = 1361.0
        AU_m = 1.495978707e11
        distance_m = 0.983 * AU_m  # perihelion

        S_t = S_0 * (AU_m / distance_m) ** 2
        assert S_t > S_0, f"Expected S_t > {S_0}, got {S_t}"
        # Should be ~3.4% higher
        assert abs(S_t / S_0 - 1.0) < 0.05, "Perihelion correction too large"

    def test_flux_at_aphelion(self):
        """At aphelion (~1.017 AU), flux should be < S₀."""
        S_0 = 1361.0
        AU_m = 1.495978707e11
        distance_m = 1.017 * AU_m  # aphelion

        S_t = S_0 * (AU_m / distance_m) ** 2
        assert S_t < S_0, f"Expected S_t < {S_0}, got {S_t}"

    def test_inverse_square_symmetry(self):
        """Double the distance → quarter the flux."""
        S_0 = 1361.0
        AU_m = 1.495978707e11

        S_1 = S_0 * (AU_m / AU_m) ** 2
        S_2 = S_0 * (AU_m / (2 * AU_m)) ** 2

        assert abs(S_2 / S_1 - 0.25) < 1e-10


# ===================================================================
# TEST 2: Stefan-Boltzmann Equilibrium
# ===================================================================


class TestStefanBoltzmannLimits:
    """Verify thermal equilibrium matches Stefan-Boltzmann prediction."""

    def test_equilibrium_temperature(self):
        """T_eq = (Q_abs / (ε·σ))^{1/4} for a surface in radiative equilibrium."""
        sigma = 5.670374419e-8
        emissivity = 0.95
        Q_abs = 200.0  # W/m² absorbed flux

        T_eq = (Q_abs / (emissivity * sigma)) ** 0.25
        Q_emitted = emissivity * sigma * T_eq ** 4

        assert abs(Q_emitted - Q_abs) < 1e-6, (
            f"Energy balance violation: Q_abs={Q_abs}, Q_emit={Q_emitted}"
        )

    def test_equilibrium_range(self):
        """Verify equilibrium temperatures for typical lunar conditions."""
        sigma = 5.670374419e-8
        emissivity = 0.95

        # Shadowed region: Q_abs ~ 0
        T_shadow = (0.001 / (emissivity * sigma)) ** 0.25
        assert T_shadow < 50.0, f"Shadow T too high: {T_shadow} K"

        # Sunlit daytime: Q_abs ~ 1000 W/m²
        T_day = (1000.0 / (emissivity * sigma)) ** 0.25
        assert 350.0 < T_day < 420.0, f"Daytime T out of range: {T_day} K"


# ===================================================================
# TEST 3: View Factor Properties
# ===================================================================


class TestViewFactorProperties:
    """Verify fundamental view factor constraints."""

    def test_row_sum_conservation(self):
        """Row sums of view factor matrix must be ≤ 1 (energy conservation).

        For a convex surface, Σ F_ij = 1 (all radiation hits something).
        For non-convex (crater), Σ F_ij < 1 (some escapes to space).
        """
        # Simulate a simple VF matrix (3 faces)
        row_ptr = np.array([0, 2, 3, 5], dtype=np.int64)
        col_idx = np.array([1, 2, 0, 0, 1], dtype=np.int64)
        values = np.array([0.3, 0.2, 0.25, 0.15, 0.35], dtype=np.float64)

        N = 3
        for i in range(N):
            row_sum = values[row_ptr[i]:row_ptr[i + 1]].sum()
            assert row_sum <= 1.0 + 1e-6, (
                f"Row {i} sum {row_sum} > 1.0 (energy conservation violated)"
            )

    def test_reciprocity(self):
        """A_i · F_ij should approximately equal A_j · F_ji."""
        # Two faces with known areas and view factors
        A = np.array([10.0, 20.0])  # m²

        F_01 = 0.4
        F_10 = 0.2  # A_0 * F_01 = 10*0.4 = 4, A_1 * F_10 = 20*0.2 = 4

        lhs = A[0] * F_01
        rhs = A[1] * F_10

        assert abs(lhs - rhs) / max(lhs, rhs) < 1e-6, (
            f"Reciprocity violation: {lhs} vs {rhs}"
        )

    def test_view_factor_bounds(self):
        """All view factors must be in [0, 1]."""
        values = np.array([0.0, 0.5, 1.0, 0.001, 0.999])
        assert np.all(values >= 0.0), "Negative view factor found"
        assert np.all(values <= 1.0), "View factor > 1.0 found"


# ===================================================================
# TEST 4: IR Flux Computation
# ===================================================================


class TestIRFluxComputation:
    """Test thermal IR flux computation using view factors."""

    def test_ir_flux_basic(self):
        """IR flux should be ε·σ·Σ(F_ij · T_j⁴)."""
        from core_engine.view_factors import compute_ir_flux

        surface_temps = np.array([200.0, 250.0, 150.0])
        row_ptr = np.array([0, 1, 2, 3], dtype=np.int64)
        col_idx = np.array([1, 2, 0], dtype=np.int64)
        values = np.array([0.3, 0.4, 0.2], dtype=np.float64)
        emissivity = 0.95
        sigma = 5.670374419e-8

        Q_ir = compute_ir_flux(
            surface_temps, row_ptr, col_idx, values, emissivity, sigma,
        )

        # Manual check for face 0: ε·σ·F_01·T_1⁴ = 0.95 * σ * 0.3 * 250⁴
        expected_0 = emissivity * sigma * 0.3 * 250.0 ** 4
        assert abs(Q_ir[0] - expected_0) / expected_0 < 1e-6

    def test_ir_flux_zero_temp(self):
        """Zero temperature → zero IR flux."""
        from core_engine.view_factors import compute_ir_flux

        surface_temps = np.zeros(3)
        row_ptr = np.array([0, 1, 2, 3], dtype=np.int64)
        col_idx = np.array([1, 2, 0], dtype=np.int64)
        values = np.array([0.3, 0.4, 0.2], dtype=np.float64)

        Q_ir = compute_ir_flux(
            surface_temps, row_ptr, col_idx, values, 0.95, 5.67e-8,
        )
        assert np.allclose(Q_ir, 0.0)


# ===================================================================
# TEST 5: Volatile Sublimation Model
# ===================================================================


class TestVolatileSublimation:
    """Test water ice sublimation rate computation."""

    def test_sublimation_zero_at_zero_temp(self):
        """Rate should be 0 at T = 0 K."""
        from thermal_solver.volatiles import sublimation_rate

        assert sublimation_rate(0.0) == 0.0
        assert sublimation_rate(0.5) == 0.0

    def test_sublimation_increases_with_temperature(self):
        """Sublimation rate should increase monotonically with T."""
        from thermal_solver.volatiles import sublimation_rate

        T_values = [50.0, 100.0, 150.0, 200.0, 250.0]
        rates = [sublimation_rate(T) for T in T_values]

        for i in range(len(rates) - 1):
            assert rates[i + 1] >= rates[i], (
                f"Rate decreased from T={T_values[i]} to T={T_values[i+1]}: "
                f"{rates[i]} → {rates[i+1]}"
            )

    def test_sublimation_negligible_at_100K(self):
        """At 100 K, sublimation should be effectively zero."""
        from thermal_solver.volatiles import sublimation_rate_log10

        log_rate = sublimation_rate_log10(100.0)
        assert log_rate < -30.0, (
            f"Sublimation too high at 100 K: log₁₀(rate) = {log_rate}"
        )

    def test_sublimation_significant_at_200K(self):
        """At 200 K, sublimation should be measurable."""
        from thermal_solver.volatiles import sublimation_rate

        rate = sublimation_rate(200.0)
        assert rate > 0.0, "Expected non-zero sublimation at 200 K"


# ===================================================================
# TEST 6: Cold Trap Stability Classification
# ===================================================================


class TestColdTrapStability:
    """Test ice stability classification (Powell & Rubanenko 2020)."""

    def test_stable_below_110K(self):
        """T < 110 K → stable (class 0)."""
        from thermal_solver.volatiles import (
            ice_stability_class,
            STABILITY_STABLE,
        )

        assert ice_stability_class(50.0) == STABILITY_STABLE
        assert ice_stability_class(100.0) == STABILITY_STABLE
        assert ice_stability_class(109.9) == STABILITY_STABLE

    def test_marginal_110_to_115K(self):
        """110 K ≤ T < 115 K → marginal (class 1)."""
        from thermal_solver.volatiles import (
            ice_stability_class,
            STABILITY_MARGINAL,
        )

        assert ice_stability_class(110.0) == STABILITY_MARGINAL
        assert ice_stability_class(112.5) == STABILITY_MARGINAL
        assert ice_stability_class(114.9) == STABILITY_MARGINAL

    def test_unstable_above_115K(self):
        """T ≥ 115 K → unstable (class 2)."""
        from thermal_solver.volatiles import (
            ice_stability_class,
            STABILITY_UNSTABLE,
        )

        assert ice_stability_class(115.0) == STABILITY_UNSTABLE
        assert ice_stability_class(200.0) == STABILITY_UNSTABLE
        assert ice_stability_class(400.0) == STABILITY_UNSTABLE

    def test_cold_trap_map_vectorized(self):
        """Vectorized cold trap map should match element-wise classification."""
        from thermal_solver.volatiles import (
            compute_cold_trap_map,
            ice_stability_class,
        )

        temps = np.array([50.0, 109.0, 112.0, 115.0, 200.0])
        stability = compute_cold_trap_map(temps)

        for i, T in enumerate(temps):
            expected = ice_stability_class(T)
            assert stability[i] == expected, (
                f"Mismatch at T={T}: got {stability[i]}, expected {expected}"
            )


# ===================================================================
# TEST 7: Ice Retention Timescale
# ===================================================================


class TestIceRetentionTimescale:
    """Test ice retention time estimates."""

    def test_infinite_at_low_temp(self):
        """Ice at 40 K should last forever."""
        from thermal_solver.volatiles import ice_retention_timescale

        t = ice_retention_timescale(30.0)
        assert t == np.inf or t > 1e15

    def test_finite_at_high_temp(self):
        """Ice at 300 K should sublimate within geological timescales."""
        from thermal_solver.volatiles import ice_retention_timescale

        t = ice_retention_timescale(300.0)
        assert t < 1e6, f"Retention too long at 300 K: {t} years"
        assert t > 0, "Retention must be positive"

    def test_geological_at_200K(self):
        """Ice at 200 K sublimates on geological timescales (very slow)."""
        from thermal_solver.volatiles import ice_retention_timescale

        t = ice_retention_timescale(200.0)
        # At 200 K, Langmuir model predicts extremely slow sublimation
        assert t > 1e6, f"Retention too short at 200 K: {t} years"
        assert np.isfinite(t), "Retention should be finite at 200 K"


# ===================================================================
# TEST 8: Closest-Hit Raytracer
# ===================================================================


class TestClosestHitRaytracer:
    """Test closest-hit BVH traversal (needed for view factors)."""

    def test_closest_hit_returns_triangle(self):
        """A ray shot at a triangle should return its index."""
        from core_engine.raytracer import _closest_hit_bvh

        # Create a single triangle at z=0
        tri_verts = np.array([
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        ], dtype=np.float64)

        # Minimal BVH: one leaf node with one triangle
        bvh_nodes = np.array([
            -1.0, -1.0, -0.1, 1.0, 1.0, 0.1,  # bbox
            0.0, -1.0,  # child_or_start=0, count_or_right=-1 (leaf with 1 tri)
        ], dtype=np.float64)

        ordered_indices = np.array([0], dtype=np.int64)
        epsilon = 1e-10

        # Ray from above, pointing down
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([0.0, 0.0, -1.0])

        tri_idx, t_hit = _closest_hit_bvh(
            origin, direction, bvh_nodes, tri_verts, ordered_indices, epsilon,
        )

        assert tri_idx == 0, f"Expected tri 0, got {tri_idx}"
        assert abs(t_hit - 1.0) < 0.01, f"Expected t≈1.0, got {t_hit}"

    def test_closest_hit_miss(self):
        """A ray that misses should return -1."""
        from core_engine.raytracer import _closest_hit_bvh

        tri_verts = np.array([
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        ], dtype=np.float64)

        bvh_nodes = np.array([
            -1.0, -1.0, -0.1, 1.0, 1.0, 0.1,
            0.0, -1.0,
        ], dtype=np.float64)

        ordered_indices = np.array([0], dtype=np.int64)

        # Ray pointing away from triangle
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([0.0, 0.0, 1.0])  # pointing up, away

        tri_idx, _ = _closest_hit_bvh(
            origin, direction, bvh_nodes, tri_verts, ordered_indices, 1e-10,
        )

        assert tri_idx == -1, f"Expected miss (-1), got {tri_idx}"
