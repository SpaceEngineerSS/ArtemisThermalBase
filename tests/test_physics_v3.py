"""Physics validation tests for v0.3.0 — Hapke BRDF & Bandfield Roughness.

Tests verify:
1. Hapke phase function normalization and symmetry
2. Opposition effect peak at zero phase angle
3. Directional-hemispherical albedo bounds and consistency
4. Roughness model effective emissivity bounds
5. Physical limits (w→0, θ̄→0)

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ===================================================================
# TEST 1: Henyey-Greenstein Phase Function
# ===================================================================


class TestHenyeyGreensteinPhase:
    """Verify the double Henyey-Greenstein phase function properties."""

    def test_backward_peak(self):
        """Phase function should peak near g=0 (opposition, cos_g=+1)
        for c=0.70 (70% backward scattering)."""
        from core_engine.reflectance import henyey_greenstein_double

        # cos_g = +1 means g = 0° (opposition/backscatter direction)
        # cos_g = -1 means g = 180° (forward scattering)
        p_opposition = henyey_greenstein_double(1.0, 0.21, 0.70)
        p_forward = henyey_greenstein_double(-1.0, 0.21, 0.70)
        # With c=0.70 (70% backward lobe), opposition peak should dominate
        assert p_opposition > p_forward, (
            f"Opposition peak not dominant: p(g=0°)={p_opposition}, "
            f"p(g=180°)={p_forward}"
        )

    def test_forward_peak_when_c_low(self):
        """With c=0.0 (equal mix), lobes are symmetric → p_fwd ≈ p_back."""
        from core_engine.reflectance import henyey_greenstein_double

        p_fwd = henyey_greenstein_double(1.0, 0.50, 0.0)
        p_back = henyey_greenstein_double(-1.0, 0.50, 0.0)
        # c=0 means 50/50 split: both lobes identical → equal values
        assert abs(p_fwd - p_back) < 0.01, (
            f"With c=0, lobes should be symmetric: "
            f"p_fwd={p_fwd}, p_back={p_back}"
        )

    def test_isotropic_when_b_zero(self):
        """With b=0 (isotropic grains), phase function should be ~1."""
        from core_engine.reflectance import henyey_greenstein_double

        p_0 = henyey_greenstein_double(0.0, 0.0, 0.5)
        p_90 = henyey_greenstein_double(1.0, 0.0, 0.5)
        assert abs(p_0 - 1.0) < 0.01, f"Not isotropic at g=90°: {p_0}"
        assert abs(p_90 - 1.0) < 0.01, f"Not isotropic at g=0°: {p_90}"

    def test_positive_definite(self):
        """Phase function must be non-negative for all angles."""
        from core_engine.reflectance import henyey_greenstein_double

        for cos_g in np.linspace(-1.0, 1.0, 100):
            p = henyey_greenstein_double(cos_g, 0.21, 0.70)
            assert p >= 0, f"Negative phase function at cos_g={cos_g}: {p}"


# ===================================================================
# TEST 2: Opposition Effect
# ===================================================================


class TestOppositionEffect:
    """Verify shadow-hiding opposition enhancement."""

    def test_peak_at_zero_phase(self):
        """B_SH should be maximum at g=0 (opposition)."""
        from core_engine.reflectance import shadow_hiding_opposition

        B_0 = shadow_hiding_opposition(1.0, 1.0, 0.065)  # g=0°
        B_90 = shadow_hiding_opposition(0.0, 1.0, 0.065)  # g=90°

        assert B_0 > B_90, f"B(0)={B_0} should exceed B(90)={B_90}"
        assert abs(B_0 - 1.0) < 0.01, f"B(0) should be ~B_SH0=1.0: {B_0}"

    def test_falls_off_with_phase(self):
        """Opposition effect should decline monotonically with g."""
        from core_engine.reflectance import shadow_hiding_opposition

        prev = shadow_hiding_opposition(1.0, 1.0, 0.065)
        for g_deg in range(5, 91, 5):
            cos_g = math.cos(math.radians(g_deg))
            B = shadow_hiding_opposition(cos_g, 1.0, 0.065)
            assert B <= prev + 1e-10, (
                f"B not monotonically decreasing at g={g_deg}°"
            )
            prev = B

    def test_zero_amplitude(self):
        """B_SH0=0 should give no opposition effect."""
        from core_engine.reflectance import shadow_hiding_opposition

        B = shadow_hiding_opposition(1.0, 0.0, 0.065)
        assert B == 0.0


# ===================================================================
# TEST 3: H-Function
# ===================================================================


class TestHFunction:
    """Verify Ambartsumian-Chandrasekhar H-function."""

    def test_no_scattering_limit(self):
        """For w=0 (no scattering), H(x) should be 1.0."""
        from core_engine.reflectance import h_function

        for x in [0.0, 0.5, 1.0]:
            H = h_function(x, 0.0)
            assert abs(H - 1.0) < 1e-10, f"H({x}, 0) = {H}, expected 1.0"

    def test_increases_with_w(self):
        """H should increase with w for fixed x."""
        from core_engine.reflectance import h_function

        H_low = h_function(0.5, 0.1)
        H_high = h_function(0.5, 0.9)
        assert H_high > H_low, f"H not increasing with w"

    def test_always_geq_1(self):
        """H(x, w) ≥ 1 for all valid inputs."""
        from core_engine.reflectance import h_function

        for x in np.linspace(0.0, 1.0, 20):
            for w in np.linspace(0.0, 0.99, 20):
                H = h_function(x, w)
                assert H >= 1.0 - 1e-10, f"H({x}, {w}) = {H} < 1.0"


# ===================================================================
# TEST 4: Full Hapke Reflectance
# ===================================================================


class TestHapkeReflectance:
    """Verify the full bidirectional reflectance function."""

    def test_positive_definite(self):
        """Reflectance must be non-negative."""
        from core_engine.reflectance import hapke_reflectance

        for cos_i in [0.1, 0.5, 1.0]:
            for cos_e in [0.1, 0.5, 1.0]:
                cos_g = cos_i * cos_e
                r = hapke_reflectance(
                    cos_i, cos_e, cos_g, 0.23, 0.21, 0.70, 1.0, 0.065
                )
                assert r >= 0, f"Negative reflectance: r={r}"

    def test_zero_albedo_gives_zero(self):
        """w=0 should give r=0 (no scattering at all)."""
        from core_engine.reflectance import hapke_reflectance

        r = hapke_reflectance(0.5, 0.5, 0.5, 0.0, 0.21, 0.70, 1.0, 0.065)
        assert abs(r) < 1e-15, f"Non-zero reflectance with w=0: {r}"

    def test_backscatter_enhancement(self):
        """Reflectance at opposition (g≈0) should be enhanced."""
        from core_engine.reflectance import hapke_reflectance

        # Near opposition: i≈e≈0, g≈0
        r_opposition = hapke_reflectance(
            0.99, 0.99, 1.0, 0.23, 0.21, 0.70, 1.0, 0.065
        )
        # Large phase angle
        r_large_phase = hapke_reflectance(
            0.99, 0.99, -0.5, 0.23, 0.21, 0.70, 1.0, 0.065
        )
        assert r_opposition > r_large_phase, (
            f"No backscatter enhancement: r(0°)={r_opposition} "
            f"vs r(120°)={r_large_phase}"
        )


# ===================================================================
# TEST 5: Directional-Hemispherical Albedo
# ===================================================================


class TestDirectionalHemisphericalAlbedo:
    """Verify the integrated DH albedo used in energy balance."""

    def test_bounded(self):
        """A_DH must be in [0, 1]."""
        from core_engine.reflectance import directional_hemispherical_albedo

        for cos_i in [0.1, 0.3, 0.5, 0.7, 1.0]:
            A = directional_hemispherical_albedo(cos_i, w=0.23)
            assert 0.0 <= A <= 1.0, f"A_DH out of bounds at cos_i={cos_i}: {A}"

    def test_consistent_with_bond_albedo(self):
        """A_DH at normal incidence should be near the Bond albedo (~0.12)."""
        from core_engine.reflectance import directional_hemispherical_albedo

        A_normal = directional_hemispherical_albedo(1.0, w=0.23)
        # Hapke A_DH at normal incidence should be roughly 0.10-0.20
        assert 0.05 < A_normal < 0.30, (
            f"A_DH(normal) = {A_normal}, expected ~0.12±0.08"
        )

    def test_zero_albedo_for_dark_surface(self):
        """w=0 should give A_DH=0."""
        from core_engine.reflectance import directional_hemispherical_albedo

        A = directional_hemispherical_albedo(0.5, w=0.0)
        assert abs(A) < 1e-10, f"A_DH should be 0 for w=0: {A}"

    def test_vectorized_consistency(self):
        """Vectorized compute_adh_array should match scalar version."""
        from core_engine.reflectance import (
            compute_adh_array,
            directional_hemispherical_albedo,
        )

        cos_arr = np.array([0.3, 0.7, 1.0])
        A_vec = compute_adh_array(cos_arr, w=0.23)

        for i, ci in enumerate(cos_arr):
            A_scalar = directional_hemispherical_albedo(ci, w=0.23)
            # Vectorized uses lower resolution (16x32 vs 32x64)
            assert abs(A_vec[i] - A_scalar) < 0.05, (
                f"Vectorized mismatch at cos_i={ci}: "
                f"vec={A_vec[i]}, scalar={A_scalar}"
            )


# ===================================================================
# TEST 6: Effective Emissivity — Roughness Model
# ===================================================================


class TestEffectiveEmissivity:
    """Verify Bandfield roughness emissivity correction."""

    def test_smooth_surface(self):
        """Zero roughness → ε_eff = ε₀."""
        from core_engine.roughness import effective_emissivity

        eps = effective_emissivity(0.95, 0.0)
        assert abs(eps - 0.95) < 1e-10, f"ε_eff(0°) = {eps}, expected 0.95"

    def test_roughness_increases_emissivity(self):
        """ε_eff ≥ ε₀ for θ̄ > 0 (cavity self-heating)."""
        from core_engine.roughness import effective_emissivity

        for theta in [5.0, 10.0, 20.0, 30.0, 45.0]:
            eps = effective_emissivity(0.95, theta)
            assert eps >= 0.95 - 1e-10, (
                f"ε_eff({theta}°) = {eps} < ε₀=0.95"
            )

    def test_bounded(self):
        """ε_eff must be in [ε₀, 1.0]."""
        from core_engine.roughness import effective_emissivity

        for theta in np.linspace(0, 90, 50):
            eps = effective_emissivity(0.95, theta)
            assert 0.95 <= eps <= 1.0 + 1e-10, (
                f"ε_eff({theta}°) = {eps} out of bounds"
            )

    def test_typical_regolith(self):
        """At θ̄=20° (typical), ε_eff should be ~0.95-0.97."""
        from core_engine.roughness import effective_emissivity

        eps = effective_emissivity(0.95, 20.0)
        assert 0.955 < eps < 0.97, (
            f"ε_eff(20°) = {eps}, expected ~0.957"
        )

    def test_extreme_roughness_approaches_blackbody(self):
        """At θ̄→90°, ε_eff should be higher than for moderate slopes."""
        from core_engine.roughness import effective_emissivity

        eps_89 = effective_emissivity(0.95, 89.0)
        eps_20 = effective_emissivity(0.95, 20.0)
        assert eps_89 > eps_20, (
            f"ε_eff(89°)={eps_89} should exceed ε_eff(20°)={eps_20}"
        )
        assert eps_89 > 0.96, f"ε_eff(89°) = {eps_89}, expected > 0.96"


# ===================================================================
# TEST 7: Face Slope Computation
# ===================================================================


class TestFaceSlopes:
    """Verify slope angle extraction from face normals."""

    def test_horizontal_face(self):
        """Horizontal face (normal = [0,0,1]) → slope = 0°."""
        from core_engine.roughness import compute_face_slopes

        normals = np.array([[0.0, 0.0, 1.0]])
        slopes = compute_face_slopes(normals)
        assert abs(slopes[0]) < 1e-10, f"Slope for flat face: {slopes[0]}°"

    def test_vertical_face(self):
        """Vertical face (normal = [1,0,0]) → slope = 90°."""
        from core_engine.roughness import compute_face_slopes

        normals = np.array([[1.0, 0.0, 0.0]])
        slopes = compute_face_slopes(normals)
        assert abs(slopes[0] - 90.0) < 1e-10, f"Slope: {slopes[0]}°"

    def test_45_degree_face(self):
        """45° tilted face → slope ≈ 45°."""
        from core_engine.roughness import compute_face_slopes

        n = np.array([[0.0, 0.0, 1.0]]) + np.array([[1.0, 0.0, 0.0]])
        n = n / np.linalg.norm(n)
        slopes = compute_face_slopes(n)
        assert abs(slopes[0] - 45.0) < 0.1, f"Slope: {slopes[0]}°"


# ===================================================================
# TEST 8: Roughness Pipeline
# ===================================================================


class TestRoughnessPipeline:
    """Verify the full roughness correction pipeline."""

    def test_pipeline_shape(self):
        """Output shape must match num_faces."""
        from core_engine.roughness import compute_roughness_correction

        normals = np.array([
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.866],
            [1.0, 0.0, 0.0],
        ])
        eps = compute_roughness_correction(normals, epsilon_0=0.95)
        assert eps.shape == (3,), f"Shape mismatch: {eps.shape}"

    def test_pipeline_monotonic(self):
        """Steeper faces should have higher ε_eff."""
        from core_engine.roughness import compute_roughness_correction

        normals = np.array([
            [0.0, 0.0, 1.0],    # flat
            [0.5, 0.0, 0.866],  # ~30° tilted
            [0.866, 0.0, 0.5],  # ~60° tilted
        ])
        eps = compute_roughness_correction(normals, epsilon_0=0.95)
        assert eps[0] <= eps[1] <= eps[2], (
            f"Not monotonic: {eps}"
        )
