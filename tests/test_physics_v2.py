"""Tests for terrain IR exchange and the volatile sublimation model.


Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import numpy as np
import pytest

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
    """Hertz-Knudsen sublimation with Murphy & Koop (2005) vapour pressure."""

    def test_sublimation_zero_at_zero_temp(self):
        """No sublimation at absolute zero."""
        from thermal_solver.volatiles import sublimation_rate

        assert sublimation_rate(0.0) == 0.0
        assert sublimation_rate(0.5) == 0.0

    def test_sublimation_increases_with_temperature(self):
        """The flux rises monotonically with temperature."""
        from thermal_solver.volatiles import sublimation_rate

        rates = [sublimation_rate(T) for T in (60.0, 100.0, 150.0, 200.0, 250.0)]
        assert all(b > a for a, b in zip(rates, rates[1:], strict=False))

    def test_vapor_pressure_reference_values(self):
        """Murphy & Koop (2005) eq. 7: about 0.16 Pa at 200 K, 611 Pa at 273.16 K."""
        from thermal_solver.volatiles import vapor_pressure_ice

        assert vapor_pressure_ice(273.16) == pytest.approx(611.7, rel=0.01)
        assert vapor_pressure_ice(200.0) == pytest.approx(0.1626, rel=0.02)

    def test_mass_flux_at_110K(self):
        """At 110 K the loss is ~4.5e-15 kg/m2/s: a few Gyr per metre of ice."""
        from thermal_solver.volatiles import ice_retention_timescale, sublimation_rate

        assert sublimation_rate(110.0) == pytest.approx(4.5e-15, rel=0.1)
        assert 1e9 < ice_retention_timescale(110.0) < 1e11

    def test_ice_is_lost_quickly_at_200K(self):
        """At 200 K a metre of exposed ice is gone within months."""
        from thermal_solver.volatiles import ice_retention_timescale

        assert ice_retention_timescale(200.0) < 1.0


# ===================================================================
# TEST 6: Cold Trap Stability Classification
# ===================================================================


class TestColdTrapStability:
    """Test ice stability classification (Powell & Rubanenko 2020)."""

    def test_stable_below_110K(self):
        """T < 110 K → stable (class 0)."""
        from thermal_solver.volatiles import (
            STABILITY_STABLE,
            ice_stability_class,
        )

        assert ice_stability_class(50.0) == STABILITY_STABLE
        assert ice_stability_class(100.0) == STABILITY_STABLE
        assert ice_stability_class(109.9) == STABILITY_STABLE

    def test_marginal_110_to_115K(self):
        """110 K ≤ T < 115 K → marginal (class 1)."""
        from thermal_solver.volatiles import (
            STABILITY_MARGINAL,
            ice_stability_class,
        )

        assert ice_stability_class(110.0) == STABILITY_MARGINAL
        assert ice_stability_class(112.5) == STABILITY_MARGINAL
        assert ice_stability_class(114.9) == STABILITY_MARGINAL

    def test_unstable_above_115K(self):
        """T ≥ 115 K → unstable (class 2)."""
        from thermal_solver.volatiles import (
            STABILITY_UNSTABLE,
            ice_stability_class,
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
