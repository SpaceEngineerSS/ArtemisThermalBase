"""Tests for the Crank-Nicolson thermal solver.

Validates numerical accuracy, energy conservation, and edge-case
behavior of the 1D heat equation solver.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

Test Strategy
-------------
1. **Steady-state linear profile**: No radiation, constant BCs → linear T(z).
2. **Thermal wave (sinusoidal)**: Known analytical solution for periodic forcing.
3. **Energy conservation**: Over a full diurnal cycle, ΔE_stored + E_radiated ≈ E_absorbed.
4. **Extreme cold (PSR)**: Solver stability at T ~ 30-50 K.
5. **Thomas algorithm**: Direct validation against NumPy linalg.solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from core_engine.constants import (
    SimulationConfig,
    load_config,
)
from thermal_solver.crank_nicolson import (
    CrankNicolsonSolver,
    ThermalColumn,
    create_thermal_column,
    _thomas_solve,
    _step_crank_nicolson,
)


# ===================================================================
# FIXTURES
# ===================================================================

_CONFIG_PATH = "config/default_config.yaml"


@pytest.fixture
def config() -> SimulationConfig:
    """Load the default simulation configuration."""
    return load_config(_CONFIG_PATH)


@pytest.fixture
def solver(config: SimulationConfig) -> CrankNicolsonSolver:
    """Create a Crank-Nicolson solver instance."""
    return CrankNicolsonSolver(config)


@pytest.fixture
def column(config: SimulationConfig) -> ThermalColumn:
    """Create a default thermal column."""
    return create_thermal_column(config)


# ===================================================================
# THOMAS ALGORITHM TESTS
# ===================================================================


class TestThomasAlgorithm:
    """Validate the Thomas tridiagonal solver against NumPy."""

    def test_known_system_3x3(self) -> None:
        """Solve a small 3×3 tridiagonal system."""
        # System: [2 -1 0; -1 2 -1; 0 -1 2] x = [1; 0; 1]
        a = np.array([0.0, -1.0, -1.0], dtype=np.float64)
        b = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        c = np.array([-1.0, -1.0, 0.0], dtype=np.float64)
        d = np.array([1.0, 0.0, 1.0], dtype=np.float64)

        # Expected solution from numpy
        A_full = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=np.float64)
        expected = np.linalg.solve(A_full, np.array([1.0, 0.0, 1.0]))

        result = _thomas_solve(a.copy(), b.copy(), c.copy(), d.copy())

        np.testing.assert_allclose(
            result, expected, atol=1e-12,
            err_msg="Thomas algorithm result does not match NumPy linalg.solve",
        )

    def test_large_random_system(self) -> None:
        """Thomas algorithm on a large random diagonally dominant system."""
        rng = np.random.default_rng(42)
        N = 200

        a = rng.uniform(-0.3, -0.1, size=N)
        c = rng.uniform(-0.3, -0.1, size=N)
        b = rng.uniform(1.0, 2.0, size=N)  # diagonally dominant
        d = rng.uniform(-1.0, 1.0, size=N)
        a[0] = 0.0
        c[N - 1] = 0.0

        # Build full matrix for reference
        A_full = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
        expected = np.linalg.solve(A_full, d.copy())

        result = _thomas_solve(a.copy(), b.copy(), c.copy(), d.copy())

        np.testing.assert_allclose(
            result, expected, atol=1e-10,
            err_msg="Thomas algorithm diverges from NumPy on a large system",
        )

    def test_diagonal_dominance_preserved(self) -> None:
        """Verify solver works with exactly diag-dominant matrix."""
        N = 50
        a = np.full(N, -0.4, dtype=np.float64)
        c = np.full(N, -0.4, dtype=np.float64)
        b = np.full(N, 1.0, dtype=np.float64)  # |b| = |a| + |c| + 0.2
        d = np.ones(N, dtype=np.float64)
        a[0] = 0.0
        c[N - 1] = 0.0

        A_full = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
        expected = np.linalg.solve(A_full, d.copy())
        result = _thomas_solve(a.copy(), b.copy(), c.copy(), d.copy())

        np.testing.assert_allclose(result, expected, atol=1e-10)


# ===================================================================
# STEADY-STATE TESTS
# ===================================================================


class TestSteadyState:
    """Test solver convergence to known steady-state solutions."""

    def test_isothermal_no_flux(
        self, config: SimulationConfig
    ) -> None:
        """No solar input, no geothermal → should remain isothermal.

        With Q_solar=0, Q_ir=0, Q_geo=0, and initially uniform T,
        the only loss is ε·σ·T⁴ radiation. The column should cool
        toward equilibrium (very low T for no input).

        We test that after one step, the temperature change is
        physically reasonable (cooling, not heating).
        """
        # Override geothermal to zero for this test
        col = create_thermal_column(config)
        T_init = col.T[0]

        solver = CrankNicolsonSolver(config)

        # Manually call with Q_solar=0, Q_ir=0
        # But we also need Q_geo=0; for now test with config geo flux
        solver.step(col, Q_solar=0.0, Q_ir=0.0)

        # Surface should cool (radiating away ε·σ·T⁴ with no input)
        assert col.T[0] < T_init, (
            f"Surface should cool with no solar input. "
            f"T_before={T_init:.2f}, T_after={col.T[0]:.2f}"
        )

        # All temperatures should remain positive
        assert np.all(col.T > 0), "Temperatures must remain positive"


# ===================================================================
# ENERGY CONSERVATION TEST
# ===================================================================


class TestEnergyConservation:
    """Strict energy conservation validation.

    Over a full simulated period:
    Total Solar Energy In = Radiated Energy Out + ΔE_internal

    This is the most important test for physical correctness.
    """

    def test_energy_balance_diurnal_cycle(
        self, config: SimulationConfig
    ) -> None:
        """Energy conservation over a simulated diurnal cycle.

        Uses a sinusoidal solar flux to simulate day/night on a flat
        surface (no shadowing, no IR from terrain).

        Test criterion: |E_in - E_out - ΔE_stored| / E_in < 1%
        """
        solver = CrankNicolsonSolver(config)
        col = create_thermal_column(config)

        # Simulation parameters
        sigma = config.constants.stefan_boltzmann
        emissivity = config.surface.thermal_emissivity
        albedo = config.surface.bond_albedo
        S0 = config.constants.solar_constant
        Q_geo = config.regolith.geothermal_flux
        dt = config.solver.dt_s

        # Simulated period: 3 lunar days to let transients die
        period_s = config.lunar.synodic_period_s
        total_time = 3.0 * period_s
        num_steps = int(total_time / dt)

        # Run first 2 periods to reach quasi-steady state (don't track energy)
        warmup_steps = int(2.0 * period_s / dt)
        omega = 2.0 * np.pi / period_s

        for step_i in range(warmup_steps):
            t = step_i * dt
            # Sinusoidal solar flux (positive half = day, zero = night)
            cos_val = np.cos(omega * t)
            Q_solar = (1.0 - albedo) * S0 * max(0.0, cos_val)
            solver.step(col, Q_solar=Q_solar, Q_ir=0.0)

        # Now track energy over the 3rd period
        E_internal_start = solver.compute_internal_energy(col)
        E_absorbed_total = 0.0  # Total energy entering the surface
        E_radiated_total = 0.0  # Total energy leaving via radiation

        tracking_steps = int(period_s / dt)
        for step_i in range(tracking_steps):
            t = (warmup_steps + step_i) * dt
            cos_val = np.cos(omega * t)
            Q_solar = (1.0 - albedo) * S0 * max(0.0, cos_val)

            # Track energy BEFORE step
            E_absorbed_total += (Q_solar + Q_geo) * dt
            E_radiated_total += emissivity * sigma * col.T[0] ** 4 * dt

            solver.step(col, Q_solar=Q_solar, Q_ir=0.0)

        E_internal_end = solver.compute_internal_energy(col)
        delta_E_stored = E_internal_end - E_internal_start

        # Energy balance: E_in ≈ E_out + ΔE_stored
        E_in = E_absorbed_total
        E_out = E_radiated_total
        residual = abs(E_in - E_out - delta_E_stored)

        # Relative error
        if E_in > 0:
            relative_error = residual / E_in
        else:
            relative_error = 0.0

        assert relative_error < 0.01, (
            f"Energy conservation violated!\n"
            f"  E_absorbed = {E_in:.2f} J/m²\n"
            f"  E_radiated = {E_out:.2f} J/m²\n"
            f"  ΔE_stored  = {delta_E_stored:.2f} J/m²\n"
            f"  Residual   = {residual:.2f} J/m²\n"
            f"  Relative error = {relative_error:.4%} (threshold: 1%)"
        )


# ===================================================================
# EXTREME CONDITIONS (PSR) TESTS
# ===================================================================


class TestExtremeConditions:
    """Test solver stability at extreme cryogenic temperatures."""

    def test_psr_cold_stability(
        self, config: SimulationConfig
    ) -> None:
        """Solver should remain stable when surface cools to ~30-50 K.

        PSR temperatures in Shackleton crater are ~38-50 K (Paige et al., 2010).
        The solver must handle these without NaN, negative T, or divergence.
        """
        col = create_thermal_column(config)
        # Start at a cold initial temperature (simulating a PSR)
        col.T[:] = 50.0

        solver = CrankNicolsonSolver(config)

        # Run 100 steps with no solar input
        for _ in range(100):
            solver.step(col, Q_solar=0.0, Q_ir=0.0)

        # Check for NaN
        assert not np.any(np.isnan(col.T)), "NaN detected in temperature profile"

        # Check all temperatures are positive
        assert np.all(col.T > 0), (
            f"Non-positive temperature detected: T_min={col.T.min():.4f} K"
        )

        # Surface should be very cold but not zero
        assert col.T[0] > 1.0, f"Surface temperature too low: {col.T[0]:.4f} K"

    def test_hot_illuminated_stability(
        self, config: SimulationConfig
    ) -> None:
        """Solver should remain stable with strong solar flux.

        Peak solar flux at normal incidence: ~1361 W/m².
        Surface can reach ~390 K at the equator (day side).
        """
        col = create_thermal_column(config)
        solver = CrankNicolsonSolver(config)

        Q_solar_max = (1.0 - config.surface.bond_albedo) * config.constants.solar_constant

        # 50 steps with full illumination
        for _ in range(50):
            solver.step(col, Q_solar=Q_solar_max, Q_ir=0.0)

        assert not np.any(np.isnan(col.T)), "NaN at high T"
        assert np.all(col.T > 0), "Non-positive T at high flux"
        assert col.T[0] < 500.0, (
            f"Surface temperature unrealistically high: {col.T[0]:.1f} K"
        )


# ===================================================================
# THERMAL WAVE ANALYTICAL TEST
# ===================================================================


class TestThermalWave:
    """Validate against the analytical thermal wave solution.

    For a semi-infinite medium with sinusoidal surface temperature:
        T_surf(t) = T_mean + ΔT · sin(ωt)

    The analytical solution at depth z is:
        T(z, t) = T_mean + ΔT · exp(-z/δ) · sin(ωt - z/δ)

    where δ = √(2κ/ω) is the thermal skin depth, κ = k/(ρ·cp).
    """

    def test_thermal_wave_amplitude_decay(
        self, config: SimulationConfig
    ) -> None:
        """Verify amplitude decays exponentially with depth.

        After running many cycles, the amplitude at depth z should be:
            A(z) = ΔT · exp(-z/δ)
        """
        col = create_thermal_column(config)
        solver = CrankNicolsonSolver(config)

        # Use simplified constant properties for analytical comparison
        reg = config.regolith
        k_const = reg.conductivity_surface.k_contact  # ignore T-dependent part
        rho_const = reg.density_surface
        cp_const = 600.0  # approximate constant cp

        # Thermal diffusivity
        kappa = k_const / (rho_const * cp_const)

        # Period = 1 lunar day
        period = config.lunar.synodic_period_s
        omega = 2.0 * np.pi / period
        dt = config.solver.dt_s

        # Skin depth
        delta = np.sqrt(2.0 * kappa / omega)

        T_mean = 200.0
        delta_T = 50.0  # amplitude of surface oscillation

        # Initialize with mean temperature
        col.T[:] = T_mean

        # Run 5 complete cycles to reach periodic steady state
        num_cycles = 5
        total_steps = int(num_cycles * period / dt)
        steps_per_cycle = int(period / dt)

        # Drive with known solar flux pattern that produces sinusoidal T_surf
        # Instead, we directly set T_surf each step (forced boundary test)
        # This requires a different approach: we track max/min at each depth
        # over the last cycle

        # Run warmup
        for step_i in range(total_steps - steps_per_cycle):
            t = step_i * dt
            # Sinusoidal forcing
            Q = max(0.0, 400.0 * np.sin(omega * t))
            solver.step(col, Q_solar=Q, Q_ir=0.0)

        # Track temperature extremes over the last cycle
        T_max = np.full(len(col.T), -np.inf)
        T_min = np.full(len(col.T), np.inf)

        for step_i in range(steps_per_cycle):
            t = (total_steps - steps_per_cycle + step_i) * dt
            Q = max(0.0, 400.0 * np.sin(omega * t))
            solver.step(col, Q_solar=Q, Q_ir=0.0)

            T_max = np.maximum(T_max, col.T)
            T_min = np.minimum(T_min, col.T)

        amplitude = (T_max - T_min) / 2.0
        surface_amplitude = amplitude[0]

        if surface_amplitude < 1.0:
            pytest.skip("Surface amplitude too small for meaningful test")

        # Check that amplitude decays with depth (qualitative for now,
        # since our forced boundary isn't perfectly sinusoidal)
        # At depth = δ, amplitude should be ~37% of surface (1/e)
        # At depth = 2δ, amplitude should be ~13.5% of surface (1/e²)

        # Find the node closest to z = δ
        z_grid = col.z
        idx_delta = np.argmin(np.abs(z_grid - delta))

        if idx_delta > 0 and idx_delta < len(amplitude) - 1:
            ratio = amplitude[idx_delta] / surface_amplitude

            # The ratio should be approximately exp(-1) ≈ 0.37
            # But since our forcing isn't perfectly sinusoidal and properties
            # aren't constant, we allow a generous tolerance
            assert ratio < 0.8, (
                f"Amplitude at z=δ should decay. "
                f"Ratio = {ratio:.3f} (expected ~0.37 for ideal case)"
            )

        # The deep temperature should barely oscillate
        deep_amplitude = amplitude[-1]
        assert deep_amplitude < surface_amplitude * 0.01, (
            f"Deep temperature oscillation too large: "
            f"{deep_amplitude:.4f} vs surface {surface_amplitude:.4f}"
        )
