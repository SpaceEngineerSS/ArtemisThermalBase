"""Crank-Nicolson 1D heat diffusion solver for lunar regolith.

Solves the 1D heat equation with:
- Non-uniform vertical grid (geometric spacing)
- Temperature-dependent thermal conductivity and specific heat
- Nonlinear surface radiation boundary (Newton-Raphson linearization)
- Constant geothermal flux bottom boundary

The Newton-linearized radiation term is folded directly into the
tridiagonal matrix coefficients, allowing the entire system to be
solved with a single Thomas algorithm sweep per Newton iteration.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

Derivation (validated via Sequential Thinking MCP — 6 steps)
------------------------------------------------------------
Heat equation:  ρ(z)·cp(T)·∂T/∂t = ∂/∂z[k(T)·∂T/∂z]

Crank-Nicolson (θ=0.5) on non-uniform grid z_j with Δz_j = z_{j+1} - z_j:

Interior nodes (j = 1, ..., N-1):
    a_j·T_{j-1}^{n+1} + b_j·T_j^{n+1} + c_j·T_{j+1}^{n+1} = d_j

where:
    α_j = k_{j-1/2} / (Δz_{j-1} · Δz̄_j)
    γ_j = k_{j+1/2} / (Δz_j · Δz̄_j)
    C_j = ρ_j · cp_j / Δt
    Δz̄_j = (Δz_j + Δz_{j-1}) / 2

    a_j = -0.5 · α_j
    b_j = C_j + 0.5 · (α_j + γ_j)
    c_j = -0.5 · γ_j
    d_j = 0.5·α_j·T_{j-1}^n + (C_j - 0.5·(α_j + γ_j))·T_j^n + 0.5·γ_j·T_{j+1}^n

Surface (j=0) — Newton-linearized radiation:
    b_0 = C̃_0 + 4·ε·σ·T̃³ + k_{1/2}/Δz_0
    c_0 = -k_{1/2}/Δz_0
    d_0 = Q_in + 3·ε·σ·T̃⁴ + C̃_0·T_0^n
    where C̃_0 = ρ_0·cp_0·(Δz_0/2)/Δt

Deep (j=N) — constant flux:
    a_N = -0.5·β_N
    b_N = C̃_N + 0.5·β_N
    d_N = 0.5·β_N·T_{N-1}^n + (C̃_N - 0.5·β_N)·T_N^n + Q_geo
    where β_N = k_{N-1/2}/Δz_{N-1}, C̃_N = ρ_N·cp_N·(Δz_{N-1}/2)/Δt

References
----------
- Crank, J. & Nicolson, P. (1947). Proc. Cambridge Phil. Soc., 43, 50-67.
- Spencer, J.R., et al. (1989). Icarus, 78, 337-354.
- Hayne, P.O., et al. (2017). JGR Planets, 122, 2371-2400.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numba import njit

from core_engine.constants import SimulationConfig

logger = logging.getLogger(__name__)


# ===================================================================
# DATA STRUCTURES
# ===================================================================


@dataclass
class ThermalColumn:
    """State of a single 1D thermal column (one DEM facet).

    Attributes
    ----------
    z : np.ndarray
        Depth grid [m]. Shape: (N+1,).
    T : np.ndarray
        Temperature at each grid node [K]. Shape: (N+1,).
    dz : np.ndarray
        Grid spacings Δz_j = z_{j+1} - z_j [m]. Shape: (N,).
    dz_bar : np.ndarray
        Averaged spacings for interior nodes. Shape: (N-1,).
        dz_bar_j = (dz_j + dz_{j-1}) / 2 for j = 1, ..., N-1.
    """

    z: np.ndarray
    T: np.ndarray
    dz: np.ndarray
    dz_bar: np.ndarray


def create_thermal_column(config: SimulationConfig) -> ThermalColumn:
    """Initialize a thermal column from configuration.

    Parameters
    ----------
    config : SimulationConfig
        Full simulation configuration.

    Returns
    -------
    ThermalColumn
        Initialized column with uniform temperature.
    """
    z = config.solver.grid.build_grid()
    N = len(z) - 1  # number of layers

    T = np.full(N + 1, config.solver.initial_temperature_K, dtype=np.float64)
    dz = np.diff(z)
    dz_bar = 0.5 * (dz[1:] + dz[:-1])  # (N-1,)

    logger.debug(
        "Thermal column initialized: %d nodes, z_max=%.3f m, T_init=%.1f K",
        N + 1,
        z[-1],
        config.solver.initial_temperature_K,
    )

    return ThermalColumn(z=z, T=T, dz=dz, dz_bar=dz_bar)


# ===================================================================
# THOMAS ALGORITHM — Numba JIT
# ===================================================================


@njit(cache=True)
def _thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system Ax = d using the Thomas algorithm.

    Modifies b and d in-place. The solution overwrites d.

    The matrix A has:
    - a[j] on the sub-diagonal (a[0] is unused)
    - b[j] on the main diagonal
    - c[j] on the super-diagonal (c[N] is unused)

    Parameters
    ----------
    a : np.ndarray
        Sub-diagonal coefficients. Shape: (N+1,). a[0] unused.
    b : np.ndarray
        Main diagonal coefficients. Shape: (N+1,). Modified in-place.
    c : np.ndarray
        Super-diagonal coefficients. Shape: (N+1,). c[N] unused.
    d : np.ndarray
        Right-hand side vector. Shape: (N+1,). Overwritten with solution.

    Returns
    -------
    np.ndarray
        Solution vector (same as d, modified in-place).

    Notes
    -----
    Stability is guaranteed because the heat equation discretization
    produces a strictly diagonally dominant matrix:
    |b_j| > |a_j| + |c_j| for all j (since C_j = ρ·cp/Δt > 0).
    """
    N = len(d) - 1

    # Forward elimination
    for j in range(1, N + 1):
        w = a[j] / b[j - 1]
        b[j] = b[j] - w * c[j - 1]
        d[j] = d[j] - w * d[j - 1]

    # Back substitution
    d[N] = d[N] / b[N]
    for j in range(N - 1, -1, -1):
        d[j] = (d[j] - c[j] * d[j + 1]) / b[j]

    return d


# ===================================================================
# SINGLE TIME STEP — Numba JIT
# ===================================================================


@njit(cache=True)
def _step_crank_nicolson(
    T: np.ndarray,
    z: np.ndarray,
    dz: np.ndarray,
    dz_bar: np.ndarray,
    dt: float,
    Q_solar: float,
    Q_ir: float,
    Q_geo: float,
    albedo: float,
    emissivity: float,
    sigma: float,
    newton_max_iter: int,
    newton_tol: float,
    newton_relax: float,
    k_func_params: np.ndarray,
    cp_func_params: np.ndarray,
    rho_func_params: np.ndarray,
) -> np.ndarray:
    """Advance one Crank-Nicolson time step with Newton surface iteration.

    Parameters
    ----------
    T : np.ndarray
        Current temperature profile [K]. Shape: (N+1,). NOT modified.
    z : np.ndarray
        Depth grid [m]. Shape: (N+1,).
    dz : np.ndarray
        Grid spacings [m]. Shape: (N,).
    dz_bar : np.ndarray
        Averaged spacings for interior nodes [m]. Shape: (N-1,).
    dt : float
        Time step [s].
    Q_solar : float
        Absorbed solar flux at surface [W/m²]. Already includes (1-A)·S·cos(θ)·f.
    Q_ir : float
        Incident IR radiation from surrounding terrain [W/m²].
    Q_geo : float
        Geothermal heat flux [W/m²].
    albedo : float
        Bond albedo [-] (for IR absorption: absorbed IR = (1 - albedo_ir) * Q_ir,
        but for thermal IR we assume full absorption, so Q_ir is already net absorbed).
    emissivity : float
        Thermal emissivity [-].
    sigma : float
        Stefan-Boltzmann constant [W/m²/K⁴].
    newton_max_iter : int
        Maximum Newton iterations for surface BC.
    newton_tol : float
        Newton convergence tolerance [K].
    newton_relax : float
        Newton under-relaxation factor (1.0 = no relaxation).
    k_func_params : np.ndarray
        Packed thermal conductivity parameters:
        [k_c_surf, k_r_surf, k_c_deep, k_r_deep, k_boundary].
    cp_func_params : np.ndarray
        Packed specific heat parameters: [c0, c1, c2, c3, c4, cp_min].
    rho_func_params : np.ndarray
        Packed density parameters: [rho_s, rho_d, H].

    Returns
    -------
    T_new : np.ndarray
        Updated temperature profile [K]. Shape: (N+1,).
    """
    N = len(T) - 1

    # --- Extract function parameters ---
    k_c_s = k_func_params[0]
    k_r_s = k_func_params[1]
    k_c_d = k_func_params[2]
    k_r_d = k_func_params[3]
    k_bnd = k_func_params[4]

    cp_c0 = cp_func_params[0]
    cp_c1 = cp_func_params[1]
    cp_c2 = cp_func_params[2]
    cp_c3 = cp_func_params[3]
    cp_c4 = cp_func_params[4]
    cp_min = cp_func_params[5]

    rho_s = rho_func_params[0]
    rho_d = rho_func_params[1]
    rho_H = rho_func_params[2]

    # --- Helper inline functions ---
    def _k(T_val: float, z_val: float) -> float:
        T_safe = max(T_val, 1.0)
        T3 = T_safe * T_safe * T_safe
        if z_val < k_bnd:
            return k_c_s + k_r_s * T3
        else:
            return k_c_d + k_r_d * T3

    def _cp(T_val: float) -> float:
        T_safe = max(T_val, 1.0)
        val = cp_c0 + cp_c1 * T_safe**0.5 + cp_c2 * T_safe + cp_c3 * T_safe**2 + cp_c4 * T_safe**3
        return max(val, cp_min)

    def _rho(z_val: float) -> float:
        return rho_d - (rho_d - rho_s) * np.exp(-z_val / rho_H)

    def _k_harmonic(k1: float, k2: float) -> float:
        """Harmonic mean of two conductivities."""
        if k1 + k2 < 1e-30:
            return 0.0
        return 2.0 * k1 * k2 / (k1 + k2)

    # Total absorbed flux at surface
    Q_in = Q_solar + Q_ir + Q_geo

    # --- Newton iteration loop ---
    T_new = T.copy()

    for newton_iter in range(newton_max_iter):
        # Allocate tridiagonal coefficients
        a = np.zeros(N + 1, dtype=np.float64)
        b = np.zeros(N + 1, dtype=np.float64)
        c = np.zeros(N + 1, dtype=np.float64)
        d = np.zeros(N + 1, dtype=np.float64)

        # --- Surface node (j = 0) ---
        # Newton-linearized radiation: ε·σ·T⁴ ≈ ε·σ·T̃⁴ + 4·ε·σ·T̃³·(T - T̃)
        #                              = -3·ε·σ·T̃⁴ + 4·ε·σ·T̃³·T
        T_guess_0 = max(T_new[0], 1.0)
        T_guess_0_3 = T_guess_0 * T_guess_0 * T_guess_0
        T_guess_0_4 = T_guess_0_3 * T_guess_0

        rho_0 = _rho(z[0])
        cp_0 = _cp(T_new[0])
        k_0 = _k(T_new[0], z[0])
        k_1 = _k(T_new[1], z[1])
        k_half_0 = _k_harmonic(k_0, k_1)

        # Surface half-cell: C̃_0 = ρ_0 · cp_0 · (Δz_0/2) / Δt
        C_tilde_0 = rho_0 * cp_0 * (dz[0] / 2.0) / dt

        a[0] = 0.0
        b[0] = C_tilde_0 + 4.0 * emissivity * sigma * T_guess_0_3 + k_half_0 / dz[0]
        c[0] = -k_half_0 / dz[0]
        d[0] = Q_in + 3.0 * emissivity * sigma * T_guess_0_4 + C_tilde_0 * T[0]

        # --- Interior nodes (j = 1, ..., N-1) ---
        for j in range(1, N):
            rho_j = _rho(z[j])
            cp_j = _cp(T_new[j])
            C_j = rho_j * cp_j / dt

            k_jm1 = _k(T_new[j - 1], z[j - 1])
            k_j = _k(T_new[j], z[j])
            k_jp1 = _k(T_new[j + 1], z[j + 1])

            k_half_minus = _k_harmonic(k_jm1, k_j)
            k_half_plus = _k_harmonic(k_j, k_jp1)

            alpha_j = k_half_minus / (dz[j - 1] * dz_bar[j - 1])
            gamma_j = k_half_plus / (dz[j] * dz_bar[j - 1])

            a[j] = -0.5 * alpha_j
            b[j] = C_j + 0.5 * (alpha_j + gamma_j)
            c[j] = -0.5 * gamma_j
            d[j] = (
                0.5 * alpha_j * T[j - 1]
                + (C_j - 0.5 * (alpha_j + gamma_j)) * T[j]
                + 0.5 * gamma_j * T[j + 1]
            )

        # --- Deep boundary node (j = N) ---
        rho_N = _rho(z[N])
        cp_N = _cp(T_new[N])
        k_Nm1 = _k(T_new[N - 1], z[N - 1])
        k_N = _k(T_new[N], z[N])
        k_half_N = _k_harmonic(k_Nm1, k_N)

        beta_N = k_half_N / dz[N - 1]
        C_tilde_N = rho_N * cp_N * (dz[N - 1] / 2.0) / dt

        a[N] = -0.5 * beta_N
        b[N] = C_tilde_N + 0.5 * beta_N
        c[N] = 0.0
        d[N] = 0.5 * beta_N * T[N - 1] + (C_tilde_N - 0.5 * beta_N) * T[N] + Q_geo

        # --- Solve tridiagonal system ---
        T_solved = _thomas_solve(a, b, c, d)

        # --- Check Newton convergence (surface node only) ---
        delta_T0 = abs(T_solved[0] - T_new[0])

        # Apply under-relaxation
        for j in range(N + 1):
            T_new[j] = T_new[j] + newton_relax * (T_solved[j] - T_new[j])

        # Clamp temperatures to physical range (T > 0)
        for j in range(N + 1):
            if T_new[j] < 1.0:
                T_new[j] = 1.0

        if delta_T0 < newton_tol:
            break

    return T_new


# ===================================================================
# HIGH-LEVEL SOLVER CLASS
# ===================================================================


class CrankNicolsonSolver:
    """1D Crank-Nicolson thermal solver for a single regolith column.

    This solver manages the thermal state of one DEM facet's subsurface
    column. It is called once per facet per time step.

    Parameters
    ----------
    config : SimulationConfig
        Full simulation configuration.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._sigma = config.constants.stefan_boltzmann
        self._albedo = config.surface.bond_albedo
        self._emissivity = config.surface.thermal_emissivity
        self._Q_geo = config.regolith.geothermal_flux
        self._dt = config.solver.dt_s
        self._newton_max = config.solver.newton.max_iterations
        self._newton_tol = config.solver.newton.tolerance_K
        self._newton_relax = config.solver.newton.relaxation

        # Pack parameters into flat arrays for Numba
        reg = config.regolith
        self._k_params = np.array(
            [
                reg.conductivity_surface.k_contact,
                reg.conductivity_surface.k_radiative,
                reg.conductivity_deep.k_contact,
                reg.conductivity_deep.k_radiative,
                reg.conductivity_layer_boundary_m,
            ],
            dtype=np.float64,
        )
        cp = reg.cp_coefficients
        self._cp_params = np.array(
            [cp[0], cp[1], cp[2], cp[3], cp[4], reg.cp_minimum],
            dtype=np.float64,
        )
        self._rho_params = np.array(
            [reg.density_surface, reg.density_deep, reg.density_transition_m],
            dtype=np.float64,
        )

        logger.info(
            "CrankNicolsonSolver initialized: dt=%.0fs, newton_max=%d, "
            "newton_tol=%.1e K, emissivity=%.3f, sigma=%.6e",
            self._dt,
            self._newton_max,
            self._newton_tol,
            self._emissivity,
            self._sigma,
        )

    def step(
        self,
        column: ThermalColumn,
        Q_solar: float,
        Q_ir: float = 0.0,
        dt: float | None = None,
    ) -> None:
        """Advance the thermal column by one time step.

        Updates ``column.T`` in-place.

        Parameters
        ----------
        column : ThermalColumn
            Thermal column state to advance.
        Q_solar : float
            Absorbed solar flux at the surface [W/m²].
            This should already include (1-A)·S·cos(θ)·f.
        Q_ir : float
            Absorbed IR from surrounding terrain [W/m²]. Default: 0.
        dt : float, optional
            Override time step [s]. If None, uses config value.
        """
        if dt is None:
            dt = self._dt

        T_new = _step_crank_nicolson(
            T=column.T,
            z=column.z,
            dz=column.dz,
            dz_bar=column.dz_bar,
            dt=dt,
            Q_solar=Q_solar,
            Q_ir=Q_ir,
            Q_geo=self._Q_geo,
            albedo=self._albedo,
            emissivity=self._emissivity,
            sigma=self._sigma,
            newton_max_iter=self._newton_max,
            newton_tol=self._newton_tol,
            newton_relax=self._newton_relax,
            k_func_params=self._k_params,
            cp_func_params=self._cp_params,
            rho_func_params=self._rho_params,
        )

        column.T[:] = T_new

    def compute_surface_radiation(self, T_surf: float) -> float:
        """Compute the outgoing thermal radiation flux.

        Parameters
        ----------
        T_surf : float
            Surface temperature [K].

        Returns
        -------
        float
            Radiated flux ε·σ·T⁴ [W/m²].
        """
        return self._emissivity * self._sigma * T_surf**4

    def compute_internal_energy(self, column: ThermalColumn) -> float:
        """Compute the total internal energy stored in the column.

        E = Σ_j ρ(z_j) · cp(T_j) · T_j · Δz_j

        Used for energy conservation verification.

        Parameters
        ----------
        column : ThermalColumn
            Thermal column state.

        Returns
        -------
        float
            Total internal energy [J/m²].
        """
        reg = self._config.regolith
        E = 0.0
        N = len(column.T) - 1

        for j in range(N + 1):
            T_j = column.T[j]
            z_j = column.z[j]

            # Density
            rho_j = reg.density_deep - (reg.density_deep - reg.density_surface) * np.exp(
                -z_j / reg.density_transition_m
            )

            # Specific heat
            c = reg.cp_coefficients
            T_safe = max(T_j, 1.0)
            cp_j = (
                c[0] + c[1] * T_safe**0.5 + c[2] * T_safe
                + c[3] * T_safe**2 + c[4] * T_safe**3
            )
            cp_j = max(cp_j, reg.cp_minimum)

            # Layer thickness (half-cells at boundaries)
            if j == 0:
                dz_j = column.dz[0] / 2.0
            elif j == N:
                dz_j = column.dz[N - 1] / 2.0
            else:
                dz_j = (column.dz[j - 1] + column.dz[j]) / 2.0

            E += rho_j * cp_j * T_j * dz_j

        return E
