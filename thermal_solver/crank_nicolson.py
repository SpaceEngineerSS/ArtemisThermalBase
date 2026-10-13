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

Derivation
----------
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
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    prange = range
else:
    from numba import prange

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
def _cp(T_val: float, cp_params: np.ndarray) -> float:
    """Specific heat polynomial of Hayne et al. (2017), floored at cp_min."""
    T = max(T_val, 1.0)
    val = (
        cp_params[0]
        + cp_params[1] * np.sqrt(T)
        + T * (cp_params[2] + T * (cp_params[3] + T * cp_params[4]))
    )
    return max(val, cp_params[5])


@njit(cache=True)
def _harmonic(k1: float, k2: float) -> float:
    s = k1 + k2
    if s < 1e-30:
        return 0.0
    return 2.0 * k1 * k2 / s


@njit(cache=True)
def _advance_column(
    T: np.ndarray,
    dz: np.ndarray,
    dz_bar: np.ndarray,
    rho: np.ndarray,
    k_contact: np.ndarray,
    k_radiative: np.ndarray,
    dt: float,
    Q_in: float,
    Q_geo: float,
    emissivity: float,
    sigma: float,
    newton_max_iter: int,
    newton_tol: float,
    newton_relax: float,
    cp_params: np.ndarray,
) -> None:
    """Advance one column by ``dt`` in place.

    Interior and bottom nodes use Crank-Nicolson. The surface half-cell is
    backward Euler: the radiative boundary on a 0.5 mm cell is very stiff,
    and the L-stable implicit step keeps the surface temperature from
    ringing at preview time steps, where Crank-Nicolson would oscillate.
    The Newton loop re-linearises εσT⁴ and stops when the largest change
    anywhere in the column is below ``newton_tol``.

    All per-node quantities that do not depend on temperature (ρ, the
    conductivity coefficients, grid spacing) are precomputed by the caller.
    """
    n = T.shape[0] - 1
    work = np.empty((7, n + 1), dtype=np.float64)
    T_old = work[0]
    a = work[1]
    b = work[2]
    c = work[3]
    d = work[4]
    k = work[5]
    cp_dt = work[6]
    for j in range(n + 1):
        T_old[j] = T[j]

    eps_sigma = emissivity * sigma
    for _ in range(newton_max_iter):
        for j in range(n + 1):
            Tj = max(T[j], 1.0)
            k[j] = k_contact[j] + k_radiative[j] * Tj * Tj * Tj
            cp_dt[j] = rho[j] * _cp(T[j], cp_params) / dt

        # Surface (j = 0): backward Euler with linearised radiation.
        T0 = max(T[0], 1.0)
        T0_3 = T0 * T0 * T0
        k_half = _harmonic(k[0], k[1])
        C0 = cp_dt[0] * 0.5 * dz[0]
        a[0] = 0.0
        b[0] = C0 + 4.0 * eps_sigma * T0_3 + k_half / dz[0]
        c[0] = -k_half / dz[0]
        d[0] = Q_in + 3.0 * eps_sigma * T0_3 * T0 + C0 * T_old[0]

        # Interior nodes: Crank-Nicolson.
        k_minus = k_half
        for j in range(1, n):
            k_plus = _harmonic(k[j], k[j + 1])
            alpha = k_minus / (dz[j - 1] * dz_bar[j - 1])
            gamma = k_plus / (dz[j] * dz_bar[j - 1])
            a[j] = -0.5 * alpha
            b[j] = cp_dt[j] + 0.5 * (alpha + gamma)
            c[j] = -0.5 * gamma
            d[j] = (
                0.5 * alpha * T_old[j - 1]
                + (cp_dt[j] - 0.5 * (alpha + gamma)) * T_old[j]
                + 0.5 * gamma * T_old[j + 1]
            )
            k_minus = k_plus

        # Bottom (j = n): Crank-Nicolson half-cell with the geothermal flux.
        beta = k_minus / dz[n - 1]
        Cn = cp_dt[n] * 0.5 * dz[n - 1]
        a[n] = -0.5 * beta
        b[n] = Cn + 0.5 * beta
        c[n] = 0.0
        d[n] = 0.5 * beta * T_old[n - 1] + (Cn - 0.5 * beta) * T_old[n] + Q_geo

        _thomas_solve(a, b, c, d)

        max_change = 0.0
        for j in range(n + 1):
            new_value = T[j] + newton_relax * (d[j] - T[j])
            if new_value < 1.0:
                new_value = 1.0
            change = abs(new_value - T[j])
            if change > max_change:
                max_change = change
            T[j] = new_value

        if max_change < newton_tol:
            break


@njit(cache=True, parallel=True)
def _advance_batch(
    temperatures: np.ndarray,
    dz: np.ndarray,
    dz_bar: np.ndarray,
    rho: np.ndarray,
    k_contact: np.ndarray,
    k_radiative: np.ndarray,
    dt: float,
    q_in: np.ndarray,
    q_geo: float,
    emissivities: np.ndarray,
    sigma: float,
    newton_max_iter: int,
    newton_tol: float,
    newton_relax: float,
    cp_params: np.ndarray,
) -> None:
    """Advance every column of a dense ``(faces, nodes)`` state in place."""
    for i in prange(temperatures.shape[0]):
        _advance_column(
            temperatures[i],
            dz,
            dz_bar,
            rho,
            k_contact,
            k_radiative,
            dt,
            q_in[i],
            q_geo,
            emissivities[i],
            sigma,
            newton_max_iter,
            newton_tol,
            newton_relax,
            cp_params,
        )


# ===================================================================
# HIGH-LEVEL SOLVER CLASS
# ===================================================================


class CrankNicolsonSolver:
    """1-D heat conduction solver for regolith columns.

    One solver serves every facet: the vertical grid and material profile
    are shared, only the temperatures and boundary fluxes differ per column.

    Parameters
    ----------
    config : SimulationConfig
        Full simulation configuration.

    """

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._sigma = config.constants.stefan_boltzmann
        self._emissivity = config.surface.thermal_emissivity
        self._Q_geo = config.regolith.geothermal_flux
        self._dt = config.solver.dt_s
        self._newton_max = config.solver.newton.max_iterations
        self._newton_tol = config.solver.newton.tolerance_K
        self._newton_relax = config.solver.newton.relaxation

        reg = config.regolith
        cp = reg.cp_coefficients
        self._cp_params = np.array(
            [cp[0], cp[1], cp[2], cp[3], cp[4], reg.cp_minimum],
            dtype=np.float64,
        )
        self._profiles: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        logger.info(
            "Thermal solver ready: dt=%.0f s, Newton max=%d, tol=%.1e K",
            self._dt,
            self._newton_max,
            self._newton_tol,
        )

    def material_profile(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Density and conductivity coefficients at each node of grid ``z``."""
        key = id(z)
        cached = self._profiles.get(key)
        if cached is not None and cached[0].shape == z.shape:
            return cached
        reg = self._config.regolith
        rho = reg.density_deep - (reg.density_deep - reg.density_surface) * np.exp(
            -z / reg.density_transition_m
        )
        upper = z < reg.conductivity_layer_boundary_m
        k_contact = np.where(
            upper, reg.conductivity_surface.k_contact, reg.conductivity_deep.k_contact
        )
        k_radiative = np.where(
            upper, reg.conductivity_surface.k_radiative, reg.conductivity_deep.k_radiative
        )
        profile = (
            np.ascontiguousarray(rho, dtype=np.float64),
            np.ascontiguousarray(k_contact, dtype=np.float64),
            np.ascontiguousarray(k_radiative, dtype=np.float64),
        )
        self._profiles[key] = profile
        return profile

    def step(
        self,
        column: ThermalColumn,
        Q_solar: float,
        Q_ir: float = 0.0,
        dt: float | None = None,
    ) -> None:
        """Advance a single column by one time step, updating ``column.T``.

        ``Q_solar`` is the absorbed solar flux, already including albedo,
        incidence and shadowing; ``Q_ir`` the absorbed terrain infrared.
        """
        rho, k_contact, k_radiative = self.material_profile(column.z)
        _advance_column(
            column.T,
            column.dz,
            column.dz_bar,
            rho,
            k_contact,
            k_radiative,
            self._dt if dt is None else dt,
            Q_solar + Q_ir,
            self._Q_geo,
            self._emissivity,
            self._sigma,
            self._newton_max,
            self._newton_tol,
            self._newton_relax,
            self._cp_params,
        )

    def step_batch(
        self,
        temperatures: np.ndarray,
        z: np.ndarray,
        dz: np.ndarray,
        dz_bar: np.ndarray,
        q_solar: np.ndarray,
        q_ir: np.ndarray | None = None,
        emissivities: np.ndarray | None = None,
        dt: float | None = None,
    ) -> None:
        """Advance a dense ``(num_faces, num_nodes)`` state in place."""
        num_faces = temperatures.shape[0]
        if q_solar.shape != (num_faces,):
            raise ValueError("q_solar must contain one value per face.")
        q_in = q_solar if q_ir is None else q_solar + q_ir
        if emissivities is None:
            emissivities = np.full(num_faces, self._emissivity, dtype=np.float64)
        if q_in.shape != (num_faces,) or emissivities.shape != (num_faces,):
            raise ValueError("q_ir and emissivities must contain one value per face.")
        rho, k_contact, k_radiative = self.material_profile(z)
        _advance_batch(
            temperatures,
            dz,
            dz_bar,
            rho,
            k_contact,
            k_radiative,
            self._dt if dt is None else dt,
            np.ascontiguousarray(q_in, dtype=np.float64),
            self._Q_geo,
            np.ascontiguousarray(emissivities, dtype=np.float64),
            self._sigma,
            self._newton_max,
            self._newton_tol,
            self._newton_relax,
            self._cp_params,
        )

    def compute_internal_energy(self, column: ThermalColumn) -> float:
        """Total internal energy ``Σ ρ cp T Δz`` of a column [J m⁻²].

        Used by the energy-conservation tests.
        """
        rho, _, _ = self.material_profile(column.z)
        n = len(column.T) - 1
        widths = np.empty(n + 1)
        widths[0] = column.dz[0] / 2.0
        widths[n] = column.dz[n - 1] / 2.0
        widths[1:n] = (column.dz[:-1] + column.dz[1:]) / 2.0
        cp = np.array([_cp(float(t), self._cp_params) for t in column.T])
        return float(np.sum(rho * cp * column.T * widths))
