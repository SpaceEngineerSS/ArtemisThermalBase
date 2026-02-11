"""Temperature-dependent regolith thermophysical properties.

Implements the Hayne et al. (2017) model for lunar regolith thermal
conductivity k(T), specific heat c_p(T), and depth-dependent density ρ(z).
All parameters are loaded from configuration — nothing is hardcoded.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

References
----------
- Hayne, P.O., et al. (2017). "Global Regolith Thermophysical Properties
  of the Moon from the Diviner Lunar Radiometer Experiment."
  JGR Planets, 122, 2371-2400.
- Hemingway, B.S., et al. (1973). "Thermophysical properties of lunar
  surface materials." USGS Professional Paper 727.
"""

from __future__ import annotations

import logging

import numpy as np
from numba import njit

from core_engine.constants import RegolithProperties

logger = logging.getLogger(__name__)


def build_property_functions(
    regolith: RegolithProperties,
) -> tuple:
    """Build Numba-compatible property functions from configuration.

    Since Numba ``@njit`` functions cannot access Python objects, this
    factory pre-extracts all numerical parameters and returns closures
    (Numba-compiled functions) that capture them.

    Parameters
    ----------
    regolith : RegolithProperties
        Regolith configuration from YAML.

    Returns
    -------
    thermal_conductivity : callable
        k(T, z) → float64. Thermal conductivity [W/m/K].
    specific_heat : callable
        cp(T) → float64. Specific heat capacity [J/kg/K].
    density : callable
        rho(z) → float64. Bulk density [kg/m³].
    """
    # Extract all parameters as plain floats
    k_c_surf = regolith.conductivity_surface.k_contact
    k_r_surf = regolith.conductivity_surface.k_radiative
    k_c_deep = regolith.conductivity_deep.k_contact
    k_r_deep = regolith.conductivity_deep.k_radiative
    k_boundary = regolith.conductivity_layer_boundary_m

    c0, c1, c2, c3, c4 = regolith.cp_coefficients
    cp_min = regolith.cp_minimum

    rho_s = regolith.density_surface
    rho_d = regolith.density_deep
    H = regolith.density_transition_m

    logger.info(
        "Building regolith property functions: "
        "k_c_surf=%.2e, k_r_surf=%.2e, cp_min=%.1f, rho_s=%.0f, rho_d=%.0f",
        k_c_surf,
        k_r_surf,
        cp_min,
        rho_s,
        rho_d,
    )

    @njit(cache=True)
    def thermal_conductivity(T: float, z: float) -> float:
        """Compute thermal conductivity k(T, z) [W/m/K].

        k(T) = k_contact + k_radiative * T^3

        Uses surface-layer parameters for z < layer_boundary,
        deep-layer parameters otherwise.

        Parameters
        ----------
        T : float
            Temperature [K]. Must be > 0.
        z : float
            Depth below surface [m]. Must be >= 0.

        Returns
        -------
        float
            Thermal conductivity [W/m/K].
        """
        # Clamp temperature to avoid negative k at T=0
        T_safe = max(T, 1.0)

        if z < k_boundary:
            k = k_c_surf + k_r_surf * T_safe * T_safe * T_safe
        else:
            k = k_c_deep + k_r_deep * T_safe * T_safe * T_safe

        return k

    @njit(cache=True)
    def specific_heat(T: float) -> float:
        """Compute specific heat capacity c_p(T) [J/kg/K].

        Polynomial fit from Hayne et al. (2017):
        c_p(T) = c0 + c1*T^0.5 + c2*T + c3*T^2 + c4*T^3

        Clamped to cp_minimum at low temperatures to prevent singularity.

        Parameters
        ----------
        T : float
            Temperature [K].

        Returns
        -------
        float
            Specific heat capacity [J/kg/K].
        """
        # Clamp temperature to avoid domain errors in sqrt
        T_safe = max(T, 1.0)

        cp = c0 + c1 * T_safe**0.5 + c2 * T_safe + c3 * T_safe**2 + c4 * T_safe**3

        # Clamp to physical minimum (avoids singularity at T→0)
        return max(cp, cp_min)

    @njit(cache=True)
    def density(z: float) -> float:
        """Compute bulk density ρ(z) [kg/m³].

        Exponential transition from surface to deep density:
        ρ(z) = ρ_deep - (ρ_deep - ρ_surface) * exp(-z / H)

        Parameters
        ----------
        z : float
            Depth below surface [m]. Must be >= 0.

        Returns
        -------
        float
            Bulk density [kg/m³].
        """
        return rho_d - (rho_d - rho_s) * np.exp(-z / H)

    return thermal_conductivity, specific_heat, density


def compute_property_profiles(
    z_grid: np.ndarray,
    T_profile: np.ndarray,
    thermal_conductivity,
    specific_heat,
    density,
) -> dict[str, np.ndarray]:
    """Compute full property profiles on the vertical grid.

    Useful for diagnostics and visualization.

    Parameters
    ----------
    z_grid : np.ndarray
        Depth grid [m]. Shape: (N+1,).
    T_profile : np.ndarray
        Temperature profile [K]. Shape: (N+1,).
    thermal_conductivity : callable
        k(T, z) function.
    specific_heat : callable
        cp(T) function.
    density : callable
        rho(z) function.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with 'k', 'cp', 'rho' arrays. Shape: (N+1,) each.
    """
    N = len(z_grid)
    k_arr = np.empty(N, dtype=np.float64)
    cp_arr = np.empty(N, dtype=np.float64)
    rho_arr = np.empty(N, dtype=np.float64)

    for i in range(N):
        k_arr[i] = thermal_conductivity(T_profile[i], z_grid[i])
        cp_arr[i] = specific_heat(T_profile[i])
        rho_arr[i] = density(z_grid[i])

    return {"k": k_arr, "cp": cp_arr, "rho": rho_arr}
