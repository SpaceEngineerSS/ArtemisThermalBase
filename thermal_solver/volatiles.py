"""Water ice sublimation and cold trap stability model.

Computes the sublimation rate of water ice on the lunar surface using
the Hertz-Knudsen (Langmuir) equation, and classifies surface locations
as thermally stable, marginal, or unstable cold traps based on
temperature thresholds from Powell & Rubanenko (2020).

Physics
-------
The sublimation mass flux of water ice at temperature T is given by
the Hertz-Knudsen equation (kinetic theory of gases):

    dm/dt = P_vap(T) / √(2π · m_mol · k_B · T)

where P_vap is the equilibrium vapor pressure of water ice.

In logarithmic form (simplified Clausius-Clapeyron):

    log₁₀(dm/dt) ≈ A - B/T - 0.5·log₁₀(T)

with Langmuir coefficients A ≈ 14.88, B ≈ 6141 for H₂O ice.

Cold Trap Stability Thresholds
------------------------------
- T < 110 K: Stable for > 1 Gyr (geologically permanent trap)
- 110 K ≤ T < 115 K: Stable for geological timescales (~100 Myr)
- T ≥ 115 K: Unstable (ice sublimates within ~1 Myr or less)

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

References
----------
- Powell, T.M. & Rubanenko, L. (2020). "Characterization of the
  distribution of water ice cold traps in the southern hemisphere
  of the Moon." AGU Fall Meeting.
- Schorghofer, N. (2008). "The lifetime of ice on main belt
  asteroids." ApJ, 682, 697-705.
- Andreas, E.L. (2007). "New estimates for the sublimation rate
  for ice on the Moon." Icarus, 186, 24-30.

"""

from __future__ import annotations

import logging

import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)


# ===================================================================
# PHYSICAL CONSTANTS
# ===================================================================

# Boltzmann constant [J/K]
_K_B: float = 1.380649e-23

# Molecular mass of water [kg]
_M_H2O: float = 2.9915e-26  # 18.015 g/mol / (6.022e23 /mol)

# Langmuir coefficients for H₂O ice sublimation (log₁₀ form)
# Derived from Clausius-Clapeyron equation
# log₁₀(dm/dt [kg/m²/s]) ≈ A - B/T - 0.5·log₁₀(T)
_LANGMUIR_A: float = 14.88
_LANGMUIR_B: float = 6141.0

# Stability thresholds [K] — Powell & Rubanenko (2020)
_THRESHOLD_STABLE: float = 110.0      # Stable for > 1 Gyr
_THRESHOLD_MARGINAL: float = 115.0    # Stable for ~100 Myr

# Stability class constants
STABILITY_STABLE: int = 0      # T < 110 K
STABILITY_MARGINAL: int = 1    # 110 K ≤ T < 115 K
STABILITY_UNSTABLE: int = 2    # T ≥ 115 K


# ===================================================================
# SUBLIMATION RATE — Numba JIT
# ===================================================================


@njit(cache=True)
def sublimation_rate_log10(T: float) -> float:
    """Compute log₁₀ of the water ice sublimation rate.

    Uses the simplified Langmuir equation:

        log₁₀(dm/dt [kg/m²/s]) = A - B/T - 0.5·log₁₀(T)

    Parameters
    ----------
    T : float
        Surface temperature [K]. Must be > 0.

    Returns
    -------
    float
        log₁₀ of sublimation rate [kg/m²/s].
        Returns -100.0 (effectively zero) for T < 1 K.

    """
    if T < 1.0:
        return -100.0
    return _LANGMUIR_A - _LANGMUIR_B / T - 0.5 * np.log10(T)


@njit(cache=True)
def sublimation_rate(T: float) -> float:
    """Compute the water ice sublimation rate [kg/m²/s].

    Parameters
    ----------
    T : float
        Surface temperature [K].

    Returns
    -------
    float
        Sublimation mass flux [kg/m²/s].
        Returns 0.0 for T < 1 K.

    """
    if T < 1.0:
        return 0.0
    log_rate = sublimation_rate_log10(T)
    if log_rate < -40.0:
        return 0.0
    return 10.0 ** log_rate


@njit(cache=True)
def sublimation_rate_hertz_knudsen(T: float) -> float:
    """Compute sublimation rate using the full Hertz-Knudsen equation.

    dm/dt = P_vap(T) / √(2π · m_H2O · k_B · T)

    where P_vap is computed from the Clausius-Clapeyron equation:

        ln(P_vap) = -L_sub / (k_B · T) + const

    Parameters
    ----------
    T : float
        Surface temperature [K].

    Returns
    -------
    float
        Sublimation mass flux [kg/m²/s].

    """
    if T < 1.0:
        return 0.0

    # Vapor pressure of water ice (Murphy & Koop, 2005)
    # Valid for 110 K < T < 273 K
    # ln(P_vap) = 9.550426 - 5723.265/T + 3.53068·ln(T) - 0.00728332·T
    ln_P = 9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T

    # Guard against overflow
    if ln_P < -200.0:
        return 0.0

    P_vap = np.exp(ln_P)

    # Hertz-Knudsen: dm/dt = P / sqrt(2π·m·k_B·T)
    denominator = np.sqrt(2.0 * np.pi * _M_H2O * _K_B * T)
    if denominator < 1e-50:
        return 0.0

    return P_vap / denominator


# ===================================================================
# ICE STABILITY CLASSIFICATION — Numba JIT
# ===================================================================


@njit(cache=True)
def ice_stability_class(T: float) -> int:
    """Classify cold trap stability based on temperature.

    Parameters
    ----------
    T : float
        Surface temperature [K].

    Returns
    -------
    int
        Stability class:
        - 0 = STABLE (T < 110 K, retains ice for > 1 Gyr)
        - 1 = MARGINAL (110 K ≤ T < 115 K, retains for ~100 Myr)
        - 2 = UNSTABLE (T ≥ 115 K, ice sublimates rapidly)

    """
    if T < _THRESHOLD_STABLE:
        return STABILITY_STABLE
    elif T < _THRESHOLD_MARGINAL:
        return STABILITY_MARGINAL
    else:
        return STABILITY_UNSTABLE


@njit(cache=True)
def ice_retention_timescale(T: float, ice_thickness_m: float = 1.0) -> float:
    """Estimate how long a given ice deposit survives at temperature T.

    Parameters
    ----------
    T : float
        Surface temperature [K].
    ice_thickness_m : float
        Ice layer thickness [m]. Default: 1 m.

    Returns
    -------
    float
        Retention timescale [years]. Returns inf for T < 40 K.

    """
    if T < 40.0:
        return np.inf

    rate = sublimation_rate(T)
    if rate < 1e-50:
        return np.inf

    # Ice density ~917 kg/m³
    ice_density = 917.0
    total_mass = ice_density * ice_thickness_m  # kg/m²
    time_s = total_mass / rate
    time_yr = time_s / (365.25 * 24.0 * 3600.0)
    return time_yr


# ===================================================================
# COLD TRAP MAP — Vectorized
# ===================================================================


@njit(cache=True, parallel=True)
def compute_cold_trap_map(
    surface_temps: np.ndarray,
) -> np.ndarray:
    """Classify each face as stable, marginal, or unstable cold trap.

    Parameters
    ----------
    surface_temps : np.ndarray
        Surface temperature per face [K]. Shape: (N,).

    Returns
    -------
    stability : np.ndarray
        Stability class per face. Shape: (N,).
        0 = STABLE, 1 = MARGINAL, 2 = UNSTABLE.

    """
    N = len(surface_temps)
    stability = np.empty(N, dtype=np.int64)

    for i in prange(N):
        stability[i] = ice_stability_class(surface_temps[i])

    return stability


@njit(cache=True, parallel=True)
def compute_sublimation_map(
    surface_temps: np.ndarray,
) -> np.ndarray:
    """Compute sublimation rate for each face.

    Parameters
    ----------
    surface_temps : np.ndarray
        Surface temperature per face [K]. Shape: (N,).

    Returns
    -------
    rates : np.ndarray
        Sublimation rate per face [kg/m²/s]. Shape: (N,).

    """
    N = len(surface_temps)
    rates = np.empty(N, dtype=np.float64)

    for i in prange(N):
        rates[i] = sublimation_rate(surface_temps[i])

    return rates
