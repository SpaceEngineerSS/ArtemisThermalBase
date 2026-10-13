"""Water-ice sublimation and cold-trap classification.

Sublimation is the Hertz-Knudsen mass flux into vacuum with unit
sticking coefficient,

    E(T) = P_vap(T) · sqrt(m / (2π k_B T))        [kg m⁻² s⁻¹]

with the vapour pressure of hexagonal ice from Murphy & Koop (2005, eq. 7).
Their fit is stated for T > 110 K; below that we extrapolate, which is the
usual practice for lunar cold traps and errs on the side of larger loss.

Reference values this module reproduces (tests pin them):

=========  ====================  ============================
T [K]      E [kg m⁻² s⁻¹]        1 m of ice lasts
=========  ====================  ============================
100        ~2 × 10⁻¹⁷            ~10¹² yr
110        ~4.5 × 10⁻¹⁵          ~6 × 10⁹ yr
200        ~2 × 10⁻⁴             ~50 days
=========  ====================  ============================

Cold-trap classes use the *maximum* surface temperature a face reaches
over the run: below the stable threshold (110 K by default) ice survives
for billions of years, between stable and marginal (115 K) for roughly
10⁸–10⁹ years, above that it does not survive geologically.

References
----------
- Murphy, D.M. & Koop, T. (2005). Review of the vapour pressures of ice and
  supercooled water for atmospheric applications. QJRMS 131, 1539-1565.
- Zhang, J.A. & Paige, D.A. (2009). Cold-trapped organic compounds at the
  poles of the Moon and Mercury. GRL 36, L16203.
- Hayne, P.O. et al. (2015). Evidence for exposed water ice in the Moon's
  south polar regions from LRO ultraviolet albedo and temperature
  measurements. Icarus 255, 58-69.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    prange = range
else:
    from numba import prange

_K_B = 1.380649e-23          # J K⁻¹ (exact, SI 2019)
_M_H2O = 2.99151e-26         # kg, 18.01528 u
_ICE_DENSITY = 917.0         # kg m⁻³
_SECONDS_PER_YEAR = 365.25 * 86400.0

STABILITY_STABLE = 0
STABILITY_MARGINAL = 1
STABILITY_UNSTABLE = 2

DEFAULT_STABLE_K = 110.0
DEFAULT_MARGINAL_K = 115.0


@njit(cache=True)
def vapor_pressure_ice(T: float) -> float:
    """Saturation vapour pressure over hexagonal ice [Pa] (Murphy & Koop 2005)."""
    if T < 1.0:
        return 0.0
    ln_p = 9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T
    if ln_p < -700.0:
        return 0.0
    return np.exp(ln_p)


@njit(cache=True)
def sublimation_rate(T: float) -> float:
    """Hertz-Knudsen sublimation mass flux into vacuum [kg m⁻² s⁻¹]."""
    if T < 1.0:
        return 0.0
    return vapor_pressure_ice(T) * np.sqrt(_M_H2O / (2.0 * np.pi * _K_B * T))


@njit(cache=True)
def ice_retention_timescale(T: float, ice_thickness_m: float = 1.0) -> float:
    """Years for a pure ice layer of the given thickness to sublimate at T."""
    rate = sublimation_rate(T)
    if rate <= 0.0:
        return np.inf
    return _ICE_DENSITY * ice_thickness_m / rate / _SECONDS_PER_YEAR


@njit(cache=True)
def ice_stability_class(
    T: float,
    stable_k: float = DEFAULT_STABLE_K,
    marginal_k: float = DEFAULT_MARGINAL_K,
) -> int:
    """Classify a temperature as stable (0), marginal (1) or unstable (2)."""
    if T < stable_k:
        return STABILITY_STABLE
    if T < marginal_k:
        return STABILITY_MARGINAL
    return STABILITY_UNSTABLE


@njit(cache=True, parallel=True)
def compute_cold_trap_map(
    max_surface_temps: np.ndarray,
    stable_k: float = DEFAULT_STABLE_K,
    marginal_k: float = DEFAULT_MARGINAL_K,
) -> np.ndarray:
    """Stability class per face from its maximum surface temperature."""
    n = max_surface_temps.shape[0]
    stability = np.empty(n, dtype=np.int8)
    for i in prange(n):
        stability[i] = ice_stability_class(max_surface_temps[i], stable_k, marginal_k)
    return stability


@njit(cache=True, parallel=True)
def compute_sublimation_map(surface_temps: np.ndarray) -> np.ndarray:
    """Sublimation flux per face [kg m⁻² s⁻¹]."""
    n = surface_temps.shape[0]
    rates = np.empty(n, dtype=np.float64)
    for i in prange(n):
        rates[i] = sublimation_rate(surface_temps[i])
    return rates
