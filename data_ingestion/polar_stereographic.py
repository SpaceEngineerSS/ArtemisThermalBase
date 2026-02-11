"""South Polar Stereographic Projection for the Lunar South Pole.

Implements the standard polar stereographic projection equations
(Snyder, 1987; Mazarico et al., 2011) for converting between geodetic
coordinates (latitude, longitude) and Cartesian (x, y) meters on a
tangent plane at the south pole.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

References
----------
- Snyder, J.P. (1987). "Map Projections — A Working Manual."
  U.S. Geological Survey Professional Paper 1395.
- Mazarico, E. et al. (2011). "Illumination conditions of the lunar
  polar regions using LOLA topography." Icarus, 211, 1066–1081.

Notes
-----
Convention: The projection is centered on the **south pole** (φ = −90°).

- Forward: (lat°, lon°) → (x, y) meters
- Inverse: (x, y) meters → (lat°, lon°)

Internally, computations use *absolute latitude* (φ_abs > 0 near the
pole) to keep the formula sign-clean:

    ρ = 2R · tan(π/4 − φ_abs/2)
    x = ρ · sin(λ)
    y = −ρ · cos(λ)

At the south pole (φ_abs = 90°), tan(0) = 0 → (x, y) = (0, 0).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LUNAR_RADIUS_M: float = 1_737_400.0  # IAU mean lunar radius [m]


# ---------------------------------------------------------------------------
# Forward Projection: (lat, lon) → (x, y)
# ---------------------------------------------------------------------------


def forward(
    lat_deg: float,
    lon_deg: float,
    R: float = _LUNAR_RADIUS_M,
) -> tuple[float, float]:
    """Convert geodetic (lat, lon) to south polar stereographic (x, y).

    Parameters
    ----------
    lat_deg : float
        Geodetic latitude in degrees. Negative for southern hemisphere.
    lon_deg : float
        Geodetic longitude in degrees [-180, 360).
    R : float
        Body radius in meters. Default: IAU lunar radius.

    Returns
    -------
    x, y : tuple[float, float]
        Cartesian coordinates in meters on the tangent plane at the
        south pole.
    """
    phi_abs = np.radians(np.abs(lat_deg))
    lam = np.radians(lon_deg)

    rho = 2.0 * R * np.tan(np.pi / 4.0 - phi_abs / 2.0)

    x = float(rho * np.sin(lam))
    y = float(-rho * np.cos(lam))

    return x, y


def forward_batch(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    R: float = _LUNAR_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized forward projection for arrays of coordinates.

    Parameters
    ----------
    lat_deg : np.ndarray
        Latitudes in degrees. Shape: (N,).
    lon_deg : np.ndarray
        Longitudes in degrees. Shape: (N,).
    R : float
        Body radius in meters.

    Returns
    -------
    x, y : tuple[np.ndarray, np.ndarray]
        Cartesian coordinates in meters. Shape: (N,) each.
    """
    phi_abs = np.radians(np.abs(lat_deg))
    lam = np.radians(lon_deg)

    rho = 2.0 * R * np.tan(np.pi / 4.0 - phi_abs / 2.0)

    x = rho * np.sin(lam)
    y = -rho * np.cos(lam)

    return x, y


# ---------------------------------------------------------------------------
# Inverse Projection: (x, y) → (lat, lon)
# ---------------------------------------------------------------------------


def inverse(
    x: float,
    y: float,
    R: float = _LUNAR_RADIUS_M,
) -> tuple[float, float]:
    """Convert south polar stereographic (x, y) to geodetic (lat, lon).

    Parameters
    ----------
    x, y : float
        Cartesian coordinates in meters on the tangent plane.
    R : float
        Body radius in meters.

    Returns
    -------
    lat_deg, lon_deg : tuple[float, float]
        Geodetic latitude (negative for south) and longitude in degrees.
    """
    rho = np.sqrt(x * x + y * y)

    # At the pole, rho = 0 → longitude is undefined; return 0 by convention
    if rho < 1e-12:
        return -90.0, 0.0

    phi_abs = np.pi / 2.0 - 2.0 * np.arctan(rho / (2.0 * R))
    lat_deg = float(-np.degrees(phi_abs))  # South pole → negative latitude

    lon_deg = float(np.degrees(np.arctan2(x, -y)))

    return lat_deg, lon_deg


def inverse_batch(
    x: np.ndarray,
    y: np.ndarray,
    R: float = _LUNAR_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized inverse projection for arrays of coordinates.

    Parameters
    ----------
    x, y : np.ndarray
        Cartesian coordinates in meters. Shape: (N,).
    R : float
        Body radius in meters.

    Returns
    -------
    lat_deg, lon_deg : tuple[np.ndarray, np.ndarray]
        Geodetic coordinates in degrees. Shape: (N,) each.
    """
    rho = np.sqrt(x * x + y * y)

    phi_abs = np.pi / 2.0 - 2.0 * np.arctan(rho / (2.0 * R))
    lat_deg = -np.degrees(phi_abs)  # South pole → negative

    # Handle rho ≈ 0 (pole) → longitude undefined
    lon_deg = np.degrees(np.arctan2(x, -y))

    # At the exact pole, set lon = 0 by convention
    pole_mask = rho < 1e-12
    lon_deg[pole_mask] = 0.0

    return lat_deg, lon_deg


# ---------------------------------------------------------------------------
# Convenience: Shackleton Crater Center
# ---------------------------------------------------------------------------


def get_shackleton_center(
    R: float = _LUNAR_RADIUS_M,
) -> dict[str, Any]:
    """Return Shackleton Crater center in both coordinate systems.

    Shackleton Crater: (−89.54°, 129.78°) — Zuber et al. (2012).

    Parameters
    ----------
    R : float
        Lunar radius in meters.

    Returns
    -------
    dict
        Keys: 'lat_deg', 'lon_deg', 'x_m', 'y_m', 'R_m'.
    """
    lat = -89.54
    lon = 129.78

    x, y = forward(lat, lon, R)

    logger.info(
        "Shackleton center: (%.2f°, %.2f°) → (%.2f m, %.2f m)",
        lat, lon, x, y,
    )

    return {
        "lat_deg": lat,
        "lon_deg": lon,
        "x_m": x,
        "y_m": y,
        "R_m": R,
    }
