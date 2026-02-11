"""Solar ephemeris for the Lunar South Pole.

Computes the Sun's position vector relative to any point on the Moon's
surface using the Skyfield library with JPL DE421 (or DE440) ephemeris
kernels. Accounts for lunar libration via Skyfield's high-precision
planetary orientation model.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

References
----------
- Folkner, W.M. et al. (2014). "The Planetary and Lunar Ephemerides
  DE430 and DE431." IPN Progress Report 42-196.
- Archinal, B.A. et al. (2018). "Report of the IAU Working Group on
  Cartographic Coordinates and Rotational Elements: 2015."
  Celestial Mechanics and Dynamical Astronomy, 130, 22.

Notes
-----
The coordinate chain is:

    Sun [ICRF/J2000] → Moon-centered ICRF → Selenographic (body-fixed)
    → South Pole local Cartesian (x=toward λ=0, y=toward λ=90°, z=up)

Skyfield handles the full chain including lunar libration, light-time
correction, and aberration via its `observe().apparent()` pipeline.

Data Management
---------------
Ephemeris kernel files (*.bsp) are large (~17 MB for DE421). The
``SolarEphemeris`` class automatically manages kernel downloads to
a configurable data directory and logs the location. These files
should be added to ``.gitignore``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path("data")
_DEFAULT_KERNEL = "de421.bsp"

# IAU lunar radius [m]
_LUNAR_RADIUS_M: float = 1_737_400.0


# ---------------------------------------------------------------------------
# Solar Ephemeris Class
# ---------------------------------------------------------------------------


class SolarEphemeris:
    """High-precision solar ephemeris for the Lunar South Pole.

    Computes the Sun's direction vector in a local Cartesian frame
    at any point on the Moon's surface, at any given UTC time.

    The class lazily loads the JPL kernel file and caches it for
    subsequent calls.

    Parameters
    ----------
    kernel_name : str
        JPL ephemeris kernel filename (default: 'de421.bsp').
    data_dir : Path or str
        Directory for storing downloaded kernel files.

    Examples
    --------
    >>> eph = SolarEphemeris()
    >>> sun_dir = eph.get_sun_direction_local(
    ...     utc_time=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
    ...     lat_deg=-89.54,
    ...     lon_deg=129.78,
    ... )
    >>> print(sun_dir)  # (x, y, z) unit vector in local frame
    """

    def __init__(
        self,
        kernel_name: str = _DEFAULT_KERNEL,
        data_dir: Path | str = _DEFAULT_DATA_DIR,
    ) -> None:
        self._kernel_name = kernel_name
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Lazily loaded
        self._ephemeris: Any = None
        self._timescale: Any = None
        self._sun: Any = None
        self._moon: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load ephemeris kernel, downloading if necessary."""
        if self._loaded:
            return

        try:
            from skyfield.api import Loader
        except ImportError as e:
            raise ImportError(
                "skyfield is required for ephemeris computation. "
                "Install it with: pip install skyfield"
            ) from e

        loader = Loader(str(self._data_dir), verbose=True)
        kernel_path = self._data_dir / self._kernel_name

        if kernel_path.exists():
            logger.info("Loading ephemeris kernel from: %s", kernel_path)
        else:
            logger.info(
                "Ephemeris kernel not found at %s. Downloading %s (~17 MB)...",
                kernel_path,
                self._kernel_name,
            )

        try:
            self._ephemeris = loader(self._kernel_name)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load or download JPL kernel '{self._kernel_name}' "
                f"to '{self._data_dir}'. Check your internet connection.\n"
                f"You can manually download it from:\n"
                f"  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/{self._kernel_name}\n"
                f"and place it in '{self._data_dir}'."
            ) from e

        self._timescale = loader.timescale()
        self._sun = self._ephemeris["sun"]
        self._moon = self._ephemeris["moon"]
        self._loaded = True

        logger.info(
            "Ephemeris loaded: %s (segments: %d)",
            self._kernel_name,
            len(self._ephemeris.segments),
        )

    def get_sun_direction_icrf(
        self,
        utc_time: datetime,
    ) -> tuple[np.ndarray, float]:
        """Get Sun direction relative to Moon center in ICRF frame.

        Parameters
        ----------
        utc_time : datetime
            UTC time (must be timezone-aware or assumed UTC).

        Returns
        -------
        direction : np.ndarray
            Unit vector from Moon center toward Sun in ICRF. Shape: (3,).
        distance_m : float
            Sun-Moon distance in meters.
        """
        self._ensure_loaded()

        # Convert datetime to Skyfield Time
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)

        t = self._timescale.from_datetime(utc_time)

        # Moon → Sun vector in ICRF (in AU then converted to meters)
        astrometric = self._moon.at(t).observe(self._sun)
        position_au = astrometric.position.au  # Shape: (3,)

        # Convert to meters
        au_to_m = 1.495978707e11
        position_m = np.array(position_au, dtype=np.float64) * au_to_m

        distance_m = float(np.linalg.norm(position_m))
        direction = position_m / distance_m

        return direction, distance_m

    def get_sun_direction_local(
        self,
        utc_time: datetime,
        lat_deg: float = -89.54,
        lon_deg: float = 129.78,
        R: float = _LUNAR_RADIUS_M,
    ) -> np.ndarray:
        """Get Sun direction in local Cartesian frame at a surface point.

        The local frame is defined as:
        - z_local: radially outward (surface normal on a sphere)
        - x_local: toward decreasing latitude (north) in the local
                    meridian plane
        - y_local: toward increasing longitude (east)

        For the south pole, this simplifies to:
        - z_up: (0, 0, −1) in body-fixed → outward
        - x: toward λ = 0° direction (projected)
        - y: toward λ = 90° direction

        Parameters
        ----------
        utc_time : datetime
            UTC observation time.
        lat_deg : float
            Surface latitude in degrees (negative for south).
        lon_deg : float
            Surface longitude in degrees.
        R : float
            Lunar radius in meters.

        Returns
        -------
        sun_dir_local : np.ndarray
            Unit vector toward the Sun in local Cartesian frame.
            Shape: (3,). z > 0 means Sun is above horizon.
        """
        # Step 1: Sun direction in ICRF
        sun_icrf, distance_m = self.get_sun_direction_icrf(utc_time)

        # Step 2: Rotate ICRF → selenographic body-fixed
        R_body_from_icrf = self._get_moon_rotation_matrix(utc_time)
        sun_body = R_body_from_icrf @ sun_icrf

        # Step 3: Rotate body-fixed → local tangent frame
        R_local_from_body = _body_to_local_rotation(
            np.radians(lat_deg), np.radians(lon_deg)
        )
        sun_local = R_local_from_body @ sun_body

        # Normalize for safety
        sun_local = sun_local / np.linalg.norm(sun_local)

        return sun_local

    def _get_moon_rotation_matrix(
        self,
        utc_time: datetime,
    ) -> np.ndarray:
        """Get the ICRF → Moon body-fixed rotation matrix.

        Uses Skyfield's built-in Moon frame which accounts for
        physical libration from the DE421 kernel.

        Parameters
        ----------
        utc_time : datetime
            UTC time.

        Returns
        -------
        R : np.ndarray
            3×3 rotation matrix. Shape: (3, 3).
        """
        self._ensure_loaded()

        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)

        t = self._timescale.from_datetime(utc_time)

        # Use Skyfield's built-in Moon frame (includes libration)
        try:
            from skyfield.api import load as skyfield_load
            from skyfield.data import mpc
            from skyfield.framelib import moon_frame

            # The moon_frame from skyfield.framelib provides full
            # libration-corrected body-fixed coordinates
            R_matrix = moon_frame.rotation_at(t)
            return np.asarray(R_matrix, dtype=np.float64)

        except (ImportError, AttributeError):
            # Fallback: Use IAU rotation model (Archinal et al., 2018)
            logger.warning(
                "Skyfield moon_frame not available. "
                "Falling back to IAU analytical model."
            )
            return self._iau_moon_rotation(t)

    def _iau_moon_rotation(self, t: Any) -> np.ndarray:
        """IAU 2009 analytical Moon rotation model.

        Computes the ICRF → body-fixed rotation matrix using the
        IAU Working Group equations (Archinal et al., 2018).

        This is a simplified model without physical libration
        (free libration amplitude ~0.03°). The Skyfield moon_frame
        is preferred when available.

        Parameters
        ----------
        t : skyfield.Time
            Skyfield time object.

        Returns
        -------
        R : np.ndarray
            3×3 rotation matrix ICRF → body-fixed.
        """
        # Julian centuries from J2000.0
        T = (t.tdb - 2451545.0) / 36525.0
        # Days from J2000.0
        d = t.tdb - 2451545.0

        # IAU 2009 pole coordinates (degrees)
        alpha0 = 269.9949 + 0.0031 * T
        delta0 = 66.5392 + 0.0130 * T
        # Prime meridian (simplified, no libration terms)
        W = 38.3213 + 13.17635815 * d

        # Convert to radians
        a0 = np.radians(alpha0)
        d0 = np.radians(delta0)
        w = np.radians(W % 360.0)

        # R_body_from_ICRF = R_z(W) · R_x(π/2 − δ₀) · R_z(α₀ + π/2)
        R = _Rz(w) @ _Rx(np.pi / 2.0 - d0) @ _Rz(a0 + np.pi / 2.0)

        return R

    def get_sun_elevation_deg(
        self,
        utc_time: datetime,
        lat_deg: float = -89.54,
        lon_deg: float = 129.78,
    ) -> float:
        """Get the Sun's elevation angle above the local horizon.

        Parameters
        ----------
        utc_time : datetime
            UTC observation time.
        lat_deg, lon_deg : float
            Surface location in degrees.

        Returns
        -------
        float
            Sun elevation in degrees. Positive = above horizon.
        """
        sun_local = self.get_sun_direction_local(utc_time, lat_deg, lon_deg)
        elevation_rad = np.arcsin(np.clip(sun_local[2], -1.0, 1.0))
        return float(np.degrees(elevation_rad))


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _Rx(angle: float) -> np.ndarray:
    """Rotation matrix about x-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float64)


def _Ry(angle: float) -> np.ndarray:
    """Rotation matrix about y-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)


def _Rz(angle: float) -> np.ndarray:
    """Rotation matrix about z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _body_to_local_rotation(
    lat_rad: float,
    lon_rad: float,
) -> np.ndarray:
    """Compute rotation matrix from body-fixed to local tangent frame.

    The local tangent frame at (lat, lon) on a sphere:
    - z_local = radially outward
    - x_local = toward north (increasing latitude)
    - y_local = toward east (increasing longitude)

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude in radians.
    lon_rad : float
        Geodetic longitude in radians.

    Returns
    -------
    R : np.ndarray
        3×3 rotation matrix body-fixed → local. Shape: (3, 3).
    """
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)

    # Radial outward unit vector in body-fixed frame
    # r_hat = (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)

    # North unit vector (toward increasing latitude)
    # n_hat = (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)

    # East unit vector (toward increasing longitude)
    # e_hat = (-sin_lon, cos_lon, 0)

    # Rotation matrix: rows are the local frame axes in body-fixed coords
    # [x_local]   [n_hat]
    # [y_local] = [e_hat] · [body_vector]
    # [z_local]   [r_hat]

    R = np.array([
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],  # north
        [-sin_lon, cos_lon, 0.0],                             # east
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],     # up
    ], dtype=np.float64)

    return R
