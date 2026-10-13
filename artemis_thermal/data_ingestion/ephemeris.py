"""Sun position sources.

Every source returns the Moon-to-Sun direction in the lunar body-fixed frame
(+z north pole, +x toward 0°E) together with the Sun distance. Turning that
into the terrain frame is the job of :mod:`artemis_thermal.core_engine.frames`, so the three
sources below cannot disagree about axis conventions.

``SyntheticSun``
    Analytical preview: the sub-solar point moves once around the Moon per
    synodic month at a fixed latitude. No data files needed.
``SolarEphemeris``
    Skyfield with a JPL DE kernel for the Moon→Sun vector (ICRF) and the IAU
    2015 rotation model for the Moon's orientation. Accurate to roughly
    0.01° in orientation; good enough for previews, not for research runs.
``SpiceSolarEphemeris`` (in :mod:`artemis_thermal.data_ingestion.spice_ephemeris`)
    NAIF SPICE with the DE440 lunar PA/ME kernels. The research path.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

References
----------
- Archinal, B.A. et al. (2018). Report of the IAU Working Group on
  Cartographic Coordinates and Rotational Elements: 2015. Celest. Mech.
  Dyn. Astr. 130, 22.
- Park, R.S. et al. (2021). The JPL Planetary and Lunar Ephemerides DE440
  and DE441. AJ 161, 105.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from artemis_thermal.core_engine.frames import synthetic_sun_body

logger = logging.getLogger(__name__)

AU_M = 149_597_870_700.0
_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)


@dataclass(frozen=True)
class SunState:
    """Sun geometry at one instant, in the lunar body-fixed frame."""

    direction_body: np.ndarray
    distance_m: float
    solar_flux: float
    extra: dict[str, Any] = field(default_factory=dict)


class SunSource(Protocol):
    """Anything that can report the Sun's position for a UTC time."""

    def sun_state(self, utc_time: datetime, solar_constant: float) -> SunState:
        """Return the Sun state at ``utc_time``."""
        ...


def _as_utc(utc_time: datetime) -> datetime:
    if utc_time.tzinfo is None:
        return utc_time.replace(tzinfo=UTC)
    return utc_time.astimezone(UTC)


# ---------------------------------------------------------------------------
# Synthetic preview source
# ---------------------------------------------------------------------------


class SyntheticSun:
    """Analytical Sun for runs without ephemeris data.

    Parameters
    ----------
    start_time : datetime
        Reference time at which the Sun is at local noon over the target.
    target_lon_deg : float
        Longitude of the terrain centre.
    subsolar_lat_deg : float
        Fixed sub-solar latitude. Negative values favour the south pole
        (southern summer); the physical range is about ±1.54°.
    synodic_period_s : float
        Length of one lunar day.

    """

    def __init__(
        self,
        start_time: datetime,
        target_lon_deg: float,
        subsolar_lat_deg: float,
        synodic_period_s: float,
    ) -> None:
        self._t0 = _as_utc(start_time)
        self._lon = target_lon_deg
        self._lat = subsolar_lat_deg
        self._period = synodic_period_s

    def sun_state(self, utc_time: datetime, solar_constant: float) -> SunState:
        """Return the analytical Sun state at ``utc_time``."""
        elapsed = (_as_utc(utc_time) - self._t0).total_seconds()
        direction = synthetic_sun_body(elapsed, self._lon, self._lat, self._period)
        return SunState(direction, AU_M, solar_constant, {"model": "synthetic"})


# ---------------------------------------------------------------------------
# IAU 2015 lunar orientation
# ---------------------------------------------------------------------------


def _rot_z(angle_rad: float) -> np.ndarray:
    """Passive (frame) rotation about z."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle_rad: float) -> np.ndarray:
    """Passive (frame) rotation about x."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def iau_moon_rotation(utc_time: datetime) -> np.ndarray:
    """ICRF-to-body rotation of the Moon from the IAU 2015 model.

    Includes the thirteen periodic terms; without them the prime meridian
    is off by up to ~3.5°. Time is treated as TDB ≈ UTC + 69.2 s, which is
    far below the model's own accuracy.
    """
    tdb_offset_s = 69.184
    d = ((_as_utc(utc_time) - _J2000).total_seconds() + tdb_offset_s) / 86400.0
    T = d / 36525.0

    E = np.radians(
        np.array(
            [
                125.045 - 0.0529921 * d,
                250.089 - 0.1059842 * d,
                260.008 + 13.0120009 * d,
                176.625 + 13.3407154 * d,
                357.529 + 0.9856003 * d,
                311.589 + 26.4057084 * d,
                134.963 + 13.0649930 * d,
                276.617 + 0.3287146 * d,
                34.226 + 1.7484877 * d,
                15.134 - 0.1589763 * d,
                119.743 + 0.0036096 * d,
                239.961 + 0.1643573 * d,
                25.053 + 12.9590088 * d,
            ]
        )
    )
    s, c = np.sin(E), np.cos(E)

    alpha0 = (
        269.9949 + 0.0031 * T
        - 3.8787 * s[0] - 0.1204 * s[1] + 0.0700 * s[2] - 0.0172 * s[3]
        + 0.0072 * s[5] - 0.0052 * s[9] + 0.0043 * s[12]
    )
    delta0 = (
        66.5392 + 0.0130 * T
        + 1.5419 * c[0] + 0.0239 * c[1] - 0.0278 * c[2] + 0.0068 * c[3]
        - 0.0029 * c[5] + 0.0009 * c[6] + 0.0008 * c[9] - 0.0009 * c[12]
    )
    W = (
        38.3213 + 13.17635815 * d - 1.4e-12 * d * d
        + 3.5610 * s[0] + 0.1208 * s[1] - 0.0642 * s[2] + 0.0158 * s[3]
        + 0.0252 * s[4] - 0.0066 * s[5] - 0.0047 * s[6] - 0.0046 * s[7]
        + 0.0028 * s[8] + 0.0052 * s[9] + 0.0040 * s[10] + 0.0019 * s[11]
        - 0.0044 * s[12]
    )
    return (
        _rot_z(np.radians(W))
        @ _rot_x(np.radians(90.0 - delta0))
        @ _rot_z(np.radians(90.0 + alpha0))
    )


# ---------------------------------------------------------------------------
# Skyfield source
# ---------------------------------------------------------------------------


class SolarEphemeris:
    """Skyfield-based Sun source (JPL DE kernel + IAU lunar orientation).

    Parameters
    ----------
    kernel_name : str
        JPL ephemeris kernel filename, e.g. ``de421.bsp``.
    data_dir : Path or str
        Directory holding (or receiving) the kernel.

    """

    def __init__(
        self,
        kernel_name: str = "de421.bsp",
        data_dir: Path | str = "data/ephemeris",
    ) -> None:
        self._kernel_name = kernel_name
        self._data_dir = Path(data_dir)
        self._timescale: Any = None
        self._sun: Any = None
        self._moon: Any = None

    def _ensure_loaded(self) -> None:
        if self._moon is not None:
            return
        try:
            from skyfield.api import Loader
        except ImportError as exc:
            raise ImportError(
                "The skyfield ephemeris needs the optional 'ephemeris' extra: "
                'pip install "artemis-thermal-base[ephemeris]"'
            ) from exc

        self._data_dir.mkdir(parents=True, exist_ok=True)
        loader = Loader(str(self._data_dir), verbose=False)
        try:
            kernel = loader(self._kernel_name)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not load or download '{self._kernel_name}' into "
                f"'{self._data_dir}'. Download it from "
                "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/ "
                "and place it there."
            ) from exc
        self._timescale = loader.timescale()
        self._sun = kernel["sun"]
        self._moon = kernel["moon"]
        logger.info("Skyfield kernel loaded: %s", self._kernel_name)

    def sun_state(self, utc_time: datetime, solar_constant: float) -> SunState:
        """Return the Sun state at ``utc_time`` from Skyfield."""
        self._ensure_loaded()
        utc_time = _as_utc(utc_time)
        t = self._timescale.from_datetime(utc_time)
        position_m = np.asarray(
            self._moon.at(t).observe(self._sun).position.m, dtype=np.float64
        )
        distance_m = float(np.linalg.norm(position_m))
        direction_body = iau_moon_rotation(utc_time) @ (position_m / distance_m)
        return SunState(
            direction_body / np.linalg.norm(direction_body),
            distance_m,
            solar_constant * (AU_M / distance_m) ** 2,
            {"model": "skyfield+iau2015", "kernel": self._kernel_name},
        )
