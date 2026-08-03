"""NAIF SPICE solar geometry in the lunar Mean Earth body-fixed frame."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from data_ingestion.ephemeris import _body_to_local_rotation
from data_ingestion.provenance import sha256_file

AU_M = 149_597_870_700.0


class SpiceSolarEphemeris:
    """Compute Moon-to-Sun geometry using versioned local NAIF kernels.

    Unlike the preview ephemeris, this class never downloads data and never
    falls back to an analytical rotation. Missing kernels are fatal so a
    research run cannot silently change its geometry model.
    """

    def __init__(
        self,
        kernel_files: Sequence[str | Path],
        *,
        frame: str = "MOON_ME",
        aberration_correction: str = "LT+S",
    ) -> None:
        try:
            import spiceypy as spice
        except ImportError as exc:
            raise ImportError(
                "spiceypy is required for the NAIF research ephemeris. "
                "Install the project with its current dependencies."
            ) from exc

        self._spice: Any = spice
        self._kernel_files = tuple(Path(item) for item in kernel_files)
        self._frame = frame
        self._abcorr = aberration_correction
        missing = [str(path) for path in self._kernel_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required NAIF kernels are missing:\n  " + "\n  ".join(missing)
                + "\nRun: python tools/fetch_research_data.py --group ephemeris"
            )
        for path in self._kernel_files:
            spice.furnsh(str(path.resolve()))

    @property
    def provenance(self) -> dict[str, Any]:
        """Return kernel names, hashes, frame, and correction settings."""
        return {
            "provider": "NAIF/JPL SPICE",
            "frame": self._frame,
            "aberration_correction": self._abcorr,
            "kernels": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in self._kernel_files
            ],
        }

    @staticmethod
    def _utc_text(utc_time: datetime) -> str:
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=UTC)
        return utc_time.astimezone(UTC).isoformat().replace("+00:00", "Z")

    def get_sun_state(
        self,
        utc_time: datetime,
        lat_deg: float = -89.67,
        lon_deg: float = 129.78,
        S_0: float = 1361.0,
    ) -> dict[str, Any]:
        """Return local Sun direction, elevation, distance, and irradiance."""
        et = self._spice.str2et(self._utc_text(utc_time))
        position_km, light_time_s = self._spice.spkpos(
            "SUN", et, self._frame, self._abcorr, "MOON"
        )
        position_m = np.asarray(position_km, dtype=np.float64) * 1000.0
        distance_m = float(np.linalg.norm(position_m))
        sun_body = position_m / distance_m
        local_rotation = _body_to_local_rotation(
            np.radians(lat_deg), np.radians(lon_deg)
        )
        sun_local = local_rotation @ sun_body
        sun_local /= np.linalg.norm(sun_local)
        elevation_deg = float(np.degrees(np.arcsin(np.clip(sun_local[2], -1.0, 1.0))))
        return {
            "direction": sun_local,
            "elevation_deg": elevation_deg,
            "distance_m": distance_m,
            "solar_flux": S_0 * (AU_M / distance_m) ** 2,
            "one_way_light_time_s": float(light_time_s),
            "frame": self._frame,
        }
