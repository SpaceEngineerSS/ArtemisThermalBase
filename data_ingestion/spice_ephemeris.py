"""NAIF SPICE solar geometry in the lunar Mean Earth body-fixed frame."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from data_ingestion.ephemeris import AU_M, SunState, _as_utc
from data_ingestion.provenance import sha256_file


class SpiceSolarEphemeris:
    """Compute Moon-to-Sun geometry using versioned local NAIF kernels.

    Unlike the preview sources, this class never downloads data and never
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
                "The SPICE ephemeris needs the optional 'ephemeris' extra: "
                'pip install "artemis-thermal-base[ephemeris]"'
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
                {"path": path.as_posix(), "sha256": sha256_file(path)}
                for path in self._kernel_files
            ],
        }

    def sun_state(self, utc_time: datetime, solar_constant: float) -> SunState:
        """Return the Sun direction in ``frame`` and its distance."""
        utc_text = _as_utc(utc_time).isoformat().replace("+00:00", "Z")
        et = self._spice.str2et(utc_text)
        position_km, light_time_s = self._spice.spkpos(
            "SUN", et, self._frame, self._abcorr, "MOON"
        )
        position_m = np.asarray(position_km, dtype=np.float64) * 1000.0
        distance_m = float(np.linalg.norm(position_m))
        return SunState(
            position_m / distance_m,
            distance_m,
            solar_constant * (AU_M / distance_m) ** 2,
            {"one_way_light_time_s": float(light_time_s), "frame": self._frame},
        )
