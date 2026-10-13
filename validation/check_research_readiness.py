"""Fail-closed readiness audit for a research run.

    python -m validation.check_research_readiness --dem data/shackleton_40m.tif

Loads the research configuration, verifies the DEM provenance and relief,
loads the SPICE kernels, evaluates the Sun at the configured start time on
the DEM's own centre, and writes a JSON report. Any failure exits non-zero.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core_engine.constants import load_config
from core_engine.frames import frame_for_dem
from data_ingestion.lola_loader import LOLALoader
from data_ingestion.spice_ephemeris import SpiceSolarEphemeris
from simulation.io_manager import _jsonable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="research_shackleton")
    parser.add_argument("--dem", required=True)
    parser.add_argument("--report", default="output/research_readiness.json")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    dem = LOLALoader(require_provenance=True).load_dem(args.dem)
    relief_m = float(np.ptp(dem.elevation))
    if relief_m < 1000.0:
        raise ValueError(
            f"DEM relief is only {relief_m:.1f} m; likely the wrong window or units."
        )
    ephemeris = SpiceSolarEphemeris(
        config.ephemeris.kernel_files,
        frame=config.ephemeris.frame,
        aberration_correction=config.ephemeris.aberration_correction,
    )
    start = datetime.fromisoformat(config.time_range.start.replace("Z", "+00:00"))
    sun = ephemeris.sun_state(start, config.constants.solar_constant)
    if not (1.3e11 < sun.distance_m < 1.7e11):
        raise ValueError("SPICE Sun-Moon distance is outside the physical range.")
    frame = frame_for_dem(
        dem.metadata, config.target.latitude_deg, config.target.longitude_deg,
        config.lunar.radius_m,
    )
    sun_mesh = frame.from_body(sun.direction_body)
    report = {
        "status": "ready_for_research_run",
        "audited_utc": datetime.now(UTC).isoformat(),
        "config": str(config.source_path),
        "dem": {k: v for k, v in dem.metadata.items() if k != "crs_wkt"},
        "dem_relief_m": relief_m,
        "mesh_frame": frame.describe(),
        "ephemeris": ephemeris.provenance,
        "sample_sun": {
            "utc": start.isoformat(),
            "elevation_deg": float(np.degrees(np.arcsin(np.clip(sun_mesh[2], -1, 1)))),
            "distance_m": sun.distance_m,
            "solar_flux_w_m2": sun.solar_flux,
        },
        "validation_status": "Diviner comparison pending",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8")
    print(f"Research readiness passed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
