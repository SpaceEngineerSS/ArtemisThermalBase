"""Fail-closed readiness audit for an Artemis research run."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core_engine.constants import load_config
from data_ingestion.lola_loader import LOLALoader
from data_ingestion.spice_ephemeris import SpiceSolarEphemeris


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research_shackleton.yaml")
    parser.add_argument("--dem", required=True)
    parser.add_argument("--report", default="output/research_readiness.json")
    args = parser.parse_args()

    config = load_config(args.config)
    dem = LOLALoader(require_provenance=True).load_dem(args.dem)
    relief_m = float(np.ptp(dem.elevation))
    if relief_m < 1000.0:
        raise ValueError(
            f"Shackleton DEM relief is only {relief_m:.1f} m; likely wrong window or units."
        )
    ephemeris = SpiceSolarEphemeris(
        config.ephemeris.kernel_files,
        frame=config.ephemeris.frame,
        aberration_correction=config.ephemeris.aberration_correction,
    )
    state = ephemeris.get_sun_state(
        datetime.fromisoformat(config.time_range.start.replace("Z", "+00:00")),
        config.target.latitude_deg,
        config.target.longitude_deg,
        config.constants.solar_constant,
    )
    if not (1.3e11 < state["distance_m"] < 1.7e11):
        raise ValueError("SPICE Sun-Moon distance is outside the physical audit range.")
    report = {
        "status": "ready_for_research_run",
        "audited_utc": datetime.now(UTC).isoformat(),
        "config": str(args.config),
        "dem": dem.metadata,
        "dem_relief_m": relief_m,
        "ephemeris": ephemeris.provenance,
        "sample_sun_state": state,
        "validation_status": "Diviner comparison pending",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Research readiness passed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
