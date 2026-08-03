"""Pure, testable adapter from simulation persistence to dashboard keys."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from simulation.io_manager import load_results


def load_dashboard_data(output_dir: str | Path) -> dict | None:
    """Load canonical simulation output and expose dashboard key names."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    saved = load_results(output_path)
    if "thermal_grid" not in saved:
        return None

    return {
        "surface_temps": saved["thermal_grid"],
        "illumination": saved.get("illumination_grid", np.array([])),
        "face_centroids": saved.get("face_centroids", np.array([])),
        "face_areas": saved.get("face_areas", np.array([])),
        "dem_elevation": saved.get("dem_grid", np.array([])),
        "probe_temps": saved.get("probe_temps", {}),
        "sun_elevations": saved.get("sun_elevations", np.array([])),
        "metadata": saved.get("metadata", {}),
    }

