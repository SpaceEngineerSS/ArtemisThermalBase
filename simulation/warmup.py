"""Compile the Numba kernels on a tiny problem so later runs start at once.

Numba writes compiled machine code to ``__pycache__`` (``cache=True``), so
this only costs time the first time after an install or upgrade.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np


def warm_up() -> float:
    """Run a 4×4-cell problem through every hot path; return seconds taken."""
    from core_engine.constants import load_config
    from simulation.runner import SimulationRunner

    started = time.perf_counter()
    config = load_config()
    tiny = replace(
        config,
        synthetic_dem=replace(
            config.synthetic_dem, radius_m=300.0, depth_m=60.0, rim_height_m=10.0,
            grid_resolution_m=100.0, domain_padding_m=100.0,
        ),
        view_factors=replace(config.view_factors, enabled=True, num_rays=8),
    )
    runner = SimulationRunner(tiny)
    for point_source in (True, False):
        runner.run(
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            duration_hours=0.5,
            dt_s=900.0,
            output_interval_s=900.0,
            point_source_mode=point_source,
            save_data=False,
        )
    from thermal_solver.volatiles import compute_cold_trap_map, sublimation_rate

    compute_cold_trap_map(np.array([100.0, 120.0]), 110.0, 115.0)
    sublimation_rate(110.0)
    return time.perf_counter() - started
