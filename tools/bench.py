"""Time the three expensive parts of a run at three grid sizes.

    python tools/bench.py                 # prints a Markdown table
    python tools/bench.py --json out.json # also writes the numbers

Each case is one simulated day of the synthetic Shackleton bowl: 16
solar-disk samples, dt 300 s, columns started at their equilibrium
temperature as in a real run. A tiny warm-up case runs first so Numba
compilation does not count. Peak memory is the process high-water mark
(needs psutil; otherwise the column is empty). Run it on an idle machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _peak_mb() -> float:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return float(getattr(info, "peak_wset", 0) or info.rss) / 1e6
    except ImportError:
        return float("nan")


def bench_case(resolution_m: float, steps: int = 288) -> dict[str, float]:
    """Time BVH build, shadows and conduction for one grid resolution."""
    from artemis_thermal.core_engine.constants import load_config
    from artemis_thermal.core_engine.frames import enu_frame, synthetic_sun_body
    from artemis_thermal.core_engine.illumination import IlluminationEngine
    from artemis_thermal.core_engine.mesh import dem_to_mesh
    from artemis_thermal.data_ingestion.ephemeris import SyntheticSun
    from artemis_thermal.data_ingestion.synthetic_dem import generate_synthetic_dem
    from artemis_thermal.simulation.runner import SimulationRunner
    from artemis_thermal.thermal_solver.crank_nicolson import (
        CrankNicolsonSolver,
        create_thermal_column,
    )

    cfg = load_config()
    cfg = replace(cfg, synthetic_dem=replace(cfg.synthetic_dem, grid_resolution_m=resolution_m))
    mesh = dem_to_mesh(generate_synthetic_dem(cfg.synthetic_dem), cfg.lunar.radius_m)
    faces = mesh.face_centroids.shape[0]

    t0 = time.perf_counter()
    engine = IlluminationEngine(
        mesh,
        solar_angular_radius_rad=cfg.lunar.solar_angular_radius_rad,
        num_samples=cfg.illumination.solar_disk_samples,
    )
    bvh_s = time.perf_counter() - t0

    frame = enu_frame(cfg.target.latitude_deg, cfg.target.longitude_deg)
    period = cfg.lunar.synodic_period_s
    dt = cfg.solver.dt_s
    shadows_s = 0.0
    thermal_s = 0.0
    solver = CrankNicolsonSolver(cfg)
    column = create_thermal_column(cfg)
    sun_source = SyntheticSun(datetime(2025, 1, 1, tzinfo=UTC), frame.longitude_deg, -1.5, period)
    t_eq = SimulationRunner(cfg)._equilibrium_temperature(
        engine, sun_source, frame, datetime(2025, 1, 1, tzinfo=UTC),
        np.full(faces, cfg.surface.thermal_emissivity),
    )
    temps = np.repeat(t_eq[:, None], column.T.size, axis=1)
    for k in range(steps):
        sun = frame.from_body(synthetic_sun_body(k * dt, frame.longitude_deg, -1.5, period))
        t = time.perf_counter()
        result = engine.compute(sun)
        shadows_s += time.perf_counter() - t
        q = 0.88 * 1361.0 * result.projected_solar_factor
        t = time.perf_counter()
        solver.step_batch(temps, column.z, column.dz, column.dz_bar, q, dt=dt)
        thermal_s += time.perf_counter() - t
    return {
        "resolution_m": resolution_m,
        "faces": faces,
        "steps": steps,
        "bvh_s": bvh_s,
        "shadows_s": shadows_s,
        "thermal_s": thermal_s,
        "total_s": bvh_s + shadows_s + thermal_s,
        "face_steps_per_s": faces * steps / (shadows_s + thermal_s),
        "peak_mb": _peak_mb(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", default=None)
    parser.add_argument("--resolutions", default="400,200,100")
    args = parser.parse_args(argv)

    bench_case(800.0, steps=2)  # compile everything once
    cases = [bench_case(float(r)) for r in args.resolutions.split(",")]
    import numba

    machine = {
        "cpu": platform.processor() or platform.machine(),
        "threads": numba.get_num_threads(),
        "python": platform.python_version(),
        "numba": numba.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "date": datetime.now(UTC).date().isoformat(),
    }
    print(f"{machine['cpu']}, {machine['threads']} threads, {machine['os']}, "
          f"Python {machine['python']}, Numba {machine['numba']}\n")
    print("| cell size | faces | BVH | shadows | conduction | one day total | peak memory |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        print(f"| {c['resolution_m']:.0f} m | {c['faces']:,} | {c['bvh_s']:.2f} s | "
              f"{c['shadows_s']:.1f} s | {c['thermal_s']:.1f} s | {c['total_s']:.1f} s | "
              f"{c['peak_mb']:.0f} MB |")
    if args.json:
        Path(args.json).write_text(json.dumps({"machine": machine, "cases": cases}, indent=2),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
