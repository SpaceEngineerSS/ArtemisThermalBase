---
title: Python API
nav: Python API
kicker: "06"
order: 6
description: Running simulations and reading results from your own scripts.
---

Everything the command line and the app do is available from Python. The
examples on this page are executed by the test suite, so they work as
written.

## Run a simulation

```python
from dataclasses import replace
from datetime import UTC, datetime

from core_engine.constants import load_config
from simulation.runner import SimulationRunner

config = load_config()                       # or load_config("shackleton_preview")
config = replace(config, synthetic_dem=replace(config.synthetic_dem,
                 radius_m=2000.0, depth_m=500.0, grid_resolution_m=200.0))

results = SimulationRunner(config).run(
    start_time=datetime(2025, 1, 1, tzinfo=UTC),
    duration_hours=6.0,
    dt_s=900.0,
    output_dir="output/api_example",
)
print(results.metadata["num_faces"], float(results.max_surface_T.max()))
```

`load_config` takes a path or the name of a bundled configuration and returns
a frozen dataclass tree; change it with `dataclasses.replace`.
`SimulationRunner.run` accepts:

| argument | default | meaning |
|---|---|---|
| `start_time` | required | UTC start of the recorded period |
| `duration_hours` | `24.0` | length of the recorded period |
| `dt_s` | config | time step |
| `output_interval_s` | config | spacing of stored maps |
| `point_source_mode` | config | `True` for one ray per facet |
| `ephemeris_mode` | config | `"synthetic"`, `"skyfield"` or `"spice"` |
| `latitude_deg`, `longitude_deg` | config | site, for synthetic terrain |
| `spinup_cycles` | config | lunar days before `start_time` |
| `save_data` | `True` | write the run folder |
| `output_dir` | config | where |
| `external_dem` | `None` | a `DEMData`, e.g. from `LOLALoader().load_dem(path)` |
| `progress` | `None` | called with a `RunProgress` after every step |
| `should_stop` | `None` | polled every step; return `True` to cancel |

It returns `SimulationResults`: per-facet arrays `final_surface_T`,
`min_surface_T`, `max_surface_T`, `mean_surface_T`, `final_illumination`,
`mean_illumination`, `sunlit_fraction`; geometry `face_centroids`,
`face_normals`, `face_areas`, `dem_elevation`, `dem_x`, `dem_y`; the series
`times`, `sun_elevations`, `sun_azimuths`, `probe_temps`; and `metadata`.

## Check the cost first

```python
from core_engine.constants import load_config
from data_ingestion.synthetic_dem import synthetic_grid_shape
from simulation.runner import estimate_run_memory_mb, faces_for_dem_shape

config = load_config()
faces = faces_for_dem_shape(*synthetic_grid_shape(config.synthetic_dem))
estimate = estimate_run_memory_mb(faces, config)
print(faces, round(estimate["total"]), "MB")
```

## Read a run folder

```python
from simulation.io_manager import ResultsBundle

run = ResultsBundle.open("output/api_example")
grid = run.raster("surface_T_max")           # north-up (rows, cols) float32
print(run.grid_shape, run.extent, run.snapshot_count)

cell = run.face_at(0.0, 0.0)                  # facet index under a point [m]
first_frame = run.raster("snapshots_surface_T", frame=0)
hours = run.timeline["hours"]
```

`ResultsBundle.open` raises `ResultsError` with a readable message if the
folder is missing, incomplete or from an older version. Arrays are opened
memory-mapped and only read when used.

## Figures

```python
from simulation.io_manager import ResultsBundle
from visualization.figures import render_run_figures

paths = render_run_figures(ResultsBundle.open("output/api_example"), "output/api_example",
                           dpi=120, hero_dpi=120)
print([p.name for p in paths])
```

The colour maps and palette live in `artemis_thermal.style`;
`visualization.figures.colormap("temperature")` returns the Matplotlib map
used everywhere.

## Lower-level pieces

| module | what you get |
|---|---|
| `core_engine.frames` | `body_to_enu`, `projected_frame`, `frame_for_dem`, `curvature_drop` |
| `core_engine.mesh` | `dem_to_mesh(dem, curvature_radius_m)` → `TriangleMesh` |
| `core_engine.illumination` | `IlluminationEngine(mesh, ...).compute(sun_dir)` → visible fraction and projected factor |
| `core_engine.raytracer` | `build_bvh`, the shadow kernels, `moller_trumbore` |
| `data_ingestion.ephemeris` | `SyntheticSun`, `SolarEphemeris` (Skyfield), `iau_moon_rotation` |
| `data_ingestion.spice_ephemeris` | `SpiceSolarEphemeris` |
| `data_ingestion.lola_loader` | `LOLALoader().load_dem(path, max_size=...)` |
| `thermal_solver.crank_nicolson` | `CrankNicolsonSolver.step_batch` on a dense `(faces, nodes)` array |
| `thermal_solver.volatiles` | `sublimation_rate`, `ice_retention_timescale`, `compute_cold_trap_map` |

A Sun source is anything with
`sun_state(utc_time, solar_constant) -> SunState`, where `SunState` carries a
body-frame unit vector, the distance and the irradiance; pass your own to
experiment with other ephemerides.
