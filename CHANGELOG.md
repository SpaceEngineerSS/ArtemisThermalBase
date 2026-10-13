# Changelog

All notable changes, newest first. Versions follow [semantic versioning](https://semver.org);
until 1.0 a minor release may change results and file formats, and says so here.

## 0.5.0 — 2026-10-13

This release changes results. Runs that used the Skyfield or SPICE ephemeris
had the Sun in the wrong place; runs with the synthetic Sun had a Sun that
rose and set like at the equator. Re-run anything you made with 0.4.

### Geometry

- **Sun direction on the terrain.** The Skyfield path imported a lunar frame
  that Skyfield does not provide and fell back to a rotation built from active
  instead of passive matrices; the Moon's pole came out about 45° from where it
  is. The local frame was north/east/up, which is left-handed and mirrors every
  azimuth, and the Sun vector was never rotated from local east/north into the
  DEM's grid axes. All Sun sources now return a body-frame vector and one
  module (`artemis_thermal/core_engine/frames.py`) rotates it onto the mesh, using the DEM's own
  CRS. The IAU 2015 lunar orientation, now with its thirteen periodic terms,
  agrees with SPICE `MOON_ME` to better than 0.003°.
- **Polar stereographic convention.** `polar_stereographic.py` used
  y = −ρ cos λ, the mirror image of the PROJ CRS our DEM tool writes. It now
  matches PROJ, and a test checks the two against each other.
- **Lunar curvature.** Terrain is lowered onto the sphere (`terrain.curvature`,
  on by default): 115 m at 20 km from the DEM centre, which matters when the Sun
  is 1.5° above the horizon.
- **Synthetic Sun.** The sub-solar point now circles the Moon once per synodic
  month at a fixed latitude (`ephemeris.synthetic_subsolar_lat_deg`, default
  −1.5°). At the pole that keeps the Sun near the horizon all day, as it is.

### Physics

- **Sublimation** is the Hertz–Knudsen mass flux with the Murphy & Koop (2005)
  vapour pressure. The old function returned a number flux labelled kg m⁻² s⁻¹
  (off by 26 orders of magnitude) and the Langmuir form used a natural-log
  constant inside a log₁₀ formula; one test asserted that ice at 200 K lasts a
  million years. It lasts about 50 days.
- **Cold traps** are classified from the hottest temperature a face reaches
  during the run, with the thresholds from the configuration, instead of from
  the last time step with hard-coded thresholds.
- **Initial condition.** Each column starts at the temperature that radiates
  away its day-mean absorbed sunlight (`solver.thermal.initial_condition:
  equilibrium`). Short runs no longer report the 200 K starting guess as their
  maximum.
- **Newton iteration** now stops when the whole column has converged, not only
  the surface node.
- Research-mode checks also run inside the runner, so `--ephemeris synthetic`
  can no longer bypass the SPICE requirement. A failing roughness model stops a
  research run instead of being silently replaced.

### Speed and memory

- **BVH construction** is compiled: 583k triangles in 1.1 s instead of 338 s.
  Shadow maps are bit-for-bit identical to 0.4.
- Ray kernels no longer allocate per ray, and view-factor post-processing is
  compiled. Reciprocity is now measured as the mismatch between the power each
  facet sends into the terrain and the power it receives (0.16 at 128 rays,
  0.08 at 512 on a test bowl). The old per-entry check with a 10⁻⁵ tolerance
  could never pass on Monte Carlo data, and printed a warning on every run.
- A run no longer keeps one full-mesh copy per output step (11 GB for a month
  at 1M faces). It keeps running minimum, maximum and mean maps and streams
  float32 snapshots to disk inside `output.max_snapshot_mb`.
- The memory check before a run covers mesh, BVH, thermal state, work arrays
  and view factors (`resources.max_memory_mb`, 1.5 GB by default, replacing
  `solver.thermal.max_state_memory_mb`).
- The thermal kernel precomputes density and conductivity profiles and updates
  columns in place; per-step logging moved to DEBUG.

### Desktop app

The Streamlit dashboard is replaced by a PySide6 app (`artemis ui`):
simulations run in a separate process with progress, estimated time and
cancel; results show as rasters with a time scrubber; clicking a cell reports
slope, aspect, temperatures, sunlit share, mean flux on a Sun-tracking panel
and the longest dark spell; two runs on the same grid can be compared as a
difference map. Light and dark themes, bundled fonts, no network access.

### Command line and files

- **One package.** Everything now lives under `artemis_thermal`
  (`artemis_thermal.simulation.runner`, `artemis_thermal.core_engine.frames`,
  …). 0.4 installed seven top-level packages with names such as `gui`,
  `simulation` and `validation` that would collide with other projects.
  Update imports by prefixing `artemis_thermal.`.
- `artemis` has subcommands: `run` (the default), `ui`, `render`, `configs`,
  `warmup`. It works from any directory; bundled configurations are found by
  name (`--config research_shackleton`).
- New run-folder format (version 2, documented in [Outputs](docs/outputs.md)),
  read lazily by `artemis_thermal.simulation.io_manager.ResultsBundle`. Folders written by 0.4
  are recognised and the app asks for a re-run.
- New configuration `shackleton_preview`: real LOLA terrain and SPICE Sun at
  laptop cost.
- Report figures are drawn from rasters and share their colours with the app.
- `tools/fetch_research_data.py` pins kernel hashes, resumes interrupted
  downloads and rejects mismatches. `tools/prepare_lola_dem.py` falls back to
  the S3 mirror of the USGS product.
- `rasterio`, `skyfield` and `spiceypy` moved to the `dem` and `ephemeris`
  extras; `tqdm`, `xarray`, `plotly` and `kaleido` are no longer needed.

### Removed

The Streamlit dashboard, `artemis_thermal/thermal_solver/regolith_properties.py` (unused, and
a copy of the solver's own property functions), `tools/download_sample_data.py`
(downloaded a shaded-relief image and treated it as elevation), unused
ray-tracer and ephemeris helpers, unused configuration keys, and tests that did
arithmetic on literals without calling the code.

## 0.4.0 — 2026-08-03

- Fail-closed research mode: provenance-verified LOLA DEM, NAIF DE440 kernels,
  minimum spin-up.
- Disk-integrated projected irradiance per solar sample.
- Geothermal heat applied once, at the bottom boundary.
- Dense batched thermal state (about 5× faster than per-face objects).
- Checksums for every saved result file.

## 0.3.0 — 2026-04-07

- Hapke directional-hemispherical albedo, Bandfield roughness emissivity.
- Sparse Monte Carlo view factors for one bounce of terrain infrared.
- Volatile stability classification.
- Streamlit dashboard (replaced in 0.5.0).

## 0.1.0 — 2026-02-12

First public version: synthetic crater, BVH shadows, Crank–Nicolson regolith
columns. (This version was tagged "v1.0" in its commit message; it was not a
stable release.)
