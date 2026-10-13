---
title: Outputs
nav: Outputs
kicker: "05"
order: 5
description: The run folder, file by file, and how to read it without this package.
---

Every run writes one folder. It holds plain NumPy arrays and one JSON file,
so any language that reads `.npy` can use it; nothing is pickled.

## Files

| file | shape | contents |
|---|---|---|
| `surface_T_final.npy` | (N,) float32 | surface temperature at the last step [K] |
| `surface_T_min.npy` | (N,) float32 | lowest surface temperature over the recorded period |
| `surface_T_max.npy` | (N,) float32 | highest surface temperature over the recorded period |
| `surface_T_mean.npy` | (N,) float32 | time mean of the surface temperature |
| `illumination_final.npy` | (N,) float32 | visible fraction of the solar disk at the last step |
| `illumination_mean.npy` | (N,) float32 | its time mean |
| `sunlit_fraction.npy` | (N,) float32 | share of steps with at least half the disk visible |
| `face_centroids.npy` | (N, 3) float32 | facet centres in the mesh frame [m] |
| `face_normals.npy` | (N, 3) float32 | unit normals |
| `face_areas.npy` | (N,) float32 | facet areas [m²] |
| `dem_elevation.npy` | (ny, nx) float32 | the input elevation grid [m] |
| `dem_x.npy`, `dem_y.npy` | (nx,), (ny,) float64 | vertex coordinates of that grid [m] |
| `timeline.npz` | | `hours`, `sun_elevation_deg`, `sun_azimuth_deg` per stored step |
| `probes.npz` | | temperature at every step for three facets, and `<name>__face` indices |
| `snapshots_surface_T.npy` | (frames, N) float32 | surface temperature at stored steps |
| `snapshots_illumination.npy` | (frames, N) float32 | visible disk fraction at stored steps |
| `metadata.json` | | settings, timing, frame, provenance, version |
| `checksums.sha256` | | SHA-256 of every file above |
| `*.png` | | report figures |

N is the number of facets: two per DEM cell. "Recorded period" means after
spin-up. Snapshots are written every `output_interval_s`; if that would exceed
`output.max_snapshot_mb`, every k-th one is kept and `metadata.json` says so in
`snapshot_stride` and `snapshot_frames`. float32 holds a temperature to about
0.00002 K at 300 K, which is far below anything the model can claim.

## From facets to a map

Facets come in pairs, cell by cell, in the row order of `dem_y`:
facet `2 * (row * (nx - 1) + col)` is the lower-left triangle of the cell,
the next one the upper-right. Averaging each pair gives a raster aligned with
the DEM:

```python
import numpy as np

run = "output/shackleton"
x = np.load(f"{run}/dem_x.npy")
y = np.load(f"{run}/dem_y.npy")
t_max = np.load(f"{run}/surface_T_max.npy")
grid = t_max.reshape(len(y) - 1, len(x) - 1, 2).mean(axis=2)
if y[-1] > y[0]:          # make row 0 the northern (top) row
    grid = grid[::-1]
print(grid.shape, float(grid.min()), float(grid.max()))
```

Inside Python, `simulation.io_manager.ResultsBundle` does this for you and
memory-maps the arrays; see [API](api.md).

## Coordinates

The mesh frame is metres, centred on the DEM. For a georeferenced DEM, $x$
and $y$ are its projected grid axes shifted to the centre, and `metadata.json`
stores the original CRS centre under `frame` together with the rotation from
the lunar body frame (`rotation_body_to_mesh`). $z$ is height above the
tangent plane at the centre, after the curvature drop; `dem_elevation.npy`
keeps the uncorrected heights.

## metadata.json

The keys you will most likely want:

| key | meaning |
|---|---|
| `artemis_version`, `format_version` | software and folder format (2) |
| `start_time`, `duration_hours`, `dt_s` | recorded period and step |
| `num_faces`, `grid_shape`, `resolution_m` | mesh size |
| `ephemeris_mode`, `point_source_mode` | Sun source and model |
| `frame` | kind, site latitude and longitude, body-to-mesh rotation |
| `spinup_cycles`, `spinup_cycle_max_delta_K`, `spinup_converged` | spin-up record |
| `volatile_thresholds_K` | used for the ice map |
| `dem_provenance`, `ephemeris_provenance` | source files and hashes |
| `estimated_memory_mb`, `wall_time_s` | cost |
| `times_utc` | ISO time of every stored step |

## Checking a folder

```bash
cd output/shackleton
sha256sum -c checksums.sha256          # Linux, macOS
```

On Windows, `Get-FileHash -Algorithm SHA256 <file>` gives the same digests.
Folders written by 0.4 or earlier use another layout; the app recognises them
and asks for a re-run instead of showing wrong numbers.
