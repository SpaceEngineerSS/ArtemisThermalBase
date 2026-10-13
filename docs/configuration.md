---
title: Configuration
nav: Configuration
kicker: "04"
order: 4
description: Every key of the YAML configuration, its default, and what changes when you change it.
---

A run is described by one YAML file. Three ship with the package:

| name | for |
|---|---|
| `default_config` | synthetic bowl, analytical Sun, 200 m cells; a minute on a laptop |
| `shackleton_preview` | real LOLA terrain and SPICE Sun with preview numerics |
| `research_shackleton` | fail-closed settings meant for results you would publish |

`artemis configs` prints where they live. Pass a name or a path:

```bash
artemis run --config shackleton_preview --dem data/processed/shackleton_lola_200m.tif
artemis run --config my_case.yaml
```

Your own file does not need to repeat the defaults. Start it with
`extends: default_config.yaml` (relative to your file, or copy the default next
to it) and write only what differs; nested sections are merged key by key.

```yaml
extends: default_config.yaml
synthetic_dem:
  radius_m: 4000.0
  grid_resolution_m: 50.0
solver:
  illumination:
    point_source_mode: true
```

Values below are the defaults from `default_config.yaml`. Command-line options
override the file for one run; the app's Run page does the same.

## target

| key | default | meaning |
|---|---|---|
| `target.name` | `Shackleton crater` | label only |
| `target.latitude_deg` | `-89.67` | site latitude; the centre of synthetic terrain |
| `target.longitude_deg` | `129.78` | site longitude, degrees east |

For a georeferenced DEM the site is the DEM centre, read from its CRS; these
two values then only matter for the synthetic Sun's starting longitude.

## constants

| key | default | meaning |
|---|---|---|
| `constants.stefan_boltzmann` | `5.670374419e-8` | W m⁻² K⁻⁴, CODATA 2018 |
| `constants.solar_constant` | `1361.0` | W m⁻² at 1 AU (Kopp & Lean 2011); scaled by 1/d² with SPICE or Skyfield |
| `constants.astronomical_unit` | `1.495978707e11` | m |
| `constants.speed_of_light` | `2.99792458e8` | m s⁻¹ |

## lunar

| key | default | meaning |
|---|---|---|
| `lunar.radius_m` | `1737400.0` | sphere used for curvature and projections |
| `lunar.sidereal_period_s` | `2360591.5104` | 27.32 d |
| `lunar.synodic_period_s` | `2551442.9` | 29.53 d; one spin-up cycle, one synthetic day |
| `lunar.obliquity_deg` | `1.5424` | to the ecliptic; sets the physical range of the sub-solar latitude |
| `lunar.solar_angular_diameter_deg` | `0.533` | size of the solar disk that is sampled for penumbra |

## surface

| key | default | meaning |
|---|---|---|
| `surface.bond_albedo` | `0.12` | used when `hapke.enabled` is false |
| `surface.thermal_emissivity` | `0.95` | intrinsic emissivity; roughness may raise the effective value |

## regolith

Depth profiles from Hayne et al. (2017). See [Physics](physics.md#regolith-properties).

| key | default | meaning |
|---|---|---|
| `regolith.density.surface_kg_m3` | `1100.0` | density at z = 0 |
| `regolith.density.deep_kg_m3` | `1800.0` | density at depth |
| `regolith.density.transition_depth_m` | `0.06` | e-folding depth H |
| `regolith.conductivity.surface.k_contact` | `7.4e-4` | W m⁻¹ K⁻¹, upper layer |
| `regolith.conductivity.surface.k_radiative` | `2.0e-11` | coefficient of T³, upper layer |
| `regolith.conductivity.deep.k_contact` | `3.4e-3` | lower layer |
| `regolith.conductivity.deep.k_radiative` | `1.0e-11` | lower layer |
| `regolith.conductivity.layer_boundary_m` | `0.02` | depth where the layers meet |
| `regolith.specific_heat.c0` … `c4` | see file | cp(T) = c0 + c1 T^½ + c2 T + c3 T² + c4 T³ |
| `regolith.specific_heat.minimum_value` | `8.0` | floor that keeps cp positive near 0 K |
| `regolith.geothermal_flux` | `0.018` | W m⁻², enters at the bottom of each column |

## solver

| key | default | meaning |
|---|---|---|
| `solver.thermal.grid.dz_surface_m` | `5.0e-4` | thickness of the top cell |
| `solver.thermal.grid.growth_ratio` | `1.07` | each cell is this much thicker than the one above |
| `solver.thermal.grid.num_layers` | `100` | 101 nodes, 6.2 m deep with the defaults |
| `solver.thermal.time.dt_s` | `300.0` | time step |
| `solver.thermal.time.max_dt_s` | `3600.0` | runs with a larger `--dt` are refused |
| `solver.thermal.newton.max_iterations` | `20` | per step and column |
| `solver.thermal.newton.tolerance_K` | `1.0e-4` | largest change anywhere in the column |
| `solver.thermal.newton.relaxation` | `1.0` | under-relaxation, 1 = none |
| `solver.thermal.initial_condition` | `equilibrium` | `equilibrium` or `uniform` |
| `solver.thermal.initial_temperature_K` | `200.0` | used by `uniform` |
| `solver.raytracer.bvh.max_leaf_triangles` | `4` | triangles per BVH leaf |
| `solver.raytracer.bvh.sah_num_bins` | `16` | bins per axis in the split search |
| `solver.raytracer.epsilon` | `1.0e-10` | ray-triangle zero test |
| `solver.illumination.solar_disk_samples` | `16` | rays per face toward the disk; 64 for research |
| `solver.illumination.point_source_mode` | `false` | one ray to the disk centre, no penumbra |

`dt` sets accuracy and cost together. We ran one column through a lunar day
with an abrupt sunrise and sunset and compared against a 30 s reference:

| dt | largest error | mean error |
|---:|---:|---:|
| 120 s | 0.03 K | < 0.001 K |
| 300 s | 0.13 K | 0.001 K |
| 600 s | 0.35 K | 0.003 K |
| 1200 s | 1.9 K | 0.008 K |
| 3600 s | 5.0 K | 0.03 K |

The largest errors sit in the hour after sunrise. 300 s is the default;
1200 s is fine for month-long previews when you care about maps more than
about the exact minute a wall warms up.

## terrain

| key | default | meaning |
|---|---|---|
| `terrain.curvature` | `true` | lower the DEM onto the lunar sphere |

## synthetic_dem

| key | default | meaning |
|---|---|---|
| `synthetic_dem.type` | `parabolic_bowl` | `parabolic_bowl`, `conical` or `flat` |
| `synthetic_dem.radius_m` | `10500.0` | crater radius |
| `synthetic_dem.depth_m` | `4200.0` | rim-to-floor depth |
| `synthetic_dem.rim_height_m` | `300.0` | raised rim above the plain |
| `synthetic_dem.grid_resolution_m` | `200.0` | cell size; halving it quadruples the faces |
| `synthetic_dem.domain_padding_m` | `3000.0` | flat ground around the rim |
| `synthetic_dem.seed` | `42` | reserved for terrain noise |

## view_factors

One bounce of thermal infrared between facets. Expensive on large grids;
matters inside craters, where walls warm the floor by tens of kelvin.

| key | default | meaning |
|---|---|---|
| `view_factors.enabled` | `false` | |
| `view_factors.num_rays` | `128` | Monte Carlo rays per face; 512 for research |
| `view_factors.seed` | `42` | |
| `view_factors.reciprocity_tolerance` | `0.2` | warn above this mismatch between power sent and received per face |
| `view_factors.max_neighbors` | `256` | distinct faces stored per row; extra hits are counted and reported |
| `view_factors.max_memory_mb` | `1024.0` | refuse before allocating more |

## volatiles

| key | default | meaning |
|---|---|---|
| `volatiles.enabled` | `true` | compute the ice-stability map |
| `volatiles.ice_thresholds_K.stable` | `110.0` | maximum T below which ice is stable for Gyr |
| `volatiles.ice_thresholds_K.marginal` | `115.0` | upper bound of the marginal class |

## hapke

| key | default | meaning |
|---|---|---|
| `hapke.enabled` | `true` | angle-dependent albedo instead of `surface.bond_albedo` |
| `hapke.single_scattering_albedo` | `0.23` | w |
| `hapke.b` | `0.21` | Henyey–Greenstein lobe width |
| `hapke.c` | `0.70` | backward fraction |
| `hapke.B_SH0` | `1.0` | shadow-hiding opposition amplitude |
| `hapke.h_s` | `0.065` | opposition width |
| `hapke.lookup_size` | `2049` | table resolution in cos i |

## roughness

| key | default | meaning |
|---|---|---|
| `roughness.enabled` | `true` | effective emissivity from sub-pixel roughness |
| `roughness.rms_slope_deg` | `20.0` | |
| `roughness.cavity_coefficient` | `0.4` | |

## ephemeris

| key | default | meaning |
|---|---|---|
| `ephemeris.mode` | `synthetic` | `synthetic`, `skyfield` or `spice` |
| `ephemeris.synthetic_subsolar_lat_deg` | `-1.5` | fixed sub-solar latitude of the synthetic Sun |
| `ephemeris.kernel_name` | `de421.bsp` | Skyfield kernel, downloaded once into `data_dir` |
| `ephemeris.data_dir` | `data/ephemeris` | |
| `ephemeris.kernel_files` | `[]` | SPICE kernels, loaded in order |
| `ephemeris.frame` | `MOON_ME` | SPICE body-fixed frame |
| `ephemeris.aberration_correction` | `LT+S` | SPICE light-time and stellar aberration |

Relative kernel paths are looked up in the working directory first, then next
to the project that holds the configuration file.

## research

| key | default | meaning |
|---|---|---|
| `research.enabled` | `false` | turn on every check below |
| `research.require_real_dem` | `true` | refuse synthetic terrain |
| `research.require_dem_provenance` | `true` | refuse a DEM without a matching provenance sidecar |
| `research.minimum_spinup_cycles` | `3` | refuse fewer spin-up lunar days |

Research mode also refuses any ephemeris other than `spice` and stops if the
roughness model fails instead of falling back. See [ADR-002](architecture/adr-002-fail-closed-research-mode.md).

## time_range

| key | default | meaning |
|---|---|---|
| `time_range.start` | `2025-01-01T00:00:00Z` | UTC start of the recorded period |
| `time_range.end` | `2025-01-02T00:00:00Z` | used when `--duration` is not given |
| `time_range.output_interval_s` | `3600.0` | spacing of stored maps |
| `time_range.spinup_cycles` | `0` | lunar days simulated before `start` and discarded |
| `time_range.spinup_convergence_tolerance_K` | `0.1` | warn if the last spin-up day changed the surface by more |

## resources

| key | default | meaning |
|---|---|---|
| `resources.max_memory_mb` | `1536.0` | the run's own estimate must stay below this |

The estimate includes mesh, BVH, the thermal state (808 bytes per face with
100 layers), work arrays and view factors. The app shows it while you type.

## output

| key | default | meaning |
|---|---|---|
| `output.directory` | `output` | default run folder |
| `output.dpi` | `150` | report figures |
| `output.hero_dpi` | `200` | the wide summary plate |
| `output.max_snapshot_mb` | `256.0` | disk budget for per-step maps; frames are thinned to fit |
