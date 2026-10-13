---
title: Validation and limitations
nav: Limitations
kicker: "09"
order: 9
description: What has been checked, what has not, and the assumptions that bound every result.
---

## What is checked

These run on every change, on Windows and Linux:

- **Geometry.** Rotations are proper (determinant +1); the map projection
  matches PROJ; our grid frame agrees with the one built from the DEM's own
  CRS through rasterio; our IAU lunar orientation agrees with SPICE `MOON_ME`
  to 0.01°; the Sun elevation over Shackleton on 2025-01-01 matches a stored
  SPICE reference to 10⁻⁶°.
- **Ray tracing.** Ray–triangle and ray–box edge cases; the compiled BVH gives
  bit-identical shadow maps to the earlier Python builder; shadows move the
  right way with Sun elevation and azimuth.
- **Heat conduction.** Energy balance over a lunar cycle; stability at very
  low and very high temperatures; decay of the thermal wave with depth; the
  geothermal flux enters once; batched and single-column solvers agree.
- **Sublimation.** Vapour pressure and mass flux against reference values.
- **Pipeline.** Run folders round-trip; incomplete folders are reported;
  research mode refuses synthetic terrain, unverified DEMs and non-SPICE Suns;
  the installed command runs from outside the source tree; the desktop app
  opens a run, inspects a site and runs a simulation in its worker process.

## What is not

**There is no comparison with LRO Diviner yet.** Diviner measured surface
brightness temperatures over the poles for more than a decade, and it is the
obvious reference. A fair comparison needs the instrument's footprint, local
time and emission angle for each observation, and the channel-to-bolometric
conversion; that work is open. The function that computes bias and RMSE while
ignoring Diviner's missing-data values (`artemis_thermal/validation/diviner.py`) exists and is
tested, the rest does not.

Until it is done, treat absolute temperatures as model output with the
uncertainties below, and differences between runs (one setting changed) as
the stronger result.

## Assumptions

| assumption | effect | how large |
|---|---|---|
| No lateral heat flow between columns | shadow edges too sharp in temperature | negligible for cells ≫ 10 cm |
| One bounce of terrain infrared, gray | crater floors too cold; no reflected infrared, no scattered sunlight | floors without IR sit near 21 K |
| Uniform regolith (Hayne 2017 profile) everywhere | no rocks, no ice-filled pores | local, unknown |
| Equatorial geothermal flux at the pole | deep temperatures | ±50 % on 18 mW m⁻² |
| Hapke parameters of mature highlands | absorbed sunlight | a few per cent |
| Simplified roughness emissivity | emission | Δε < 0.02 |
| Sphere, not ellipsoid; DEM from LOLA 118 m resampled | geometry | sub-pixel |
| No Earthshine, no dust | night-side heating | Earthshine is at most about 0.1 W m⁻² |

## Numerical choices

| choice | default | how we know it is enough |
|---|---|---|
| time step | 300 s | 0.13 K from a 30 s reference over a lunar day ([table](configuration.md#solver)) |
| depth grid | 100 layers to 6.2 m | below the annual skin depth; tested wave decay |
| solar disk samples | 16 | 64 for research; a refinement sweep is part of any publication |
| view-factor rays | 128 | 512 for research; reciprocity reported per run |
| spin-up | 0 (preview) | research mode requires 3 lunar days and reports the last change |

A result you intend to publish needs its own convergence sweep in these
parameters. One run is not evidence of convergence.
