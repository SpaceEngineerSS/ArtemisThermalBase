# Roadmap

What we plan to do next, roughly in order. Nothing here is promised for a
date; issues and pull requests on any of it are welcome.

## Science

1. **Diviner comparison.** Match model output to Diviner observations by
   footprint, local time and emission angle; report bias, RMSE and coverage
   per region. Until this exists we do not call the model validated.
2. **Multi-bounce radiosity.** Iterate terrain infrared to convergence and add
   scattered sunlight, both conserving energy under view-factor reciprocity.
   Crater floors are where the current one-bounce model is weakest.
3. **Convergence sweeps as a command.** `artemis sweep` over disk samples,
   view-factor rays, time step and grid, with a report, so that every published
   result can carry its own convergence evidence.
4. **Enthalpy-based energy check.** Replace the ρ·cp(T)·T diagnostic with the
   integral of heat capacity, which is the correct energy for temperature-
   dependent cp.

## Speed

5. **Horizon maps.** Precompute each facet's horizon elevation against azimuth
   once; a month of shadows then costs table lookups instead of ray tracing.
   This is how long illumination studies (Mazarico et al. 2011) stay
   tractable, and it would make year-long runs practical on a laptop.
6. **Cache geometry by DEM hash.** Reuse the BVH, view factors and horizon maps
   between runs on the same terrain.

## App

7. **3D terrain view** with level of detail, optional, for presentations.
8. **Paths.** Draw a traverse on the map and get sunlit time, temperature
   range and longest dark spell along it.
9. **GeoTIFF export** of any layer in the DEM's own CRS, for GIS tools.

## Done in 0.5.0

Compiled BVH builder (Phase 6.1 of the old plan), correct Sun geometry,
desktop app, run-folder format with snapshots, pinned research data. See the
[changelog](CHANGELOG.md).
