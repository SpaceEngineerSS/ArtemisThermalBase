# ADR-003: The Sun arrives in the body frame and is rotated once

## Status

Accepted — 2026-10-13

## Context

Up to 0.4, each Sun source produced a "local" vector itself: the Skyfield path
through its own lunar rotation, the SPICE path through a north/east/up matrix,
and the synthetic path directly in an assumed local frame. Three conventions
for one quantity drifted apart. The Skyfield rotation was wrong by tens of
degrees, the local frame was left-handed, and none of them was rotated into the
axes of a projected DEM. Every test passed, because every test checked one
source against itself.

## Decision

- Every Sun source returns the Moon→Sun unit vector in the lunar body frame
  (`MOON_ME` or the IAU mean-Earth frame) and the distance. Nothing else.
- The mesh frame is derived from the terrain: the DEM's CRS for georeferenced
  data, local east/north/up for synthetic terrain.
- One function builds the body→mesh rotation; the runner applies it once per
  step for every source alike.
- Tests compare across boundaries: our IAU rotation against SPICE, our map
  projection against PROJ, our grid frame against rasterio's.

## Consequences

- A new Sun source (another ephemeris, a test double) only has to produce a
  body-frame vector.
- The rotation and the site coordinates are written to every run's metadata,
  so a result can be checked without re-running.
- Results with real ephemerides changed in 0.5.0; the changelog says so.
