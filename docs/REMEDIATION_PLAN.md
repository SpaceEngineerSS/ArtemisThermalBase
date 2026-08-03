# ArtemisThermalBase Improvement Plan

This plan tracks the findings from the August 2026 architecture, physics, and
performance audit. “Complete” means implemented and covered by an automated
check; it does not mean the scientific model is validated against observations.

## Phase 1 — Operability and Configuration

Status: **Complete**

- Parse target, view-factor, Hapke, roughness, volatile, ephemeris, time-range,
  spin-up, and memory settings into typed configuration objects.
- Restore the installed `artemis` command.
- Unify dashboard and simulation persistence file names.
- Replace unsafe defaults with a runnable 200 m preview profile.
- Add hard thermal-state and view-factor allocation preflights.

Acceptance evidence:

- CLI entry point imports and `python main.py --help` succeeds.
- Dashboard adapter round-trips canonical saved output in tests.
- Invalid or oversized allocations fail before expensive computation.

## Phase 2 — Physics Integration Corrections

Status: **Complete**

- Inject geothermal heat only at the deep boundary.
- Pass per-face roughness emissivity into the radiative boundary condition.
- Apply gray-surface source emission and receiver absorption to one-bounce IR.
- Preserve the visible part of the extended solar disk when its center is just
  below the horizon.
- Precompute a Hapke albedo lookup table instead of re-integrating it for every
  face and time step.

Acceptance evidence:

- Energy-budget regression measures one geothermal flux.
- Per-face emissivity changes the resulting surface temperature.
- Extended-source regression detects partial illumination at -0.1° elevation.

## Phase 3 — Batched Thermal Core

Status: **Complete**

- Replace per-face Python objects with one dense temperature matrix.
- Share immutable depth arrays between every face.
- Run the Crank-Nicolson columns in Numba-parallel face batches.
- Bound the temporary update matrix with `batch_size_faces`.

Measured result on the audit workstation:

- Old scalar call path: approximately 67,000 face-steps/s in the small profile.
- Batched path: approximately 347,000 face-steps/s at 100,000 faces and 100
  depth layers, about a 5× thermal-kernel improvement.

## Phase 4 — Time, Location, and Input Integrity

Status: **Complete**

- Connect runner and CLI to target latitude, longitude, UTC start time, and the
  optional Skyfield/JPL ephemeris path.
- Keep a clearly labeled synthetic offline preview mode.
- Add fixed lunar-cycle spin-up and cycle-to-cycle surface convergence metrics.
- Reject angular-degree DEMs and rotated/sheared rasters instead of silently
  treating them as metric north-up grids.
- Correct final-state timestamps.

## Phase 5 — Quality Gates and Documentation

Status: **Complete**

- Add cross-module regression tests for the repaired integration paths.
- Make Ruff and Mypy pass with scientific naming conventions documented in the
  tool configuration.
- Add ADR-001 for the dense batched CPU architecture.
- Remove claims of completed Diviner validation and multi-bounce radiosity.

## Phase 6 — Illumination and Geometry Performance

Status: **Planned**

1. Replace the Python recursive SAH builder with a compiled Morton/LBVH builder
   or a benchmarked native backend.
2. Serialize and cache BVH data by DEM/mesh hash.
3. Cache or interpolate illumination maps over slowly varying solar azimuth and
   elevation, with an explicit interpolation-error tolerance.
4. **Complete:** integrate `max(0, n·s_sample)` over visible solar-disk samples
   in the same ray pass.

Acceptance criteria:

- At least 10× faster BVH construction at 100,000 faces.
- Cached geometry reproduces uncached arrays bit-for-bit.
- Solar-disk irradiance converges under 16/32/64/128 sample refinement.

## Phase 7 — Terrain Radiosity and Scientific Validation

Status: **Planned**

1. Add iterative gray-surface radiosity for reflected thermal IR, conserving
   energy under view-factor reciprocity.
2. Replace the diagnostic `rho*cp(T)*T` energy estimate with the temperature
   integral of heat capacity (enthalpy) for temperature-dependent properties.
3. **Partial:** explicit fail-closed research configuration is present; the
   convergence matrix still needs to be executed and archived.
4. Build the LRO Diviner comparison pipeline with observation geometry,
   uncertainty bars, and reproducible input provenance.
5. Only after validation, reassess C++/GPU acceleration using end-to-end
   profiles rather than isolated ray benchmarks.

Acceptance criteria:

- Closed-cavity radiosity conserves energy within a documented tolerance.
- Numerical refinement produces stable temperatures and energy residuals.
- Diviner comparison reports bias, RMSE, spatial coverage, and uncertainty;
  documentation may claim validation only after these artifacts exist.
