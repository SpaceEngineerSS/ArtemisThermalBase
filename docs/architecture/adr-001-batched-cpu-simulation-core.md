# ADR-001: Batched CPU Simulation Core

## Status

Accepted

## Context

The original runner created one Python object and four NumPy arrays for every
terrain triangle. The default 20 m Shackleton mesh produced 3.645 million
faces, more than 11 GB of thermal-array payload, and billions of Python-level
solver calls. Expensive view-factor physics was also enabled without a hard
memory guard.

The project is a research-oriented, single-machine Python application. It must
remain reproducible and easy to inspect before adding C++ or GPU backends.

## Decision

- Store temperatures in one dense `(num_faces, num_depth_nodes)` array.
- Share the immutable vertical grid across every thermal column.
- Advance faces in one Numba-parallel batch call.
- Keep the scalar solver API as a reference and compatibility path.
- Use a runnable 200 m preview configuration by default.
- Make Monte Carlo terrain IR opt-in and enforce hard allocation preflights.
- Keep Skyfield ephemeris opt-in so offline preview runs do not download data.

## Rationale

This removes Python object amplification while preserving the existing 1-D
Crank-Nicolson equations. It provides a measurable CPU speedup without creating
a second implementation language before the physics and validation suite are
stable.

## Trade-offs

- A dense temperature matrix can still be large at research resolution.
- The batch kernel allocates a second matrix during each update.
- The default preview is not publication resolution.
- Skyfield mode requires a JPL kernel and may download it on first use.

## Consequences

- Thermal state memory is predictable and checked before allocation.
- Per-face roughness emissivity can be passed directly to the boundary solver.
- Future C++/GPU backends can consume the same dense state contract.
- Research runs must explicitly select resolution, spin-up cycles, solar-disk
  samples, and terrain-IR settings and should record the resulting config.

## Revisit Trigger

Reconsider the CPU/Numba backend when profiling shows the batched thermal
kernel, rather than BVH construction or shadow rays, dominates a validated
research workload.

