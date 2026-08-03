# Scientific Validation Status

Status: **research-capable implementation; observational validation pending**.

Completed checks:

- unit and integration tests for ray intersections, thermal stepping, geothermal
  boundary accounting, configuration, I/O, provenance, and missing Diviner data;
- batched and scalar thermal solvers agree in regression tests;
- NAIF DE440 `MOON_ME` smoke calculation passed with pinned kernel hashes;
- solar-disk projected irradiance is integrated per visible sample, including a
  solar centre below the horizon;
- research mode rejects synthetic or unverified DEM input and insufficient spin-up.

Not completed:

- spatial, temporal, footprint, emission-angle, and channel matching to Diviner;
- published bias/RMSE/coverage and uncertainty budget;
- timestep, DEM, disk-sample, vertical-grid, spin-up, and view-factor convergence matrix;
- iterative multi-bounce thermal radiosity and global energy-closure diagnostic;
- independent reproduction on a second machine.

Until those artifacts exist, outputs must not be described as “validated against
Diviner,” “flight qualified,” or predictive ground truth. Suitable wording is
“model result generated from provenance-tracked LOLA topography and NAIF DE440
geometry under the documented assumptions.”
