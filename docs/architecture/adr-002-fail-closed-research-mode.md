# ADR-002: Fail-Closed Research Data Mode

## Status

Accepted — 2026-08-03

## Context

The application needs a fast offline preview and a traceable publication path.
Silent fallback from unavailable real data to synthetic terrain, or from lunar
libration to an analytical rotation, can create plausible but mislabelled output.

## Decision

Keep preview and research modes in one solver, but make research mode fail closed.
It requires a real DEM with a verified SHA-256 sidecar, pinned NAIF kernels,
`MOON_ME` geometry, extended solar disk, and minimum spin-up cycles. Synthetic
fallback remains available only through explicit preview commands.

## Trade-offs

- Research setup is less convenient and requires external files.
- Kernel and DEM binaries are too large for Git and need manifest-driven setup.
- The same physics core avoids a second implementation drifting from preview.

## Revisit trigger

Revisit after a versioned public data bundle and completed Diviner validation
pipeline are available.
