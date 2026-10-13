# ADR-004: The desktop app runs simulations in a separate process

## Status

Accepted — 2026-10-13

## Context

The Streamlit dashboard of 0.3–0.4 re-ran its whole script on every click and
could not start a simulation at all. A desktop app has to start runs, show
progress and stay smooth on modest laptops.

The simulation loop holds the Python interpreter lock between its Numba calls.
In a thread it would make the interface stutter for the whole run, and there
is no clean way to interrupt a thread that is inside a compiled kernel.

## Decision

- Runs execute in a child process started with `spawn`. The child reports
  progress and log lines through a queue; the window drains it about 15 times
  a second.
- Cancel sets an event the runner checks every step; closing the window
  terminates a child that does not stop within two seconds.
- Reading run folders and computing summaries happen in a thread pool; only
  the newest request per view delivers its result.
- Maps are images (one pixel per DEM cell) drawn with pyqtgraph, and run
  folders are memory-mapped.

## Consequences

- The window never blocks, and a crash in a run cannot take the app down.
- The child pays Numba's start-up once per run (seconds, from the on-disk
  cache).
- Progress arrives a little delayed and coarsened, which is all a person
  watching needs.
