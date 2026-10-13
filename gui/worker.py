"""Run simulations in a child process and report back to the UI.

The physics loop holds the GIL between Numba calls, so a thread would make
the window stutter. A separate process keeps the UI at full frame rate and
makes cancel immediate. Messages travel through a ``multiprocessing.Queue``
that a QTimer drains about 15 times a second.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal


@dataclass(frozen=True)
class RunRequest:
    """Everything the run form collects. ``None`` means "use the config"."""

    config: str | None = None
    output_dir: str = "output/run"
    dem_path: str | None = None
    dem_max_size: int | None = None
    crater_radius_m: float | None = None
    grid_resolution_m: float | None = None
    duration_hours: float = 24.0
    dt_s: float | None = None
    output_interval_s: float | None = None
    start_utc: str | None = None
    ephemeris_mode: str | None = None
    point_source: bool | None = None
    disk_samples: int | None = None
    spinup_cycles: int | None = None
    view_factors: bool | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def build_config(request: RunRequest):
    """Load the base configuration and apply the form's overrides."""
    from core_engine.constants import load_config

    cfg = load_config(request.config)
    if request.grid_resolution_m is not None:
        cfg = replace(cfg, synthetic_dem=replace(
            cfg.synthetic_dem, grid_resolution_m=request.grid_resolution_m))
    if request.crater_radius_m is not None:
        cfg = replace(cfg, synthetic_dem=replace(
            cfg.synthetic_dem, radius_m=request.crater_radius_m))
    if request.disk_samples is not None:
        cfg = replace(cfg, illumination=replace(
            cfg.illumination, solar_disk_samples=request.disk_samples))
    if request.view_factors is not None:
        cfg = replace(cfg, view_factors=replace(cfg.view_factors, enabled=request.view_factors))
    return cfg


class _QueueLogHandler(logging.Handler):
    def __init__(self, out: Any) -> None:
        super().__init__(logging.INFO)
        self._out = out

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._out.put_nowait(("log", f"{record.levelname.lower():7s} {record.getMessage()}"))
        except Exception:  # noqa: S110 - a full queue must never kill the run
            pass


def _child(request: RunRequest, out: Any, stop: Any) -> None:
    """Entry point of the worker process."""
    root = logging.getLogger()
    root.handlers[:] = [_QueueLogHandler(out)]
    root.setLevel(logging.INFO)
    try:
        from simulation.io_manager import ResultsBundle
        from simulation.runner import RunCancelledError, SimulationRunner

        cfg = build_config(request)
        dem = None
        if request.dem_path:
            from data_ingestion.lola_loader import LOLALoader

            dem = LOLALoader(
                require_provenance=cfg.research.enabled and cfg.research.require_dem_provenance
            ).load_dem(request.dem_path, max_size=request.dem_max_size)
        start_text = request.start_utc or cfg.time_range.start
        start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
        start = start.replace(tzinfo=UTC) if start.tzinfo is None else start

        last = [0.0]

        def report(p) -> None:
            now = time.monotonic()
            if now - last[0] > 0.08 or p.step == p.total_steps:
                last[0] = now
                out.put(("progress", p))

        try:
            SimulationRunner(cfg).run(
                start_time=start,
                duration_hours=request.duration_hours,
                dt_s=request.dt_s,
                output_interval_s=request.output_interval_s,
                point_source_mode=request.point_source,
                ephemeris_mode=request.ephemeris_mode,
                spinup_cycles=request.spinup_cycles,
                output_dir=request.output_dir,
                external_dem=dem,
                progress=report,
                should_stop=stop.is_set,
            )
        except RunCancelledError as exc:
            out.put(("cancelled", str(exc)))
            return
        out.put(("log", "info    rendering report figures"))
        from visualization.figures import render_run_figures

        render_run_figures(ResultsBundle.open(request.output_dir), request.output_dir,
                           dpi=cfg.output.dpi, hero_dpi=cfg.output.hero_dpi)
        out.put(("done", str(Path(request.output_dir).resolve())))
    except Exception as exc:
        out.put(("error", f"{type(exc).__name__}: {exc}", traceback.format_exc()))


class RunController(QObject):
    """Starts one worker process at a time and turns its messages into signals."""

    progress = Signal(object)    # RunProgress
    log = Signal(str)
    finished = Signal(str)       # run folder
    failed = Signal(str, str)    # message, traceback
    cancelled = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._ctx = mp.get_context("spawn")
        self._process: Any = None
        self._queue: Any = None
        self._stop: Any = None
        self._timer = QTimer(self)
        self._timer.setInterval(66)
        self._timer.timeout.connect(self._drain)

    @property
    def running(self) -> bool:
        """True while a worker process is alive."""
        return self._process is not None and self._process.is_alive()

    def start(self, request: RunRequest) -> None:
        """Launch a run; ignored if one is already going."""
        if self.running:
            return
        self._queue = self._ctx.Queue(maxsize=2000)
        self._stop = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_child, args=(request, self._queue, self._stop), daemon=True,
            name="artemis-run",
        )
        self._process.start()
        self._timer.start()

    def cancel(self) -> None:
        """Ask the worker to stop at the next time step."""
        if self._stop is not None:
            self._stop.set()

    def shutdown(self) -> None:
        """Stop the worker (used when the window closes)."""
        self.cancel()
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
        self._timer.stop()

    def _drain(self) -> None:
        if self._queue is None:
            return
        latest = None
        try:
            for _ in range(500):
                kind, *payload = self._queue.get_nowait()
                if kind == "progress":
                    latest = payload[0]
                elif kind == "log":
                    self.log.emit(payload[0])
                elif kind == "done":
                    self._finish()
                    self.finished.emit(payload[0])
                    return
                elif kind == "cancelled":
                    self._finish()
                    self.cancelled.emit(payload[0])
                    return
                elif kind == "error":
                    self._finish()
                    self.failed.emit(payload[0], payload[1])
                    return
        except queue.Empty:
            pass
        if latest is not None:
            self.progress.emit(latest)
        if self._process is not None and not self._process.is_alive() and self._queue.empty():
            code = self._process.exitcode
            self._finish()
            self.failed.emit(f"The worker process exited unexpectedly (code {code}).", "")

    def _finish(self) -> None:
        self._timer.stop()
        if self._process is not None:
            self._process.join(timeout=1.0)
        self._process = None
