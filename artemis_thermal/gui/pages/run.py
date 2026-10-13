"""Run page: set up a simulation, watch it, open the result."""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from artemis_thermal.core_engine.constants import list_bundled_configs, load_config
from artemis_thermal.data_ingestion.synthetic_dem import synthetic_grid_shape
from artemis_thermal.gui.theme import Theme
from artemis_thermal.gui.widgets import ProgressLine, ReadoutGrid, button, label, rule, section
from artemis_thermal.gui.worker import RunController, RunRequest, build_config
from artemis_thermal.simulation.runner import estimate_run_memory_mb, faces_for_dem_shape


def _spin(lo: float, hi: float, value: float, decimals: int = 0, step: float = 1.0,
          suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setSuffix(f"  {suffix}" if suffix else "")
    box.setKeyboardTracking(False)
    box.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
    return box


def _ispin(lo: int, hi: int, value: int, suffix: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(lo, hi)
    box.setValue(value)
    box.setSuffix(f"  {suffix}" if suffix else "")
    box.setKeyboardTracking(False)
    box.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
    return box


class _Form(QWidget):
    """Two-column grid: small grey label on the left, input on the right."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(7)
        self._grid.setColumnMinimumWidth(0, 130)
        self._grid.setColumnStretch(1, 1)
        self._row = 0

    def heading(self, text: str) -> None:
        if self._row:
            self._grid.setRowMinimumHeight(self._row, 10)
            self._row += 1
        self._grid.addWidget(label(text.upper(), "kicker"), self._row, 0, 1, 2)
        self._row += 1

    def add(self, caption: str, widget: QWidget, hint: str = "") -> QWidget:
        cap = label(caption, "label")
        if hint:
            cap.setToolTip(hint)
            widget.setToolTip(hint)
        self._grid.addWidget(cap, self._row, 0, Qt.AlignmentFlag.AlignVCenter)
        self._grid.addWidget(widget, self._row, 1)
        self._row += 1
        return widget


class RunPage(QWidget):
    """Form on the left, estimate and progress on the right."""

    run_finished = Signal(str)
    status = Signal(str)

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Page")
        self._theme = theme
        self._settings = QSettings()
        self.controller = RunController(self)
        self.controller.progress.connect(self._on_progress)
        self.controller.log.connect(self._append_log)
        self.controller.finished.connect(self._on_finished)
        self.controller.failed.connect(self._on_failed)
        self.controller.cancelled.connect(self._on_cancelled)
        self._started = 0.0
        self._faces = 0
        self._steps = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)
        content = QWidget()
        content.setObjectName("Page")
        scroll.setWidget(content)
        body = QVBoxLayout(content)
        body.setContentsMargins(32, 26, 32, 28)
        body.setSpacing(18)
        body.addWidget(section(
            "01  Run", "New simulation",
            "Defaults finish in well under a minute on a four-core laptop. "
            "Hover a label for what it controls.",
        ))

        columns = QHBoxLayout()
        columns.setSpacing(36)
        body.addLayout(columns, 1)

        form = _Form()
        form.setMaximumWidth(520)
        columns.addWidget(form, 1, Qt.AlignmentFlag.AlignTop)

        form.heading("Configuration")
        self.preset = QComboBox()
        for name in list_bundled_configs():
            self.preset.addItem(name, name)
        self.preset.addItem("Other file…", "__browse__")
        self.preset.activated.connect(self._preset_chosen)
        form.add("preset", self.preset, "Base configuration. The fields below override it.")

        form.heading("Terrain")
        self.terrain = QComboBox()
        self.terrain.addItem("Synthetic bowl", "synthetic")
        self.terrain.addItem("GeoTIFF DEM", "dem")
        self.terrain.currentIndexChanged.connect(self._sync_terrain)
        form.add("source", self.terrain)
        dem_row = QWidget()
        dem_layout = QHBoxLayout(dem_row)
        dem_layout.setContentsMargins(0, 0, 0, 0)
        self.dem_path = QLineEdit()
        self.dem_path.setPlaceholderText("projected GeoTIFF, e.g. from tools/prepare_lola_dem.py")
        dem_layout.addWidget(self.dem_path, 1)
        dem_layout.addWidget(button("Browse", None, self._browse_dem))
        self._dem_row = form.add("DEM file", dem_row)
        self.dem_size = _ispin(64, 8192, 600, "px")
        self._dem_size_row = form.add("max DEM side", self.dem_size,
                                      "Larger DEMs are downsampled so neither side exceeds this.")
        self.radius = _spin(0.2, 60.0, 10.5, 2, 0.5, "km")
        self._radius_row = form.add("crater radius", self.radius)
        self.resolution = _spin(5.0, 2000.0, 200.0, 0, 10.0, "m")
        self._res_row = form.add("cell size", self.resolution,
                                 "Grid spacing of the synthetic terrain. Halving it quadruples "
                                 "the number of faces.")

        form.heading("Time")
        self.start = QLineEdit("2025-01-01T00:00:00Z")
        form.add("start (UTC)", self.start, "ISO-8601 time; only matters for Skyfield/SPICE.")
        self.duration = _spin(0.1, 24 * 400, 24.0, 1, 1.0, "h")
        form.add("duration", self.duration)
        self.dt = _spin(10.0, 3600.0, 300.0, 0, 30.0, "s")
        form.add("time step", self.dt)
        self.interval = _spin(0.1, 240.0, 1.0, 1, 0.5, "h")
        form.add("map every", self.interval, "Interval between stored temperature maps.")
        self.spinup = _ispin(0, 10, 0, "lunar days")
        form.add("spin-up", self.spinup, "Lunar days simulated before recording, so the "
                                         "subsurface reaches its periodic state.")

        form.heading("Sun and physics")
        self.ephemeris = QComboBox()
        self.ephemeris.addItem("Synthetic (no data needed)", "synthetic")
        self.ephemeris.addItem("Skyfield + DE421", "skyfield")
        self.ephemeris.addItem("SPICE DE440 (research)", "spice")
        form.add("Sun position", self.ephemeris)
        self.sun_model = QComboBox()
        self.sun_model.addItem("Solar disk, penumbra", False)
        self.sun_model.addItem("Point source (faster)", True)
        self.sun_model.currentIndexChanged.connect(self._update_estimate)
        form.add("Sun model", self.sun_model)
        self.samples = _ispin(4, 256, 16, "rays")
        form.add("disk samples", self.samples, "Rays per face toward the solar disk.")
        self.view_factors = QCheckBox("terrain infrared (view factors)")
        form.add("", self.view_factors, "One bounce of thermal emission between facets. "
                                        "Accurate in craters, costly on large grids.")

        form.heading("Output")
        out_row = QWidget()
        out_layout = QHBoxLayout(out_row)
        out_layout.setContentsMargins(0, 0, 0, 0)
        self.output = QLineEdit(str(Path.cwd() / "output" / "run"))
        out_layout.addWidget(self.output, 1)
        out_layout.addWidget(button("Browse", None, self._browse_output))
        form.add("run folder", out_row)

        for widget in (self.radius, self.resolution, self.duration, self.dt, self.interval,
                       self.spinup, self.samples, self.dem_size):
            widget.valueChanged.connect(self._update_estimate)
        self.view_factors.toggled.connect(self._update_estimate)

        # Right column: estimate + progress ---------------------------------
        right = QVBoxLayout()
        right.setSpacing(10)
        side = QWidget()
        side.setMinimumWidth(320)
        side.setMaximumWidth(460)
        side.setLayout(right)
        columns.addWidget(side, 1, Qt.AlignmentFlag.AlignTop)

        right.addWidget(label("ESTIMATE", "kicker"))
        self.estimate = ReadoutGrid(2)
        self.estimate.add("faces", "faces", "", 0)
        self.estimate.add("steps", "time steps", "", 0)
        self.estimate.add("memory", "memory", "MB", 0)
        self.estimate.add("time", "run time, rough", "", 0)
        right.addWidget(self.estimate)
        self.limit_note = label("", "caption")
        self.limit_note.setWordWrap(True)
        right.addWidget(self.limit_note)

        right.addSpacing(8)
        buttons = QHBoxLayout()
        self.start_btn = button("Start run", "primary", self.start_run)
        self.cancel_btn = button("Cancel", None, self.controller.cancel)
        self.cancel_btn.setEnabled(False)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addStretch(1)
        right.addLayout(buttons)
        right.addSpacing(6)

        self.progress = ProgressLine(theme.color("rule"), theme.color("accent"))
        right.addWidget(self.progress)
        self.progress_text = label("Idle.", "mono")
        self.progress_text.setWordWrap(True)
        right.addWidget(self.progress_text)
        right.addWidget(rule())
        right.addWidget(label("LOG", "kicker"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumHeight(220)
        right.addWidget(self.log)

        self._config = load_config()
        self._load_from_config(self._config)
        self._sync_terrain()

    # -- config <-> form --------------------------------------------------

    def apply_theme(self, theme: Theme) -> None:
        """Restyle custom-painted parts."""
        self._theme = theme
        self.progress.set_colors(theme.color("rule"), theme.color("accent"))

    def _preset_chosen(self, index: int) -> None:
        data = self.preset.itemData(index)
        if data == "__browse__":
            path, _ = QFileDialog.getOpenFileName(self, "Configuration", str(Path.cwd()),
                                                  "YAML (*.yaml *.yml)")
            if not path:
                self.preset.setCurrentIndex(0)
                return
            self.preset.insertItem(index, Path(path).name, path)
            self.preset.setCurrentIndex(index)
            data = path
        try:
            self._config = load_config(data)
        except (ValueError, FileNotFoundError) as exc:
            self.status.emit(str(exc))
            return
        self._load_from_config(self._config)

    def _load_from_config(self, cfg) -> None:
        self.radius.setValue(cfg.synthetic_dem.radius_m / 1000.0)
        self.resolution.setValue(cfg.synthetic_dem.grid_resolution_m)
        self.dt.setValue(cfg.solver.dt_s)
        self.interval.setValue(cfg.time_range.output_interval_s / 3600.0)
        self.spinup.setValue(cfg.time_range.spinup_cycles)
        self.samples.setValue(cfg.illumination.solar_disk_samples)
        self.view_factors.setChecked(cfg.view_factors.enabled)
        self.sun_model.setCurrentIndex(1 if cfg.illumination.point_source_mode else 0)
        index = self.ephemeris.findData(cfg.ephemeris.mode)
        self.ephemeris.setCurrentIndex(max(index, 0))
        self.start.setText(cfg.time_range.start)
        self._update_estimate()

    def _sync_terrain(self) -> None:
        dem = self.terrain.currentData() == "dem"
        for row in (self._dem_row, self._dem_size_row):
            row.setEnabled(dem)
        for row in (self._radius_row, self._res_row):
            row.setEnabled(not dem)
        self._update_estimate()

    def _browse_dem(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "DEM", str(Path.cwd()),
                                              "GeoTIFF (*.tif *.tiff)")
        if path:
            self.dem_path.setText(path)
            self.terrain.setCurrentIndex(1)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Run folder", self.output.text())
        if folder:
            self.output.setText(folder)

    def request(self) -> RunRequest:
        """Form contents as a :class:`RunRequest`."""
        dem = self.terrain.currentData() == "dem"
        preset = self.preset.currentData()
        return RunRequest(
            config=None if preset in (None, "__browse__") else preset,
            output_dir=self.output.text().strip(),
            dem_path=self.dem_path.text().strip() if dem else None,
            dem_max_size=self.dem_size.value() if dem else None,
            crater_radius_m=None if dem else self.radius.value() * 1000.0,
            grid_resolution_m=None if dem else self.resolution.value(),
            duration_hours=self.duration.value(),
            dt_s=self.dt.value(),
            output_interval_s=self.interval.value() * 3600.0,
            start_utc=self.start.text().strip() or None,
            ephemeris_mode=self.ephemeris.currentData(),
            point_source=bool(self.sun_model.currentData()),
            disk_samples=self.samples.value(),
            spinup_cycles=self.spinup.value(),
            view_factors=self.view_factors.isChecked(),
        )

    # -- estimate ---------------------------------------------------------

    def _rate(self) -> float:
        """Measured face-steps per second on this machine (or a modest default)."""
        value = self._settings.value("perf/face_steps_per_s", 0.0)
        try:
            return float(str(value)) or 1.2e6
        except (TypeError, ValueError):
            return 1.2e6

    def _update_estimate(self) -> None:
        req = self.request()
        cfg = build_config(replace(req, config=req.config))
        if req.dem_path:
            side = req.dem_max_size or 600
            faces = faces_for_dem_shape(side, side)
            approx = "≤ "
        else:
            faces = faces_for_dem_shape(*synthetic_grid_shape(cfg.synthetic_dem))
            approx = ""
        steps = int(round(req.duration_hours * 3600.0 / (req.dt_s or cfg.solver.dt_s)))
        steps += (req.spinup_cycles or 0) * round(cfg.lunar.synodic_period_s / (req.dt_s or 300.0))
        cfg = replace(cfg, view_factors=replace(cfg.view_factors, enabled=req.view_factors))
        memory = estimate_run_memory_mb(faces, cfg)["total"]
        samples = 1 if req.point_source else (req.disk_samples or 16)
        seconds = faces * steps / self._rate() * (0.6 + 0.4 * samples / 16.0)
        self._faces, self._steps = faces, steps
        e = self.estimate.items
        e["faces"].set_text(approx + f"{faces:,}".replace(",", " "))
        e["steps"].set_value(float(steps))
        e["memory"].set_value(memory)
        e["time"].set_text(_duration_text(seconds))
        limit = cfg.resources.max_memory_mb
        if memory > limit:
            self.limit_note.setProperty("role", "error")
            self.limit_note.setText(
                f"Above the {limit:.0f} MB limit in resources.max_memory_mb. "
                "Use a coarser cell size or turn off terrain infrared."
            )
        else:
            self.limit_note.setProperty("role", "caption")
            self.limit_note.setText(f"Limit {limit:.0f} MB (resources.max_memory_mb).")
        self.limit_note.style().unpolish(self.limit_note)
        self.limit_note.style().polish(self.limit_note)

    # -- running ----------------------------------------------------------

    def start_run(self) -> None:
        """Validate the form and launch the worker."""
        req = self.request()
        if not req.output_dir:
            self.status.emit("Choose a run folder first.")
            return
        if req.dem_path and not Path(req.dem_path).is_file():
            self.status.emit(f"DEM not found: {req.dem_path}")
            return
        try:
            datetime.fromisoformat((req.start_utc or "").replace("Z", "+00:00"))
        except ValueError:
            self.status.emit("Start time must be ISO-8601, e.g. 2025-01-01T00:00:00Z.")
            return
        self.log.clear()
        self.progress.set_fraction(0.0)
        self.progress_text.setText("Starting worker, compiling kernels on first use…")
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._started = time.monotonic()
        self.controller.start(req)
        self.status.emit("Run started.")

    def _on_progress(self, p) -> None:
        self.progress.set_fraction(p.fraction)
        elapsed = p.elapsed_s
        remaining = elapsed / max(p.fraction, 1e-6) - elapsed if p.fraction > 0.02 else None
        eta = f" · about {_duration_text(remaining)} left" if remaining is not None else ""
        self.progress_text.setText(
            f"{p.step:,} / {p.total_steps:,} steps ({p.phase})".replace(",", " ")
            + f"{eta}\nSun {p.sun_elevation_deg:+.2f}°  ·  surface "
            f"{p.surface_min_K:.1f}–{p.surface_max_K:.1f} K"
        )

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _done(self) -> None:
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_finished(self, folder: str) -> None:
        self._done()
        wall = time.monotonic() - self._started
        if wall > 2.0 and self._faces and self._steps:
            self._settings.setValue("perf/face_steps_per_s", self._faces * self._steps / wall)
        self.progress.set_fraction(1.0)
        self.progress_text.setText(f"Finished in {_duration_text(wall)}.")
        self.status.emit(f"Run written to {folder}")
        self._update_estimate()
        self.run_finished.emit(folder)

    def _on_failed(self, message: str, details: str) -> None:
        self._done()
        self.progress_text.setText(message)
        if details:
            self.log.appendPlainText(details)
        self.status.emit("Run failed: " + message)

    def _on_cancelled(self, message: str) -> None:
        self._done()
        self.progress_text.setText("Cancelled. " + message)
        self.status.emit("Run cancelled.")


def _duration_text(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"
