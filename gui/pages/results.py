"""Results page: one run folder as maps, numbers and a timeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from artemis_thermal import style
from gui import run_data
from gui.mapview import Layer, MapView
from gui.plots import restyle, series_pens, styled_plot
from gui.tasks import Background
from gui.theme import Theme, lut
from gui.widgets import (
    FigureCaption,
    ReadoutGrid,
    Segmented,
    button,
    format_number,
    label,
    rule,
    section,
)
from simulation.io_manager import ResultsBundle, ResultsError

ICE_NAMES = ("stable", "marginal", "not stable")


class ResultsPage(QWidget):
    """Shows one run. Empty until :meth:`open_run` is called."""

    status = Signal(str)

    def __init__(self, theme: Theme, background: Background, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Page")
        self._theme = theme
        self._bg = background
        self._bundle: ResultsBundle | None = None
        self._layer_key = "temperature"
        self._layers: dict[str, run_data.LayerSpec] = {}
        self._shade: np.ndarray | None = None
        self._frame: int | None = None
        self._site: run_data.SiteReport | None = None
        self._play = QTimer(self)
        self._play.setInterval(125)
        self._play.timeout.connect(self._advance_frame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._stack = QStackedWidget()
        outer.addWidget(self._stack)

        # Empty state ------------------------------------------------------
        empty = QWidget()
        empty.setObjectName("Page")
        ev = QVBoxLayout(empty)
        ev.setContentsMargins(40, 36, 40, 36)
        ev.addWidget(section("02  Results", "No run open",
                             "Run a simulation from the Run page, or open an existing run "
                             "folder."))
        ev.addSpacing(18)
        row = QHBoxLayout()
        row.addWidget(button("Open run folder…", "primary", self.browse))
        row.addStretch(1)
        ev.addLayout(row)
        self._empty_note = label("", "error")
        self._empty_note.setWordWrap(True)
        ev.addWidget(self._empty_note)
        ev.addStretch(1)
        self._stack.addWidget(empty)

        # Content ----------------------------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        content.setObjectName("Page")
        scroll.setWidget(content)
        self._stack.addWidget(scroll)
        body = QVBoxLayout(content)
        body.setContentsMargins(32, 26, 32, 28)
        body.setSpacing(14)

        head = QHBoxLayout()
        self._heading = QVBoxLayout()
        self._kicker = label("02  RESULTS", "kicker")
        self._title = label("", "title")
        self._subtitle = label("", "subtitle")
        self._subtitle.setWordWrap(True)
        for w in (self._kicker, self._title, self._subtitle):
            self._heading.addWidget(w)
        head.addLayout(self._heading, 1)
        actions = QVBoxLayout()
        actions.setSpacing(2)
        actions.addWidget(button("Open another run…", "link", self.browse))
        actions.addWidget(button("Show folder", "link", self._show_folder))
        actions.addWidget(button("Export figures", "link", self._export_figures))
        actions.addStretch(1)
        head.addLayout(actions)
        body.addLayout(head)

        self._tabs = Segmented(run_data.LAYER_KEYS)
        self._tabs.changed.connect(self._select_layer)
        body.addWidget(self._tabs)

        columns = QHBoxLayout()
        columns.setSpacing(28)
        body.addLayout(columns, 1)

        left = QVBoxLayout()
        left.setSpacing(6)
        self._map = MapView(theme)
        self._map.setMinimumHeight(420)
        self._map.hovered.connect(self._on_hover)
        self._hover = label("", "mono")
        self._map.left.connect(lambda: self._hover.setText(""))
        self._map.picked.connect(self._pick)
        left.addWidget(self._map, 1)
        cap_row = QHBoxLayout()
        self._caption = FigureCaption(1)
        cap_row.addWidget(self._caption, 1)
        cap_row.addWidget(self._hover, 0, Qt.AlignmentFlag.AlignTop)
        left.addLayout(cap_row)

        scrub = QHBoxLayout()
        scrub.setSpacing(10)
        self._play_btn = button("Play", None, self._toggle_play)
        self._play_btn.setFixedWidth(64)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.valueChanged.connect(self._show_frame)
        self._time = label("", "mono")
        self._time.setMinimumWidth(110)
        scrub.addWidget(self._play_btn)
        scrub.addWidget(self._slider, 1)
        scrub.addWidget(self._time)
        self._scrub_box = QWidget()
        self._scrub_box.setLayout(scrub)
        left.addWidget(self._scrub_box)
        columns.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(10)
        side = QWidget()
        side.setFixedWidth(300)
        side.setLayout(right)
        right.addWidget(label("RUN SUMMARY", "kicker"))
        self._summary = ReadoutGrid(2)
        self._summary.add("t_min", "lowest T", "K", 1)
        self._summary.add("t_max", "highest T", "K", 1)
        self._summary.add("t_mean", "mean T, area-weighted", "K", 1)
        self._summary.add("area", "terrain area", "km²", 1)
        self._summary.add("never_lit", "never lit", "km²", 2)
        self._summary.add("sunlit", "lit ≥ half the time", "km²", 2)
        self._summary.add("stable", "ice stable", "km²", 2)
        self._summary.add("marginal", "ice marginal", "km²", 2)
        right.addWidget(self._summary)
        right.addSpacing(6)
        right.addWidget(rule())
        right.addSpacing(4)
        right.addWidget(label("SITE", "kicker"))
        self._site_hint = label("Click the map to inspect one cell.", "caption")
        self._site_hint.setWordWrap(True)
        right.addWidget(self._site_hint)
        self._site_grid = ReadoutGrid(2)
        for key, cap, unit, dec in (
            ("elev", "elevation", "m", 0), ("slope", "slope", "°", 1),
            ("aspect", "faces (grid az.)", "°", 0), ("t_now", "T, last step", "K", 1),
            ("t_min", "T min", "K", 1), ("t_max", "T max", "K", 1),
            ("sunlit", "sunlit", "%", 0), ("flux", "tracking panel, mean", "W/m²", 0),
            ("dark", "longest dark spell", "h", 1), ("ice", "ice", "", 0),
        ):
            self._site_grid.add(key, cap, unit, dec)
        self._site_grid.setVisible(False)
        right.addWidget(self._site_grid)
        self._site_plot = styled_plot(theme, 120)
        self._site_plot.setVisible(False)
        right.addWidget(self._site_plot)
        self._copy_site = button("Copy site data as CSV", "link", self._copy_site_csv)
        self._copy_site.setVisible(False)
        right.addWidget(self._copy_site)
        right.addStretch(1)
        columns.addWidget(side, 0, Qt.AlignmentFlag.AlignTop)

        body.addWidget(rule())
        self._timeline = styled_plot(theme, 190)
        self._timeline.getPlotItem().setLabel("bottom", "hours since start")
        self._timeline.getPlotItem().setLabel("left", "surface T [K]")
        self._legend = self._timeline.getPlotItem().addLegend(offset=(-10, 6), labelTextSize="8pt")
        body.addWidget(self._timeline)
        self._timeline_caption = FigureCaption(2, "Probe temperatures (lowest, highest and "
                                                  "mid-height faces) and Sun elevation.")
        body.addWidget(self._timeline_caption)
        self._sun_plot = styled_plot(theme, 80)
        self._sun_plot.getPlotItem().setLabel("left", "Sun el. [°]")
        self._sun_plot.setXLink(self._timeline)
        body.addWidget(self._sun_plot)

        self._stack.setCurrentIndex(0)
        self._map.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # -- public ------------------------------------------------------------

    @property
    def bundle(self) -> ResultsBundle | None:
        """The open run, if any."""
        return self._bundle

    def apply_theme(self, theme: Theme) -> None:
        """Restyle after a theme switch."""
        self._theme = theme
        self._map.apply_theme(theme)
        for plot in (self._timeline, self._sun_plot, self._site_plot):
            restyle(plot, theme)
        if self._bundle is not None:
            self._layers.clear()
            self._render_layer(animate=False)
            self._draw_timeline()
            if self._site is not None:
                self._show_site(self._site)

    def browse(self) -> None:
        """Pick a run folder."""
        folder = QFileDialog.getExistingDirectory(self, "Open run folder", str(Path.cwd()))
        if folder:
            self.open_run(folder)

    def open_run(self, folder: str) -> None:
        """Load a run folder in the background, then show it."""
        self.status.emit(f"Opening {folder}")

        def load():
            bundle = run_data.open_bundle(folder)
            summary = run_data.summarize(bundle)
            shade = run_data.hillshade(bundle)
            return bundle, summary, shade

        self._bg.submit("results", load, self._on_loaded, self._on_load_error)

    # -- loading -----------------------------------------------------------

    def _on_load_error(self, message: str) -> None:
        first = message.split("\n", 1)[0]
        self._empty_note.setText(first)
        self._stack.setCurrentIndex(0)
        self.status.emit(first)

    def _on_loaded(self, result) -> None:
        bundle, summary, shade = result
        self._bundle = bundle
        self._layers.clear()
        self._shade = shade
        self._site = None
        self._frame = None
        self._play.stop()
        self._play_btn.setText("Play")
        self._map.set_marker(None, None)
        self._site_grid.setVisible(False)
        self._site_plot.setVisible(False)
        self._copy_site.setVisible(False)
        self._site_hint.setVisible(True)

        self._title.setText(bundle.path.name)
        self._subtitle.setText(run_data.describe(bundle))
        self._map.set_extent(bundle.extent)
        self._map.set_relief(shade, lut(style.RELIEF_STOPS))
        s = summary
        values = {"t_min": s.t_min, "t_max": s.t_max, "t_mean": s.t_mean_area,
                  "area": s.area_km2, "never_lit": s.never_lit_km2,
                  "sunlit": s.sunlit_half_km2, "stable": s.stable_km2,
                  "marginal": s.marginal_km2}
        for key, value in values.items():
            self._summary.items[key].set_value(value)

        frames = bundle.snapshot_count
        self._scrub_box.setVisible(frames > 1)
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(frames - 1, 0))
        self._slider.setValue(max(frames - 1, 0))
        self._slider.blockSignals(False)
        self._update_time_label(max(frames - 1, 0))

        self._stack.setCurrentIndex(1)
        self._render_layer(animate=True)
        self._draw_timeline()
        self._map.reset_view()
        self.status.emit(f"{bundle.path.name}: {bundle.num_faces:,} faces".replace(",", " "))

    # -- layers ------------------------------------------------------------

    def _spec(self, key: str) -> run_data.LayerSpec:
        if key not in self._layers:
            assert self._bundle is not None
            self._layers[key] = run_data.build_layer(self._bundle, key, self._theme.pal)
        return self._layers[key]

    def _select_layer(self, key: str) -> None:
        self._layer_key = key
        self._frame = None
        if self._bundle is not None and self._bundle.snapshot_count > 1:
            self._slider.blockSignals(True)
            self._slider.setValue(self._slider.maximum())
            self._slider.blockSignals(False)
            self._update_time_label(self._slider.maximum())
        self._render_layer(animate=True)

    def _render_layer(self, animate: bool) -> None:
        if self._bundle is None:
            return
        spec = self._spec(self._layer_key)
        layer = Layer(spec.key, spec.title, spec.data, lut(spec.stops, len(spec.stops)
                      if spec.ticks else 256), spec.levels, spec.unit, spec.decimals, spec.ticks)
        self._map.set_layer(layer, opacity=0.78 if spec.over_relief else 1.0, animate=animate)
        self._caption.set_text(spec.title + ".")
        self._scrub_box.setEnabled(self._layer_key == "temperature")

    # -- time scrubbing ---------------------------------------------------

    def _update_time_label(self, frame: int) -> None:
        if self._bundle is None or not self._bundle.snapshot_count:
            self._time.setText("")
            return
        stride = int(self._bundle.metadata.get("snapshot_stride", 1))
        hours = self._bundle.timeline["hours"]
        index = min(frame * stride, len(hours) - 1) if hours.size else frame
        value = hours[index] if hours.size else float(frame)
        self._time.setText(f"t = {value:7.1f} h")

    def _show_frame(self, frame: int) -> None:
        if self._bundle is None or self._layer_key != "temperature":
            return
        self._update_time_label(frame)
        data = self._bundle.raster("snapshots_surface_T", frame)
        self._map.update_data(data)
        self._caption.set_text("Surface temperature, " + self._time.text().strip() + ".")

    def _toggle_play(self) -> None:
        if self._play.isActive():
            self._play.stop()
            self._play_btn.setText("Play")
        else:
            if self._slider.value() >= self._slider.maximum():
                self._slider.setValue(0)
            self._play.start()
            self._play_btn.setText("Pause")

    def _advance_frame(self) -> None:
        if self._slider.value() >= self._slider.maximum():
            self._toggle_play()
            return
        self._slider.setValue(self._slider.value() + 1)

    # -- hover and site ---------------------------------------------------

    def _on_hover(self, x_m: float, y_m: float) -> None:
        value = self._map.value_at(x_m, y_m)
        spec = self._layers.get(self._layer_key)
        text = f"x {x_m / 1e3:7.2f} km   y {y_m / 1e3:7.2f} km"
        if value is not None and spec is not None:
            if spec.ticks:
                text += f"   {ICE_NAMES[int(round(value))]}"
            else:
                text += f"   {format_number(value, max(spec.decimals, 1))} {spec.unit}"
        self._hover.setText(text)

    def _pick(self, x_m: float, y_m: float) -> None:
        if self._bundle is None:
            return
        self._map.set_marker(x_m, y_m)
        bundle = self._bundle
        self._bg.submit("site", lambda: run_data.site_report(bundle, x_m, y_m), self._show_site)

    def _show_site(self, report: run_data.SiteReport) -> None:
        self._site = report
        g = self._site_grid.items
        g["elev"].set_value(report.elevation_m)
        g["slope"].set_value(report.slope_deg)
        g["aspect"].set_value(report.aspect_deg)
        g["t_now"].set_value(report.t_final)
        g["t_min"].set_value(report.t_min)
        g["t_max"].set_value(report.t_max)
        g["sunlit"].set_value(report.sunlit_pct)
        g["flux"].set_value(report.tracking_flux)
        g["dark"].set_value(report.longest_dark_h)
        g["ice"].set_text(ICE_NAMES[report.ice_class])
        self._site_hint.setText(
            f"Cell at x {report.x_m / 1e3:.2f} km, y {report.y_m / 1e3:.2f} km."
        )
        self._site_grid.setVisible(True)
        self._copy_site.setVisible(True)
        plot = self._site_plot
        plot.clear()
        if report.series.size > 1:
            plot.plot(report.hours, report.series, pen=series_pens(self._theme)[1])
            plot.setVisible(True)
        else:
            plot.setVisible(False)

    def _copy_site_csv(self) -> None:
        report = self._site
        if report is None:
            return
        rows = [
            ("x_m", report.x_m), ("y_m", report.y_m), ("elevation_m", report.elevation_m),
            ("slope_deg", report.slope_deg), ("aspect_deg", report.aspect_deg),
            ("t_last_K", report.t_final), ("t_min_K", report.t_min),
            ("t_max_K", report.t_max), ("t_mean_K", report.t_mean),
            ("sunlit_pct", report.sunlit_pct), ("tracking_flux_W_m2", report.tracking_flux),
            ("longest_dark_h", report.longest_dark_h), ("ice", ICE_NAMES[report.ice_class]),
        ]
        lines = ["quantity,value"] + [f"{k},{v:.6g}" if isinstance(v, float) else f"{k},{v}"
                                      for k, v in rows]
        if report.series.size:
            lines += ["", "hours,surface_T_K"] + [
                f"{h:.4f},{t:.4f}" for h, t in zip(report.hours, report.series, strict=False)
            ]
        QGuiApplication.clipboard().setText("\n".join(lines) + "\n")
        self.status.emit("Site data copied to the clipboard.")

    # -- timeline ---------------------------------------------------------

    def _draw_timeline(self) -> None:
        bundle = self._bundle
        if bundle is None:
            return
        plot = self._timeline.getPlotItem()
        plot.clear()
        if self._legend is not None:
            self._legend.clear()
        dt_h = float(bundle.metadata.get("dt_s", 300.0)) / 3600.0
        for (name, (_face, series)), pen in zip(
            bundle.probes.items(), series_pens(self._theme), strict=False
        ):
            hours = (np.arange(series.size) + 1) * dt_h
            plot.plot(hours, series, pen=pen, name=name.replace("_", " "))
        sun = self._sun_plot.getPlotItem()
        sun.clear()
        timeline = bundle.timeline
        if timeline["hours"].size:
            sun.plot(timeline["hours"], timeline["sun_elevation_deg"],
                     pen=pg.mkPen(self._theme.pal.accent_ink, width=1.3))
            sun.addLine(y=0.0, pen=pg.mkPen(self._theme.pal.rule_strong, width=1))

    # -- actions ----------------------------------------------------------

    def _show_folder(self) -> None:
        if self._bundle is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._bundle.path)))

    def _export_figures(self) -> None:
        bundle = self._bundle
        if bundle is None:
            return
        target = QFileDialog.getExistingDirectory(self, "Export figures to", str(bundle.path))
        if not target:
            return
        self.status.emit("Rendering figures…")

        def render():
            from visualization.figures import render_run_figures

            return render_run_figures(bundle, target)

        self._bg.submit("export", render,
                        lambda paths: self.status.emit(f"Wrote {len(paths)} figures to {target}"),
                        lambda msg: self.status.emit(msg.split("\n", 1)[0]))

    def export_site_csv(self, path: str) -> None:
        """Write the current site report to ``path`` (used by tests)."""
        report = self._site
        if report is None:
            raise ResultsError("No site selected.")
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["hours", "surface_T_K"])
            writer.writerows(zip(report.hours, report.series, strict=False))
