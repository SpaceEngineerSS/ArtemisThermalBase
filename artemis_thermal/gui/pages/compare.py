"""Compare page: difference of one quantity between two runs on the same grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout, QWidget

from artemis_thermal import style
from artemis_thermal.gui import run_data
from artemis_thermal.gui.mapview import Layer, MapView
from artemis_thermal.gui.tasks import Background
from artemis_thermal.gui.theme import Theme, lut
from artemis_thermal.gui.widgets import (
    FigureCaption,
    ReadoutGrid,
    Segmented,
    button,
    label,
    rule,
    section,
)

QUANTITIES = (
    ("surface_T_max", "Max T"),
    ("surface_T_mean", "Mean T"),
    ("surface_T_final", "Last-step T"),
    ("sunlit_fraction", "Sunlit time"),
)


class ComparePage(QWidget):
    """Pick run A and run B; see B − A."""

    status = Signal(str)

    def __init__(self, theme: Theme, background: Background, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Page")
        self._theme = theme
        self._bg = background
        self._paths: dict[str, str | None] = {"a": None, "b": None}
        self._quantity = QUANTITIES[0][0]

        body = QVBoxLayout(self)
        body.setContentsMargins(32, 26, 32, 28)
        body.setSpacing(14)
        body.addWidget(section(
            "03  Compare", "Difference between two runs",
            "Both runs must share the same terrain grid, for example the same DEM with a "
            "different Sun model, time step or physics switch.",
        ))

        pick = QHBoxLayout()
        pick.setSpacing(24)
        self._labels: dict[str, object] = {}
        for key, title in (("a", "Run A (reference)"), ("b", "Run B")):
            box = QVBoxLayout()
            box.addWidget(label(title.upper(), "kicker"))
            name = label("not chosen", "mono")
            self._labels[key] = name
            box.addWidget(name)
            box.addWidget(button("Choose…", "link", lambda _=False, k=key: self._choose(k)))
            pick.addLayout(box)
        pick.addStretch(1)
        body.addLayout(pick)

        self._tabs = Segmented(QUANTITIES)
        self._tabs.changed.connect(self._set_quantity)
        body.addWidget(self._tabs)

        columns = QHBoxLayout()
        columns.setSpacing(28)
        left = QVBoxLayout()
        self._map = MapView(theme)
        self._map.setMinimumHeight(380)
        self._map.hovered.connect(self._hover)
        left.addWidget(self._map, 1)
        self._caption = FigureCaption(3, "Choose two runs.")
        left.addWidget(self._caption)
        columns.addLayout(left, 1)

        side = QWidget()
        side.setFixedWidth(260)
        right = QVBoxLayout(side)
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(label("B − A", "kicker"))
        self._stats = ReadoutGrid(1)
        self._stats.add("mean", "mean difference", "", 2)
        self._stats.add("rms", "RMS difference", "", 2)
        self._stats.add("max", "largest |difference|", "", 2)
        self._stats.add("changed", "cells changed by > 1 unit", "%", 1)
        right.addWidget(self._stats)
        right.addWidget(rule())
        self._note = label("", "caption")
        self._note.setWordWrap(True)
        right.addWidget(self._note)
        right.addStretch(1)
        columns.addWidget(side, 0, Qt.AlignmentFlag.AlignTop)
        body.addLayout(columns, 1)

    def apply_theme(self, theme: Theme) -> None:
        """Restyle."""
        self._theme = theme
        self._map.apply_theme(theme)
        self._refresh()

    def set_run(self, key: str, folder: str) -> None:
        """Set run A or B programmatically."""
        self._paths[key] = folder
        self._labels[key].setText(Path(folder).name)  # type: ignore[attr-defined]
        self._refresh()

    def _choose(self, key: str) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Run folder", str(Path.cwd()))
        if folder:
            self.set_run(key, folder)

    def _set_quantity(self, key: str) -> None:
        self._quantity = key
        self._refresh()

    def _refresh(self) -> None:
        a, b = self._paths["a"], self._paths["b"]
        if not a or not b:
            return
        quantity = self._quantity

        def compute():
            run_a = run_data.open_bundle(a)
            run_b = run_data.open_bundle(b)
            if run_a.grid_shape != run_b.grid_shape:
                raise ValueError(
                    f"Grids differ: {run_a.grid_shape} vs {run_b.grid_shape} cells."
                )
            if not (run_a.has(quantity) and run_b.has(quantity)):
                raise ValueError(f"{quantity} is missing in one of the runs.")
            diff = run_b.raster(quantity) - run_a.raster(quantity)
            if quantity == "sunlit_fraction":
                diff = diff * 100.0
            return run_a, diff

        self._bg.submit("compare", compute, self._show, self._error)

    def _error(self, message: str) -> None:
        first = message.split("\n", 1)[0]
        self._note.setText(first)
        self._caption.set_text(first)

    def _show(self, result) -> None:
        run_a, diff = result
        unit = "%" if self._quantity == "sunlit_fraction" else "K"
        span = float(np.nanpercentile(np.abs(diff), 99.5)) or 1.0
        self._map.set_extent(run_a.extent)
        self._map.set_layer(Layer("diff", "difference", diff, lut(style.DIVERGING_STOPS),
                                  (-span, span), unit, 2))
        name = dict(QUANTITIES)[self._quantity]
        self._caption.set_text(f"{name}: run B minus run A. Orange is warmer or longer lit in B.")
        s = self._stats.items
        s["mean"].set_value(float(np.nanmean(diff)))
        s["rms"].set_value(float(np.sqrt(np.nanmean(diff * diff))))
        s["max"].set_value(float(np.nanmax(np.abs(diff))))
        s["changed"].set_value(100.0 * float(np.mean(np.abs(diff) > 1.0)))
        for key, item in s.items():
            item.set_unit("%" if key == "changed" else unit)
        self._note.setText(f"Both runs: {run_a.grid_shape[0]} × {run_a.grid_shape[1]} cells.")

    def _hover(self, x_m: float, y_m: float) -> None:
        value = self._map.value_at(x_m, y_m)
        if value is not None:
            self.status.emit(f"x {x_m / 1e3:.2f} km  y {y_m / 1e3:.2f} km  B − A = {value:+.2f}")
