"""Raster map with km axes, a colour scale, hover readout and click-to-pick."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from artemis_thermal.gui import motion
from artemis_thermal.gui.theme import Theme
from artemis_thermal.gui.widgets import format_number


@dataclass
class Layer:
    """One map layer: a north-up raster and how to colour it."""

    key: str
    title: str
    data: np.ndarray
    lut: np.ndarray
    levels: tuple[float, float]
    unit: str
    decimals: int = 1
    ticks: tuple[tuple[float, str], ...] = ()


class ColorScale(QWidget):
    """Horizontal colour strip with end values and unit."""

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._layer: Layer | None = None
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_theme(self, theme: Theme) -> None:
        """Recolour."""
        self._theme = theme
        self.update()

    def set_layer(self, layer: Layer | None) -> None:
        """Show the scale of ``layer``."""
        self._layer = layer
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: D102
        layer = self._layer
        if layer is None:
            return
        t = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        strip = QRectF(0, 2, self.width() - 1, 7)
        if layer.ticks:
            n = len(layer.lut)
            w = strip.width() / n
            for i, rgb in enumerate(layer.lut):
                painter.fillRect(QRectF(strip.left() + i * w, strip.top(), w, strip.height()),
                                 QColor(*rgb))
        else:
            grad = QLinearGradient(strip.topLeft(), strip.topRight())
            for i in range(0, len(layer.lut), 16):
                grad.setColorAt(i / (len(layer.lut) - 1), QColor(*layer.lut[i]))
            grad.setColorAt(1.0, QColor(*layer.lut[-1]))
            painter.fillRect(strip, grad)
        painter.setPen(QPen(t.color("rule_strong"), 1))
        painter.drawRect(strip)
        painter.setFont(t.mono(7.5))
        painter.setPen(t.color("ink_2"))
        y = 24
        if layer.ticks:
            n = len(layer.ticks)
            for i, (_value, text) in enumerate(layer.ticks):
                x = strip.left() + (i + 0.5) * strip.width() / n
                painter.drawText(QRectF(x - 80, y - 12, 160, 14), Qt.AlignmentFlag.AlignCenter,
                                 text)
        else:
            lo, hi = layer.levels
            painter.drawText(QRectF(0, y - 12, 120, 14), Qt.AlignmentFlag.AlignLeft,
                             format_number(lo, layer.decimals))
            painter.drawText(QRectF(self.width() - 121, y - 12, 120, 14),
                             Qt.AlignmentFlag.AlignRight,
                             f"{format_number(hi, layer.decimals)} {layer.unit}")
            mid = 0.5 * (lo + hi)
            painter.drawText(QRectF(self.width() / 2 - 60, y - 12, 120, 14),
                             Qt.AlignmentFlag.AlignCenter, format_number(mid, layer.decimals))
        painter.end()


class MapView(QWidget):
    """Interactive raster map in mesh coordinates (shown in km)."""

    hovered = Signal(float, float)   # metres
    picked = Signal(float, float)    # metres
    left = Signal()

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._extent_km = (0.0, 1.0, 0.0, 1.0)
        self._layer: Layer | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._view = pg.PlotWidget()
        self._view.setMinimumSize(280, 240)
        self._plot = self._view.getPlotItem()
        self._plot.setAspectLocked(True)
        self._plot.hideButtons()
        self._plot.setMenuEnabled(False)
        for side in ("top", "right"):
            self._plot.showAxis(side)
            self._plot.getAxis(side).setStyle(showValues=False)
        self._plot.setLabel("bottom", "x", units="")
        self._plot.setLabel("left", "y", units="")

        self._relief = pg.ImageItem(autoDownsample=True)
        self._image = pg.ImageItem(autoDownsample=True)
        self._plot.addItem(self._relief)
        self._plot.addItem(self._image)
        self._marker = pg.ScatterPlotItem(size=13, symbol="+", pxMode=True)
        self._plot.addItem(self._marker)

        self.scale = ColorScale(theme)
        layout.addWidget(self._view, 1)
        layout.addWidget(self.scale)

        self._hover_proxy = pg.SignalProxy(
            self._view.scene().sigMouseMoved, rateLimit=30, slot=self._on_move
        )
        self._view.scene().sigMouseClicked.connect(self._on_click)
        self.apply_theme(theme)

    # -- appearance --------------------------------------------------------

    def apply_theme(self, theme: Theme) -> None:
        """Restyle axes, background and marker."""
        self._theme = theme
        p = theme.pal
        self._view.setBackground(p.paper)
        axis_pen = pg.mkPen(p.rule_strong, width=1)
        text_pen = pg.mkPen(p.ink_2)
        for side in ("left", "bottom", "top", "right"):
            axis = self._plot.getAxis(side)
            axis.setPen(axis_pen)
            axis.setTextPen(text_pen)
            axis.setTickFont(theme.mono(7.5))
            axis.setStyle(tickLength=-5)
            axis.enableAutoSIPrefix(False)
        self._plot.getAxis("bottom").setLabel("x  [km]", color=p.ink_2)
        self._plot.getAxis("left").setLabel("y  [km]", color=p.ink_2)
        self._marker.setPen(pg.mkPen(p.accent, width=2))
        self._marker.setBrush(pg.mkBrush(p.accent))
        self.scale.set_theme(theme)

    # -- data -------------------------------------------------------------

    def set_extent(self, extent_m: tuple[float, float, float, float]) -> None:
        """Map extent (left, right, bottom, top) in metres."""
        left, right, bottom, top = extent_m
        self._extent_km = (left / 1e3, right / 1e3, bottom / 1e3, top / 1e3)
        rect = self._rect()
        self._image.setRect(rect)
        self._relief.setRect(rect)
        self._plot.setRange(rect, padding=0.02)

    def _rect(self) -> QRectF:
        left, right, bottom, top = self._extent_km
        return QRectF(left, bottom, right - left, top - bottom)

    def set_relief(self, shade: np.ndarray | None, lut: np.ndarray | None = None) -> None:
        """Grayscale hillshade drawn under translucent layers."""
        if shade is None:
            self._relief.clear()
            return
        self._relief.setImage(np.ascontiguousarray(shade[::-1]), levels=(0.0, 1.0), lut=lut)
        self._relief.setRect(self._rect())

    def set_layer(self, layer: Layer, opacity: float = 1.0, animate: bool = True) -> None:
        """Show ``layer``; fades in when the content changes."""
        changed = self._layer is None or self._layer.key != layer.key
        self._layer = layer
        self._image.setImage(np.ascontiguousarray(layer.data[::-1]), levels=layer.levels,
                             lut=layer.lut)
        self._image.setRect(self._rect())
        self.scale.set_layer(layer)
        if animate and changed:
            self._image.setOpacity(0.0)
            motion.tween(self, 0.0, opacity, self._image.setOpacity, duration=motion.BASE_MS)
        else:
            self._image.setOpacity(opacity)

    def update_data(self, data: np.ndarray) -> None:
        """Swap pixels of the current layer without resetting levels (scrubbing)."""
        if self._layer is None:
            return
        self._layer.data = data
        self._image.setImage(np.ascontiguousarray(data[::-1]), levels=self._layer.levels,
                             lut=self._layer.lut, autoLevels=False)
        self._image.setRect(self._rect())

    def set_marker(self, x_m: float | None, y_m: float | None) -> None:
        """Place (or clear) the site marker."""
        if x_m is None or y_m is None:
            self._marker.clear()
        else:
            self._marker.setData([x_m / 1e3], [y_m / 1e3])

    def value_at(self, x_m: float, y_m: float) -> float | None:
        """Layer value under a point, or None outside the map."""
        if self._layer is None:
            return None
        left, right, bottom, top = self._extent_km
        x, y = x_m / 1e3, y_m / 1e3
        if not (left <= x <= right and bottom <= y <= top):
            return None
        rows, cols = self._layer.data.shape
        col = min(int((x - left) / (right - left) * cols), cols - 1)
        row = min(int((top - y) / (top - bottom) * rows), rows - 1)
        return float(self._layer.data[row, col])

    def reset_view(self) -> None:
        """Zoom back to the full extent."""
        self._plot.setRange(self._rect(), padding=0.02)

    # -- events -----------------------------------------------------------

    def _scene_to_m(self, pos: QPointF) -> tuple[float, float] | None:
        if not self._plot.sceneBoundingRect().contains(pos):
            return None
        point = self._plot.vb.mapSceneToView(pos)
        left, right, bottom, top = self._extent_km
        if not (left <= point.x() <= right and bottom <= point.y() <= top):
            return None
        return point.x() * 1e3, point.y() * 1e3

    def _on_move(self, event) -> None:
        where = self._scene_to_m(event[0])
        if where is None:
            self.left.emit()
        else:
            self.hovered.emit(*where)

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if event.double():
            self.reset_view()
            return
        where = self._scene_to_m(event.scenePos())
        if where is not None:
            self.picked.emit(*where)
