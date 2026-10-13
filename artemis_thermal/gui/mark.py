"""The ArtemisThermalBase mark, drawn in code so it follows the theme.

A crater rim seen from above, a floor contour broken on the shadowed side,
the pole as a small tick just inside the rim, and the Sun as a short dash on
the horizon. ``artemis_thermal/assets/mark.svg`` is the same drawing.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget


def _rim() -> QPainterPath:
    path = QPainterPath(QPointF(4.2, 12.6))
    path.cubicTo(4.0, 7.9, 7.6, 4.3, 12.3, 4.2)
    path.cubicTo(17.0, 4.1, 20.1, 7.4, 19.9, 12.0)
    path.cubicTo(19.7, 16.6, 16.3, 19.9, 11.8, 19.8)
    path.cubicTo(7.4, 19.7, 4.4, 17.0, 4.2, 12.6)
    return path


def _floor() -> QPainterPath:
    path = QPainterPath(QPointF(8.3, 14.6))
    path.cubicTo(7.5, 11.6, 9.4, 8.9, 12.4, 8.8)
    path.cubicTo(14.6, 8.7, 16.0, 10.0, 16.2, 11.6)
    return path


def paint_mark(painter: QPainter, rect: QRectF, ink: QColor, accent: QColor) -> None:
    """Draw the mark scaled into ``rect`` (designed on a 24 × 24 grid)."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    scale = min(rect.width(), rect.height()) / 24.0
    painter.translate(rect.center().x() - 12 * scale, rect.center().y() - 12 * scale)
    painter.scale(scale, scale)
    pen = QPen(ink, 1.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawPath(_rim())
    pen.setWidthF(1.0)
    painter.setPen(pen)
    painter.drawPath(_floor())
    accent_pen = QPen(accent, 1.2)
    accent_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    painter.setPen(accent_pen)
    painter.drawLine(QPointF(14.9, 6.9), QPointF(14.9, 9.7))
    painter.drawLine(QPointF(13.5, 8.3), QPointF(16.3, 8.3))
    painter.drawLine(QPointF(0.8, 16.9), QPointF(3.4, 16.4))
    painter.restore()


def mark_icon(ink: str = "#1D1F21", accent: str = "#E8641B", paper: str = "#F4F2EC") -> QIcon:
    """Window icon at the usual sizes."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pix = QPixmap(size, size)
        pix.fill(QColor(paper))
        painter = QPainter(pix)
        inset = size * 0.08
        paint_mark(painter, QRectF(inset, inset, size - 2 * inset, size - 2 * inset),
                   QColor(ink), QColor(accent))
        painter.end()
        icon.addPixmap(pix)
    return icon


class MarkWidget(QWidget):
    """The mark as a fixed-size widget for the header."""

    def __init__(self, ink: QColor, accent: QColor, size: int = 22, parent=None) -> None:
        super().__init__(parent)
        self._ink = ink
        self._accent = accent
        self.setFixedSize(QSize(size, size))

    def set_colors(self, ink: QColor, accent: QColor) -> None:
        """Recolour after a theme change."""
        self._ink, self._accent = ink, accent
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: D102
        painter = QPainter(self)
        paint_mark(painter, QRectF(self.rect()), self._ink, self._accent)
        painter.end()
