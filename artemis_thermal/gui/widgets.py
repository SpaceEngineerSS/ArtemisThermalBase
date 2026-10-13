"""Small building blocks shared by the pages."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from artemis_thermal.gui import motion


def label(text: str = "", role: str | None = None, parent: QWidget | None = None) -> QLabel:
    """QLabel with a style role (see ``theme.style_sheet``)."""
    widget = QLabel(text, parent)
    if role:
        widget.setProperty("role", role)
    return widget


def rule(vertical: bool = False) -> QFrame:
    """A 1 px hairline."""
    line = QFrame()
    line.setProperty("role", "vrule" if vertical else "rule")
    return line


def button(text: str, kind: str | None = None, slot: Callable | None = None) -> QPushButton:
    """Push button with a style kind: ``primary``, ``link``, ``rail``, ``segment``."""
    widget = QPushButton(text)
    if kind:
        widget.setProperty("kind", kind)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    if slot is not None:
        widget.clicked.connect(slot)
    return widget


def format_number(value: float, decimals: int) -> str:
    """Thin-space thousands separators, fixed decimals, en dash for NaN."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ").replace("-", "−")


class Readout(QWidget):
    """A labelled number with a unit; changes glide to the new value."""

    def __init__(self, caption: str, unit: str = "", decimals: int = 1,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._decimals = decimals
        self._value: float | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        self._caption = label(caption, "label")
        row = QHBoxLayout()
        row.setSpacing(4)
        self._number = label("–", "value")
        self._unit = label(unit, "unit")
        row.addWidget(self._number)
        row.addWidget(self._unit, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        layout.addWidget(self._caption)
        layout.addLayout(row)

    def set_value(self, value: float | None, animate: bool = True) -> None:
        """Show ``value``; ``None`` shows an en dash."""
        if value is None or not math.isfinite(value):
            self._value = None
            self._number.setText("–")
            return
        start = self._value if (animate and self._value is not None) else value
        self._value = float(value)
        motion.tween(self, start, float(value),
                     lambda v: self._number.setText(format_number(v, self._decimals)))

    def set_unit(self, unit: str) -> None:
        """Change the unit label."""
        self._unit.setText(unit)

    def set_text(self, text: str) -> None:
        """Show free text instead of a number."""
        self._value = None
        self._number.setText(text)


class ReadoutGrid(QWidget):
    """Readouts laid out in a fixed number of columns."""

    def __init__(self, columns: int = 2, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(18)
        self._grid.setVerticalSpacing(12)
        self._columns = columns
        self.items: dict[str, Readout] = {}

    def add(self, key: str, caption: str, unit: str = "", decimals: int = 1) -> Readout:
        """Append a readout and return it."""
        index = len(self.items)
        item = Readout(caption, unit, decimals)
        self._grid.addWidget(item, index // self._columns, index % self._columns)
        self.items[key] = item
        return item


class Segmented(QWidget):
    """Exclusive text tabs with an accent underline."""

    changed = Signal(str)

    def __init__(self, options: Sequence[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        for key, text in options:
            btn = button(text, "segment")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            self._group.addButton(btn)
            layout.addWidget(btn)
            self._buttons[key] = btn
        layout.addStretch(1)
        if options:
            self._buttons[options[0][0]].setChecked(True)

    def select(self, key: str, emit: bool = False) -> None:
        """Select ``key`` programmatically."""
        if key in self._buttons:
            self._buttons[key].setChecked(True)
            if emit:
                self.changed.emit(key)

    def set_enabled(self, key: str, enabled: bool) -> None:
        """Enable or disable one option."""
        if key in self._buttons:
            self._buttons[key].setEnabled(enabled)

    def current(self) -> str:
        """Key of the checked option."""
        for key, btn in self._buttons.items():
            if btn.isChecked():
                return key
        return ""


class ProgressLine(QWidget):
    """A 2 px progress rule that glides between values."""

    def __init__(self, track: QColor, fill: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._track = track
        self._fill = fill
        self._value = 0.0
        self.setFixedHeight(2)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_colors(self, track: QColor, fill: QColor) -> None:
        """Recolour after a theme change."""
        self._track, self._fill = track, fill
        self.update()

    def set_fraction(self, value: float) -> None:
        """Move to ``value`` in [0, 1]."""
        value = min(max(value, 0.0), 1.0)
        motion.tween(self, self._value, value, self._apply, duration=motion.FAST_MS)

    def _apply(self, value: float) -> None:
        self._value = value
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: D102
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._track)
        width = self.width() * self._value
        painter.fillRect(QRectF(0, 0, width, self.height()), self._fill)
        painter.end()


class FigureCaption(QWidget):
    """'Fig. n' in accent followed by a caption, like a printed report."""

    def __init__(self, number: int, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)
        self._number = label(f"Fig. {number}", "figno")
        self._text = label(text, "caption")
        self._text.setWordWrap(True)
        layout.addWidget(self._number, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._text, 1)

    def set_text(self, text: str) -> None:
        """Replace the caption text."""
        self._text.setText(text)


def section(kicker: str, title: str, subtitle: str = "") -> QWidget:
    """Page heading: small accent kicker, a title, an optional subtitle."""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)
    layout.addWidget(label(kicker.upper(), "kicker"))
    layout.addWidget(label(title, "title"))
    if subtitle:
        sub = label(subtitle, "subtitle")
        sub.setWordWrap(True)
        layout.addWidget(sub)
    return box
