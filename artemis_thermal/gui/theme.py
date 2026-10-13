"""Turn the shared design tokens into a Qt style sheet and pyqtgraph settings.

The widgets stay plain Qt widgets; all of the look comes from here, so the
light and dark themes are one function call apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

from artemis_thermal import style

SANS = style.SANS
MONO = style.MONO


@dataclass(frozen=True)
class Theme:
    """Palette plus the derived Qt objects the widgets need."""

    dark: bool
    pal: style.Palette

    def color(self, name: str) -> QColor:
        """A palette colour as a QColor."""
        return QColor(getattr(self.pal, name))

    def mono(self, size: float = 9.0, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
        """Monospaced font for numbers."""
        font = QFont(MONO)
        font.setPointSizeF(size)
        font.setWeight(weight)
        font.setStyleHint(QFont.StyleHint.Monospace)
        return font

    def sans(self, size: float = 9.5, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
        """Text font."""
        font = QFont(SANS)
        font.setPointSizeF(size)
        font.setWeight(weight)
        return font


_fonts_loaded = False


def load_fonts() -> bool:
    """Register the bundled IBM Plex files with Qt. Returns True on success."""
    global _fonts_loaded
    if _fonts_loaded:
        return True
    ok = False
    for path in style.font_files():
        ok = QFontDatabase.addApplicationFont(str(path)) >= 0 or ok
    _fonts_loaded = ok
    return ok


def system_prefers_dark() -> bool:
    """True if the OS is in dark mode (Qt 6.5+); False when unknown."""
    hints = QGuiApplication.styleHints()
    scheme = getattr(hints, "colorScheme", None)
    if scheme is None:
        return False
    return scheme() == Qt.ColorScheme.Dark


def make_theme(dark: bool) -> Theme:
    """Theme for the requested mode."""
    return Theme(dark=dark, pal=style.palette(dark))


def style_sheet(t: Theme) -> str:
    """The application-wide QSS."""
    p = t.pal
    return f"""
* {{
    font-family: "{SANS}";
    font-size: 9.5pt;
    color: {p.ink};
    outline: 0;
}}
QMainWindow, QWidget#Page, QWidget#Root, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {p.paper};
}}
QWidget#Rail {{
    background: {p.paper};
    border-right: 1px solid {p.rule};
}}
QWidget#Header {{
    background: {p.paper};
    border-bottom: 1px solid {p.rule};
}}
QLabel {{ background: transparent; }}
QLabel[role="kicker"] {{
    color: {p.accent_ink};
    font-size: 8pt;
    font-weight: 500;
    letter-spacing: 1.2px;
}}
QLabel[role="title"] {{ font-size: 15pt; font-weight: 400; }}
QLabel[role="subtitle"] {{ color: {p.ink_2}; }}
QLabel[role="caption"] {{ color: {p.ink_2}; font-size: 8.5pt; }}
QLabel[role="figno"] {{ color: {p.accent_ink}; font-size: 8.5pt; font-weight: 500; }}
QLabel[role="label"] {{ color: {p.ink_3}; font-size: 8pt; }}
QLabel[role="value"] {{ font-family: "{MONO}"; font-size: 11pt; }}
QLabel[role="unit"] {{ color: {p.ink_3}; font-family: "{MONO}"; font-size: 8.5pt; }}
QLabel[role="mono"] {{ font-family: "{MONO}"; font-size: 9pt; color: {p.ink_2}; }}
QLabel[role="empty"] {{ color: {p.ink_3}; font-size: 10pt; }}
QLabel[role="error"] {{ color: {p.warn}; }}

QPushButton {{
    background: transparent;
    border: 1px solid {p.rule_strong};
    border-radius: 2px;
    padding: 5px 12px;
    color: {p.ink};
}}
QPushButton:hover {{ border-color: {p.ink_2}; }}
QPushButton:pressed {{ background: {p.sheet}; }}
QPushButton:disabled {{ color: {p.ink_3}; border-color: {p.rule}; }}
QPushButton[kind="primary"] {{
    background: {p.ink};
    color: {p.paper};
    border-color: {p.ink};
}}
QPushButton[kind="primary"]:hover {{ background: {p.accent_ink}; border-color: {p.accent_ink}; }}
QPushButton[kind="primary"]:disabled {{ background: {p.rule}; border-color: {p.rule}; color: {p.ink_3}; }}
QPushButton[kind="link"] {{
    border: none; padding: 2px 0; color: {p.ink_2}; text-align: left;
}}
QPushButton[kind="link"]:hover {{ color: {p.accent_ink}; }}

QPushButton[kind="rail"] {{
    border: none;
    border-left: 2px solid transparent;
    border-radius: 0;
    padding: 7px 14px 7px 16px;
    text-align: left;
    color: {p.ink_2};
}}
QPushButton[kind="rail"]:hover {{ color: {p.ink}; }}
QPushButton[kind="rail"]:checked {{
    color: {p.ink};
    border-left: 2px solid {p.accent};
    font-weight: 500;
}}
QPushButton[kind="segment"] {{
    border: none;
    border-bottom: 1px solid {p.rule};
    border-radius: 0;
    padding: 4px 10px 5px 10px;
    color: {p.ink_2};
}}
QPushButton[kind="segment"]:hover {{ color: {p.ink}; }}
QPushButton[kind="segment"]:checked {{
    color: {p.ink};
    border-bottom: 2px solid {p.accent};
    padding-bottom: 4px;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit {{
    background: {p.sheet};
    border: 1px solid {p.rule};
    border-radius: 2px;
    padding: 3px 6px;
    min-height: 20px;
    font-family: "{MONO}";
    font-size: 9pt;
    selection-background-color: {p.accent};
    selection-color: {p.paper};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QDateTimeEdit:focus {{
    border-color: {p.ink_2};
    border-bottom: 1px solid {p.accent};
}}
QComboBox {{ font-family: "{SANS}"; font-size: 9.5pt; }}
QComboBox::drop-down {{ border: none; width: 16px; }}
QComboBox QAbstractItemView {{
    background: {p.sheet};
    border: 1px solid {p.rule_strong};
    selection-background-color: {p.paper};
    selection-color: {p.accent_ink};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}

QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{
    width: 11px; height: 11px;
    border: 1px solid {p.rule_strong};
    background: {p.sheet};
    border-radius: 1px;
}}
QCheckBox::indicator:checked {{ background: {p.ink}; border-color: {p.ink}; }}
QCheckBox::indicator:hover {{ border-color: {p.ink_2}; }}

QSlider::groove:horizontal {{ height: 1px; background: {p.rule_strong}; }}
QSlider::sub-page:horizontal {{ height: 1px; background: {p.accent}; }}
QSlider::handle:horizontal {{
    width: 9px; height: 9px; margin: -4px 0;
    background: {p.paper}; border: 1px solid {p.ink}; border-radius: 0;
}}
QSlider::handle:horizontal:hover {{ border-color: {p.accent_ink}; }}

QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {p.rule}; min-height: 30px; border-radius: 0; }}
QScrollBar::handle:vertical:hover {{ background: {p.rule_strong}; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {p.rule}; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QPlainTextEdit {{
    background: {p.sheet};
    border: 1px solid {p.rule};
    font-family: "{MONO}";
    font-size: 8.5pt;
    color: {p.ink_2};
}}
QListWidget {{
    background: transparent;
    border: none;
    font-size: 9pt;
}}
QListWidget::item {{ padding: 4px 6px 4px 16px; color: {p.ink_2}; border: none; }}
QListWidget::item:hover {{ color: {p.ink}; }}
QListWidget::item:selected {{ color: {p.accent_ink}; background: transparent; }}
QToolTip {{
    background: {p.ink};
    color: {p.paper};
    border: none;
    padding: 4px 6px;
    font-size: 8.5pt;
}}
QFrame[role="rule"] {{ background: {p.rule}; border: none; max-height: 1px; min-height: 1px; }}
QFrame[role="vrule"] {{ background: {p.rule}; border: none; max-width: 1px; min-width: 1px; }}
QMenu {{ background: {p.sheet}; border: 1px solid {p.rule_strong}; padding: 4px 0; }}
QMenu::item {{ padding: 4px 18px; }}
QMenu::item:selected {{ background: {p.paper}; color: {p.accent_ink}; }}
"""


def apply(app: QApplication, t: Theme) -> None:
    """Apply palette, style sheet and pyqtgraph defaults."""
    app.setStyle("Fusion")
    qp = QPalette()
    for role, name in (
        (QPalette.ColorRole.Window, "paper"),
        (QPalette.ColorRole.Base, "sheet"),
        (QPalette.ColorRole.AlternateBase, "paper"),
        (QPalette.ColorRole.Text, "ink"),
        (QPalette.ColorRole.WindowText, "ink"),
        (QPalette.ColorRole.ButtonText, "ink"),
        (QPalette.ColorRole.Button, "paper"),
        (QPalette.ColorRole.Highlight, "accent"),
        (QPalette.ColorRole.HighlightedText, "paper"),
        (QPalette.ColorRole.PlaceholderText, "ink_3"),
        (QPalette.ColorRole.ToolTipBase, "ink"),
        (QPalette.ColorRole.ToolTipText, "paper"),
    ):
        qp.setColor(role, t.color(name))
    app.setPalette(qp)
    app.setFont(t.sans())
    app.setStyleSheet(style_sheet(t))
    pg.setConfigOptions(
        background=t.pal.paper,
        foreground=t.pal.ink_2,
        antialias=True,
        imageAxisOrder="row-major",
    )


def lut(stops: tuple[str, ...], n: int = 256):
    """Lookup table for pyqtgraph (n × 3 uint8) from colour stops."""
    import numpy as np

    rgb = np.array([style.hex_to_rgb(c) for c in stops])
    x = np.linspace(0.0, 1.0, len(stops))
    xi = np.linspace(0.0, 1.0, n)
    table = np.stack([np.interp(xi, x, rgb[:, k]) for k in range(3)], axis=1)
    return (table * 255.0 + 0.5).astype(np.uint8)
