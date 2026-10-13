"""Design tokens shared by the desktop app, the figures and the docs site.

The look is a printed technical report: warm off-white paper, graphite ink,
hairline rules, one sodium-orange accent used sparingly. The dark theme is
the same report on a graphite desk. Everything visual reads its colours from
here, so a change in one place shows up in the app, the PNGs and the site.

Colour maps are built so that lightness rises monotonically from the first
stop to the last (checked in ``tests/test_style.py``); a map that only
changes hue would hide gradients in grayscale prints and to readers with
colour-vision deficiencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
FONTS = ASSETS / "fonts"

SANS = "IBM Plex Sans"
MONO = "IBM Plex Mono"


@dataclass(frozen=True)
class Palette:
    """Named colours of one theme."""

    paper: str        # page background
    sheet: str        # raised surfaces: panels, inputs
    rule: str         # hairlines and borders
    rule_strong: str  # axis lines, focused borders
    ink: str          # primary text
    ink_2: str        # secondary text, labels
    ink_3: str        # tertiary: ticks, placeholders
    accent: str       # sodium orange: selection, the one thing to look at
    accent_ink: str   # accent dark enough for text on ``paper``
    cold: str         # ice-stable class
    cold_soft: str    # ice-marginal class
    warn: str         # errors


LIGHT = Palette(
    paper="#F4F2EC",
    sheet="#FBFAF6",
    rule="#D8D3C8",
    rule_strong="#A9A397",
    ink="#1D1F21",
    ink_2="#55585B",
    ink_3="#8C8E8F",
    accent="#E8641B",
    accent_ink="#B44A0E",
    cold="#2F6690",
    cold_soft="#93B5CC",
    warn="#A8322A",
)

DARK = Palette(
    paper="#17191B",
    sheet="#1F2225",
    rule="#2F3337",
    rule_strong="#4A4F54",
    ink="#E8E5DD",
    ink_2="#A7A9AA",
    ink_3="#73777A",
    accent="#F07A35",
    accent_ink="#F59A63",
    cold="#6FA3C8",
    cold_soft="#3E5F77",
    warn="#E06C5F",
)

# Surface temperature: graphite → slate blue → sand → sodium → pale ochre.
TEMPERATURE_STOPS = (
    "#0E1116",
    "#1E2A45",
    "#2D4B6E",
    "#4A7088",
    "#7E9293",
    "#B99A74",
    "#E3A35C",
    "#F3C47E",
    "#F8E6B8",
)

# Illumination fraction: night graphite → paper white, slightly warm.
ILLUMINATION_STOPS = ("#101214", "#3A3C3D", "#77766F", "#B9B5A8", "#F4F1E6")

# Elevation hillshade underlay.
RELIEF_STOPS = ("#2A2C2E", "#6E6D68", "#B4B0A5", "#E9E5DA")

# Signed differences (compare view): cold blue ← paper → sodium.
DIVERGING_STOPS = ("#1F4E79", "#6F9CC0", "#EDEAE2", "#E8A06E", "#B44A0E")

# Global temperature limits used when a figure should be comparable to others.
T_RANGE_K = (40.0, 330.0)


def palette(dark: bool = False) -> Palette:
    """The palette for the requested theme."""
    return DARK if dark else LIGHT


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    """``#RRGGBB`` → floats in [0, 1]."""
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def relative_lightness(color: str) -> float:
    """CIE L* (0–100) of an sRGB colour."""

    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(c) for c in hex_to_rgb(color))
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 116.0 * y ** (1.0 / 3.0) - 16.0 if y > 0.008856 else 903.3 * y


def font_files() -> list[Path]:
    """Bundled TTF files (empty if the assets were stripped)."""
    return sorted(FONTS.glob("*.ttf")) if FONTS.is_dir() else []
