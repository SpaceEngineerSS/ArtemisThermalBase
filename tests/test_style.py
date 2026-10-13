"""Colour maps must read correctly in grayscale and for colour-blind readers."""

from __future__ import annotations

import pytest

from artemis_thermal import style


@pytest.mark.parametrize(
    "stops",
    [style.TEMPERATURE_STOPS, style.ILLUMINATION_STOPS, style.RELIEF_STOPS],
    ids=["temperature", "illumination", "relief"],
)
def test_sequential_maps_get_lighter(stops) -> None:
    lightness = [style.relative_lightness(c) for c in stops]
    assert all(b > a + 2.0 for a, b in zip(lightness, lightness[1:], strict=False))


def test_diverging_map_is_light_in_the_middle() -> None:
    lightness = [style.relative_lightness(c) for c in style.DIVERGING_STOPS]
    middle = len(lightness) // 2
    assert lightness[middle] == max(lightness)


@pytest.mark.parametrize("palette", [style.LIGHT, style.DARK], ids=["light", "dark"])
def test_text_contrast(palette) -> None:
    def contrast(a: str, b: str) -> float:
        def lum(c: str) -> float:
            def lin(v: float) -> float:
                return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            r, g, b_ = (lin(v) for v in style.hex_to_rgb(c))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b_
        hi, lo = sorted((lum(a), lum(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    assert contrast(palette.ink, palette.paper) >= 12.0
    assert contrast(palette.ink_2, palette.paper) >= 4.5
    assert contrast(palette.accent_ink, palette.paper) >= 4.5
