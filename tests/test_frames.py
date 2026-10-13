"""Reference-frame conventions: the Sun must land on the mesh the right way round."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from core_engine.frames import (
    body_to_enu,
    curvature_drop,
    enu_frame,
    projected_frame,
    surface_normal,
    synthetic_sun_body,
)
from data_ingestion import polar_stereographic as ps

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "data" / "ephemeris"
SMOKE = ROOT / "validation" / "reference" / "naif_de440_smoke.json"
KERNEL_FILES = [
    "naif0012.tls",
    "de440s.bsp",
    "moon_pa_de440_200625.bpc",
    "moon_de440_250416.tf",
]


def _inverse(x: float, y: float) -> tuple[float, float]:
    return ps.inverse(x, y)


@pytest.mark.parametrize("lat, lon", [(-89.67, 129.78), (-45.0, 10.0), (30.0, 250.0)])
def test_enu_is_a_proper_rotation(lat: float, lon: float) -> None:
    R = body_to_enu(lat, lon)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(R @ surface_normal(lat, lon), [0.0, 0.0, 1.0])


def test_enu_east_and_north_point_the_right_way() -> None:
    R = body_to_enu(0.0, 0.0)
    assert np.allclose(R @ np.array([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0])  # +y body is east
    assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0])  # +z body is north


def test_stereographic_matches_proj_convention() -> None:
    """PROJ south-polar stereographic: lon 0 lies along +y, lon 90E along +x."""
    x0, y0 = ps.forward(-89.0, 0.0)
    x90, y90 = ps.forward(-89.0, 90.0)
    assert x0 == pytest.approx(0.0, abs=1e-6) and y0 > 0
    assert x90 > 0 and y90 == pytest.approx(0.0, abs=1e-6)


def test_projected_frame_is_right_handed_with_up_on_z() -> None:
    x0, y0 = ps.forward(-89.67, 129.78)
    frame = projected_frame(_inverse, x0, y0)
    R = frame.rotation
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert frame.latitude_deg == pytest.approx(-89.67, abs=1e-9)
    assert np.allclose(R @ surface_normal(-89.67, 129.78), [0.0, 0.0, 1.0], atol=1e-9)


def test_projected_frame_agrees_with_rasterio_crs() -> None:
    rasterio = pytest.importorskip("rasterio")
    from core_engine.frames import rasterio_inverse

    crs = rasterio.crs.CRS.from_proj4(
        "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs"
    )
    x0, y0 = ps.forward(-89.5, 40.0)
    ours = projected_frame(_inverse, x0, y0).rotation
    theirs = projected_frame(rasterio_inverse(crs.to_wkt(), 1737400.0), x0, y0).rotation
    assert np.allclose(ours, theirs, atol=1e-6)


def test_grid_azimuth_at_the_pole_follows_longitude() -> None:
    """A Sun over 90°E, seen from the pole, sits toward grid +x."""
    frame = projected_frame(_inverse, 1.0, 1.0)
    sun = frame.from_body(surface_normal(0.0, 90.0))
    assert sun[0] > 0.99


def test_curvature_drop_at_20_km() -> None:
    drop = curvature_drop(np.array([20_000.0]), np.array([0.0]), 1_737_400.0)[0]
    assert drop == pytest.approx(115.1, abs=0.2)


def test_synthetic_sun_circles_at_constant_elevation_near_the_pole() -> None:
    frame = enu_frame(-90.0 + 1e-6, 0.0)
    period = 2_551_442.9
    elevations = []
    for k in range(8):
        sun = frame.from_body(synthetic_sun_body(period * k / 8, 0.0, -1.5, period))
        elevations.append(np.degrees(np.arcsin(sun[2])))
    assert np.ptp(elevations) < 1e-3
    assert elevations[0] == pytest.approx(1.5, abs=1e-3)


def _spice_or_skip():
    spice = pytest.importorskip("spiceypy")
    missing = [name for name in KERNEL_FILES if not (KERNELS / name).is_file()]
    if missing:
        pytest.skip(f"NAIF kernels not downloaded: {', '.join(missing)}")
    return spice


def test_iau_rotation_tracks_spice_moon_me() -> None:
    spice = _spice_or_skip()
    from data_ingestion.ephemeris import iau_moon_rotation

    for name in KERNEL_FILES:
        spice.furnsh(str(KERNELS / name))
    for text in ("2025-01-01T00:00:00", "2027-07-15T06:00:00"):
        R_spice = np.array(spice.pxform("J2000", "MOON_ME", spice.str2et(text)))
        R_iau = iau_moon_rotation(datetime.fromisoformat(text).replace(tzinfo=UTC))
        angle = np.degrees(np.arccos(np.clip((np.trace(R_spice @ R_iau.T) - 1) / 2, -1, 1)))
        assert angle < 0.01


def test_spice_sun_elevation_matches_reference_fixture() -> None:
    _spice_or_skip()
    from data_ingestion.spice_ephemeris import SpiceSolarEphemeris

    ref = json.loads(SMOKE.read_text(encoding="utf-8"))
    eph = SpiceSolarEphemeris([KERNELS / name for name in KERNEL_FILES])
    when = datetime.fromisoformat(ref["observation_time_utc"].replace("Z", "+00:00"))
    sun = eph.sun_state(when, 1361.0)
    frame = enu_frame(ref["target_latitude_deg"], ref["target_longitude_deg_east"])
    elevation = np.degrees(np.arcsin(frame.from_body(sun.direction_body)[2]))
    assert elevation == pytest.approx(ref["sun_elevation_deg"], abs=1e-6)
    assert sun.solar_flux == pytest.approx(ref["solar_flux_w_m2"], rel=1e-9)
