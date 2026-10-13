"""Reference frames used to bring the Sun vector onto the terrain mesh.

Three frames meet here:

``body``
    Lunar body-fixed frame (SPICE ``MOON_ME`` or the IAU mean-Earth/polar
    axis frame). +z is the north rotation pole, +x points to 0°E.
``enu``
    Local East-North-Up tangent frame at a surface point. Right-handed.
``mesh``
    The frame the triangle mesh lives in. For a projected DEM its x/y axes
    are the grid axes of the map projection at the DEM centre and z is the
    local vertical there. For the synthetic DEM it is plain ENU at the target.

Everything that produces a Sun direction (synthetic model, Skyfield, SPICE)
returns a unit vector in the body frame. :class:`MeshFrame` turns that into
the mesh frame, so the ray tracer never sees an ephemeris-specific convention.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

InverseProjection = Callable[[float, float], tuple[float, float]]


def surface_normal(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return the outward unit normal of a sphere at (lat, lon), body frame."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=np.float64,
    )


def body_to_enu(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rotation matrix from the body frame to local East-North-Up.

    Rows are the E, N and U unit vectors expressed in body coordinates,
    so ``body_to_enu(lat, lon) @ v_body`` gives ``v_enu``. The determinant
    is +1.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )


def body_to_grid(
    inverse_projection: InverseProjection,
    x0: float,
    y0: float,
    step_m: float = 10.0,
) -> tuple[np.ndarray, float, float]:
    """Rotation from the body frame to a projected grid frame at (x0, y0).

    The grid +x and +y directions are found by stepping the inverse
    projection, then orthonormalised against the local vertical. This works
    for any conformal projection, so we do not hard-code one map convention.

    Returns
    -------
    rotation, lat_deg, lon_deg
        Body-to-grid rotation (rows x, y, up) and the geographic position of
        the grid point.

    """
    lat0, lon0 = inverse_projection(x0, y0)
    up = surface_normal(lat0, lon0)
    east_x = surface_normal(*inverse_projection(x0 + step_m, y0)) - up
    east_y = surface_normal(*inverse_projection(x0, y0 + step_m)) - up

    ex = east_x - np.dot(east_x, up) * up
    ex /= np.linalg.norm(ex)
    ey = east_y - np.dot(east_y, up) * up - np.dot(east_y, ex) * ex
    ey /= np.linalg.norm(ey)

    if np.dot(np.cross(ex, ey), up) < 0.0:
        raise ValueError(
            "Projected grid axes form a left-handed frame; the DEM would be "
            "mirrored relative to the Sun. Check the CRS of the input raster."
        )
    return np.vstack([ex, ey, up]), lat0, lon0


@dataclass(frozen=True)
class MeshFrame:
    """Body-to-mesh rotation plus the geographic anchor of the mesh origin."""

    rotation: np.ndarray
    latitude_deg: float
    longitude_deg: float
    kind: str

    def from_body(self, vector_body: np.ndarray) -> np.ndarray:
        """Rotate a body-frame vector into the mesh frame and normalise it."""
        v = self.rotation @ np.asarray(vector_body, dtype=np.float64)
        return v / np.linalg.norm(v)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for run metadata."""
        return {
            "kind": self.kind,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "rotation_body_to_mesh": self.rotation.tolist(),
        }


def enu_frame(lat_deg: float, lon_deg: float) -> MeshFrame:
    """Mesh frame for terrain laid out in local ENU at (lat, lon)."""
    return MeshFrame(body_to_enu(lat_deg, lon_deg), lat_deg, lon_deg, "enu")


def projected_frame(
    inverse_projection: InverseProjection,
    x0: float,
    y0: float,
) -> MeshFrame:
    """Mesh frame for a projected DEM whose centre is grid point (x0, y0)."""
    rotation, lat, lon = body_to_grid(inverse_projection, x0, y0)
    return MeshFrame(rotation, lat, lon, "projected")


def rasterio_inverse(crs_wkt: str, radius_m: float) -> InverseProjection:
    """Build an inverse projection from a raster CRS using rasterio."""
    from rasterio.crs import CRS
    from rasterio.warp import transform

    src = CRS.from_wkt(crs_wkt)
    dst = CRS.from_proj4(f"+proj=longlat +a={radius_m} +b={radius_m} +no_defs")

    def _inverse(x: float, y: float) -> tuple[float, float]:
        lons, lats = transform(src, dst, [x], [y])
        return float(lats[0]), float(lons[0])

    return _inverse


def frame_for_dem(
    dem_metadata: dict[str, Any],
    target_lat_deg: float,
    target_lon_deg: float,
    radius_m: float,
) -> MeshFrame:
    """Choose the mesh frame that matches how a DEM was built.

    A georeferenced DEM carries its CRS and projected centre; we use those.
    Anything else (synthetic terrain, legacy arrays) is treated as ENU at the
    configured target.
    """
    crs = dem_metadata.get("crs_wkt")
    centre = dem_metadata.get("projected_center")
    if crs and centre is not None:
        inverse = rasterio_inverse(str(crs), radius_m)
        return projected_frame(inverse, float(centre[0]), float(centre[1]))
    return enu_frame(target_lat_deg, target_lon_deg)


def curvature_drop(x: np.ndarray, y: np.ndarray, radius_m: float) -> np.ndarray:
    """Height of a sphere below its tangent plane at horizontal offset (x, y).

    ``R - sqrt(R² - d²)``, which is ≈ d²/2R for d ≪ R (115 m at 20 km on
    the Moon).
    """
    d2 = x * x + y * y
    return radius_m - np.sqrt(np.maximum(radius_m * radius_m - d2, 0.0))


def synthetic_sun_body(
    elapsed_s: float,
    target_lon_deg: float,
    subsolar_lat_deg: float,
    synodic_period_s: float,
) -> np.ndarray:
    """Sun direction in the body frame for the analytical preview model.

    The sub-solar point sits at a fixed latitude and moves west once per
    synodic month, starting at local noon over the target. At the pole this
    makes the Sun circle the horizon at an almost constant elevation, which
    is what real polar illumination looks like, rather than rising and
    setting like at the equator.
    """
    subsolar_lon = target_lon_deg - 360.0 * elapsed_s / synodic_period_s
    return surface_normal(subsolar_lat_deg, subsolar_lon)
