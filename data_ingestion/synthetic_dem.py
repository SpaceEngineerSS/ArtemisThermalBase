"""Synthetic DEM generator for Milestone 1 validation.

Generates parametric crater topographies (parabolic bowl, conical, flat) for
testing the raytracing and thermal solver pipeline before using real PDS data.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

Notes
-----
The synthetic crater is centered at the origin (x=0, y=0) with z representing
elevation relative to the surrounding flat terrain (z=0). The parabolic bowl
profile follows:

    z(r) = -D * (1 - (r/R)^2)     for r <= R  (inside crater)
    z(r) = H * exp(-(r-R)^2 / w^2) for r > R  (raised rim, Gaussian profile)
    z(r) = 0                       for r >> R  (flat surroundings)

where D = depth, R = radius, H = rim height, w = rim width (~0.1*R).

References
----------
- Pike, R.J. (1977). "Size-dependence in the shape of fresh impact craters
  on the Moon." In: Impact and Explosion Cratering, pp. 489-509.
- Zuber, M.T., et al. (2012). "Constraints on the volatile distribution
  within Shackleton crater." Nature, 486, 378-381.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from core_engine.constants import SyntheticDEMConfig

logger = logging.getLogger(__name__)


@dataclass
class DEMData:
    """Container for a Digital Elevation Model dataset.

    Attributes
    ----------
    elevation : np.ndarray
        2D elevation grid [m]. Shape: (ny, nx). dtype: float64.
    x_coords : np.ndarray
        1D array of x coordinates [m]. Shape: (nx,).
    y_coords : np.ndarray
        1D array of y coordinates [m]. Shape: (ny,).
    resolution_m : float
        Grid spacing [m/pixel].
    metadata : dict
        Additional metadata about the DEM (type, parameters, etc.).
    """

    elevation: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    resolution_m: float
    metadata: dict


def generate_synthetic_dem(config: SyntheticDEMConfig) -> DEMData:
    """Generate a synthetic crater DEM from configuration.

    Dispatches to the appropriate generator function based on ``config.crater_type``.

    Parameters
    ----------
    config : SyntheticDEMConfig
        Synthetic DEM configuration loaded from YAML.

    Returns
    -------
    DEMData
        Generated DEM with elevation grid and coordinate arrays.

    Raises
    ------
    ValueError
        If ``config.crater_type`` is not recognized.
    """
    rng = np.random.default_rng(config.seed)
    logger.info(
        "Generating synthetic DEM: type=%s, radius=%.0f m, depth=%.0f m, "
        "resolution=%.1f m/px, seed=%d",
        config.crater_type,
        config.radius_m,
        config.depth_m,
        config.grid_resolution_m,
        config.seed,
    )

    generators = {
        "parabolic_bowl": _generate_parabolic_bowl,
        "conical": _generate_conical,
        "flat": _generate_flat,
    }

    if config.crater_type not in generators:
        raise ValueError(
            f"Unknown crater type '{config.crater_type}'. "
            f"Valid options: {list(generators.keys())}"
        )

    return generators[config.crater_type](config, rng)


def _generate_parabolic_bowl(
    config: SyntheticDEMConfig,
    rng: np.random.Generator,
) -> DEMData:
    """Generate a parabolic bowl crater DEM.

    The crater profile is a downward-opening parabola inside radius R,
    with a Gaussian raised rim outside R, and flat terrain beyond.

    Profile
    -------
    For radial distance r from crater center:
        r <= R:  z(r) = -D * (1 - (r/R)^2)
        r >  R:  z(r) = H * exp(-(r - R)^2 / (2 * w^2))

    where w = 0.1 * R (rim width parameter).

    Parameters
    ----------
    config : SyntheticDEMConfig
        Configuration with crater geometry.
    rng : np.random.Generator
        Random number generator (for optional noise).

    Returns
    -------
    DEMData
        Parabolic bowl crater DEM.
    """
    R = config.radius_m
    D = config.depth_m
    H = config.rim_height_m
    res = config.grid_resolution_m
    pad = config.domain_padding_m

    # Domain extends from -(R + pad) to +(R + pad) in both x and y
    half_extent = R + pad
    x = np.arange(-half_extent, half_extent + res, res, dtype=np.float64)
    y = np.arange(-half_extent, half_extent + res, res, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    # Radial distance from crater center
    r = np.sqrt(xx**2 + yy**2)

    # Initialize elevation to zero (flat terrain)
    elevation = np.zeros_like(r, dtype=np.float64)

    # Rim width parameter (controls how wide the raised rim is)
    rim_width = 0.1 * R

    # Inside the crater: parabolic bowl
    inside_mask = r <= R
    r_norm = r[inside_mask] / R  # Normalized radius [0, 1]
    elevation[inside_mask] = -D * (1.0 - r_norm**2)

    # Outside the crater: Gaussian raised rim
    outside_mask = ~inside_mask
    elevation[outside_mask] = H * np.exp(
        -((r[outside_mask] - R) ** 2) / (2.0 * rim_width**2)
    )

    ny, nx = elevation.shape
    logger.info(
        "Parabolic bowl DEM generated: %d x %d pixels (%.1f x %.1f km), "
        "depth=%.0f m, rim=%.0f m, z_min=%.1f m, z_max=%.1f m",
        nx,
        ny,
        (nx * res) / 1000.0,
        (ny * res) / 1000.0,
        D,
        H,
        elevation.min(),
        elevation.max(),
    )

    metadata = {
        "type": "parabolic_bowl",
        "radius_m": R,
        "depth_m": D,
        "rim_height_m": H,
        "rim_width_m": rim_width,
        "resolution_m": res,
        "domain_extent_m": half_extent * 2,
        "num_pixels_x": nx,
        "num_pixels_y": ny,
        "seed": config.seed,
        "z_min_m": float(elevation.min()),
        "z_max_m": float(elevation.max()),
    }

    return DEMData(
        elevation=elevation,
        x_coords=x,
        y_coords=y,
        resolution_m=res,
        metadata=metadata,
    )


def _generate_conical(
    config: SyntheticDEMConfig,
    rng: np.random.Generator,
) -> DEMData:
    """Generate a conical crater DEM.

    The crater profile is a linear cone (V-shape) inside radius R.

    Profile
    -------
    For radial distance r from crater center:
        r <= R:  z(r) = -D * (1 - r/R)
        r >  R:  z(r) = H * exp(-(r - R)^2 / (2 * w^2))

    Parameters
    ----------
    config : SyntheticDEMConfig
        Configuration with crater geometry.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    DEMData
        Conical crater DEM.
    """
    R = config.radius_m
    D = config.depth_m
    H = config.rim_height_m
    res = config.grid_resolution_m
    pad = config.domain_padding_m

    half_extent = R + pad
    x = np.arange(-half_extent, half_extent + res, res, dtype=np.float64)
    y = np.arange(-half_extent, half_extent + res, res, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(xx**2 + yy**2)

    elevation = np.zeros_like(r, dtype=np.float64)
    rim_width = 0.1 * R

    inside_mask = r <= R
    elevation[inside_mask] = -D * (1.0 - r[inside_mask] / R)

    outside_mask = ~inside_mask
    elevation[outside_mask] = H * np.exp(
        -((r[outside_mask] - R) ** 2) / (2.0 * rim_width**2)
    )

    ny, nx = elevation.shape
    logger.info("Conical DEM generated: %d x %d pixels", nx, ny)

    metadata = {
        "type": "conical",
        "radius_m": R,
        "depth_m": D,
        "rim_height_m": H,
        "resolution_m": res,
        "num_pixels_x": nx,
        "num_pixels_y": ny,
        "seed": config.seed,
        "z_min_m": float(elevation.min()),
        "z_max_m": float(elevation.max()),
    }

    return DEMData(
        elevation=elevation,
        x_coords=x,
        y_coords=y,
        resolution_m=res,
        metadata=metadata,
    )


def _generate_flat(
    config: SyntheticDEMConfig,
    rng: np.random.Generator,
) -> DEMData:
    """Generate a flat terrain DEM (no crater).

    Useful as a baseline test case for the thermal solver (uniform illumination).

    Parameters
    ----------
    config : SyntheticDEMConfig
        Configuration (radius/depth are ignored for flat terrain).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    DEMData
        Flat terrain DEM with z=0 everywhere.
    """
    R = config.radius_m
    res = config.grid_resolution_m
    pad = config.domain_padding_m

    half_extent = R + pad
    x = np.arange(-half_extent, half_extent + res, res, dtype=np.float64)
    y = np.arange(-half_extent, half_extent + res, res, dtype=np.float64)

    elevation = np.zeros((len(y), len(x)), dtype=np.float64)

    ny, nx = elevation.shape
    logger.info("Flat DEM generated: %d x %d pixels", nx, ny)

    metadata = {
        "type": "flat",
        "resolution_m": res,
        "num_pixels_x": nx,
        "num_pixels_y": ny,
        "seed": config.seed,
        "z_min_m": 0.0,
        "z_max_m": 0.0,
    }

    return DEMData(
        elevation=elevation,
        x_coords=x,
        y_coords=y,
        resolution_m=res,
        metadata=metadata,
    )


def compute_dem_statistics(dem: DEMData) -> dict:
    """Compute summary statistics for a DEM.

    Parameters
    ----------
    dem : DEMData
        DEM dataset.

    Returns
    -------
    dict
        Statistics including min, max, mean, std of elevation,
        total area, number of pixels, etc.
    """
    elev = dem.elevation
    res = dem.resolution_m
    ny, nx = elev.shape

    return {
        "num_pixels_x": nx,
        "num_pixels_y": ny,
        "total_pixels": nx * ny,
        "resolution_m": res,
        "domain_x_km": (nx * res) / 1000.0,
        "domain_y_km": (ny * res) / 1000.0,
        "total_area_km2": (nx * ny * res**2) / 1e6,
        "z_min_m": float(elev.min()),
        "z_max_m": float(elev.max()),
        "z_mean_m": float(elev.mean()),
        "z_std_m": float(elev.std()),
        "z_range_m": float(elev.max() - elev.min()),
        "estimated_triangles": 2 * (nx - 1) * (ny - 1),
    }
