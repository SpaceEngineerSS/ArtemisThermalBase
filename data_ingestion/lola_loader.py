"""LOLA GeoTIFF Loader — ingest real NASA LRO LOLA DEM data.

Loads real Lunar surface elevation data from GeoTIFF files produced
by the Lunar Orbiter Laser Altimeter (LOLA) instrument aboard the
Lunar Reconnaissance Orbiter (LRO).

Key concerns:
- NASA DEMs use Float32 with NoData sentinels (-3.4028235e+38).
- Elevation is in meters relative to the lunar reference sphere (1737.4 km).
- For the thermal solver, we center and zero-reference the elevation.
- CRS is typically IAU_2015:30100 (Moon, equidistant cylindrical) or
  polar stereographic for south pole tiles.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

References
----------
- Smith et al. (2017). "Summary of the results from the Lunar Orbiter
  Laser Altimeter after seven years." Icarus, 283, 70-91.
- LOLA PDS Archive: https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _import_rasterio():
    """Lazy import rasterio with a helpful error message."""
    try:
        import rasterio
        return rasterio
    except ImportError as e:
        raise ImportError(
            "rasterio is required for loading GeoTIFF DEMs. "
            "Install it with: pip install rasterio>=1.3.9"
        ) from e


class LOLALoader:
    """Loader for NASA LRO LOLA Digital Elevation Models in GeoTIFF format.

    Reads GeoTIFF DEMs, handles NoData masking, optional spatial subsetting,
    and outputs a DEMData object compatible with the simulation pipeline.

    Parameters
    ----------
    nodata_threshold : float
        Values below this threshold are treated as NoData.
        NASA LOLA NoData is typically -3.4028235e+38.
    fill_nodata : bool
        If True, interpolate NoData pixels from neighbours.
        If False, replace NoData with the minimum valid elevation.
    center_elevation : bool
        If True, subtract the mean elevation so the DEM is centered
        around z=0.  This improves numerical conditioning.
    """

    NODATA_DEFAULT = -3.4028235e+38

    def __init__(
        self,
        nodata_threshold: float = -1.0e+30,
        fill_nodata: bool = True,
        center_elevation: bool = True,
    ) -> None:
        self._nodata_threshold = nodata_threshold
        self._fill_nodata = fill_nodata
        self._center_elevation = center_elevation

    def load_dem(
        self,
        file_path: str | Path,
        bounds: tuple[float, float, float, float] | None = None,
        max_size: int | None = None,
    ):
        """Load a GeoTIFF DEM and return a DEMData object.

        Parameters
        ----------
        file_path : str or Path
            Path to the GeoTIFF file (.tif/.tiff).
        bounds : tuple, optional
            Spatial subset (xmin, ymin, xmax, ymax) in the CRS of the file.
            If None, loads the full raster.
        max_size : int, optional
            Maximum grid dimension (pixels). If the loaded DEM exceeds this,
            it will be downsampled by skipping rows/columns.

        Returns
        -------
        DEMData
            DEM data ready for the simulation pipeline.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file contains no valid elevation data.
        """
        from data_ingestion.synthetic_dem import DEMData

        rasterio = _import_rasterio()
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"DEM file not found: {file_path}")

        logger.info("Loading LOLA DEM: %s", file_path)

        with rasterio.open(file_path) as src:
            # Log metadata
            logger.info(
                "  CRS: %s, Size: %d x %d, Bands: %d, dtype: %s",
                src.crs, src.width, src.height, src.count, src.dtypes[0],
            )
            logger.info(
                "  Bounds: (%.4f, %.4f) → (%.4f, %.4f)",
                src.bounds.left, src.bounds.bottom,
                src.bounds.right, src.bounds.top,
            )
            logger.info("  NoData value: %s", src.nodata)

            # Read the first band
            if bounds is not None:
                # Compute the window for the requested bounds
                from rasterio.windows import from_bounds
                window = from_bounds(*bounds, transform=src.transform)
                elevation = src.read(1, window=window).astype(np.float64)
                transform = src.window_transform(window)
            else:
                elevation = src.read(1).astype(np.float64)
                transform = src.transform

            # Store CRS and transform for metadata
            crs_str = str(src.crs) if src.crs else "unknown"
            file_nodata = src.nodata

        ny, nx = elevation.shape
        logger.info("  Raw grid: %d x %d pixels (%.1f MB)",
                     nx, ny, elevation.nbytes / 1e6)

        # Step 1: Mask NoData values
        nodata_mask = self._build_nodata_mask(elevation, file_nodata)
        num_nodata = np.count_nonzero(nodata_mask)
        pct_nodata = 100.0 * num_nodata / elevation.size

        if num_nodata > 0:
            logger.info("  NoData pixels: %d (%.1f%%)", num_nodata, pct_nodata)

        if num_nodata == elevation.size:
            raise ValueError(
                f"DEM contains no valid elevation data. "
                f"All {elevation.size} pixels are NoData."
            )

        # Step 2: Fill or replace NoData
        if num_nodata > 0:
            elevation = self._handle_nodata(elevation, nodata_mask)

        # Step 3: Optional downsampling
        if max_size is not None and max(ny, nx) > max_size:
            step = max(ny, nx) // max_size
            elevation = elevation[::step, ::step]
            ny, nx = elevation.shape
            # Adjust the effective transform
            transform = rasterio.Affine(
                transform.a * step, transform.b, transform.c,
                transform.d, transform.e * step, transform.f,
            )
            logger.info("  Downsampled to %d x %d (step=%d)", nx, ny, step)

        # Step 4: Build coordinate arrays
        # The transform maps pixel (col, row) → projected (x, y)
        resolution_x = abs(transform.a)
        resolution_y = abs(transform.e)
        resolution_m = (resolution_x + resolution_y) / 2.0

        x_coords = np.array(
            [transform.c + (j + 0.5) * transform.a for j in range(nx)],
            dtype=np.float64,
        )
        y_coords = np.array(
            [transform.f + (i + 0.5) * transform.e for i in range(ny)],
            dtype=np.float64,
        )

        # Center the coordinate system around the DEM center
        x_center = (x_coords[0] + x_coords[-1]) / 2.0
        y_center = (y_coords[0] + y_coords[-1]) / 2.0
        x_coords -= x_center
        y_coords -= y_center

        # Step 5: Center elevation if requested
        if self._center_elevation:
            z_mean = np.nanmean(elevation)
            elevation -= z_mean
            logger.info("  Elevation centered: subtracted mean=%.1f m", z_mean)

        logger.info(
            "  Final DEM: %d x %d, res=%.2f m, z=[%.1f, %.1f] m",
            nx, ny, resolution_m,
            elevation.min(), elevation.max(),
        )

        metadata = {
            "type": "lola_geotiff",
            "source_file": str(file_path),
            "crs": crs_str,
            "original_size": f"{src.width}x{src.height}" if 'src' in dir() else "unknown",
            "num_pixels_x": nx,
            "num_pixels_y": ny,
            "resolution_m": resolution_m,
            "z_min_m": float(elevation.min()),
            "z_max_m": float(elevation.max()),
            "z_range_m": float(elevation.max() - elevation.min()),
            "nodata_pixels": int(num_nodata),
            "nodata_pct": float(pct_nodata),
        }

        return DEMData(
            elevation=elevation,
            x_coords=x_coords,
            y_coords=y_coords,
            resolution_m=resolution_m,
            metadata=metadata,
        )

    def _build_nodata_mask(
        self,
        elevation: np.ndarray,
        file_nodata: float | None,
    ) -> np.ndarray:
        """Build a boolean mask of NoData pixels.

        Combines the file's declared NoData value with a threshold check
        to catch both standard sentinels and unexpected fill values.
        """
        mask = np.zeros_like(elevation, dtype=bool)

        # Check declared NoData
        if file_nodata is not None:
            mask |= np.isclose(elevation, file_nodata, rtol=1e-5)

        # Check threshold (catches -3.4e38 style sentinels)
        mask |= elevation < self._nodata_threshold

        # Check NaN/Inf
        mask |= ~np.isfinite(elevation)

        return mask

    def _handle_nodata(
        self,
        elevation: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Handle NoData pixels by filling or replacing them.

        Parameters
        ----------
        elevation : np.ndarray
            Elevation grid with NoData values.
        mask : np.ndarray
            Boolean mask where True = NoData.

        Returns
        -------
        np.ndarray
            Elevation grid with NoData handled.
        """
        elevation = elevation.copy()

        if self._fill_nodata:
            # Use scipy nearest-neighbour interpolation for NoData fill
            try:
                from scipy.ndimage import distance_transform_edt

                # Replace NoData with NaN temporarily
                elevation[mask] = np.nan

                # Find nearest valid pixel for each NoData pixel
                valid_mask = ~mask
                if np.any(valid_mask):
                    # Get indices of nearest valid pixels
                    _, indices = distance_transform_edt(
                        mask, return_distances=True, return_indices=True,
                    )
                    elevation[mask] = elevation[
                        indices[0][mask], indices[1][mask]
                    ]
                    logger.info("  NoData filled via nearest-neighbour interpolation")
                else:
                    elevation[mask] = 0.0
            except ImportError:
                # Fallback: replace with minimum valid value
                valid_min = np.nanmin(elevation[~mask]) if np.any(~mask) else 0.0
                elevation[mask] = valid_min
                logger.warning(
                    "  scipy not available; replaced NoData with min=%.1f m",
                    valid_min,
                )
        else:
            # Simple replacement with minimum valid elevation
            valid_min = np.nanmin(elevation[~mask]) if np.any(~mask) else 0.0
            elevation[mask] = valid_min
            logger.info("  NoData replaced with min valid elevation: %.1f m", valid_min)

        return elevation
