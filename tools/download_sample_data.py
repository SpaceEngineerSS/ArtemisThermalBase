"""Download or generate sample LOLA DEM data for testing.

Attempts to download a real LOLA GeoTIFF tile from NASA PDS archives.
If the download fails (network issues, URL changes), falls back to
generating a high-fidelity "semi-synthetic" Shackleton-like crater DEM
with Perlin-style fractal noise for realistic terrain texture.

Usage
-----
    python tools/download_sample_data.py
    python tools/download_sample_data.py --output data/shackleton_dem.tif
    python tools/download_sample_data.py --synthetic  # Force synthetic mode

Output
------
    data/sample_lola_dem.tif — GeoTIFF suitable for lola_loader.py

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shackleton Crater parameters (real measurements)
# ---------------------------------------------------------------------------
# Zuber et al. (2012), Nature 486:378-381
# Smith et al. (2010), GRL 37:L18204
SHACKLETON = {
    "lat_deg": -89.9,
    "lon_deg": 0.0,
    "radius_m": 10500.0,      # ~21 km diameter
    "depth_m": 4200.0,         # 4.2 km deep
    "rim_height_m": 300.0,     # rim above terrain
    "d_over_D": 0.20,          # depth-to-diameter ratio
}


# ---------------------------------------------------------------------------
# Real Data Download
# ---------------------------------------------------------------------------

# LOLA south pole reduced DEM (20m/px) from USGS Astrogeology
# This is a 1° × 1° tile covering the south pole (~89-90°S)
_LOLA_URLS = [
    # USGS LOLA South Pole 20m GeoTIFF
    "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_ClrShade_Global_128ppd_v04.tif",
    # PDS direct (WAC + LOLA merged DEM, 100m/pixel)
    "https://pds.lroc.asu.edu/data/LRO-L-LOLA-3-RDR-V1.0/LRO_LOLA/DATA/"
    "LOLA_GDR/POLAR/SOUTH/LDEM_80S_20M.TIF",
]


def try_download_lola(output_path: Path) -> bool:
    """Attempt to download a real LOLA DEM tile.

    Tries multiple URLs in order, returns True on success.
    Clips to a manageable subset (~20 km around the south pole).
    """
    try:
        import urllib.request
        import tempfile
    except ImportError:
        return False

    for url in _LOLA_URLS:
        try:
            logger.info("Attempting download: %s", url[:80] + "...")

            # Download to temporary file (could be very large)
            tmp_path = output_path.parent / "_download_tmp.tif"

            request = urllib.request.Request(url)
            request.add_header("User-Agent", "ArtemisThermalBase/0.1")

            with urllib.request.urlopen(request, timeout=30) as response:
                total_size = int(response.headers.get("content-length", 0))
                if total_size > 500_000_000:  # Skip files >500MB
                    logger.warning("  File too large (%.0f MB), skipping", total_size / 1e6)
                    continue

                with open(tmp_path, "wb") as f:
                    chunk_size = 1024 * 1024  # 1 MB
                    downloaded = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = 100 * downloaded / total_size
                            logger.info("  Downloaded %.1f MB (%.0f%%)",
                                        downloaded / 1e6, pct)

            # Try to clip the south pole region from the full file
            _clip_south_pole(tmp_path, output_path)
            tmp_path.unlink(missing_ok=True)
            logger.info("Real LOLA DEM saved: %s", output_path)
            return True

        except Exception as e:
            logger.warning("  Download failed: %s", e)
            Path(output_path.parent / "_download_tmp.tif").unlink(missing_ok=True)
            continue

    return False


def _clip_south_pole(src_path: Path, dst_path: Path) -> None:
    """Clip a global/polar DEM to a ~25 km region around Shackleton."""
    try:
        import rasterio
        from rasterio.windows import from_bounds

        with rasterio.open(src_path) as src:
            # Estimate bounds: ~25 km around south pole in projected coords
            # or ~0.5° in geographic coords
            if src.crs and "4326" in str(src.crs):
                # Geographic CRS
                bounds = (-1.0, -90.0, 1.0, -89.5)
            else:
                # Projected CRS (assume meters)
                bounds = (-25000, -25000, 25000, 25000)

            try:
                window = from_bounds(*bounds, transform=src.transform)
                data = src.read(1, window=window)
                transform = src.window_transform(window)
            except Exception:
                # Fallback: read the center 500x500 region
                cx, cy = src.width // 2, src.height // 2
                half = 250
                window = rasterio.windows.Window(cx - half, cy - half, half * 2, half * 2)
                data = src.read(1, window=window)
                transform = src.window_transform(window)

            # Write clipped GeoTIFF
            profile = src.profile.copy()
            profile.update(
                width=data.shape[1],
                height=data.shape[0],
                transform=transform,
                compress="deflate",
            )

            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data, 1)

    except ImportError:
        # No rasterio available, just copy the file
        import shutil
        shutil.copy2(src_path, dst_path)


# ---------------------------------------------------------------------------
# Semi-Synthetic Shackleton Generator (Fallback)
# ---------------------------------------------------------------------------


def _fbm_noise_2d(
    shape: tuple[int, int],
    scale: float = 100.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate fractal Brownian motion noise on a 2D grid.

    Uses summed sine waves as a lightweight substitute for true Perlin noise
    (no external dependencies needed).

    Parameters
    ----------
    shape : tuple
        (ny, nx) grid dimensions.
    scale : float
        Spatial scale of the largest octave.
    octaves : int
        Number of noise octaves.
    persistence : float
        Amplitude decay per octave.
    lacunarity : float
        Frequency multiplier per octave.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noise field in [-1, 1], shape (ny, nx).
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    noise = np.zeros(shape, dtype=np.float64)

    y_idx = np.arange(ny, dtype=np.float64)
    x_idx = np.arange(nx, dtype=np.float64)
    xx, yy = np.meshgrid(x_idx, y_idx)

    amplitude = 1.0
    frequency = 1.0 / scale
    max_amp = 0.0

    for _ in range(octaves):
        # Random phase and direction for this octave
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        angle = rng.uniform(0, 2 * np.pi)

        # Rotated coordinates for anisotropy
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xr = xx * cos_a - yy * sin_a
        yr = xx * sin_a + yy * cos_a

        noise += amplitude * np.sin(
            2 * np.pi * frequency * xr + phase_x
        ) * np.cos(
            2 * np.pi * frequency * yr + phase_y
        )

        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize to [-1, 1]
    noise /= max_amp
    return noise


def generate_semi_synthetic_shackleton(
    output_path: Path,
    grid_size: int = 501,
    resolution_m: float = 20.0,
    noise_amplitude_m: float = 50.0,
    seed: int = 42,
) -> Path:
    """Generate a semi-synthetic Shackleton crater GeoTIFF.

    Combines a parametric parabolic bowl profile (matching real Shackleton
    dimensions) with fractal noise to create realistic terrain texture.

    Parameters
    ----------
    output_path : Path
        Output GeoTIFF path.
    grid_size : int
        Grid dimension (pixels).  501 → ~10 km domain at 20 m/px.
    resolution_m : float
        Grid spacing [m/pixel].
    noise_amplitude_m : float
        Peak-to-peak noise amplitude [m].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Path
        Path to the generated GeoTIFF.
    """
    R = SHACKLETON["radius_m"]
    D = SHACKLETON["depth_m"]
    H = SHACKLETON["rim_height_m"]

    logger.info(
        "Generating semi-synthetic Shackleton DEM: "
        "%d x %d px, res=%.0f m, R=%.0f m, D=%.0f m",
        grid_size, grid_size, resolution_m, R, D,
    )

    # Build coordinate grid centered at crater
    half_extent = (grid_size - 1) * resolution_m / 2
    x = np.linspace(-half_extent, half_extent, grid_size, dtype=np.float64)
    y = np.linspace(-half_extent, half_extent, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(xx**2 + yy**2)

    # Base crater profile (parabolic bowl)
    elevation = np.zeros_like(r, dtype=np.float64)

    inside = r <= R
    r_norm = r[inside] / R
    elevation[inside] = -D * (1.0 - r_norm**2)

    outside = ~inside
    rim_width = 0.12 * R
    elevation[outside] = H * np.exp(
        -((r[outside] - R) ** 2) / (2.0 * rim_width**2)
    )

    # Add fractal noise for realistic terrain texture
    noise = _fbm_noise_2d(
        shape=(grid_size, grid_size),
        scale=grid_size / 8.0,
        octaves=6,
        persistence=0.55,
        lacunarity=2.1,
        seed=seed,
    )

    # Scale noise: stronger inside crater (slumping), weaker on flat terrain
    depth_factor = np.clip(1.0 - r / (R * 1.5), 0.2, 1.0)
    elevation += noise * noise_amplitude_m * depth_factor

    # Add small-scale roughness (boulder fields)
    fine_noise = _fbm_noise_2d(
        shape=(grid_size, grid_size),
        scale=grid_size / 30.0,
        octaves=3,
        persistence=0.4,
        lacunarity=2.5,
        seed=seed + 1,
    )
    elevation += fine_noise * 5.0  # ±5 m micro-roughness

    logger.info(
        "  Elevation range: [%.1f, %.1f] m, noise σ=%.1f m",
        elevation.min(), elevation.max(), noise.std() * noise_amplitude_m,
    )

    # Write as GeoTIFF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import rasterio
        from rasterio.transform import from_origin

        # Origin at top-left corner, y decreasing downward
        transform = from_origin(
            west=x[0] - resolution_m / 2,
            north=y[-1] + resolution_m / 2,
            xsize=resolution_m,
            ysize=resolution_m,
        )

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": grid_size,
            "height": grid_size,
            "count": 1,
            "crs": None,  # Local Cartesian (no CRS)
            "transform": transform,
            "compress": "deflate",
            "nodata": float(np.finfo(np.float32).min),
        }

        # Flip vertically so row 0 = northernmost
        data_to_write = np.flipud(elevation).astype(np.float32)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data_to_write, 1)
            dst.update_tags(
                DESCRIPTION="Semi-synthetic Shackleton crater DEM",
                SOURCE="ArtemisThermalBase parametric + fBm noise",
                CRATER_RADIUS_M=str(R),
                CRATER_DEPTH_M=str(D),
                RESOLUTION_M=str(resolution_m),
                SEED=str(seed),
            )

        size_mb = output_path.stat().st_size / 1e6
        logger.info("Semi-synthetic DEM saved: %s (%.1f MB)", output_path, size_mb)

    except ImportError:
        # Fallback: save as raw numpy
        npy_path = output_path.with_suffix(".npy")
        np.save(npy_path, elevation)
        logger.warning(
            "rasterio not available. Saved raw NumPy instead: %s", npy_path
        )
        return npy_path

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Download or generate sample LOLA DEM data."""
    parser = argparse.ArgumentParser(
        description="Download or generate sample LOLA DEM for ArtemisThermalBase",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/sample_lola_dem.tif",
        help="Output file path (default: data/sample_lola_dem.tif)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip download, generate semi-synthetic Shackleton DEM",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=501,
        help="Grid size in pixels for synthetic DEM (default: 501)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=20.0,
        help="Grid resolution in m/pixel (default: 20.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        logger.info("Generating semi-synthetic Shackleton DEM (forced)...")
        generate_semi_synthetic_shackleton(
            output_path=output_path,
            grid_size=args.grid_size,
            resolution_m=args.resolution,
            seed=args.seed,
        )
        return 0

    # Try real download first
    logger.info("Attempting to download real LOLA data...")
    success = try_download_lola(output_path)

    if not success:
        logger.warning(
            "Real data download failed. Generating semi-synthetic fallback..."
        )
        generate_semi_synthetic_shackleton(
            output_path=output_path,
            grid_size=args.grid_size,
            resolution_m=args.resolution,
            seed=args.seed,
        )

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    sys.exit(main())
