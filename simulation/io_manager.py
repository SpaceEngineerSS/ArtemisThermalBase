"""Data I/O manager — persist simulation results as NumPy arrays.

Saves and loads raw simulation data (thermal grids, illumination maps,
DEM elevations, probe time series) to allow re-rendering without
re-running the heavy physics loop.

File layout under output_dir/:
    thermal_grid.npy       — Final surface temperatures [K], shape (N_faces,)
    illumination_grid.npy  — Final illumination fractions, shape (N_faces,)
    dem_grid.npy           — DEM elevation grid [m], shape (ny, nx)
    face_centroids.npy     — Face centroid coordinates [m], shape (N_faces, 3)
    face_areas.npy         — Face areas [m²], shape (N_faces,)
    probe_temps.npz        — Probe temperature time series (one key per probe)
    sun_elevations.npy     — Sun elevation [deg] per output snapshot
    metadata.json          — Simulation metadata (JSON)

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def save_results(
    output_dir: Path | str,
    surface_temps: np.ndarray,
    illumination: np.ndarray,
    dem_elevation: np.ndarray,
    face_centroids: np.ndarray,
    face_areas: np.ndarray,
    probe_temps: dict[str, list[float]],
    sun_elevations: list[float],
    metadata: dict,
) -> list[Path]:
    """Save all simulation results to disk as NumPy arrays + JSON.

    Parameters
    ----------
    output_dir : Path or str
        Output directory (created if needed).
    surface_temps : np.ndarray
        Final surface temperatures [K]. Shape: (N_faces,).
    illumination : np.ndarray
        Final illumination fractions. Shape: (N_faces,).
    dem_elevation : np.ndarray
        DEM elevation grid [m]. Shape: (ny, nx).
    face_centroids : np.ndarray
        Face centroid coordinates. Shape: (N_faces, 3).
    face_areas : np.ndarray
        Face areas [m²]. Shape: (N_faces,).
    probe_temps : dict[str, list[float]]
        Probe temperature time series.
    sun_elevations : list[float]
        Sun elevation [deg] per output snapshot.
    metadata : dict
        Simulation metadata.

    Returns
    -------
    list[Path]
        Paths to all saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    # Core arrays
    for name, arr in [
        ("thermal_grid.npy", surface_temps),
        ("illumination_grid.npy", illumination),
        ("dem_grid.npy", dem_elevation),
        ("face_centroids.npy", face_centroids),
        ("face_areas.npy", face_areas),
        ("sun_elevations.npy", np.array(sun_elevations, dtype=np.float64)),
    ]:
        path = output_dir / name
        np.save(path, arr)
        saved.append(path)
        logger.debug("Saved %s: shape=%s, dtype=%s", name, arr.shape, arr.dtype)

    # Probe time series (multiple arrays in one file)
    if probe_temps:
        probe_path = output_dir / "probe_temps.npz"
        arrays = {k: np.array(v, dtype=np.float64) for k, v in probe_temps.items()}
        np.savez_compressed(probe_path, **arrays)
        saved.append(probe_path)
        logger.debug("Saved probe_temps.npz: %d probes", len(arrays))

    # Metadata
    meta_path = output_dir / "metadata.json"
    # Sanitize metadata for JSON serialization
    safe_meta = _sanitize_for_json(metadata)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(safe_meta, f, indent=2, ensure_ascii=False)
    saved.append(meta_path)

    logger.info(
        "Saved %d files to %s (thermal: %s, illum: %s)",
        len(saved), output_dir, surface_temps.shape, illumination.shape,
    )

    return saved


def load_results(
    output_dir: Path | str,
) -> dict[str, np.ndarray | dict]:
    """Load previously saved simulation results.

    Parameters
    ----------
    output_dir : Path or str
        Directory containing saved results.

    Returns
    -------
    dict
        Keys: 'thermal_grid', 'illumination_grid', 'dem_grid',
        'face_centroids', 'face_areas', 'sun_elevations',
        'probe_temps', 'metadata'.
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    data: dict = {}

    # Core arrays
    for key, filename in [
        ("thermal_grid", "thermal_grid.npy"),
        ("illumination_grid", "illumination_grid.npy"),
        ("dem_grid", "dem_grid.npy"),
        ("face_centroids", "face_centroids.npy"),
        ("face_areas", "face_areas.npy"),
        ("sun_elevations", "sun_elevations.npy"),
    ]:
        path = output_dir / filename
        if path.exists():
            data[key] = np.load(path)
            logger.debug("Loaded %s: shape=%s", key, data[key].shape)
        else:
            logger.warning("Missing file: %s", path)
            data[key] = None

    # Probe time series
    probe_path = output_dir / "probe_temps.npz"
    if probe_path.exists():
        npz = np.load(probe_path)
        data["probe_temps"] = {k: npz[k] for k in npz.files}
        logger.debug("Loaded probe_temps: %d probes", len(data["probe_temps"]))
    else:
        data["probe_temps"] = {}

    # Metadata
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            data["metadata"] = json.load(f)
    else:
        data["metadata"] = {}

    logger.info("Loaded results from %s (%d keys)", output_dir, len(data))

    return data


def _sanitize_for_json(obj: object) -> object:
    """Recursively convert NumPy types and other non-JSON types to Python natives."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
