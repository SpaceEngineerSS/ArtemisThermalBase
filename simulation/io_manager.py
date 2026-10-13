"""Reading and writing run folders.

A run folder holds plain ``.npy`` arrays plus a JSON metadata file, so it
can be opened from any language without this package. Layout (format 2):

==============================  ==============================================
``surface_T_final.npy``         surface temperature at the last step [K]
``surface_T_min/max/mean.npy``  per-face statistics over the production run
``illumination_final.npy``      lit fraction of the solar disk, last step
``illumination_mean.npy``       time-averaged lit fraction
``sunlit_fraction.npy``         share of steps with ≥ half the disk visible
``face_centroids.npy``          (N, 3) mesh-frame centroids [m]
``face_normals.npy``            (N, 3) unit normals
``face_areas.npy``              (N,) areas [m²]
``dem_elevation.npy``           (ny, nx) input elevation [m]
``dem_x.npy`` / ``dem_y.npy``   vertex coordinates of the DEM grid [m]
``timeline.npz``                ``hours``, ``sun_elevation_deg``,
                                ``sun_azimuth_deg`` per output step
``probes.npz``                  one series per probe; ``<name>__face`` index
``snapshots_*.npy``             optional (frames, N) float32 maps
``metadata.json``               run settings, timing, provenance
``checksums.sha256``            SHA-256 of every file above
==============================  ==============================================

Per-face maps are stored as float32: that is ~10⁻⁵ K resolution at 300 K
and halves the size of every run folder.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from data_ingestion.provenance import sha256_file

if TYPE_CHECKING:
    from simulation.runner import SimulationResults

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2

_FACE_MAPS = {
    "surface_T_final": "final_surface_T",
    "surface_T_min": "min_surface_T",
    "surface_T_max": "max_surface_T",
    "surface_T_mean": "mean_surface_T",
    "illumination_final": "final_illumination",
    "illumination_mean": "mean_illumination",
    "sunlit_fraction": "sunlit_fraction",
}


class ResultsError(RuntimeError):
    """A run folder is missing, incomplete or from an unknown format."""


def save_results(output_dir: Path | str, results: SimulationResults) -> list[Path]:
    """Write a run folder and return the files written."""
    from artemis_thermal import __version__

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def put(name: str, array: np.ndarray) -> None:
        path = out / f"{name}.npy"
        np.save(path, array)
        written.append(path)

    for name, attr in _FACE_MAPS.items():
        put(name, np.asarray(getattr(results, attr), dtype=np.float32))
    put("face_centroids", results.face_centroids.astype(np.float32))
    put("face_normals", results.face_normals.astype(np.float32))
    put("face_areas", results.face_areas.astype(np.float32))
    put("dem_elevation", results.dem_elevation.astype(np.float32))
    put("dem_x", np.asarray(results.dem_x, dtype=np.float64))
    put("dem_y", np.asarray(results.dem_y, dtype=np.float64))

    start = datetime.fromisoformat(str(results.metadata["start_time"]))
    hours = np.array(
        [(t - start).total_seconds() / 3600.0 for t in results.times], dtype=np.float64
    )
    timeline = out / "timeline.npz"
    np.savez(
        timeline,
        hours=hours,
        sun_elevation_deg=np.asarray(results.sun_elevations, dtype=np.float64),
        sun_azimuth_deg=np.asarray(results.sun_azimuths, dtype=np.float64),
    )
    written.append(timeline)

    probes = out / "probes.npz"
    probe_arrays: dict[str, Any] = {
        name: np.asarray(series, dtype=np.float64)
        for name, series in results.probe_temps.items()
    }
    probe_arrays.update(
        {f"{name}__face": np.int64(index) for name, index in results.probe_faces.items()}
    )
    np.savez(probes, **probe_arrays)
    written.append(probes)

    for snapshot in ("snapshots_surface_T.npy", "snapshots_illumination.npy"):
        if (out / snapshot).is_file():
            written.append(out / snapshot)

    meta = dict(results.metadata)
    meta["format_version"] = FORMAT_VERSION
    meta["artemis_version"] = __version__
    meta["times_utc"] = [t.isoformat() for t in results.times]
    meta_path = out / "metadata.json"
    meta_path.write_text(
        json.dumps(_jsonable(meta), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    written.append(meta_path)

    checksum_path = out / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in sorted(written, key=lambda p: p.name)),
        encoding="ascii",
    )
    written.append(checksum_path)
    logger.info("Saved %d files to %s", len(written), out)
    return written


def _jsonable(obj: object) -> object:
    """Convert NumPy scalars/arrays and tuples to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@dataclass
class ResultsBundle:
    """Lazy, memory-mapped view of a run folder.

    Arrays are opened with ``mmap_mode="r"`` on first access, so opening a
    500k-face run costs almost nothing until a map is actually drawn.
    """

    path: Path
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    REQUIRED = ("surface_T_final", "face_centroids", "dem_x", "dem_y")

    @classmethod
    def open(cls, path: Path | str) -> ResultsBundle:
        """Open a run folder, raising :class:`ResultsError` if it is unusable."""
        folder = Path(path).expanduser()
        if not folder.is_dir():
            raise ResultsError(f"No run folder at {folder}.")
        meta_path = folder / "metadata.json"
        if not meta_path.is_file():
            raise ResultsError(f"{folder} has no metadata.json; it is not a run folder.")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ResultsError(f"metadata.json in {folder} is not valid JSON: {exc}") from exc
        if metadata.get("format_version") is None and (folder / "thermal_grid.npy").is_file():
            raise ResultsError(
                f"{folder} was written by ArtemisThermalBase 0.4 or earlier. "
                "Re-run the simulation with 0.5 to view it here."
            )
        missing = [name for name in cls.REQUIRED if not (folder / f"{name}.npy").is_file()]
        if missing:
            raise ResultsError(
                f"{folder} is incomplete; missing: {', '.join(n + '.npy' for n in missing)}."
            )
        return cls(path=folder, metadata=metadata)

    def has(self, name: str) -> bool:
        """True if ``<name>.npy`` exists in the folder."""
        return (self.path / f"{name}.npy").is_file()

    def array(self, name: str) -> np.ndarray:
        """Memory-mapped array ``<name>.npy``."""
        file = self.path / f"{name}.npy"
        if not file.is_file():
            raise ResultsError(f"{self.path.name}: {name}.npy is missing.")
        return np.load(file, mmap_mode="r")

    @cached_property
    def grid_shape(self) -> tuple[int, int]:
        """Cell grid (rows, cols)."""
        return (len(self.array("dem_y")) - 1, len(self.array("dem_x")) - 1)

    @cached_property
    def extent(self) -> tuple[float, float, float, float]:
        """(left, right, bottom, top) of the raster in mesh metres."""
        x = self.array("dem_x")
        y = self.array("dem_y")
        return (float(x.min()), float(x.max()), float(y.min()), float(y.max()))

    @cached_property
    def num_faces(self) -> int:
        """Number of mesh faces."""
        return int(self.array("surface_T_final").shape[0])

    def raster(self, name: str, frame: int | None = None) -> np.ndarray:
        """Per-face map ``name`` as a north-up (rows, cols) float32 raster.

        With ``frame``, ``name`` refers to a snapshot stack such as
        ``snapshots_surface_T``.
        """
        values = self.array(name)
        if frame is not None:
            values = values[frame]
        return faces_to_raster(np.asarray(values), self.grid_shape, self._y_ascending)

    def raster_from(self, values: np.ndarray) -> np.ndarray:
        """Turn any per-face array of this run into a north-up raster."""
        return faces_to_raster(values, self.grid_shape, self._y_ascending)

    @cached_property
    def _y_ascending(self) -> bool:
        y = self.array("dem_y")
        return bool(y[-1] > y[0])

    def elevation(self) -> np.ndarray:
        """DEM elevation averaged to cell centres, north-up, same shape as rasters."""
        z = np.asarray(self.array("dem_elevation"), dtype=np.float32)
        cells = 0.25 * (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:])
        return cells[::-1] if self._y_ascending else cells

    @cached_property
    def timeline(self) -> dict[str, np.ndarray]:
        """Output-step timeline arrays (may be empty)."""
        path = self.path / "timeline.npz"
        if not path.is_file():
            return {"hours": np.array([]), "sun_elevation_deg": np.array([]),
                    "sun_azimuth_deg": np.array([])}
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    @cached_property
    def probes(self) -> dict[str, tuple[int, np.ndarray]]:
        """Probe name → (face index, temperature series)."""
        path = self.path / "probes.npz"
        if not path.is_file():
            return {}
        out: dict[str, tuple[int, np.ndarray]] = {}
        with np.load(path) as data:
            for key in data.files:
                if key.endswith("__face"):
                    continue
                face = int(data[f"{key}__face"]) if f"{key}__face" in data.files else -1
                out[key] = (face, data[key])
        return out

    @property
    def snapshot_count(self) -> int:
        """Number of stored snapshot frames (0 if none)."""
        if not self.has("snapshots_surface_T"):
            return 0
        written = self.metadata.get("snapshot_frames")
        stored = self.array("snapshots_surface_T").shape[0]
        return int(min(written, stored)) if written is not None else int(stored)

    def face_at(self, x: float, y: float) -> int:
        """Index of the lower-left triangle of the cell containing (x, y)."""
        rows, cols = self.grid_shape
        xs = self.array("dem_x")
        ys = self.array("dem_y")
        col = int(np.clip(np.searchsorted(xs, x) - 1, 0, cols - 1))
        if self._y_ascending:
            row = int(np.clip(np.searchsorted(ys, y) - 1, 0, rows - 1))
        else:
            row = int(np.clip(np.searchsorted(-ys, -y) - 1, 0, rows - 1))
        return 2 * (row * cols + col)


def faces_to_raster(
    values: np.ndarray,
    grid_shape: tuple[int, int],
    y_ascending: bool,
) -> np.ndarray:
    """Average the two triangles of each DEM cell into a north-up raster."""
    rows, cols = grid_shape
    grid = np.asarray(values, dtype=np.float32).reshape(rows, cols, 2).mean(axis=2)
    return grid[::-1] if y_ascending else grid


def list_runs(root: Path | str) -> list[Path]:
    """Run folders directly under ``root`` (and ``root`` itself), newest first."""
    base = Path(root)
    if not base.is_dir():
        return []
    found = [p for p in [base, *base.iterdir()] if (p / "metadata.json").is_file()]
    return sorted(found, key=lambda p: (p / "metadata.json").stat().st_mtime, reverse=True)
