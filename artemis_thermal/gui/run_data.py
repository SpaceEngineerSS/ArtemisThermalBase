"""Turn a run folder into map layers, summary numbers and site reports.

Nothing here touches Qt widgets, so it can be tested and reused headless.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from artemis_thermal import style
from artemis_thermal.simulation.io_manager import ResultsBundle
from artemis_thermal.thermal_solver.volatiles import compute_cold_trap_map

SOLAR_CONSTANT = 1361.0

LAYER_KEYS = (
    ("temperature", "Surface T"),
    ("t_max", "Max T"),
    ("t_mean", "Mean T"),
    ("sunlit", "Sunlit time"),
    ("ice", "Ice stability"),
    ("relief", "Relief"),
)


@dataclass(frozen=True)
class LayerSpec:
    """Raw data and colouring for one layer, before it becomes a Qt Layer."""

    key: str
    title: str
    data: np.ndarray
    stops: tuple[str, ...]
    levels: tuple[float, float]
    unit: str
    decimals: int
    ticks: tuple[tuple[float, str], ...] = ()
    over_relief: bool = False


def _nice_levels(*arrays: np.ndarray, step: float = 10.0) -> tuple[float, float]:
    lo = min(float(np.nanmin(a)) for a in arrays)
    hi = max(float(np.nanmax(a)) for a in arrays)
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
    return (lo, hi if hi > lo else lo + step)


def thresholds(bundle: ResultsBundle) -> tuple[float, float]:
    """Cold-trap thresholds recorded with the run."""
    stable, marginal = bundle.metadata.get("volatile_thresholds_K", [110.0, 115.0])
    return float(stable), float(marginal)


def ice_classes(bundle: ResultsBundle) -> np.ndarray:
    """Per-face stability class from the run's maximum temperature."""
    name = "surface_T_max" if bundle.has("surface_T_max") else "surface_T_final"
    t_max = np.asarray(bundle.array(name), dtype=np.float64)
    return compute_cold_trap_map(t_max, *thresholds(bundle))


def hillshade(bundle: ResultsBundle) -> np.ndarray:
    """Hillshade of the run's DEM on the raster grid."""
    from artemis_thermal.visualization.figures import hillshade as _hillshade

    return _hillshade(bundle.elevation(), float(bundle.metadata.get("resolution_m") or 1.0))


def build_layer(bundle: ResultsBundle, key: str, pal: style.Palette) -> LayerSpec:
    """Build one layer by key (see ``LAYER_KEYS``)."""
    temp_stops = style.TEMPERATURE_STOPS
    if key in ("temperature", "t_max", "t_mean"):
        final = bundle.raster("surface_T_final")
        t_max = bundle.raster("surface_T_max") if bundle.has("surface_T_max") else final
        levels = _nice_levels(final, t_max)
        data = {"temperature": final, "t_max": t_max,
                "t_mean": bundle.raster("surface_T_mean") if bundle.has("surface_T_mean")
                else final}[key]
        title = {"temperature": "Surface temperature, last step",
                 "t_max": "Maximum surface temperature over the run",
                 "t_mean": "Mean surface temperature over the run"}[key]
        return LayerSpec(key, title, data, temp_stops, levels, "K", 0)
    if key == "sunlit":
        name = "sunlit_fraction" if bundle.has("sunlit_fraction") else "illumination_mean"
        return LayerSpec(key, "Share of the run with at least half the Sun visible",
                         bundle.raster(name) * 100.0, style.ILLUMINATION_STOPS, (0.0, 100.0),
                         "%", 0, over_relief=True)
    if key == "ice":
        stable, marginal = thresholds(bundle)
        grid = np.rint(bundle.raster_from(ice_classes(bundle).astype(np.float32)))
        return LayerSpec(
            key, "Water-ice stability from the hottest temperature reached", grid,
            (pal.cold, pal.cold_soft, pal.sheet), (-0.5, 2.5), "", 0,
            ticks=((0, f"< {stable:.0f} K stable"), (1, f"{stable:.0f}–{marginal:.0f} K"),
                   (2, f"≥ {marginal:.0f} K")),
            over_relief=True,
        )
    if key == "relief":
        z = bundle.elevation()
        return LayerSpec(key, "Elevation", z, style.RELIEF_STOPS,
                         _nice_levels(z, step=100.0), "m", 0)
    raise KeyError(key)


@dataclass(frozen=True)
class Summary:
    """Headline numbers for a run."""

    t_min: float
    t_max: float
    t_mean_area: float
    never_lit_km2: float
    stable_km2: float
    marginal_km2: float
    sunlit_half_km2: float
    area_km2: float


def summarize(bundle: ResultsBundle) -> Summary:
    """Area-weighted headline numbers."""
    areas = np.asarray(bundle.array("face_areas"), dtype=np.float64)
    final = np.asarray(bundle.array("surface_T_final"), dtype=np.float64)
    t_max = np.asarray(bundle.array("surface_T_max"), dtype=np.float64) \
        if bundle.has("surface_T_max") else final
    t_min = np.asarray(bundle.array("surface_T_min"), dtype=np.float64) \
        if bundle.has("surface_T_min") else final
    mean = np.asarray(bundle.array("surface_T_mean"), dtype=np.float64) \
        if bundle.has("surface_T_mean") else final
    lit_name = "illumination_mean" if bundle.has("illumination_mean") else "illumination_final"
    lit = np.asarray(bundle.array(lit_name), dtype=np.float64)
    sunlit = np.asarray(bundle.array("sunlit_fraction"), dtype=np.float64) \
        if bundle.has("sunlit_fraction") else lit
    classes = ice_classes(bundle)
    km2 = 1e-6
    return Summary(
        t_min=float(t_min.min()),
        t_max=float(t_max.max()),
        t_mean_area=float(np.sum(mean * areas) / np.sum(areas)),
        never_lit_km2=float(areas[lit <= 0.0].sum() * km2),
        stable_km2=float(areas[classes == 0].sum() * km2),
        marginal_km2=float(areas[classes == 1].sum() * km2),
        sunlit_half_km2=float(areas[sunlit >= 0.5].sum() * km2),
        area_km2=float(areas.sum() * km2),
    )


@dataclass(frozen=True)
class SiteReport:
    """What we know about one DEM cell."""

    x_m: float
    y_m: float
    face: int
    elevation_m: float
    slope_deg: float
    aspect_deg: float
    t_final: float
    t_min: float
    t_max: float
    t_mean: float
    sunlit_pct: float
    tracking_flux: float
    ice_class: int
    longest_dark_h: float | None
    hours: np.ndarray
    series: np.ndarray


def _cell_mean(bundle: ResultsBundle, name: str, face: int) -> float:
    if not bundle.has(name):
        return float("nan")
    values = bundle.array(name)
    return float(0.5 * (float(values[face]) + float(values[face + 1])))


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def site_report(bundle: ResultsBundle, x_m: float, y_m: float) -> SiteReport:
    """Facts for the DEM cell under (x, y)."""
    face = bundle.face_at(x_m, y_m)
    normals = np.asarray(bundle.array("face_normals")[face : face + 2], dtype=np.float64)
    normal = normals.mean(axis=0)
    normal /= np.linalg.norm(normal)
    slope = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    aspect = float(np.degrees(np.arctan2(normal[0], normal[1])) % 360.0)

    rows, cols = bundle.grid_shape
    cell = face // 2
    row, col = divmod(cell, cols)
    elev = bundle.elevation()
    ascending = bool(bundle.array("dem_y")[-1] > bundle.array("dem_y")[0])
    elevation = float(elev[rows - 1 - row, col] if ascending else elev[row, col])

    lit_mean = _cell_mean(bundle, "illumination_mean", face)
    sunlit = _cell_mean(bundle, "sunlit_fraction", face)
    stable, marginal = thresholds(bundle)
    t_max = _cell_mean(bundle, "surface_T_max", face)
    ice = 0 if t_max < stable else (1 if t_max < marginal else 2)

    hours = np.array([])
    series = np.array([])
    longest = None
    count = bundle.snapshot_count
    if count:
        stride = int(bundle.metadata.get("snapshot_stride", 1))
        all_hours = bundle.timeline["hours"]
        hours = all_hours[::stride][:count] if all_hours.size else np.arange(count, dtype=float)
        temps = bundle.array("snapshots_surface_T")[:count, face : face + 2]
        series = np.asarray(temps, dtype=np.float64).mean(axis=1)
        if bundle.has("snapshots_illumination"):
            illum = np.asarray(bundle.array("snapshots_illumination")[:count, face : face + 2],
                               dtype=np.float64).mean(axis=1)
            step_h = float(np.median(np.diff(hours))) if hours.size > 1 else 0.0
            longest = _longest_run(illum < 0.5) * step_h

    return SiteReport(
        x_m=x_m, y_m=y_m, face=face, elevation_m=elevation, slope_deg=slope,
        aspect_deg=aspect,
        t_final=_cell_mean(bundle, "surface_T_final", face),
        t_min=_cell_mean(bundle, "surface_T_min", face),
        t_max=t_max,
        t_mean=_cell_mean(bundle, "surface_T_mean", face),
        sunlit_pct=100.0 * sunlit,
        tracking_flux=SOLAR_CONSTANT * lit_mean,
        ice_class=ice,
        longest_dark_h=longest,
        hours=hours,
        series=series,
    )


@lru_cache(maxsize=4)
def _open(path: str, stamp: float) -> ResultsBundle:
    return ResultsBundle.open(path)


def open_bundle(path: str) -> ResultsBundle:
    """:meth:`ResultsBundle.open`, cached until the folder is rewritten."""
    from pathlib import Path

    meta = Path(path) / "metadata.json"
    stamp = meta.stat().st_mtime if meta.is_file() else 0.0
    return _open(str(Path(path).resolve()), stamp)


def describe(bundle: ResultsBundle) -> str:
    """One line describing the run for page subtitles."""
    m = bundle.metadata
    frame = m.get("frame", {})
    lat, lon = frame.get("latitude_deg"), frame.get("longitude_deg")
    site = f"{abs(lat):.2f}°{'S' if lat < 0 else 'N'} {lon % 360:.2f}°E" if lat is not None \
        else "unknown site"
    terrain = "LOLA DEM" if m.get("dem_type") == "lola_geotiff" else "synthetic bowl"
    sun = "point Sun" if m.get("point_source_mode") else "solar disk"
    when = str(m.get("start_time", ""))[:16].replace("T", " ")
    return (f"{terrain}, {m.get('resolution_m', 0):.0f} m cells · {site} · "
            f"{m.get('duration_hours', 0):.1f} h from {when} UTC · {m.get('ephemeris_mode', '')}"
            f" ephemeris, {sun}")
