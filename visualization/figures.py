"""Report figures for a finished run.

Every figure is drawn from rasters (two triangles per DEM cell averaged
into one pixel), never from scattered face centroids, so a 500k-face run
renders in about a second and the PNG shows the actual grid.

Figures written by :func:`render_run_figures`:

``temperature.png``   final and maximum surface temperature
``illumination.png``  time-averaged lit fraction over relief
``ice_stability.png`` cold-trap classes from the maximum temperature
``timeline.png``      probe temperatures and Sun elevation
``hero.png``          a single wide plate for the README and the docs

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.text

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402

from artemis_thermal import style  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from simulation.io_manager import ResultsBundle

logger = logging.getLogger(__name__)

_FONTS_READY = False


def _setup(dark: bool = False) -> style.Palette:
    global _FONTS_READY
    if not _FONTS_READY:
        for path in style.font_files():
            font_manager.fontManager.addfont(str(path))
        _FONTS_READY = True
    pal = style.palette(dark)
    family = style.SANS if style.font_files() else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": 8.5,
        "axes.facecolor": pal.paper,
        "figure.facecolor": pal.paper,
        "savefig.facecolor": pal.paper,
        "axes.edgecolor": pal.rule_strong,
        "axes.linewidth": 0.6,
        "axes.labelcolor": pal.ink_2,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlecolor": pal.ink,
        "axes.titlepad": 6,
        "xtick.color": pal.ink_3,
        "ytick.color": pal.ink_3,
        "xtick.labelcolor": pal.ink_2,
        "ytick.labelcolor": pal.ink_2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "text.color": pal.ink,
        "lines.linewidth": 1.1,
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
    })
    return pal


def colormap(name: str) -> LinearSegmentedColormap:
    """Matplotlib colormap from the shared stops (``temperature`` etc.)."""
    stops = {
        "temperature": style.TEMPERATURE_STOPS,
        "illumination": style.ILLUMINATION_STOPS,
        "relief": style.RELIEF_STOPS,
        "diverging": style.DIVERGING_STOPS,
    }[name]
    return LinearSegmentedColormap.from_list(f"artemis_{name}", list(stops), N=256)


def hillshade(elevation: np.ndarray, cell_m: float, azimuth_deg: float = 315.0,
              altitude_deg: float = 40.0) -> np.ndarray:
    """Lambertian hillshade in [0, 1] of a north-up elevation grid."""
    dzdy, dzdx = np.gradient(elevation.astype(np.float64), cell_m)
    dzdy = -dzdy  # row index increases southward
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, dzdy)
    az = np.radians(azimuth_deg)
    alt = np.radians(altitude_deg)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0.0, 1.0)


def _cell_m(bundle: ResultsBundle) -> float:
    return float(bundle.metadata.get("resolution_m") or 1.0)


def _km_extent(bundle: ResultsBundle) -> tuple[float, float, float, float]:
    left, right, bottom, top = bundle.extent
    return (left / 1e3, right / 1e3, bottom / 1e3, top / 1e3)


def _frame_axes(ax: Axes) -> None:
    ax.set_xlabel("x  [km]")
    ax.set_ylabel("y  [km]")
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)


def _caption(fig: Figure, number: int, text: str, pal: style.Palette) -> None:
    label = fig.text(0.012, 0.018, f"Fig. {number}", fontsize=7.5, color=pal.accent_ink,
                     fontweight="medium", va="bottom", ha="left")
    fig.add_artist(matplotlib.text.Annotation(
        text, xy=(1.0, 0.0), xycoords=label, xytext=(8, 0), textcoords="offset points",
        fontsize=7.5, color=pal.ink_2, va="bottom", ha="left",
    ))


def _site_label(bundle: ResultsBundle) -> str:
    frame = bundle.metadata.get("frame", {})
    lat = frame.get("latitude_deg")
    lon = frame.get("longitude_deg")
    terrain = "synthetic bowl" if bundle.metadata.get("dem_type") in (
        "parabolic_bowl", "conical", "flat") else "LOLA DEM"
    cell = bundle.metadata.get("resolution_m")
    where = f"{abs(lat):.2f}°{'S' if lat < 0 else 'N'} {lon % 360:.2f}°E" if lat is not None else ""
    text = ", ".join(x for x in (terrain, f"{cell:.0f} m cells" if cell else "", where) if x)
    return text[:1].upper() + text[1:]


def _colorbar(fig: Figure, mappable, ax, label: str, pal: style.Palette, **kw) -> None:
    if "cax" not in kw:
        kw.update(ax=ax, fraction=0.035, pad=0.02)
    cbar = fig.colorbar(mappable, **kw)
    outline: Any = cbar.outline
    outline.set_linewidth(0.5)
    outline.set_edgecolor(pal.rule_strong)
    cbar.ax.tick_params(labelsize=7, length=2, width=0.5)
    cbar.set_label(label, color=pal.ink_2, fontsize=7.5)


def _save(fig: Figure, path: Path, dpi: int) -> Path:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def figure_temperature(bundle: ResultsBundle, path: Path, dpi: int) -> Path:
    """Final and maximum surface temperature side by side."""
    pal = _setup()
    final = bundle.raster("surface_T_final")
    hottest = bundle.raster("surface_T_max") if bundle.has("surface_T_max") else final
    vmin = float(np.floor(min(np.nanmin(final), np.nanmin(hottest)) / 10.0) * 10.0)
    vmax = float(np.ceil(np.nanmax(hottest) / 10.0) * 10.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    cmap = colormap("temperature")
    extent = _km_extent(bundle)
    hours = bundle.timeline["hours"]
    t_end = f"t = {hours[-1]:.1f} h" if hours.size else "last step"
    for ax, grid, title in ((axes[0], final, f"Surface temperature, {t_end}"),
                            (axes[1], hottest, "Maximum over the run")):
        image = ax.imshow(grid, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation="nearest")
        ax.set_title(title)
        _frame_axes(ax)
    axes[1].set_ylabel("")
    fig.subplots_adjust(left=0.07, right=0.9, top=0.9, bottom=0.17)
    box = axes[1].get_position()
    cax = fig.add_axes((0.915, box.y0, 0.012, box.height))
    _colorbar(fig, image, None, "K", pal, cax=cax)
    _caption(fig, 1, f"Surface temperature. {_site_label(bundle)}.", pal)
    return _save(fig, path, dpi)


def figure_illumination(bundle: ResultsBundle, path: Path, dpi: int) -> Path:
    """Time-averaged lit fraction over a hillshade of the terrain."""
    pal = _setup()
    lit = bundle.raster("illumination_mean" if bundle.has("illumination_mean")
                        else "illumination_final")
    shade = hillshade(bundle.elevation(), _cell_m(bundle))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    extent = _km_extent(bundle)
    ax.imshow(shade, extent=extent, cmap=colormap("relief"), vmin=0, vmax=1,
              interpolation="bilinear")
    image = ax.imshow(lit, extent=extent, cmap=colormap("illumination"), vmin=0, vmax=1,
                      alpha=0.82, interpolation="nearest")
    never = np.ma.masked_where(lit > 0.0, np.ones_like(lit))
    ax.imshow(never, extent=extent, cmap=ListedColormap([pal.cold]), alpha=0.55,
              interpolation="nearest")
    ax.set_title("Mean illuminated fraction of the solar disk")
    _frame_axes(ax)
    _colorbar(fig, image, ax, "fraction", pal)
    dark_pct = 100.0 * float(np.mean(lit <= 0.0))
    fig.subplots_adjust(left=0.1, right=0.88, top=0.92, bottom=0.15)
    _caption(fig, 2, f"Illumination. Blue: never lit during the run ({dark_pct:.1f} % of the "
             "area).", pal)
    return _save(fig, path, dpi)


def figure_ice(bundle: ResultsBundle, path: Path, dpi: int) -> Path:
    """Cold-trap classes from the maximum surface temperature."""
    from thermal_solver.volatiles import compute_cold_trap_map

    pal = _setup()
    stable_k, marginal_k = bundle.metadata.get("volatile_thresholds_K", [110.0, 115.0])
    t_max = np.asarray(bundle.array("surface_T_max" if bundle.has("surface_T_max")
                                    else "surface_T_final"), dtype=np.float64)
    classes = compute_cold_trap_map(t_max, float(stable_k), float(marginal_k))
    areas = np.asarray(bundle.array("face_areas"), dtype=np.float64)
    km2 = [float(areas[classes == k].sum()) / 1e6 for k in range(3)]
    grid = np.rint(bundle.raster_from(classes.astype(np.float32))).astype(np.int8)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    extent = _km_extent(bundle)
    shade = hillshade(bundle.elevation(), _cell_m(bundle))
    ax.imshow(shade, extent=extent, cmap=colormap("relief"), vmin=0, vmax=1,
              interpolation="bilinear")
    cmap = ListedColormap([pal.cold, pal.cold_soft, pal.paper])
    ax.imshow(np.ma.masked_equal(grid, 2), extent=extent, cmap=cmap,
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), interpolation="nearest", alpha=0.92)
    ax.set_title("Water-ice stability against sublimation")
    _frame_axes(ax)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=pal.cold, label=f"T_max < {stable_k:.0f} K   {km2[0]:.2f} km²"),
            Patch(facecolor=pal.cold_soft,
                  label=f"{stable_k:.0f}–{marginal_k:.0f} K   {km2[1]:.2f} km²"),
            Patch(facecolor=pal.sheet, edgecolor=pal.rule_strong, linewidth=0.5,
                  label=f"≥ {marginal_k:.0f} K   {km2[2]:.2f} km²"),
        ],
        loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.2, borderaxespad=0,
    )
    fig.subplots_adjust(left=0.1, right=0.66, top=0.92, bottom=0.15)
    _caption(fig, 3, "Cold traps, classified by the hottest temperature each cell reached.",
             pal)
    return _save(fig, path, dpi)


def figure_timeline(bundle: ResultsBundle, path: Path, dpi: int) -> Path:
    """Probe temperatures over time, with the Sun's elevation underneath."""
    pal = _setup()
    probes = bundle.probes
    timeline = bundle.timeline
    dt_h = float(bundle.metadata.get("dt_s", 300.0)) / 3600.0
    fig, (ax_t, ax_s) = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})
    shades = [pal.ink, pal.accent, pal.cold]
    for (name, (_face, series)), color in zip(probes.items(), shades, strict=False):
        hours = (np.arange(series.size) + 1) * dt_h
        ax_t.plot(hours, series, color=color, label=name.replace("_", " "))
    ax_t.set_ylabel("surface T  [K]")
    ax_t.set_title("Probe temperatures")
    ax_t.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=3, borderaxespad=0.2)
    if timeline["hours"].size:
        ax_s.plot(timeline["hours"], timeline["sun_elevation_deg"], color=pal.accent_ink)
        ax_s.axhline(0.0, color=pal.rule_strong, linewidth=0.5)
    ax_s.set_ylabel("Sun el.  [°]")
    ax_s.set_xlabel("time since start  [h]")
    for ax in (ax_t, ax_s):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.17)
    _caption(fig, 4, "Temperature at the lowest, highest and mid-height faces.", pal)
    return _save(fig, path, dpi)


def _nice_length(span_m: float) -> float:
    target = span_m / 5.0
    exponent = np.floor(np.log10(target))
    for step in (1.0, 2.0, 5.0, 10.0):
        if step * 10**exponent >= target:
            return float(step * 10**exponent)
    return float(10 ** (exponent + 1))


def figure_hero(bundle: ResultsBundle, path: Path, dpi: int) -> Path:
    """Wide plate: maximum temperature draped on relief, with a data block."""
    pal = _setup(dark=True)
    t_grid = bundle.raster("surface_T_max" if bundle.has("surface_T_max") else "surface_T_final")
    shade = hillshade(bundle.elevation(), _cell_m(bundle), azimuth_deg=300.0, altitude_deg=25.0)
    vmin, vmax = float(np.nanpercentile(t_grid, 1)), float(np.nanpercentile(t_grid, 99.5))
    rgb = colormap("temperature")(np.clip((t_grid - vmin) / max(vmax - vmin, 1e-6), 0, 1))[..., :3]
    rgb = rgb * (0.55 + 0.45 * shade[..., None])

    fig = plt.figure(figsize=(12.0, 6.0))
    ax = fig.add_axes((0.03, 0.08, 0.6, 0.84))
    left, right, bottom, top = bundle.extent
    ax.imshow(rgb, extent=(left, right, bottom, top), interpolation="bilinear")
    ax.set_axis_off()

    bar = _nice_length(right - left)
    x0 = left + 0.05 * (right - left)
    y0 = bottom + 0.05 * (top - bottom)
    ax.plot([x0, x0 + bar], [y0, y0], color=pal.ink, linewidth=1.4, solid_capstyle="butt")
    ax.text(x0, y0 + 0.02 * (top - bottom),
            f"{bar / 1000:g} km" if bar >= 1000 else f"{bar:g} m",
            color=pal.ink, fontsize=8, family=style.MONO if style.font_files() else None)

    meta = bundle.metadata
    frame = meta.get("frame", {})
    lat, lon = frame.get("latitude_deg"), frame.get("longitude_deg")
    mono = style.MONO if style.font_files() else "DejaVu Sans Mono"
    fig.text(0.67, 0.86, "ARTEMIS THERMAL BASE", fontsize=9, color=pal.accent,
             fontweight="medium")
    fig.text(0.67, 0.78, "Hottest surface\ntemperature of the run", fontsize=17, color=pal.ink,
             linespacing=1.15, va="top")
    rows = [
        ("site", f"{abs(lat):.2f}°{'S' if lat < 0 else 'N'}  {lon % 360:.2f}°E"
         if lat is not None else "–"),
        ("terrain", _site_label(bundle).split(",")[0].replace("Synthetic", "synthetic")),
        ("faces", f"{bundle.num_faces:,}".replace(",", " ")),
        ("span", f"{meta.get('duration_hours', 0):.1f} h, dt {meta.get('dt_s', 0):.0f} s"),
        ("Sun", "point" if meta.get("point_source_mode") else "disk, penumbra"),
        ("T range", f"{np.nanmin(t_grid):.0f} – {np.nanmax(t_grid):.0f} K"),
    ]
    for i, (key, value) in enumerate(rows):
        y = 0.56 - i * 0.052
        fig.text(0.67, y, key, fontsize=8, color=pal.ink_3)
        fig.text(0.76, y, value, fontsize=8.5, color=pal.ink, family=mono)
    fig.add_artist(plt.Line2D([0.67, 0.96], [0.6, 0.6], color=pal.rule_strong, linewidth=0.6))

    cax = fig.add_axes((0.67, 0.13, 0.27, 0.022))
    sm = plt.cm.ScalarMappable(cmap=colormap("temperature"), norm=plt.Normalize(vmin, vmax))
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    hero_outline: Any = cbar.outline
    hero_outline.set_linewidth(0)
    cbar.ax.tick_params(labelsize=7, length=2, colors=pal.ink_2)
    cbar.set_label("K", color=pal.ink_2, fontsize=7.5, loc="right")
    return _save(fig, path, dpi)


def render_run_figures(
    bundle: ResultsBundle,
    out_dir: Path | str,
    dpi: int = 150,
    hero_dpi: int = 200,
) -> list[Path]:
    """Write every report figure for ``bundle`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [
        figure_temperature(bundle, out / "temperature.png", dpi),
        figure_illumination(bundle, out / "illumination.png", dpi),
        figure_ice(bundle, out / "ice_stability.png", dpi),
        figure_timeline(bundle, out / "timeline.png", dpi),
        figure_hero(bundle, out / "hero.png", hero_dpi),
    ]
    logger.info("Rendered %d figures into %s", len(written), out)
    return written
