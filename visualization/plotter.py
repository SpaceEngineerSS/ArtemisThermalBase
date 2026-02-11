"""Visualization module for illumination and thermal maps.

Generates publication-quality figures using matplotlib:
- Illumination maps (grayscale)
- Surface temperature maps (inferno colormap)
- Temperature vs. time series for probe locations
- Crater rim overlay for spatial context

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color Configuration
# ---------------------------------------------------------------------------

_ILLUMINATION_CMAP = "gray"
_THERMAL_CMAP = "inferno"
_THERMAL_VMIN = 40.0   # K
_THERMAL_VMAX = 300.0  # K
_DPI = 150


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_illumination_map(
    face_centroids: np.ndarray,
    illumination: np.ndarray,
    title: str = "Illumination Map",
    output_path: Path | str | None = None,
    dpi: int = _DPI,
) -> plt.Figure:
    """Plot a 2D illumination map (grayscale).

    Parameters
    ----------
    face_centroids : np.ndarray
        Face centroid (x, y, z) coordinates. Shape: (N, 3).
    illumination : np.ndarray
        Per-face illumination fraction [0, 1]. Shape: (N,).
    title : str
        Figure title.
    output_path : Path or str, optional
        If provided, save figure to this path.
    dpi : int
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    x = face_centroids[:, 0]
    y = face_centroids[:, 1]

    scatter = ax.scatter(
        x, y,
        c=illumination,
        cmap=_ILLUMINATION_CMAP,
        s=0.5,
        vmin=0.0,
        vmax=1.0,
        edgecolors="none",
        rasterized=True,
    )

    cbar = fig.colorbar(scatter, ax=ax, label="Illumination Fraction", shrink=0.8)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    ax.set_xlabel("X [m]", color="white")
    ax.set_ylabel("Y [m]", color="white")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax.set_aspect("equal")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
        logger.info("Illumination map saved: %s", output_path)

    plt.close(fig)
    return fig


def plot_thermal_map(
    face_centroids: np.ndarray,
    surface_temp: np.ndarray,
    title: str = "Surface Temperature [K]",
    output_path: Path | str | None = None,
    vmin: float = _THERMAL_VMIN,
    vmax: float = _THERMAL_VMAX,
    dpi: int = _DPI,
) -> plt.Figure:
    """Plot a 2D surface temperature map (inferno colormap).

    Parameters
    ----------
    face_centroids : np.ndarray
        Face centroid coordinates. Shape: (N, 3).
    surface_temp : np.ndarray
        Surface temperature per face [K]. Shape: (N,).
    title : str
        Figure title.
    output_path : Path or str, optional
        If provided, save figure to this path.
    vmin, vmax : float
        Temperature range for colorbar.
    dpi : int
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    x = face_centroids[:, 0]
    y = face_centroids[:, 1]

    scatter = ax.scatter(
        x, y,
        c=surface_temp,
        cmap=_THERMAL_CMAP,
        s=0.5,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        rasterized=True,
    )

    cbar = fig.colorbar(scatter, ax=ax, label="Temperature [K]", shrink=0.8)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    ax.set_xlabel("X [m]", color="white")
    ax.set_ylabel("Y [m]", color="white")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax.set_aspect("equal")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
        logger.info("Thermal map saved: %s", output_path)

    plt.close(fig)
    return fig


def plot_time_series(
    times_hours: np.ndarray | list[float],
    probe_data: dict[str, list[float]],
    title: str = "Temperature vs. Time",
    output_path: Path | str | None = None,
    dpi: int = _DPI,
) -> plt.Figure:
    """Plot temperature vs. time for probe locations.

    Parameters
    ----------
    times_hours : array-like
        Time axis in hours.
    probe_data : dict[str, list[float]]
        Probe name → temperature time series [K].
    title : str
        Figure title.
    output_path : Path or str, optional
        If provided, save figure to this path.
    dpi : int
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    colors = ["#ff6b6b", "#51cf66", "#69db7c", "#ffd43b", "#748ffc", "#e599f7"]

    for i, (name, temps) in enumerate(probe_data.items()):
        color = colors[i % len(colors)]
        # Resample to match time axis if needed
        t = np.linspace(0, times_hours[-1] if len(times_hours) > 0 else 1.0, len(temps))
        ax.plot(t, temps, label=name, color=color, linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Time [hours]", color="white", fontsize=12)
    ax.set_ylabel("Temperature [K]", color="white", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, color="white")

    legend = ax.legend(facecolor="#1a1a2e", edgecolor="#444", fontsize=10)
    for text in legend.get_texts():
        text.set_color("white")

    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
        logger.info("Time series saved: %s", output_path)

    plt.close(fig)
    return fig


def plot_sun_elevation(
    times_hours: np.ndarray | list[float],
    sun_elevations: list[float],
    title: str = "Sun Elevation vs. Time",
    output_path: Path | str | None = None,
    dpi: int = _DPI,
) -> plt.Figure:
    """Plot sun elevation angle vs. time.

    Parameters
    ----------
    times_hours : array-like
        Time axis in hours.
    sun_elevations : list[float]
        Sun elevation in degrees at each output snapshot.
    title : str
        Figure title.
    output_path : Path or str, optional
        If provided, save figure to output path.
    dpi : int
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    ax.plot(
        times_hours, sun_elevations,
        color="#ffd43b", linewidth=2.0, label="Sun Elevation",
    )
    ax.axhline(0.0, color="#555", linestyle="--", linewidth=0.8, label="Horizon")

    ax.fill_between(
        times_hours, sun_elevations, 0,
        where=[e > 0 for e in sun_elevations],
        color="#ffd43b", alpha=0.15,
    )

    ax.set_xlabel("Time [hours]", color="white", fontsize=12)
    ax.set_ylabel("Elevation [°]", color="white", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, color="white")

    legend = ax.legend(facecolor="#1a1a2e", edgecolor="#444")
    for text in legend.get_texts():
        text.set_color("white")

    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
        logger.info("Sun elevation plot saved: %s", output_path)

    plt.close(fig)
    return fig


def generate_all_plots(
    results: "SimulationResults",
    output_dir: Path | str = "output",
    dpi: int = _DPI,
) -> list[Path]:
    """Generate all standard plots from simulation results.

    Parameters
    ----------
    results : SimulationResults
        Full simulation results.
    output_dir : Path or str
        Directory for output plots.
    dpi : int
        Figure resolution.

    Returns
    -------
    list[Path]
        Paths to all generated plot files.
    """
    # Lazy import to avoid circular
    from simulation.runner import SimulationResults

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Final illumination map
    if results.illumination_maps:
        p = output_dir / "illumination_map.png"
        plot_illumination_map(
            results.face_centroids,
            results.illumination_maps[-1],
            title="Final Illumination Map",
            output_path=p,
            dpi=dpi,
        )
        saved.append(p)

    # 2. Final thermal map
    if results.surface_temps:
        p = output_dir / "thermal_map.png"
        plot_thermal_map(
            results.face_centroids,
            results.surface_temps[-1],
            title="Final Surface Temperature [K]",
            output_path=p,
            dpi=dpi,
        )
        saved.append(p)

    # 3. Temperature time series
    if results.probe_temps:
        dt_s = results.metadata.get("dt_s", 600.0)
        max_steps = max(len(v) for v in results.probe_temps.values())
        times_hours = [i * dt_s / 3600.0 for i in range(max_steps)]

        p = output_dir / "time_series.png"
        plot_time_series(
            times_hours,
            results.probe_temps,
            title="Temperature Probes vs. Time",
            output_path=p,
            dpi=dpi,
        )
        saved.append(p)

    # 4. Sun elevation
    if results.sun_elevations and results.times:
        t0 = results.times[0]
        times_hours = [(t - t0).total_seconds() / 3600.0 for t in results.times]
        p = output_dir / "sun_elevation.png"
        plot_sun_elevation(
            times_hours,
            results.sun_elevations,
            title="Sun Elevation vs. Time",
            output_path=p,
            dpi=dpi,
        )
        saved.append(p)

    logger.info("Generated %d plots in %s", len(saved), output_dir)
    return saved
