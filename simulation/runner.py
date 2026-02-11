"""Simulation Runner — time-stepping loop for illumination + thermal.

Orchestrates the full simulation pipeline:
1. Generate / load DEM → mesh → BVH
2. Initialize thermal columns (one per face)
3. Time loop: ephemeris → illumination → heat equation → store results
4. Return results for visualization

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

Notes
-----
The Q_solar absorbed by each face is:

    Q_solar = (1 − A) · S₀ · cos(θ_incidence) · f_illum

where:
- A = bond albedo
- S₀ = solar constant (1361 W/m²)
- θ_incidence = angle between face normal and sun direction
- f_illum = illumination fraction [0, 1] from raytracer
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from core_engine.constants import SimulationConfig, load_config
from core_engine.illumination import IlluminationEngine, IlluminationResult
from core_engine.mesh import TriangleMesh, dem_to_mesh
from data_ingestion.synthetic_dem import generate_synthetic_dem
from thermal_solver.crank_nicolson import (
    CrankNicolsonSolver,
    ThermalColumn,
    create_thermal_column,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Container
# ---------------------------------------------------------------------------


@dataclass
class SimulationResults:
    """Container for simulation output data.

    Attributes
    ----------
    times : list[datetime]
        UTC timestamps for each output snapshot.
    surface_temps : list[np.ndarray]
        Surface temperature maps [K] at each snapshot. Each: (num_faces,).
    illumination_maps : list[np.ndarray]
        Illumination fractions at each snapshot. Each: (num_faces,).
    sun_elevations : list[float]
        Sun elevation [deg] at each timestep.
    probe_temps : dict[str, list[float]]
        Temperature time series for named probe locations.
    face_centroids : np.ndarray
        Face centroid coordinates. Shape: (num_faces, 3).
    face_areas : np.ndarray
        Face areas [m²]. Shape: (num_faces,).
    metadata : dict
        Simulation metadata (config, timing, etc.).
    """

    times: list[datetime] = field(default_factory=list)
    surface_temps: list[np.ndarray] = field(default_factory=list)
    illumination_maps: list[np.ndarray] = field(default_factory=list)
    sun_elevations: list[float] = field(default_factory=list)
    probe_temps: dict[str, list[float]] = field(default_factory=dict)
    face_centroids: np.ndarray = field(default_factory=lambda: np.array([]))
    face_areas: np.ndarray = field(default_factory=lambda: np.array([]))
    dem_elevation: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Probe Location
# ---------------------------------------------------------------------------


@dataclass
class ProbeLocation:
    """A named temperature probe at a specific mesh face.

    Attributes
    ----------
    name : str
        Human-readable name (e.g., "crater_floor", "rim_east").
    face_index : int
        Index into the mesh face arrays.
    """

    name: str
    face_index: int


# ---------------------------------------------------------------------------
# Simulation Runner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """Main simulation runner orchestrating illumination + thermal solver.

    Parameters
    ----------
    config : SimulationConfig
        Full simulation configuration loaded from YAML.
    crater_radius_m : float, optional
        Override crater radius [m]. If None, uses config value.
    """

    def __init__(
        self,
        config: SimulationConfig,
        crater_radius_m: float | None = None,
    ) -> None:
        self._config = config

        # Override crater radius if specified
        if crater_radius_m is not None:
            # Mutate config (not frozen — SimulationConfig is mutable)
            from dataclasses import replace as dc_replace

            self._config = SimulationConfig(
                constants=config.constants,
                lunar=config.lunar,
                surface=config.surface,
                regolith=config.regolith,
                solver=config.solver,
                raytracer=config.raytracer,
                illumination=config.illumination,
                synthetic_dem=type(config.synthetic_dem)(
                    crater_type=config.synthetic_dem.crater_type,
                    radius_m=crater_radius_m,
                    depth_m=config.synthetic_dem.depth_m,
                    rim_height_m=config.synthetic_dem.rim_height_m,
                    grid_resolution_m=config.synthetic_dem.grid_resolution_m,
                    domain_padding_m=config.synthetic_dem.domain_padding_m,
                    seed=config.synthetic_dem.seed,
                ),
                assumptions=config.assumptions,
            )

        # Derived constants
        self._solar_constant = config.constants.solar_constant
        self._albedo = config.surface.bond_albedo
        self._dt = config.solver.dt_s

        logger.info(
            "SimulationRunner initialized: S₀=%.0f W/m², A=%.2f, dt=%.0fs",
            self._solar_constant, self._albedo, self._dt,
        )

    def run(
        self,
        start_time: datetime,
        duration_hours: float = 24.0,
        dt_s: float | None = None,
        output_interval_s: float = 3600.0,
        num_probes: int = 3,
        point_source_mode: bool | None = None,
        save_data: bool = True,
        output_dir: Path | str = "output",
        external_dem=None,
    ) -> SimulationResults:
        """Execute the simulation time loop.

        Parameters
        ----------
        start_time : datetime
            UTC start time.
        duration_hours : float
            Simulation duration in hours.
        dt_s : float, optional
            Override time step [s]. Default: from config.
        output_interval_s : float
            Save output every N seconds.
        num_probes : int
            Number of temperature probe locations to track.
        point_source_mode : bool, optional
            Override illumination mode.

        Returns
        -------
        SimulationResults
            All output data.
        """
        if dt_s is None:
            dt_s = self._dt

        total_seconds = duration_hours * 3600.0
        num_steps = int(total_seconds / dt_s)
        output_every = max(1, int(output_interval_s / dt_s))

        logger.info(
            "Starting simulation: %s → +%.1f hours (%d steps, dt=%.0fs)",
            start_time.isoformat(),
            duration_hours,
            num_steps,
            dt_s,
        )

        # Step 1: Generate or load DEM
        wall_start = time.perf_counter()
        if external_dem is not None:
            logger.info("Step 1/4: Using external DEM (%s)...",
                        external_dem.metadata.get("type", "unknown"))
            dem = external_dem
        else:
            logger.info("Step 1/4: Generating synthetic DEM...")
            dem = generate_synthetic_dem(self._config.synthetic_dem)
        mesh = dem_to_mesh(dem)
        num_faces = mesh.face_centroids.shape[0]

        # Step 2: Build illumination engine
        logger.info("Step 2/4: Building illumination engine...")
        use_point_source = (
            point_source_mode
            if point_source_mode is not None
            else self._config.illumination.point_source_mode
        )
        engine = IlluminationEngine(
            mesh=mesh,
            solar_angular_radius_rad=self._config.lunar.solar_angular_radius_rad,
            num_samples=self._config.illumination.solar_disk_samples,
            point_source_mode=use_point_source,
            epsilon=self._config.raytracer.epsilon,
            max_leaf_triangles=self._config.raytracer.max_leaf_triangles,
        )

        # Step 3: Initialize thermal solver + columns
        logger.info("Step 3/4: Initializing thermal solver (%d columns)...", num_faces)
        solver = CrankNicolsonSolver(self._config)
        template_column = create_thermal_column(self._config)

        # Create one thermal column per face (sharing the grid structure)
        columns: list[ThermalColumn] = []
        for _ in range(num_faces):
            col = ThermalColumn(
                z=template_column.z.copy(),
                T=template_column.T.copy(),
                dz=template_column.dz.copy(),
                dz_bar=template_column.dz_bar.copy(),
            )
            columns.append(col)

        # Set up probe locations
        probes = self._select_probes(mesh, num_probes)
        results = SimulationResults(
            face_centroids=mesh.face_centroids,
            face_areas=mesh.face_areas,
            dem_elevation=dem.elevation,
            metadata={
                "config_crater_radius_m": self._config.synthetic_dem.radius_m,
                "config_depth_m": self._config.synthetic_dem.depth_m,
                "num_faces": num_faces,
                "num_steps": num_steps,
                "dt_s": dt_s,
                "start_time": start_time.isoformat(),
                "duration_hours": duration_hours,
                "point_source_mode": use_point_source,
            },
        )
        for p in probes:
            results.probe_temps[p.name] = []

        # Step 4: Time loop
        logger.info("Step 4/4: Running time loop (%d steps)...", num_steps)
        current_time = start_time

        for step_i in range(num_steps):
            # 4a: Compute sun direction
            # Use a synthetic sun that rises from 0° and sweeps over 24h
            # This models a synodic period at the pole
            phase = (step_i * dt_s) / (self._config.lunar.synodic_period_s)
            sun_elevation_rad = np.radians(
                1.5 * np.sin(2 * np.pi * phase)  # max 1.5° elevation
            )
            sun_azimuth_rad = 2 * np.pi * phase

            # Convert spherical to Cartesian (in local frame: z=up, x=north)
            cos_el = np.cos(sun_elevation_rad)
            sun_dir = np.array([
                cos_el * np.cos(sun_azimuth_rad),
                cos_el * np.sin(sun_azimuth_rad),
                np.sin(sun_elevation_rad),
            ], dtype=np.float64)
            sun_dir /= np.linalg.norm(sun_dir)

            # 4b: Compute illumination
            illum_result = engine.compute(sun_dir)
            illumination = illum_result.illumination
            sun_elev_deg = illum_result.sun_elevation_deg

            # 4c: Compute absorbed solar flux per face
            # Q_solar = (1 − A) · S₀ · cos(θ) · f_illum
            cos_incidence = np.maximum(
                0.0,
                np.einsum("ij,j->i", mesh.face_normals, sun_dir),
            )
            Q_solar_per_face = (
                (1.0 - self._albedo)
                * self._solar_constant
                * cos_incidence
                * illumination
            )

            # 4d: Advance thermal state for each column
            for fi in range(num_faces):
                solver.step(columns[fi], Q_solar_per_face[fi], dt=dt_s)

            # 4e: Record probe temperatures every step
            for p in probes:
                results.probe_temps[p.name].append(
                    float(columns[p.face_index].T[0])
                )

            # 4f: Save output snapshots at intervals
            if step_i % output_every == 0:
                surface_T = np.array(
                    [columns[fi].T[0] for fi in range(num_faces)],
                    dtype=np.float64,
                )
                results.times.append(current_time)
                results.surface_temps.append(surface_T)
                results.illumination_maps.append(illumination.copy())
                results.sun_elevations.append(sun_elev_deg)

            # Progress logging
            if step_i % max(1, num_steps // 10) == 0:
                surface_T_snap = np.array(
                    [columns[fi].T[0] for fi in range(num_faces)]
                )
                logger.info(
                    "  Step %d/%d (t=%.1f hrs): "
                    "sun_elev=%.2f°, T_min=%.1f K, T_max=%.1f K, T_mean=%.1f K",
                    step_i, num_steps,
                    step_i * dt_s / 3600.0,
                    sun_elev_deg,
                    surface_T_snap.min(),
                    surface_T_snap.max(),
                    surface_T_snap.mean(),
                )

            current_time += timedelta(seconds=dt_s)

        # Always capture the final state
        if num_steps > 0:
            final_surface_T = np.array(
                [columns[fi].T[0] for fi in range(num_faces)],
                dtype=np.float64,
            )
            # Avoid duplicate if last step was already an output step
            if (num_steps - 1) % output_every != 0:
                results.times.append(current_time)
                results.surface_temps.append(final_surface_T)
                results.illumination_maps.append(illumination.copy())
                results.sun_elevations.append(sun_elev_deg)
            else:
                # Update the last snapshot with the final state
                results.surface_temps[-1] = final_surface_T

        wall_elapsed = time.perf_counter() - wall_start
        results.metadata["wall_time_s"] = wall_elapsed
        results.metadata["steps_per_second"] = num_steps / wall_elapsed

        logger.info(
            "Simulation complete: %.1f seconds wall time (%.1f steps/s)",
            wall_elapsed,
            num_steps / wall_elapsed,
        )

        # Save raw data for re-rendering
        if save_data and results.surface_temps:
            from simulation.io_manager import save_results

            save_results(
                output_dir=output_dir,
                surface_temps=results.surface_temps[-1],
                illumination=results.illumination_maps[-1],
                dem_elevation=results.dem_elevation,
                face_centroids=results.face_centroids,
                face_areas=results.face_areas,
                probe_temps=results.probe_temps,
                sun_elevations=results.sun_elevations,
                metadata=results.metadata,
            )

        return results

    def _select_probes(
        self,
        mesh: TriangleMesh,
        num_probes: int,
    ) -> list[ProbeLocation]:
        """Select representative probe locations on the mesh.

        Selects:
        - crater_floor: face closest to the center (lowest elevation)
        - crater_rim: face near the rim (highest elevation)
        - mid_slope: face at mid-depth between floor and rim

        Parameters
        ----------
        mesh : TriangleMesh
            Terrain mesh.
        num_probes : int
            Number of probes (up to 3).

        Returns
        -------
        list[ProbeLocation]
            Named probe locations.
        """
        centroids = mesh.face_centroids
        z = centroids[:, 2]
        r = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)

        probes: list[ProbeLocation] = []

        if num_probes >= 1:
            # Floor: lowest z, closest to center
            floor_idx = int(np.argmin(z))
            probes.append(ProbeLocation("crater_floor", floor_idx))
            logger.info(
                "Probe 'crater_floor': face %d at z=%.1f m, r=%.1f m",
                floor_idx, z[floor_idx], r[floor_idx],
            )

        if num_probes >= 2:
            # Rim: highest z
            rim_idx = int(np.argmax(z))
            probes.append(ProbeLocation("crater_rim", rim_idx))
            logger.info(
                "Probe 'crater_rim': face %d at z=%.1f m, r=%.1f m",
                rim_idx, z[rim_idx], r[rim_idx],
            )

        if num_probes >= 3:
            # Mid-slope: median elevation
            z_mid = (z.min() + z.max()) / 2.0
            mid_idx = int(np.argmin(np.abs(z - z_mid)))
            probes.append(ProbeLocation("mid_slope", mid_idx))
            logger.info(
                "Probe 'mid_slope': face %d at z=%.1f m, r=%.1f m",
                mid_idx, z[mid_idx], r[mid_idx],
            )

        return probes
