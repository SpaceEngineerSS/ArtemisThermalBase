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
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from core_engine.constants import SimulationConfig
from core_engine.illumination import IlluminationEngine
from core_engine.mesh import TriangleMesh, dem_to_mesh
from data_ingestion.synthetic_dem import generate_synthetic_dem
from thermal_solver.crank_nicolson import CrankNicolsonSolver, create_thermal_column

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
            self._config = replace(
                config,
                synthetic_dem=replace(config.synthetic_dem, radius_m=crater_radius_m),
            )

        # Derived constants
        self._solar_constant = self._config.constants.solar_constant
        self._albedo = self._config.surface.bond_albedo
        self._emissivity = self._config.surface.thermal_emissivity
        self._sigma = self._config.constants.stefan_boltzmann
        self._dt = self._config.solver.dt_s

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
        ephemeris_mode: str | None = None,
        latitude_deg: float | None = None,
        longitude_deg: float | None = None,
        spinup_cycles: int | None = None,
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

        if spinup_cycles is None:
            spinup_cycles = self._config.time_range.spinup_cycles
        if spinup_cycles < 0:
            raise ValueError("spinup_cycles cannot be negative.")
        if self._config.research.enabled:
            research = self._config.research
            if research.require_real_dem and external_dem is None:
                raise ValueError(
                    "Research mode requires a real external DEM; "
                    "synthetic terrain is forbidden."
                )
            if (
                research.require_dem_provenance
                and external_dem is not None
                and not external_dem.metadata.get("provenance_verified", False)
            ):
                raise ValueError("Research mode requires a provenance-verified DEM.")
            if spinup_cycles < research.minimum_spinup_cycles:
                raise ValueError(
                    "Research mode requires at least "
                    f"{research.minimum_spinup_cycles} spin-up cycles."
                )
        production_steps = int(duration_hours * 3600.0 / dt_s)
        steps_per_cycle = max(1, round(self._config.lunar.synodic_period_s / dt_s))
        spinup_steps = spinup_cycles * steps_per_cycle
        num_steps = spinup_steps + production_steps
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
        thermal_peak_mb = (
            (
                num_faces
                + min(num_faces, self._config.solver.batch_size_faces)
            )
            * (self._config.solver.grid.num_layers + 1)
            * np.dtype(np.float64).itemsize
            / 1e6
        )
        if thermal_peak_mb > self._config.solver.max_state_memory_mb:
            raise MemoryError(
                "Thermal-state preflight requires approximately "
                f"{thermal_peak_mb:.0f} MB, above the configured "
                f"{self._config.solver.max_state_memory_mb:.0f} MB limit. "
                "Reduce DEM resolution/layers or explicitly raise "
                "solver.thermal.max_state_memory_mb."
            )

        # Step 2: Build illumination engine
        logger.info("Step 2/4: Building illumination engine...")
        use_point_source = (
            point_source_mode
            if point_source_mode is not None
            else self._config.illumination.point_source_mode
        )
        solar_mode = ephemeris_mode or self._config.ephemeris.mode
        if solar_mode not in {"synthetic", "skyfield", "spice"}:
            raise ValueError("ephemeris_mode must be 'synthetic', 'skyfield', or 'spice'.")
        target_lat = (
            self._config.target.latitude_deg if latitude_deg is None else latitude_deg
        )
        target_lon = (
            self._config.target.longitude_deg if longitude_deg is None else longitude_deg
        )
        ephemeris: Any = None
        if solar_mode == "skyfield":
            from data_ingestion.ephemeris import SolarEphemeris

            ephemeris = SolarEphemeris(
                kernel_name=self._config.ephemeris.kernel_name,
                data_dir=self._config.ephemeris.data_dir,
            )
        elif solar_mode == "spice":
            from data_ingestion.spice_ephemeris import SpiceSolarEphemeris

            ephemeris = SpiceSolarEphemeris(
                self._config.ephemeris.kernel_files,
                frame=self._config.ephemeris.frame,
                aberration_correction=self._config.ephemeris.aberration_correction,
            )
        engine = IlluminationEngine(
            mesh=mesh,
            solar_angular_radius_rad=self._config.lunar.solar_angular_radius_rad,
            num_samples=self._config.illumination.solar_disk_samples,
            point_source_mode=use_point_source,
            epsilon=self._config.raytracer.epsilon,
            max_leaf_triangles=self._config.raytracer.max_leaf_triangles,
        )

        # Step 2.5: Compute view factors for IR scattering
        view_factors = None
        if self._config.view_factors.enabled:
            vf_cfg = self._config.view_factors
            logger.info(
                "Step 2.5/5: Computing view factors "
                "(Monte Carlo, %d rays/face)...", vf_cfg.num_rays,
            )
            from core_engine.view_factors import compute_view_factor_matrix

            bvh_nodes, tri_verts, ordered_indices = engine.get_bvh_data()
            view_factors = compute_view_factor_matrix(
                    face_centroids=mesh.face_centroids,
                    face_normals=mesh.face_normals,
                    face_areas=mesh.face_areas,
                    bvh_nodes=bvh_nodes,
                    tri_verts=tri_verts,
                    ordered_tri_indices=ordered_indices,
                    mesh_triangles=mesh.triangles,
                    num_rays=vf_cfg.num_rays,
                    max_neighbors=vf_cfg.max_neighbors,
                    max_memory_mb=vf_cfg.max_memory_mb,
                    epsilon=self._config.raytracer.epsilon,
                    seed=vf_cfg.seed,
                    reciprocity_tol=vf_cfg.reciprocity_tolerance,
                )
            logger.info(
                    "View factors ready: %d nnz, %.1f MB",
                    view_factors.nnz, view_factors.memory_mb,
                )
        else:
            logger.info("View factors disabled by config.")

        # Step 2.7: Precompute roughness correction (geometry-only)
        roughness_emissivity = np.full(num_faces, self._emissivity)
        hapke_enabled = self._config.hapke.enabled
        roughness_enabled = self._config.roughness.enabled
        rms_slope_deg = self._config.roughness.rms_slope_deg

        if roughness_enabled:
            try:
                from core_engine.roughness import compute_roughness_correction
                roughness_emissivity = compute_roughness_correction(
                    face_normals=mesh.face_normals,
                    epsilon_0=self._emissivity,
                    rms_slope_deg=rms_slope_deg,
                    K=self._config.roughness.cavity_coefficient,
                )
                logger.info(
                    "Roughness correction applied: ε_eff range [%.4f, %.4f] "
                    "(ε₀=%.2f, θ̄=%.1f°)",
                    roughness_emissivity.min(), roughness_emissivity.max(),
                    self._emissivity, rms_slope_deg,
                )
            except Exception as e:
                logger.warning(
                    "Roughness correction failed, using flat emissivity: %s", e
                )

        # Hapke parameters (loaded from config or defaults)
        hapke_w = self._config.hapke.single_scattering_albedo
        hapke_b = self._config.hapke.b
        hapke_c = self._config.hapke.c
        hapke_B_SH0 = self._config.hapke.b_sh0
        hapke_h_s = self._config.hapke.h_s
        hapke_mu_grid = None
        hapke_albedo_grid = None
        if hapke_enabled:
            from core_engine.reflectance import compute_adh_array

            hapke_mu_grid = np.linspace(0.0, 1.0, self._config.hapke.lookup_size)
            hapke_albedo_grid = compute_adh_array(
                hapke_mu_grid,
                w=hapke_w,
                b=hapke_b,
                c=hapke_c,
                B_SH0=hapke_B_SH0,
                h_s=hapke_h_s,
            )

        # Step 3: Initialize thermal solver + columns
        logger.info("Step 3/5: Initializing thermal solver (%d columns)...", num_faces)
        solver = CrankNicolsonSolver(self._config)
        template_column = create_thermal_column(self._config)

        # Dense state removes millions of Python objects and shares the
        # immutable vertical grid across all independent 1-D columns.
        temperatures = np.broadcast_to(
            template_column.T,
            (num_faces, template_column.T.size),
        ).copy()
        previous_cycle_surface = temperatures[:, 0].copy()
        spinup_cycle_deltas_k: list[float] = []

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
                "num_steps": production_steps,
                "total_compute_steps": num_steps,
                "spinup_cycles": spinup_cycles,
                "spinup_steps": spinup_steps,
                "dt_s": dt_s,
                "start_time": start_time.isoformat(),
                "duration_hours": duration_hours,
                "point_source_mode": use_point_source,
                "ephemeris_mode": solar_mode,
                "target_latitude_deg": target_lat,
                "target_longitude_deg": target_lon,
                "research_mode": self._config.research.enabled,
                "dem_provenance": dem.metadata.get("provenance"),
            },
        )
        if solar_mode == "spice" and ephemeris is not None:
            results.metadata["ephemeris_provenance"] = ephemeris.provenance
        for p in probes:
            results.probe_temps[p.name] = []

        # Step 4: Time loop
        logger.info("Step 4/5: Running time loop (%d steps)...", num_steps)
        current_time = start_time - timedelta(seconds=spinup_steps * dt_s)

        for step_i in range(num_steps):
            output_step = step_i - spinup_steps
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

            # 4b: Distance-corrected solar flux (inverse-square law)
            # S(t) = S₀ × (1 AU / d_sun)² — varies ±3.4% over year
            # For synthetic sun, use fixed S₀ (real ephemeris would use
            # get_solar_flux() with actual UTC time)
            solar_flux_t = self._solar_constant
            if ephemeris is not None:
                sun_state = ephemeris.get_sun_state(
                    current_time,
                    lat_deg=target_lat,
                    lon_deg=target_lon,
                    S_0=self._solar_constant,
                )
                sun_dir = np.asarray(sun_state["direction"], dtype=np.float64)
                solar_flux_t = float(sun_state["solar_flux"])

            # 4c: Compute illumination
            illum_result = engine.compute(sun_dir)
            illumination = illum_result.illumination
            projected_solar_factor = illum_result.projected_solar_factor
            sun_elev_deg = illum_result.sun_elevation_deg

            # 4d: Compute absorbed solar flux per face
            # Q_solar = [1 − A_DH(θ_i)] · S(t) · cos(θ_i) · f_illum
            cos_incidence = np.maximum(
                0.0,
                np.einsum("ij,j->i", mesh.face_normals, sun_dir),
            )

            # Hapke: angle-dependent directional-hemispherical albedo
            if hapke_enabled:
                assert hapke_mu_grid is not None
                assert hapke_albedo_grid is not None
                A_DH = np.interp(cos_incidence, hapke_mu_grid, hapke_albedo_grid)
                Q_solar_per_face = (
                    (1.0 - A_DH)
                    * solar_flux_t
                    * projected_solar_factor
                )
            else:
                Q_solar_per_face = (
                    (1.0 - self._albedo)
                    * solar_flux_t
                    * projected_solar_factor
                )

            # 4e: Compute IR flux from terrain (view factor radiosity)
            Q_ir_per_face = np.zeros(num_faces, dtype=np.float64)
            if view_factors is not None:
                surface_T_current = temperatures[:, 0]
                from core_engine.view_factors import compute_ir_flux_gray
                Q_ir_per_face = compute_ir_flux_gray(
                    surface_T_current,
                    view_factors.row_ptr,
                    view_factors.col_idx,
                    view_factors.values,
                    roughness_emissivity,
                    self._sigma,
                )

            # 4f: Advance all independent columns in one parallel call.
            solver.step_batch(
                temperatures,
                template_column.z,
                template_column.dz,
                template_column.dz_bar,
                Q_solar_per_face,
                q_ir=Q_ir_per_face,
                emissivities=roughness_emissivity,
                dt=dt_s,
            )
            state_time = current_time + timedelta(seconds=dt_s)

            if step_i < spinup_steps and (step_i + 1) % steps_per_cycle == 0:
                cycle_surface = temperatures[:, 0]
                cycle_delta = float(
                    np.max(np.abs(cycle_surface - previous_cycle_surface))
                )
                spinup_cycle_deltas_k.append(cycle_delta)
                previous_cycle_surface = cycle_surface.copy()
                logger.info(
                    "Spin-up cycle %d/%d: max surface delta=%.4f K",
                    len(spinup_cycle_deltas_k),
                    spinup_cycles,
                    cycle_delta,
                )

            # 4e: Record probe temperatures every step
            if output_step >= 0:
                for p in probes:
                    results.probe_temps[p.name].append(
                        float(temperatures[p.face_index, 0])
                    )

            # 4f: Save output snapshots at intervals
            if output_step >= 0 and output_step % output_every == 0:
                surface_T = temperatures[:, 0].copy()
                results.times.append(state_time)
                results.surface_temps.append(surface_T)
                results.illumination_maps.append(illumination.copy())
                results.sun_elevations.append(sun_elev_deg)

            # Progress logging
            if step_i % max(1, num_steps // 10) == 0:
                surface_T_snap = temperatures[:, 0]
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

            current_time = state_time

        # Always capture the final state
        if production_steps > 0:
            final_surface_T = temperatures[:, 0].copy()
            # Avoid duplicate if last step was already an output step
            if (production_steps - 1) % output_every != 0:
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
        results.metadata["spinup_cycle_max_delta_K"] = spinup_cycle_deltas_k
        if spinup_cycle_deltas_k:
            tolerance = self._config.time_range.spinup_convergence_tolerance_k
            converged = spinup_cycle_deltas_k[-1] <= tolerance
            results.metadata["spinup_converged"] = converged
            if not converged:
                logger.warning(
                    "Spin-up did not meet %.3f K tolerance; final cycle delta=%.3f K",
                    tolerance,
                    spinup_cycle_deltas_k[-1],
                )

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
