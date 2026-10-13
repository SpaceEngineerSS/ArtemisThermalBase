"""Simulation runner: terrain → shadows → heat conduction, step by step.

One run does the following:

1. Build the terrain mesh from a real or synthetic DEM, lowered onto the
   lunar sphere, and work out how the mesh axes sit in the Moon's body frame.
2. Estimate the memory the run needs and refuse to start above the limit.
3. Build the BVH, optional terrain view factors, roughness and Hapke tables.
4. March in time. Each step asks the Sun source for a body-frame direction,
   rotates it onto the mesh, traces shadows, and advances every regolith
   column in one parallel call.
5. Keep running statistics (min/max/mean surface temperature, mean
   illumination) instead of a full-mesh copy per output step, so memory does
   not grow with run length. Snapshots, if wanted, stream to disk.

Absorbed solar flux on a facet is

    Q_solar = (1 − A(μ)) · S(t) · ⟨max(0, n·s_k) · V_k⟩_k

where the average runs over solar-disk samples s_k with visibility V_k, and
A(μ) is the Hapke directional-hemispherical albedo (or the Bond albedo when
Hapke is off).

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from core_engine.constants import SimulationConfig
from core_engine.frames import MeshFrame, frame_for_dem
from core_engine.illumination import IlluminationEngine
from core_engine.mesh import TriangleMesh, dem_to_mesh
from data_ingestion.ephemeris import SunSource, SyntheticSun
from data_ingestion.synthetic_dem import DEMData, generate_synthetic_dem
from thermal_solver.crank_nicolson import CrankNicolsonSolver, create_thermal_column

logger = logging.getLogger(__name__)

EPHEMERIS_MODES = ("synthetic", "skyfield", "spice")


class RunCancelledError(RuntimeError):
    """Raised when a caller asks a running simulation to stop."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class SimulationResults:
    """Everything a run produces.

    Per-face arrays have shape ``(num_faces,)``. ``sunlit_fraction`` is the
    share of production steps in which at least half of the solar disk was
    visible from the face. Time series have one entry
    per output step (``times``, ``sun_elevations``) or per time step
    (``probe_temps``).
    """

    final_surface_T: np.ndarray
    final_illumination: np.ndarray
    min_surface_T: np.ndarray
    max_surface_T: np.ndarray
    mean_surface_T: np.ndarray
    mean_illumination: np.ndarray
    sunlit_fraction: np.ndarray
    face_centroids: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    dem_elevation: np.ndarray
    dem_x: np.ndarray
    dem_y: np.ndarray
    times: list[datetime] = field(default_factory=list)
    sun_elevations: list[float] = field(default_factory=list)
    sun_azimuths: list[float] = field(default_factory=list)
    probe_temps: dict[str, list[float]] = field(default_factory=dict)
    probe_faces: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Cell grid (rows, cols) of the DEM the mesh came from."""
        return (len(self.dem_y) - 1, len(self.dem_x) - 1)


@dataclass(frozen=True)
class ProbeLocation:
    """A named temperature probe at one mesh face."""

    name: str
    face_index: int


@dataclass(frozen=True)
class RunProgress:
    """Progress report handed to the optional callback once per step."""

    step: int
    total_steps: int
    phase: str
    sim_time: datetime
    sun_elevation_deg: float
    surface_min_K: float
    surface_max_K: float
    elapsed_s: float

    @property
    def fraction(self) -> float:
        """Completed fraction in [0, 1]."""
        return self.step / max(self.total_steps, 1)


ProgressCallback = Callable[[RunProgress], None]
StopCheck = Callable[[], bool]


# ---------------------------------------------------------------------------
# Memory estimate
# ---------------------------------------------------------------------------


def estimate_run_memory_mb(
    num_faces: int,
    config: SimulationConfig,
    view_factors: bool | None = None,
) -> dict[str, float]:
    """Approximate peak memory of a run, broken down by component [MB].

    The numbers are deliberately a little pessimistic. They cover the mesh,
    the BVH (including its build buffers), the dense thermal state, the
    per-step work arrays and, when enabled, the sparse view-factor matrix
    and its Monte-Carlo buffers.
    """
    f8 = 8.0
    nodes = config.solver.grid.num_layers + 1
    use_vf = config.view_factors.enabled if view_factors is None else view_factors
    parts = {
        "mesh": num_faces * (0.5 * 24 + 24 + 24 + 24 + 8),
        "bvh": num_faces * (72 + 2 * 64 + 8 + 3 * f8),
        "thermal_state": num_faces * nodes * f8,
        "work_arrays": num_faces * 14 * f8,
        "view_factors": 0.0,
    }
    if use_vf:
        width = min(config.view_factors.num_rays, config.view_factors.max_neighbors)
        parts["view_factors"] = num_faces * width * (8 + 8) * 2.0
    mb = {key: value / 1e6 for key, value in parts.items()}
    mb["total"] = sum(mb.values())
    return mb


def faces_for_dem_shape(ny: int, nx: int) -> int:
    """Number of triangles a ``ny × nx`` DEM turns into."""
    return 2 * max(ny - 1, 0) * max(nx - 1, 0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """Run illumination and heat conduction over a terrain patch.

    Parameters
    ----------
    config : SimulationConfig
        Full simulation configuration.
    crater_radius_m : float, optional
        Override for the synthetic crater radius.

    """

    def __init__(
        self,
        config: SimulationConfig,
        crater_radius_m: float | None = None,
    ) -> None:
        if crater_radius_m is not None:
            config = replace(
                config,
                synthetic_dem=replace(config.synthetic_dem, radius_m=crater_radius_m),
            )
        self._config = config

    @property
    def config(self) -> SimulationConfig:
        """The configuration this runner uses."""
        return self._config

    # -- validation -------------------------------------------------------

    def _check_research_mode(
        self,
        external_dem: DEMData | None,
        ephemeris_mode: str,
        spinup_cycles: int,
    ) -> None:
        research = self._config.research
        if not research.enabled:
            return
        if ephemeris_mode != "spice":
            raise ValueError(
                "Research mode requires the SPICE ephemeris; "
                f"'{ephemeris_mode}' was requested."
            )
        if research.require_real_dem and external_dem is None:
            raise ValueError("Research mode requires a real DEM; synthetic terrain is refused.")
        if (
            research.require_dem_provenance
            and external_dem is not None
            and not external_dem.metadata.get("provenance_verified", False)
        ):
            raise ValueError("Research mode requires a provenance-verified DEM.")
        if spinup_cycles < research.minimum_spinup_cycles:
            raise ValueError(
                f"Research mode requires at least {research.minimum_spinup_cycles} "
                "spin-up cycles."
            )

    def _sun_source(self, mode: str, start_time: datetime, frame: MeshFrame) -> SunSource:
        eph = self._config.ephemeris
        if mode == "skyfield":
            from data_ingestion.ephemeris import SolarEphemeris

            return SolarEphemeris(kernel_name=eph.kernel_name, data_dir=eph.data_dir)
        if mode == "spice":
            from data_ingestion.spice_ephemeris import SpiceSolarEphemeris

            return SpiceSolarEphemeris(
                eph.kernel_files,
                frame=eph.frame,
                aberration_correction=eph.aberration_correction,
            )
        return SyntheticSun(
            start_time=start_time,
            target_lon_deg=frame.longitude_deg,
            subsolar_lat_deg=eph.synthetic_subsolar_lat_deg,
            synodic_period_s=self._config.lunar.synodic_period_s,
        )

    # -- main loop --------------------------------------------------------

    def run(
        self,
        start_time: datetime,
        duration_hours: float = 24.0,
        dt_s: float | None = None,
        output_interval_s: float | None = None,
        num_probes: int = 3,
        point_source_mode: bool | None = None,
        ephemeris_mode: str | None = None,
        latitude_deg: float | None = None,
        longitude_deg: float | None = None,
        spinup_cycles: int | None = None,
        save_data: bool = True,
        output_dir: Path | str | None = None,
        external_dem: DEMData | None = None,
        progress: ProgressCallback | None = None,
        should_stop: StopCheck | None = None,
    ) -> SimulationResults:
        """Run the simulation and return its results.

        ``progress`` is called after every time step; ``should_stop`` is
        polled as well and, when it returns True, the run raises
        :class:`RunCancelledError`. Both are meant for the desktop app.
        """
        cfg = self._config
        dt_s = cfg.solver.dt_s if dt_s is None else float(dt_s)
        if dt_s <= 0:
            raise ValueError("dt must be positive.")
        if dt_s > cfg.solver.max_dt_s:
            raise ValueError(
                f"dt = {dt_s:g} s exceeds solver.thermal.time.max_dt_s = "
                f"{cfg.solver.max_dt_s:g} s."
            )
        if duration_hours <= 0:
            raise ValueError("duration must be positive.")
        output_interval_s = (
            cfg.time_range.output_interval_s if output_interval_s is None else output_interval_s
        )
        if output_interval_s <= 0:
            raise ValueError("output interval must be positive.")
        spinup_cycles = cfg.time_range.spinup_cycles if spinup_cycles is None else spinup_cycles
        if spinup_cycles < 0:
            raise ValueError("spinup_cycles cannot be negative.")
        solar_mode = ephemeris_mode or cfg.ephemeris.mode
        if solar_mode not in EPHEMERIS_MODES:
            raise ValueError(f"ephemeris mode must be one of {', '.join(EPHEMERIS_MODES)}.")
        self._check_research_mode(external_dem, solar_mode, spinup_cycles)

        production_steps = max(1, int(round(duration_hours * 3600.0 / dt_s)))
        steps_per_cycle = max(1, round(cfg.lunar.synodic_period_s / dt_s))
        spinup_steps = spinup_cycles * steps_per_cycle
        num_steps = spinup_steps + production_steps
        output_every = max(1, int(round(output_interval_s / dt_s)))
        wall_start = time.perf_counter()

        # 1. Terrain and frames ------------------------------------------
        dem = external_dem if external_dem is not None else generate_synthetic_dem(
            cfg.synthetic_dem
        )
        target_lat = cfg.target.latitude_deg if latitude_deg is None else latitude_deg
        target_lon = cfg.target.longitude_deg if longitude_deg is None else longitude_deg
        frame = frame_for_dem(dem.metadata, target_lat, target_lon, cfg.lunar.radius_m)
        num_faces = faces_for_dem_shape(*dem.elevation.shape)

        memory = estimate_run_memory_mb(num_faces, cfg)
        if memory["total"] > cfg.resources.max_memory_mb:
            raise MemoryError(
                f"This run needs about {memory['total']:.0f} MB "
                f"({num_faces:,} faces), above resources.max_memory_mb = "
                f"{cfg.resources.max_memory_mb:.0f} MB. Use a coarser grid "
                "(synthetic_dem.grid_resolution_m or a downsampled DEM), turn off "
                "view factors, or raise the limit if the machine has the room."
            )

        mesh = dem_to_mesh(
            dem, curvature_radius_m=cfg.lunar.radius_m if cfg.terrain.curvature else None
        )
        logger.info(
            "Terrain: %d faces, frame=%s at %.3f°, %.3f°; estimated memory %.0f MB",
            num_faces, frame.kind, frame.latitude_deg, frame.longitude_deg, memory["total"],
        )

        # 2. Shadows, view factors, optics -------------------------------
        use_point_source = (
            cfg.illumination.point_source_mode if point_source_mode is None else point_source_mode
        )
        sun_source = self._sun_source(solar_mode, start_time, frame)
        engine = IlluminationEngine(
            mesh=mesh,
            solar_angular_radius_rad=cfg.lunar.solar_angular_radius_rad,
            num_samples=cfg.illumination.solar_disk_samples,
            point_source_mode=use_point_source,
            epsilon=cfg.raytracer.epsilon,
            max_leaf_triangles=cfg.raytracer.max_leaf_triangles,
            sah_num_bins=cfg.raytracer.sah_num_bins,
        )

        view_factors = None
        if cfg.view_factors.enabled:
            from core_engine.view_factors import compute_view_factor_matrix

            vf = cfg.view_factors
            bvh_nodes, tri_verts, ordered = engine.get_bvh_data()
            view_factors = compute_view_factor_matrix(
                face_centroids=mesh.face_centroids,
                face_normals=mesh.face_normals,
                face_areas=mesh.face_areas,
                bvh_nodes=bvh_nodes,
                tri_verts=tri_verts,
                ordered_tri_indices=ordered,
                mesh_triangles=mesh.triangles,
                num_rays=vf.num_rays,
                max_neighbors=vf.max_neighbors,
                max_memory_mb=vf.max_memory_mb,
                epsilon=cfg.raytracer.epsilon,
                seed=vf.seed,
                reciprocity_tol=vf.reciprocity_tolerance,
            )

        emissivity = np.full(num_faces, cfg.surface.thermal_emissivity)
        if cfg.roughness.enabled:
            from core_engine.roughness import compute_roughness_correction

            try:
                emissivity = compute_roughness_correction(
                    face_normals=mesh.face_normals,
                    epsilon_0=cfg.surface.thermal_emissivity,
                    rms_slope_deg=cfg.roughness.rms_slope_deg,
                    K=cfg.roughness.cavity_coefficient,
                )
            except Exception:
                if cfg.research.enabled:
                    raise
                logger.warning("Roughness correction failed; using flat emissivity.",
                               exc_info=True)

        hapke_mu = hapke_albedo = None
        if cfg.hapke.enabled:
            from core_engine.reflectance import compute_adh_array

            hapke_mu = np.linspace(0.0, 1.0, cfg.hapke.lookup_size)
            hapke_albedo = compute_adh_array(
                hapke_mu,
                w=cfg.hapke.single_scattering_albedo,
                b=cfg.hapke.b,
                c=cfg.hapke.c,
                B_SH0=cfg.hapke.b_sh0,
                h_s=cfg.hapke.h_s,
            )

        # 3. Thermal state -----------------------------------------------
        solver = CrankNicolsonSolver(cfg)
        column = create_thermal_column(cfg)
        temperatures = np.broadcast_to(column.T, (num_faces, column.T.size)).copy()
        if cfg.solver.initial_condition == "equilibrium":
            t_eq = self._equilibrium_temperature(
                engine, sun_source, frame, start_time, emissivity,
            )
            temperatures[:] = t_eq[:, None]
        previous_cycle_surface = temperatures[:, 0].copy()
        spinup_deltas: list[float] = []

        probes = self._select_probes(mesh, num_probes)
        stats_min = np.full(num_faces, np.inf)
        stats_max = np.full(num_faces, -np.inf)
        stats_sum = np.zeros(num_faces)
        illum_sum = np.zeros(num_faces)
        sunlit_steps = np.zeros(num_faces, dtype=np.int32)
        stats_count = 0

        out_dir = Path(output_dir if output_dir is not None else cfg.output.directory)
        snapshots = _SnapshotWriter.create(
            out_dir if save_data else None,
            num_faces=num_faces,
            num_frames=-(-production_steps // output_every),
            budget_mb=cfg.output.max_snapshot_mb,
        )

        results_times: list[datetime] = []
        sun_elevations: list[float] = []
        sun_azimuths: list[float] = []
        probe_temps: dict[str, list[float]] = {p.name: [] for p in probes}
        q_ir = np.zeros(num_faces)
        illumination = np.zeros(num_faces)
        sun_elev_deg = 0.0

        # 4. Time loop ---------------------------------------------------
        current_time = start_time - timedelta(seconds=spinup_steps * dt_s)
        for step_i in range(num_steps):
            if should_stop is not None and should_stop():
                raise RunCancelledError(f"Stopped at step {step_i} of {num_steps}.")
            output_step = step_i - spinup_steps

            sun = sun_source.sun_state(current_time, cfg.constants.solar_constant)
            sun_dir = frame.from_body(sun.direction_body)
            illum_result = engine.compute(sun_dir)
            illumination = illum_result.illumination
            sun_elev_deg = illum_result.sun_elevation_deg

            absorptance: np.ndarray | float
            if hapke_mu is not None and hapke_albedo is not None:
                mu = np.maximum(mesh.face_normals @ sun_dir, 0.0)
                absorptance = 1.0 - np.interp(mu, hapke_mu, hapke_albedo)
            else:
                absorptance = 1.0 - cfg.surface.bond_albedo
            q_solar = absorptance * sun.solar_flux * illum_result.projected_solar_factor

            if view_factors is not None:
                from core_engine.view_factors import compute_ir_flux_gray

                q_ir = compute_ir_flux_gray(
                    temperatures[:, 0],
                    view_factors.row_ptr,
                    view_factors.col_idx,
                    view_factors.values,
                    emissivity,
                    cfg.constants.stefan_boltzmann,
                )

            solver.step_batch(
                temperatures, column.z, column.dz, column.dz_bar,
                q_solar, q_ir=q_ir, emissivities=emissivity, dt=dt_s,
            )
            current_time = current_time + timedelta(seconds=dt_s)
            surface = temperatures[:, 0]

            if step_i < spinup_steps:
                if (step_i + 1) % steps_per_cycle == 0:
                    delta = float(np.max(np.abs(surface - previous_cycle_surface)))
                    spinup_deltas.append(delta)
                    previous_cycle_surface = surface.copy()
                    logger.info("Spin-up cycle %d/%d: max surface change %.4f K",
                                len(spinup_deltas), spinup_cycles, delta)
            else:
                np.minimum(stats_min, surface, out=stats_min)
                np.maximum(stats_max, surface, out=stats_max)
                stats_sum += surface
                illum_sum += illumination
                sunlit_steps += illumination >= 0.5
                stats_count += 1
                for probe in probes:
                    probe_temps[probe.name].append(float(surface[probe.face_index]))
                is_last = step_i == num_steps - 1
                if (output_step + 1) % output_every == 0 or is_last:
                    results_times.append(current_time)
                    sun_elevations.append(sun_elev_deg)
                    sun_azimuths.append(float(np.degrees(np.arctan2(sun_dir[0], sun_dir[1]))))
                    snapshots.write(surface, illumination)

            if progress is not None:
                progress(RunProgress(
                    step=step_i + 1,
                    total_steps=num_steps,
                    phase="spin-up" if step_i < spinup_steps else "production",
                    sim_time=current_time,
                    sun_elevation_deg=sun_elev_deg,
                    surface_min_K=float(surface.min()),
                    surface_max_K=float(surface.max()),
                    elapsed_s=time.perf_counter() - wall_start,
                ))
            if step_i % max(1, num_steps // 10) == 0:
                logger.info(
                    "Step %d/%d: sun %.2f°, T %.1f–%.1f K",
                    step_i + 1, num_steps, sun_elev_deg, surface.min(), surface.max(),
                )

        snapshots.close()
        wall = time.perf_counter() - wall_start

        metadata: dict[str, Any] = {
            "num_faces": num_faces,
            "grid_shape": [len(dem.y_coords) - 1, len(dem.x_coords) - 1],
            "resolution_m": dem.resolution_m,
            "num_steps": production_steps,
            "total_compute_steps": num_steps,
            "spinup_cycles": spinup_cycles,
            "spinup_steps": spinup_steps,
            "spinup_cycle_max_delta_K": spinup_deltas,
            "dt_s": dt_s,
            "output_interval_s": output_every * dt_s,
            "start_time": start_time.isoformat(),
            "duration_hours": duration_hours,
            "point_source_mode": use_point_source,
            "ephemeris_mode": solar_mode,
            "frame": frame.describe(),
            "curvature": cfg.terrain.curvature,
            "research_mode": cfg.research.enabled,
            "dem_type": dem.metadata.get("type"),
            "dem_provenance": dem.metadata.get("provenance"),
            "crater_radius_m": cfg.synthetic_dem.radius_m if external_dem is None else None,
            "hapke": cfg.hapke.enabled,
            "roughness": cfg.roughness.enabled,
            "view_factors": cfg.view_factors.enabled,
            "volatile_thresholds_K": [
                cfg.volatiles.stable_threshold_k,
                cfg.volatiles.marginal_threshold_k,
            ],
            "estimated_memory_mb": memory,
            "snapshot_frames": snapshots.frames_written,
            "snapshot_stride": snapshots.stride,
            "wall_time_s": wall,
            "steps_per_second": num_steps / wall if wall > 0 else 0.0,
        }
        if spinup_deltas:
            converged = spinup_deltas[-1] <= cfg.time_range.spinup_convergence_tolerance_k
            metadata["spinup_converged"] = converged
            if not converged:
                logger.warning("Spin-up did not converge: last cycle changed by %.3f K.",
                               spinup_deltas[-1])
        if hasattr(sun_source, "provenance"):
            metadata["ephemeris_provenance"] = sun_source.provenance

        count = max(stats_count, 1)
        results = SimulationResults(
            final_surface_T=temperatures[:, 0].copy(),
            final_illumination=illumination.copy(),
            min_surface_T=stats_min,
            max_surface_T=stats_max,
            mean_surface_T=stats_sum / count,
            mean_illumination=illum_sum / count,
            sunlit_fraction=sunlit_steps / count,
            face_centroids=mesh.face_centroids,
            face_normals=mesh.face_normals,
            face_areas=mesh.face_areas,
            dem_elevation=dem.elevation,
            dem_x=dem.x_coords,
            dem_y=dem.y_coords,
            times=results_times,
            sun_elevations=sun_elevations,
            sun_azimuths=sun_azimuths,
            probe_temps=probe_temps,
            probe_faces={p.name: p.face_index for p in probes},
            metadata=metadata,
        )
        logger.info("Run finished in %.1f s (%.1f steps/s).", wall, metadata["steps_per_second"])

        if save_data:
            from simulation.io_manager import save_results

            save_results(out_dir, results)
        return results

    def _equilibrium_temperature(
        self,
        engine: IlluminationEngine,
        sun_source: SunSource,
        frame: MeshFrame,
        start_time: datetime,
        emissivity: np.ndarray,
        samples: int = 24,
    ) -> np.ndarray:
        """Per-face temperature that radiates away the day-mean absorbed flux.

        The Sun is sampled at ``samples`` instants spread over one synodic
        month with point-source shadows. Starting from here instead of a
        uniform guess removes most of the initial transient, so short runs
        without spin-up do not report the start temperature as their maximum.
        Columns are set isothermal; spin-up still matters for research runs.
        """
        cfg = self._config
        period = cfg.lunar.synodic_period_s
        mean_flux = np.zeros_like(emissivity)
        for k in range(samples):
            when = start_time + timedelta(seconds=period * k / samples)
            sun = sun_source.sun_state(when, cfg.constants.solar_constant)
            lit = engine.compute(frame.from_body(sun.direction_body), point_source_override=True)
            mean_flux += sun.solar_flux * lit.projected_solar_factor
        mean_flux *= (1.0 - cfg.surface.bond_albedo) / samples
        q = mean_flux + cfg.regolith.geothermal_flux
        return (q / (emissivity * cfg.constants.stefan_boltzmann)) ** 0.25

    @staticmethod
    def _select_probes(mesh: TriangleMesh, num_probes: int) -> list[ProbeLocation]:
        """Pick the lowest face, the highest face and one at mid-height."""
        z = mesh.face_centroids[:, 2]
        probes = [
            ProbeLocation("floor", int(np.argmin(z))),
            ProbeLocation("rim", int(np.argmax(z))),
            ProbeLocation("mid_slope", int(np.argmin(np.abs(z - 0.5 * (z.min() + z.max()))))),
        ]
        return probes[: max(0, min(num_probes, 3))]


class _SnapshotWriter:
    """Streams per-output surface temperature and illumination to disk.

    Frames are stored as float32 in memory-mapped ``.npy`` files. If the
    run would exceed the size budget, every k-th frame is kept instead.
    """

    def __init__(self, temps: Any, illum: Any, stride: int) -> None:
        self._temps = temps
        self._illum = illum
        self.stride = stride
        self._seen = 0
        self.frames_written = 0

    @classmethod
    def create(
        cls,
        output_dir: Path | None,
        num_faces: int,
        num_frames: int,
        budget_mb: float,
    ) -> _SnapshotWriter:
        if output_dir is None or budget_mb <= 0 or num_faces == 0:
            return cls(None, None, 1)
        frame_mb = 2 * num_faces * 4 / 1e6
        stride = max(1, int(np.ceil(num_frames * frame_mb / budget_mb)))
        kept = (num_frames + stride - 1) // stride
        output_dir.mkdir(parents=True, exist_ok=True)
        open_memmap = np.lib.format.open_memmap
        temps = open_memmap(output_dir / "snapshots_surface_T.npy", mode="w+",
                            dtype=np.float32, shape=(kept, num_faces))
        illum = open_memmap(output_dir / "snapshots_illumination.npy", mode="w+",
                            dtype=np.float32, shape=(kept, num_faces))
        if stride > 1:
            logger.info("Snapshot budget: keeping every %d-th output frame.", stride)
        return cls(temps, illum, stride)

    def write(self, surface: np.ndarray, illumination: np.ndarray) -> None:
        if self._temps is None:
            return
        if self._seen % self.stride == 0 and self.frames_written < self._temps.shape[0]:
            self._temps[self.frames_written] = surface
            self._illum[self.frames_written] = illumination
            self.frames_written += 1
        self._seen += 1

    def close(self) -> None:
        for array in (self._temps, self._illum):
            if array is not None:
                array.flush()
        self._temps = self._illum = None
