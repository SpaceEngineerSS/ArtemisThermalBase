"""Regression tests for cross-module correctness and performance fixes."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from core_engine.constants import load_config
from core_engine.illumination import IlluminationEngine
from core_engine.mesh import dem_to_mesh
from core_engine.view_factors import compute_view_factor_matrix
from data_ingestion.synthetic_dem import generate_synthetic_dem
from gui.data_loader import load_dashboard_data
from simulation.cli import main as installed_main
from simulation.io_manager import save_results
from simulation.runner import SimulationRunner
from thermal_solver.crank_nicolson import CrankNicolsonSolver, create_thermal_column


@pytest.fixture(scope="module")
def config():
    return load_config("config/default_config.yaml")


def test_advanced_yaml_sections_are_loaded(config) -> None:
    assert config.view_factors.enabled is False
    assert config.view_factors.num_rays == 128
    assert config.hapke.lookup_size == 2049
    assert config.roughness.cavity_coefficient == pytest.approx(0.4)
    assert config.target.latitude_deg == pytest.approx(-89.67)
    assert config.ephemeris.mode == "synthetic"


def test_installed_cli_entrypoint_exists() -> None:
    assert callable(installed_main)


def test_dashboard_reads_canonical_simulation_files(tmp_path) -> None:
    save_results(
        tmp_path,
        surface_temps=np.array([120.0, 130.0]),
        illumination=np.array([0.0, 1.0]),
        dem_elevation=np.array([[0.0, 1.0], [2.0, 3.0]]),
        face_centroids=np.zeros((2, 3)),
        face_areas=np.ones(2),
        probe_temps={"floor": [120.0]},
        sun_elevations=[0.1],
        metadata={"num_faces": 2},
    )
    dashboard_data = load_dashboard_data(tmp_path)
    assert dashboard_data is not None
    assert np.array_equal(dashboard_data["surface_temps"], [120.0, 130.0])
    assert np.array_equal(dashboard_data["illumination"], [0.0, 1.0])
    assert dashboard_data["metadata"]["num_faces"] == 2


def test_batch_solver_matches_scalar_solver(config) -> None:
    chunked_config = replace(
        config,
        solver=replace(config.solver, batch_size_faces=1),
    )
    solver = CrankNicolsonSolver(chunked_config)
    first = create_thermal_column(chunked_config)
    second = create_thermal_column(chunked_config)
    q_solar = np.array([0.0, 250.0], dtype=np.float64)
    q_ir = np.array([1.0, 2.0], dtype=np.float64)
    temperatures = np.vstack([first.T.copy(), second.T.copy()])

    solver.step(first, q_solar[0], Q_ir=q_ir[0], dt=60.0)
    solver.step(second, q_solar[1], Q_ir=q_ir[1], dt=60.0)
    solver.step_batch(
        temperatures,
        first.z,
        first.dz,
        first.dz_bar,
        q_solar,
        q_ir=q_ir,
        dt=60.0,
    )

    assert np.allclose(temperatures[0], first.T, atol=1e-10)
    assert np.allclose(temperatures[1], second.T, atol=1e-10)


def test_per_face_emissivity_changes_thermal_boundary(config) -> None:
    solver = CrankNicolsonSolver(config)
    column = create_thermal_column(config)
    temperatures = np.vstack([column.T.copy(), column.T.copy()])
    solver.step_batch(
        temperatures,
        column.z,
        column.dz,
        column.dz_bar,
        np.zeros(2),
        emissivities=np.array([0.80, 0.99]),
        dt=300.0,
    )
    assert temperatures[1, 0] < temperatures[0, 0]


def test_geothermal_flux_enters_column_once(config) -> None:
    density = 1500.0
    heat_capacity = 1000.0
    regolith = replace(
        config.regolith,
        density_surface=density,
        density_deep=density,
        cp_coefficients=(heat_capacity, 0.0, 0.0, 0.0, 0.0),
        cp_minimum=heat_capacity,
    )
    thermal_config = replace(
        config,
        regolith=regolith,
        surface=replace(config.surface, thermal_emissivity=1e-12),
    )
    solver = CrankNicolsonSolver(thermal_config)
    column = create_thermal_column(thermal_config)

    cell_widths = np.empty_like(column.T)
    cell_widths[0] = column.dz[0] / 2.0
    cell_widths[-1] = column.dz[-1] / 2.0
    cell_widths[1:-1] = (column.dz[:-1] + column.dz[1:]) / 2.0

    energy_before = float(
        np.sum(density * heat_capacity * column.T * cell_widths)
    )
    solver.step(column, Q_solar=0.0, Q_ir=0.0, dt=1.0)
    energy_after = float(
        np.sum(density * heat_capacity * column.T * cell_widths)
    )

    measured_flux = energy_after - energy_before
    assert measured_flux == pytest.approx(regolith.geothermal_flux, rel=1e-3)


def test_extended_sun_remains_partial_below_center_horizon(config) -> None:
    dem_cfg = replace(
        config.synthetic_dem,
        crater_type="flat",
        radius_m=10.0,
        domain_padding_m=0.0,
        grid_resolution_m=10.0,
    )
    mesh = dem_to_mesh(generate_synthetic_dem(dem_cfg))
    engine = IlluminationEngine(
        mesh,
        solar_angular_radius_rad=np.radians(0.533 / 2.0),
        num_samples=64,
        point_source_mode=False,
    )
    elevation = np.radians(-0.10)
    sun_dir = np.array([np.cos(elevation), 0.0, np.sin(elevation)])
    result = engine.compute(sun_dir)
    assert result.mode == "extended_source"
    assert np.any(result.illumination > 0.0)
    assert np.all(result.illumination < 1.0)
    assert np.all(result.projected_solar_factor > 0.0)
    # Centre-ray cosine is negative here; disk integration still captures
    # the physically visible upper solar limb.
    assert np.dot(mesh.face_normals[0], sun_dir) < 0.0


def test_research_config_is_fail_closed_without_real_dem() -> None:
    research_config = load_config("config/research_shackleton.yaml")
    assert research_config.research.enabled is True
    assert research_config.ephemeris.mode == "spice"
    with pytest.raises(ValueError, match="real external DEM"):
        SimulationRunner(research_config).run(
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            duration_hours=0.0,
            save_data=False,
        )


def test_view_factor_preflight_fails_before_large_allocation(config) -> None:
    dem_cfg = replace(
        config.synthetic_dem,
        crater_type="flat",
        radius_m=10.0,
        domain_padding_m=0.0,
        grid_resolution_m=10.0,
    )
    mesh = dem_to_mesh(generate_synthetic_dem(dem_cfg))
    engine = IlluminationEngine(mesh, point_source_mode=True)
    bvh_nodes, tri_verts, ordered = engine.get_bvh_data()
    with pytest.raises(MemoryError, match="preflight"):
        compute_view_factor_matrix(
            mesh.face_centroids,
            mesh.face_normals,
            mesh.face_areas,
            bvh_nodes,
            tri_verts,
            ordered,
            mesh.triangles,
            num_rays=512,
            max_neighbors=256,
            max_memory_mb=0.0001,
        )


def test_runner_uses_dense_state_pipeline(config) -> None:
    run_config = replace(
        config,
        synthetic_dem=replace(
            config.synthetic_dem,
            crater_type="flat",
            radius_m=10.0,
            domain_padding_m=0.0,
            grid_resolution_m=10.0,
        ),
        hapke=replace(config.hapke, enabled=False),
        roughness=replace(config.roughness, enabled=False),
    )
    result = SimulationRunner(run_config).run(
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        duration_hours=1.0 / 3600.0,
        dt_s=1.0,
        output_interval_s=1.0,
        point_source_mode=True,
        save_data=False,
    )
    assert result.metadata["num_steps"] == 1
    assert result.metadata["total_compute_steps"] == 1
    assert len(result.surface_temps) == 1
    assert result.times[0] == datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)


def test_runner_skyfield_mode_uses_target_and_distance_flux(
    config, monkeypatch
) -> None:
    calls: list[tuple[float, float, float]] = []

    class FakeEphemeris:
        def __init__(self, **_kwargs) -> None:
            pass

        def get_sun_state(self, _time, lat_deg, lon_deg, S_0):
            calls.append((lat_deg, lon_deg, S_0))
            return {
                "direction": np.array([0.0, 0.0, 1.0]),
                "solar_flux": 1400.0,
            }

    import data_ingestion.ephemeris as ephemeris_module

    monkeypatch.setattr(ephemeris_module, "SolarEphemeris", FakeEphemeris)
    run_config = replace(
        config,
        synthetic_dem=replace(
            config.synthetic_dem,
            crater_type="flat",
            radius_m=10.0,
            domain_padding_m=0.0,
            grid_resolution_m=10.0,
        ),
        hapke=replace(config.hapke, enabled=False),
        roughness=replace(config.roughness, enabled=False),
    )
    result = SimulationRunner(run_config).run(
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        duration_hours=1.0 / 3600.0,
        dt_s=1.0,
        output_interval_s=1.0,
        point_source_mode=True,
        ephemeris_mode="skyfield",
        latitude_deg=-88.0,
        longitude_deg=45.0,
        save_data=False,
    )

    assert calls == [(-88.0, 45.0, config.constants.solar_constant)]
    assert result.metadata["ephemeris_mode"] == "skyfield"
    assert np.all(result.illumination_maps[0] == 1.0)
