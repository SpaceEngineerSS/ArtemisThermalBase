"""Physical constants, lunar parameters, and configuration loader.

All numerical values are loaded from YAML configuration files.
NO physical constants are hardcoded in this module or anywhere else.
This module provides a typed, validated interface to the configuration.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

References
----------
- CODATA 2018 for fundamental constants
- Hayne et al. (2017) for regolith thermophysical properties
- Vasavada et al. (2012), Bandfield et al. (2015) for surface optical properties
- Langseth et al. (1976) for geothermal heat flux
"""

from __future__ import annotations

import hashlib
import logging
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration Data Classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FundamentalConstants:
    """Fundamental physical constants (CODATA 2018 / IAU 2012).

    Attributes
    ----------
    stefan_boltzmann : float
        Stefan-Boltzmann constant [W/m²/K⁴].
    solar_constant : float
        Total Solar Irradiance at 1 AU [W/m²].
    astronomical_unit : float
        1 Astronomical Unit [m].
    speed_of_light : float
        Speed of light in vacuum [m/s].
    """

    stefan_boltzmann: float
    solar_constant: float
    astronomical_unit: float
    speed_of_light: float


@dataclass(frozen=True)
class LunarParameters:
    """Physical parameters of the Moon.

    Attributes
    ----------
    radius_m : float
        Mean lunar radius [m].
    sidereal_period_s : float
        Sidereal rotation period [s].
    synodic_period_s : float
        Synodic period / one lunar day [s].
    obliquity_deg : float
        Obliquity to ecliptic plane [deg].
    solar_angular_diameter_deg : float
        Mean angular diameter of the Sun as seen from the Moon [deg].
    """

    radius_m: float
    sidereal_period_s: float
    synodic_period_s: float
    obliquity_deg: float
    solar_angular_diameter_deg: float

    @property
    def solar_angular_radius_rad(self) -> float:
        """Half angular diameter of the Sun in radians."""
        return np.radians(self.solar_angular_diameter_deg / 2.0)


@dataclass(frozen=True)
class SurfaceProperties:
    """Optical surface properties.

    Attributes
    ----------
    bond_albedo : float
        Bond albedo [-].
    thermal_emissivity : float
        Broadband thermal emissivity [-].
    reflectance_model : str
        Reflectance model name (e.g., 'lambertian').
    """

    bond_albedo: float
    thermal_emissivity: float
    reflectance_model: str


@dataclass(frozen=True)
class ConductivityLayer:
    """Thermal conductivity parameters for one regolith layer.

    k(T) = k_contact + k_radiative * T^3

    Attributes
    ----------
    k_contact : float
        Contact (phonon) conductivity [W/m/K].
    k_radiative : float
        Radiative conductivity coefficient [W/m/K⁴].
    """

    k_contact: float
    k_radiative: float


@dataclass(frozen=True)
class RegolithProperties:
    """Thermophysical properties of lunar regolith.

    All values from Hayne et al. (2017), JGR Planets.

    Attributes
    ----------
    density_surface : float
        Surface bulk density [kg/m³].
    density_deep : float
        Deep bulk density [kg/m³].
    density_transition_m : float
        e-folding depth for density transition [m].
    conductivity_surface : ConductivityLayer
        Surface layer conductivity model.
    conductivity_deep : ConductivityLayer
        Deep layer conductivity model.
    conductivity_layer_boundary_m : float
        Depth at which conductivity model transitions [m].
    cp_coefficients : tuple[float, float, float, float, float]
        Polynomial coefficients for c_p(T) = c0 + c1*T^0.5 + c2*T + c3*T^2 + c4*T^3.
    cp_minimum : float
        Clamping floor for c_p to avoid singularity at T→0 [J/kg/K].
    geothermal_flux : float
        Interior heat flux [W/m²].
    """

    density_surface: float
    density_deep: float
    density_transition_m: float
    conductivity_surface: ConductivityLayer
    conductivity_deep: ConductivityLayer
    conductivity_layer_boundary_m: float
    cp_coefficients: tuple[float, float, float, float, float]
    cp_minimum: float
    geothermal_flux: float


@dataclass(frozen=True)
class ThermalGridConfig:
    """Vertical grid configuration for subsurface solver.

    Attributes
    ----------
    dz_surface_m : float
        Finest grid spacing at z=0 [m].
    growth_ratio : float
        Geometric growth factor for grid spacing.
    num_layers : int
        Number of subsurface grid layers.
    """

    dz_surface_m: float
    growth_ratio: float
    num_layers: int

    def build_grid(self) -> np.ndarray:
        """Generate the non-uniform depth grid.

        Returns
        -------
        z : np.ndarray
            Depth values [m] of each grid point, starting at z=0.
            Shape: (num_layers + 1,).
        """
        dz = np.array(
            [self.dz_surface_m * (self.growth_ratio ** i) for i in range(self.num_layers)],
            dtype=np.float64,
        )
        z = np.zeros(self.num_layers + 1, dtype=np.float64)
        z[1:] = np.cumsum(dz)
        return z


@dataclass(frozen=True)
class NewtonConfig:
    """Newton iteration settings for nonlinear surface boundary condition.

    Attributes
    ----------
    max_iterations : int
        Maximum Newton iterations per time step.
    tolerance_K : float
        Convergence tolerance [K].
    relaxation : float
        Under-relaxation factor (1.0 = no relaxation).
    """

    max_iterations: int
    tolerance_K: float
    relaxation: float


@dataclass(frozen=True)
class SolverConfig:
    """Complete thermal solver configuration.

    Attributes
    ----------
    method : str
        Solver method name (e.g., 'crank_nicolson').
    grid : ThermalGridConfig
        Vertical grid configuration.
    dt_s : float
        Default time step [s].
    max_dt_s : float
        Maximum allowed time step [s].
    newton : NewtonConfig
        Newton iteration settings.
    initial_temperature_K : float
        Uniform initial temperature guess [K].
    """

    method: str
    grid: ThermalGridConfig
    dt_s: float
    max_dt_s: float
    newton: NewtonConfig
    initial_temperature_K: float


@dataclass(frozen=True)
class RaytracerConfig:
    """BVH raytracer configuration.

    Attributes
    ----------
    max_leaf_triangles : int
        Maximum triangles per BVH leaf node.
    sah_num_bins : int
        Number of bins for SAH cost sweep.
    epsilon : float
        Zero-test epsilon for Möller-Trumbore algorithm.
    precision : str
        Floating point precision ('float32' or 'float64').
    """

    max_leaf_triangles: int
    sah_num_bins: int
    epsilon: float
    precision: str


@dataclass(frozen=True)
class IlluminationConfig:
    """Illumination computation configuration.

    Attributes
    ----------
    solar_disk_samples : int
        Number of sample points on the solar disk.
    sampling_method : str
        Sampling pattern ('fibonacci', 'concentric', 'random').
    point_source_mode : bool
        If True, treat the Sun as a point source (fast preview).
    """

    solar_disk_samples: int
    sampling_method: str
    point_source_mode: bool


@dataclass(frozen=True)
class SyntheticDEMConfig:
    """Configuration for synthetic DEM generation.

    Attributes
    ----------
    crater_type : str
        Crater shape type ('parabolic_bowl', 'conical', 'flat').
    radius_m : float
        Crater radius [m].
    depth_m : float
        Crater depth from rim to floor [m].
    rim_height_m : float
        Rim elevation above surrounding terrain [m].
    grid_resolution_m : float
        Grid spacing [m/pixel].
    domain_padding_m : float
        Flat terrain padding around crater rim [m].
    seed : int
        Random seed for reproducibility.
    """

    crater_type: str
    radius_m: float
    depth_m: float
    rim_height_m: float
    grid_resolution_m: float
    domain_padding_m: float
    seed: int


@dataclass(frozen=True)
class Assumption:
    """A documented model assumption.

    Attributes
    ----------
    parameter : str
        Name of the assumed parameter.
    value : str
        Assumed value (string representation).
    source : str
        Literature source or rationale.
    uncertainty : str
        Uncertainty estimate or 'N/A'.
    """

    parameter: str
    value: str
    source: str
    uncertainty: str


@dataclass
class SimulationConfig:
    """Top-level simulation configuration loaded from YAML.

    Attributes
    ----------
    constants : FundamentalConstants
        Fundamental physical constants.
    lunar : LunarParameters
        Lunar body parameters.
    surface : SurfaceProperties
        Surface optical properties.
    regolith : RegolithProperties
        Regolith thermophysical properties.
    solver : SolverConfig
        Thermal solver configuration.
    raytracer : RaytracerConfig
        Raytracer configuration.
    illumination : IlluminationConfig
        Illumination computation configuration.
    synthetic_dem : SyntheticDEMConfig
        Synthetic DEM generation configuration.
    assumptions : list[Assumption]
        Registry of documented model assumptions.
    """

    constants: FundamentalConstants
    lunar: LunarParameters
    surface: SurfaceProperties
    regolith: RegolithProperties
    solver: SolverConfig
    raytracer: RaytracerConfig
    illumination: IlluminationConfig
    synthetic_dem: SyntheticDEMConfig
    assumptions: list[Assumption] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> SimulationConfig:
    """Load and validate a simulation configuration from a YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    SimulationConfig
        Fully populated, typed configuration object.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If required configuration keys are missing or values are invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    logger.info("Loading configuration from: %s", config_path)

    # --- Parse fundamental constants ---
    c = raw["constants"]
    constants = FundamentalConstants(
        stefan_boltzmann=float(c["stefan_boltzmann"]),
        solar_constant=float(c["solar_constant"]),
        astronomical_unit=float(c["astronomical_unit"]),
        speed_of_light=float(c["speed_of_light"]),
    )

    # --- Parse lunar parameters ---
    lun = raw["lunar"]
    lunar = LunarParameters(
        radius_m=float(lun["radius_m"]),
        sidereal_period_s=float(lun["sidereal_period_s"]),
        synodic_period_s=float(lun["synodic_period_s"]),
        obliquity_deg=float(lun["obliquity_deg"]),
        solar_angular_diameter_deg=float(lun["solar_angular_diameter_deg"]),
    )

    # --- Parse surface properties ---
    surf = raw["surface"]
    surface = SurfaceProperties(
        bond_albedo=float(surf["bond_albedo"]),
        thermal_emissivity=float(surf["thermal_emissivity"]),
        reflectance_model=str(surf["reflectance_model"]),
    )

    # --- Parse regolith properties ---
    reg = raw["regolith"]
    dens = reg["density"]
    cond = reg["conductivity"]
    cp = reg["specific_heat"]

    regolith = RegolithProperties(
        density_surface=float(dens["surface_kg_m3"]),
        density_deep=float(dens["deep_kg_m3"]),
        density_transition_m=float(dens["transition_depth_m"]),
        conductivity_surface=ConductivityLayer(
            k_contact=float(cond["surface"]["k_contact"]),
            k_radiative=float(cond["surface"]["k_radiative"]),
        ),
        conductivity_deep=ConductivityLayer(
            k_contact=float(cond["deep"]["k_contact"]),
            k_radiative=float(cond["deep"]["k_radiative"]),
        ),
        conductivity_layer_boundary_m=float(cond["layer_boundary_m"]),
        cp_coefficients=(
            float(cp["c0"]),
            float(cp["c1"]),
            float(cp["c2"]),
            float(cp["c3"]),
            float(cp["c4"]),
        ),
        cp_minimum=float(cp["minimum_value"]),
        geothermal_flux=float(reg["geothermal_flux"]),
    )

    # --- Parse solver config ---
    slv = raw["solver"]
    th = slv["thermal"]
    grid_cfg = th["grid"]
    newton_cfg = th["newton"]
    time_cfg = th["time"]

    solver = SolverConfig(
        method=str(th["method"]),
        grid=ThermalGridConfig(
            dz_surface_m=float(grid_cfg["dz_surface_m"]),
            growth_ratio=float(grid_cfg["growth_ratio"]),
            num_layers=int(grid_cfg["num_layers"]),
        ),
        dt_s=float(time_cfg["dt_s"]),
        max_dt_s=float(time_cfg["max_dt_s"]),
        newton=NewtonConfig(
            max_iterations=int(newton_cfg["max_iterations"]),
            tolerance_K=float(newton_cfg["tolerance_K"]),
            relaxation=float(newton_cfg["relaxation"]),
        ),
        initial_temperature_K=float(th["initial_temperature_K"]),
    )

    # --- Parse raytracer config ---
    rt = slv["raytracer"]
    bvh_cfg = rt["bvh"]
    raytracer = RaytracerConfig(
        max_leaf_triangles=int(bvh_cfg["max_leaf_triangles"]),
        sah_num_bins=int(bvh_cfg["sah_num_bins"]),
        epsilon=float(rt["epsilon"]),
        precision=str(rt["precision"]),
    )

    # --- Parse illumination config ---
    ill = slv["illumination"]
    illumination = IlluminationConfig(
        solar_disk_samples=int(ill["solar_disk_samples"]),
        sampling_method=str(ill["sampling_method"]),
        point_source_mode=bool(ill["point_source_mode"]),
    )

    # --- Parse synthetic DEM config ---
    sdem = raw["synthetic_dem"]
    synthetic_dem = SyntheticDEMConfig(
        crater_type=str(sdem["type"]),
        radius_m=float(sdem["radius_m"]),
        depth_m=float(sdem["depth_m"]),
        rim_height_m=float(sdem["rim_height_m"]),
        grid_resolution_m=float(sdem["grid_resolution_m"]),
        domain_padding_m=float(sdem["domain_padding_m"]),
        seed=int(sdem["seed"]),
    )

    # --- Build assumptions registry ---
    assumptions = _build_assumptions_registry(surface, regolith)

    config = SimulationConfig(
        constants=constants,
        lunar=lunar,
        surface=surface,
        regolith=regolith,
        solver=solver,
        raytracer=raytracer,
        illumination=illumination,
        synthetic_dem=synthetic_dem,
        assumptions=assumptions,
    )

    _validate_config(config)
    logger.info("Configuration loaded successfully. %d assumptions registered.", len(assumptions))

    return config


def _build_assumptions_registry(
    surface: SurfaceProperties,
    regolith: RegolithProperties,
) -> list[Assumption]:
    """Build the documented assumptions registry.

    Parameters
    ----------
    surface : SurfaceProperties
        Loaded surface properties.
    regolith : RegolithProperties
        Loaded regolith properties.

    Returns
    -------
    list[Assumption]
        List of all documented assumptions.
    """
    return [
        Assumption("Bond Albedo", str(surface.bond_albedo), "Vasavada et al., 2012", "±0.03"),
        Assumption(
            "Thermal Emissivity",
            str(surface.thermal_emissivity),
            "Bandfield et al., 2015",
            "±0.02",
        ),
        Assumption(
            "Geothermal Flux",
            f"{regolith.geothermal_flux} W/m²",
            "Langseth et al., 1976 (Apollo 15/17, equatorial)",
            "±50%",
        ),
        Assumption(
            "Surface Density",
            f"{regolith.density_surface} kg/m³",
            "Hayne et al., 2017",
            "±200 kg/m³",
        ),
        Assumption(
            "Deep Density",
            f"{regolith.density_deep} kg/m³",
            "Hayne et al., 2017",
            "±200 kg/m³",
        ),
        Assumption(
            "Surface k_contact",
            f"{regolith.conductivity_surface.k_contact} W/m/K",
            "Hayne et al., 2017",
            "±50%",
        ),
        Assumption(
            "Reflectance Model",
            surface.reflectance_model,
            "User requirement (Lambertian)",
            "N/A",
        ),
        Assumption("No Dust Levitation", "Excluded", "User requirement", "N/A"),
        Assumption("No Sub-pixel Roughness", "Excluded", "User requirement", "N/A"),
        Assumption(
            "Spatially Uniform Regolith",
            "Depth-dependent only",
            "Simplification",
            "Unknown",
        ),
    ]


def _validate_config(config: SimulationConfig) -> None:
    """Validate physical constraints on configuration values.

    Parameters
    ----------
    config : SimulationConfig
        Configuration to validate.

    Raises
    ------
    ValueError
        If any value is physically invalid.
    """
    if not (0.0 <= config.surface.bond_albedo <= 1.0):
        raise ValueError(
            f"Bond albedo must be in [0, 1], got {config.surface.bond_albedo}"
        )
    if not (0.0 < config.surface.thermal_emissivity <= 1.0):
        raise ValueError(
            f"Thermal emissivity must be in (0, 1], got {config.surface.thermal_emissivity}"
        )
    if config.constants.stefan_boltzmann <= 0:
        raise ValueError("Stefan-Boltzmann constant must be positive.")
    if config.constants.solar_constant <= 0:
        raise ValueError("Solar constant must be positive.")
    if config.regolith.density_surface <= 0 or config.regolith.density_deep <= 0:
        raise ValueError("Regolith densities must be positive.")
    if config.regolith.geothermal_flux < 0:
        raise ValueError("Geothermal flux cannot be negative.")
    if config.solver.grid.dz_surface_m <= 0:
        raise ValueError("Surface grid spacing must be positive.")
    if config.solver.grid.growth_ratio <= 1.0:
        raise ValueError("Grid growth ratio must be > 1.0.")
    if config.solver.dt_s <= 0:
        raise ValueError("Time step must be positive.")
    if config.solver.newton.tolerance_K <= 0:
        raise ValueError("Newton tolerance must be positive.")
    if config.raytracer.epsilon <= 0:
        raise ValueError("Raytracer epsilon must be positive.")

    logger.debug("Configuration validation passed.")


def log_assumptions(config: SimulationConfig) -> None:
    """Log all documented model assumptions to the logger.

    Parameters
    ----------
    config : SimulationConfig
        Configuration with populated assumptions registry.
    """
    logger.info("=" * 70)
    logger.info("MODEL ASSUMPTIONS REGISTRY")
    logger.info("=" * 70)
    for i, a in enumerate(config.assumptions, 1):
        logger.info(
            "  [%02d] %-25s = %-20s | Source: %-40s | Uncertainty: %s",
            i,
            a.parameter,
            a.value,
            a.source,
            a.uncertainty,
        )
    logger.info("=" * 70)


def log_platform_info() -> None:
    """Log platform and library version information for reproducibility."""
    logger.info("=" * 70)
    logger.info("PLATFORM INFORMATION (for reproducibility)")
    logger.info("=" * 70)
    logger.info("  Python:    %s", sys.version)
    logger.info("  Platform:  %s", platform.platform())
    logger.info("  Processor: %s", platform.processor())
    logger.info("  NumPy:     %s", np.__version__)
    logger.info("  Float64 eps: %e", np.finfo(np.float64).eps)
    logger.info("=" * 70)


def hash_array(arr: np.ndarray) -> str:
    """Compute SHA-256 hash of a NumPy array for reproducibility verification.

    Parameters
    ----------
    arr : np.ndarray
        Array to hash.

    Returns
    -------
    str
        Hex digest of the SHA-256 hash.
    """
    return hashlib.sha256(arr.tobytes()).hexdigest()
