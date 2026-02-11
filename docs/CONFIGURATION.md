# ArtemisThermalBase — Configuration Reference

> Complete guide to `config/default_config.yaml`
>
> Every tunable parameter with type, units, default value, valid range, and physical implications.

---

## Table of Contents

1. [Metadata](#1-metadata)
2. [Target Region](#2-target-region)
3. [Physical Constants](#3-physical-constants)
4. [Lunar Parameters](#4-lunar-parameters)
5. [Surface Optical Properties](#5-surface-optical-properties)
6. [Regolith Thermophysical Properties](#6-regolith-thermophysical-properties)
7. [Solver Configuration](#7-solver-configuration)
8. [Synthetic DEM](#8-synthetic-dem)
9. [Output & Visualization](#9-output--visualization)
10. [Reproducibility](#10-reproducibility)
11. [Parameter Sensitivity Guide](#11-parameter-sensitivity-guide)

---

## 1. Metadata

```yaml
metadata:
  project_name: "ArtemisThermalBase"
  version: "0.1.0"
  author: "Mehmet Gümüş"
```

Informational only. Logged for reproducibility and included in output metadata files.

---

## 2. Target Region

```yaml
target:
  name: "Shackleton Crater"
  latitude_deg: -89.54
  longitude_deg: 129.78
  diameter_km: 21.0
  depth_km: 4.2
```

| Parameter | Type | Units | Range | Source |
|-----------|------|-------|-------|--------|
| `latitude_deg` | float | degrees | [−90, 90] | Zuber et al. (2012) |
| `longitude_deg` | float | degrees | [−180, 360] | Zuber et al. (2012) |
| `diameter_km` | float | km | > 0 | LOLA measurements |
| `depth_km` | float | km | > 0 | Topographic analysis |

**Physical implication**: These coordinates are passed to the Skyfield ephemeris engine to compute solar elevation and azimuth angles at the target site. Changing the latitude significantly affects shadow geometry — equatorial sites receive direct sunlight for ~50% of a lunar day, while polar sites may receive near-zero direct illumination.

---

## 3. Physical Constants

```yaml
constants:
  stefan_boltzmann: 5.670374419e-8    # [W/m²/K⁴]
  solar_constant: 1361.0              # [W/m²]
  astronomical_unit: 1.495978707e11   # [m]
  speed_of_light: 2.99792458e8        # [m/s]
```

| Parameter | Type | Units | Source |
|-----------|------|-------|--------|
| `stefan_boltzmann` | float | W/m²/K⁴ | CODATA 2018 |
| `solar_constant` | float | W/m² | Kopp & Lean (2011) |
| `astronomical_unit` | float | m | IAU 2012 |

> [!CAUTION]
> **Do not modify these values** unless you are implementing a specific physical scenario (e.g., a different epoch of solar luminosity). Incorrect values will produce physically meaningless temperatures.

---

## 4. Lunar Parameters

```yaml
lunar:
  radius_m: 1737400.0
  sidereal_period_s: 2360591.5104
  synodic_period_s: 2551442.9
  obliquity_deg: 1.5424
  solar_angular_diameter_deg: 0.533
```

| Parameter | Type | Units | Range | Source |
|-----------|------|-------|-------|--------|
| `radius_m` | float | m | — | IAU mean radius |
| `sidereal_period_s` | float | s | — | 27.321661 days |
| `synodic_period_s` | float | s | — | 29.530589 days |
| `obliquity_deg` | float | degrees | — | Rambaux & Williams (2011) |
| `solar_angular_diameter_deg` | float | degrees | (0, 1] | Mean Sun diameter from Moon |

**Physical implication**: The `solar_angular_diameter_deg` controls the size of penumbra regions. A larger value produces wider penumbra bands at shadow edges. Setting this value to 0 would be equivalent to a point-source sun (no penumbra).

---

## 5. Surface Optical Properties

```yaml
surface:
  bond_albedo: 0.12
  thermal_emissivity: 0.95
  reflectance_model: "lambertian"
```

| Parameter | Type | Units | Valid Range | Default | Source |
|-----------|------|-------|-------------|---------|--------|
| `bond_albedo` | float | — | [0.0, 1.0] | 0.12 | Vasavada et al. (2012) |
| `thermal_emissivity` | float | — | (0.0, 1.0] | 0.95 | Bandfield et al. (2015) |
| `reflectance_model` | string | — | `"lambertian"` | `"lambertian"` | — |

**Physical implications**:
- **Increasing `bond_albedo`** → more solar flux reflected → lower equilibrium temperatures. A 10% increase in albedo reduces peak temperatures by ~15–20 K.
- **Decreasing `thermal_emissivity`** → less efficient radiative cooling → higher nighttime temperatures. PSR temperature estimates are particularly sensitive to this parameter.
- **Reflectance model**: Only Lambertian is currently supported. A future Hapke model (Milestone 5) would compute direction-dependent reflectance.

---

## 6. Regolith Thermophysical Properties

### 6.1 Density Profile

```yaml
regolith:
  density:
    surface_kg_m3: 1100.0
    deep_kg_m3: 1800.0
    transition_depth_m: 0.06
```

| Parameter | Type | Units | Valid Range | Default | Source |
|-----------|------|-------|-------------|---------|--------|
| `surface_kg_m3` | float | kg/m³ | [500, 2000] | 1100 | Hayne et al. (2017) |
| `deep_kg_m3` | float | kg/m³ | [1000, 3000] | 1800 | Hayne et al. (2017) |
| `transition_depth_m` | float | m | (0, 1] | 0.06 | Hayne et al. (2017) |

**Physical implication**: The density profile controls thermal inertia ($I = \sqrt{k \rho c_p}$). Higher density → higher thermal inertia → slower temperature response. The transition depth `H` determines how quickly the density increases with depth.

### 6.2 Thermal Conductivity

```yaml
  conductivity:
    surface:
      k_contact: 7.4e-4       # [W/m/K]
      k_radiative: 2.0e-11    # [W/m/K⁴]
    deep:
      k_contact: 3.4e-3       # [W/m/K]
      k_radiative: 1.0e-11    # [W/m/K⁴]
    layer_boundary_m: 0.02     # [m]
```

| Parameter | Type | Units | Valid Range | Source |
|-----------|------|-------|-------------|--------|
| `k_contact` | float | W/m/K | [10⁻⁴, 10⁻²] | Hayne et al. (2017) |
| `k_radiative` | float | W/m/K⁴ | [10⁻¹², 10⁻¹⁰] | Hayne et al. (2017) |
| `layer_boundary_m` | float | m | (0, 1] | Hayne et al. (2017) |

**Physical implications**:
- **`k_contact`** dominates at low temperatures (< 200 K). Increasing it raises PSR floor temperatures.
- **`k_radiative`** dominates at high temperatures (> 300 K). Controls how quickly heat percolates into the subsurface during daytime.
- **Uncertainty is ±50%** — this is the single largest source of uncertainty in lunar thermal models.

### 6.3 Specific Heat Capacity

```yaml
  specific_heat:
    c0: -3.6125
    c1: 2.7431
    c2: 2.3616e-3
    c3: -1.2340e-5
    c4: 8.9093e-9
    minimum_value: 8.0       # [J/kg/K]
```

Polynomial fit valid for 80 K < T < 400 K. The `minimum_value` prevents division by zero at $T \to 0$ in the CN solver.

### 6.4 Geothermal Heat Flux

```yaml
  geothermal_flux: 0.018     # [W/m²]
```

| Parameter | Type | Units | Valid Range | Source |
|-----------|------|-------|-------------|--------|
| `geothermal_flux` | float | W/m² | [0, 0.1] | Langseth et al. (1976) |

**Physical implication**: This flux sets the deep temperature boundary condition. At 0.018 W/m², the deep subsurface temperature is ~250 K. For PSR studies, this flux provides a lower limit on cold trap temperatures. **Uncertainty: ±50%** (equatorial measurement applied to polar region).

---

## 7. Solver Configuration

### 7.1 Thermal Solver

```yaml
solver:
  thermal:
    method: "crank_nicolson"
    grid:
      dz_surface_m: 5.0e-4          # [m] = 0.5 mm
      growth_ratio: 1.07
      num_layers: 100
    time:
      dt_s: 120.0                    # [s] = 2 minutes
      max_dt_s: 3600.0               # [s]
    newton:
      max_iterations: 20
      tolerance_K: 1.0e-4            # [K]
      relaxation: 1.0
    initial_temperature_K: 200.0     # [K]
```

| Parameter | Type | Units | Valid Range | Default |
|-----------|------|-------|-------------|---------|
| `dz_surface_m` | float | m | [10⁻⁴, 10⁻²] | 5×10⁻⁴ |
| `growth_ratio` | float | — | (1.0, 2.0] | 1.07 |
| `num_layers` | int | — | [10, 500] | 100 |
| `dt_s` | float | s | [1, 3600] | 120 |
| `tolerance_K` | float | K | [10⁻⁶, 10⁻¹] | 10⁻⁴ |
| `initial_temperature_K` | float | K | [50, 400] | 200 |

**Physical implications**:
- **Increasing `dt_s`** speeds up the simulation but may cause temporal discretization errors during fast thermal transients (e.g., sunrise/sunset). The Crank-Nicolson scheme is unconditionally stable, but accuracy degrades with large time steps. For accurate penumbra thermal transients, use `dt ≤ 300 s`.
- **Decreasing `dz_surface_m`** improves resolution of the thermal skin depth but increases the number of layers and computation time.
- **`growth_ratio`** close to 1.0 gives near-uniform spacing (expensive but accurate). Values > 1.2 may under-resolve the thermal wave at intermediate depths.
- **`num_layers = 100`** with `growth_ratio = 1.07` gives a grid ~2 m deep, sufficient for thermal isolation.

### 7.2 Raytracer

```yaml
  raytracer:
    bvh:
      max_leaf_triangles: 4
      sah_num_bins: 16
    epsilon: 1.0e-10
    precision: "float64"
```

| Parameter | Type | Units | Valid Range | Default |
|-----------|------|-------|-------------|---------|
| `max_leaf_triangles` | int | — | [1, 16] | 4 |
| `sah_num_bins` | int | — | [4, 64] | 16 |
| `epsilon` | float | — | [10⁻¹², 10⁻⁶] | 10⁻¹⁰ |

**Physical implications**:
- **`epsilon`** is the zero-test threshold for Möller-Trumbore intersection. Too large → misses grazing rays at shadow edges. Too small → floating-point cancellation causes ray leakage. The default 10⁻¹⁰ balances both risks.
- **`max_leaf_triangles = 4`** is optimal for cache efficiency on modern CPUs.

### 7.3 Illumination

```yaml
  illumination:
    solar_disk_samples: 64
    sampling_method: "fibonacci"
    point_source_mode: false
```

| Parameter | Type | Units | Valid Range | Default |
|-----------|------|-------|-------------|---------|
| `solar_disk_samples` | int | — | [1, 256] | 64 |
| `point_source_mode` | bool | — | — | false |

**Physical implications**:
- **`solar_disk_samples`** controls the Monte Carlo resolution of penumbra. 64 samples gives ~1.5% noise at shadow edges. 32 is acceptable for fast previews; 128+ for publication quality.
- **`point_source_mode = true`** disables penumbra entirely (binary shadow, ~64× faster). Useful for testing but produces unrealistic sharp shadow boundaries.

---

## 8. Synthetic DEM

```yaml
synthetic_dem:
  type: "parabolic_bowl"
  radius_m: 10500.0
  depth_m: 4200.0
  rim_height_m: 300.0
  grid_resolution_m: 20.0
  domain_padding_m: 3000.0
  seed: 42
```

| Parameter | Type | Units | Valid Range | Default |
|-----------|------|-------|-------------|---------|
| `type` | string | — | `"parabolic_bowl"` | `"parabolic_bowl"` |
| `radius_m` | float | m | [100, 100000] | 10500 |
| `depth_m` | float | m | [10, 10000] | 4200 |
| `rim_height_m` | float | m | [0, 2000] | 300 |
| `grid_resolution_m` | float | m/px | [1, 500] | 20 |
| `seed` | int | — | — | 42 |

**Physical implications**:
- **Smaller `radius_m`** → faster simulation (fewer mesh triangles) but may not capture large-scale shadow geometry.
- **`grid_resolution_m`** controls the DEM pixel size and thus the number of triangles. At 20 m/px a 21 km crater produces ~500k triangles. At 10 m/px the count quadruples.
- **`seed`** ensures reproducible synthetic terrain noise. Change for ensemble simulations.

---

## 9. Output & Visualization

```yaml
output:
  directory: "results/"
  formats:
    illumination_map: "npy"
    thermal_map: "npy"
    time_series: "csv"
  save_subsurface_profiles: true
  compress: true

visualization:
  colormap: "inferno"
  dpi: 150
  interactive: false
```

**Note**: The `output.directory` is overridden by the `--output` CLI argument at runtime.

---

## 10. Reproducibility

```yaml
reproducibility:
  random_seed: 42
  float_precision: "float64"
  log_platform_info: true
  hash_outputs: true
```

- **`log_platform_info`**: Logs Python version, OS, CPU, NumPy version for debugging cross-platform discrepancies.
- **`hash_outputs`**: Computes SHA-256 hash of output arrays for bitwise reproducibility verification.

---

## 11. Parameter Sensitivity Guide

| Parameter | Sensitivity | Effect on PSR Temperature |
|-----------|-------------|--------------------------|
| `k_contact` (surface) | **High** | ±50% change → ±15–25 K |
| `thermal_emissivity` | **High** | ±0.02 → ±5–10 K |
| `bond_albedo` | **Medium** | ±0.03 → ±3–5 K (dayside only) |
| `geothermal_flux` | **Medium** | ±50% → ±5–8 K (deep temps) |
| `density_surface` | **Low** | ±200 kg/m³ → ±2–3 K |
| `dt_s` | **N/A** | Numerical parameter, no physical effect if converged |
| `solar_disk_samples` | **N/A** | Controls penumbra noise, not equilibrium temperatures |
