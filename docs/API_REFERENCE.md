# ArtemisThermalBase — API Reference

> Module-based reference for key classes and functions.
>
> For detailed physics, see [PHYSICS_MODEL.md](PHYSICS_MODEL.md). For configuration, see [CONFIGURATION.md](CONFIGURATION.md).

---

## Table of Contents

1. [Simulation Orchestration](#1-simulation-orchestration)
2. [Core Engine — Raytracing & Illumination](#2-core-engine)
3. [Thermal Solver](#3-thermal-solver)
4. [Data Ingestion](#4-data-ingestion)
5. [Visualization](#5-visualization)
6. [Data Structures](#6-data-structures)

---

## 1. Simulation Orchestration

### `simulation.runner.SimulationRunner`

**The main entry point for running a simulation.**

```python
from simulation.runner import SimulationRunner
from core_engine.constants import load_config

config = load_config("config/default_config.yaml")
runner = SimulationRunner(config=config, crater_radius_m=2500.0)
```

#### Constructor

```python
SimulationRunner(
    config: SimulationConfig,
    crater_radius_m: float | None = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `SimulationConfig` | Full configuration loaded from YAML |
| `crater_radius_m` | `float \| None` | Override crater radius [m]. If None, uses config value |

#### `run()`

```python
runner.run(
    start_time: datetime,
    duration_hours: float = 24.0,
    dt_s: float | None = None,
    output_interval_s: float = 3600.0,
    num_probes: int = 3,
    point_source_mode: bool | None = None,
    save_data: bool = True,
    output_dir: Path | str = "output",
    external_dem: DEMData | None = None,
) -> SimulationResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_time` | `datetime` | — | UTC simulation start time |
| `duration_hours` | `float` | 24.0 | Simulation duration [hours] |
| `dt_s` | `float \| None` | None | Override time step [s] |
| `output_interval_s` | `float` | 3600.0 | Snapshot save interval [s] |
| `num_probes` | `int` | 3 | Number of temperature probes |
| `point_source_mode` | `bool \| None` | None | Override penumbra mode |
| `save_data` | `bool` | True | Save results to disk |
| `output_dir` | `Path \| str` | `"output"` | Output directory |
| `external_dem` | `DEMData \| None` | None | Pre-loaded DEM (bypasses synthetic) |

**Returns**: `SimulationResults` — Container with all output data.

---

### `simulation.io_manager`

```python
from simulation.io_manager import save_results, load_results

save_results(results, output_dir="output")
data = load_results("output")  # Returns dict of numpy arrays
```

| Function | Description |
|----------|-------------|
| `save_results(results, output_dir)` | Serialize `SimulationResults` to `.npy` + `.json` |
| `load_results(data_dir) → dict` | Load saved arrays for re-rendering |

---

## 2. Core Engine

### `core_engine.raytracer`

#### `build_bvh()`

```python
build_bvh(
    mesh: TriangleMesh,
    max_leaf_triangles: int = 4,
    sah_num_bins: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

**Returns**: `(bvh_nodes, tri_verts, ordered_indices)` — Flattened BVH arrays for Numba traversal.

#### `compute_shadow_map_point_source()`

```python
compute_shadow_map_point_source(
    face_centroids: np.ndarray,   # (N, 3)
    face_normals: np.ndarray,     # (N, 3)
    sun_dir: np.ndarray,          # (3,)
    bvh_nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered_tri_indices: np.ndarray,
    epsilon: float,
) -> np.ndarray  # (N,) float64, 0.0 or 1.0
```

#### `compute_shadow_map_extended_source()`

```python
compute_shadow_map_extended_source(
    face_centroids: np.ndarray,   # (N, 3)
    face_normals: np.ndarray,     # (N, 3)
    sun_samples: np.ndarray,      # (M, 3) — disk sample directions
    bvh_nodes: np.ndarray,
    tri_verts: np.ndarray,
    ordered_tri_indices: np.ndarray,
    epsilon: float,
) -> np.ndarray  # (N,) float64, [0.0, 1.0]
```

---

### `core_engine.illumination.IlluminationEngine`

```python
from core_engine.illumination import IlluminationEngine

engine = IlluminationEngine(
    mesh=mesh,
    num_samples=64,
    point_source_mode=False,
)
result = engine.compute(sun_dir=sun_direction_vector)
```

#### Constructor

```python
IlluminationEngine(
    mesh: TriangleMesh,
    bvh_data: tuple | None = None,
    solar_angular_radius_rad: float = np.radians(0.533 / 2.0),
    num_samples: int = 64,
    point_source_mode: bool = False,
    epsilon: float = 1e-10,
    max_leaf_triangles: int = 4,
)
```

#### `compute()`

```python
engine.compute(
    sun_dir: np.ndarray,
    point_source_override: bool | None = None,
) -> IlluminationResult
```

**Returns**: `IlluminationResult` with fields `illumination`, `sun_dir`, `sun_elevation_deg`, `num_samples`, `mode`, `stats`.

---

### `core_engine.solar_disk`

```python
from core_engine.solar_disk import generate_solar_disk_samples

samples = generate_solar_disk_samples(
    sun_center_dir=np.array([0.0, 0.1, 0.995]),
    angular_radius_rad=np.radians(0.266),
    num_samples=64,
)  # Returns (64, 3) array of unit vectors
```

---

### `core_engine.mesh`

```python
from core_engine.mesh import dem_to_mesh, TriangleMesh

mesh = dem_to_mesh(dem_data)  # DEMData → TriangleMesh
```

`TriangleMesh` fields: `vertices`, `triangles`, `face_normals`, `face_centroids`, `face_areas`, `num_faces`.

---

## 3. Thermal Solver

### `thermal_solver.crank_nicolson.CrankNicolsonSolver`

```python
from thermal_solver.crank_nicolson import CrankNicolsonSolver, create_thermal_column

solver = CrankNicolsonSolver(config)
column = create_thermal_column(config)

# Advance one time step
solver.step(column, Q_solar=150.0, Q_ir=0.0, dt=120.0)

# Query output
T_surface = column.T[0]
E_total = solver.compute_internal_energy(column)
Q_rad = solver.compute_surface_radiation(T_surface)
```

#### `step()`

```python
solver.step(
    column: ThermalColumn,
    Q_solar: float,          # Absorbed solar flux [W/m²]
    Q_ir: float = 0.0,      # Absorbed IR from terrain [W/m²]
    dt: float | None = None, # Override time step [s]
) -> None  # Modifies column.T in-place
```

#### `compute_surface_radiation(T_surf: float) → float`

Returns $\varepsilon \sigma T^4$ [W/m²].

#### `compute_internal_energy(column: ThermalColumn) → float`

Returns total stored energy [J/m²] for conservation verification.

---

### `thermal_solver.regolith_properties`

```python
from thermal_solver.regolith_properties import build_property_functions

k_func, cp_func, rho_func = build_property_functions(config.regolith)

k = k_func(T=200.0, z=0.01)   # W/m/K
cp = cp_func(T=200.0)          # J/kg/K
rho = rho_func(z=0.05)         # kg/m³
```

All returned functions are **Numba JIT-compiled** for use inside `@njit` code.

---

## 4. Data Ingestion

### `data_ingestion.lola_loader.LOLALoader`

```python
from data_ingestion.lola_loader import LOLALoader

loader = LOLALoader(
    nodata_threshold=-1.0e30,
    fill_nodata=True,
    center_elevation=True,
)
dem = loader.load_dem("data/sample_lola_dem.tif")
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nodata_threshold` | `float` | -1e30 | Values below this are treated as NoData |
| `fill_nodata` | `bool` | True | Fill NoData with nearest-neighbor interpolation |
| `center_elevation` | `bool` | True | Subtract mean to center around zero |

#### `load_dem()`

```python
loader.load_dem(
    file_path: str | Path,
    bounds: tuple[float, float, float, float] | None = None,
    max_size: int | None = None,
) -> DEMData
```

**Returns**: `DEMData` with fields `elevation`, `x_coords`, `y_coords`, `resolution_m`, `metadata`.

---

### `data_ingestion.synthetic_dem`

```python
from data_ingestion.synthetic_dem import generate_synthetic_dem

dem = generate_synthetic_dem(config.synthetic_dem)
```

---

### `data_ingestion.ephemeris`

```python
from data_ingestion.ephemeris import get_sun_direction

sun_dir = get_sun_direction(
    time_utc=datetime(2025, 1, 1),
    lat_deg=-89.54,
    lon_deg=129.78,
)  # Returns (3,) unit vector
```

---

## 5. Visualization

### `visualization.hero_renderer`

```python
from visualization.hero_renderer import render_hero_image, render_from_saved_data

# From live data
path = render_hero_image(
    face_centroids=centroids,
    thermal_grid=temperatures,
    illumination_grid=illumination,
    output_path="output/hero_artemis.png",
    dpi=300,
)

# From saved files
path = render_from_saved_data(data_dir="output", dpi=600)
```

### `visualization.plotter`

```python
from visualization.plotter import plot_illumination_map, plot_thermal_map

plot_illumination_map(centroids, illumination, "output/illum.png")
plot_thermal_map(centroids, temperatures, "output/thermal.png")
```

---

## 6. Data Structures

### `DEMData`

```python
@dataclass
class DEMData:
    elevation: np.ndarray     # (ny, nx) float64 [m]
    x_coords: np.ndarray      # (nx,) float64 [m]
    y_coords: np.ndarray      # (ny,) float64 [m]
    resolution_m: float       # Grid spacing [m]
    metadata: dict             # Source info
```

### `TriangleMesh`

```python
@dataclass
class TriangleMesh:
    vertices: np.ndarray       # (V, 3)
    triangles: np.ndarray      # (F, 3) int
    face_normals: np.ndarray   # (F, 3)
    face_centroids: np.ndarray # (F, 3)
    face_areas: np.ndarray     # (F,)
    num_faces: int
```

### `ThermalColumn`

```python
@dataclass
class ThermalColumn:
    z: np.ndarray    # (N+1,) depth grid [m]
    T: np.ndarray    # (N+1,) temperatures [K]
    dz: np.ndarray   # (N,) grid spacings [m]
    dz_bar: np.ndarray  # (N-1,) averaged spacings [m]
```

### `SimulationResults`

```python
@dataclass
class SimulationResults:
    times: list[datetime]
    surface_temps: list[np.ndarray]
    illumination_maps: list[np.ndarray]
    sun_elevations: list[float]
    probe_data: dict[str, list[float]]
    face_centroids: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    dem_elevation: np.ndarray
    metadata: dict
```

### `IlluminationResult`

```python
@dataclass
class IlluminationResult:
    illumination: np.ndarray    # (F,) [0, 1]
    sun_dir: np.ndarray         # (3,)
    sun_elevation_deg: float
    num_samples: int
    mode: str                    # "point_source" or "extended_source"
    stats: dict[str, float]
```
