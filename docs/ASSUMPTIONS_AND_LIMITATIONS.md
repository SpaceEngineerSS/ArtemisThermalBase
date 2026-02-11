# ArtemisThermalBase — Assumptions & Limitations

> **Academic Transparency Document**
>
> This document explicitly states every assumption, simplification, and known limitation of the ArtemisThermalBase thermal simulation engine. Scientific credibility requires honest disclosure of what the model *can* and *cannot* do.

---

## 1. Registered Model Assumptions

The following 10 assumptions are programmatically registered in `core_engine/constants.py` and logged at the start of every simulation run.

| # | Parameter | Value | Source | Uncertainty |
|---|-----------|-------|--------|-------------|
| 1 | Bond Albedo | 0.12 | Vasavada et al., 2012 | ±0.03 |
| 2 | Thermal Emissivity | 0.95 | Bandfield et al., 2015 | ±0.02 |
| 3 | Geothermal Flux | 0.018 W/m² | Langseth et al., 1976 (Apollo 15/17) | ±50% |
| 4 | Surface Density | 1100 kg/m³ | Hayne et al., 2017 | ±200 kg/m³ |
| 5 | Deep Density | 1800 kg/m³ | Hayne et al., 2017 | ±200 kg/m³ |
| 6 | Surface k_contact | 7.4×10⁻⁴ W/m/K | Hayne et al., 2017 | ±50% |
| 7 | Reflectance Model | Lambertian | User requirement | N/A |
| 8 | No Dust Levitation | Excluded | User requirement | N/A |
| 9 | No Sub-pixel Roughness | Excluded | User requirement | N/A |
| 10 | Spatially Uniform Regolith | Depth-dependent only | Simplification | Unknown |

---

## 2. Known Limitations

### 2.1 No Multi-Bounce IR Scattering

**Impact: HIGH — PSR temperatures may be underestimated by ~10–20 K**

The current model computes only **direct solar illumination** and does not include infrared thermal radiation exchange between terrain facets. In deep craters like Shackleton, sunlit rim surfaces emit thermal IR that illuminates/warms the permanently shadowed floor. Without this term, cold trap temperatures are biased low.

This is a planned **Milestone 4** feature requiring view-factor computation between all $N^2$ face pairs.

### 2.2 No Temperature-Dependent Albedo

**Impact: LOW–MEDIUM**

The Bond albedo is treated as spatially and thermally uniform (A = 0.12). In reality, albedo varies with:
- Composition (highlands vs. mare)
- Solar incidence angle (opposition surge)
- Temperature (minor effect)

A future Hapke reflectance model would address direction-dependent effects (Milestone 5).

### 2.3 1D Heat Flow Assumption (Lateral Conduction Ignored)

**Impact: LOW for most surfaces; MEDIUM at sharp shadow boundaries**

Each DEM facet is treated as an independent 1D thermal column with no lateral heat conduction. This is valid when:

$$
L_{\text{face}} \gg d_{\text{skin}} \approx \sqrt{\frac{k P}{\pi \rho c_p}}
$$

For typical regolith properties and a 29.5-day lunar period, $d_{\text{skin}} \approx 0.3$ m. At 20 m/px DEM resolution, each face is ~20 m wide, so $L/d \approx 67$ — lateral conduction is negligible.

**Exception**: At sharp PSR boundaries where temperature gradients exceed ~100 K/m, 3D effects may become significant.

### 2.4 Equatorial Geothermal Flux Applied to Polar Region

**Impact: MEDIUM for deep temperatures**

The geothermal heat flux (0.018 W/m²) was measured at the Apollo 15 and 17 equatorial landing sites. Polar regions may have different internal heat distribution due to:
- Crustal thickness variations
- Tidal heating anisotropy
- Compositional differences

The ±50% uncertainty partially accounts for this.

### 2.5 No Dust Levitation or Electrostatic Transport

**Impact: LOW**

Electrostatically charged dust particles can be lofted from the surface at the terminator (Day/night boundary) due to photoelectric charging. This may affect local albedo and surface thermal properties. The effect is excluded as it requires particle transport modeling.

### 2.6 No Sub-pixel Roughness

**Impact: MEDIUM for thermal emission models**

Surface roughness at scales smaller than the DEM resolution (< 20 m) affects:
- Thermal inradiance at grazing solar angles
- Effective emissivity (cavity effect)
- Shadow fraction near the terminator

Roughness could be parameterized using a Gaussian surface model (Bandfield et al., 2015).

### 2.7 Spatially Uniform Regolith

**Impact: MEDIUM**

Regolith properties (k, ρ, cₚ) vary only with depth, not laterally. In reality:
- Highland regolith differs from mare regolith
- Rocky ejecta near craters has different thermal inertia
- PSR regolith may contain water ice (higher thermal conductivity)

### 2.8 No Orbital Eccentricity / Distance Correction

**Impact: LOW**

The solar constant is fixed at 1361 W/m² (1 AU). Earth-Moon distance variation (~±1.7%) causes ~±3.4% flux variation throughout the year. This is currently not modeled.

### 2.9 No Atmospheric Effects

**Impact: NONE**

The Moon has no significant atmosphere. This is correctly handled — no atmospheric scattering, absorption, or convection is included.

### 2.10 No Spacecraft/Instrument Self-Heating

**Impact: N/A**

The simulation models the natural thermal environment only. Spacecraft thermal interactions are out of scope for this version.

---

## 3. Numerical Approximations

| Approximation | Details | Error Bound |
|---------------|---------|-------------|
| Crank-Nicolson temporal discretization | $O(\Delta t^2)$ | < 0.1 K at dt = 120 s |
| Non-uniform FD spatial discretization | $O(\Delta z^2)$ | < 0.05 K with geometric grid |
| Newton linearization of $T^4$ | Converges in 2–4 iterations | $10^{-4}$ K tolerance |
| Fibonacci disk sampling (64 points) | ~1.5% noise at shadow edges | < 0.02 illumination fraction |
| Harmonic mean for interface conductivity | Exact for piecewise-constant k | N/A (exact) |
| Thomas algorithm (TDMA) | Exact for tridiagonal systems | Machine precision |

---

## 4. Validation Status

| Against | Status | Reference |
|---------|--------|-----------|
| LRO Diviner surface temperatures | Planned (Milestone 5) | Paige et al. (2010) |
| Analytical solutions (flat surface) | Partial — initial testing | Spencer et al. (1989) |
| Energy conservation (internal check) | ✅ Implemented | `compute_internal_energy()` |

---

## 5. Improvement Roadmap

| Milestone | Feature | Impact on Accuracy |
|-----------|---------|--------------------|
| 3 | C++ BVH raytracer (pybind11) | Performance only (no accuracy change) |
| 4 | Multi-bounce IR view factors | +10–20 K in PSRs |
| 5 | Diviner validation + Hapke reflectance | Quantified error bars |
| Future | 3D heat conduction | ±1–3 K at shadow boundaries |
| Future | Orbital eccentricity correction | ±3% flux |
| Future | Water ice thermal properties | Critical for volatile stability |
