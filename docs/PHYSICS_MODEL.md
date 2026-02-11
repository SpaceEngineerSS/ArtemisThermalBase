# ArtemisThermalBase — Physical Model

> **Technical Whitepaper: First-Principles Derivation of the Thermal Simulation Engine**
>
> Author: Mehmet Gümüş · Version 0.1.0 · February 2026

---

## Table of Contents

1. [Surface Energy Balance](#1-surface-energy-balance)
2. [Extended Solar Source Model](#2-extended-solar-source-model)
3. [BVH-Accelerated Raytracing](#3-bvh-accelerated-raytracing)
4. [Subsurface Heat Equation](#4-subsurface-heat-equation)
5. [Crank-Nicolson Discretization](#5-crank-nicolson-discretization)
6. [Newton-Raphson Surface Boundary Condition](#6-newton-raphson-surface-boundary-condition)
7. [Thomas Algorithm (Tridiagonal Solver)](#7-thomas-algorithm)
8. [Regolith Thermophysical Properties](#8-regolith-thermophysical-properties)
9. [Coordinate Systems](#9-coordinate-systems)
10. [References](#10-references)

---

## 1. Surface Energy Balance

At each triangular DEM facet $i$, the surface temperature $T_s$ is governed by a radiative–conductive energy balance at the regolith–vacuum interface ($z = 0$):

$$
(1 - A) \cdot S_0 \cdot \cos\theta_i \cdot f_i + Q_{\text{IR},i} + Q_{\text{geo}} = \varepsilon \sigma T_s^4 + q_{\text{cond}}
$$

where each term is derived as follows:

### Term 1: Absorbed Solar Flux

$$
Q_{\text{solar},i} = (1 - A) \cdot S_0 \cdot \cos\theta_i \cdot f_i
$$

| Symbol | Meaning | Value | Source |
|--------|---------|-------|--------|
| $A$ | Bond albedo | 0.12 | Vasavada et al. (2012) |
| $S_0$ | Solar constant at 1 AU | 1361 W/m² | Kopp & Lean (2011) |
| $\cos\theta_i$ | Cosine of solar incidence angle | $\hat{n}_i \cdot \hat{s}$ | Geometry |
| $f_i$ | Illumination fraction | $[0, 1]$ | Raytracer output |

The incidence angle $\theta_i$ is the angle between the face normal $\hat{n}_i$ and the sun direction vector $\hat{s}$. When $\cos\theta_i < 0$, the facet is self-shadowed (facing away from the sun) and $Q_{\text{solar}} = 0$.

The illumination fraction $f_i$ captures topographic shadowing:
- **Point source mode**: $f_i \in \{0, 1\}$ (binary shadow)
- **Extended source mode**: $f_i \in [0, 1]$ (fractional penumbra from solar disk sampling)

### Term 2: Outgoing Thermal Radiation

$$
Q_{\text{rad}} = \varepsilon \sigma T_s^4
$$

| Symbol | Meaning | Value | Source |
|--------|---------|-------|--------|
| $\varepsilon$ | Broadband thermal emissivity | 0.95 | Bandfield et al. (2015) |
| $\sigma$ | Stefan-Boltzmann constant | $5.670374 \times 10^{-8}$ W/m²/K⁴ | CODATA 2018 |

### Term 3: Geothermal Heat Flux

$$
Q_{\text{geo}} = 0.018 \text{ W/m²}
$$

Source: Apollo 15/17 heat flow measurements (Langseth et al., 1976). This equatorial measurement is applied uniformly to the south pole region — a known assumption documented in the model registry.

### Term 4: Conductive Heat Flux

$$
q_{\text{cond}} = -k(T) \cdot \left.\frac{\partial T}{\partial z}\right|_{z=0}
$$

This couples the surface energy balance to the subsurface heat equation (Section 4).

> [!NOTE]
> **Current simplification**: $Q_{\text{IR},i} = 0$ (no multi-bounce infrared scattering between terrain facets). This is a planned Milestone 4 feature. PSR temperatures may be underestimated by ~10–20 K due to this omission (Paige et al., 2010).

---

## 2. Extended Solar Source Model

### Solar Disk Geometry

The Sun is not a point source. From the Moon's surface, it subtends a finite angular diameter:

$$
\theta_{\text{sun}} = 0.533° \quad \Rightarrow \quad \theta_r = \frac{\theta_{\text{sun}}}{2} \approx 4.65 \times 10^{-3} \text{ rad}
$$

This produces **penumbra** — partially shadowed regions at the edges of topographic shadows — critical for accurate thermal modeling near PSR boundaries.

### Solid Angle

The exact solid angle of the solar disk (spherical cap, not small-angle approximation):

$$
\Omega_{\text{sun}} = 2\pi(1 - \cos\theta_r) \approx 6.79 \times 10^{-5} \text{ sr}
$$

### Fibonacci Spiral Sampling

To approximate the integral over the solar disk, we distribute $N$ sample directions uniformly across the disk using a **Fibonacci spiral pattern** on a spherical cap.

For sample index $i = 0, 1, \ldots, N-1$:

$$
\phi_i = i \cdot \Phi_{\text{golden}}, \qquad \Phi_{\text{golden}} = \pi(3 - \sqrt{5}) \approx 2.3996 \text{ rad}
$$

$$
h_i = 1 - \frac{i + 0.5}{N} \cdot (1 - \cos\theta_r)
$$

$$
r_i = \sqrt{1 - h_i^2}
$$

The sample direction in the local frame (z-axis = sun center):

$$
\hat{d}_i^{\text{local}} = \begin{pmatrix} r_i \cos\phi_i \\ r_i \sin\phi_i \\ h_i \end{pmatrix}
$$

These are then rotated to the global frame using a **Rodrigues rotation matrix** $\mathbf{R}$ that maps $\hat{z} \to \hat{s}_{\text{center}}$:

$$
\hat{d}_i = \mathbf{R} \cdot \hat{d}_i^{\text{local}}
$$

### Illumination Fraction

Each sample direction gets a shadow ray test via the BVH raytracer. The illumination fraction is:

$$
f_i = \frac{1}{N} \sum_{k=1}^{N} V(\hat{d}_k)
$$

where $V(\hat{d}_k) = 1$ if the ray is unoccluded, and $V(\hat{d}_k) = 0$ if blocked. This provides a Monte Carlo estimate of the visible fraction of the solar disk as seen from facet $i$.

---

## 3. BVH-Accelerated Raytracing

### Möller-Trumbore Ray-Triangle Intersection

For a ray $\mathbf{R}(t) = \mathbf{O} + t\hat{\mathbf{d}}$ and triangle with vertices $\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2$:

$$
\mathbf{e}_1 = \mathbf{v}_1 - \mathbf{v}_0, \qquad \mathbf{e}_2 = \mathbf{v}_2 - \mathbf{v}_0
$$

$$
\mathbf{h} = \hat{\mathbf{d}} \times \mathbf{e}_2, \qquad a = \mathbf{e}_1 \cdot \mathbf{h}
$$

If $|a| < \epsilon$, the ray is parallel to the triangle plane (no hit). Otherwise:

$$
f = 1/a, \qquad \mathbf{s} = \mathbf{O} - \mathbf{v}_0
$$

$$
u = f \cdot (\mathbf{s} \cdot \mathbf{h}), \qquad \mathbf{q} = \mathbf{s} \times \mathbf{e}_1
$$

$$
v = f \cdot (\hat{\mathbf{d}} \cdot \mathbf{q}), \qquad t = f \cdot (\mathbf{e}_2 \cdot \mathbf{q})
$$

The intersection is valid if $u \geq 0$, $v \geq 0$, $u + v \leq 1$, and $t > 0$.

> [!IMPORTANT]
> The epsilon value ($\epsilon = 10^{-10}$) is critical. Too large → misses grazing rays near shadow boundaries. Too small → floating-point cancellation causes ray leakage at triangle edges, fatal for PSR accuracy.

### BVH Construction (SAH)

The Bounding Volume Hierarchy uses the **Surface Area Heuristic (SAH)** for optimal spatial partitioning:

$$
C_{\text{SAH}} = C_{\text{trav}} + \frac{S_L}{S_P} \cdot N_L \cdot C_{\text{isect}} + \frac{S_R}{S_P} \cdot N_R \cdot C_{\text{isect}}
$$

| Symbol | Meaning | Default |
|--------|---------|---------|
| $S_L, S_R$ | Surface area of left/right child bounding box | — |
| $S_P$ | Surface area of parent bounding box | — |
| $N_L, N_R$ | Number of triangles in left/right child | — |
| $C_{\text{trav}}$ | Traversal cost (set to 1.0) | 1.0 |
| $C_{\text{isect}}$ | Intersection test cost (set to 1.0) | 1.0 |

Configuration: **16 SAH bins** per axis, **4 triangles maximum** per leaf node.

### Shadow Ray Query

For shadow testing, we use **early exit**: the traversal terminates as soon as ANY occlusion is found (no need for closest-hit). This provides significant speedup since most rays in crater interiors will be occluded quickly by nearby rim geometry.

Traversal is **stack-based** (iterative, no recursion) for Numba JIT compatibility.

---

## 4. Subsurface Heat Equation

### Governing PDE

The 1D heat equation in a semi-infinite regolith column with depth $z$ (positive downward):

$$
\rho(z) \cdot c_p(T) \cdot \frac{\partial T}{\partial t} = \frac{\partial}{\partial z}\left[k(T, z) \cdot \frac{\partial T}{\partial z}\right]
$$

### Boundary Conditions

**Upper boundary** ($z = 0$, regolith–vacuum interface):

$$
k(T) \cdot \left.\frac{\partial T}{\partial z}\right|_{z=0} = Q_{\text{solar}} + Q_{\text{IR}} - \varepsilon\sigma T_s^4
$$

This is a **nonlinear Robin boundary condition** due to the $T^4$ radiation term.

**Lower boundary** ($z = z_{\max}$, deep regolith):

$$
-k(T) \cdot \left.\frac{\partial T}{\partial z}\right|_{z=z_{\max}} = Q_{\text{geo}}
$$

A constant geothermal flux condition (Neumann type).

### Non-Uniform Grid

The vertical grid uses **geometric spacing** to concentrate resolution near the surface where thermal gradients are steepest:

$$
\Delta z_j = \Delta z_0 \cdot r^j, \qquad j = 0, 1, \ldots, N-1
$$

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Surface spacing | $\Delta z_0$ | 0.5 mm |
| Growth ratio | $r$ | 1.07 |
| Number of layers | $N$ | 100 |
| Maximum depth | $z_{\max}$ | $\sim$ 2.0 m |

The thermal skin depth for a 29.5-day lunar cycle is $\sim$ 0.3 m, so the 2 m grid depth is sufficient for thermal isolation.

---

## 5. Crank-Nicolson Discretization

### Semi-Discrete Form

Applying the Crank-Nicolson scheme ($\theta = 0.5$, second-order accurate in time and space) to the heat equation:

$$
\rho_j \cdot c_{p,j} \cdot \frac{T_j^{n+1} - T_j^n}{\Delta t} = \frac{1}{2}\left[\mathcal{L}(T^{n+1}) + \mathcal{L}(T^n)\right]
$$

where $\mathcal{L}$ is the spatial operator:

$$
\mathcal{L}(T)_j = \frac{1}{\overline{\Delta z}_j}\left[\frac{k_{j+1/2}}{\Delta z_j}(T_{j+1} - T_j) - \frac{k_{j-1/2}}{\Delta z_{j-1}}(T_j - T_{j-1})\right]
$$

### Interface Conductivities

At the half-grid points, we use the **harmonic mean** to correctly handle discontinuities in $k$:

$$
k_{j+1/2} = \frac{2 k_j k_{j+1}}{k_j + k_{j+1}}
$$

### Tridiagonal Coefficients — Interior Nodes ($j = 1, \ldots, N-1$)

Define auxiliary quantities:

$$
\alpha_j = \frac{k_{j-1/2}}{\Delta z_{j-1} \cdot \overline{\Delta z}_j}, \qquad \gamma_j = \frac{k_{j+1/2}}{\Delta z_j \cdot \overline{\Delta z}_j}, \qquad C_j = \frac{\rho_j \cdot c_{p,j}}{\Delta t}
$$

where $\overline{\Delta z}_j = (\Delta z_j + \Delta z_{j-1})/2$ is the averaged spacing.

The tridiagonal system $a_j T_{j-1}^{n+1} + b_j T_j^{n+1} + c_j T_{j+1}^{n+1} = d_j$ has coefficients:

$$
\boxed{
\begin{aligned}
a_j &= -\tfrac{1}{2}\alpha_j \\[4pt]
b_j &= C_j + \tfrac{1}{2}(\alpha_j + \gamma_j) \\[4pt]
c_j &= -\tfrac{1}{2}\gamma_j \\[4pt]
d_j &= \tfrac{1}{2}\alpha_j T_{j-1}^n + \left(C_j - \tfrac{1}{2}(\alpha_j + \gamma_j)\right)T_j^n + \tfrac{1}{2}\gamma_j T_{j+1}^n
\end{aligned}
}
$$

> [!TIP]
> **Diagonal dominance is guaranteed**: $|b_j| = C_j + \frac{1}{2}(\alpha_j + \gamma_j) > \frac{1}{2}\alpha_j + \frac{1}{2}\gamma_j = |a_j| + |c_j|$, since $C_j = \rho c_p / \Delta t > 0$. This ensures stability and uniqueness of the Thomas algorithm (Section 7).

### Surface Node ($j = 0$) — with Newton-linearized Radiation

The surface half-cell integrates the energy balance:

$$
\tilde{C}_0 = \frac{\rho_0 \cdot c_{p,0} \cdot \Delta z_0/2}{\Delta t}
$$

$$
\begin{aligned}
a_0 &= 0 \\[4pt]
b_0 &= \tilde{C}_0 + 4\varepsilon\sigma\tilde{T}_0^3 + \frac{k_{1/2}}{\Delta z_0} \\[4pt]
c_0 &= -\frac{k_{1/2}}{\Delta z_0} \\[4pt]
d_0 &= Q_{\text{in}} + 3\varepsilon\sigma\tilde{T}_0^4 + \tilde{C}_0 \cdot T_0^n
\end{aligned}
$$

where $\tilde{T}_0$ is the current Newton guess and $Q_{\text{in}} = Q_{\text{solar}} + Q_{\text{IR}} + Q_{\text{geo}}$.

### Deep Boundary Node ($j = N$) — Constant Flux

$$
\beta_N = \frac{k_{N-1/2}}{\Delta z_{N-1}}, \qquad \tilde{C}_N = \frac{\rho_N \cdot c_{p,N} \cdot \Delta z_{N-1}/2}{\Delta t}
$$

$$
\begin{aligned}
a_N &= -\tfrac{1}{2}\beta_N \\[4pt]
b_N &= \tilde{C}_N + \tfrac{1}{2}\beta_N \\[4pt]
c_N &= 0 \\[4pt]
d_N &= \tfrac{1}{2}\beta_N T_{N-1}^n + (\tilde{C}_N - \tfrac{1}{2}\beta_N)T_N^n + Q_{\text{geo}}
\end{aligned}
$$

---

## 6. Newton-Raphson Surface Boundary Condition

The $\varepsilon\sigma T_s^4$ term in the surface energy balance makes the boundary condition nonlinear. We linearize it using a Taylor expansion around the current guess $\tilde{T}$:

$$
\varepsilon\sigma T^4 \approx \varepsilon\sigma\tilde{T}^4 + 4\varepsilon\sigma\tilde{T}^3(T - \tilde{T}) = -3\varepsilon\sigma\tilde{T}^4 + 4\varepsilon\sigma\tilde{T}^3 \cdot T
$$

This rearrangement places the $4\varepsilon\sigma\tilde{T}^3 \cdot T$ term on the left-hand side (into $b_0$) and the remaining $-3\varepsilon\sigma\tilde{T}^4$ contributes to $d_0$ as $+3\varepsilon\sigma\tilde{T}^4$.

### Iteration Procedure

1. Initialize $\tilde{T} = T^n$ (previous time-step temperature)
2. Build tridiagonal coefficients using $\tilde{T}$
3. Solve tridiagonal system → $T^*$
4. Apply under-relaxation: $\tilde{T} \leftarrow \tilde{T} + \omega(T^* - \tilde{T})$, where $\omega \in (0, 1]$
5. If $|\tilde{T}_0^{\text{new}} - \tilde{T}_0^{\text{old}}| < \epsilon_{\text{Newton}}$, converge; else repeat from step 2

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 20 | Maximum Newton iterations per time step |
| `tolerance_K` | $10^{-4}$ K | Surface temperature convergence criterion |
| `relaxation` | 1.0 | Under-relaxation factor ($\omega$) |

> [!NOTE]
> Convergence is typically achieved in 2–4 iterations for realistic temperature ranges. The relaxation factor can be reduced below 1.0 if convergence issues arise at extreme temperature gradients.

---

## 7. Thomas Algorithm

The Thomas algorithm (TDMA) solves the tridiagonal system $\mathbf{A}\mathbf{x} = \mathbf{d}$ in $O(N)$ time and $O(1)$ extra space.

For the matrix:
$$
\mathbf{A} = \begin{pmatrix}
b_0 & c_0 & & & \\
a_1 & b_1 & c_1 & & \\
& a_2 & b_2 & c_2 & \\
& & \ddots & \ddots & \ddots \\
& & & a_N & b_N
\end{pmatrix}
$$

**Forward elimination** ($j = 1, \ldots, N$):
$$
w = \frac{a_j}{b_{j-1}}, \qquad b_j \leftarrow b_j - w \cdot c_{j-1}, \qquad d_j \leftarrow d_j - w \cdot d_{j-1}
$$

**Back substitution** ($j = N-1, \ldots, 0$):
$$
x_N = \frac{d_N}{b_N}, \qquad x_j = \frac{d_j - c_j \cdot x_{j+1}}{b_j}
$$

Stability of the Thomas algorithm is guaranteed by diagonal dominance of our Crank-Nicolson matrix (proven in Section 5).

---

## 8. Regolith Thermophysical Properties

All properties follow the **Hayne et al. (2017)** model, validated against LRO Diviner observations.

### Thermal Conductivity

$$
k(T, z) = k_c(z) + k_r(z) \cdot T^3
$$

The first term represents **phonon conduction** through grain contacts; the second represents **radiative heat transfer** through pore spaces.

| Layer | $k_c$ [W/m/K] | $k_r$ [W/m/K⁴] | Depth |
|-------|----------------|------------------|-------|
| Surface | $7.4 \times 10^{-4}$ | $2.0 \times 10^{-11}$ | $z < 0.02$ m |
| Deep | $3.4 \times 10^{-3}$ | $1.0 \times 10^{-11}$ | $z \geq 0.02$ m |

### Specific Heat Capacity

$$
c_p(T) = c_0 + c_1 T^{1/2} + c_2 T + c_3 T^2 + c_4 T^3
$$

| Coefficient | Value | Units |
|-------------|-------|-------|
| $c_0$ | −3.6125 | J/kg/K |
| $c_1$ | 2.7431 | J/kg/K^{3/2} |
| $c_2$ | $2.3616 \times 10^{-3}$ | J/kg/K² |
| $c_3$ | $-1.2340 \times 10^{-5}$ | J/kg/K³ |
| $c_4$ | $8.9093 \times 10^{-9}$ | J/kg/K⁴ |

Clamped to a minimum of 8.0 J/kg/K at low temperatures to prevent singularity as $T \to 0$.

### Bulk Density

$$
\rho(z) = \rho_{\text{deep}} - (\rho_{\text{deep}} - \rho_{\text{surf}}) \cdot \exp(-z / H)
$$

| Parameter | Value | Source |
|-----------|-------|--------|
| $\rho_{\text{surf}}$ | 1100 kg/m³ | Hayne et al. (2017) |
| $\rho_{\text{deep}}$ | 1800 kg/m³ | Hayne et al. (2017) |
| $H$ | 0.06 m | e-folding depth |

---

## 9. Coordinate Systems

### Selenographic Coordinates

The simulation target is specified in **selenographic coordinates**:
- Latitude: $-89.54°$ (Shackleton Crater, Zuber et al., 2012)
- Longitude: $129.78°$

### Local Cartesian Frame

For the DEM, we use a local tangent plane with:
- **x**: East
- **y**: North
- **z**: Radial (up)

Origin at the mean elevation of the DEM grid. The **polar stereographic projection** is used for coordinate transforms between selenographic and local Cartesian coordinates.

### Sun Direction Vector

The sun direction $\hat{s}$ is computed via the **Skyfield** ephemeris library using JPL DE421 planetary ephemerides. The computation chain:

1. Skyfield computes geocentric sun position
2. Transform to selenocentric coordinates
3. Apply polar stereographic projection to get $\hat{s}$ in the local DEM frame

---

## 10. References

1. Bandfield, J.L., et al. (2015). "Lunar surface roughness derived from LRO Diviner Radiometer observations." *Icarus*, 248, 357-372.
2. Crank, J. & Nicolson, P. (1947). "A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type." *Proc. Cambridge Phil. Soc.*, 43, 50-67.
3. Hayne, P.O., et al. (2017). "Global regolith thermophysical properties of the Moon from the Diviner Lunar Radiometer Experiment." *JGR Planets*, 122, 2371-2400.
4. Kopp, G. & Lean, J.L. (2011). "A new, lower value of total solar irradiance." *Geophys. Res. Lett.*, 38, L01706.
5. Langseth, M.G., Keihm, S.J. & Peters, K. (1976). "Revised lunar heat-flow values." *Proc. 7th Lunar Science Conf.*, 3143-3171.
6. Mazarico, E., et al. (2011). "Illumination conditions of the lunar polar regions using LOLA topography." *Icarus*, 211, 1066-1081.
7. Möller, T. & Trumbore, B. (1997). "Fast, minimum storage ray-triangle intersection." *J. Graphics Tools*, 2(1), 21-28.
8. Paige, D.A., et al. (2010). "Diviner Lunar Radiometer observations of cold traps in the Moon's south polar region." *Science*, 330, 479-482.
9. Spencer, J.R., Lebofsky, L.A. & Sykes, M.V. (1989). "Systematic biases in radiometric diameter determinations." *Icarus*, 78, 337-354.
10. Vasavada, A.R., et al. (2012). "Lunar equatorial surface temperatures and regolith properties from the Diviner Lunar Radiometer Experiment." *JGR Planets*, 117, E00H18.
11. Zuber, M.T., et al. (2012). "Constraints on the volatile distribution within Shackleton crater at the Moon's south pole." *Nature*, 486, 378-381.
12. Wald, I. (2007). "On fast construction of SAH-based bounding volume hierarchies." *Proc. IEEE Symp. Interactive Ray Tracing*, 33-40.

---

> Implementation: [crank_nicolson.py](file:///c:/Users/mehme/Desktop/ArtemisThermalBase/thermal_solver/crank_nicolson.py) · [raytracer.py](file:///c:/Users/mehme/Desktop/ArtemisThermalBase/core_engine/raytracer.py) · [solar_disk.py](file:///c:/Users/mehme/Desktop/ArtemisThermalBase/core_engine/solar_disk.py) · [regolith_properties.py](file:///c:/Users/mehme/Desktop/ArtemisThermalBase/thermal_solver/regolith_properties.py)
