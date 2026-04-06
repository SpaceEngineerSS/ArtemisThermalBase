"""ArtemisThermalBase — Mission Control Dashboard.

Interactive Streamlit GUI for lunar thermal simulation analysis,
3D terrain visualization, ice stability mapping, and rover power
simulation.

Launch:
    streamlit run gui/dashboard.py

Architecture:
    - LOCAL-FIRST: Reads output/*.npy + metadata.json (no database)
    - 3D Viewer: PyVista mesh rendered via stpyvista
    - Charts: Plotly for interactive time series + heatmaps
    - Rover Sim: Click-to-inspect on 2D scatter map

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:
    raise ImportError(
        "Streamlit is required for the GUI. Install with:\n"
        "  pip install streamlit stpyvista pyarrow"
    )

import plotly.graph_objects as go
import plotly.express as px


# ===================================================================
# PAGE CONFIG
# ===================================================================

st.set_page_config(
    page_title="ArtemisThermalBase — Mission Control",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .main-header p {
        font-size: 0.9rem;
        opacity: 0.85;
        margin-top: 0.3rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: white;
    }
    .metric-card h3 {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ffd700, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117, #161b22);
    }
    div[data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)


# ===================================================================
# DATA LOADING (LOCAL-FIRST)
# ===================================================================


@st.cache_data(ttl=30)
def load_simulation_data(output_dir: str) -> dict | None:
    """Load simulation results from local files.

    Reads:
    - surface_temps.npy
    - illumination.npy
    - face_centroids.npy
    - face_areas.npy
    - dem_elevation.npy
    - metadata.json
    - probe_temps.json (if exists)

    Parameters
    ----------
    output_dir : str
        Path to simulation output directory.

    Returns
    -------
    dict or None
        Dictionary of loaded arrays and metadata, or None if not found.
    """
    out = Path(output_dir)
    if not out.exists():
        return None

    data = {}
    files = {
        "surface_temps": "surface_temps.npy",
        "illumination": "illumination.npy",
        "face_centroids": "face_centroids.npy",
        "face_areas": "face_areas.npy",
        "dem_elevation": "dem_elevation.npy",
    }

    for key, fname in files.items():
        fpath = out / fname
        if fpath.exists():
            data[key] = np.load(fpath)

    # Metadata
    meta_path = out / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)

    # Probe temperatures
    probe_path = out / "probe_temps.json"
    if probe_path.exists():
        with open(probe_path) as f:
            data["probe_temps"] = json.load(f)

    # Sun elevations
    sun_path = out / "sun_elevations.npy"
    if sun_path.exists():
        data["sun_elevations"] = np.load(sun_path)

    if "surface_temps" not in data:
        return None

    return data


def downsample_for_3d(centroids: np.ndarray, values: np.ndarray,
                       max_points: int = 80000) -> tuple:
    """Downsample mesh data for 3D visualization.

    Parameters
    ----------
    centroids : np.ndarray
        Face centroids (N, 3).
    values : np.ndarray
        Per-face values (N,).
    max_points : int
        Maximum number of points for 3D rendering.

    Returns
    -------
    tuple
        (downsampled_centroids, downsampled_values, indices)
    """
    n = len(values)
    if n <= max_points:
        return centroids, values, np.arange(n)

    indices = np.random.default_rng(42).choice(n, max_points, replace=False)
    indices.sort()
    return centroids[indices], values[indices], indices


# ===================================================================
# HEADER
# ===================================================================

st.markdown("""
<div class="main-header">
    <h1>🌙 ArtemisThermalBase — Mission Control</h1>
    <p>High-Fidelity Lunar South Pole Thermal Simulation Dashboard</p>
</div>
""", unsafe_allow_html=True)


# ===================================================================
# SIDEBAR
# ===================================================================

with st.sidebar:
    st.image(
        "https://img.shields.io/badge/v0.3.0-Scientific%20Explorer-blueviolet"
        "?style=for-the-badge",
        use_container_width=True,
    )
    st.markdown("---")

    st.markdown("### ⚙️ Data Source")
    output_dir = st.text_input(
        "Output Directory",
        value="output",
        help="Path to simulation output directory with .npy files",
    )

    st.markdown("---")
    st.markdown("### 🎨 Visualization")

    colormap = st.selectbox(
        "Temperature Colormap",
        ["inferno", "magma", "plasma", "turbo", "hot", "jet"],
        index=0,
    )

    max_3d_points = st.slider(
        "3D Point Limit",
        min_value=10000,
        max_value=200000,
        value=80000,
        step=10000,
        help="Maximum points for 3D rendering (lower = faster)",
    )

    st.markdown("---")
    st.markdown("### 🔬 Physics Models")

    show_hapke = st.checkbox("Show Hapke A_DH curve", value=True)
    show_roughness = st.checkbox("Show Roughness ε_eff", value=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; opacity:0.5; font-size:0.75rem;'>"
        "Developed by Mehmet Gümüş<br>"
        "<a href='https://github.com/SpaceEngineerSS' "
        "style='color:#58a6ff;'>github.com/SpaceEngineerSS</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ===================================================================
# LOAD DATA
# ===================================================================

data = load_simulation_data(output_dir)

if data is None:
    st.warning(
        f"⚠️ No simulation data found in `{output_dir}/`. "
        "Run a simulation first:\n\n"
        "```bash\npython main.py --dem data/sample_lola_dem.tif --duration 6\n```"
    )
    st.stop()


# ===================================================================
# METRICS ROW
# ===================================================================

surface_T = data["surface_temps"]
centroids = data.get("face_centroids", np.array([]))
face_areas = data.get("face_areas", np.array([]))
metadata = data.get("metadata", {})
illum = data.get("illumination", np.array([]))

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f'<div class="metric-card"><h3>Min Temp</h3>'
        f'<div class="value">{surface_T.min():.1f} K</div></div>',
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f'<div class="metric-card"><h3>Max Temp</h3>'
        f'<div class="value">{surface_T.max():.1f} K</div></div>',
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f'<div class="metric-card"><h3>Mean Temp</h3>'
        f'<div class="value">{surface_T.mean():.1f} K</div></div>',
        unsafe_allow_html=True,
    )

with col4:
    n_faces = metadata.get("num_faces", len(surface_T))
    st.markdown(
        f'<div class="metric-card"><h3>Mesh Faces</h3>'
        f'<div class="value">{n_faces:,}</div></div>',
        unsafe_allow_html=True,
    )

with col5:
    wall_t = metadata.get("wall_time_s", 0)
    st.markdown(
        f'<div class="metric-card"><h3>Wall Time</h3>'
        f'<div class="value">{wall_t:.0f}s</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")


# ===================================================================
# TABS
# ===================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🌡️ Thermal Overview",
    "🧊 Ice Stability",
    "🤖 Rover Simulator",
    "📊 Physics Models",
])


# ===================================================================
# TAB 1: THERMAL OVERVIEW
# ===================================================================

with tab1:
    col_3d, col_map = st.columns([1, 1])

    with col_3d:
        st.markdown("#### 3D Terrain Temperature")

        if len(centroids) > 0:
            c_ds, t_ds, idx_ds = downsample_for_3d(
                centroids, surface_T, max_3d_points
            )

            try:
                import pyvista as pv
                from stpyvista import stpyvista

                cloud = pv.PolyData(c_ds)
                cloud["Temperature [K]"] = t_ds

                plotter = pv.Plotter(window_size=[600, 500])
                plotter.set_background("#0d1117")
                plotter.add_mesh(
                    cloud,
                    scalars="Temperature [K]",
                    cmap=colormap,
                    point_size=3,
                    render_points_as_spheres=True,
                    clim=[surface_T.min(), surface_T.max()],
                )
                plotter.view_isometric()

                stpyvista(plotter, key="thermal_3d")

            except ImportError:
                st.info(
                    "Install `stpyvista` for 3D rendering:\n"
                    "`pip install stpyvista`"
                )

                # Fallback: Plotly 3D scatter
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=c_ds[:, 0],
                    y=c_ds[:, 1],
                    z=c_ds[:, 2],
                    mode="markers",
                    marker=dict(
                        size=1.5,
                        color=t_ds,
                        colorscale=colormap.capitalize(),
                        colorbar=dict(title="T [K]"),
                        cmin=float(surface_T.min()),
                        cmax=float(surface_T.max()),
                    ),
                )])
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title="X [m]",
                        yaxis_title="Y [m]",
                        zaxis_title="Z [m]",
                        bgcolor="#0d1117",
                    ),
                    paper_bgcolor="#0d1117",
                    font_color="white",
                    height=500,
                    margin=dict(l=0, r=0, t=0, b=0),
                )
                st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("No centroid data available for 3D view.")

    with col_map:
        st.markdown("#### 2D Temperature Map")

        if len(centroids) > 0:
            fig_heat = go.Figure(data=[go.Scatter(
                x=centroids[::max(1, len(centroids) // max_3d_points), 0],
                y=centroids[::max(1, len(centroids) // max_3d_points), 1],
                mode="markers",
                marker=dict(
                    size=3,
                    color=surface_T[::max(1, len(surface_T) // max_3d_points)],
                    colorscale=colormap.capitalize(),
                    colorbar=dict(title="T [K]"),
                    cmin=float(surface_T.min()),
                    cmax=float(surface_T.max()),
                ),
            )])
            fig_heat.update_layout(
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                font_color="white",
                height=500,
                margin=dict(l=60, r=20, t=20, b=60),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # Probe time series
    probe_data = data.get("probe_temps", {})
    if probe_data:
        st.markdown("#### 📈 Probe Temperature Time Series")

        fig_ts = go.Figure()
        colors = ["#ffd700", "#ff6b6b", "#4ecdc4", "#45b7d1"]
        for i, (name, temps) in enumerate(probe_data.items()):
            fig_ts.add_trace(go.Scatter(
                y=temps,
                mode="lines",
                name=name,
                line=dict(
                    color=colors[i % len(colors)],
                    width=2,
                ),
            ))
        fig_ts.update_layout(
            xaxis_title="Time Step",
            yaxis_title="Temperature [K]",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font_color="white",
            legend=dict(
                bgcolor="rgba(0,0,0,0.5)",
                font_color="white",
            ),
            height=350,
            margin=dict(l=60, r=20, t=30, b=60),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # Temperature histogram
    st.markdown("#### 📊 Temperature Distribution")
    fig_hist = go.Figure(data=[go.Histogram(
        x=surface_T,
        nbinsx=80,
        marker_color="#ffd700",
        opacity=0.8,
    )])
    fig_hist.update_layout(
        xaxis_title="Surface Temperature [K]",
        yaxis_title="Face Count",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="white",
        height=300,
        margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ===================================================================
# TAB 2: ICE STABILITY
# ===================================================================

with tab2:
    st.markdown("#### 🧊 Water Ice Cold Trap Stability Map")
    st.markdown(
        "Classification based on Powell & Rubanenko (2020): "
        "**Stable** (T < 110 K, > 1 Gyr), "
        "**Marginal** (110–115 K, ~100 Myr), "
        "**Unstable** (T > 115 K)."
    )

    if len(centroids) > 0:
        # Classify stability
        stability = np.zeros(len(surface_T), dtype=np.int32)
        stability[surface_T < 110.0] = 0   # Stable
        stability[(surface_T >= 110.0) & (surface_T < 115.0)] = 1  # Marginal
        stability[surface_T >= 115.0] = 2  # Unstable

        # Color mapping
        stability_colors = np.where(
            stability == 0, "#00bfff",
            np.where(stability == 1, "#ffd700", "#1a1a2e")
        )

        n_stable = np.sum(stability == 0)
        n_marginal = np.sum(stability == 1)
        n_unstable = np.sum(stability == 2)
        total = len(stability)

        # Stats
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("🔵 Stable (< 110 K)", f"{n_stable:,} ({100*n_stable/total:.1f}%)")
        with col_s2:
            st.metric("🟡 Marginal (110–115 K)", f"{n_marginal:,} ({100*n_marginal/total:.1f}%)")
        with col_s3:
            st.metric("⚫ Unstable (> 115 K)", f"{n_unstable:,} ({100*n_unstable/total:.1f}%)")

        # Ice stability scatter
        step = max(1, len(centroids) // max_3d_points)
        fig_ice = go.Figure(data=[go.Scatter(
            x=centroids[::step, 0],
            y=centroids[::step, 1],
            mode="markers",
            marker=dict(
                size=3,
                color=surface_T[::step],
                colorscale=[
                    [0.0, "#00bfff"],
                    [0.25, "#00bfff"],
                    [0.4, "#ffd700"],
                    [0.45, "#ffd700"],
                    [0.5, "#1a1a2e"],
                    [1.0, "#1a1a2e"],
                ],
                cmin=80.0,
                cmax=200.0,
                colorbar=dict(
                    title="T [K]",
                    tickvals=[90, 110, 115, 150, 200],
                    ticktext=["90 (Stable)", "110", "115", "150", "200"],
                ),
            ),
        )])
        fig_ice.update_layout(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font_color="white",
            height=600,
            margin=dict(l=60, r=20, t=20, b=60),
        )
        st.plotly_chart(fig_ice, use_container_width=True)

        # Area calculation
        if len(face_areas) > 0:
            stable_area_km2 = face_areas[stability == 0].sum() / 1e6
            st.info(
                f"**Cold trap area**: {stable_area_km2:.2f} km² "
                f"(faces with T < 110 K)"
            )


# ===================================================================
# TAB 3: ROVER SIMULATOR
# ===================================================================

with tab3:
    st.markdown("#### 🤖 Rover Power & Thermal Analysis")
    st.markdown(
        "Select a landing site on the map. The panel shows "
        "estimated sink temperature, sky view factor, and solar "
        "power availability at that location."
    )

    if len(centroids) > 0:
        col_rover_map, col_rover_info = st.columns([2, 1])

        with col_rover_map:
            step = max(1, len(centroids) // max_3d_points)
            x_coords = centroids[::step, 0]
            y_coords = centroids[::step, 1]
            z_coords = centroids[::step, 2]
            temps = surface_T[::step]

            fig_rover = go.Figure(data=[go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=3,
                    color=temps,
                    colorscale="Inferno",
                    colorbar=dict(title="T [K]"),
                ),
                customdata=np.column_stack([z_coords, temps]),
                hovertemplate=(
                    "X: %{x:.0f} m<br>"
                    "Y: %{y:.0f} m<br>"
                    "Elev: %{customdata[0]:.1f} m<br>"
                    "T: %{customdata[1]:.1f} K"
                    "<extra></extra>"
                ),
            )])
            fig_rover.update_layout(
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                font_color="white",
                height=500,
                margin=dict(l=60, r=20, t=20, b=60),
                dragmode="pan",
            )
            st.plotly_chart(fig_rover, use_container_width=True)

        with col_rover_info:
            st.markdown("##### 📍 Landing Site Analysis")

            # Manual coordinate input
            rover_x = st.number_input(
                "X coordinate [m]",
                value=0.0,
                step=100.0,
            )
            rover_y = st.number_input(
                "Y coordinate [m]",
                value=0.0,
                step=100.0,
            )

            if st.button("🔍 Analyze Site", type="primary"):
                # Find nearest face
                xy = centroids[:, :2]
                dists = np.sqrt(
                    (xy[:, 0] - rover_x) ** 2 +
                    (xy[:, 1] - rover_y) ** 2
                )
                nearest_idx = int(np.argmin(dists))

                T_site = float(surface_T[nearest_idx])
                z_site = float(centroids[nearest_idx, 2])

                # Estimate sky view factor from surface normal
                # (simplified: VF_sky ≈ (1 + cos(slope)) / 2)
                from core_engine.roughness import compute_face_slopes
                try:
                    face_normals_data = np.load(
                        Path(output_dir) / "face_normals.npy"
                    )
                    slopes = compute_face_slopes(
                        face_normals_data[nearest_idx:nearest_idx+1]
                    )
                    slope_deg = slopes[0]
                except Exception:
                    slope_deg = 0.0

                vf_sky = (1.0 + math.cos(math.radians(slope_deg))) / 2.0

                # Radiative sink temperature
                sigma = 5.670374419e-8
                T_sink = (T_site ** 4 * 0.95 * sigma / sigma) ** 0.25

                # Estimated solar power (1361 W/m² × VF_sky × panel_eff)
                panel_eff = 0.28  # typical space-grade solar cell
                P_solar = 1361.0 * vf_sky * panel_eff

                # Ice stability
                if T_site < 110:
                    ice_status = "🔵 STABLE (> 1 Gyr)"
                elif T_site < 115:
                    ice_status = "🟡 MARGINAL (~100 Myr)"
                else:
                    ice_status = "⚫ UNSTABLE"

                st.markdown("---")
                st.markdown(f"**Nearest Face**: #{nearest_idx}")
                st.markdown(f"**Elevation**: {z_site:.1f} m")
                st.markdown(f"**Slope**: {slope_deg:.1f}°")
                st.markdown("---")

                st.metric("🌡️ Surface Temperature", f"{T_site:.1f} K")
                st.metric("🌌 Sky View Factor", f"{vf_sky:.3f}")
                st.metric("☀️ Solar Power", f"{P_solar:.1f} W/m²")
                st.metric("🧊 Ice Status", ice_status)

                st.markdown("---")
                st.markdown(
                    f"**T_sink** = {T_sink:.1f} K"
                    f"\n\n**Panel output** = {P_solar:.1f} W/m² "
                    f"(η={panel_eff*100:.0f}%)"
                )


# ===================================================================
# TAB 4: PHYSICS MODELS
# ===================================================================

with tab4:
    st.markdown("#### 📊 Physics Model Diagnostics")

    col_hapke, col_rough = st.columns(2)

    with col_hapke:
        if show_hapke:
            st.markdown("##### Hapke Directional-Hemispherical Albedo")
            st.markdown(
                "A_DH(θ_i) replaces constant Bond albedo. "
                "Shows how reflectance varies with solar incidence angle."
            )

            try:
                from core_engine.reflectance import directional_hemispherical_albedo

                angles = np.linspace(0, 89, 45)
                adh_values = []
                for a in angles:
                    cos_i = math.cos(math.radians(a))
                    adh = directional_hemispherical_albedo(cos_i, w=0.23)
                    adh_values.append(adh)

                fig_hapke = go.Figure()
                fig_hapke.add_trace(go.Scatter(
                    x=angles,
                    y=adh_values,
                    mode="lines+markers",
                    name="Hapke A_DH",
                    line=dict(color="#ffd700", width=2),
                    marker=dict(size=4),
                ))
                fig_hapke.add_hline(
                    y=0.12,
                    line_dash="dash",
                    line_color="#ff6b6b",
                    annotation_text="Bond Albedo (Lambertian)",
                )
                fig_hapke.update_layout(
                    xaxis_title="Incidence Angle θ_i [°]",
                    yaxis_title="Directional-Hemispherical Albedo",
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#161b22",
                    font_color="white",
                    height=400,
                    margin=dict(l=60, r=20, t=30, b=60),
                )
                st.plotly_chart(fig_hapke, use_container_width=True)

            except ImportError:
                st.warning("Hapke module not available.")

    with col_rough:
        if show_roughness:
            st.markdown("##### Bandfield Effective Emissivity")
            st.markdown(
                "ε_eff(θ̄) from sub-pixel cavity self-heating. "
                "Higher roughness → higher effective emissivity."
            )

            try:
                from core_engine.roughness import effective_emissivity

                slopes = np.linspace(0, 60, 30)
                eps_values = [effective_emissivity(0.95, s) for s in slopes]

                fig_rough = go.Figure()
                fig_rough.add_trace(go.Scatter(
                    x=slopes,
                    y=eps_values,
                    mode="lines+markers",
                    name="ε_eff",
                    line=dict(color="#4ecdc4", width=2),
                    marker=dict(size=4),
                ))
                fig_rough.add_hline(
                    y=0.95,
                    line_dash="dash",
                    line_color="#ff6b6b",
                    annotation_text="ε₀ (smooth)",
                )
                fig_rough.update_layout(
                    xaxis_title="RMS Slope Angle θ̄ [°]",
                    yaxis_title="Effective Emissivity ε_eff",
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#161b22",
                    font_color="white",
                    height=400,
                    yaxis_range=[0.94, 1.001],
                    margin=dict(l=60, r=20, t=30, b=60),
                )
                st.plotly_chart(fig_rough, use_container_width=True)

            except ImportError:
                st.warning("Roughness module not available.")

    # Phase function plot
    st.markdown("##### Henyey-Greenstein Phase Function p(g)")
    try:
        from core_engine.reflectance import henyey_greenstein_double

        g_angles = np.linspace(0, 180, 181)
        p_values = [
            henyey_greenstein_double(math.cos(math.radians(g)), 0.21, 0.70)
            for g in g_angles
        ]

        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(
            x=g_angles,
            y=p_values,
            mode="lines",
            name="p(g)",
            line=dict(color="#ff6b6b", width=2),
        ))
        fig_phase.add_vline(
            x=0,
            line_dash="dot",
            line_color="#ffd700",
            annotation_text="Opposition",
        )
        fig_phase.update_layout(
            xaxis_title="Phase Angle g [°]",
            yaxis_title="Phase Function p(g)",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font_color="white",
            height=350,
            margin=dict(l=60, r=20, t=30, b=60),
        )
        st.plotly_chart(fig_phase, use_container_width=True)

    except ImportError:
        st.warning("Reflectance module not available.")


# ===================================================================
# FOOTER
# ===================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; opacity:0.5; font-size:0.8rem;'>"
    "ArtemisThermalBase v0.3.0 — "
    "Developed by <a href='https://github.com/SpaceEngineerSS' "
    "style='color:#58a6ff;'>Mehmet Gümüş</a>"
    " | MIT License"
    "</div>",
    unsafe_allow_html=True,
)
