"""Make every image used by the documentation.

    python tools/make_figures.py diagrams
    python tools/make_figures.py run output/shackleton      # report figures of a run
    python tools/make_figures.py bench bench.json           # performance chart

Diagrams are written as SVG by hand (no plotting library) so they stay crisp
and small, and they carry their own light/dark colours from the shared
palette. Everything lands in docs/img/.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from artemis_thermal import style  # noqa: E402

IMG = ROOT / "docs" / "img"


def _svg(width: int, height: int, body: str, title: str) -> str:
    light, dark = style.LIGHT, style.DARK

    def rules(p: style.Palette) -> str:
        return (
            f".bg{{fill:{p.paper}}}.ink{{fill:{p.ink}}}.ink2{{fill:{p.ink_2}}}"
            f".ink3{{fill:{p.ink_3}}}.acc{{fill:{p.accent_ink}}}"
            f".s-ink{{stroke:{p.ink}}}.s-ink2{{stroke:{p.ink_2}}}.s-rule{{stroke:{p.rule_strong}}}"
            f".s-faint{{stroke:{p.rule}}}.s-acc{{stroke:{p.accent}}}.s-cold{{stroke:{p.cold}}}"
            f".f-sheet{{fill:{p.sheet}}}.f-acc{{fill:{p.accent}}}.f-cold{{fill:{p.cold}}}"
            f".f-soft{{fill:{p.cold_soft}}}.f-rule{{fill:{p.rule}}}"
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{title}">\n'
        f"<title>{title}</title>\n<style>"
        "text{font-family:'IBM Plex Mono','Plex Mono',ui-monospace,Consolas,monospace;"
        "font-size:12px}.sans{font-family:'IBM Plex Sans','Plex Sans',system-ui,sans-serif}"
        ".small{font-size:10.5px}.big{font-size:14px}"
        "line,path,polyline,circle,rect,ellipse{fill:none;stroke-width:1;"
        "vector-effect:non-scaling-stroke}"
        f"{rules(light)}"
        f"@media (prefers-color-scheme:dark){{{rules(dark)}}}"
        "</style>\n"
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" style="stroke:none"/>\n'
        f"{body}\n</svg>\n"
    )


def _arrow(x1: float, y1: float, x2: float, y2: float, cls: str = "s-ink", head: float = 6.0,
           dash: bool = False) -> str:
    angle = math.atan2(y2 - y1, x2 - x1)
    a1 = angle + math.radians(155)
    a2 = angle - math.radians(155)
    hx1, hy1 = x2 + head * math.cos(a1), y2 + head * math.sin(a1)
    hx2, hy2 = x2 + head * math.cos(a2), y2 + head * math.sin(a2)
    style_dash = ' stroke-dasharray="4 3"' if dash else ""
    return (f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
            f'{style_dash}/>'
            f'<polyline class="{cls}" points="{hx1:.1f},{hy1:.1f} {x2:.1f},{y2:.1f} '
            f'{hx2:.1f},{hy2:.1f}"/>')


def _text(x: float, y: float, text: str, cls: str = "ink", anchor: str = "start",
          extra: str = "") -> str:
    return (f'<text class="{cls} {extra}" x="{x:.1f}" y="{y:.1f}" '
            f'text-anchor="{anchor}">{text}</text>')


# ---------------------------------------------------------------------------
# 1. Pipeline
# ---------------------------------------------------------------------------


def pipeline() -> str:
    """Data flow of one run, drawn as a numbered process sheet."""
    w, h = 760, 360
    parts = []

    def box(x, y, bw, bh, num, title, sub, accent=False):
        cls = "s-acc" if accent else "s-ink2"
        parts.append(f'<rect class="{cls} f-sheet" x="{x}" y="{y}" width="{bw}" height="{bh}"/>')
        parts.append(_text(x + 10, y + 18, num, "acc", extra="small"))
        parts.append(_text(x + 10, y + 36, title, "ink", extra="sans big"))
        for i, line in enumerate(sub):
            parts.append(_text(x + 10, y + 54 + 14 * i, line, "ink2", extra="small"))

    box(20, 30, 190, 92, "01", "Terrain", ["LOLA GeoTIFF or", "synthetic bowl"])
    box(20, 210, 190, 92, "02", "Sun", ["SPICE DE440, Skyfield", "or analytical model"])
    box(285, 30, 190, 92, "03", "Mesh on the sphere", ["2 triangles per cell,", "BVH (binned SAH)"])
    box(285, 210, 190, 92, "04", "Sun on the mesh", ["body frame → grid axes", "disk samples s_k"])
    box(550, 120, 190, 92, "05", "Shadow rays",
        ["visible disk fraction,", "mean of max(0, n·s)·V"], True)
    box(550, 250, 190, 92, "06", "Regolith columns", ["1-D heat, 101 nodes,", "one parallel call"])
    parts.append(_arrow(210, 76, 283, 76))
    parts.append(_arrow(210, 256, 283, 256))
    parts.append(_arrow(475, 76, 548, 150))
    parts.append(_arrow(475, 256, 548, 182))
    parts.append(_arrow(645, 212, 645, 248))
    parts.append('<path class="s-rule" d="M550 296 H520 V330 H245" stroke-dasharray="4 3"/>')
    parts.append(_arrow(260, 330, 240, 330, "s-rule"))
    parts.append(_text(232, 334, "run folder → figures, app", "ink2", "end", "small"))
    parts.append(_text(20, 20, "ONE TIME STEP", "acc", extra="small"))
    parts.append(_text(740, 20, "steps 04–06 repeat every dt", "ink3", "end", "small"))
    return _svg(w, h, "\n".join(parts), "Pipeline of one simulation run")


# ---------------------------------------------------------------------------
# 2. Frames
# ---------------------------------------------------------------------------


def frames() -> str:
    """Top-down view of the south pole: map grid axes versus local east/north."""
    w, h = 760, 380
    cx, cy, r = 210, 195, 150
    parts = [_text(20, 24, "LOOKING DOWN ON THE SOUTH POLE", "acc", extra="small")]
    for rr in (r, r * 0.66, r * 0.33):
        parts.append(f'<circle class="s-faint" cx="{cx}" cy="{cy}" r="{rr:.1f}"/>')
    labels = {0: "0°", 90: "90°E", 180: "180°", 270: "90°W"}
    for lon in range(0, 360, 30):
        # PROJ south polar stereographic: x = rho sin(lon), y = rho cos(lon) (y up on the page)
        a = math.radians(lon)
        x2, y2 = cx + r * math.sin(a), cy - r * math.cos(a)
        parts.append(f'<line class="s-faint" x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
        if lon in labels:
            lx, ly = cx + (r + 16) * math.sin(a), cy - (r + 16) * math.cos(a) + 4
            parts.append(_text(lx, ly, labels[lon], "ink3", "middle", "small"))
    parts.append(_arrow(cx, cy, cx + 78, cy, "s-ink", 7))
    parts.append(_arrow(cx, cy, cx, cy - 78, "s-ink", 7))
    parts.append(_text(cx + 82, cy + 4, "grid x", "ink", extra="small"))
    parts.append(_text(cx + 4, cy - 84, "grid y", "ink", extra="small"))
    parts.append(f'<circle class="s-ink f-acc" cx="{cx}" cy="{cy}" r="2.5" style="stroke:none"/>')
    parts.append(_text(cx - 8, cy + 16, "pole", "ink3", "end", "small"))
    lon = math.radians(129.78)
    rho = r * 0.38
    sx, sy = cx + rho * math.sin(lon), cy - rho * math.cos(lon)
    parts.append(f'<circle class="f-acc" cx="{sx:.1f}" cy="{sy:.1f}" r="3.5" style="stroke:none"/>')
    parts.append(_text(sx + 8, sy + 16, "site 129.78°E", "acc", extra="small"))
    ex, ey = math.cos(lon), math.sin(lon)  # east on the page, rotated with longitude
    nx, ny = math.sin(lon), -math.cos(lon)  # north points away from the south pole
    parts.append(_arrow(sx, sy, sx + 52 * ex, sy + 52 * ey, "s-acc", 6))
    parts.append(_arrow(sx, sy, sx + 52 * nx, sy + 52 * ny, "s-acc", 6))
    parts.append(_text(sx + 60 * ex, sy + 60 * ey + 4, "E", "acc", "middle", "small"))
    parts.append(_text(sx + 60 * nx, sy + 60 * ny + 4, "N", "acc", "middle", "small"))

    tx = 430
    notes = [
        ("Sun sources", "SPICE, Skyfield and the synthetic model all return"),
        ("", "a unit vector s in the Moon's body frame (MOON_ME)."),
        ("Mesh frame", "A projected DEM keeps its own grid axes. At the DEM"),
        ("", "centre we step the inverse projection along x and y,"),
        ("", "project out the local vertical and get R = [x̂; ŷ; û]."),
        ("Rotation", "s_mesh = R · s_body, once per step, for every Sun"),
        ("", "source alike. det R = +1 is checked when R is built."),
        ("Synthetic", "Synthetic terrain uses local east/north/up at the"),
        ("", "configured target instead of grid axes."),
    ]
    y = 64
    for head, line in notes:
        if head:
            y += 10
            parts.append(_text(tx, y, head.upper(), "acc", extra="small"))
            y += 16
        parts.append(_text(tx, y, line, "ink2", extra="sans small"))
        y += 16
    return _svg(w, h, "\n".join(parts), "Reference frames at the south pole")


# ---------------------------------------------------------------------------
# 3. Shadow geometry
# ---------------------------------------------------------------------------


def geometry() -> str:
    """Cross-section: low Sun, rim shadow line, a facet in penumbra."""
    w, h = 760, 330
    ground = [(20, 165), (150, 160), (210, 148), (230, 146), (250, 152), (300, 230),
              (340, 262), (400, 270), (470, 268), (520, 250), (580, 205), (620, 165),
              (650, 142), (670, 140), (700, 146), (740, 150)]
    parts = [_text(20, 24, "CROSS-SECTION THROUGH A POLAR CRATER, ANGLES EXAGGERATED", "acc",
                   extra="small")]
    outline = "M" + " L".join(f"{x} {y}" for x, y in ground)
    parts.append(f'<path class="f-rule" style="stroke:none" opacity="0.6" '
                 f'd="{outline} L740 292 L20 292 Z"/>')
    shade = [(230, 146), (595, 191), (580, 205), (520, 250), (470, 268), (400, 270),
             (340, 262), (300, 230), (250, 152)]
    parts.append('<path class="ink" style="stroke:none" opacity="0.13" d="M'
                 + " L".join(f"{x} {y}" for x, y in shade) + ' Z"/>')
    parts.append(f'<path class="s-ink" d="{outline}"/>')
    parts.append('<line class="s-rule" x1="230" y1="146" x2="595" y2="191" '
                 'stroke-dasharray="3 3"/>')
    parts.append(_text(470, 196, "rim shadow line", "ink3", "middle", "small"))

    sun = (46, 120)
    parts.append(f'<circle class="s-acc" cx="{sun[0]}" cy="{sun[1]}" r="14"/>')
    parts.append(_text(68, 96, "solar disk (0.53°, drawn larger)", "acc", extra="small"))
    facet = (598, 187)
    for dy, blocked in ((-10, False), (0, False), (10, True)):
        x0, y0 = sun[0] + 4, sun[1] + dy
        if blocked:
            t = (226 - x0) / (facet[0] - x0)
            xe, ye = 226, y0 + t * (facet[1] - y0)
            parts.append(f'<line class="s-rule" x1="{x0}" y1="{y0}" x2="{xe:.1f}" y2="{ye:.1f}" '
                         'stroke-dasharray="4 3"/>')
            parts.append(f'<path class="s-ink2" d="M{xe - 4:.1f} {ye - 4:.1f} l8 8 m0 -8 l-8 8"/>')
        else:
            parts.append(_arrow(x0, y0, facet[0] - 3, facet[1] - 1, "s-acc", 5))
    parts.append(_arrow(facet[0], facet[1], facet[0] - 24, facet[1] - 24, "s-ink", 5))
    parts.append(_text(facet[0] - 30, facet[1] - 28, "n", "ink", "end"))
    parts.append(f'<circle class="f-acc" style="stroke:none" cx="{facet[0]}" cy="{facet[1]}" '
                 'r="3"/>')
    parts.append(_text(612, 214, "facet in penumbra:", "ink", extra="small"))
    parts.append(_text(612, 229, "2 of 3 disk samples", "ink2", extra="small"))
    parts.append(_text(612, 244, "reach it, f = 0.67", "ink2", extra="small"))
    parts.append(_text(232, 176, "rim", "ink2", "middle", "small"))
    parts.append(_text(395, 256, "never lit", "ink2", "middle", "small"))
    parts.append(_text(20, 316, "Q = (1 − A(μ)) · S(t) · mean over samples of max(0, n·s) · V",
                       "ink2", extra="small"))
    return _svg(w, h, "\n".join(parts), "Shadowing and penumbra geometry")


# ---------------------------------------------------------------------------
# 4. Regolith column
# ---------------------------------------------------------------------------


def column() -> str:
    """The 1-D column under every facet: grid, properties, boundary conditions."""
    from core_engine.constants import load_config

    cfg = load_config()
    z = cfg.solver.grid.build_grid()
    w, h = 760, 420
    top, bottom = 70, 380
    zmax = float(z[-1])

    def ypos(depth: float) -> float:
        return top + (bottom - top) * math.log10(1 + depth / 0.002) / math.log10(1 + zmax / 0.002)

    parts = [_text(20, 24, "ONE REGOLITH COLUMN (DEPTH ON A LOG SCALE)", "acc", extra="small")]
    x0, x1 = 120, 300
    for depth in z:
        y = ypos(float(depth))
        parts.append(f'<line class="s-faint" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
    parts.append(f'<rect class="s-ink2" x="{x0}" y="{top}" width="{x1 - x0}" '
                 f'height="{bottom - top}"/>')
    parts.append(f'<line class="s-acc" x1="{x0 - 10}" y1="{top}" x2="{x1 + 10}" y2="{top}"/>')
    for depth, text in ((0.0, "0"), (0.02, "2 cm"), (0.1, "10 cm"), (1.0, "1 m"),
                        (zmax, f"{zmax:.1f} m")):
        y = ypos(depth)
        parts.append(f'<line class="s-ink2" x1="{x0 - 6}" y1="{y:.1f}" x2="{x0}" y2="{y:.1f}"/>')
        parts.append(_text(x0 - 10, y + 4, text, "ink2", "end", "small"))
    yb = ypos(cfg.regolith.conductivity_layer_boundary_m)
    parts.append(f'<line class="s-cold" x1="{x0}" y1="{yb:.1f}" x2="{x1}" y2="{yb:.1f}" '
                 'stroke-dasharray="5 3"/>')
    parts.append(_text(x0 + 8, yb - 6, "conductivity layer boundary", "ink3", extra="small"))
    parts.append(_text((x0 + x1) / 2, top - 30, "Sun, terrain IR", "acc", "middle", "small"))
    parts.append(_arrow((x0 + x1) / 2 - 30, top - 22, (x0 + x1) / 2 - 30, top - 3, "s-acc", 5))
    parts.append(_arrow((x0 + x1) / 2 + 30, top - 3, (x0 + x1) / 2 + 30, top - 22, "s-ink2", 5))
    parts.append(_text((x0 + x1) / 2 + 38, top - 10, "εσT⁴", "ink2", extra="small"))
    parts.append(_arrow((x0 + x1) / 2, bottom + 26, (x0 + x1) / 2, bottom + 3, "s-ink2", 5))
    parts.append(_text((x0 + x1) / 2 + 8, bottom + 22,
                       f"geothermal {cfg.regolith.geothermal_flux * 1000:.0f} mW/m²", "ink2",
                       extra="small"))
    tx = 420
    notes = [
        ("Surface node", "Half-cell, backward Euler. εσT⁴ is linearised"),
        ("", "and iterated (Newton) until the whole column changes"),
        ("", f"by less than {cfg.solver.newton.tolerance_K:g} K."),
        ("Interior", "Crank–Nicolson on a geometric grid: first cell"),
        ("", f"{cfg.solver.grid.dz_surface_m * 1000:.1f} mm, "
             f"growth ×{cfg.solver.grid.growth_ratio}, {cfg.solver.grid.num_layers + 1} nodes."),
        ("Properties", "ρ(z) rises from 1100 to 1800 kg/m³ (H = 6 cm);"),
        ("", "k(T) = k_c + k_r T³; cp(T) polynomial (Hayne 2017)."),
        ("Bottom", "Constant geothermal flux into the last half-cell."),
        ("Per facet", "Every facet has its own column; all columns"),
        ("", "advance together in one Numba parallel call."),
    ]
    y = 74
    for head, line in notes:
        if head:
            y += 10
            parts.append(_text(tx, y, head.upper(), "acc", extra="small"))
            y += 16
        parts.append(_text(tx, y, line, "ink2", extra="sans small"))
        y += 16
    return _svg(w, h, "\n".join(parts), "Regolith column and boundary conditions")


def diagrams() -> None:
    """Write all SVG diagrams."""
    out = IMG / "diagrams"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("pipeline", pipeline), ("frames", frames), ("geometry", geometry),
                     ("column", column)):
        (out / f"{name}.svg").write_text(fn(), encoding="utf-8")
        print(f"wrote docs/img/diagrams/{name}.svg")


def run_figures(folder: str) -> None:
    """Render a run's report figures into docs/img/run/."""
    from simulation.io_manager import ResultsBundle
    from visualization.figures import render_run_figures

    out = IMG / "run"
    out.mkdir(parents=True, exist_ok=True)
    for path in render_run_figures(ResultsBundle.open(folder), out, dpi=130, hero_dpi=150):
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    shutil.copy(out / "hero.png", IMG / "hero.png")


def bench_chart(results_file: str) -> None:
    """Bar chart of benchmark wall times by component."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualization.figures import _setup

    pal = _setup()
    rows = json.loads(Path(results_file).read_text(encoding="utf-8"))["cases"]
    labels = [f"{r['faces'] / 1000:.0f}k faces" for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    left = [0.0] * len(rows)
    for key, color, name in (("bvh_s", style.LIGHT.ink_3, "BVH build"),
                             ("shadows_s", style.LIGHT.accent, "shadows"),
                             ("thermal_s", style.LIGHT.cold, "heat conduction")):
        values = [r[key] for r in rows]
        ax.barh(labels, values, left=left, color=color, height=0.5, label=name)
        left = [a + b for a, b in zip(left, values, strict=True)]
    ax.set_xlabel("seconds for one simulated day (dt 300 s, 16 disk samples)")
    ax.legend(loc="lower right", ncols=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.22)
    out = IMG / "bench.png"
    fig.savefig(out, dpi=150, facecolor=pal.paper)
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "diagrams":
        diagrams()
    elif argv[0] == "run":
        run_figures(argv[1])
    elif argv[0] == "bench":
        bench_chart(argv[1])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
