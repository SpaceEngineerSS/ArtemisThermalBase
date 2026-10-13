"""Reference page: the surface models in use, drawn from the active config."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QGridLayout, QScrollArea, QVBoxLayout, QWidget

from core_engine.constants import SimulationConfig, load_config
from gui.plots import restyle, series_pens, styled_plot
from gui.theme import Theme
from gui.widgets import FigureCaption, label, section


class ReferencePage(QWidget):
    """Hapke albedo, roughness emissivity, regolith column and assumptions."""

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Page")
        self._theme = theme
        self._built = False
        self._config: SimulationConfig | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)
        self._content = QWidget()
        self._content.setObjectName("Page")
        scroll.setWidget(self._content)
        self._body = QVBoxLayout(self._content)
        self._body.setContentsMargins(32, 26, 32, 28)
        self._body.setSpacing(14)
        self._body.addWidget(section(
            "04  Reference", "Surface models",
            "Curves are computed from the default configuration the first time this page "
            "opens. The physics behind them is in docs/physics.md.",
        ))
        self._plots: list[pg.PlotWidget] = []

    def showEvent(self, event) -> None:  # noqa: D102
        if not self._built:
            self._build()
        super().showEvent(event)

    def apply_theme(self, theme: Theme) -> None:
        """Restyle and redraw."""
        self._theme = theme
        for plot in self._plots:
            restyle(plot, theme)
        if self._built:
            self._draw()

    def _build(self) -> None:
        self._built = True
        self._config = load_config()
        grid = QGridLayout()
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(6)
        captions = (
            "Directional-hemispherical albedo of the Hapke model against incidence angle. "
            "Absorbed sunlight is (1 − A) S cos i.",
            "Effective emissivity from sub-pixel roughness (Bandfield et al. 2015) against "
            "facet slope.",
            "Bulk density against depth in the top half metre (Hayne et al. 2017).",
            "Thermal conductivity against depth at 100 K (ink) and 250 K (orange); the "
            "radiative term makes warm regolith conduct better.",
        )
        for index, caption in enumerate(captions):
            plot = styled_plot(self._theme, 220)
            self._plots.append(plot)
            cell = QVBoxLayout()
            cell.addWidget(plot)
            cell.addWidget(FigureCaption(index + 4, caption))
            grid.addLayout(cell, index // 2, index % 2)
        self._body.addLayout(grid)
        self._assumptions = label("", "caption")
        self._assumptions.setWordWrap(True)
        self._body.addWidget(label("ASSUMPTIONS IN THE ACTIVE CONFIGURATION", "kicker"))
        self._body.addWidget(self._assumptions)
        self._body.addStretch(1)
        self._draw()

    def _draw(self) -> None:
        cfg = self._config
        if cfg is None:
            return
        from core_engine.reflectance import compute_adh_array
        from core_engine.roughness import effective_emissivity

        pens = series_pens(self._theme)
        hp = cfg.hapke
        incidence = np.linspace(0.0, 89.5, 120)
        albedo = compute_adh_array(np.cos(np.radians(incidence)), w=hp.single_scattering_albedo,
                                   b=hp.b, c=hp.c, B_SH0=hp.b_sh0, h_s=hp.h_s)
        p0 = self._plots[0].getPlotItem()
        p0.clear()
        p0.plot(incidence, albedo, pen=pens[0])
        p0.addLine(y=cfg.surface.bond_albedo, pen=pg.mkPen(self._theme.pal.accent, width=1,
                                                           style=pg.QtCore.Qt.PenStyle.DashLine))
        p0.setLabel("bottom", "incidence [°]")
        p0.setLabel("left", "albedo")

        slopes = np.linspace(0.0, 45.0, 90)
        emis = [effective_emissivity(cfg.surface.thermal_emissivity, s,
                                     cfg.roughness.cavity_coefficient) for s in slopes]
        p1 = self._plots[1].getPlotItem()
        p1.clear()
        p1.plot(slopes, emis, pen=pens[1])
        p1.setLabel("bottom", "slope [°]")
        p1.setLabel("left", "ε eff")

        z = cfg.solver.grid.build_grid()
        z = z[z <= 0.5]
        reg = cfg.regolith
        rho = reg.density_deep - (reg.density_deep - reg.density_surface) * np.exp(
            -z / reg.density_transition_m)
        p2 = self._plots[2].getPlotItem()
        p2.clear()
        p2.plot(rho, -z, pen=pens[0])
        p2.setLabel("bottom", "density [kg/m³]")
        p2.setLabel("left", "depth [m]")

        upper = z < reg.conductivity_layer_boundary_m
        k_c = np.where(upper, reg.conductivity_surface.k_contact, reg.conductivity_deep.k_contact)
        k_r = np.where(upper, reg.conductivity_surface.k_radiative,
                       reg.conductivity_deep.k_radiative)
        p3 = self._plots[3].getPlotItem()
        p3.clear()
        for temperature, pen in ((100.0, pens[0]), (250.0, pens[1])):
            p3.plot((k_c + k_r * temperature**3) * 1000.0, -z, pen=pen)
        p3.setLabel("bottom", "conductivity [mW/m/K]")
        p3.setLabel("left", "depth [m]")

        self._assumptions.setText("\n".join(
            f"{a.parameter}: {a.value}  ({a.source}; {a.uncertainty})" for a in cfg.assumptions
        ))
