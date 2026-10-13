"""pyqtgraph line plots styled like the report figures."""

from __future__ import annotations

import pyqtgraph as pg

from gui.theme import Theme


def styled_plot(theme: Theme, height: int | None = None) -> pg.PlotWidget:
    """A PlotWidget with hairline axes, inward ticks and mono tick labels."""
    widget = pg.PlotWidget()
    if height:
        widget.setFixedHeight(height)
    restyle(widget, theme)
    item = widget.getPlotItem()
    item.hideButtons()
    item.setMenuEnabled(False)
    item.getViewBox().setMouseEnabled(x=True, y=False)
    return widget


def restyle(widget: pg.PlotWidget, theme: Theme) -> None:
    """Apply ``theme`` to an existing plot."""
    p = theme.pal
    widget.setBackground(p.paper)
    item = widget.getPlotItem()
    for side in ("left", "bottom"):
        axis = item.getAxis(side)
        axis.setPen(pg.mkPen(p.rule_strong, width=1))
        axis.setTextPen(pg.mkPen(p.ink_2))
        axis.setTickFont(theme.mono(7.5))
        axis.setStyle(tickLength=-4)
        axis.enableAutoSIPrefix(False)


def series_pens(theme: Theme) -> list:
    """Pens for up to four series: ink, accent, cold, secondary ink."""
    p = theme.pal
    return [pg.mkPen(c, width=1.4) for c in (p.ink, p.accent, p.cold, p.ink_3)]
