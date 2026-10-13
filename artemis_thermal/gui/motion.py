"""Small, purposeful motion.

Three kinds only: a page fades in when you switch to it, numbers glide to
their new value, and maps cross-fade when the layer or time step changes.
Everything is short (120–220 ms) and uses one easing curve. With reduced
motion on (setting or ``ARTEMIS_REDUCE_MOTION=1``) every helper jumps
straight to the end state.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from PySide6.QtCore import QEasingCurve, QObject, QPropertyAnimation, QVariantAnimation
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget

FAST_MS = 120
BASE_MS = 180
SLOW_MS = 220

_reduce = os.environ.get("ARTEMIS_REDUCE_MOTION", "") not in ("", "0")


def set_reduced(value: bool) -> None:
    """Turn reduced motion on or off for the whole app."""
    global _reduce
    _reduce = bool(value)


def reduced() -> bool:
    """True when animations are disabled."""
    return _reduce


def _curve() -> QEasingCurve:
    # A quick start and a long, soft landing: cubic-bezier(0.2, 0.7, 0.2, 1).
    curve = QEasingCurve(QEasingCurve.Type.BezierSpline)
    curve.addCubicBezierSegment(
        _pt(0.2, 0.7), _pt(0.2, 1.0), _pt(1.0, 1.0)
    )
    return curve


def _pt(x: float, y: float):
    from PySide6.QtCore import QPointF

    return QPointF(x, y)


def fade_in(widget: QWidget, duration: int = BASE_MS) -> None:
    """Fade a widget from transparent to opaque."""
    if _reduce:
        widget.setGraphicsEffect(None)  # type: ignore[arg-type]
        return
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(0.0)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(_curve())
    # Drop the effect afterwards: an idle opacity effect forces offscreen
    # rendering of the whole subtree, which is slow on weak GPUs.
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))  # type: ignore[arg-type]
    anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


def tween(
    owner: QObject,
    start: float,
    end: float,
    apply: Callable[[float], None],
    duration: int = SLOW_MS,
) -> QVariantAnimation | None:
    """Animate a float from ``start`` to ``end``, calling ``apply`` each frame."""
    if _reduce or start == end:
        apply(end)
        return None
    anim = QVariantAnimation(owner)
    anim.setDuration(duration)
    anim.setStartValue(float(start))
    anim.setEndValue(float(end))
    anim.setEasingCurve(_curve())
    anim.valueChanged.connect(lambda v: apply(float(v)))
    anim.start(QVariantAnimation.DeletionPolicy.DeleteWhenStopped)
    return anim
