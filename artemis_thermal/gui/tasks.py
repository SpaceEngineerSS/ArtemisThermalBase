"""Run short blocking jobs (file reads, statistics) off the UI thread."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Relay(QObject):
    done = Signal(object, object)   # token, result
    failed = Signal(object, str)    # token, message


class _Job(QRunnable):
    def __init__(self, fn: Callable[[], Any], token: object, relay: _Relay) -> None:
        super().__init__()
        self._fn = fn
        self._token = token
        self._relay = relay

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:
            self._relay.failed.emit(self._token, f"{exc}\n\n{traceback.format_exc()}")
        else:
            self._relay.done.emit(self._token, result)


class Background(QObject):
    """Submit jobs; only the result of the latest job per channel is delivered.

    Switching runs quickly therefore never shows a stale result that arrives
    after a newer one.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._relay = _Relay(self)
        self._relay.done.connect(self._on_done)
        self._relay.failed.connect(self._on_failed)
        self._latest: dict[str, int] = {}
        self._handlers: dict[int, tuple[Callable, Callable | None]] = {}
        self._counter = 0
        self._pool = QThreadPool.globalInstance()

    def submit(
        self,
        channel: str,
        fn: Callable[[], Any],
        on_done: Callable[[Any], None],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        """Run ``fn`` in the pool; call ``on_done`` on the UI thread."""
        self._counter += 1
        token = self._counter
        self._latest[channel] = token
        self._handlers[token] = (on_done, on_error)
        self._pool.start(_Job(fn, (channel, token), self._relay))

    def _on_done(self, key: tuple[str, int], result: object) -> None:
        channel, token = key
        handlers = self._handlers.pop(token, None)
        if handlers and self._latest.get(channel) == token:
            handlers[0](result)

    def _on_failed(self, key: tuple[str, int], message: str) -> None:
        channel, token = key
        handlers = self._handlers.pop(token, None)
        if handlers and handlers[1] and self._latest.get(channel) == token:
            handlers[1](message)
