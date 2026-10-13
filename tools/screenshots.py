"""Capture the desktop app's pages as PNG files (used for the docs).

    python tools/screenshots.py output/demo docs/img/app [output/other_run]

With a second run folder, the Compare page shows the second run minus the
first.

Runs Qt offscreen, so it works on a headless machine or in CI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ARTEMIS_REDUCE_MOTION", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    run_folder = sys.argv[1] if len(sys.argv) > 1 else "output/demo"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/img/app")
    other = sys.argv[3] if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)

    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    QCoreApplication.setOrganizationName("ArtemisThermalBase-screenshots")
    _app = QApplication.instance() or QApplication([])
    from artemis_thermal.gui import theme
    from artemis_thermal.gui.main_window import MainWindow

    theme.load_fonts()

    def settle(ms: int = 600) -> None:
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    for dark in (False, True):
        window = MainWindow(dark=dark)
        window.resize(1440, 920)
        window.show()
        window.open_run(run_folder)
        for _ in range(100):
            if window.results_page.bundle is not None:
                break
            settle(200)
        settle(600)
        bundle = window.results_page.bundle
        if bundle is not None:
            left, right, bottom, top = bundle.extent
            window.results_page._pick(0.35 * left + 0.65 * right, 0.7 * bottom + 0.3 * top)
            settle(1200)
        if other:
            window.compare_page.set_run("a", run_folder)
            window.compare_page.set_run("b", other)
            settle(2500)
        suffix = "dark" if dark else "light"
        for key in ("results", "run", "reference", "compare"):
            window.go(key, animate=False)
            settle(900)
            window.grab().save(str(out / f"{key}-{suffix}.png"))
        window.results_page._tabs.select("ice", emit=True)
        window.go("results", animate=False)
        settle(700)
        window.grab().save(str(out / f"results-ice-{suffix}.png"))
        window.close()
    print(f"Saved screenshots to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
