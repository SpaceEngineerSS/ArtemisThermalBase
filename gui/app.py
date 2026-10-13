"""Start the desktop app: ``artemis ui [RUN_FOLDER]`` or ``artemis-ui``."""

from __future__ import annotations

import multiprocessing as mp
import sys


def main(run_folder: str | None = None) -> int:
    """Create the application and the main window."""
    mp.freeze_support()
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QApplication

    from artemis_thermal import __version__

    QCoreApplication.setOrganizationName("ArtemisThermalBase")
    QCoreApplication.setApplicationName("ArtemisThermalBase")
    QCoreApplication.setApplicationVersion(__version__)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication(sys.argv[:1])
    # Numbers are scientific data: always "1234.5", never "1.234,5".
    from PySide6.QtCore import QLocale

    locale = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
    locale.setNumberOptions(QLocale.NumberOption.OmitGroupSeparator)
    QLocale.setDefault(locale)

    from gui import theme
    from gui.main_window import MainWindow

    theme.load_fonts()
    window = MainWindow()
    if run_folder is None and len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
            and sys.argv[0].endswith(("artemis-ui", "artemis-ui.exe", "app.py")):
        run_folder = sys.argv[1]
    if run_folder:
        window.open_run(run_folder)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
