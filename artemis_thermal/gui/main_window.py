"""Main window: header strip, numbered section rail, pages."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from artemis_thermal import __version__
from artemis_thermal.gui import motion, theme
from artemis_thermal.gui.mark import MarkWidget, mark_icon
from artemis_thermal.gui.pages.compare import ComparePage
from artemis_thermal.gui.pages.reference import ReferencePage
from artemis_thermal.gui.pages.results import ResultsPage
from artemis_thermal.gui.pages.run import RunPage
from artemis_thermal.gui.tasks import Background
from artemis_thermal.gui.widgets import button, label, rule
from artemis_thermal.simulation.io_manager import list_runs

SECTIONS = (
    ("run", "01", "Run"),
    ("results", "02", "Results"),
    ("compare", "03", "Compare"),
    ("reference", "04", "Reference"),
)


class MainWindow(QMainWindow):
    """The application window."""

    def __init__(self, dark: bool | None = None) -> None:
        super().__init__()
        self._settings = QSettings()
        stored = self._settings.value("ui/theme", "system")
        if dark is None:
            dark = theme.system_prefers_dark() if stored == "system" else stored == "dark"
        motion.set_reduced(
            motion.reduced() or self._settings.value("ui/reduce_motion", "false") == "true"
        )
        self.theme = theme.make_theme(dark)
        theme.apply(QApplication.instance(), self.theme)  # type: ignore[arg-type]

        self.setWindowTitle(f"ArtemisThermalBase {__version__}")
        self.setWindowIcon(mark_icon(self.theme.pal.ink, self.theme.pal.accent,
                                     self.theme.pal.paper))
        self.resize(1360, 880)
        self.setMinimumSize(1000, 640)

        self.background = Background(self)
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_header())
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        outer.addLayout(middle, 1)
        middle.addWidget(self._build_rail())

        self.pages = QStackedWidget()
        self.run_page = RunPage(self.theme)
        self.results_page = ResultsPage(self.theme, self.background)
        self.compare_page = ComparePage(self.theme, self.background)
        self.reference_page = ReferencePage(self.theme)
        self._page_index = {}
        for key, page in (("run", self.run_page), ("results", self.results_page),
                          ("compare", self.compare_page), ("reference", self.reference_page)):
            self._page_index[key] = self.pages.addWidget(page)
        middle.addWidget(self.pages, 1)

        self.run_page.run_finished.connect(self._run_finished)
        for page in (self.run_page, self.results_page, self.compare_page):
            page.status.connect(self.show_status)

        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self._status.setText(""))
        self._install_shortcuts()
        self.refresh_recent()
        self.go("run", animate=False)

    # -- chrome -----------------------------------------------------------

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("Header")
        header.setFixedHeight(46)
        row = QHBoxLayout(header)
        row.setContentsMargins(18, 0, 18, 0)
        row.setSpacing(10)
        self._mark = MarkWidget(self.theme.color("ink"), self.theme.color("accent"), 22)
        row.addWidget(self._mark)
        name = label("ARTEMIS THERMAL BASE", "kicker")
        row.addWidget(name)
        row.addWidget(label(__version__, "mono"))
        row.addSpacing(16)
        self._status = label("", "caption")
        row.addWidget(self._status, 1)
        self._theme_btn = button("", "link", self.toggle_theme)
        self._motion_btn = button("", "link", self.toggle_motion)
        row.addWidget(self._motion_btn)
        row.addWidget(label("·", "label"))
        row.addWidget(self._theme_btn)
        self._sync_header_buttons()
        return header

    def _build_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("Rail")
        rail.setFixedWidth(208)
        col = QVBoxLayout(rail)
        col.setContentsMargins(0, 22, 0, 16)
        col.setSpacing(0)
        self._rail_group = QButtonGroup(self)
        self._rail_group.setExclusive(True)
        self._rail_buttons = {}
        for key, number, text in SECTIONS:
            btn = button(f"{number}   {text}", "rail", lambda _=False, k=key: self.go(k))
            btn.setCheckable(True)
            self._rail_group.addButton(btn)
            self._rail_buttons[key] = btn
            col.addWidget(btn)
        col.addSpacing(22)
        rule_box = QHBoxLayout()
        rule_box.setContentsMargins(16, 0, 16, 0)
        rule_box.addWidget(rule())
        col.addLayout(rule_box)
        col.addSpacing(12)
        heading = label("RECENT RUNS", "kicker")
        heading.setContentsMargins(16, 0, 0, 4)
        col.addWidget(heading)
        self._recent = QListWidget()
        self._recent.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._recent.itemActivated.connect(self._open_recent)
        self._recent.itemClicked.connect(self._open_recent)
        col.addWidget(self._recent, 1)
        open_btn = button("Open folder…", "link", self._browse_run)
        open_btn.setContentsMargins(16, 0, 0, 0)
        open_row = QHBoxLayout()
        open_row.setContentsMargins(16, 6, 0, 0)
        open_row.addWidget(open_btn)
        col.addLayout(open_row)
        return rail

    def _install_shortcuts(self) -> None:
        for index, (key, _n, _t) in enumerate(SECTIONS, start=1):
            action = QAction(self)
            action.setShortcut(QKeySequence(f"Ctrl+{index}"))
            action.triggered.connect(lambda _=False, k=key: self.go(k))
            self.addAction(action)
        open_action = QAction(self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._browse_run)
        self.addAction(open_action)
        run_action = QAction(self)
        run_action.setShortcut(QKeySequence("Ctrl+Return"))
        run_action.triggered.connect(self.run_page.start_run)
        self.addAction(run_action)

    # -- navigation -------------------------------------------------------

    def go(self, key: str, animate: bool = True) -> None:
        """Show a section."""
        self._rail_buttons[key].setChecked(True)
        index = self._page_index[key]
        if self.pages.currentIndex() != index:
            self.pages.setCurrentIndex(index)
            current = self.pages.currentWidget()
            if animate and current is not None:
                motion.fade_in(current)

    def open_run(self, folder: str) -> None:
        """Open a run folder on the Results page."""
        self.results_page.open_run(folder)
        self.compare_page.set_run("b", folder)
        self.go("results")
        self._remember(folder)

    def _run_finished(self, folder: str) -> None:
        previous = self.results_page.bundle
        if previous is not None and str(previous.path) != folder:
            self.compare_page.set_run("a", str(previous.path))
        self.open_run(folder)

    def _browse_run(self) -> None:
        self.results_page.browse()
        self.go("results")

    def _open_recent(self, item: QListWidgetItem) -> None:
        folder = item.data(Qt.ItemDataRole.UserRole)
        if folder:
            self.open_run(folder)

    def _remember(self, folder: str) -> None:
        folder = str(Path(folder).resolve())
        recent = [f for f in self._recent_setting() if f != folder]
        recent.insert(0, folder)
        self._settings.setValue("runs/recent", recent[:12])
        self.refresh_recent()

    def _recent_setting(self) -> list[str]:
        value = self._settings.value("runs/recent", [])
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value] if isinstance(value, list) else []

    def refresh_recent(self) -> None:
        """Fill the recent-runs list from settings and ./output."""
        known = [str(Path(f).resolve()) for f in self._recent_setting()
                 if (Path(f) / "metadata.json").is_file()]
        for found in list_runs(Path.cwd() / "output"):
            if str(found) not in known:
                known.append(str(found))
        self._recent.clear()
        for folder in list(dict.fromkeys(known))[:12]:
            item = QListWidgetItem(Path(folder).name)
            item.setToolTip(folder)
            item.setData(Qt.ItemDataRole.UserRole, folder)
            self._recent.addItem(item)

    # -- settings ---------------------------------------------------------

    def show_status(self, text: str) -> None:
        """Show a transient status message in the header."""
        self._status.setText(text)
        self._status_timer.start(8000)

    def _sync_header_buttons(self) -> None:
        self._theme_btn.setText("Dark theme" if self.theme.dark else "Light theme")
        self._theme_btn.setToolTip("Click to switch theme")
        self._motion_btn.setText("Motion off" if motion.reduced() else "Motion on")
        self._motion_btn.setToolTip("Click to turn interface animations on or off")

    def toggle_theme(self) -> None:
        """Switch between light and dark."""
        self.theme = theme.make_theme(not self.theme.dark)
        self._settings.setValue("ui/theme", "dark" if self.theme.dark else "light")
        theme.apply(QApplication.instance(), self.theme)  # type: ignore[arg-type]
        self._mark.set_colors(self.theme.color("ink"), self.theme.color("accent"))
        for page in (self.run_page, self.results_page, self.compare_page, self.reference_page):
            page.apply_theme(self.theme)
        self._sync_header_buttons()

    def toggle_motion(self) -> None:
        """Turn animations off or on."""
        motion.set_reduced(not motion.reduced())
        self._settings.setValue("ui/reduce_motion", "true" if motion.reduced() else "false")
        self._sync_header_buttons()

    def closeEvent(self, event) -> None:  # noqa: D102
        self.run_page.controller.shutdown()
        super().closeEvent(event)
