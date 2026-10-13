"""Desktop app: data layer, window smoke test, and a real worker-process run."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["ARTEMIS_REDUCE_MOTION"] = "1"

from core_engine.constants import load_config  # noqa: E402
from simulation.runner import SimulationRunner  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def run_folder(tmp_path_factory):
    config = load_config()
    config = replace(
        config,
        synthetic_dem=replace(config.synthetic_dem, radius_m=2000.0, depth_m=600.0,
                              grid_resolution_m=250.0, domain_padding_m=500.0),
    )
    folder = tmp_path_factory.mktemp("run")
    SimulationRunner(config).run(
        start_time=datetime(2025, 1, 1, tzinfo=UTC),
        duration_hours=6.0,
        dt_s=900.0,
        output_interval_s=1800.0,
        output_dir=folder,
    )
    return folder


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    QCoreApplication.setOrganizationName("ArtemisThermalBase-tests")
    app = QApplication.instance() or QApplication([])
    yield app


def _wait(predicate, timeout_ms: int = 20000) -> bool:
    from PySide6.QtCore import QEventLoop, QTimer

    waited = 0
    while waited < timeout_ms:
        if predicate():
            return True
        loop = QEventLoop()
        QTimer.singleShot(50, loop.quit)
        loop.exec()
        waited += 50
    return predicate()


def test_layers_and_summary(run_folder) -> None:
    from artemis_thermal import style
    from gui import run_data

    bundle = run_data.open_bundle(str(run_folder))
    for key, _title in run_data.LAYER_KEYS:
        spec = run_data.build_layer(bundle, key, style.LIGHT)
        assert spec.data.shape == bundle.grid_shape
        assert spec.levels[1] > spec.levels[0]
    summary = run_data.summarize(bundle)
    assert summary.t_min <= summary.t_mean_area <= summary.t_max
    assert summary.stable_km2 + summary.marginal_km2 <= summary.area_km2 + 1e-9


def test_site_report_is_physical(run_folder) -> None:
    from gui import run_data

    bundle = run_data.open_bundle(str(run_folder))
    report = run_data.site_report(bundle, 0.0, 0.0)  # crater floor
    assert 0.0 <= report.slope_deg < 10.0
    assert report.t_min <= report.t_mean <= report.t_max
    assert 0.0 <= report.sunlit_pct <= 100.0
    assert report.series.size == bundle.snapshot_count
    assert report.longest_dark_h is not None


def test_window_opens_run_and_inspects_a_site(qapp, run_folder) -> None:
    from gui.main_window import MainWindow

    window = MainWindow(dark=False)
    window.show()
    window.open_run(str(run_folder))
    assert _wait(lambda: window.results_page.bundle is not None)
    window.results_page._pick(0.0, 0.0)
    assert _wait(lambda: window.results_page._site is not None)
    window.toggle_theme()
    assert window.theme.dark
    for key in ("run", "compare", "reference", "results"):
        window.go(key)
    window.close()


def test_worker_process_runs_and_reports(qapp, tmp_path) -> None:
    from gui.worker import RunController, RunRequest

    controller = RunController()
    events: dict[str, object] = {}
    controller.progress.connect(lambda p: events.setdefault("progress", p))
    controller.finished.connect(lambda folder: events.__setitem__("done", folder))
    controller.failed.connect(lambda msg, tb: events.__setitem__("error", msg + tb))
    controller.start(RunRequest(
        output_dir=str(tmp_path / "worker_run"),
        grid_resolution_m=2000.0,
        duration_hours=1.0,
        dt_s=900.0,
        output_interval_s=1800.0,
        point_source=True,
    ))
    assert _wait(lambda: "done" in events or "error" in events, timeout_ms=180000)
    assert "error" not in events, events.get("error")
    assert (tmp_path / "worker_run" / "surface_T_final.npy").is_file()
    assert (tmp_path / "worker_run" / "hero.png").is_file()
    assert np.isfinite(np.load(tmp_path / "worker_run" / "surface_T_final.npy")).all()
