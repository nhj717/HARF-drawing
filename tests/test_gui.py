"""Headless offscreen Qt GUI tests for the 5-tab workstation (A1-A4).

Run with ``QT_QPA_PLATFORM=offscreen uv run pytest tests/test_gui.py``.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.main import main  # noqa: E402
from app.main_window import MainWindow  # noqa: E402
from app.widgets.plot_canvas import SCHEMATIC_VIEW  # noqa: E402
from app.workers import OptimizationWorker, SimulationWorker  # noqa: E402
from src.io_utils import read_results_table  # noqa: E402
from src.models import (  # noqa: E402
    OptimizationInputs,
    OptimizationResult,
    SimulationResult,
    SimulationStatus,
    default_optimization_inputs,
    default_simulation_inputs,
    um_to_m,
)
from src.solvers import run_direct_simulation, run_optimization  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication(["test-harf"])
    yield app


@pytest.fixture()
def window(qapp) -> MainWindow:
    win = MainWindow()
    yield win
    win.close()


def _wait_for_result(qapp, tab, timeout_s: float = 60.0) -> SimulationResult:
    """Pump the event loop until the worker delivers a result."""
    deadline = time.monotonic() + timeout_s
    while tab.result is None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert tab.result is not None, "worker never produced a result"
    return tab.result


# ---------------------------------------------------------------------------
# A1: tabs, defaults, dynamic rings, badges, formulas
# ---------------------------------------------------------------------------


def test_all_tabs_instantiated(window: MainWindow) -> None:
    labels = [
        window.tabs.tabText(i) for i in range(window.tabs.count())
    ]
    assert labels == [
        "Parameters", "Jacket", "Capillaries", "Direct computation",
        "Optimization",
    ]
    assert window.windowTitle() == "HARF Drawing Simulation Workstation"


def test_exact_v15_defaults(window: MainWindow) -> None:
    p, j, c, d, o = (
        window.parameter_tab, window.jacket_tab, window.capillaries_tab,
        window.direct_tab, window.optimization_tab,
    )
    assert p.drawTensionEdit.text() == "84"
    assert p.feedSpeedEdit.text() == "4"
    assert p.lengthEdit.text() == "0.1"
    assert p.textEdit.text() == "20"
    assert p.dT0Edit.text() == "500"
    assert p.LfourEdit.text() == "0.065"
    assert j.D0Edit.text() == "8.3556"
    assert j.d0Edit.text() == "2.4"
    assert c.ncapSpinBox.value() == 2
    assert d.TmaxEdit.text() == "1806.9"
    assert d.PcoreEdit.text() == "15.6"
    assert d.pcapEdits[0].text() == "63.4"
    assert o.tfEdit.text() == "3.5"
    assert o.Dcf1Edit.text() == "110"
    assert o.DcoreEdit.text() == "156"
    assert o.tmaxMinEdit.text() == "1750"
    assert o.tmaxMaxEdit.text() == "1850"
    assert o.pcoreMinEdit.text() == "10"
    assert o.pcoreMaxEdit.text() == "20"
    assert o.pcapMinEdit.text() == "50"
    assert o.pcapMaxEdit.text() == "150"


def test_build_inputs_matches_defaults(window: MainWindow) -> None:
    inputs = window.build_inputs()
    reference = default_simulation_inputs(nb_nested=1)
    assert inputs.r0 == pytest.approx(reference.r0)
    assert inputs.R0 == pytest.approx(reference.R0)
    assert inputs.rings[0].Rc0 == pytest.approx(reference.rings[0].Rc0)
    assert inputs.rings[0].rc0 == pytest.approx(reference.rings[0].rc0)
    assert inputs.w0 == pytest.approx(reference.w0)
    assert inputs.tension == pytest.approx(reference.tension)
    assert inputs.L == pytest.approx(reference.L)
    assert inputs.Lfour == pytest.approx(reference.Lfour)
    assert inputs.Tmax_C == pytest.approx(reference.Tmax_C)
    assert inputs.Pcore == pytest.approx(reference.Pcore)
    assert inputs.Pcap == pytest.approx(reference.Pcap)
    assert inputs.n_capillaries == reference.n_capillaries
    assert inputs.furnace_profile_expr == reference.furnace_profile_expr
    assert inputs.viscosity_expr == reference.viscosity_expr


def test_nested_ring_dynamic_toggle(window: MainWindow) -> None:
    c = window.capillaries_tab
    assert c.ring_boxes[0].isEnabled()
    assert not c.ring_boxes[1].isEnabled()
    assert not c.ring_boxes[2].isEnabled()
    c.nestedSpinBox.setValue(2)
    assert c.ring_boxes[1].isEnabled()
    assert not c.ring_boxes[2].isEnabled()
    assert window.build_inputs().nb_nested == 2
    c.nestedSpinBox.setValue(3)
    assert all(box.isEnabled() for box in c.ring_boxes)
    assert window.build_inputs().nb_nested == 3
    c.nestedSpinBox.setValue(1)
    assert c.ring_boxes[0].isEnabled()
    assert not c.ring_boxes[2].isEnabled()


def test_geometry_and_contact_badges(window: MainWindow) -> None:
    j, c = window.jacket_tab, window.capillaries_tab
    # Default d0/D0 = 2.4/8.3556 < 1 -> green.
    assert j.dRatioBadge.is_ok is True
    j.d0Edit.setText("10")
    assert j.dRatioBadge.is_ok is False
    j.d0Edit.setText("2.4")
    # Default ring ratios are nested -> green.
    assert c.ratioBadges[0].is_ok is True
    assert c.ratioBadges[1].is_ok is True
    # Default Rc01 = 0.24 mm < Rcontact = r0/2 = 0.6 mm -> green.
    assert c.contactBadge.is_ok is True
    c.dcEdits[0].setText("2.4")  # Rc01 = 1.2 mm > 0.6 mm -> red
    assert c.contactBadge.is_ok is False
    c.dcEdits[0].setText("0.48")
    # Nested badge Dc02/dc01 turns red when ring 2 exceeds ring 1 bore.
    c.nestedSpinBox.setValue(2)
    c.dccEdits[0].setText("0.1")  # dc01 = 0.05 mm
    c.dcEdits[1].setText("0.4")   # Dc02 = 0.2 mm > 0.05 mm
    assert c.nestedBadges[0].is_ok is False


def test_live_formula_validation_styling(window: MainWindow) -> None:
    j = window.jacket_tab
    assert j.viscosityEdit.property("formula_valid") is True
    j.viscosityEdit.setText("10^(-7.946)*exp(66337/T")
    assert j.viscosityEdit.property("formula_valid") is False
    assert "background" in j.viscosityEdit.styleSheet()
    j.gammaEdit.setText("__import__('os')")
    assert j.gammaEdit.property("formula_valid") is False
    j.viscosityEdit.setText("10^(-7.946)*exp(66337/T)")
    assert j.viscosityEdit.property("formula_valid") is True


def test_invalid_formula_run_is_graceful(window: MainWindow, qapp) -> None:
    window.jacket_tab.viscosityEdit.setText("not a formula ((")
    window.direct_tab.runButton.click()
    result = _wait_for_result(qapp, window.direct_tab)
    assert result.status == SimulationStatus.INPUT_ERROR
    assert window.direct_tab.banner_visible()
    assert "Invalid" in window.direct_tab.diag_banner.text()
    assert result.finite()


# ---------------------------------------------------------------------------
# A2: 12 plot views, cursor readout, zmSlider, partial trajectories
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def success_result() -> SimulationResult:
    return run_direct_simulation(default_simulation_inputs())


def test_all_plot_views_render(window: MainWindow, success_result) -> None:
    d = window.direct_tab
    d.set_result(success_result)
    assert not d.banner_visible()
    combo = d.viewCombo
    for i in range(combo.count()):
        combo.setCurrentIndex(i)
        view = combo.currentText()
        if view == SCHEMATIC_VIEW:
            assert d.view_stack.currentIndex() == 1
            assert d.crossSection.z_label.text() != "z = -- mm"
        else:
            assert d.view_stack.currentIndex() == 0
            assert d.plotCanvas.has_finite_data(), f"view {view!r} empty"
    assert combo.count() == 12


def test_cursor_readout_live(window: MainWindow, success_result) -> None:
    d = window.direct_tab
    d.viewCombo.setCurrentIndex(1)  # jacket diameter D
    d.plotCanvas.draw_axial_view(success_result, d.viewCombo.currentText())
    event = SimpleNamespace(
        inaxes=d.plotCanvas.ax, xdata=42.0, ydata=8.2
    )
    d.plotCanvas._on_motion(event)
    assert "42" in d.plotCanvas.cursor_label.text()
    assert "8.2" in d.plotCanvas.cursor_label.text()
    outside = SimpleNamespace(inaxes=None, xdata=None, ydata=None)
    d.plotCanvas._on_motion(outside)  # must not raise


def test_zmslider_updates_cross_section(
    window: MainWindow, success_result, qapp
) -> None:
    d = window.direct_tab
    d.set_result(success_result)
    d.viewCombo.setCurrentText(SCHEMATIC_VIEW)
    xs = d.crossSection
    z_end_mm = success_result.z_end * 1e3
    xs.zmSlider.setValue(0)
    text0 = xs.z_label.text()
    assert text0 == "z = 0.00 mm"
    xs.zmSlider.setValue(500)
    qapp.processEvents()
    assert xs.z_label.text() != text0
    xs.zmSlider.setValue(1000)
    qapp.processEvents()
    z_mm = float(xs.z_label.text().split("=")[1].replace("mm", "").strip())
    assert z_mm == pytest.approx(z_end_mm, abs=0.5)
    assert xs.zmSlider.maximum() == 1000


def test_partial_failure_trajectory_and_banner(window: MainWindow) -> None:
    d = window.direct_tab
    bad = default_simulation_inputs().with_process(Pcore=1e6)
    result = run_direct_simulation(bad)
    assert result.status != SimulationStatus.SUCCESS
    assert 0.0 < result.z_end < result.inputs.L
    d.set_result(result)
    assert d.banner_visible()
    text = d.diag_banner.text()
    assert "z =" in text and "mm" in text
    assert "Advice" in text
    # Partial curves still render in every axial view.
    d.viewCombo.setCurrentIndex(1)
    assert d.plotCanvas.has_finite_data()
    d.viewCombo.setCurrentText(SCHEMATIC_VIEW)
    z_mm = float(
        d.crossSection.z_label.text().split("=")[1].replace("mm", "").strip()
    )
    assert 0.0 <= z_mm <= result.z_end * 1e3 + 1e-6


def test_post_run_contact_badges(window: MainWindow, success_result) -> None:
    d = window.direct_tab
    d.set_result(success_result)
    assert d.maxContactBadge.is_ok is True
    assert d.maxContactBadge.ratio is not None


def test_jacket_property_preview(window: MainWindow) -> None:
    j = window.jacket_tab
    j.propertyCombo.setCurrentText("Viscosity (log)")
    j.refresh_preview()
    assert j.plotCanvas.has_finite_data()
    j.viscosityEdit.setText("bad((")
    j.refresh_preview()
    assert not j.plotCanvas.has_finite_data()
    j.viscosityEdit.setText("10^(-7.946)*exp(66337/T)")


def test_furnace_profile_preview(window: MainWindow) -> None:
    window.jacket_tab.plot_furnace_profile(default_simulation_inputs())
    assert window.jacket_tab.plotCanvas.has_finite_data()


# ---------------------------------------------------------------------------
# A3: workers, cancellation, apply-to-direct
# ---------------------------------------------------------------------------


def test_simulation_worker_runs_async(qapp, window: MainWindow) -> None:
    inputs = window.build_inputs()
    worker = SimulationWorker(inputs)
    results: list = []
    worker.result_ready.connect(results.append)
    t0 = time.perf_counter()
    worker.start()
    deadline = time.perf_counter() + 60.0
    while not results and time.perf_counter() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    worker.wait(5000)
    assert results, "SimulationWorker never emitted a result"
    result = results[0]
    assert result.status == SimulationStatus.SUCCESS
    assert result.finite()
    assert time.perf_counter() - t0 < 60.0


def test_simulation_worker_cancel_within_half_second(qapp) -> None:
    worker = SimulationWorker(default_simulation_inputs())
    results: list = []
    worker.result_ready.connect(results.append)
    worker.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    worker.cancel()
    finished = worker.wait(5000)
    elapsed = time.perf_counter() - t0
    assert finished, "worker did not finish after cancel"
    assert elapsed < 0.5, f"cancel took {elapsed:.3f} s"
    if results:
        assert results[0].status == SimulationStatus.CANCELLED


def test_optimization_worker_cancel_within_half_second(qapp) -> None:
    worker = OptimizationWorker(
        default_simulation_inputs(), default_optimization_inputs()
    )
    worker.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    worker.cancel()
    assert worker.wait(10000), "optimization worker did not finish"
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"cancel took {elapsed:.3f} s"


def test_run_direct_button_async(qapp, window: MainWindow) -> None:
    window.direct_tab.runButton.click()
    result = _wait_for_result(qapp, window.direct_tab)
    assert result.ok
    assert window.direct_tab.result is result
    assert not window.direct_tab.is_running()


def test_apply_to_direct_computation(qapp, window: MainWindow) -> None:
    o, d = window.optimization_tab, window.direct_tab
    solution = OptimizationResult(
        status=SimulationStatus.SUCCESS,
        targets=None,
        Tmax_C=1810.25,
        Pcore=1637.9,
        Pcap=(6422.9, 7100.0, 7300.0),
    )
    o.set_solution(solution)
    assert o.applyButton.isEnabled()
    o.applyButton.click()
    qapp.processEvents()
    assert d.TmaxEdit.text() == "1810.25"
    assert float(d.PcoreEdit.text()) == pytest.approx(16.379, abs=1e-3)
    assert float(d.pcapEdits[0].text()) == pytest.approx(64.229, abs=1e-3)
    assert float(d.pcapEdits[1].text()) == pytest.approx(71.0, abs=1e-3)
    assert float(d.pcapEdits[2].text()) == pytest.approx(73.0, abs=1e-3)
    assert window.tabs.currentWidget() is d


def test_derived_target_preview_and_warning(window: MainWindow) -> None:
    o = window.optimization_tab
    o.update_preview()
    text = o.preview_label.text()
    assert "wf" in text and "Rcf1" in text
    o.tfEdit.setText("200")  # tf > Rcf -> impossible targets
    o.update_preview()
    assert o.warn_banner.isVisibleTo(o)
    assert not o.applyButton.isEnabled()
    o.tfEdit.setText("3.5")


def test_optimization_failure_banner(window: MainWindow) -> None:
    o = window.optimization_tab
    bad = OptimizationInputs(
        tf=um_to_m(200.0),  # impossible: tf >> Dcf1
        Dcf1=um_to_m(110.0),
        Dcore=um_to_m(156.0),
    )
    result = run_optimization(default_simulation_inputs(), bad)
    assert result.status == SimulationStatus.INFEASIBLE
    o.set_solution(result)
    assert o.warn_banner.isVisibleTo(o)
    assert not o.applyButton.isEnabled()


# ---------------------------------------------------------------------------
# A4: JSON / CSV / TXT round-trips, entry point
# ---------------------------------------------------------------------------


def test_config_json_round_trip(window: MainWindow, tmp_path) -> None:
    path = tmp_path / "config.json"
    window.direct_tab.TmaxEdit.setText("1799.5")
    window.capillaries_tab.nestedSpinBox.setValue(2)
    window.save_config(path)
    payload = json.loads(path.read_text())
    assert payload["simulation"]["Tmax_C"] == pytest.approx(1799.5)
    assert len(payload["simulation"]["rings"]) == 2
    # Change the GUI, then load back.
    window.direct_tab.TmaxEdit.setText("1234")
    window.capillaries_tab.nestedSpinBox.setValue(1)
    window.load_config(path)
    assert window.direct_tab.TmaxEdit.text() == "1799.5"
    assert window.capillaries_tab.nestedSpinBox.value() == 2


def test_config_dialog_round_trip(
    window: MainWindow, tmp_path, monkeypatch
) -> None:
    import app.main_window as mw

    path = str(tmp_path / "dialog_config.json")
    monkeypatch.setattr(
        mw.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (path, "HARF config (*.json)")),
    )
    monkeypatch.setattr(
        mw.QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (path, "HARF config (*.json)")),
    )
    window.direct_tab.PcoreEdit.setText("17.3")
    window.save_action.trigger()
    window.direct_tab.PcoreEdit.setText("1.0")
    window.load_action.trigger()
    assert window.direct_tab.PcoreEdit.text() == "17.3"


def test_export_csv_txt_via_dialog(
    window: MainWindow, tmp_path, monkeypatch, success_result
) -> None:
    import app.tabs.tab_direct as td

    csv_path = str(tmp_path / "out.csv")
    txt_path = str(tmp_path / "out.txt")

    def fake_save(parent, title, default, flt):
        return (csv_path if default.endswith(".csv") else txt_path, flt)

    monkeypatch.setattr(td.QFileDialog, "getSaveFileName", staticmethod(fake_save))
    window.direct_tab.set_result(success_result)
    window.direct_tab.saveCsvButton.click()
    window.direct_tab.saveTxtButton.click()
    meta, names, cols = read_results_table(csv_path)
    assert "z_m" in names and "D_um" in names
    assert len(cols["z_m"]) == len(success_result.z)
    assert meta["status"] == "success"
    meta2, names2, cols2 = read_results_table(txt_path)
    assert names2 == names
    assert len(cols2["z_m"]) == len(cols["z_m"])


def test_export_without_result_is_graceful(window: MainWindow) -> None:
    window.direct_tab._result = None
    window.direct_tab.saveCsvButton.click()
    assert "Nothing to export" in window.direct_tab.status_text()


def test_close_event_autosaves_config(qapp, tmp_path) -> None:
    path = tmp_path / "autosave.json"
    win = MainWindow(config_path=path)
    win.direct_tab.TmaxEdit.setText("1800.4")
    win.close()
    payload = json.loads(path.read_text())
    assert payload["simulation"]["Tmax_C"] == pytest.approx(1800.4)
    # Loading that config restores the values.
    win2 = MainWindow(config_path=path)
    assert win2.direct_tab.TmaxEdit.text() == "1800.4"
    win2.close()


def test_main_help_exits_cleanly(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "harf-drawing" in capsys.readouterr().out


def test_cross_section_geometry_shape(window: MainWindow, success_result):
    """Ring-1 circles stay inside the jacket bore at every scrub point."""
    d = window.direct_tab
    d.set_result(success_result)
    d.viewCombo.setCurrentText(SCHEMATIC_VIEW)
    xs = d.crossSection
    for value in (0, 250, 500, 750, 1000):
        xs.zmSlider.setValue(value)
        idx = xs.current_index
        r = float(success_result.r[idx])
        rc1 = float(success_result.Rc[0][idx])
        assert rc1 < r
        assert (r - rc1) > 0.0
