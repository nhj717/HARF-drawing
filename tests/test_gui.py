"""Headless offscreen Qt GUI tests for the 6-tab workstation (A1-A4, A8).

Run with ``QT_QPA_PLATFORM=offscreen uv run pytest tests/test_gui.py``.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import erf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QLocale, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QScrollArea,
    QSplitter,
)

from app.dialogs.sem_inspector import SemApplyPayload, SemInspectorDialog
from app.main import main
from app.main_window import MainWindow
from app.tabs.tab_scaling import (
    MODE_BACKCALC,
    SUBMODE_FIXED_JACKET,
    ScalingTab,
)
from app.theme import (
    build_cyberpunk_qss,
    mono_font_family,
)
from app.widgets.cross_section_canvas import (
    compute_capillary_centers,
)
from app.widgets.plot_canvas import (
    SCHEMATIC_VIEW,
    TEMPERATURE_VIEW,
    TENSION_VIEW,
)
from app.widgets.validators import read_double
from app.workers import OptimizationWorker, SimulationWorker
from src.io_utils import read_results_table
from src.models import (
    DEFAULT_RHO,
    OptimizationInputs,
    OptimizationResult,
    SimulationResult,
    SimulationStatus,
    default_optimization_inputs,
    default_simulation_inputs,
    m_per_s_to_m_per_min,
    mbar_to_pa,
    um_to_m,
)
from src.scaling import compute_forward_scaling, get_scaling_presets
from src.sem_metrology import ScaleCalibration
from src.solvers import run_direct_simulation, run_optimization


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
        "Optimization", "Cane & Stack",
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
    assert o.pcapMinEdits[0].text() == "50"
    assert o.pcapMaxEdits[0].text() == "150"
    assert o.pcapMinEdits[1].text() == "50"
    assert o.pcapMaxEdits[2].text() == "150"


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
    j.propertyCombo.setCurrentText("Viscosity")
    assert j.plotCanvas.has_finite_data()
    # Multi-curve overlay: cyan Jacket + pink Cap #1.
    assert len(j.plotCanvas.ax.get_lines()) == 2
    j.viscosityEdit.setText("bad((")
    # Only the (still valid) Cap #1 curve remains.
    assert len(j.plotCanvas.ax.get_lines()) == 1
    j.viscosityEdit.setText("10^(-7.946)*exp(66337/T)")
    assert len(j.plotCanvas.ax.get_lines()) == 2
    # Single-curve property with a broken formula -> empty axes.
    j.propertyCombo.setCurrentText("Volumic mass")
    j.rhoEdit.setText("bad((")
    assert not j.plotCanvas.has_finite_data()
    j.rhoEdit.setText(DEFAULT_RHO)
    assert j.plotCanvas.has_finite_data()


def test_furnace_profile_preview(window: MainWindow) -> None:
    p = window.parameter_tab
    p.refresh_furnace_profile()
    assert p.plotCanvas.has_finite_data()
    assert p.plotCanvas.ax.get_xlabel() == "z (m)"
    assert p.plotCanvas.ax.get_xlim()[1] == pytest.approx(0.065)


# ---------------------------------------------------------------------------
# A1+: Cyberpunk Neon theme, live startup plots, cross-tab refresh
# ---------------------------------------------------------------------------


def test_cyberpunk_theme_applied(window: MainWindow) -> None:
    from matplotlib.colors import to_hex

    from app.theme import BG_DARK, CYAN, SURFACE_LOWEST

    qss = window.styleSheet()
    assert BG_DARK in qss
    assert CYAN in qss
    canvas = window.jacket_tab.plotCanvas
    assert to_hex(canvas.figure.patch.get_facecolor()) == BG_DARK
    assert to_hex(canvas.ax.get_facecolor()) == SURFACE_LOWEST
    for spine in canvas.ax.spines.values():
        assert to_hex(spine.get_edgecolor()) != "#000000"


def test_all_tabs_startup_live_plots(window: MainWindow) -> None:
    p = window.parameter_tab
    j = window.jacket_tab
    c = window.capillaries_tab
    d = window.direct_tab
    o = window.optimization_tab
    # Tab 1: normalized furnace profile on [0, Lfour].
    assert p.plotCanvas.has_finite_data()
    assert p.plotCanvas.ax.get_xlim()[1] == pytest.approx(0.065)
    # Tabs 2 + 3: material curves over [250, 2500] K, 250 K ticks.
    for tab in (j, c):
        assert tab.plotCanvas.has_finite_data()
        ax = tab.plotCanvas.ax
        expected = [float(t) for t in np.arange(250.0, 2501.0, 250.0)]
        assert [float(t) for t in ax.get_xticks()] == expected
        assert ax.get_xlim() == (250.0, 2500.0)
    # Tabs 4 + 5: pre-run furnace temperature profile by default.
    assert d.viewCombo.currentText() == TEMPERATURE_VIEW
    assert d.plotCanvas.has_finite_data()
    assert d.plotCanvas.ax.get_title() == "Furnace temperature profile"
    assert o.viewCombo.currentText() == TEMPERATURE_VIEW
    assert o.plotCanvas.has_finite_data()
    assert o.plotCanvas.ax.get_title() == "Furnace temperature profile"


def test_cross_tab_live_refresh(window: MainWindow) -> None:
    j = window.jacket_tab
    j.propertyCombo.setCurrentText("Viscosity")
    assert len(j.plotCanvas.ax.get_lines()) == 2  # Jacket + Cap #1
    # Nnested change adds a Cap #2 curve on the Jacket preview.
    window.capillaries_tab.nestedSpinBox.setValue(2)
    assert len(j.plotCanvas.ax.get_lines()) == 3
    # Lfour edit rescales the ParameterTab furnace profile.
    p = window.parameter_tab
    assert p.plotCanvas.ax.get_xlim()[1] == pytest.approx(0.065)
    p.LfourEdit.setText("0.05")
    assert p.plotCanvas.ax.get_xlim()[1] == pytest.approx(0.05)
    # Tmax edit rescales the DirectTab pre-run profile.
    d = window.direct_tab
    ys = d.plotCanvas.ax.get_lines()[0].get_ydata()
    assert float(np.max(ys)) == pytest.approx(1806.9, abs=1.0)
    d.TmaxEdit.setText("2000")
    ys = d.plotCanvas.ax.get_lines()[0].get_ydata()
    assert float(np.max(ys)) == pytest.approx(2000.0, abs=1.0)
    # Optimization pre-run profile follows too.
    o = window.optimization_tab
    ys = o.plotCanvas.ax.get_lines()[0].get_ydata()
    assert float(np.max(ys)) == pytest.approx(2000.0, abs=1.0)


# ---------------------------------------------------------------------------
# A2+: post-run KPI readouts, draw tension in g, optimization views
# ---------------------------------------------------------------------------


def test_direct_post_run_kpi_readouts(
    window: MainWindow, success_result: SimulationResult
) -> None:
    d = window.direct_tab
    d.set_result(success_result)
    assert d.finalOutputBox.isVisibleTo(d)
    g = float(success_result.inputs.g)
    assert float(d.kpiVf.value_text()) == pytest.approx(
        success_result.w[-1] * 60.0, rel=1e-3
    )
    assert float(d.kpiDf.value_text()) == pytest.approx(
        2.0 * success_result.R[-1] * 1e6, rel=1e-3
    )
    assert float(d.kpiDfBore.value_text()) == pytest.approx(
        2.0 * success_result.r[-1] * 1e6, rel=1e-3
    )
    assert float(d.kpiDcore.value_text()) == pytest.approx(
        success_result.D_core[-1] * 1e6, rel=1e-3
    )
    assert float(d.kpiDcf.value_text()) == pytest.approx(
        2.0 * success_result.Rc[0][-1] * 1e6, rel=1e-3
    )
    assert float(d.kpiTf.value_text()) == pytest.approx(
        (success_result.Rc[0][-1] - success_result.rc[0][-1]) * 1e6,
        rel=1e-3,
    )
    assert d.finalContactBadge.ratio is not None
    assert d.ringRatioBadges[0].ratio is not None
    # Draw tension is plotted in grams.
    d.viewCombo.setCurrentText(TENSION_VIEW)
    assert d.plotCanvas.has_finite_data()
    assert d.plotCanvas.ax.get_ylabel() == "Draw tension (g)"
    line = d.plotCanvas.ax.get_lines()[0]
    assert line.get_ydata()[-1] == pytest.approx(
        success_result.tau[-1] / g * 1000.0, rel=1e-6
    )
    # A failed run keeps the always-visible panel with placeholders.
    bad = run_direct_simulation(
        default_simulation_inputs().with_process(Pcore=1e6)
    )
    d.set_result(bad)
    assert d.finalOutputBox.isVisibleTo(d)
    assert d.kpiVf.value_text() == "--"
    assert d.finalContactBadge.ratio is None


def test_optimization_views_and_zmslider2(
    window: MainWindow, success_result: SimulationResult
) -> None:
    o = window.optimization_tab
    assert o.viewCombo.count() == 12
    assert o.viewCombo.currentText() == TEMPERATURE_VIEW
    solution = OptimizationResult(
        status=SimulationStatus.SUCCESS,
        targets=None,
        Tmax_C=1806.9,
        Pcore=1560.0,
        Pcap=(6340.0,),
        result=success_result,
        cost=0.0,
    )
    o.set_solution(solution)
    assert o.plotCanvas.has_finite_data()
    for i in range(o.viewCombo.count()):
        o.viewCombo.setCurrentIndex(i)
        view = o.viewCombo.currentText()
        if view == SCHEMATIC_VIEW:
            assert o.view_stack.currentIndex() == 1
            assert o.crossSection.z_label.text() != "z = -- mm"
        else:
            assert o.view_stack.currentIndex() == 0
            assert o.plotCanvas.has_finite_data(), f"view {view!r} empty"
    assert o.zmSlider_2 is o.crossSection.zmSlider
    o.viewCombo.setCurrentText(SCHEMATIC_VIEW)
    o.zmSlider_2.setValue(1000)
    assert o.crossSection.z_label.text() != "z = -- mm"


def test_optimization_per_ring_pcap_windows(window: MainWindow) -> None:
    o = window.optimization_tab
    assert o.pcapMinEdits[0].isEnabled()
    assert not o.pcapMinEdits[1].isEnabled()
    assert not o.pcapMaxEdits[2].isEnabled()
    o.pcapMinEdits[1].setText("80")
    o.pcapMaxEdits[1].setText("200")
    o.pcapMinEdits[2].setText("120")
    o.pcapMaxEdits[2].setText("300")
    window.capillaries_tab.nestedSpinBox.setValue(3)
    assert o.pcapMinEdits[1].isEnabled()
    assert o.pcapMaxEdits[2].isEnabled()
    opt = o.to_inputs(3)
    assert opt.Pcap_min == (
        mbar_to_pa(50.0), mbar_to_pa(80.0), mbar_to_pa(120.0),
    )
    assert opt.Pcap_max[2] == pytest.approx(mbar_to_pa(300.0))
    window.capillaries_tab.nestedSpinBox.setValue(1)
    assert not o.pcapMinEdits[2].isEnabled()
    assert len(o.to_inputs(1).Pcap_min) == 1


def test_optimization_live_kpi_preview(window: MainWindow) -> None:
    o = window.optimization_tab
    assert o.kpiVf.value_text() != "--"
    assert o.boreRatioBadge.ratio is not None
    assert o.contactBadge.ratio is not None
    # Impossible targets flag the warning banner and clear the cards.
    o.tfEdit.setText("200")
    assert o.warn_banner.isVisibleTo(o)
    assert o.kpiVf.value_text() == "--"
    o.tfEdit.setText("3.5")
    assert not o.warn_banner.isVisibleTo(o)
    assert o.kpiVf.value_text() != "--"


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
    _meta2, names2, cols2 = read_results_table(txt_path)
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


# ---------------------------------------------------------------------------
# A5: oven thermal profile regressions (units, validators, view gating,
# layout stretch) — deep offscreen pixel & keystroke audit fixes.
# ---------------------------------------------------------------------------


def _oven_curve(ax):
    """The rendered oven curve line (long, finite) of ``ax``."""
    lines = [
        line for line in ax.get_lines()
        if len(line.get_xdata()) > 200
        and np.isfinite(line.get_xdata()).all()
        and np.isfinite(line.get_ydata()).all()
    ]
    assert lines, "no oven curve rendered on the axes"
    return lines[0]


def _assert_curve_in_bounds(ax) -> None:
    """All points inside ``xlim`` (inclusive) and strictly inside ``ylim``."""
    line = _oven_curve(ax)
    x, y = line.get_xdata(), line.get_ydata()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert x0 <= x.min() < x.max() <= x1, "xdata escapes xlim"
    assert y0 < y.min() and y.max() < y1, "ydata not strictly inside ylim"
    assert x.size > 200, f"only {x.size} visible in-bounds points"


def test_tab1_oven_curve_units_and_bounds(window: MainWindow) -> None:
    """Tab 1 renders all 400 points in metres on ``[0, Lfour]``."""
    p = window.parameter_tab
    ax = p.plotCanvas.ax
    assert ax.get_xlabel() == "z (m)"
    assert ax.get_xlim() == (0.0, pytest.approx(0.065))
    line = _oven_curve(ax)
    assert len(line.get_xdata()) == 400
    assert float(np.max(line.get_xdata())) == pytest.approx(0.065)
    _assert_curve_in_bounds(ax)


def test_tab1_oven_curve_tracks_lfour_and_live_tmax(
    window: MainWindow,
) -> None:
    """Curve follows ``Lfour`` edits and the live ``Tmax`` from Tab 4."""
    p, d = window.parameter_tab, window.direct_tab
    p.LfourEdit.setText("0.08")
    assert p.plotCanvas.ax.get_xlim()[1] == pytest.approx(0.08)
    _assert_curve_in_bounds(p.plotCanvas.ax)
    assert float(
        np.max(_oven_curve(p.plotCanvas.ax).get_xdata())
    ) == pytest.approx(0.08)
    d.TmaxEdit.setText("2000")
    assert "2000.0" in p.plotCanvas.ax.get_title()
    _assert_curve_in_bounds(p.plotCanvas.ax)


def test_tabs45_pre_run_curve_bounds_and_view_gating(
    window: MainWindow,
) -> None:
    """Tabs 4-5 render the pre-run furnace curve only for the view."""
    d, o = window.direct_tab, window.optimization_tab
    for tab in (d, o):
        assert tab.viewCombo.currentText() == TEMPERATURE_VIEW
        _assert_curve_in_bounds(tab.plotCanvas.ax)
        # Un-run non-temperature views keep the empty-state prompt.
        tab.viewCombo.setCurrentText("Velocity V (m/min)")
        assert not tab.plotCanvas.has_finite_data()
        tab.viewCombo.setCurrentText(TEMPERATURE_VIEW)
        _assert_curve_in_bounds(tab.plotCanvas.ax)


def test_formula_typing_not_blocked_by_validators(
    window: MainWindow,
) -> None:
    """Real keystrokes type letters/operators; constants plot flat."""
    p = window.parameter_tab
    edit = p.furnaceProfileEdit
    assert edit.validator() is None  # no QDoubleValidator on formula box
    edit.clear()
    QTest.keyClicks(edit, "exp(-(z-0.03)^2/0.001)")
    assert edit.text() == "exp(-(z-0.03)^2/0.001)"
    p.refresh_furnace_profile()
    assert len(_oven_curve(p.plotCanvas.ax).get_xdata()) == 400
    _assert_curve_in_bounds(p.plotCanvas.ax)

    edit.clear()
    QTest.keyClicks(edit, "1")
    assert edit.text() == "1"
    p.refresh_furnace_profile()
    line = _oven_curve(p.plotCanvas.ax)
    y = np.asarray(line.get_ydata(), dtype=float)
    assert y.size == 400 and np.isfinite(y).all()
    assert np.allclose(y, 1.0)  # 0-d constant broadcast, no shape error
    _assert_curve_in_bounds(p.plotCanvas.ax)

    j = window.jacket_tab
    assert j.viscosityEdit.validator() is None
    j.viscosityEdit.clear()
    QTest.keyClicks(j.viscosityEdit, "10^(-7.946)*exp(66337/T)")
    assert j.viscosityEdit.text() == "10^(-7.946)*exp(66337/T)"
    assert j.viscosityEdit.property("formula_valid") is True


def test_invalid_furnace_profile_edit_is_caught(window: MainWindow) -> None:
    """Invalid/constant edits never abort the cross-tab refresh."""
    p = window.parameter_tab
    p.furnaceProfileEdit.setText("bad((")
    assert not p.plotCanvas.has_finite_data()
    p.furnaceProfileEdit.setText("0.95")
    assert p.plotCanvas.has_finite_data()
    assert np.allclose(
        _oven_curve(p.plotCanvas.ax).get_ydata(), 0.95
    )


def test_plot_canvas_fills_panel_and_window_fits_laptop(qapp) -> None:
    """Matplotlib canvas >= 70% of the panel; window fits 1280x800."""
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    qapp.processEvents()
    assert win.minimumSizeHint().height() <= 820
    for tab in (
        win.parameter_tab, win.direct_tab, win.optimization_tab,
    ):
        win.tabs.setCurrentWidget(tab)
        qapp.processEvents()
        pc = tab.plotCanvas
        assert pc.height() > 0
        assert pc.canvas.height() >= 0.7 * pc.height()
    win.close()


# ---------------------------------------------------------------------------
# A6: German-locale decimal fix, full Nnested gating, horizontal Direct
# splitter, and last-session persistence (ui_state round-trips).
# ---------------------------------------------------------------------------


def test_german_locale_forced_to_c(qapp) -> None:
    """A de_DE OS locale never leaks ',' into numeric fields."""
    german = QLocale(QLocale.Language.German, QLocale.Country.Germany)
    QLocale.setDefault(german)
    try:
        win = MainWindow()
        assert QLocale().language() == QLocale.Language.C
        assert QLocale().decimalPoint() == "."
        j, d, o = (
            win.jacket_tab, win.direct_tab, win.optimization_tab
        )
        assert j.d0Edit.text() == "2.4"  # '.', never ','
        assert d.TmaxEdit.text() == "1806.9"
        assert o.tfEdit.text() == "3.5"
        # Typing '.' decimals works via real keystrokes.
        j.d0Edit.clear()
        QTest.keyClicks(j.d0Edit, "2.65")
        qapp.processEvents()
        assert j.d0Edit.text() == "2.65"
        d.PcoreEdit.clear()
        QTest.keyClicks(d.PcoreEdit, "12.75")
        qapp.processEvents()
        assert d.PcoreEdit.text() == "12.75"
        win.close()
    finally:
        QLocale.setDefault(QLocale.system())


def test_nb_nested_gating_across_all_tabs(window: MainWindow) -> None:
    """Nnested blocks/enables Ring 2 + Ring 3 widgets on all 3 tabs."""
    c, d, o = (
        window.capillaries_tab, window.direct_tab,
        window.optimization_tab,
    )

    def assert_ring2(enabled: bool) -> None:
        # Capillaries: Ring-2 box + Dc02/dc01 badge.
        assert c.ring_boxes[1].isEnabled() is enabled
        assert c.nestedBadges[0].isEnabled() is enabled
        # Direct: Pcap2 input/label, per-ring badge, inter-ring badge.
        assert d.pcapEdits[1].isEnabled() is enabled
        assert d.pcapLabels[1].isEnabled() is enabled
        assert d.ringRatioBadges[1].isEnabled() is enabled
        assert d.maxNest2Badge.isEnabled() is enabled
        # Optimization: Pcap2 bounds + labels, badges, solved readout.
        assert o.pcapMinEdits[1].isEnabled() is enabled
        assert o.pcapMaxEdits[1].isEnabled() is enabled
        assert o.pcapMinLabels[1].isEnabled() is enabled
        assert o.pcapMaxLabels[1].isEnabled() is enabled
        assert o.nestedBadges[0].isEnabled() is enabled
        assert o.solvedPcapLabels[1].isEnabled() is enabled

    def assert_ring3(enabled: bool) -> None:
        assert c.ring_boxes[2].isEnabled() is enabled
        assert c.nestedBadges[1].isEnabled() is enabled
        assert d.pcapEdits[2].isEnabled() is enabled
        assert d.pcapLabels[2].isEnabled() is enabled
        assert d.ringRatioBadges[2].isEnabled() is enabled
        assert d.maxNest3Badge.isEnabled() is enabled
        assert o.pcapMinEdits[2].isEnabled() is enabled
        assert o.pcapMaxEdits[2].isEnabled() is enabled
        assert o.pcapMinLabels[2].isEnabled() is enabled
        assert o.pcapMaxLabels[2].isEnabled() is enabled
        assert o.nestedBadges[1].isEnabled() is enabled
        assert o.solvedPcapLabels[2].isEnabled() is enabled

    # Nnested = 1 (default): Ring 2 and Ring 3 fully blocked.
    assert_ring2(False)
    assert_ring3(False)
    # Nnested = 2: Ring 2 unlocked, Ring 3 still blocked.
    c.nestedSpinBox.setValue(2)
    assert_ring2(True)
    assert_ring3(False)
    # Nnested = 3: everything unlocked.
    c.nestedSpinBox.setValue(3)
    assert_ring2(True)
    assert_ring3(True)
    # Back to 1: blocked again.
    c.nestedSpinBox.setValue(1)
    assert_ring2(False)
    assert_ring3(False)


def test_direct_tab_horizontal_splitter(qapp) -> None:
    """Direct tab: controls left, plot/schematic right (>= 450 px tall)."""
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    qapp.processEvents()
    d = win.direct_tab
    win.tabs.setCurrentWidget(d)
    qapp.processEvents()
    splitter = d.splitter
    assert isinstance(splitter, QSplitter)
    assert splitter.orientation() == Qt.Orientation.Horizontal
    assert splitter.count() == 2
    # Left pane: scroll-wrapped controls hosting the process group box.
    assert isinstance(splitter.widget(0), QScrollArea)
    assert splitter.widget(0).widget().isAncestorOf(
        d.pcapEdits[0]
    )
    # Right pane: view dropdown + view stack (plot + cross-section).
    assert splitter.widget(1).isAncestorOf(d.view_stack)
    assert splitter.widget(1).isAncestorOf(d.plotCanvas)
    assert splitter.sizes()[1] > splitter.sizes()[0]
    assert d.plotCanvas.canvas.height() >= 450
    win.close()


def test_main_defaults_to_last_session_config(tmp_path, monkeypatch) -> None:
    """``uv run harf-drawing`` reuses the user's default config path."""
    import app.main as m

    session = tmp_path / "session.json"
    monkeypatch.setattr(
        m, "default_user_config_path", lambda: session
    )
    assert m._resolve_config_path(m._parse_args([])) == session
    explicit = tmp_path / "explicit.json"
    args = m._parse_args(["--config", str(explicit)])
    assert m._resolve_config_path(args) == explicit


def test_close_reopen_session_persistence_full_restore(
    qapp, tmp_path
) -> None:
    """Closing + reopening restores all 5 tabs incl. inactive rings."""
    path = tmp_path / "fresh" / "deep" / "session.json"
    win = MainWindow(config_path=path)
    # Parameters / Jacket / Capillaries (incl. inactive Ring 2 edits).
    win.parameter_tab.drawTensionEdit.setText("90")
    win.jacket_tab.d0Edit.setText("2.6")
    win.capillaries_tab.dcEdits[1].setText("0.36")
    win.capillaries_tab.dccEdits[1].setText("0.30")
    # Direct: process values incl. the inactive Ring-2 Pcap.
    win.direct_tab.TmaxEdit.setText("1799.5")
    win.direct_tab.pcapEdits[1].setText("77.7")
    # Optimization: every target and bound, incl. inactive rings.
    o = win.optimization_tab
    o.tfEdit.setText("3.8")
    o.Dcf1Edit.setText("115")
    o.DcoreEdit.setText("160")
    o.tmaxMinEdit.setText("1700")
    o.tmaxMaxEdit.setText("1900")
    o.pcoreMinEdit.setText("11")
    o.pcoreMaxEdit.setText("21")
    o.pcapMinEdits[0].setText("55")
    o.pcapMaxEdits[0].setText("160")
    o.pcapMinEdits[1].setText("60")
    o.pcapMaxEdits[1].setText("170")
    win.close()
    assert path.exists()  # parent dirs auto-created on first save
    payload = json.loads(path.read_text())
    assert payload["ui_state"]["direct_pcap"][1] == "77.7"
    assert payload["ui_state"]["optim_pcap_min"][1] == "60"

    win2 = MainWindow(config_path=path)
    assert win2.parameter_tab.drawTensionEdit.text() == "90"
    assert win2.jacket_tab.d0Edit.text() == "2.6"
    assert win2.direct_tab.TmaxEdit.text() == "1799.5"
    assert win2.direct_tab.pcapEdits[1].text() == "77.7"
    assert win2.capillaries_tab.dcEdits[1].text() == "0.36"
    assert win2.capillaries_tab.dccEdits[1].text() == "0.30"
    o2 = win2.optimization_tab
    assert o2.tfEdit.text() == "3.8"
    assert o2.Dcf1Edit.text() == "115"
    assert o2.DcoreEdit.text() == "160"
    assert o2.tmaxMinEdit.text() == "1700"
    assert o2.tmaxMaxEdit.text() == "1900"
    assert o2.pcoreMinEdit.text() == "11"
    assert o2.pcoreMaxEdit.text() == "21"
    assert o2.pcapMinEdits[0].text() == "55"
    assert o2.pcapMaxEdits[0].text() == "160"
    assert o2.pcapMinEdits[1].text() == "60"
    assert o2.pcapMaxEdits[1].text() == "170"
    # Switching Nnested keeps every typed ring value and unlocks it.
    win2.capillaries_tab.nestedSpinBox.setValue(2)
    assert win2.capillaries_tab.dcEdits[1].text() == "0.36"
    assert win2.capillaries_tab.dcEdits[1].isEnabled()
    assert o2.pcapMinEdits[1].isEnabled()
    win2.close()


def test_save_load_config_restores_ui_state(
    window: MainWindow, tmp_path
) -> None:
    """Save/Load Config round-trips inactive Ring 2/3 fields too."""
    path = tmp_path / "manual.json"
    window.capillaries_tab.dcEdits[2].setText("0.25")
    window.optimization_tab.pcapMaxEdits[2].setText("310")
    window.save_config(path)
    window.capillaries_tab.dcEdits[2].setText("0.99")
    window.optimization_tab.pcapMaxEdits[2].setText("999")
    window.load_config(path)
    assert window.capillaries_tab.dcEdits[2].text() == "0.25"
    assert window.optimization_tab.pcapMaxEdits[2].text() == "310"


# ---------------------------------------------------------------------------
# A7: Vf m/min units, closed-form contact badge, nested schematic,
#     compact final-outputs table, resolved fonts & fast tab switches.
# ---------------------------------------------------------------------------


def test_m_per_s_to_m_per_min_conversion() -> None:
    """1 m/s == 60 m/min; arrays pass through elementwise."""
    assert m_per_s_to_m_per_min(1.0 / 60.0) == pytest.approx(1.0)
    arr = np.array([1.0 / 60.0, 2.0 / 60.0])
    out = m_per_s_to_m_per_min(arr)
    assert isinstance(out, np.ndarray)
    assert out.shape == arr.shape
    assert out[1] == pytest.approx(2.0)


def test_feed_speed_w0_stays_mm_per_min(window: MainWindow) -> None:
    """w0 keeps the mm/min convention: 4 mm/min == 4/60000 m/s."""
    assert window.build_inputs().w0 == pytest.approx(4.0 / 60000.0)


def test_optimization_contact_badge_closed_form(
    window: MainWindow,
) -> None:
    """Default preview: Rcf1/Rcontact,f = 55/94; red from Ncap = 8."""
    o = window.optimization_tab
    o.update_preview()
    assert o.contactBadge.ratio == pytest.approx(55.0 / 94.0, rel=1e-6)
    assert o.contactBadge.is_ok is True
    window.capillaries_tab.ncapSpinBox.setValue(8)
    assert o.contactBadge.ratio is not None
    assert o.contactBadge.ratio >= 1.0
    assert o.contactBadge.is_ok is False
    window.capillaries_tab.ncapSpinBox.setValue(2)
    assert o.contactBadge.is_ok is True


def test_compute_capillary_centers_nested_rings() -> None:
    """3 nested rings x 6 azimuths -> [6, 6, 6] centers at 60/76/85."""
    rings = compute_capillary_centers(
        100.0, [40.0, 20.0, 8.0], [36.0, 17.0, 6.0], 6
    )
    assert [len(ring) for ring in rings] == [6, 6, 6]
    for ring, dist in zip(rings, [60.0, 76.0, 85.0], strict=True):
        for x, y in ring:
            assert float(np.hypot(x, y)) == pytest.approx(dist)
    # Exactly one capillary per azimuth per ring: theta_k = 2*pi*k/6.
    for ring in rings:
        angles = sorted(
            float(np.arctan2(y, x)) % (2.0 * np.pi) for x, y in ring
        )
        assert angles == pytest.approx(
            [2.0 * np.pi * k / 6.0 for k in range(6)]
        )


def test_solved_max_contact_badge(
    window: MainWindow, success_result: SimulationResult
) -> None:
    """set_solution fills solvedMaxContactBadge = max Rc1/Rcontact."""
    o = window.optimization_tab
    expected = float(np.max(success_result.Rc[0] / success_result.R_contact))
    assert success_result.max_rc1_over_rcontact == pytest.approx(expected)
    solution = OptimizationResult(
        status=SimulationStatus.SUCCESS,
        targets=None,
        Tmax_C=1806.9,
        Pcore=1560.0,
        Pcap=(6340.0,),
        result=success_result,
        cost=0.0,
    )
    o.set_solution(solution)
    assert o.solvedMaxContactBadge.ratio == pytest.approx(expected, rel=1e-6)
    if expected < 1.0:
        assert o.solvedMaxContactBadge.is_ok is True
    else:
        assert o.solvedMaxContactBadge.is_ok is False
    # Red styling contract: a ratio >= 1 flips the badge red.
    o.solvedMaxContactBadge.set_ratio(1.5, 1.0)
    assert o.solvedMaxContactBadge.is_ok is False
    # A failed optimization resets the badge to the placeholder state.
    bad = run_optimization(
        default_simulation_inputs(),
        OptimizationInputs(
            tf=um_to_m(200.0),  # impossible: tf >> Dcf1
            Dcf1=um_to_m(110.0),
            Dcore=um_to_m(156.0),
        ),
    )
    assert bad.status == SimulationStatus.INFEASIBLE
    o.set_solution(bad)
    assert o.solvedMaxContactBadge.ratio is None


def test_velocity_view_and_legend_in_m_per_min(
    window: MainWindow, success_result: SimulationResult
) -> None:
    """Velocity view + end-value legend report w*60 m/min."""
    d = window.direct_tab
    d.set_result(success_result)
    d.viewCombo.setCurrentText("Velocity V (m/min)")
    line = d.plotCanvas.ax.get_lines()[0]
    y = np.asarray(line.get_ydata(), dtype=float)
    assert y[-1] == pytest.approx(success_result.w[-1] * 60.0, rel=1e-6)
    assert d.plotCanvas.ax.get_ylabel() == "w (m/min)"
    legend_texts = [
        t.get_text() for t in d.plotCanvas.ax.get_legend().get_texts()
    ]
    vf_texts = [t for t in legend_texts if t.startswith("Vf =")]
    assert vf_texts, "no Vf end-value legend entry"
    assert "m/min" in vf_texts[0]
    assert float(vf_texts[0].split("=")[1].split()[0]) == pytest.approx(
        success_result.w[-1] * 60.0, abs=0.01
    )


def test_final_output_box_compact_and_always_visible(qapp) -> None:
    """Final-outputs table fits the left pane at 1280x800, pre-run too."""
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    qapp.processEvents()
    d = win.direct_tab
    win.tabs.setCurrentWidget(d)
    qapp.processEvents()
    # Visible pre-run with placeholder values (no run yet).
    assert d.finalOutputBox.isVisibleTo(d)
    assert d.kpiVf.value_text() == "--"
    assert d.finalContactBadge.ratio is None
    # Compact: fits the left scroll pane with zero horizontal overflow.
    scroll = d.splitter.widget(0)
    assert isinstance(scroll, QScrollArea)
    assert d.finalOutputBox.sizeHint().width() <= scroll.viewport().width()
    assert scroll.horizontalScrollBar().maximum() == 0
    win.close()


def test_tab_switch_latency_below_16ms(
    qapp, success_result: SimulationResult
) -> None:
    """Average tab-switch cost stays under 16 ms with results loaded."""
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    qapp.processEvents()
    d, o = win.direct_tab, win.optimization_tab
    d.set_result(success_result)
    d.viewCombo.setCurrentText(SCHEMATIC_VIEW)
    o.set_solution(OptimizationResult(
        status=SimulationStatus.SUCCESS,
        targets=None,
        Tmax_C=1806.9,
        Pcore=1560.0,
        Pcap=(6340.0,),
        result=success_result,
        cost=0.0,
    ))
    qapp.processEvents()
    # Warm up every tab once (deferred rasters, layout passes).
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        qapp.processEvents()
    times: list[float] = []
    for _ in range(3):
        for i in range(win.tabs.count()):
            t0 = time.perf_counter()
            win.tabs.setCurrentIndex(i)
            times.append(time.perf_counter() - t0)
            qapp.processEvents()
    avg = sum(times) / len(times)
    assert avg < 0.016, f"avg tab switch {avg * 1e3:.1f} ms"
    win.close()


def test_qss_font_declarations_resolved(qapp) -> None:
    """QSS font-family lists only resolved families (no bare monospace)."""
    from PySide6.QtGui import QFontDatabase

    installed = set(QFontDatabase.families())
    qss = build_cyberpunk_qss()
    declarations = re.findall(r"font-family:\s*([^;]+);", qss)
    assert declarations
    for decl in declarations:
        names = re.findall(r"'([^']+)'", decl)
        assert names, f"unresolved font-family: {decl!r}"
        for name in names:
            assert name in installed or name == "DejaVu Sans Mono"
            assert name.lower() not in ("monospace", "sans-serif", "serif")
    # The resolved mono family is a real installed family, never a generic.
    mono = mono_font_family()
    assert mono in installed or mono == "DejaVu Sans Mono"


def test_matplotlib_monospace_stack_resolves_silently() -> None:
    """The Matplotlib mono stack resolves with zero findfont warnings."""
    import logging

    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties, findfont

    from app.theme import configure_matplotlib_fonts

    configure_matplotlib_fonts()
    families = list(mpl.rcParams["font.monospace"])
    assert families[-1] == "DejaVu Sans Mono"
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Capture()
    logger = logging.getLogger("matplotlib.font_manager")
    logger.addHandler(handler)
    try:
        resolved = findfont(FontProperties(family=families))
    finally:
        logger.removeHandler(handler)
    assert not messages, f"findfont warnings: {messages}"
    assert FontProperties(fname=resolved).get_name() in families


# ---------------------------------------------------------------------------
# A8: cane & stack scaling tab
# ---------------------------------------------------------------------------


def test_scaling_tab_forward_matches_engine(window: MainWindow) -> None:
    """Preset-0 forward run matches the engine stage-by-stage."""
    preset = get_scaling_presets()[0]
    res = compute_forward_scaling(preset.cane, preset.jacket, preset.factors)
    s = window.scaling_tab
    assert isinstance(s, ScalingTab)
    assert s._result is not None
    for name in ("E", "F", "G", "H", "I"):
        assert s._result.stages[name].zeta == pytest.approx(
            res.stages[name].zeta, rel=1e-9
        )
    zeta_row = next(
        i for i, (_label, attr) in enumerate(ScalingTab.STAGE_ROWS)
        if attr == "zeta"
    )
    assert float(s.stageTable.item(zeta_row, 1).text()) == pytest.approx(
        res.stages["E"].zeta * 1e3, rel=1e-6
    )
    assert s.allOkBadge.is_ok is True
    assert not s.caneCanvas.slider_row.isVisibleTo(s.caneCanvas)
    assert not s.fiberCanvas.slider_row.isVisibleTo(s.fiberCanvas)


def test_scaling_tab_back_calc_recovers_preset(qapp) -> None:
    """Back-calc with a fixed jacket recovers the preset-0 cane."""
    win = MainWindow()
    s = win.scaling_tab
    s.modeCombo.setCurrentText(MODE_BACKCALC)
    s.subModeCombo.setCurrentText(SUBMODE_FIXED_JACKET)
    assert read_double(s.odcEdit) == pytest.approx(3.8, rel=1e-6)
    assert read_double(s.dcorecEdit) == pytest.approx(1.4, rel=1e-6)
    assert read_double(s.dcEdits[0]) == pytest.approx(0.45, rel=1e-6)
    assert read_double(s.tcEdits[0]) == pytest.approx(0.05, rel=1e-6)
    assert read_double(s.sjEdit) == pytest.approx(40.0, rel=1e-6)
    assert s._result is not None
    assert s._result.feasibility.all_ok
    win.close()


def test_scaling_tab_push_pull_round_trip(window: MainWindow) -> None:
    """Push writes the solver tabs; pull+push keeps D0/d0 stable."""
    s = window.scaling_tab
    j = window.jacket_tab
    c = window.capillaries_tab
    o = window.optimization_tab
    s.pushButton.click()
    handoff = s.build_handoff()
    assert read_double(j.D0Edit) == pytest.approx(handoff.D0 * 1e3, rel=1e-9)
    assert read_double(j.d0Edit) == pytest.approx(handoff.d0 * 1e3, rel=1e-9)
    assert c.ncapSpinBox.value() == handoff.n_cap
    assert c.nestedSpinBox.value() == len(handoff.D_c0)
    for i, (dc, dcc) in enumerate(
        zip(handoff.D_c0, handoff.d_c0, strict=True)
    ):
        assert read_double(c.dcEdits[i]) == pytest.approx(
            dc * 1e3, rel=1e-9
        )
        assert read_double(c.dccEdits[i]) == pytest.approx(
            dcc * 1e3, rel=1e-9
        )
    assert read_double(o.tfEdit) == pytest.approx(handoff.tf * 1e6, rel=1e-9)
    assert read_double(o.Dcf1Edit) == pytest.approx(
        handoff.Dcf1 * 1e6, rel=1e-9
    )
    assert read_double(o.DcoreEdit) == pytest.approx(
        handoff.Dcore * 1e6, rel=1e-9
    )
    d0_before = read_double(j.d0Edit)
    s.pullButton.click()
    s.pushButton.click()
    assert read_double(j.D0Edit) == pytest.approx(handoff.D0 * 1e3, rel=1e-6)
    assert read_double(j.d0Edit) == pytest.approx(d0_before, rel=1e-6)


def test_scaling_ui_state_round_trip(qapp, tmp_path) -> None:
    """Scaling-tab edits survive a config save / reload cycle."""
    cfg = tmp_path / "cfg.json"
    win = MainWindow(config_path=cfg)
    win.scaling_tab.sieEdit.setText("1.75")
    win.close()
    win2 = MainWindow(config_path=cfg)
    assert win2.scaling_tab.sieEdit.text() == "1.75"
    win2.close()


# ---------------------------------------------------------------------------
# A9: un-elided tab bar & SEM / microscope cross-section import
# ---------------------------------------------------------------------------


def test_tab_bar_never_elides(qapp, window) -> None:
    """All six tab labels stay fully visible, even at narrow widths."""
    tabs = window.tabs
    bar = tabs.tabBar()
    assert tabs.elideMode() == Qt.TextElideMode.ElideNone
    assert bar.elideMode() == Qt.TextElideMode.ElideNone
    assert not bar.expanding()
    fm = bar.fontMetrics()
    window.show()
    for width in (1280, 900, 760):
        window.resize(width, 800)
        qapp.processEvents()
        for j in range(tabs.count()):
            text = tabs.tabText(j)
            assert "…" not in text
            assert bar.tabRect(j).width() >= fm.horizontalAdvance(text)


def test_scaling_tab_sem_import_buttons(qapp, monkeypatch) -> None:
    """SEM import buttons open a file dialog; cancel leaves state alone."""
    win = MainWindow()
    s = win.scaling_tab
    before = s.odcEdit.text()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: ("", "")),
    )
    s.importCaneSemButton.click()
    s.importFiberSemButton.click()
    assert not s.statusLabel.text().startswith("Scaling error")
    assert s.odcEdit.text() == before
    win.close()


_SEM_N = 1024
_SEM_CENTER = 512.0
_SEM_BANNER_START = 905
_SEM_EDGE_SIGMA = 0.6


def _sem_inside(r: np.ndarray, r0: float) -> np.ndarray:
    """Smooth (erf) indicator of ``r < r0``."""
    s = _SEM_EDGE_SIGMA * math.sqrt(2.0)
    return 0.5 * (1.0 - erf((r - r0) / s))


def _sem_two_ring_image() -> np.ndarray:
    """Synthetic bright-silica 2-ring cane micrograph with banner."""
    rings = ((215.0, 85.0, 105.0), (245.0, 45.0, 55.0))
    n_cap, r_zeta, od = 6, 320.0, 365.0
    xx, yy = np.meshgrid(
        np.arange(_SEM_N, dtype=float), np.arange(_SEM_N, dtype=float)
    )
    r = np.hypot(xx - _SEM_CENTER, yy - _SEM_CENTER)
    comps = [_sem_inside(r, od) - _sem_inside(r, r_zeta)]
    rho1, _rc1, rcap1 = rings[0]
    comps.append(_sem_inside(r, rho1 - rcap1))
    for k in range(n_cap):
        th = k * 2.0 * math.pi / n_cap
        for rho, rc, rcap in rings:
            d = np.hypot(
                xx - (_SEM_CENTER + rho * math.cos(th)),
                yy - (_SEM_CENTER + rho * math.sin(th)),
            )
            comps.append(_sem_inside(d, rcap) - _sem_inside(d, rc))
    silica = np.ones((_SEM_N, _SEM_N))
    for comp in comps:
        silica *= 1.0 - comp
    silica = 1.0 - silica
    rng = np.random.default_rng(42)
    img = np.clip(silica + rng.normal(0.0, 0.04, (_SEM_N, _SEM_N)), 0.0, 1.0)
    img[_SEM_BANNER_START:, :] = np.clip(
        0.90 + rng.normal(0.0, 0.04, (_SEM_N - _SEM_BANNER_START, _SEM_N)),
        0.0,
        1.0,
    )
    return img


def test_sem_inspector_wall_only_payload(qapp) -> None:
    """Wall-only apply exports just the ring thicknesses in SI metres."""
    dlg = SemInspectorDialog(
        image=_sem_two_ring_image(),
        calibration=ScaleCalibration(2e-6),
        stage="cane",
    )
    dlg.nestedSpin.setValue(2)
    dlg.redetectButton.click()
    dlg.wallOnlyButton.click()
    p = dlg.payload()
    assert p.stage == "cane"
    assert p.od is None
    assert p.d_core is None
    assert p.n_cap is None
    assert p.ring_bore == ()
    assert len(p.ring_wall) == 2
    assert p.ring_wall[0] == pytest.approx(40e-6, abs=4e-6)
    assert p.ring_wall[1] == pytest.approx(20e-6, abs=3e-6)
    assert p.nested == 2
    dlg.close()


def test_scaling_tab_apply_sem_payload_selective(qapp) -> None:
    """A wall-only payload touches only the matching thickness edits."""
    win = MainWindow()
    s = win.scaling_tab
    s.unitCombo.setCurrentText("mm")
    s.odcEdit.setText("3.8")
    s.dcorecEdit.setText("1.4")
    for i, (d, t) in enumerate(
        ((0.085, 0.005), (0.22, 0.006), (0.16, 0.007))
    ):
        s.dcEdits[i].setText(f"{d}")
        s.tcEdits[i].setText(f"{t}")
    s.apply_sem_payload(
        SemApplyPayload(stage="cane", ring_wall=(40e-6, 20e-6), nested=2)
    )
    assert read_double(s.tcEdits[0]) == pytest.approx(0.04, rel=1e-6)
    assert read_double(s.tcEdits[1]) == pytest.approx(0.02, rel=1e-6)
    assert read_double(s.tcEdits[2]) == pytest.approx(0.007, rel=1e-9)
    assert read_double(s.odcEdit) == pytest.approx(3.8, rel=1e-9)
    assert read_double(s.dcorecEdit) == pytest.approx(1.4, rel=1e-9)
    assert read_double(s.dcEdits[0]) == pytest.approx(0.085, rel=1e-9)
    assert read_double(s.dcEdits[1]) == pytest.approx(0.22, rel=1e-9)

    s.unitCombo.setCurrentText("µm")
    s.dfEdit.setText("125")
    s.dcorefEdit.setText("35")
    for edit, value in zip(s.dfEdits, ("0.3", "0.2", "0.15"), strict=True):
        edit.setText(value)
    for edit, value in zip(s.tfEdits, ("0.3", "0.2", "0.15"), strict=True):
        edit.setText(value)
    s.apply_sem_payload(
        SemApplyPayload(
            stage="fiber", od=124.8e-6, ring_wall=(4e-6, 2e-6), nested=2
        )
    )
    assert read_double(s.tfEdits[0]) == pytest.approx(4.0, rel=1e-6)
    assert read_double(s.tfEdits[1]) == pytest.approx(2.0, rel=1e-6)
    assert read_double(s.tfEdits[2]) == pytest.approx(0.15, rel=1e-9)
    assert read_double(s.dfEdit) == pytest.approx(124.8, rel=1e-6)
    win.close()
