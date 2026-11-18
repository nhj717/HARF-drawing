"""Main workstation window: 5-tab layout, menu/toolbar, config I/O.

``MainWindow`` wires the five tabs together, owns the worker lifecycle
(asynchronous runs with cooperative cancellation and a graceful
``closeEvent`` shutdown), and provides **Load Config** / **Save Config**
(``.json``) and **Reset Defaults** actions.  When a ``config_path`` is
supplied the current configuration is auto-saved there on close, so
tests never touch the user's home directory.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QLocale, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QStatusBar,
    QTabWidget,
)

from app.tabs.tab_capillaries import CapillariesTab
from app.tabs.tab_direct import DirectTab
from app.tabs.tab_jacket import JacketTab
from app.tabs.tab_optimization import OptimizationTab
from app.tabs.tab_parameters import ParameterTab
from app.theme import build_cyberpunk_qss
from app.workers import OptimizationWorker, SimulationWorker
from src.io_utils import (
    load_config_json,
    read_ui_state,
    save_config_json,
)
from src.models import (
    OptimizationResult,
    SimulationInputs,
    default_optimization_inputs,
    default_simulation_inputs,
)

__all__ = ["MainWindow"]

_GITHUB_URL = "https://github.com/nhj717/HARF-drawing"
_LOGO_PATH = Path("reference/Logo.png")
_GITHUB_ICON_PATH = Path("reference/github-mark.png")


class MainWindow(QMainWindow):
    """The HARF Drawing Simulation Workstation main window."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__()
        # Force the C locale ('.' decimal separator) on every widget of
        # this window, even when the OS runs e.g. German (de_DE).
        QLocale.setDefault(QLocale.c())
        self.setWindowTitle("HARF Drawing Simulation Workstation")
        self.resize(1280, 800)
        self.setStyleSheet(build_cyberpunk_qss())
        self.config_path = Path(config_path) if config_path else None

        self.parameter_tab = ParameterTab()
        self.jacket_tab = JacketTab()
        self.capillaries_tab = CapillariesTab()
        self.direct_tab = DirectTab()
        self.optimization_tab = OptimizationTab()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.parameter_tab, "Parameters")
        self.tabs.addTab(self.jacket_tab, "Jacket")
        self.tabs.addTab(self.capillaries_tab, "Capillaries")
        self.tabs.addTab(self.direct_tab, "Direct computation")
        self.tabs.addTab(self.optimization_tab, "Optimization")
        self.setCentralWidget(self.tabs)

        self._direct_worker: SimulationWorker | None = None
        self._optimization_worker: OptimizationWorker | None = None

        self._build_menu()
        self._build_statusbar()
        self._wire_tabs()
        self._apply_icons()

        if self.config_path is not None and self.config_path.exists():
            try:
                self.load_config(self.config_path)
            except Exception:  # noqa: BLE001 - keep the GUI alive
                self.statusBar().showMessage(
                    f"Could not load config {self.config_path}; "
                    "defaults kept."
                )

    # -- UI scaffolding ------------------------------------------------------
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self.load_action = QAction("&Load Config...", self)
        self.load_action.setShortcut("Ctrl+O")
        self.load_action.triggered.connect(self._on_load_config)
        file_menu.addAction(self.load_action)

        self.save_action = QAction("&Save Config...", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self._on_save_config)
        file_menu.addAction(self.save_action)

        file_menu.addSeparator()
        self.reset_action = QAction("&Reset Defaults", self)
        self.reset_action.triggered.connect(self.reset_defaults)
        file_menu.addAction(self.reset_action)

        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addAction(self.load_action)
        toolbar.addAction(self.save_action)
        toolbar.addAction(self.reset_action)

        if _GITHUB_ICON_PATH.exists():
            self.github_action = QAction(
                QIcon(str(_GITHUB_ICON_PATH)), "Open GitHub page", self
            )
            self.github_action.triggered.connect(
                lambda: QDesktopServices.openUrl(QUrl(_GITHUB_URL))
            )
            toolbar.addAction(self.github_action)

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage("Ready.")

    def _apply_icons(self) -> None:
        if _LOGO_PATH.exists():
            self.setWindowIcon(QIcon(QPixmap(str(_LOGO_PATH))))

    def _wire_tabs(self) -> None:
        self.parameter_tab.inputs_changed.connect(
            self._refresh_cross_tab_state
        )
        self.jacket_tab.geometry_changed.connect(
            self._refresh_cross_tab_state
        )
        self.jacket_tab.expressions_changed.connect(
            self._refresh_cross_tab_state
        )
        self.capillaries_tab.inputs_changed.connect(
            self._refresh_cross_tab_state
        )
        self.direct_tab.inputs_changed.connect(self._refresh_cross_tab_state)
        self.direct_tab.run_requested.connect(self.run_direct)
        self.direct_tab.result_ready.connect(self._on_direct_result)
        self.optimization_tab.run_requested.connect(self.run_optimization)
        self.optimization_tab.apply_requested.connect(self._on_apply_solution)
        self.optimization_tab.set_inputs_provider(self.build_inputs)
        self._refresh_cross_tab_state()

    # -- cross-tab state --------------------------------------------------------
    def _refresh_cross_tab_state(self) -> None:
        """Propagate every tab edit into all live plots and badges."""
        r0, _ = self.jacket_tab.geometry()
        self.capillaries_tab.update_contact_badge(
            r0, int(self.capillaries_tab.ncapSpinBox.value())
        )
        nb = self.capillaries_tab.nb_nested
        self.direct_tab.set_nb_nested(nb)
        self.optimization_tab.set_nb_nested(nb)
        try:
            inputs = self.build_inputs()
        except (TypeError, ValueError):  # invalid intermediate state
            self.optimization_tab.update_preview()
            return
        self.parameter_tab.refresh_furnace_profile(
            inputs, tmax_c=inputs.Tmax_C
        )
        self.jacket_tab.refresh_material_comparison(
            self.capillaries_tab.material_exprs()
        )
        self.capillaries_tab.refresh_material_plot(
            self.jacket_tab.material_exprs()
        )
        self.direct_tab.refresh_pre_run_profile(inputs)
        self.optimization_tab.refresh_pre_run_profile(inputs)
        self.optimization_tab.update_preview(inputs)

    # -- input assembly -----------------------------------------------------------
    def build_inputs(self) -> SimulationInputs:
        """Merge every tab into one validated-later SimulationInputs."""
        fields: dict = {}
        fields.update(self.parameter_tab.to_fields())
        fields.update(self.jacket_tab.to_fields())
        fields.update(self.capillaries_tab.to_fields())
        nb_nested = self.capillaries_tab.nb_nested
        fields.update(self.direct_tab.to_fields(nb_nested))
        return SimulationInputs(**fields)

    # -- worker orchestration ------------------------------------------------------
    def run_direct(self) -> None:
        if self._direct_worker is not None and self._direct_worker.isRunning():
            self.statusBar().showMessage("A computation is already running.")
            return
        inputs = self.build_inputs()
        self._direct_worker = SimulationWorker(inputs, parent=self)
        self.direct_tab.attach_worker(self._direct_worker)
        self.statusBar().showMessage("Direct computation running...")
        self._autosave()

    def run_optimization(self) -> None:
        if (
            self._optimization_worker is not None
            and self._optimization_worker.isRunning()
        ):
            self.statusBar().showMessage("An optimization is already running.")
            return
        inputs = self.build_inputs()
        opt = self.optimization_tab.to_inputs(inputs.nb_nested)
        self._optimization_worker = OptimizationWorker(
            inputs, opt, parent=self
        )
        self.optimization_tab.attach_worker(self._optimization_worker)
        self.statusBar().showMessage("Optimization running...")
        self._autosave()

    def _on_direct_result(self, result) -> None:
        if result is not None and result.ok:
            self.statusBar().showMessage(
                f"Direct computation finished ({result.elapsed_s:.1f} s)."
            )

    def _on_apply_solution(self, solution: OptimizationResult) -> None:
        if solution is None or not solution.ok:
            return
        self.direct_tab.apply_process(
            solution.Tmax_C, solution.Pcore, solution.Pcap
        )
        if solution.result is not None:
            self.direct_tab.set_result(solution.result)
        self.tabs.setCurrentWidget(self.direct_tab)
        self.statusBar().showMessage(
            "Solved parameters applied to Direct Computation."
        )

    # -- config I/O --------------------------------------------------------------------
    def _on_load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load config", "", "HARF config (*.json)"
        )
        if path:
            self.load_config(path)

    def _on_save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", "harf_config.json", "HARF config (*.json)"
        )
        if path:
            self.save_config(path)
            self._autosave()

    def _collect_ui_state(self) -> dict:
        """GUI-only values the dataclasses cannot round-trip.

        Captures every ring-aware text field (Capillaries ring
        geometry/laws, Direct ``Pcap i`` and Optimization ``Pcap i``
        bounds) for ALL 3 rings, so values typed while a ring is
        inactive survive closing, reopening and ``Nnested`` switches.
        """
        c, d, o = (
            self.capillaries_tab, self.direct_tab, self.optimization_tab
        )
        return {
            "ring_dc": [e.text() for e in c.dcEdits],
            "ring_dcc": [e.text() for e in c.dccEdits],
            "ring_gamma": [e.text() for e in c.gammaEdits],
            "ring_visc": [e.text() for e in c.viscEdits],
            "direct_pcap": [e.text() for e in d.pcapEdits],
            "optim_pcap_min": [e.text() for e in o.pcapMinEdits],
            "optim_pcap_max": [e.text() for e in o.pcapMaxEdits],
        }

    def _apply_ui_state(self, state: dict) -> None:
        """Restore ring-aware text fields captured by ``_collect_ui_state``."""
        if not state:
            return
        c, d, o = (
            self.capillaries_tab, self.direct_tab, self.optimization_tab
        )
        for key, edits in (
            ("ring_dc", c.dcEdits),
            ("ring_dcc", c.dccEdits),
            ("ring_gamma", c.gammaEdits),
            ("ring_visc", c.viscEdits),
            ("direct_pcap", d.pcapEdits),
            ("optim_pcap_min", o.pcapMinEdits),
            ("optim_pcap_max", o.pcapMaxEdits),
        ):
            for i, text in enumerate(state.get(key, [])[: len(edits)]):
                if isinstance(text, str):
                    edits[i].setText(text)

    def load_config(self, path: str | Path) -> None:
        inputs, opt = load_config_json(path)
        self.parameter_tab.apply_inputs(inputs)
        self.jacket_tab.apply_inputs(inputs)
        self.capillaries_tab.apply_inputs(inputs)
        self.direct_tab.apply_inputs(inputs)
        if opt is not None:
            self.optimization_tab.apply_inputs(inputs, opt)
        self._apply_ui_state(read_ui_state(path))
        self._refresh_cross_tab_state()
        self.statusBar().showMessage(f"Config loaded from {path}.")

    def save_config(self, path: str | Path) -> None:
        inputs = self.build_inputs()
        opt = self.optimization_tab.to_inputs(inputs.nb_nested)
        save_config_json(
            inputs, opt, path, ui_state=self._collect_ui_state()
        )
        self.statusBar().showMessage(f"Config saved to {path}.")

    def _autosave(self) -> None:
        """Silently persist the session to the startup config path."""
        if self.config_path is None:
            return
        try:
            inputs = self.build_inputs()
            opt = self.optimization_tab.to_inputs(inputs.nb_nested)
            save_config_json(
                inputs,
                opt,
                self.config_path,
                ui_state=self._collect_ui_state(),
            )
        except Exception:  # noqa: BLE001 - persistence must never crash
            pass

    def reset_defaults(self) -> None:
        nb = self.capillaries_tab.nb_nested
        inputs = default_simulation_inputs(nb)
        opt = default_optimization_inputs(nb)
        self.parameter_tab.apply_inputs(inputs)
        self.jacket_tab.apply_inputs(inputs)
        self.capillaries_tab.apply_inputs(inputs)
        self.direct_tab.apply_inputs(inputs)
        self.optimization_tab.apply_inputs(inputs, opt)
        self._refresh_cross_tab_state()
        self._autosave()
        self.statusBar().showMessage("Defaults restored.")

    # -- shutdown -------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        for worker in (self._direct_worker, self._optimization_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(5000)
        self._autosave()
        super().closeEvent(event)
