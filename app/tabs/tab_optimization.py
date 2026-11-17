"""Tab 5: optimization targets, windows, live KPI preview, views, apply.

Live derived-target previews follow the exact ``RunButton_2Pushed``
rules: KPI cards (``Vf``, ``Df``) and red/green ratio badges
(``df/Df``, ``dcf1/Dcf1``, ``Rcf1/Rcontact``, per-ring
``Dcf_i/dcf_i-1``) update on every edit; impossible targets show a
warning instead of crashing.  Distinct per-ring ``Pcap1..3`` search
windows are enabled by the nested-ring count.  The right column hosts
the 12-view combo (MATLAB ``UIAxes_2`` + ``zmSlider_2`` cross-section)
defaulting to the pre-run ``Temperature (°C)`` furnace profile, and
renders the optimized run after ``set_solution``.  *"Apply to Direct
Computation"* transfers the solved ``(Tmax, Pcore, Pcap1..3)``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.theme import RED, SURFACE_LOWEST, TEXT_MUTED, make_kpi_card
from app.widgets.cross_section_canvas import CrossSectionCanvas
from app.widgets.plot_canvas import (
    AXIAL_VIEWS,
    SCHEMATIC_VIEW,
    TEMPERATURE_VIEW,
    PlotCanvas,
)
from app.widgets.validators import RatioBadge, make_double_edit, read_double
from app.workers import OptimizationWorker
from src.io_utils import export_results_table
from src.models import (
    DEFAULT_OPTIM_DCF1_UM,
    DEFAULT_OPTIM_DCORE_UM,
    DEFAULT_OPTIM_PCAP_MAX_MBAR,
    DEFAULT_OPTIM_PCAP_MIN_MBAR,
    DEFAULT_OPTIM_PCORE_MAX_MBAR,
    DEFAULT_OPTIM_PCORE_MIN_MBAR,
    DEFAULT_OPTIM_TF_UM,
    DEFAULT_OPTIM_TMAX_MAX_C,
    DEFAULT_OPTIM_TMAX_MIN_C,
    MAX_NESTED_RINGS,
    DerivedTargets,
    InputError,
    OptimizationInputs,
    OptimizationResult,
    SimulationInputs,
    TargetError,
    compute_derived_targets,
    mbar_to_pa,
    r_contact_radius,
    um_to_m,
)

__all__ = ["OptimizationTab"]


class OptimizationTab(QWidget):
    """Optimization tab: targets, windows, preview, views, apply, export."""

    run_requested = Signal()
    apply_requested = Signal(object)  # OptimizationResult

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: OptimizationResult | None = None
        self._worker: OptimizationWorker | None = None
        self._inputs_provider = None
        self._last_inputs: SimulationInputs | None = None

        # -- final-fiber targets ------------------------------------------
        self.tfEdit = make_double_edit(
            DEFAULT_OPTIM_TF_UM, 0.0, 1e4,
            tooltip="Target capillary wall thickness tf [um]",
        )
        self.Dcf1Edit = make_double_edit(
            DEFAULT_OPTIM_DCF1_UM, 0.0, 1e4,
            tooltip="Target ring-1 outer diameter Dcf1 [um]",
        )
        self.DcoreEdit = make_double_edit(
            DEFAULT_OPTIM_DCORE_UM, 0.0, 1e4,
            tooltip="Target core diameter Dcore [um]",
        )
        targets_box = QGroupBox("Final-fiber targets")
        form = QFormLayout(targets_box)
        form.addRow("Wall thickness tf [um]", self.tfEdit)
        form.addRow("Ring-1 outer diameter Dcf1 [um]", self.Dcf1Edit)
        form.addRow("Core diameter Dcore [um]", self.DcoreEdit)

        # -- search windows (distinct per-ring Pcap pairs) -----------------
        self.tmaxMinEdit = make_double_edit(
            DEFAULT_OPTIM_TMAX_MIN_C, 0.0, 5000.0
        )
        self.tmaxMaxEdit = make_double_edit(
            DEFAULT_OPTIM_TMAX_MAX_C, 0.0, 5000.0
        )
        self.pcoreMinEdit = make_double_edit(
            DEFAULT_OPTIM_PCORE_MIN_MBAR, -1e5, 1e5
        )
        self.pcoreMaxEdit = make_double_edit(
            DEFAULT_OPTIM_PCORE_MAX_MBAR, -1e5, 1e5
        )
        self.pcapMinEdits = []
        self.pcapMaxEdits = []
        for i in range(MAX_NESTED_RINGS):
            self.pcapMinEdits.append(
                make_double_edit(
                    DEFAULT_OPTIM_PCAP_MIN_MBAR, -1e5, 1e5,
                    tooltip=f"Ring {i + 1} Pcap min [mbar]",
                )
            )
            self.pcapMaxEdits.append(
                make_double_edit(
                    DEFAULT_OPTIM_PCAP_MAX_MBAR, -1e5, 1e5,
                    tooltip=f"Ring {i + 1} Pcap max [mbar]",
                )
            )
        bounds_box = QGroupBox("Search windows")
        bform = QFormLayout(bounds_box)
        bform.addRow("Tmax min [C]", self.tmaxMinEdit)
        bform.addRow("Tmax max [C]", self.tmaxMaxEdit)
        bform.addRow("Pcore min [mbar]", self.pcoreMinEdit)
        bform.addRow("Pcore max [mbar]", self.pcoreMaxEdit)
        for i in range(MAX_NESTED_RINGS):
            bform.addRow(f"Pcap{i + 1} min [mbar]", self.pcapMinEdits[i])
            bform.addRow(f"Pcap{i + 1} max [mbar]", self.pcapMaxEdits[i])

        # -- live derived-target KPI preview --------------------------------
        self.kpiVf = make_kpi_card("Vf", unit="mm/min")
        self.kpiDf = make_kpi_card("Df", unit="um")
        kpi_row = QHBoxLayout()
        kpi_row.addWidget(self.kpiVf)
        kpi_row.addWidget(self.kpiDf)
        kpi_row.addStretch(1)

        self.boreRatioBadge = RatioBadge(
            "df / Df", "Final jacket bore/outer diameter ratio."
        )
        self.capRatioBadge = RatioBadge(
            "dcf1 / Dcf1", "Final ring-1 bore/outer diameter ratio."
        )
        self.contactBadge = RatioBadge(
            "Rcf1 / Rcontact",
            "Ring-1 final outer radius over the packing contact radius "
            "(red when the capillaries overlap).",
        )
        self.nestedBadges = []
        for i in range(2, MAX_NESTED_RINGS + 1):
            self.nestedBadges.append(
                RatioBadge(
                    f"Dcf{i} / dcf{i - 1}",
                    f"Ring-{i} outer diameter over ring-{i - 1} bore "
                    "(red when not nested).",
                )
            )
        badge_grid = QHBoxLayout()
        badge_grid.addWidget(self.boreRatioBadge)
        badge_grid.addWidget(self.capRatioBadge)
        badge_grid.addWidget(self.contactBadge)
        badge_grid.addStretch(1)
        nested_row = QHBoxLayout()
        for badge in self.nestedBadges:
            nested_row.addWidget(badge)
        nested_row.addStretch(1)

        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-family: monospace;"
        )
        preview_box = QGroupBox("Derived targets (live preview)")
        pv = QVBoxLayout(preview_box)
        pv.addLayout(kpi_row)
        pv.addLayout(badge_grid)
        pv.addLayout(nested_row)
        pv.addWidget(self.preview_label)

        # -- run controls -----------------------------------------------------
        self.runButton = QPushButton("Run optimization")
        self.runButton.setObjectName("btn_primary")
        self.runButton.clicked.connect(self.run_requested.emit)
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setObjectName("btn_danger")
        self.cancelButton.setEnabled(False)
        self.cancelButton.clicked.connect(self._on_cancel)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Idle.")
        self.warn_banner = QLabel("")
        self.warn_banner.setWordWrap(True)
        self.warn_banner.setStyleSheet(
            f"background: {SURFACE_LOWEST}; border: 1px solid {RED}; "
            f"padding: 6px; color: {RED};"
        )
        self.warn_banner.hide()

        solved_box = QGroupBox("Solved parameters")
        self.solvedTmaxLabel = QLabel("Tmax = --")
        self.solvedPcoreLabel = QLabel("Pcore = --")
        self.solvedPcapLabel = QLabel("Pcap = --")
        self.solvedCostLabel = QLabel("Cost = --")
        solved_form = QFormLayout(solved_box)
        solved_form.addRow(self.solvedTmaxLabel)
        solved_form.addRow(self.solvedPcoreLabel)
        solved_form.addRow(self.solvedPcapLabel)
        solved_form.addRow(self.solvedCostLabel)

        self.applyButton = QPushButton("Apply to Direct Computation")
        self.applyButton.setObjectName("btn_accent")
        self.applyButton.setEnabled(False)
        self.applyButton.clicked.connect(self._on_apply)
        self.saveCsvButton = QPushButton("Save results (.csv)")
        self.saveCsvButton.setObjectName("btn_export")
        self.saveCsvButton.clicked.connect(lambda: self._export("csv"))
        self.saveTxtButton = QPushButton("Save results (.txt)")
        self.saveTxtButton.setObjectName("btn_export")
        self.saveTxtButton.clicked.connect(lambda: self._export("txt"))

        run_row = QHBoxLayout()
        run_row.addWidget(self.runButton)
        run_row.addWidget(self.cancelButton)
        run_row.addStretch(1)
        action_row = QHBoxLayout()
        action_row.addWidget(self.applyButton)
        action_row.addStretch(1)
        action_row.addWidget(self.saveCsvButton)
        action_row.addWidget(self.saveTxtButton)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(targets_box)
        left_layout.addWidget(bounds_box)
        left_layout.addWidget(preview_box)
        left_layout.addLayout(run_row)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.warn_banner)
        left_layout.addWidget(solved_box)
        left_layout.addLayout(action_row)
        left_layout.addStretch(1)

        # -- right column: 12 views (UIAxes_2 + zmSlider_2) --------------------
        self.viewCombo = QComboBox()
        self.viewCombo.addItems(AXIAL_VIEWS)
        self.viewCombo.setCurrentText(TEMPERATURE_VIEW)
        self.viewCombo.currentTextChanged.connect(self._on_view_changed)
        self.plotCanvas = PlotCanvas()
        self.crossSection = CrossSectionCanvas()
        self.zmSlider_2 = self.crossSection.zmSlider
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.plotCanvas)  # index 0
        self.view_stack.addWidget(self.crossSection)  # index 1
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))
        view_row.addWidget(self.viewCombo)
        view_row.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addLayout(view_row)
        right_layout.addWidget(self.view_stack, stretch=1)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([520, 620])

        layout = QHBoxLayout(self)
        layout.addWidget(self.splitter)

        for edit in (
            self.tfEdit, self.Dcf1Edit, self.DcoreEdit, self.tmaxMinEdit,
            self.tmaxMaxEdit, self.pcoreMinEdit, self.pcoreMaxEdit,
            *self.pcapMinEdits, *self.pcapMaxEdits,
        ):
            edit.textChanged.connect(lambda *_: self.update_preview())
        self.set_nb_nested(1)
        self._on_view_changed(self.viewCombo.currentText())

    # -- previews --------------------------------------------------------------
    def update_preview(self, inputs: SimulationInputs | None = None) -> None:
        """Live derived-target KPI preview; warnings instead of crashes."""
        if inputs is None:
            if self._inputs_provider is None:
                return
            try:
                inputs = self._inputs_provider()
            except Exception:  # noqa: BLE001 - invalid intermediate state
                inputs = None
        if inputs is None:
            return
        self._last_inputs = inputs
        nb = len(inputs.rings)
        try:
            targets = compute_derived_targets(inputs, self.to_inputs(nb))
        except (TargetError, InputError, TypeError, ValueError) as exc:
            self.preview_label.setText("--")
            self._invalidate_preview()
            self.warn_banner.setText(f"Targets cannot be derived: {exc}")
            self.warn_banner.show()
            return
        self.warn_banner.hide()
        self._render_targets(targets, inputs)

    def _render_targets(
        self, targets: DerivedTargets, inputs: SimulationInputs
    ) -> None:
        self.kpiVf.set_data(f"{targets.wf * 60.0 * 1e3:.4g}", "mm/min")
        self.kpiDf.set_data(f"{2.0 * targets.Rf * 1e6:.4g}", "um")
        self.boreRatioBadge.set_ratio(2.0 * targets.rf, 2.0 * targets.Rf)
        self.capRatioBadge.set_ratio(2.0 * targets.rcf, 2.0 * targets.Rcf)
        self.contactBadge.set_ratio(
            targets.Rcf, r_contact_radius(inputs.r0, inputs.n_capillaries)
        )
        for j, badge in enumerate(self.nestedBadges):
            index = j + 1  # ring j+2 vs ring j+1
            if index < len(targets.ring_targets):
                dcf = 2.0 * targets.ring_targets[index][0]
                dch = 2.0 * targets.ring_targets[index - 1][1]
                badge.set_ratio(dcf, dch)
            else:
                badge.set_ratio(float("nan"), 1.0)
        self.preview_label.setText(self._format_targets(targets))

    def _invalidate_preview(self) -> None:
        self.kpiVf.set_data("--", "")
        self.kpiDf.set_data("--", "")
        for badge in (
            self.boreRatioBadge, self.capRatioBadge, self.contactBadge,
            *self.nestedBadges,
        ):
            badge.set_ratio(float("nan"), 1.0)

    @staticmethod
    def _format_targets(t: DerivedTargets) -> str:
        lines = [
            f"wf = {t.wf * 60.0 * 1e3:.4g} mm/min, "
            f"rf = {t.rf * 1e6:.4g} um, Rf = {t.Rf * 1e6:.4g} um",
        ]
        for i, (rcf_o, rcf_i) in enumerate(t.ring_targets, start=1):
            lines.append(
                f"Rcf{i} = {rcf_o * 1e6:.4g} um, rcf{i} = {rcf_i * 1e6:.4g} um"
            )
        return "\n".join(lines)

    # -- views (UIAxes_2 + zmSlider_2) ------------------------------------------
    def _on_view_changed(self, view: str) -> None:
        if view == SCHEMATIC_VIEW:
            self.view_stack.setCurrentIndex(1)
            self.crossSection.set_result(self._solution_result())
            return
        self.view_stack.setCurrentIndex(0)
        result = self._solution_result()
        if result is not None:
            self.plotCanvas.draw_axial_view(result, view)
        elif self._last_inputs is not None and view.startswith("Temperature"):
            self.refresh_pre_run_profile(self._last_inputs)
        else:
            self.plotCanvas.clear("No data yet — run an optimization.")

    def _solution_result(self):
        inner = self._result.result if self._result is not None else None
        return inner

    def refresh_pre_run_profile(self, inputs: SimulationInputs) -> None:
        """Live pre-run furnace temperature profile (no result installed)."""
        if self._result is not None or self.view_stack.currentIndex() == 1:
            return
        self._last_inputs = inputs
        self.plotCanvas.plot_initial_furnace_temperature(
            inputs.L,
            inputs.Lfour,
            inputs.Tmax_C,
            inputs.Text_C,
            inputs.furnace_profile_expr,
        )

    # -- run / cancel ------------------------------------------------------------
    def set_inputs_provider(self, provider) -> None:
        self._inputs_provider = provider
        self.update_preview()

    def attach_worker(self, worker: OptimizationWorker) -> None:
        self._worker = worker
        worker.progress.connect(self._on_progress)
        worker.status_changed.connect(self.status_label.setText)
        worker.result_ready.connect(self.set_solution)
        worker.error_occurred.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)
        self.runButton.setEnabled(False)
        self.cancelButton.setEnabled(True)
        self.progress_bar.setValue(0)
        worker.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self.status_label.setText("Cancelling...")
            self._worker.cancel()

    def _on_progress(self, message: str, frac: float) -> None:
        self.status_label.setText(message)
        self.progress_bar.setValue(int(max(0.0, min(1.0, frac)) * 100))

    def _on_worker_error(self, message: str) -> None:
        self.warn_banner.setText(message)
        self.warn_banner.show()

    def _on_worker_finished(self) -> None:
        self.runButton.setEnabled(True)
        self.cancelButton.setEnabled(False)
        self.progress_bar.setValue(100)
        self._worker = None

    # -- results ---------------------------------------------------------------------
    @property
    def result(self) -> OptimizationResult | None:
        return self._result

    def set_solution(self, result: OptimizationResult) -> None:
        self._result = result
        if result is None:
            return
        self.solvedTmaxLabel.setText(
            f"Tmax = {result.Tmax_C:.6g} C" if result.ok else "Tmax = --"
        )
        self.solvedPcoreLabel.setText(
            f"Pcore = {result.Pcore / 100.0:.6g} mbar"
            if result.ok else "Pcore = --"
        )
        pcap_txt = ", ".join(
            f"{p / 100.0:.6g}" for p in result.Pcap
        ) or "--"
        self.solvedPcapLabel.setText(f"Pcap = [{pcap_txt}] mbar")
        cost = "n/a" if result.cost is None else f"{result.cost:.3e}"
        self.solvedCostLabel.setText(f"Cost = {cost}")
        if result.ok:
            self.warn_banner.hide()
            self.applyButton.setEnabled(True)
            self.status_label.setText(
                f"Optimization finished in {result.elapsed_s:.1f} s."
            )
        else:
            self.applyButton.setEnabled(False)
            message = result.message or "The optimization failed."
            advice = result.advice or "No action required."
            self.warn_banner.setText(
                f"<b>{result.status.value}</b>: {message}<br><i>{advice}</i>"
            )
            self.warn_banner.show()
        self._on_view_changed(self.viewCombo.currentText())

    def _on_apply(self) -> None:
        if self._result is not None:
            self.apply_requested.emit(self._result)

    def _export(self, fmt: str) -> None:
        if self._result is None or self._result.result is None:
            self.status_label.setText("Nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save optimized run", f"harf_optimized.{fmt}",
            f"Results (*.{fmt})",
        )
        if not path:
            return
        try:
            export_results_table(self._result.result, path, fmt=fmt)
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Export failed: {exc}")
            return
        self.status_label.setText(f"Results saved to {path}")

    # -- data exchange ------------------------------------------------------------
    def set_nb_nested(self, nb_nested: int) -> None:
        """Enable exactly the first ``nb_nested`` Pcap window pairs."""
        nb = max(1, min(int(nb_nested), MAX_NESTED_RINGS))
        for i in range(MAX_NESTED_RINGS):
            enabled = i < nb
            self.pcapMinEdits[i].setEnabled(enabled)
            self.pcapMaxEdits[i].setEnabled(enabled)

    def to_inputs(self, nb_nested: int) -> OptimizationInputs:
        nb = max(1, int(nb_nested))
        pcap_min = tuple(
            mbar_to_pa(
                read_double(
                    self.pcapMinEdits[i], DEFAULT_OPTIM_PCAP_MIN_MBAR
                )
            )
            for i in range(nb)
        )
        pcap_max = tuple(
            mbar_to_pa(
                read_double(
                    self.pcapMaxEdits[i], DEFAULT_OPTIM_PCAP_MAX_MBAR
                )
            )
            for i in range(nb)
        )
        return OptimizationInputs(
            tf=um_to_m(read_double(self.tfEdit, DEFAULT_OPTIM_TF_UM)),
            Dcf1=um_to_m(read_double(self.Dcf1Edit, DEFAULT_OPTIM_DCF1_UM)),
            Dcore=um_to_m(read_double(self.DcoreEdit, DEFAULT_OPTIM_DCORE_UM)),
            Tmax_min_C=read_double(self.tmaxMinEdit, DEFAULT_OPTIM_TMAX_MIN_C),
            Tmax_max_C=read_double(self.tmaxMaxEdit, DEFAULT_OPTIM_TMAX_MAX_C),
            Pcore_min=mbar_to_pa(
                read_double(self.pcoreMinEdit, DEFAULT_OPTIM_PCORE_MIN_MBAR)
            ),
            Pcore_max=mbar_to_pa(
                read_double(self.pcoreMaxEdit, DEFAULT_OPTIM_PCORE_MAX_MBAR)
            ),
            Pcap_min=pcap_min,
            Pcap_max=pcap_max,
        )

    def apply_inputs(self, inputs: SimulationInputs, opt: OptimizationInputs) -> None:
        self.tfEdit.setText(f"{opt.tf * 1e6:.10g}")
        self.Dcf1Edit.setText(f"{opt.Dcf1 * 1e6:.10g}")
        self.DcoreEdit.setText(f"{opt.Dcore * 1e6:.10g}")
        self.tmaxMinEdit.setText(f"{opt.Tmax_min_C:.10g}")
        self.tmaxMaxEdit.setText(f"{opt.Tmax_max_C:.10g}")
        self.pcoreMinEdit.setText(f"{opt.Pcore_min / 100.0:.10g}")
        self.pcoreMaxEdit.setText(f"{opt.Pcore_max / 100.0:.10g}")
        for i in range(MAX_NESTED_RINGS):
            lo = opt.Pcap_min[i] if i < len(opt.Pcap_min) else opt.Pcap_min[0]
            hi = opt.Pcap_max[i] if i < len(opt.Pcap_max) else opt.Pcap_max[0]
            self.pcapMinEdits[i].setText(f"{lo / 100.0:.10g}")
            self.pcapMaxEdits[i].setText(f"{hi / 100.0:.10g}")
        self.set_nb_nested(len(inputs.rings))
