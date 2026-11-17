"""Tab 5: optimization targets, windows, solved cards, and apply/export.

Live derived-target previews (``Rf``, ``rf``, ``wf``, ``Rcf_i``,
``rcf_i``) follow the exact ``RunButton_2Pushed`` rules; impossible
targets show a warning instead of crashing.  *"Apply to Direct
Computation"* transfers the solved ``(Tmax, Pcore, Pcap1..3)`` into the
Direct Computation tab.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.widgets.validators import make_double_edit, read_double
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
    DerivedTargets,
    InputError,
    OptimizationInputs,
    OptimizationResult,
    SimulationInputs,
    TargetError,
    compute_derived_targets,
    mbar_to_pa,
    um_to_m,
)

__all__ = ["OptimizationTab"]


class OptimizationTab(QWidget):
    """Optimization tab: targets, windows, run controls, apply, export."""

    run_requested = Signal()
    apply_requested = Signal(object)  # OptimizationResult

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: OptimizationResult | None = None
        self._worker: OptimizationWorker | None = None
        self._inputs_provider = None

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
        self.pcapMinEdit = make_double_edit(
            DEFAULT_OPTIM_PCAP_MIN_MBAR, -1e5, 1e5
        )
        self.pcapMaxEdit = make_double_edit(
            DEFAULT_OPTIM_PCAP_MAX_MBAR, -1e5, 1e5
        )
        bounds_box = QGroupBox("Search windows (applied to every ring)")
        form = QFormLayout(bounds_box)
        form.addRow("Tmax min [C]", self.tmaxMinEdit)
        form.addRow("Tmax max [C]", self.tmaxMaxEdit)
        form.addRow("Pcore min [mbar]", self.pcoreMinEdit)
        form.addRow("Pcore max [mbar]", self.pcoreMaxEdit)
        form.addRow("Pcap min [mbar]", self.pcapMinEdit)
        form.addRow("Pcap max [mbar]", self.pcapMaxEdit)

        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("color: #37474f;")
        preview_box = QGroupBox("Derived targets (live preview)")
        pv = QVBoxLayout(preview_box)
        pv.addWidget(self.preview_label)

        self.runButton = QPushButton("Run optimization")
        self.runButton.clicked.connect(self.run_requested.emit)
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setEnabled(False)
        self.cancelButton.clicked.connect(self._on_cancel)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Idle.")
        self.warn_banner = QLabel("")
        self.warn_banner.setWordWrap(True)
        self.warn_banner.setStyleSheet(
            "background: #fff3e0; border: 1px solid #ef6c00; "
            "padding: 6px; color: #bf360c;"
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
        self.applyButton.setEnabled(False)
        self.applyButton.clicked.connect(self._on_apply)
        self.saveCsvButton = QPushButton("Save results (.csv)")
        self.saveCsvButton.clicked.connect(lambda: self._export("csv"))
        self.saveTxtButton = QPushButton("Save results (.txt)")
        self.saveTxtButton.clicked.connect(lambda: self._export("txt"))

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(targets_box)
        top.addWidget(bounds_box)
        layout.addLayout(top)
        layout.addWidget(preview_box)
        run_row = QHBoxLayout()
        run_row.addWidget(self.runButton)
        run_row.addWidget(self.cancelButton)
        run_row.addStretch(1)
        layout.addLayout(run_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.warn_banner)
        layout.addWidget(solved_box)
        action_row = QHBoxLayout()
        action_row.addWidget(self.applyButton)
        action_row.addStretch(1)
        action_row.addWidget(self.saveCsvButton)
        action_row.addWidget(self.saveTxtButton)
        layout.addLayout(action_row)
        layout.addStretch(1)

        for edit in (
            self.tfEdit, self.Dcf1Edit, self.DcoreEdit, self.tmaxMinEdit,
            self.tmaxMaxEdit, self.pcoreMinEdit, self.pcoreMaxEdit,
            self.pcapMinEdit, self.pcapMaxEdit,
        ):
            edit.textChanged.connect(self.update_preview)

    # -- previews --------------------------------------------------------------
    def update_preview(self) -> None:
        """Live derived-target preview; warnings instead of crashes."""
        if self._inputs_provider is None:
            return
        try:
            inputs = self._inputs_provider()
            nb = len(inputs.rings)
            targets = compute_derived_targets(inputs, self.to_inputs(nb))
        except (TargetError, InputError, TypeError, ValueError) as exc:
            self.preview_label.setText("--")
            self.warn_banner.setText(f"Targets cannot be derived: {exc}")
            self.warn_banner.show()
            return
        self.warn_banner.hide()
        self.preview_label.setText(self._format_targets(targets))

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
    def to_inputs(self, nb_nested: int) -> OptimizationInputs:
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
            Pcap_min=(mbar_to_pa(
                read_double(self.pcapMinEdit, DEFAULT_OPTIM_PCAP_MIN_MBAR)
            ),) * nb_nested,
            Pcap_max=(mbar_to_pa(
                read_double(self.pcapMaxEdit, DEFAULT_OPTIM_PCAP_MAX_MBAR)
            ),) * nb_nested,
        )

    def apply_inputs(self, inputs: SimulationInputs, opt: OptimizationInputs) -> None:
        self.tfEdit.setText(f"{opt.tf * 1e6:.10g}")
        self.Dcf1Edit.setText(f"{opt.Dcf1 * 1e6:.10g}")
        self.DcoreEdit.setText(f"{opt.Dcore * 1e6:.10g}")
        self.tmaxMinEdit.setText(f"{opt.Tmax_min_C:.10g}")
        self.tmaxMaxEdit.setText(f"{opt.Tmax_max_C:.10g}")
        self.pcoreMinEdit.setText(f"{opt.Pcore_min / 100.0:.10g}")
        self.pcoreMaxEdit.setText(f"{opt.Pcore_max / 100.0:.10g}")
        self.pcapMinEdit.setText(f"{opt.Pcap_min[0] / 100.0:.10g}")
        self.pcapMaxEdit.setText(f"{opt.Pcap_max[0] / 100.0:.10g}")
