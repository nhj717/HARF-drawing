"""Tab 4: direct computation (Tmax / pressures), plots, and exports.

Hosts the 12-view plot area (11 axial views + the ``Schematic``
cross-section scrubbed by ``zmSlider``), the run/cancel controls, the
non-crashing diagnostic banner (failure status, ``z_stop`` in mm, and
actionable advice), post-run contact badges, and the ``.csv``/``.txt``
result-export buttons.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.widgets.cross_section_canvas import CrossSectionCanvas
from app.widgets.plot_canvas import AXIAL_VIEWS, SCHEMATIC_VIEW, PlotCanvas
from app.widgets.validators import (
    RatioBadge,
    make_double_edit,
    read_double,
)
from app.workers import SimulationWorker
from src.io_utils import export_results_table
from src.models import (
    DEFAULT_PCAP_MBAR,
    DEFAULT_PCORE_MBAR,
    DEFAULT_TMAX_C,
    SimulationInputs,
    SimulationResult,
    mbar_to_pa,
)

__all__ = ["DirectTab"]


class DirectTab(QWidget):
    """Direct simulation tab: process inputs, run controls, plots, I/O."""

    run_requested = Signal()
    result_ready = Signal(object)  # SimulationResult

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: SimulationResult | None = None
        self._worker: SimulationWorker | None = None

        self.TmaxEdit = make_double_edit(
            DEFAULT_TMAX_C, 0.0, 5000.0,
            tooltip="Maximum furnace temperature [C]",
        )
        self.PcoreEdit = make_double_edit(
            DEFAULT_PCORE_MBAR, -1e5, 1e5,
            tooltip="Core pressure Pcore [mbar]",
        )
        self.pcapEdits = []
        for i in range(3):
            edit = make_double_edit(
                DEFAULT_PCAP_MBAR, -1e5, 1e5,
                tooltip=f"Capillary {i + 1} absolute pressure [mbar]",
            )
            self.pcapEdits.append(edit)

        process_box = QGroupBox("Process conditions")
        form = QFormLayout(process_box)
        form.addRow("Furnace Tmax [C]", self.TmaxEdit)
        form.addRow("Core pressure Pcore [mbar]", self.PcoreEdit)
        for i, edit in enumerate(self.pcapEdits):
            form.addRow(f"Capillary Pcap{i + 1} [mbar]", edit)

        self.runButton = QPushButton("Run computation")
        self.runButton.clicked.connect(self._on_run)
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setEnabled(False)
        self.cancelButton.clicked.connect(self._on_cancel)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Idle.")
        self.diag_banner = QLabel("")
        self.diag_banner.setWordWrap(True)
        self.diag_banner.setStyleSheet(
            "background: #fff3e0; border: 1px solid #ef6c00; "
            "padding: 6px; color: #bf360c;"
        )
        self.diag_banner.hide()

        run_row = QHBoxLayout()
        run_row.addWidget(self.runButton)
        run_row.addWidget(self.cancelButton)
        run_row.addStretch(1)

        self.maxContactBadge = RatioBadge(
            "max Rc1/Rcontact", "Post-run maximum of Rc1/Rcontact."
        )
        self.maxNest2Badge = RatioBadge(
            "max Dc2/dc1", "Post-run maximum of Dc2/dc1."
        )
        self.maxNest3Badge = RatioBadge(
            "max Dc3/dc2", "Post-run maximum of Dc3/dc2."
        )
        badge_row = QHBoxLayout()
        badge_row.addWidget(self.maxContactBadge)
        badge_row.addWidget(self.maxNest2Badge)
        badge_row.addWidget(self.maxNest3Badge)
        badge_row.addStretch(1)

        self.viewCombo = QComboBox()
        self.viewCombo.addItems(AXIAL_VIEWS)
        self.viewCombo.currentTextChanged.connect(self._on_view_changed)
        self.plotCanvas = PlotCanvas()
        self.crossSection = CrossSectionCanvas()
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.plotCanvas)  # index 0
        self.view_stack.addWidget(self.crossSection)  # index 1

        self.saveCsvButton = QPushButton("Save results (.csv)")
        self.saveCsvButton.clicked.connect(lambda: self._export("csv"))
        self.saveTxtButton = QPushButton("Save results (.txt)")
        self.saveTxtButton.clicked.connect(lambda: self._export("txt"))

        layout = QVBoxLayout(self)
        layout.addWidget(process_box)
        layout.addLayout(run_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.diag_banner)
        layout.addLayout(badge_row)
        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("View:"))
        plot_row.addWidget(self.viewCombo)
        plot_row.addStretch(1)
        plot_row.addWidget(self.saveCsvButton)
        plot_row.addWidget(self.saveTxtButton)
        layout.addLayout(plot_row)
        layout.addWidget(self.view_stack, stretch=1)
        self._on_view_changed(self.viewCombo.currentText())

    # -- run / cancel --------------------------------------------------------
    def _on_run(self) -> None:
        self.run_requested.emit()  # MainWindow builds inputs and starts

    def attach_worker(self, worker: SimulationWorker) -> None:
        """Wire a freshly created worker and start it."""
        self._worker = worker
        worker.progress.connect(self._on_progress)
        worker.status_changed.connect(self.status_label.setText)
        worker.result_ready.connect(self._on_worker_result)
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
        self.status_label.setText("Error.")
        self.diag_banner.setText(message)
        self.diag_banner.show()

    def _on_worker_finished(self) -> None:
        self.runButton.setEnabled(True)
        self.cancelButton.setEnabled(False)
        self.progress_bar.setValue(100)
        self._worker = None

    def _on_worker_result(self, result: SimulationResult) -> None:
        self.set_result(result)
        self.result_ready.emit(result)

    # -- result handling -------------------------------------------------------
    @property
    def result(self) -> SimulationResult | None:
        return self._result

    def set_result(self, result: SimulationResult | None) -> None:
        """Install a result: banner, badges, plots, and cross-section."""
        self._result = result
        self._update_post_run_badges(result)
        if result is not None and not result.ok:
            diag = result.diagnostics
            z_stop_mm = (diag.z_stop if diag else 0.0) * 1e3
            lines = [
                f"<b>Run failed: {result.status.value}</b>",
                f"Stopped at z = {z_stop_mm:.2f} mm.",
                diag.message if diag else "",
                f"<i>Advice: {diag.advice if diag else ''}</i>",
            ]
            self.diag_banner.setText("<br>".join(x for x in lines if x))
            self.diag_banner.show()
        else:
            self.diag_banner.hide()
        self._on_view_changed(self.viewCombo.currentText())

    def _update_post_run_badges(self, result: SimulationResult | None) -> None:
        if result is None or len(result.z) < 2 or not result.Rc:
            for badge in (
                self.maxContactBadge, self.maxNest2Badge, self.maxNest3Badge,
            ):
                badge.set_ratio(float("nan"), 1.0)
            return
        rc1 = np.asarray(result.Rc[0])
        contact = np.asarray(result.R_contact)
        good = (rc1 > 0.0) & (contact > 0.0)
        if good.any():
            # Point-wise axial max at matching z (MATLAB max(Rc./rcontact)).
            self.maxContactBadge.set_ratio(
                float(np.max(rc1[good] / contact[good])), 1.0
            )
        else:
            self.maxContactBadge.set_ratio(float("nan"), 1.0)
        for badge, index in (
            (self.maxNest2Badge, 1), (self.maxNest3Badge, 2),
        ):
            if index < len(result.Rc) and index < len(result.rc):
                dc = 2.0 * np.asarray(result.Rc[index])
                dch = 2.0 * np.asarray(result.rc[index - 1])
                good = (dc > 0.0) & (dch > 0.0)
                if good.any():
                    ratio = float(np.max(dc[good] / dch[good]))
                    badge.set_ratio(ratio, 1.0)
                    continue
            badge.set_ratio(float("nan"), 1.0)

    # -- views -------------------------------------------------------------------
    def _on_view_changed(self, view: str) -> None:
        if view == SCHEMATIC_VIEW:
            self.view_stack.setCurrentIndex(1)
            self.crossSection.set_result(self._result)
        else:
            self.view_stack.setCurrentIndex(0)
            self.plotCanvas.draw_axial_view(self._result, view)

    # -- export --------------------------------------------------------------------
    def _export(self, fmt: str) -> None:
        if self._result is None:
            self.status_label.setText("Nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save results", f"harf_results.{fmt}",
            f"Results (*.{fmt})",
        )
        if not path:
            return
        try:
            export_results_table(self._result, path, fmt=fmt)
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Export failed: {exc}")
            return
        self.status_label.setText(f"Results saved to {path}")

    # -- data exchange -----------------------------------------------------------------
    def to_fields(self, nb_nested: int) -> dict:
        pcap = tuple(
            mbar_to_pa(read_double(edit, DEFAULT_PCAP_MBAR))
            for edit in self.pcapEdits[:nb_nested]
        )
        return {
            "Tmax_C": read_double(self.TmaxEdit, DEFAULT_TMAX_C),
            "Pcore": mbar_to_pa(read_double(self.PcoreEdit, DEFAULT_PCORE_MBAR)),
            "Pcap": pcap,
        }

    def apply_inputs(self, inputs: SimulationInputs) -> None:
        self.TmaxEdit.setText(f"{inputs.Tmax_C:.10g}")
        self.PcoreEdit.setText(f"{inputs.Pcore / 100.0:.10g}")
        for i, edit in enumerate(self.pcapEdits):
            value = (
                inputs.Pcap[i] / 100.0 if i < len(inputs.Pcap)
                else DEFAULT_PCAP_MBAR
            )
            edit.setText(f"{value:.10g}")

    def apply_process(
        self, t_max_c: float, p_core_pa: float, pcap_pa: tuple[float, ...]
    ) -> None:
        """Programmatic transfer used by "Apply to Direct Computation"."""
        self.TmaxEdit.setText(f"{float(t_max_c):.10g}")
        self.PcoreEdit.setText(f"{float(p_core_pa) / 100.0:.10g}")
        for i, edit in enumerate(self.pcapEdits):
            if i < len(pcap_pa):
                edit.setText(f"{float(pcap_pa[i]) / 100.0:.10g}")

    # -- helpers for tests ------------------------------------------------------------
    def status_text(self) -> str:
        return self.status_label.text()

    def banner_visible(self) -> bool:
        return self.diag_banner.isVisibleTo(self)

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()
