"""Tab 4: direct computation (Tmax / pressures), plots, and exports.

Hosts the 12-view plot area (11 axial views + the ``Schematic``
cross-section scrubbed by ``zmSlider``), the run/cancel controls, the
non-crashing diagnostic banner (failure status, ``z_stop`` in mm, and
actionable advice), post-run contact badges, and the ``.csv``/``.txt``
result-export buttons.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
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

from app.theme import RED, SURFACE_LOWEST, TEXT_MUTED, TEXT_PRIMARY, mono_font_family
from app.widgets.cross_section_canvas import CrossSectionCanvas
from app.widgets.plot_canvas import (
    AXIAL_VIEWS,
    SCHEMATIC_VIEW,
    TEMPERATURE_VIEW,
    PlotCanvas,
)
from app.widgets.scroll import wrap_scroll
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
    MAX_NESTED_RINGS,
    SimulationInputs,
    SimulationResult,
    mbar_to_pa,
)

__all__ = ["DirectTab"]


class _ValueReadout(QLabel):
    """Compact monospaced value cell of the final-outputs table.

    Mirrors the lightweight ``set_data`` / ``value_text`` / ``reset``
    accessors of :class:`~app.theme.KpiCard` so callers and tests stay
    ergonomic, without the wide neon-card chrome.
    """

    def __init__(self, unit: str = "", parent=None) -> None:
        super().__init__("\u2014", parent)
        self.unit = unit
        self._value_text = "--"
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-weight: 700; "
            f"font-family: '{mono_font_family()}';"
        )

    def set_data(self, value: str, unit: str | None = None) -> None:
        self._value_text = str(value)
        if unit is not None:
            self.unit = unit
        self.setText(self._value_text)

    def value_text(self) -> str:
        """Current numeric readout text (``--`` before any run)."""
        return self._value_text

    def reset(self) -> None:
        """Pre-run placeholder state."""
        self._value_text = "--"
        self.setText("\u2014")


class DirectTab(QWidget):
    """Direct simulation tab: process inputs, run controls, plots, I/O."""

    run_requested = Signal()
    result_ready = Signal(object)  # SimulationResult
    inputs_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: SimulationResult | None = None
        self._worker: SimulationWorker | None = None
        self._last_inputs: SimulationInputs | None = None

        self.TmaxEdit = make_double_edit(
            DEFAULT_TMAX_C, 0.0, 5000.0,
            tooltip="Maximum furnace temperature [C]",
        )
        self.PcoreEdit = make_double_edit(
            DEFAULT_PCORE_MBAR, -1e5, 1e5,
            tooltip="Core pressure Pcore [mbar]",
        )
        self.pcapEdits = []
        self.pcapLabels: list[QLabel] = []
        for i in range(MAX_NESTED_RINGS):
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
            label = QLabel(f"Capillary Pcap{i + 1} [mbar]")
            self.pcapLabels.append(label)
            form.addRow(label, edit)

        self.runButton = QPushButton("Run computation")
        self.runButton.setObjectName("btn_primary")
        self.runButton.clicked.connect(self._on_run)
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setObjectName("btn_danger")
        self.cancelButton.setEnabled(False)
        self.cancelButton.clicked.connect(self._on_cancel)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Idle.")
        self.diag_banner = QLabel("")
        self.diag_banner.setWordWrap(True)
        self.diag_banner.setStyleSheet(
            f"background: {SURFACE_LOWEST}; border: 1px solid {RED}; "
            f"padding: 6px; color: {RED};"
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
        self.viewCombo.setCurrentText(TEMPERATURE_VIEW)
        self.viewCombo.currentTextChanged.connect(self._on_view_changed)
        self.plotCanvas = PlotCanvas()
        self.crossSection = CrossSectionCanvas()
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.plotCanvas)  # index 0
        self.view_stack.addWidget(self.crossSection)  # index 1

        self.saveCsvButton = QPushButton("Save results (.csv)")
        self.saveCsvButton.setObjectName("btn_export")
        self.saveCsvButton.clicked.connect(lambda: self._export("csv"))
        self.saveTxtButton = QPushButton("Save results (.txt)")
        self.saveTxtButton.setObjectName("btn_export")
        self.saveTxtButton.clicked.connect(lambda: self._export("txt"))

        # Post-run final-fiber KPI readouts (MATLAB output panel).
        self.finalOutputBox = self._build_final_output_panel()

        # Horizontal side-by-side splitter (controls left, plots right),
        # matching Tabs 1, 2, 3 and 5.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(process_box)
        left_layout.addLayout(run_row)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.diag_banner)
        left_layout.addLayout(badge_row)
        left_layout.addWidget(self.finalOutputBox)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("View:"))
        plot_row.addWidget(self.viewCombo)
        plot_row.addStretch(1)
        plot_row.addWidget(self.saveCsvButton)
        plot_row.addWidget(self.saveTxtButton)
        right_layout.addLayout(plot_row)
        right_layout.addWidget(self.view_stack, stretch=1)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(wrap_scroll(left))
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([430, 830])

        layout = QHBoxLayout(self)
        layout.addWidget(self.splitter)
        for edit in (self.TmaxEdit, self.PcoreEdit, *self.pcapEdits):
            edit.textChanged.connect(self.inputs_changed)
        self.set_nb_nested(1)
        self._on_view_changed(self.viewCombo.currentText())

    def _build_final_output_panel(self) -> QGroupBox:
        """Compact solved-parameters-style table for the final fiber.

        Always visible (``--`` placeholders pre-run): a 2-column grid
        of labeled value rows plus a 2-column badge summary, matching
        the ``Solved parameters`` aesthetic of the Optimization tab —
        no horizontal sprawl, no scrolling required at 1280x800.
        """
        box = QGroupBox("Final fiber outputs")
        self.kpiVf = _ValueReadout("m/min")
        self.kpiDf = _ValueReadout("um")
        self.kpiDfBore = _ValueReadout("um")
        self.kpiDcore = _ValueReadout("um")
        self.kpiDcf = _ValueReadout("um")
        self.kpiDcfBore = _ValueReadout("um")
        self.kpiTf = _ValueReadout("um")
        self.finalContactBadge = RatioBadge(
            "Rcf1 / Rcontact",
            "Final ring-1 outer radius over the packing contact radius "
            "(red when the capillaries overlap).",
        )
        self.ringRatioBadges = [
            RatioBadge(
                f"dcf{i + 1} / Dcf{i + 1}",
                f"Final ring-{i + 1} bore/outer diameter ratio.",
            )
            for i in range(3)
        ]
        rows = (
            ("Vf", self.kpiVf), ("Df", self.kpiDf),
            ("df", self.kpiDfBore), ("Dcore", self.kpiDcore),
            ("Dcf1", self.kpiDcf), ("dcf1", self.kpiDcfBore),
            ("tf1", self.kpiTf),
        )
        badges = (self.finalContactBadge,) + tuple(self.ringRatioBadges)
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 4, 8, 6)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)
        for i, (name, readout) in enumerate(rows):
            row, col = divmod(i, 2)
            label = QLabel(f"{name} [{readout.unit}]")
            label.setStyleSheet(f"color: {TEXT_MUTED};")
            grid.addWidget(label, row, 2 * col, Qt.AlignVCenter)
            grid.addWidget(readout, row, 2 * col + 1)
        badge_start = (len(rows) + 1) // 2
        for i, badge in enumerate(badges):
            row, col = divmod(i, 2)
            grid.addWidget(badge, badge_start + row, 2 * col, 1, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return box

    # -- dynamic state --------------------------------------------------------
    def set_nb_nested(self, nb_nested: int) -> None:
        """Enable exactly the first ``nb_nested`` capillary widgets.

        Disables the ``Pcap i`` inputs and their form labels, the
        per-ring post-run ratio badges and the inter-ring badges
        (``max Dc2/dc1`` needs 2 rings, ``max Dc3/dc2`` needs 3).
        """
        nb = max(1, min(int(nb_nested), MAX_NESTED_RINGS))
        for i in range(MAX_NESTED_RINGS):
            enabled = i < nb
            self.pcapEdits[i].setEnabled(enabled)
            self.pcapLabels[i].setEnabled(enabled)
            self.ringRatioBadges[i].setEnabled(enabled)
        self.maxNest2Badge.setEnabled(nb >= 2)
        self.maxNest3Badge.setEnabled(nb >= 3)

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
        """Install a result: banner, badges, KPI panel, plots, cross-section."""
        self._result = result
        self._update_post_run_badges(result)
        self._render_final_outputs(result)
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

    # -- final-fiber readouts --------------------------------------------
    def _render_final_outputs(self, result: SimulationResult | None) -> None:
        """Fill (or reset) the post-run value table and ratio badges."""
        cards = (
            self.kpiVf, self.kpiDf, self.kpiDfBore, self.kpiDcore,
            self.kpiDcf, self.kpiDcfBore, self.kpiTf,
        )
        if (
            result is None or not result.ok
            or len(result.z) == 0 or len(result.w) == 0
        ):
            for card in cards:
                card.reset()
            self.finalContactBadge.reset()
            for badge in self.ringRatioBadges:
                badge.reset()
            return
        self.kpiVf.set_data(f"{result.w[-1] * 60.0:.4g}")
        self.kpiDf.set_data(f"{2.0 * result.R[-1] * 1e6:.4g}")
        self.kpiDfBore.set_data(f"{2.0 * result.r[-1] * 1e6:.4g}")
        self.kpiDcore.set_data(f"{result.D_core[-1] * 1e6:.4g}")
        if result.Rc and len(result.Rc[0]):
            self.kpiDcf.set_data(
                f"{2.0 * result.Rc[0][-1] * 1e6:.4g}"
            )
        if result.rc and len(result.rc[0]):
            self.kpiDcfBore.set_data(
                f"{2.0 * result.rc[0][-1] * 1e6:.4g}"
            )
        if (
            result.Rc and result.rc
            and len(result.Rc[0]) and len(result.rc[0])
        ):
            self.kpiTf.set_data(
                f"{(result.Rc[0][-1] - result.rc[0][-1]) * 1e6:.4g}"
            )
        if result.Rc and len(result.Rc[0]) and len(result.R_contact):
            self.finalContactBadge.set_ratio(
                float(result.Rc[0][-1]), float(result.R_contact[-1])
            )
        else:
            self.finalContactBadge.reset()
        for i, badge in enumerate(self.ringRatioBadges):
            if (
                i < len(result.Rc) and i < len(result.rc)
                and len(result.Rc[i]) and len(result.rc[i])
            ):
                badge.set_ratio(
                    2.0 * float(result.rc[i][-1]),
                    2.0 * float(result.Rc[i][-1]),
                )
            else:
                badge.reset()

    # -- views -------------------------------------------------------------------
    def _on_view_changed(self, view: str) -> None:
        if view == SCHEMATIC_VIEW:
            self.view_stack.setCurrentIndex(1)
            self.crossSection.set_result(self._result)
        elif self._result is not None:
            self.view_stack.setCurrentIndex(0)
            self.plotCanvas.draw_axial_view(self._result, view)
        elif self._last_inputs is not None and view.startswith("Temperature"):
            self.view_stack.setCurrentIndex(0)
            self.refresh_pre_run_profile(self._last_inputs)
        else:
            self.view_stack.setCurrentIndex(0)
            self.plotCanvas.clear("No data yet — run a computation.")

    def refresh_pre_run_profile(self, inputs: SimulationInputs) -> None:
        """Live pre-run axial furnace profile (no result installed)."""
        if self._result is not None or self.view_stack.currentIndex() == 1:
            return
        self._last_inputs = inputs
        if self.viewCombo.currentText() != TEMPERATURE_VIEW:
            # Un-run views keep their empty-state prompt.
            return
        self.plotCanvas.plot_initial_furnace_temperature(
            inputs.L,
            inputs.Lfour,
            inputs.Tmax_C,
            inputs.Text_C,
            inputs.furnace_profile_expr,
        )

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
