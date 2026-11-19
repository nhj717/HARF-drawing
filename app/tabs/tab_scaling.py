"""Tab 6: cane & stack scaling (port of ``6Ring_Scaling.xlsx``).

Stage-by-stage closed-form geometry mapping between the cane / jacket
preform, the stage-1 stack and the final microstructured fiber:

* **Forward** (cane -> fiber): fill the cane, jacket and draw factors,
  watch the E -> I stage table, feasibility flags and cross-section
  schematics update live.
* **Back-calc** (fiber -> cane): give a target fiber and recover the
  cane geometry (fixed-jacket or manual ``S_j`` sub-mode).
* **Stack tools**: scale the cane up to the stage-1 stack (and back).
* **Solver bridge**: pull the Jacket / Capillaries / Optimization tab
  state into this tab, or push a solved cane/fiber handoff back.

All lengths are SI metres in :mod:`src.scaling`; the GUI shows them in
mm (cane / jacket / stack) or um (fiber) with a global unit toggle.
"""

from __future__ import annotations

import contextlib
import math

from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.theme import GREEN, RED, TEXT_MUTED
from app.widgets.cross_section_canvas import CrossSectionCanvas
from app.widgets.scroll import wrap_scroll
from app.widgets.validators import make_double_edit, read_double
from src.models import MAX_NESTED_RINGS
from src.scaling import (
    CO_FED_VACUUM,
    CaneGeometry,
    CouplingMode,
    FiberGeometry,
    InverseSubMode,
    JacketGeometry,
    ScalingError,
    ScalingFactors,
    ScalingResult,
    SolverHandoff,
    StackGeometry,
    cane_to_stack,
    compute_forward_scaling,
    get_scaling_presets,
    inverse_jacket_od,
    scaling_to_solver_inputs,
    solve_inverse_cane,
    stack_to_cane,
)

__all__ = [
    "MODE_BACKCALC",
    "MODE_FORWARD",
    "SUBMODE_FIXED_JACKET",
    "SUBMODE_MANUAL_SJ",
    "ScalingTab",
]


MODE_FORWARD = "Forward (cane -> fiber)"
MODE_BACKCALC = "Back-calc (fiber -> cane)"
SUBMODE_FIXED_JACKET = "Fixed jacket (OD_c given)"
SUBMODE_MANUAL_SJ = "Manual S_j (+ t_sleeve)"

_COUPLING_ITEMS: tuple[tuple[str, CouplingMode], ...] = (
    ("Fed vacuum (area conserved)", CO_FED_VACUUM),
    ("Legacy Excel", CouplingMode.LEGACY_EXCEL),
)
_TEXT_FOR_COUPLING = {mode: text for text, mode in _COUPLING_ITEMS}
_COUPLING_FOR_TEXT = dict(_COUPLING_ITEMS)

_UNIT_ITEMS = ("mm", "µm")
_STAGE_COLS = ("E", "F", "G", "H", "I")

STAGE_ROWS: tuple[tuple[str, str], ...] = (
    ("ζ envelope", "zeta"),
    ("D_core", "D_core"),
    ("d (bores)", "d"),
    ("t (walls)", "t"),
    ("Δ (ring OD)", "Delta"),
    ("OD_cane", "OD_cane"),
    ("ID_j", "ID_j"),
    ("OD_j", "OD_j"),
    ("Γ = d/Δ", "Gamma"),
    ("η = d1/D_core", "eta"),
    ("δ (cap gap)", "delta"),
    ("γ (ring area %)", "gamma_pct"),
)

_LENGTH_ATTRS = frozenset(
    {"zeta", "D_core", "d", "t", "Delta", "OD_cane", "ID_j", "OD_j",
     "delta"}
)

_FLAGS: tuple[tuple[str, str], ...] = (
    ("jacket_valid", "Jacket OD > ID"),
    ("cane_wall_positive", "Cane wall"),
    ("factors_positive", "Factors > 0"),
    ("fiber_walls_positive", "Fiber walls"),
    ("core_positive", "Cores > 0"),
    ("capillary_gap_positive", "Capillary gap"),
    ("jacket_area_conserved", "Jacket area"),
    ("ring_areas_conserved", "Ring areas"),
)

_STACK_SPECS = ("S_stack", "ID_stack", "OD_stack")


def _form_box(title: str, rows: list[tuple[str, QWidget]]) -> QGroupBox:
    """A titled group box with a two-column label / widget grid."""
    box = QGroupBox(title)
    layout = QGridLayout(box)
    for r, (label, widget) in enumerate(rows):
        layout.addWidget(QLabel(label), r, 0)
        layout.addWidget(widget, r, 1)
    layout.setColumnStretch(1, 1)
    return box


class _FlagBadge(QLabel):
    """Compact OK / FAIL / ``--`` chip for one feasibility flag."""

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self._label = label
        self.is_ok: bool | None = None
        self.setToolTip(label)
        self.set_state(None)

    def set_state(self, ok: bool | None) -> None:
        self.is_ok = ok
        if ok is None:
            self.setText(f"{self._label}: --")
            self.setStyleSheet(
                f"color: {TEXT_MUTED}; border: 1px solid #3a494a; "
                "border-radius: 3px; padding: 1px 8px;"
            )
        elif ok:
            self.setText(f"{self._label}: OK")
            self.setStyleSheet(
                f"color: {GREEN}; background: rgba(0, 230, 118, 0.10); "
                f"border: 1px solid {GREEN}; border-radius: 3px; "
                "padding: 1px 8px; font-weight: bold;"
            )
        else:
            self.setText(f"{self._label}: FAIL")
            self.setStyleSheet(
                f"color: {RED}; background: rgba(255, 82, 82, 0.10); "
                f"border: 1px solid {RED}; border-radius: 3px; "
                "padding: 1px 8px; font-weight: bold;"
            )


class ScalingTab(QWidget):
    """Cane & stack scaling workstation tab."""

    pull_requested = Signal()
    push_requested = Signal()

    # Class-level alias so tests (and external callers) can reach the
    # stage-table row map as ``ScalingTab.STAGE_ROWS``.
    STAGE_ROWS = STAGE_ROWS

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating = False
        self._result: ScalingResult | None = None
        self._stack: StackGeometry | None = None

        # -- selectors -----------------------------------------------------
        self.presetCombo = QComboBox()
        for preset in get_scaling_presets():
            self.presetCombo.addItem(preset.name)
        self.modeCombo = QComboBox()
        self.modeCombo.addItems((MODE_FORWARD, MODE_BACKCALC))
        self.couplingCombo = QComboBox()
        for text, _mode in _COUPLING_ITEMS:
            self.couplingCombo.addItem(text)
        self.unitCombo = QComboBox()
        self.unitCombo.addItems(_UNIT_ITEMS)
        self.ncapSpinBox = QSpinBox()
        self.ncapSpinBox.setRange(1, 12)
        self.ncapSpinBox.setValue(6)
        self.ncapSpinBox.setLocale(QLocale.c())
        self.nestedSpinBox = QSpinBox()
        self.nestedSpinBox.setRange(1, MAX_NESTED_RINGS)
        self.nestedSpinBox.setValue(1)
        self.nestedSpinBox.setLocale(QLocale.c())

        # -- cane edits ----------------------------------------------------
        self.odcEdit = make_double_edit(
            3.8, 0.0, 1e3, tooltip="Cane outer diameter OD_c"
        )
        self.dcorecEdit = make_double_edit(
            1.4, 0.0, 1e3, tooltip="Cane core diameter D_core_c"
        )
        self.dcEdits = [
            make_double_edit(
                v, 0.0, 1e3, tooltip=f"Ring {i + 1} cane bore d_c,{i + 1}"
            )
            for i, v in enumerate((0.45, 0.22, 0.11))
        ]
        self.tcEdits = [
            make_double_edit(
                v, 0.0, 1e3,
                tooltip=f"Ring {i + 1} cane wall t_c,{i + 1}",
            )
            for i, v in enumerate((0.05, 0.04, 0.025))
        ]

        # -- jacket edits --------------------------------------------------
        self.odjEdit = make_double_edit(
            10.0, 0.0, 1e3, tooltip="Jacket outer diameter OD_j"
        )
        self.idjEdit = make_double_edit(
            4.0, 0.0, 1e3, tooltip="Jacket bore diameter ID_j"
        )

        # -- factor edits --------------------------------------------------
        self.sjEdit = make_double_edit(
            40.0, 0.0, 1e5, tooltip="Jacket draw ratio S_j"
        )
        self.sieEdit = make_double_edit(
            1.6, 0.0, 1e5, tooltip="Ring-1 inflation factor S_ie"
        )
        self.sceEdits = [
            make_double_edit(
                v, 0.0, 1e5, tooltip=f"Ring {i + 1} expansion S_ce,{i + 1}"
            )
            for i, v in enumerate((2.5, 2.0, 1.8))
        ]

        # -- fiber edits ---------------------------------------------------
        self.dfEdit = make_double_edit(
            0.26, 0.0, 1e3, tooltip="Fiber outer diameter D_f"
        )
        self.dcorefEdit = make_double_edit(
            0.04, 0.0, 1e3, tooltip="Fiber core diameter D_core_f"
        )
        self.dfEdits = [
            make_double_edit(
                v, 0.0, 1e3, tooltip=f"Ring {i + 1} fiber bore d_f,{i + 1}"
            )
            for i, v in enumerate((0.026, 0.013, 0.0065))
        ]
        self.tfEdits = [
            make_double_edit(
                v, 0.0, 1e3, tooltip=f"Ring {i + 1} fiber wall t_f,{i + 1}"
            )
            for i, v in enumerate((0.0003, 0.0002, 0.0001))
        ]

        # -- back-calc / stack widgets ---------------------------------------
        self.subModeCombo = QComboBox()
        self.subModeCombo.addItems((SUBMODE_FIXED_JACKET, SUBMODE_MANUAL_SJ))
        self.tSleeveEdit = make_double_edit(
            0.5, 0.0, 1e3,
            tooltip="Cane sleeve wall t_sleeve (manual S_j sub-mode)",
        )
        self.stackSpecCombo = QComboBox()
        self.stackSpecCombo.addItems(_STACK_SPECS)
        self.stackParamLabel = QLabel("S_stack (–)")
        self.stackParamEdit = make_double_edit(
            2.5, 0.0, 1e5, tooltip="Stack scale / diameter parameter"
        )
        self.sieStEdit = make_double_edit(
            1.7, 0.0, 1e5, tooltip="Stack ring-1 inflation S_ie_st"
        )
        self.scestEdits = [
            make_double_edit(
                v, 0.0, 1e5, tooltip=f"Stack ring {i + 1} expansion"
            )
            for i, v in enumerate((2.0, 1.9, 1.8))
        ]
        self.stackForwardButton = QPushButton("Cane -> Stack")
        self.stackInverseButton = QPushButton("Stack -> Cane")
        self.stackResultLabel = QLabel("Stack: not computed yet.")
        self.stackResultLabel.setWordWrap(True)

        # -- bridge / status -------------------------------------------------
        self.pullButton = QPushButton("Pull from solver tabs")
        self.pullButton.setObjectName("btn_accent")
        self.pushButton = QPushButton("Push to solver tabs")
        self.pushButton.setObjectName("btn_primary")
        self.statusLabel = QLabel("Ready.")
        self.statusLabel.setWordWrap(True)

        # -- feasibility flags ------------------------------------------------
        self._flag_badges: dict[str, _FlagBadge] = {}
        flags_box = QGroupBox("Feasibility flags")
        flags_grid = QGridLayout(flags_box)
        badges = [_FlagBadge(label) for _field, label in _FLAGS]
        self._flag_badges = {
            field: badge
            for (field, _label), badge in zip(_FLAGS, badges, strict=True)
        }
        self.allOkBadge = _FlagBadge("All checks")
        for i, badge in enumerate((*badges, self.allOkBadge)):
            flags_grid.addWidget(badge, i // 3, i % 3)

        # -- stage table -------------------------------------------------------
        self.tableBox = QGroupBox("Stage table")
        table_layout = QVBoxLayout(self.tableBox)
        self.stageTable = QTableWidget(len(STAGE_ROWS), 1 + len(_STAGE_COLS))
        self.stageTable.setHorizontalHeaderLabels(("Parameter",) + _STAGE_COLS)
        self.stageTable.verticalHeader().setVisible(False)
        self.stageTable.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        for r, (label, _attr) in enumerate(STAGE_ROWS):
            self.stageTable.setItem(r, 0, QTableWidgetItem(label))
        self.stageTable.horizontalHeader().setStretchLastSection(True)
        self.stageTable.setColumnWidth(0, 150)
        table_layout.addWidget(self.stageTable)

        # -- canvases ----------------------------------------------------------
        self.caneCanvas = CrossSectionCanvas()
        self.fiberCanvas = CrossSectionCanvas()

        # -- layout --------------------------------------------------------------
        selector_rows = [
            ("Preset", self.presetCombo),
            ("Mode", self.modeCombo),
            ("Coupling", self.couplingCombo),
            ("Units", self.unitCombo),
            ("Capillaries (n_cap)", self.ncapSpinBox),
            ("Nested rings", self.nestedSpinBox),
        ]
        cane_rows: list[tuple[str, QWidget]] = [
            ("OD_c", self.odcEdit),
            ("D_core_c", self.dcorecEdit),
        ]
        for i in range(MAX_NESTED_RINGS):
            cane_rows.append((f"d_c,{i + 1}", self.dcEdits[i]))
            cane_rows.append((f"t_c,{i + 1}", self.tcEdits[i]))
        factors_rows: list[tuple[str, QWidget]] = [
            ("S_j", self.sjEdit),
            ("S_ie", self.sieEdit),
        ]
        for i in range(MAX_NESTED_RINGS):
            factors_rows.append((f"S_ce,{i + 1}", self.sceEdits[i]))
        fiber_rows: list[tuple[str, QWidget]] = [
            ("D_f", self.dfEdit),
            ("D_core_f", self.dcorefEdit),
        ]
        for i in range(MAX_NESTED_RINGS):
            fiber_rows.append((f"d_f,{i + 1}", self.dfEdits[i]))
            fiber_rows.append((f"t_f,{i + 1}", self.tfEdits[i]))

        self.caneBox = _form_box("Cane (stage E)", cane_rows)
        self.jacketBox = _form_box(
            "Jacket", [("OD_j", self.odjEdit), ("ID_j", self.idjEdit)]
        )
        self.factorsBox = _form_box("Scaling factors", factors_rows)
        self.fiberBox = _form_box("Desired fiber (stage I, optional)", fiber_rows)

        self.backcalcBox = QGroupBox("Back-calculation")
        backcalc_layout = QGridLayout(self.backcalcBox)
        backcalc_layout.addWidget(QLabel("Sub-mode"), 0, 0)
        backcalc_layout.addWidget(self.subModeCombo, 0, 1)
        backcalc_layout.addWidget(QLabel("t_sleeve"), 1, 0)
        backcalc_layout.addWidget(self.tSleeveEdit, 1, 1)
        backcalc_layout.setColumnStretch(1, 1)
        self.backcalcBox.setEnabled(False)
        self.tSleeveEdit.setEnabled(False)

        self.stackBox = QGroupBox("Stage-1 stack")
        stack_grid = QGridLayout(self.stackBox)
        stack_grid.addWidget(self.stackSpecCombo, 0, 0, 1, 2)
        stack_grid.addWidget(self.stackParamLabel, 1, 0)
        stack_grid.addWidget(self.stackParamEdit, 1, 1)
        stack_grid.addWidget(QLabel("S_ie_st"), 2, 0)
        stack_grid.addWidget(self.sieStEdit, 2, 1)
        for i in range(MAX_NESTED_RINGS):
            stack_grid.addWidget(QLabel(f"S_ce_st,{i + 1}"), 3 + i, 0)
            stack_grid.addWidget(self.scestEdits[i], 3 + i, 1)
        stack_buttons = QHBoxLayout()
        stack_buttons.addWidget(self.stackForwardButton)
        stack_buttons.addWidget(self.stackInverseButton)
        stack_grid.addLayout(stack_buttons, 6, 0, 1, 2)
        stack_grid.addWidget(self.stackResultLabel, 7, 0, 1, 2)
        stack_grid.setColumnStretch(1, 1)

        bridge_box = QGroupBox("Solver bridge")
        bridge_layout = QHBoxLayout(bridge_box)
        bridge_layout.addWidget(self.pullButton)
        bridge_layout.addWidget(self.pushButton)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(_form_box("Presets & mode", selector_rows))
        left_layout.addWidget(self.caneBox)
        left_layout.addWidget(self.jacketBox)
        left_layout.addWidget(self.factorsBox)
        left_layout.addWidget(self.fiberBox)
        left_layout.addWidget(self.backcalcBox)
        left_layout.addWidget(self.stackBox)
        left_layout.addWidget(bridge_box)
        left_layout.addWidget(self.statusLabel)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(flags_box)
        right_layout.addWidget(self.tableBox)
        canvases = QHBoxLayout()
        canvases.addWidget(self.caneCanvas)
        canvases.addWidget(self.fiberCanvas)
        right_layout.addLayout(canvases, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(wrap_scroll(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        # -- signal wiring ----------------------------------------------------
        self.presetCombo.currentIndexChanged.connect(self._on_preset)
        self.unitCombo.currentTextChanged.connect(self._on_unit)
        self.modeCombo.currentTextChanged.connect(self._on_mode)
        self.couplingCombo.currentTextChanged.connect(
            lambda _text: self._recompute()
        )
        self.subModeCombo.currentTextChanged.connect(self._on_submode)
        self.stackSpecCombo.currentTextChanged.connect(self._on_stack_spec)
        self.ncapSpinBox.valueChanged.connect(
            lambda _value: self._recompute()
        )
        self.nestedSpinBox.valueChanged.connect(self._on_nested)
        for edit in (
            self.odcEdit, self.dcorecEdit, *self.dcEdits, *self.tcEdits,
            self.odjEdit, self.idjEdit, self.sjEdit, self.sieEdit,
            *self.sceEdits, self.dfEdit, self.dcorefEdit, *self.dfEdits,
            *self.tfEdits, self.tSleeveEdit, self.stackParamEdit,
            self.sieStEdit, *self.scestEdits,
        ):
            edit.textChanged.connect(lambda _text: self._recompute())
        self.stackForwardButton.clicked.connect(self._on_stack_forward)
        self.stackInverseButton.clicked.connect(self._on_stack_inverse)
        self.pullButton.clicked.connect(self.pull_requested.emit)
        self.pushButton.clicked.connect(self.push_requested.emit)

        self._on_preset(0)

    # -- helpers -----------------------------------------------------------
    @contextlib.contextmanager
    def _guard(self):
        self._updating = True
        try:
            yield
        finally:
            self._updating = False

    def coupling(self) -> CouplingMode:
        """The currently selected jacket <-> cane coupling mode."""
        return _COUPLING_FOR_TEXT.get(
            self.couplingCombo.currentText(), CO_FED_VACUUM
        )

    def _lscale(self) -> float:
        """Display-unit scale: metres per displayed unit (1e-3 / 1e-6)."""
        return 1e-6 if self.unitCombo.currentText() == "µm" else 1e-3

    def _unit_name(self) -> str:
        return self.unitCombo.currentText() or "mm"

    def _read_cane(self) -> CaneGeometry:
        sc = self._lscale()
        nb = self.nestedSpinBox.value()
        return CaneGeometry(
            OD_c=read_double(self.odcEdit) * sc,
            D_core_c=read_double(self.dcorecEdit) * sc,
            d_c=tuple(read_double(e) * sc for e in self.dcEdits[:nb]),
            t_c=tuple(read_double(e) * sc for e in self.tcEdits[:nb]),
            n_cap=self.ncapSpinBox.value(),
        )

    def _read_jacket(self) -> JacketGeometry:
        sc = self._lscale()
        return JacketGeometry(
            OD_j=read_double(self.odjEdit) * sc,
            ID_j=read_double(self.idjEdit) * sc,
        )

    def _read_factors(self) -> ScalingFactors:
        nb = self.nestedSpinBox.value()
        return ScalingFactors(
            S_j=read_double(self.sjEdit),
            S_ie=read_double(self.sieEdit),
            S_ce=tuple(read_double(e) for e in self.sceEdits[:nb]),
            coupling=self.coupling(),
        )

    def _read_fiber(self) -> FiberGeometry:
        sc = self._lscale()
        nb = self.nestedSpinBox.value()
        return FiberGeometry(
            D_f=read_double(self.dfEdit) * sc,
            D_core_f=read_double(self.dcorefEdit) * sc,
            d_f=tuple(read_double(e) * sc for e in self.dfEdits[:nb]),
            t_f=tuple(read_double(e) * sc for e in self.tfEdits[:nb]),
        )

    def _fiber_given(self) -> FiberGeometry | None:
        """The desired fiber when every field is finite and positive."""
        fiber = self._read_fiber()
        values = (fiber.D_f, fiber.D_core_f) + fiber.d_f + fiber.t_f
        if all(math.isfinite(v) and v > 0.0 for v in values):
            return fiber
        return None

    def _fill_cane(
        self,
        cane: CaneGeometry,
        odj: float | None = None,
        sj: float | None = None,
    ) -> None:
        """Write a solved cane (and optional jacket / S_j) into the edits."""
        sc = self._lscale()
        with self._guard():
            self.odcEdit.setText(f"{cane.OD_c / sc:.10g}")
            self.dcorecEdit.setText(f"{cane.D_core_c / sc:.10g}")
            for i, (d, t) in enumerate(
                zip(cane.d_c, cane.t_c, strict=True)
            ):
                self.dcEdits[i].setText(f"{d / sc:.10g}")
                self.tcEdits[i].setText(f"{t / sc:.10g}")
            if odj is not None:
                self.odjEdit.setText(f"{odj / sc:.10g}")
            if sj is not None:
                self.sjEdit.setText(f"{sj:.10g}")

    def _apply_gating(self, nb: int) -> None:
        """Enable only the first ``nb`` ring rows of every ring group."""
        nb = max(1, min(int(nb), MAX_NESTED_RINGS))
        for group in (
            self.dcEdits,
            self.tcEdits,
            self.sceEdits,
            self.dfEdits,
            self.tfEdits,
            self.scestEdits,
        ):
            for i, edit in enumerate(group):
                edit.setEnabled(i < nb)

    def _refresh_titles(self) -> None:
        unit = self._unit_name()
        backcalc = self.modeCombo.currentText() == MODE_BACKCALC
        self.caneBox.setTitle(f"Cane (stage E) [{unit}]")
        self.jacketBox.setTitle(f"Jacket [{unit}]")
        self.fiberBox.setTitle(
            "Target fiber (stage I)"
            if backcalc
            else "Desired fiber (stage I, optional)"
        )
        self.tableBox.setTitle(f"Stage table E -> I [{unit}]")

    def _refresh_stack_label(self) -> None:
        spec = self.stackSpecCombo.currentText()
        if spec == "S_stack":
            self.stackParamLabel.setText("S_stack (–)")
        else:
            self.stackParamLabel.setText(f"{spec} [{self._unit_name()}]")

    def _length_edits(self) -> list[QLineEdit]:
        edits = [
            self.odcEdit,
            self.dcorecEdit,
            *self.dcEdits,
            *self.tcEdits,
            self.odjEdit,
            self.idjEdit,
            self.dfEdit,
            self.dcorefEdit,
            *self.dfEdits,
            *self.tfEdits,
            self.tSleeveEdit,
        ]
        if self.stackSpecCombo.currentText() != "S_stack":
            edits.append(self.stackParamEdit)
        return edits

    @staticmethod
    def _scale_edit(edit: QLineEdit, factor: float) -> None:
        edit.setText(f"{read_double(edit) * factor:.10g}")

    # -- slots ---------------------------------------------------------------
    def _on_unit(self, unit: str) -> None:
        if self._updating:
            return
        factor = 1e3 if unit == "µm" else 1e-3
        for edit in self._length_edits():
            self._scale_edit(edit, factor)
        self._refresh_titles()
        self._refresh_stack_label()
        self._recompute()

    def _on_mode(self, mode: str) -> None:
        backcalc = mode == MODE_BACKCALC
        self.backcalcBox.setEnabled(backcalc)
        self._refresh_titles()
        self._recompute()

    def _on_submode(self, submode: str) -> None:
        self.tSleeveEdit.setEnabled(submode == SUBMODE_MANUAL_SJ)
        self._recompute()

    def _on_stack_spec(self, _spec: str) -> None:
        self._refresh_stack_label()

    def _on_nested(self, nb: int) -> None:
        self._apply_gating(nb)
        self._recompute()

    def _on_preset(self, index: int) -> None:
        presets = get_scaling_presets()
        if not 0 <= index < len(presets):
            return
        preset = presets[index]
        sc = self._lscale()
        fiber = preset.desired_fiber
        if fiber is None:
            try:
                fiber = compute_forward_scaling(
                    preset.cane, preset.jacket, preset.factors
                ).achieved_fiber
            except ScalingError:
                fiber = None
        with self._guard():
            self.couplingCombo.setCurrentText(
                _TEXT_FOR_COUPLING[preset.factors.coupling]
            )
            self.ncapSpinBox.setValue(preset.cane.n_cap)
            self.nestedSpinBox.setValue(len(preset.cane.d_c))
            self.odcEdit.setText(f"{preset.cane.OD_c / sc:.10g}")
            self.dcorecEdit.setText(f"{preset.cane.D_core_c / sc:.10g}")
            for i, (d, t) in enumerate(
                zip(preset.cane.d_c, preset.cane.t_c, strict=True)
            ):
                self.dcEdits[i].setText(f"{d / sc:.10g}")
                self.tcEdits[i].setText(f"{t / sc:.10g}")
            self.odjEdit.setText(f"{preset.jacket.OD_j / sc:.10g}")
            self.idjEdit.setText(f"{preset.jacket.ID_j / sc:.10g}")
            self.sjEdit.setText(f"{preset.factors.S_j:.10g}")
            self.sieEdit.setText(f"{preset.factors.S_ie:.10g}")
            for i, s in enumerate(preset.factors.S_ce):
                self.sceEdits[i].setText(f"{s:.10g}")
            if fiber is not None:
                self.dfEdit.setText(f"{fiber.D_f / sc:.10g}")
                self.dcorefEdit.setText(f"{fiber.D_core_f / sc:.10g}")
                for i, (d, t) in enumerate(
                    zip(fiber.d_f, fiber.t_f, strict=True)
                ):
                    self.dfEdits[i].setText(f"{d / sc:.10g}")
                    self.tfEdits[i].setText(f"{t / sc:.10g}")
        self._refresh_titles()
        self._apply_gating(len(preset.cane.d_c))
        self._recompute()

    def _on_stack_forward(self) -> None:
        cane = self._read_cane()
        sc = self._lscale()
        nb = self.nestedSpinBox.value()
        sie_st = read_double(self.sieStEdit)
        scest = tuple(read_double(e) for e in self.scestEdits[:nb])
        spec = self.stackSpecCombo.currentText()
        param = read_double(self.stackParamEdit)
        try:
            if spec == "S_stack":
                stack = cane_to_stack(
                    cane, sie_st, scest, S_stack=param
                )
            elif spec == "ID_stack":
                stack = cane_to_stack(
                    cane, sie_st, scest, ID_stack=param * sc
                )
            else:
                stack = cane_to_stack(
                    cane, sie_st, scest, OD_stack=param * sc
                )
        except ScalingError as exc:
            self.stackResultLabel.setText(f"Stack error: {exc}")
            return
        self._stack = stack
        self._render_stack(stack)

    def _on_stack_inverse(self) -> None:
        if self._stack is None:
            self.stackResultLabel.setText(
                "Run 'Cane -> Stack' before inverting."
            )
            return
        try:
            cane = stack_to_cane(self._stack)
        except ScalingError as exc:
            self.stackResultLabel.setText(f"Stack error: {exc}")
            return
        self._fill_cane(cane)
        self._apply_gating(len(cane.d_c))
        self._recompute()

    def _render_stack(self, stack: StackGeometry) -> None:
        inv = 1.0 / self._lscale()
        parts = (
            f"S_stack = {stack.S_stack:.6g}",
            f"ID_stack = {stack.ID_stack * inv:.6g}",
            f"OD_stack = {stack.OD_stack * inv:.6g}",
            f"ζ_st = {stack.zeta_st * inv:.6g}",
            f"D_core_st = {stack.D_core_st * inv:.6g}",
            f"d_st,1 = {stack.d_st[0] * inv:.6g}",
            f"t_st,1 = {stack.t_st[0] * inv:.6g}",
        )
        self.stackResultLabel.setText("Stack: " + ", ".join(parts))

    # -- computation ----------------------------------------------------------
    def _recompute(self) -> None:
        if self._updating:
            return
        result = self._compute_scaling()
        if result is None:
            for badge in self._flag_badges.values():
                badge.set_state(None)
            self.allOkBadge.set_state(None)
            return
        self._result = result
        self._render_result(result)

    def _compute_scaling(self) -> ScalingResult | None:
        try:
            if self.modeCombo.currentText() == MODE_BACKCALC:
                return self._back_calc()
            return compute_forward_scaling(
                self._read_cane(),
                self._read_jacket(),
                self._read_factors(),
                desired_fiber=self._fiber_given(),
            )
        except ScalingError as exc:
            self._show_error(exc)
            return None

    def _back_calc(self) -> ScalingResult:
        fiber = self._read_fiber()
        jacket = self._read_jacket()
        sc = self._lscale()
        nb = self.nestedSpinBox.value()
        sie = read_double(self.sieEdit)
        sce = tuple(read_double(e) for e in self.sceEdits[:nb])
        coupling = self.coupling()
        n_cap = self.ncapSpinBox.value()
        od_c_read = read_double(self.odcEdit) * sc
        manual = self.subModeCombo.currentText() == SUBMODE_MANUAL_SJ
        sub_mode = (
            InverseSubMode.MANUAL_SJ if manual else InverseSubMode.FIXED_JACKET
        )
        if sub_mode is InverseSubMode.FIXED_JACKET:
            cane, S_c = solve_inverse_cane(
                fiber, jacket, sie, sce, sub_mode, coupling,
                OD_c=od_c_read, n_cap=n_cap,
            )
            odj = jacket.OD_j
            sj_edit = (
                S_c if coupling is CO_FED_VACUUM
                else S_c * jacket.ID_j / od_c_read
            )
        else:
            S_j = read_double(self.sjEdit)
            t_sleeve = read_double(self.tSleeveEdit) * sc
            if t_sleeve > 0.0:
                cane, S_c = solve_inverse_cane(
                    fiber, jacket, sie, sce, sub_mode, coupling,
                    t_sleeve=t_sleeve, S_j=S_j, n_cap=n_cap,
                )
            else:
                cane, S_c = solve_inverse_cane(
                    fiber, jacket, sie, sce, sub_mode, coupling,
                    OD_c=od_c_read, S_j=S_j, n_cap=n_cap,
                )
            odj = inverse_jacket_od(
                fiber, jacket.ID_j, cane.OD_c, S_j, sie, coupling
            )
            sj_edit = S_j
        self._fill_cane(
            cane,
            odj=(odj if sub_mode is InverseSubMode.MANUAL_SJ else None),
            sj=sj_edit,
        )
        jacket2 = JacketGeometry(OD_j=odj, ID_j=jacket.ID_j)
        factors = ScalingFactors(
            S_j=sj_edit, S_ie=sie, S_ce=sce, coupling=coupling
        )
        return compute_forward_scaling(
            cane, jacket2, factors, desired_fiber=fiber
        )

    def _render_result(self, result: ScalingResult) -> None:
        feasibility = result.feasibility
        for field, _label in _FLAGS:
            self._flag_badges[field].set_state(
                getattr(feasibility, field)
            )
        self.allOkBadge.set_state(feasibility.all_ok)
        if feasibility.all_ok:
            self._set_status(
                f"S_c = {result.S_c:.6g}; stages E -> I consistent.", True
            )
        else:
            self._set_status(
                "Scaling warnings: " + "; ".join(feasibility.warnings),
                False,
            )
        self._render_table(result)
        self._update_canvases(result)

    def _render_table(self, result: ScalingResult) -> None:
        inv = 1.0 / self._lscale()
        for r, (_label, attr) in enumerate(STAGE_ROWS):
            is_length = attr in _LENGTH_ATTRS
            for c, name in enumerate(_STAGE_COLS, start=1):
                value = getattr(result.stages[name], attr)
                if isinstance(value, tuple):
                    value = value[0] if value else math.nan
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = math.nan
                if is_length:
                    number *= inv
                text = f"{number:.6g}" if math.isfinite(number) else "--"
                item = self.stageTable.item(r, c)
                if item is None:
                    self.stageTable.setItem(r, c, QTableWidgetItem(text))
                else:
                    item.setText(text)

    def _update_canvases(self, result: ScalingResult) -> None:
        unit = self._unit_name()
        stage_e = result.stages["E"]
        stage_i = result.stages["I"]
        self.caneCanvas.set_static_geometry(
            f"Cane (stage E) [{unit}]",
            r_outer=stage_e.OD_j * 5e5,
            r_bore=stage_e.ID_j * 5e5,
            r_env=stage_e.zeta * 5e5,
            r_core=stage_e.D_core * 5e5,
            ring_outer=tuple(D * 5e5 for D in stage_e.Delta),
            ring_bore=tuple(d * 5e5 for d in stage_e.d),
            n_cap=result.n_cap,
        )
        self.fiberCanvas.set_static_geometry(
            f"Final fiber (stage I) [{unit}]",
            r_outer=stage_i.OD_j * 5e5,
            r_bore=stage_i.zeta * 5e5,
            r_env=stage_i.zeta * 5e5,
            r_core=stage_i.D_core * 5e5,
            ring_outer=tuple(D * 5e5 for D in stage_i.Delta),
            ring_bore=tuple(d * 5e5 for d in stage_i.d),
            n_cap=result.n_cap,
        )

    def _set_status(self, text: str, ok: bool) -> None:
        color = GREEN if ok else RED
        self.statusLabel.setText(text)
        self.statusLabel.setStyleSheet(
            f"color: {color}; border: 1px solid {color}; "
            "border-radius: 4px; padding: 4px 8px;"
        )

    def _show_error(self, exc: Exception) -> None:
        self._set_status(f"Scaling error: {exc}", False)

    # -- solver bridge ---------------------------------------------------------
    def build_handoff(self) -> SolverHandoff:
        """The current cane / jacket / fiber as solver-tab inputs."""
        fiber = self._fiber_given()
        if fiber is None and self._result is not None:
            fiber = self._result.fiber
        if fiber is None:
            raise ScalingError(
                "No fiber geometry available for the solver handoff."
            )
        return scaling_to_solver_inputs(
            self._read_cane(), self._read_jacket(), fiber, self.coupling()
        )

    def apply_pulled(self, state) -> None:
        """Fill this tab from the solver tabs (via ``PulledScalingState``)."""
        with self._guard():
            self.unitCombo.setCurrentText("mm")
            self.couplingCombo.setCurrentText(
                _TEXT_FOR_COUPLING[state.coupling]
            )
            self.ncapSpinBox.setValue(int(state.n_cap))
            self.nestedSpinBox.setValue(int(state.nb_nested))
            sc = 1e-3
            self.odcEdit.setText(f"{state.cane.OD_c / sc:.10g}")
            self.dcorecEdit.setText(f"{state.cane.D_core_c / sc:.10g}")
            for i, (d, t) in enumerate(
                zip(state.cane.d_c, state.cane.t_c, strict=True)
            ):
                self.dcEdits[i].setText(f"{d / sc:.10g}")
                self.tcEdits[i].setText(f"{t / sc:.10g}")
            self.odjEdit.setText(f"{state.jacket.OD_j / sc:.10g}")
            self.idjEdit.setText(f"{state.jacket.ID_j / sc:.10g}")
            if state.fiber is not None:
                self.dfEdit.setText(f"{state.fiber.D_f / sc:.10g}")
                self.dcorefEdit.setText(
                    f"{state.fiber.D_core_f / sc:.10g}"
                )
                for i, (d, t) in enumerate(
                    zip(state.fiber.d_f, state.fiber.t_f, strict=True)
                ):
                    self.dfEdits[i].setText(f"{d / sc:.10g}")
                    self.tfEdits[i].setText(f"{t / sc:.10g}")
            if state.factors is not None:
                self.sjEdit.setText(f"{state.factors.S_j:.10g}")
                self.sieEdit.setText(f"{state.factors.S_ie:.10g}")
                for i, s in enumerate(state.factors.S_ce):
                    self.sceEdits[i].setText(f"{s:.10g}")
        self._refresh_titles()
        self._apply_gating(int(state.nb_nested))
        self._recompute()

    # -- persistence ------------------------------------------------------------
    def collect_state(self) -> dict:
        """GUI-only values that must survive a config round-trip."""
        return {
            "unit": self.unitCombo.currentText(),
            "mode": self.modeCombo.currentText(),
            "submode": self.subModeCombo.currentText(),
            "coupling": self.couplingCombo.currentText(),
            "ncap": self.ncapSpinBox.value(),
            "nested": self.nestedSpinBox.value(),
            "odc": self.odcEdit.text(),
            "dcorec": self.dcorecEdit.text(),
            "dc": [e.text() for e in self.dcEdits],
            "tc": [e.text() for e in self.tcEdits],
            "odj": self.odjEdit.text(),
            "idj": self.idjEdit.text(),
            "sj": self.sjEdit.text(),
            "sie": self.sieEdit.text(),
            "sce": [e.text() for e in self.sceEdits],
            "df": self.dfEdit.text(),
            "dcoref": self.dcorefEdit.text(),
            "df_": [e.text() for e in self.dfEdits],
            "tf_": [e.text() for e in self.tfEdits],
            "tsleeve": self.tSleeveEdit.text(),
            "stack_spec": self.stackSpecCombo.currentText(),
            "stack_param": self.stackParamEdit.text(),
            "sie_st": self.sieStEdit.text(),
            "scest": [e.text() for e in self.scestEdits],
        }

    def apply_state(self, state: dict) -> None:
        """Restore the state captured by :meth:`collect_state`."""
        with self._guard():
            self.unitCombo.setCurrentText(state.get("unit", "mm"))
            self.modeCombo.setCurrentText(state.get("mode", MODE_FORWARD))
            self.subModeCombo.setCurrentText(
                state.get("submode", SUBMODE_FIXED_JACKET)
            )
            self.couplingCombo.setCurrentText(
                state.get("coupling", _COUPLING_ITEMS[0][0])
            )
            self.ncapSpinBox.setValue(int(state.get("ncap", 6)))
            self.nestedSpinBox.setValue(int(state.get("nested", 1)))
            for key, edit in (
                ("odc", self.odcEdit),
                ("dcorec", self.dcorecEdit),
                ("odj", self.odjEdit),
                ("idj", self.idjEdit),
                ("sj", self.sjEdit),
                ("sie", self.sieEdit),
                ("df", self.dfEdit),
                ("dcoref", self.dcorefEdit),
                ("tsleeve", self.tSleeveEdit),
                ("stack_param", self.stackParamEdit),
                ("sie_st", self.sieStEdit),
            ):
                text = state.get(key)
                if isinstance(text, str):
                    edit.setText(text)
            for key, edits in (
                ("dc", self.dcEdits),
                ("tc", self.tcEdits),
                ("sce", self.sceEdits),
                ("df_", self.dfEdits),
                ("tf_", self.tfEdits),
                ("scest", self.scestEdits),
            ):
                for i, text in enumerate(
                    state.get(key, [])[: len(edits)]
                ):
                    if isinstance(text, str):
                        edits[i].setText(text)
            spec = state.get("stack_spec")
            if isinstance(spec, str):
                self.stackSpecCombo.setCurrentText(spec)
            backcalc = self.modeCombo.currentText() == MODE_BACKCALC
            self.backcalcBox.setEnabled(backcalc)
            self.tSleeveEdit.setEnabled(
                self.subModeCombo.currentText() == SUBMODE_MANUAL_SJ
            )
        self._refresh_titles()
        self._refresh_stack_label()
        self._apply_gating(self.nestedSpinBox.value())
        self._recompute()
