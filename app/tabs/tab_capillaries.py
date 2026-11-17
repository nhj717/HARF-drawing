"""Tab 3: capillary structure (Ncap azimuthal capillaries, nested rings).

Split layout: the left column hosts the spinners, the Ring 1..3
geometry group boxes (with their ratio badges and per-ring formulas);
the right column hosts a ``propertyCombo`` (``Surface tension``,
``Viscosity``) and the ``PlotCanvas`` (MATLAB ``UIAxes_5``) rendering
immediately at startup over ``T in [250, 2500] K`` (250 K ticks) with
the Jacket + ``Cap #1..3`` neon multi-curve overlay, live-updating on
any Jacket/Capillary formula or ``NestedCapillaries`` change.

``Nnested`` in ``1..3`` dynamically enables the Ring 2 / Ring 3 widget
groups.  Contact badges (``Rc01/Rcontact``, ``Dc02/dc01``,
``Dc03/dc02``) turn red at ``>= 1`` exactly like the MATLAB v1.5
alerts.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.widgets.plot_canvas import PlotCanvas
from app.widgets.validators import (
    RatioBadge,
    attach_formula_validation,
    make_double_edit,
    read_double,
)
from src.models import (
    DEFAULT_GAMMA,
    DEFAULT_VISCOSITY,
    MAX_NESTED_RINGS,
    RingSpec,
    SimulationInputs,
    default_ring_specs,
    mm_to_m,
    r_contact_radius,
)

__all__ = ["CapillariesTab"]

_CAP_PROPERTY_ITEMS = ("Surface tension", "Viscosity")


class CapillariesTab(QWidget):
    """Capillary count, nesting depth and per-ring geometry/laws."""

    inputs_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._jacket_materials: tuple[str, str] = (
            DEFAULT_GAMMA,
            DEFAULT_VISCOSITY,
        )
        self.ncapSpinBox = QSpinBox()
        self.ncapSpinBox.setRange(1, 12)
        self.ncapSpinBox.setValue(2)
        self.nestedSpinBox = QSpinBox()
        self.nestedSpinBox.setRange(1, MAX_NESTED_RINGS)
        self.nestedSpinBox.setValue(1)

        self.contactBadge = RatioBadge(
            "Rc01 / Rcontact",
            "Ring-1 outer radius over the packing contact radius "
            "(red when the capillaries overlap).",
        )

        top_box = QGroupBox("Structure")
        form = QFormLayout(top_box)
        form.addRow("Number of capillaries Ncap", self.ncapSpinBox)
        form.addRow("Nested rings Nnested", self.nestedSpinBox)
        form.addRow(self.contactBadge)

        defaults = default_ring_specs(MAX_NESTED_RINGS)
        self.ring_boxes: list[QGroupBox] = []
        self.dcEdits: list[object] = []
        self.dccEdits: list[object] = []
        self.gammaEdits: list[object] = []
        self.viscEdits: list[object] = []
        self.ratioBadges: list[RatioBadge] = []
        self.nestedBadges: list[RatioBadge] = []

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(top_box)
        for i in range(MAX_NESTED_RINGS):
            box, badge, nested_badge = self._build_ring_group(i, defaults[i])
            self.ring_boxes.append(box)
            self.ratioBadges.append(badge)
            if nested_badge is not None:
                self.nestedBadges.append(nested_badge)
            left_layout.addWidget(box)
        left_layout.addStretch(1)

        preview_box = QGroupBox("Material property (UIAxes_5)")
        self.propertyCombo = QComboBox()
        self.propertyCombo.addItems(_CAP_PROPERTY_ITEMS)
        self.propertyCombo.currentTextChanged.connect(self._auto_refresh_plot)
        self.plotCanvas = PlotCanvas()
        row = QHBoxLayout()
        row.addWidget(self.propertyCombo)
        row.addStretch(1)
        pv = QVBoxLayout(preview_box)
        pv.addLayout(row)
        pv.addWidget(self.plotCanvas)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(preview_box)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([460, 600])

        layout = QHBoxLayout(self)
        layout.addWidget(self.splitter)

        self.nestedSpinBox.valueChanged.connect(self.apply_nested_state)
        for spin in (self.ncapSpinBox, self.nestedSpinBox):
            spin.valueChanged.connect(self.inputs_changed)
        self.apply_nested_state()
        self.refresh_material_plot()

    def _build_ring_group(self, index: int, spec: RingSpec):
        number = index + 1
        dc_edit = make_double_edit(
            spec.Rc0 * 2e3, 0.0, 1e3,
            tooltip=f"Ring {number} outer diameter Dc0 [{number}] [mm]",
        )
        dcc_edit = make_double_edit(
            spec.rc0 * 2e3, 0.0, 1e3,
            tooltip=f"Ring {number} bore diameter dc0 [{number}] [mm]",
        )
        gamma_edit = make_double_edit(0.3)
        gamma_edit.setText(DEFAULT_GAMMA)
        visc_edit = make_double_edit(1.0)
        visc_edit.setText(DEFAULT_VISCOSITY)
        attach_formula_validation(gamma_edit, ("T",))
        attach_formula_validation(visc_edit, ("T",))
        badge = RatioBadge(
            f"dc0{number} / Dc0{number}",
            f"Ring {number} bore/outer diameter ratio (must stay < 1).",
        )
        nested_badge = None
        if number >= 2:
            host = number - 1
            nested_badge = RatioBadge(
                f"Dc0{number} / dc0{host}",
                f"Ring {number} outer diameter over ring {host} bore "
                "(red when not nested).",
            )

        box = QGroupBox(f"Ring {number}")
        form = QFormLayout(box)
        form.addRow(f"Outer diameter Dc0 [{number}] [mm]", dc_edit)
        form.addRow(f"Bore diameter dc0 [{number}] [mm]", dcc_edit)
        form.addRow(f"Surface tension gamma_{number}(T) [N/m]", gamma_edit)
        form.addRow(f"Viscosity mu_{number}(T) [Pa s]", visc_edit)
        form.addRow(badge)
        if nested_badge is not None:
            form.addRow(nested_badge)

        for edit in (dc_edit, dcc_edit):
            edit.textChanged.connect(self._update_badges)
            edit.textChanged.connect(self.inputs_changed)
        for edit in (gamma_edit, visc_edit):
            edit.textChanged.connect(self.inputs_changed)

        self.dcEdits.append(dc_edit)
        self.dccEdits.append(dcc_edit)
        self.gammaEdits.append(gamma_edit)
        self.viscEdits.append(visc_edit)
        return box, badge, nested_badge

    # -- dynamic state --------------------------------------------------------
    def apply_nested_state(self) -> None:
        """Enable exactly the first ``Nnested`` ring groups."""
        nb = self.nestedSpinBox.value()
        for i, box in enumerate(self.ring_boxes):
            box.setEnabled(i < nb)
            box.setVisible(True)
        self._update_badges()

    @property
    def nb_nested(self) -> int:
        return self.nestedSpinBox.value()

    # -- badges ---------------------------------------------------------------
    def _update_badges(self) -> None:
        for i in range(MAX_NESTED_RINGS):
            dcc = read_double(self.dccEdits[i], 0.0)
            dc = read_double(self.dcEdits[i], 0.0)
            self.ratioBadges[i].set_ratio(dcc, dc)
        for j, badge in enumerate(self.nestedBadges):
            number = j + 2  # badge j compares ring (j+2) with ring (j+1)
            dc = read_double(self.dcEdits[number - 1], 0.0)
            dcc_host = read_double(self.dccEdits[number - 2], 0.0)
            badge.set_ratio(dc, dcc_host)

    def update_contact_badge(self, r0: float, n_cap: int) -> None:
        """Refresh ``Rc01 / Rcontact`` from the jacket bore and packing."""
        rc01 = mm_to_m(read_double(self.dcEdits[0], 0.0)) / 2.0
        contact = r_contact_radius(r0, n_cap)
        self.contactBadge.set_ratio(rc01, contact)

    # -- live material comparison (UIAxes_5) ------------------------------------
    def material_exprs(self) -> tuple[tuple[str, str], ...]:
        """``((gamma_i, visc_i), ...)`` for every enabled nested ring."""
        return tuple(
            (
                self.gammaEdits[i].text().strip(),
                self.viscEdits[i].text().strip(),
            )
            for i in range(self.nb_nested)
        )

    def _auto_refresh_plot(self, *_args) -> None:
        self.refresh_material_plot()

    def refresh_material_plot(
        self, jacket_materials: tuple[str, str] | None = None
    ) -> None:
        """Plot the selected property over ``[250, 2500] K``.

        ``jacket_materials`` is ``(gamma_expr, viscosity_expr)`` of the
        jacket; the Jacket curve is overlaid with ``Cap #1..3``.
        """
        if jacket_materials is not None:
            self._jacket_materials = (
                str(jacket_materials[0]),
                str(jacket_materials[1]),
            )
        prop = self.propertyCombo.currentText()
        if prop == "Viscosity":
            jacket_expr = self._jacket_materials[1]
        else:
            jacket_expr = self._jacket_materials[0]
        cap_exprs = tuple(
            m[1] if prop == "Viscosity" else m[0]
            for m in self.material_exprs()
        )
        self.plotCanvas.plot_material_comparison(
            prop, jacket_expr, cap_exprs, len(cap_exprs)
        )

    # -- data exchange -----------------------------------------------------------
    def to_fields(self) -> dict:
        nb = self.nb_nested
        rings = []
        for i in range(nb):
            rings.append(
                RingSpec(
                    Rc0=mm_to_m(read_double(self.dcEdits[i], 0.48)) / 2.0,
                    rc0=mm_to_m(read_double(self.dccEdits[i], 0.3936)) / 2.0,
                    gamma_expr=self.gammaEdits[i].text().strip(),
                    viscosity_expr=self.viscEdits[i].text().strip(),
                )
            )
        return {
            "rings": tuple(rings),
            "n_capillaries": int(self.ncapSpinBox.value()),
        }

    def apply_inputs(self, inputs: SimulationInputs) -> None:
        self.ncapSpinBox.setValue(int(inputs.n_capillaries))
        self.nestedSpinBox.setValue(len(inputs.rings))
        for i, ring in enumerate(inputs.rings):
            self.dcEdits[i].setText(f"{ring.Rc0 * 2e3:.10g}")
            self.dccEdits[i].setText(f"{ring.rc0 * 2e3:.10g}")
            self.gammaEdits[i].setText(ring.gamma_expr)
            self.viscEdits[i].setText(ring.viscosity_expr)
        self.apply_nested_state()
