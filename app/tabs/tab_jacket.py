"""Tab 2: jacket geometry and user-editable material laws.

Right-side live material-property preview (``UIAxes_4``): a
``propertyCombo`` selects one of ``Volumic mass``, ``Surface tension``,
``Specific heat``, ``Viscosity`` or ``Emissivity`` and the ``PlotCanvas``
renders it over ``T in [250, 2500] K`` (250 K ticks).  ``Viscosity`` and
``Surface tension`` overlay every enabled capillary ring (``Cap #1..3``,
neon pink/green/amber dashed) on the cyan Jacket curve, auto-refreshing
live on any Jacket/Capillary formula edit or ``NestedCapillaries``
change.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
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
    DEFAULT_ALPHA,
    DEFAULT_CP,
    DEFAULT_GAMMA,
    DEFAULT_RHO,
    DEFAULT_VISCOSITY,
    SimulationInputs,
    mm_to_m,
)

__all__ = ["JacketTab"]

_PROPERTY_ITEMS = (
    "Volumic mass",
    "Surface tension",
    "Specific heat",
    "Viscosity",
    "Emissivity",
)


class JacketTab(QWidget):
    """Jacket (outer preform) geometry and material expressions."""

    geometry_changed = Signal()
    expressions_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cap_materials: tuple[tuple[str, str], ...] = ()
        self.D0Edit = make_double_edit(
            8.3556, 0.0, 1e3, tooltip="Jacket outer diameter D0 [mm]"
        )
        self.d0Edit = make_double_edit(
            2.4, 0.0, 1e3, tooltip="Jacket bore diameter d0 [mm]"
        )
        self.dRatioBadge = RatioBadge(
            "d0 / D0", "Jacket bore/outer diameter ratio (must stay < 1)."
        )

        geom_box = QGroupBox("Jacket preform geometry")
        form = QFormLayout(geom_box)
        form.addRow("Outer diameter D0 [mm]", self.D0Edit)
        form.addRow("Bore diameter d0 [mm]", self.d0Edit)
        form.addRow(self.dRatioBadge)

        self.rhoEdit = make_double_edit(
            2200.0, tooltip="Jacket density rho(T) [kg m-3]"
        )
        self.rhoEdit.setText(DEFAULT_RHO)
        self.gammaEdit = make_double_edit(
            0.3, tooltip="Surface tension gamma(T) [N m-1]"
        )
        self.gammaEdit.setText(DEFAULT_GAMMA)
        self.cpEdit = make_double_edit(
            1000.0, tooltip="Heat capacity Cp(T) [J kg-1 K-1]"
        )
        self.cpEdit.setText(DEFAULT_CP)
        self.viscosityEdit = make_double_edit(
            1.0, tooltip="Dynamic viscosity mu(T) [Pa s] (positive)"
        )
        self.viscosityEdit.setText(DEFAULT_VISCOSITY)
        self.alphaEdit = make_double_edit(
            0.885, tooltip="Total hemispherical absorptivity alpha(T)"
        )
        self.alphaEdit.setText(DEFAULT_ALPHA)

        material_box = QGroupBox("Material laws (expressions of T in K)")
        form = QFormLayout(material_box)
        form.addRow("Viscosity mu(T) [Pa s]", self.viscosityEdit)
        form.addRow("Surface tension gamma(T) [N/m]", self.gammaEdit)
        form.addRow("Density rho(T) [kg/m^3]", self.rhoEdit)
        form.addRow("Heat capacity Cp(T) [J/(kg K)]", self.cpEdit)
        form.addRow("Absorptivity alpha(T)", self.alphaEdit)

        preview_box = QGroupBox("Material property (UIAxes_4)")
        self.propertyCombo = QComboBox()
        self.propertyCombo.addItems(_PROPERTY_ITEMS)
        self.propertyCombo.currentTextChanged.connect(self._auto_refresh)
        self.refreshButton = QPushButton("Refresh preview")
        self.refreshButton.setObjectName("btn_accent")
        self.refreshButton.clicked.connect(self._auto_refresh)
        self.plotCanvas = PlotCanvas()

        row = QHBoxLayout()
        row.addWidget(self.propertyCombo)
        row.addWidget(self.refreshButton)
        row.addStretch(1)
        inner = QVBoxLayout(preview_box)
        inner.addLayout(row)
        inner.addWidget(self.plotCanvas)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(geom_box)
        left_layout.addWidget(material_box)
        left_layout.addStretch(1)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(preview_box)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([420, 640])

        layout = QHBoxLayout(self)
        layout.addWidget(self.splitter)

        for edit in (self.D0Edit, self.d0Edit):
            edit.textChanged.connect(self._update_geometry_badge)
            edit.textChanged.connect(self.geometry_changed)
        for edit in (
            self.rhoEdit, self.gammaEdit,
            self.cpEdit, self.viscosityEdit, self.alphaEdit,
        ):
            attach_formula_validation(edit, ("T",))
            edit.textChanged.connect(self.expressions_changed)

        self._update_geometry_badge()
        self.refresh_material_comparison()

    # -- helpers -------------------------------------------------------------
    def _update_geometry_badge(self) -> None:
        d0 = read_double(self.d0Edit, 2.4)
        D0 = read_double(self.D0Edit, 8.3556)
        self.dRatioBadge.set_ratio(d0, D0)

    def _edit_for(self, prop: str):
        return {
            "Viscosity": self.viscosityEdit,
            "Surface tension": self.gammaEdit,
            "Volumic mass": self.rhoEdit,
            "Specific heat": self.cpEdit,
            "Emissivity": self.alphaEdit,
        }[prop]

    def material_exprs(self) -> tuple[str, str]:
        """``(gamma_expr, viscosity_expr)`` for the capillaries preview."""
        return (
            self.gammaEdit.text().strip(),
            self.viscosityEdit.text().strip(),
        )

    # -- live multi-curve preview (UIAxes_4) -----------------------------------
    def _auto_refresh(self, *_args) -> None:
        self.refresh_material_comparison()

    def refresh_material_comparison(
        self,
        cap_materials: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        """Plot the selected property over ``[250, 2500] K``.

        ``cap_materials`` is ``((gamma_i, visc_i), ...)`` for every
        enabled nested ring; ``Viscosity`` and ``Surface tension``
        overlay them on the Jacket curve.
        """
        if cap_materials is not None:
            self._cap_materials = tuple(cap_materials)
        prop = self.propertyCombo.currentText()
        edit = self._edit_for(prop)
        cap_exprs: tuple[str, ...] = ()
        if prop == "Viscosity":
            cap_exprs = tuple(m[1] for m in self._cap_materials)
        elif prop == "Surface tension":
            cap_exprs = tuple(m[0] for m in self._cap_materials)
        self.plotCanvas.plot_material_comparison(
            prop, edit.text(), cap_exprs, len(cap_exprs)
        )

    # -- data exchange ---------------------------------------------------------
    def geometry(self) -> tuple[float, float]:
        """``(r0, R0)`` jacket radii in metres."""
        return (
            mm_to_m(read_double(self.d0Edit, 2.4)) / 2.0,
            mm_to_m(read_double(self.D0Edit, 8.3556)) / 2.0,
        )

    def to_fields(self) -> dict:
        return {
            "r0": mm_to_m(read_double(self.d0Edit, 2.4)) / 2.0,
            "R0": mm_to_m(read_double(self.D0Edit, 8.3556)) / 2.0,
            "rho_expr": self.rhoEdit.text().strip(),
            "gamma_expr": self.gammaEdit.text().strip(),
            "cp_expr": self.cpEdit.text().strip(),
            "viscosity_expr": self.viscosityEdit.text().strip(),
            "alpha_expr": self.alphaEdit.text().strip(),
        }

    def apply_inputs(self, inputs: SimulationInputs) -> None:
        self.D0Edit.setText(f"{inputs.R0 * 2e3:.10g}")
        self.d0Edit.setText(f"{inputs.r0 * 2e3:.10g}")
        self.rhoEdit.setText(inputs.rho_expr)
        self.gammaEdit.setText(inputs.gamma_expr)
        self.cpEdit.setText(inputs.cp_expr)
        self.viscosityEdit.setText(inputs.viscosity_expr)
        self.alphaEdit.setText(inputs.alpha_expr)
