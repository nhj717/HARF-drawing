"""Tab 2: jacket geometry and user-editable material laws.

Includes a live material-property preview (viscosity on a log axis,
surface tension, density, heat capacity) and the furnace profile
``T/Tmax`` vs ``z``, both rendered with :class:`PlotCanvas`.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
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
from src.expressions import CompiledExpression, ExpressionError
from src.models import (
    DEFAULT_ALPHA,
    DEFAULT_CP,
    DEFAULT_FURNACE_PROFILE,
    DEFAULT_GAMMA,
    DEFAULT_RHO,
    DEFAULT_VISCOSITY,
    SimulationInputs,
    mm_to_m,
)

__all__ = ["JacketTab"]

_PROPERTY_ITEMS = (
    "Viscosity (log)",
    "Surface tension",
    "Density",
    "Heat capacity",
    "Absorptivity",
)


class JacketTab(QWidget):
    """Jacket (outer preform) geometry and material expressions."""

    geometry_changed = Signal()
    expressions_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
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

        self.furnaceProfileEdit = make_double_edit(
            1.0, tooltip="Furnace profile(z) in [0, 1], expression of z [m]"
        )
        self.furnaceProfileEdit.setText(DEFAULT_FURNACE_PROFILE)
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
        form.addRow("Furnace profile(z) [0,1]", self.furnaceProfileEdit)

        preview_box = QGroupBox("Live previews")
        self.propertyCombo = QComboBox()
        self.propertyCombo.addItems(_PROPERTY_ITEMS)
        self.refreshButton = QPushButton("Refresh preview")
        self.refreshButton.clicked.connect(self.refresh_preview)
        self.plotCanvas = PlotCanvas()

        row = QHBoxLayout()
        row.addWidget(self.propertyCombo)
        row.addWidget(self.refreshButton)
        inner = QVBoxLayout(preview_box)
        inner.addLayout(row)
        inner.addWidget(self.plotCanvas)

        layout = QVBoxLayout(self)
        layout.addWidget(geom_box)
        layout.addWidget(material_box)
        layout.addWidget(preview_box, stretch=1)

        for edit in (self.D0Edit, self.d0Edit):
            edit.textChanged.connect(self._update_geometry_badge)
            edit.textChanged.connect(self.geometry_changed)
        for edit in (
            self.furnaceProfileEdit, self.rhoEdit, self.gammaEdit,
            self.cpEdit, self.viscosityEdit, self.alphaEdit,
        ):
            variables = ("z",) if edit is self.furnaceProfileEdit else ("T",)
            attach_formula_validation(edit, variables)
            edit.textChanged.connect(self.expressions_changed)

        self._update_geometry_badge()
        self.refresh_preview()

    # -- helpers -------------------------------------------------------------
    def _update_geometry_badge(self) -> None:
        d0 = read_double(self.d0Edit, 2.4)
        D0 = read_double(self.D0Edit, 8.3556)
        self.dRatioBadge.set_ratio(d0, D0)

    # -- previews ------------------------------------------------------------
    def refresh_preview(self) -> None:
        """Plot the selected material property vs T (or furnace profile)."""
        name = self.propertyCombo.currentText()
        if name == "Viscosity (log)":
            self._plot_single(self.viscosityEdit, "mu (Pa s)", log_y=True)
        elif name == "Surface tension":
            self._plot_single(self.gammaEdit, "gamma (N/m)")
        elif name == "Density":
            self._plot_single(self.rhoEdit, "rho (kg/m^3)")
        elif name == "Heat capacity":
            self._plot_single(self.cpEdit, "Cp (J/(kg K))")
        elif name == "Absorptivity":
            self._plot_single(self.alphaEdit, "alpha")
        else:  # pragma: no cover - combo is exhaustive
            pass

    def _plot_single(self, edit, ylabel: str, log_y: bool = False) -> None:
        t_grid = np.linspace(800.0, 2500.0, 400)
        try:
            expr = CompiledExpression(edit.text(), ("T",))
            values = expr.vector(t_grid)
        except ExpressionError:
            self.plotCanvas.clear("Invalid formula — cannot preview.")
            return
        good = np.isfinite(values)
        if not good.any():
            self.plotCanvas.clear(
                "Formula yields no finite values on T in [800, 2500] K."
            )
            return
        self.plotCanvas.plot_property(
            t_grid[good], [(values[good], "value")], ylabel,
            log_y=log_y, title=self.propertyCombo.currentText(),
        )

    def plot_furnace_profile(self, inputs: SimulationInputs) -> None:
        """Render the current furnace profile T/Tmax vs z."""
        try:
            expr = CompiledExpression(self.furnaceProfileEdit.text(), ("z",))
        except ExpressionError:
            self.plotCanvas.clear("Invalid furnace profile formula.")
            return
        z_grid = np.linspace(0.0, inputs.L, 400)
        values = np.clip(expr.vector(z_grid), 0.0, 1.0)
        self.plotCanvas.plot_furnace_profile(
            z_grid, values, inputs.Tmax_C
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
            "furnace_profile_expr": self.furnaceProfileEdit.text().strip(),
            "rho_expr": self.rhoEdit.text().strip(),
            "gamma_expr": self.gammaEdit.text().strip(),
            "cp_expr": self.cpEdit.text().strip(),
            "viscosity_expr": self.viscosityEdit.text().strip(),
            "alpha_expr": self.alphaEdit.text().strip(),
        }

    def apply_inputs(self, inputs: SimulationInputs) -> None:
        self.D0Edit.setText(f"{inputs.R0 * 2e3:.10g}")
        self.d0Edit.setText(f"{inputs.r0 * 2e3:.10g}")
        self.furnaceProfileEdit.setText(inputs.furnace_profile_expr)
        self.rhoEdit.setText(inputs.rho_expr)
        self.gammaEdit.setText(inputs.gamma_expr)
        self.cpEdit.setText(inputs.cp_expr)
        self.viscosityEdit.setText(inputs.viscosity_expr)
        self.alphaEdit.setText(inputs.alpha_expr)
