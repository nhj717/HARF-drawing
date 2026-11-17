"""Tab 1: process, thermal and furnace parameters (MATLAB v1.5 defaults).

Split layout: the left column hosts the Drawing parameters, Constants
and Furnace (``Lfour``, ``profil(z)``) group boxes plus the GitHub
button; the right column hosts the ``PlotCanvas`` (MATLAB ``UIAxes_3``)
rendering the normalized furnace temperature profile ``T/Tmax`` vs
``z (m)`` on ``[0, Lfour]`` immediately at startup and live on every
edit of ``profil(z)`` or ``Lfour``.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
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
    attach_formula_validation,
    make_double_edit,
    read_double,
)
from src.expressions import CompiledExpression, ExpressionError
from src.models import (
    DEFAULT_FURNACE_PROFILE,
    DEFAULT_TEXT_C,
    SimulationInputs,
    grams_to_newtons,
    mm_per_min_to_m_per_s,
)

__all__ = ["ParameterTab"]

_GITHUB_URL = "https://github.com/nhj717/HARF-drawing"


class ParameterTab(QWidget):
    """Draw tension, feed speed, geometry lengths and thermal inputs."""

    inputs_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.drawTensionEdit = make_double_edit(
            84.0, 0.0, 1e9, tooltip="Target draw tension at z = L [g]"
        )
        self.feedSpeedEdit = make_double_edit(
            4.0, 0.0, 1e9, tooltip="Preform feed speed w0 [mm/min]"
        )
        self.lengthEdit = make_double_edit(
            0.1, 0.0, 10.0, tooltip="Total drawing-zone length L [m]"
        )
        self.textEdit = make_double_edit(
            20.0, -273.15, 1e4, tooltip="External temperature Text [C]"
        )
        self.dT0Edit = make_double_edit(
            500.0, 0.0, 1e4,
            tooltip="Initial preform under-cooling dT0 = T(0) - T0 [K]",
        )
        self.LfourEdit = make_double_edit(
            0.065, 0.0, 10.0, tooltip="Furnace (hot-zone) length [m]"
        )
        self.sigmaEdit = make_double_edit(
            5.67e-8, 0.0, 1.0,
            tooltip="Stefan-Boltzmann constant [W m-2 K-4]",
        )
        self.nHeatEdit = make_double_edit(
            100.0, 0.0, 1e6, tooltip="Convective heat-transfer coefficient N"
        )
        self.gEdit = make_double_edit(
            9.81, 0.0, 100.0, tooltip="Gravitational acceleration [m s-2]"
        )

        process_box = QGroupBox("Process parameters")
        form = QFormLayout(process_box)
        form.addRow("Draw tension [g]", self.drawTensionEdit)
        form.addRow("Feed speed [mm/min]", self.feedSpeedEdit)
        form.addRow("Drawing-zone length L [m]", self.lengthEdit)

        furnace_box = QGroupBox("Furnace")
        self.furnaceProfileEdit = make_double_edit(
            1.0, tooltip="Furnace profile(z) in [0, 1], expression of z [m]"
        )
        self.furnaceProfileEdit.setText(DEFAULT_FURNACE_PROFILE)
        fform = QFormLayout(furnace_box)
        fform.addRow("Furnace length Lfour [m]", self.LfourEdit)
        fform.addRow("Furnace profile(z) [0,1]", self.furnaceProfileEdit)

        thermal_box = QGroupBox("Thermal parameters")
        form = QFormLayout(thermal_box)
        form.addRow("External temperature Text [C]", self.textEdit)
        form.addRow("Initial under-cooling dT0 [K]", self.dT0Edit)

        advanced_box = QGroupBox("Advanced constants")
        form = QFormLayout(advanced_box)
        form.addRow("Stefan-Boltzmann sigma", self.sigmaEdit)
        form.addRow("Heat-transfer coefficient N", self.nHeatEdit)
        form.addRow("Gravity g [m/s^2]", self.gEdit)

        self.githubButton = QPushButton("GitHub")
        self.githubButton.setObjectName("btn_accent")
        self.githubButton.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(_GITHUB_URL))
        )
        link_row = QHBoxLayout()
        link_row.addWidget(self.githubButton)
        link_row.addStretch(1)

        # Left column (controls) + right column (UIAxes_3 live plot).
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(process_box)
        left_layout.addWidget(furnace_box)
        left_layout.addWidget(thermal_box)
        left_layout.addWidget(advanced_box)
        left_layout.addLayout(link_row)
        left_layout.addStretch(1)

        preview_box = QGroupBox("Furnace temperature profile (UIAxes_3)")
        pv = QVBoxLayout(preview_box)
        self.plotCanvas = PlotCanvas()
        pv.addWidget(self.plotCanvas)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(preview_box)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([420, 640])

        layout = QHBoxLayout(self)
        layout.addWidget(self.splitter)

        for edit in (
            self.drawTensionEdit, self.feedSpeedEdit, self.lengthEdit,
            self.textEdit, self.dT0Edit, self.LfourEdit, self.sigmaEdit,
            self.nHeatEdit, self.gEdit,
        ):
            edit.textChanged.connect(self.inputs_changed)
        attach_formula_validation(self.furnaceProfileEdit, ("z",))
        self.furnaceProfileEdit.textChanged.connect(self.inputs_changed)

        self.refresh_furnace_profile()

    # -- live furnace profile (UIAxes_3) ----------------------------------
    def refresh_furnace_profile(
        self, inputs: SimulationInputs | None = None
    ) -> None:
        """Render the normalized furnace profile ``T/Tmax`` on ``[0, Lfour]``."""
        if inputs is not None:
            lf = inputs.Lfour
        else:
            lf = read_double(self.LfourEdit, 0.065)
        lf = max(float(lf), 1e-6)
        try:
            expr = CompiledExpression(self.furnaceProfileEdit.text(), ("z",))
        except ExpressionError:
            self.plotCanvas.clear("Invalid furnace profile formula.")
            return
        z_grid = np.linspace(0.0, lf, 400)
        values = np.clip(expr.vector(z_grid), 0.0, 1.0)
        if not np.isfinite(values).any():
            self.plotCanvas.clear(
                "Furnace profile yields no finite values on [0, Lfour]."
            )
            return
        self.plotCanvas.plot_furnace_profile(z_grid, values, 1806.9)
        self.plotCanvas.ax.set_xlabel("z (m)")
        self.plotCanvas.ax.set_xlim(0.0, lf)
        self.plotCanvas.figure.tight_layout()
        self.plotCanvas.canvas.draw_idle()

    # -- data exchange -----------------------------------------------------
    def to_fields(self) -> dict:
        """SI-valued SimulationInputs fields owned by this tab."""
        return {
            "tension": grams_to_newtons(
                read_double(self.drawTensionEdit, 84.0)
            ),
            "w0": mm_per_min_to_m_per_s(
                read_double(self.feedSpeedEdit, 4.0)
            ),
            "L": read_double(self.lengthEdit, 0.1),
            "Lfour": read_double(self.LfourEdit, 0.065),
            "Text_C": read_double(self.textEdit, DEFAULT_TEXT_C),
            "dT0": read_double(self.dT0Edit, 500.0),
            "sigma": read_double(self.sigmaEdit, 5.67e-8),
            "N_heat": read_double(self.nHeatEdit, 100.0),
            "g": read_double(self.gEdit, 9.81),
            "furnace_profile_expr": self.furnaceProfileEdit.text().strip(),
        }

    def apply_inputs(self, inputs: SimulationInputs) -> None:
        g_per_n = 1000.0 / 9.81
        self.drawTensionEdit.setText(f"{inputs.tension * g_per_n:.10g}")
        self.feedSpeedEdit.setText(f"{inputs.w0 * 60.0 * 1e3:.10g}")
        self.lengthEdit.setText(f"{inputs.L:.10g}")
        self.LfourEdit.setText(f"{inputs.Lfour:.10g}")
        self.furnaceProfileEdit.setText(inputs.furnace_profile_expr)
        self.textEdit.setText(f"{inputs.Text_C:.10g}")
        self.dT0Edit.setText(f"{inputs.dT0:.10g}")
        self.sigmaEdit.setText(f"{inputs.sigma:.10g}")
        self.nHeatEdit.setText(f"{inputs.N_heat:.10g}")
        self.gEdit.setText(f"{inputs.g:.10g}")
