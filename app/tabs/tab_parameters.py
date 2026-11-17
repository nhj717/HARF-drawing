"""Tab 1: process and thermal parameters (MATLAB v1.5 defaults)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from app.widgets.validators import make_double_edit, read_double
from src.models import (
    DEFAULT_TEXT_C,
    SimulationInputs,
    grams_to_newtons,
    mm_per_min_to_m_per_s,
)

__all__ = ["ParameterTab"]


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
        form.addRow("Furnace length Lfour [m]", self.LfourEdit)

        thermal_box = QGroupBox("Thermal parameters")
        form = QFormLayout(thermal_box)
        form.addRow("External temperature Text [C]", self.textEdit)
        form.addRow("Initial under-cooling dT0 [K]", self.dT0Edit)

        advanced_box = QGroupBox("Advanced constants")
        form = QFormLayout(advanced_box)
        form.addRow("Stefan-Boltzmann sigma", self.sigmaEdit)
        form.addRow("Heat-transfer coefficient N", self.nHeatEdit)
        form.addRow("Gravity g [m/s^2]", self.gEdit)

        layout = QVBoxLayout(self)
        layout.addWidget(process_box)
        layout.addWidget(thermal_box)
        layout.addWidget(advanced_box)
        layout.addStretch(1)

        for edit in (
            self.drawTensionEdit, self.feedSpeedEdit, self.lengthEdit,
            self.textEdit, self.dT0Edit, self.LfourEdit, self.sigmaEdit,
            self.nHeatEdit, self.gEdit,
        ):
            edit.textChanged.connect(self.inputs_changed)

    # -- data exchange -----------------------------------------------------
    def to_fields(self) -> dict:
        """SI-valued SimulationInputs fields owned by this tab."""
        return {
            "tension": grams_to_newtons(read_double(self.drawTensionEdit, 84.0)),
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
        }

    def apply_inputs(self, inputs: SimulationInputs) -> None:
        g_per_n = 1000.0 / 9.81
        self.drawTensionEdit.setText(f"{inputs.tension * g_per_n:.10g}")
        self.feedSpeedEdit.setText(f"{inputs.w0 * 60.0 * 1e3:.10g}")
        self.lengthEdit.setText(f"{inputs.L:.10g}")
        self.LfourEdit.setText(f"{inputs.Lfour:.10g}")
        self.textEdit.setText(f"{inputs.Text_C:.10g}")
        self.dT0Edit.setText(f"{inputs.dT0:.10g}")
        self.sigmaEdit.setText(f"{inputs.sigma:.10g}")
        self.nHeatEdit.setText(f"{inputs.N_heat:.10g}")
        self.gEdit.setText(f"{inputs.g:.10g}")
