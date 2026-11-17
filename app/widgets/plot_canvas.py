"""Embedded Matplotlib canvas with toolbar and live cursor readout.

Renders every plot of the workstation:

* axial draw views (velocity, diameters, wall thickness, core
  diameter, ratios, temperature, draw tension, contact ratio),
* material property vs ``T`` curves (log-y for viscosity, multi-ring
  overlays),
* the furnace profile ``T/Tmax`` vs ``z``.

Both full ``[0, L]`` and partial ``[0, z_stop]`` trajectories are
handled cleanly (including empty INPUT_ERROR results).
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.models import SimulationResult

__all__ = ["AXIAL_VIEWS", "PlotCanvas", "SCHEMATIC_VIEW"]

SCHEMATIC_VIEW = "Schematic (cross-section)"

AXIAL_VIEWS: tuple[str, ...] = (
    "Velocity V (mm/min)",
    "Jacket diameter D (mm)",
    "Jacket bore diameter d (mm)",
    "Capillary 1 outer Dc (um)",
    "Capillary 1 bore dc (um)",
    "Capillary wall t (um)",
    "Core diameter Dcore (um)",
    "dc / Dc",
    "Temperature (C)",
    "Draw tension (N)",
    "Rc1 / Rcontact",
    SCHEMATIC_VIEW,
)


class PlotCanvas(QWidget):
    """Matplotlib figure + navigation toolbar + ``(x, y)`` cursor label."""

    def __init__(self, parent: QWidget | None = None, dpi: int = 100) -> None:
        super().__init__(parent)
        self.figure = Figure(dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.cursor_label = QLabel("x = --, y = --")
        self.cursor_label.setAlignment(Qt.AlignRight)
        self.cursor_label.setStyleSheet("color: #546e7a; font-family: monospace;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.cursor_label)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.clear("No data yet — run a computation.")

    # -- cursor ---------------------------------------------------------
    def _on_motion(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None or not np.isfinite([x, y]).all():
            return
        self.cursor_label.setText(f"x = {x:.6g}, y = {y:.6g}")

    # -- generic drawing --------------------------------------------------
    def clear(self, message: str = "") -> None:
        self.ax.clear()
        if message:
            self.ax.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=10,
                color="#90a4ae",
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        self.canvas.draw_idle()

    def plot_curves(
        self,
        x: np.ndarray,
        curves: list[tuple[np.ndarray, str]],
        xlabel: str,
        ylabel: str,
        log_y: bool = False,
        title: str = "",
    ) -> None:
        """Plot ``curves = [(y, label), ...]`` against ``x``."""
        self.ax.clear()
        if x is None or len(x) == 0:
            self.clear("No data available for this view.")
            return
        for y, label in curves:
            if y is None or len(y) != len(x):
                continue
            self.ax.plot(x, y, label=label)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        if log_y:
            self.ax.set_yscale("log")
        if len(curves) > 1:
            self.ax.legend(fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def plot_property(
        self,
        t_grid: np.ndarray,
        curves: list[tuple[np.ndarray, str]],
        ylabel: str,
        log_y: bool = False,
        title: str = "",
    ) -> None:
        """Material property vs temperature curves (log-y for viscosity)."""
        self.plot_curves(
            t_grid, curves, "T (K)", ylabel, log_y=log_y, title=title
        )

    def plot_furnace_profile(
        self, z_m: np.ndarray, profile: np.ndarray, t_max_c: float
    ) -> None:
        """Furnace profile ``T/Tmax`` (and the hot/cold zone split)."""
        self.ax.clear()
        self.ax.plot(z_m * 1e3, profile, label="T/Tmax")
        self.ax.set_xlabel("z (mm)")
        self.ax.set_ylabel("T / Tmax (normalized)")
        self.ax.set_title(f"Furnace profile (Tmax = {t_max_c:.1f} C)")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    # -- axial draw views -------------------------------------------------
    def draw_axial_view(self, result: SimulationResult | None, view: str) -> None:
        """Render one of the 11 axial views of a simulation result."""
        if result is None:
            self.clear("No data yet — run a computation.")
            return
        z_mm = np.asarray(result.z, dtype=float) * 1e3
        if len(z_mm) < 2 or result.z_end <= 0.0:
            self.clear(
                "No trajectory available (the run failed before integration)."
            )
            return
        partial = result.z_end < result.inputs.L - 1e-12

        def _has(name: str, index: int = 0) -> np.ndarray | None:
            arrays = getattr(result, name, ())
            if index >= len(arrays):
                return None
            arr = np.asarray(arrays[index], dtype=float)
            return arr if len(arr) == len(z_mm) else None

        if view.startswith("Velocity"):
            self.plot_curves(
                z_mm, [(result.w * 60.0 * 1e3, "w(z)")],
                "z (mm)", "w (mm/min)", title=view,
            )
        elif view.startswith("Jacket diameter"):
            self.plot_curves(
                z_mm, [(2.0 * result.R * 1e3, "D(z)")], "z (mm)", "D (mm)",
                title=view,
            )
        elif view.startswith("Jacket bore"):
            self.plot_curves(
                z_mm, [(2.0 * result.r * 1e3, "d(z)")], "z (mm)", "d (mm)",
                title=view,
            )
        elif view.startswith("Capillary 1 outer"):
            rc = _has("Rc")
            self.plot_curves(
                z_mm,
                [(2.0 * rc * 1e6, "Dc1(z)")] if rc is not None else [],
                "z (mm)", "Dc1 (um)", title=view,
            )
        elif view.startswith("Capillary 1 bore"):
            rc = _has("rc")
            self.plot_curves(
                z_mm,
                [(2.0 * rc * 1e6, "dc1(z)")] if rc is not None else [],
                "z (mm)", "dc1 (um)", title=view,
            )
        elif view.startswith("Capillary wall"):
            rc_o, rc_i = _has("Rc"), _has("rc")
            curves = []
            if rc_o is not None and rc_i is not None:
                curves.append(((rc_o - rc_i) * 1e6, "t(z)"))
            for i in range(1, len(result.Rc)):
                ro, ri = _has("Rc", i), _has("rc", i)
                if ro is not None and ri is not None:
                    curves.append(((ro - ri) * 1e6, f"t{i + 1}(z)"))
            self.plot_curves(
                z_mm, curves, "z (mm)", "wall t (um)", title=view
            )
        elif view.startswith("Core diameter"):
            self.plot_curves(
                z_mm, [(result.D_core * 1e6, "Dcore(z)")],
                "z (mm)", "Dcore (um)", title=view,
            )
        elif view.startswith("dc / Dc"):
            rc_o, rc_i = _has("Rc"), _has("rc")
            curves = []
            if rc_o is not None and rc_i is not None:
                curves.append((rc_i / rc_o, "dc1/Dc1"))
            for i in range(1, len(result.Rc)):
                ro, ri = _has("Rc", i), _has("rc", i)
                if ro is not None and ri is not None:
                    curves.append((ri / ro, f"dc{i + 1}/Dc{i + 1}"))
            self.plot_curves(
                z_mm, curves, "z (mm)", "dc / Dc (dimensionless)",
                title=view,
            )
        elif view.startswith("Temperature"):
            self.plot_curves(
                z_mm,
                [
                    (result.T - 273.15, "Preform T"),
                    (result.Ta - 273.15, "Furnace Ta"),
                ],
                "z (mm)", "T (C)", title=view,
            )
        elif view.startswith("Draw tension"):
            self.plot_curves(
                z_mm, [(result.tau, "tau(z)")], "z (mm)", "tau (N)",
                title=view,
            )
        elif view.startswith("Rc1 / Rcontact"):
            rc_o = _has("Rc")
            contact = result.R_contact
            curves = []
            if rc_o is not None:
                curves.append((rc_o / contact, "Rc1/Rcontact"))
            self.plot_curves(
                z_mm, curves, "z (mm)", "Rc1 / Rcontact", title=view
            )
        else:  # pragma: no cover - unknown view
            self.clear(f"Unknown view: {view}")
            return
        if partial:
            self.ax.axvline(
                result.z_end * 1e3, color="#c62828", linestyle="--",
                alpha=0.7, label=f"stop z = {result.z_end * 1e3:.1f} mm",
            )
            self.ax.legend(fontsize=8)
            self.canvas.draw_idle()

    def has_finite_data(self) -> bool:
        """True when the axes hold at least one line with finite points."""
        for line in self.ax.get_lines():
            xy = line.get_xydata()
            if len(xy) and np.isfinite(xy).all():
                return True
        return False
