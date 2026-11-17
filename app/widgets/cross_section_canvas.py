"""Interactive 2D fiber cross-section renderer (the "Schematic" view).

A Matplotlib axes (equal aspect) linked to a ``zmSlider`` scrubbing the
axial position ``z`` over ``[0, z_end]``.  At each index the renderer
draws:

* the jacket annulus (outer radius ``R(z)``, bore ``r(z)``),
* ``N_cap`` capillaries of ring 1 tangent inside the jacket bore wall,
  centred at radius ``r(z) - Rc1(z)``,
* nested rings ``2..N_nested`` tangent inside each host capillary bore,
  centred at radius ``rc_{i-1}(z) - Rc_i(z)``.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Wedge
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.models import SimulationResult

__all__ = ["CrossSectionCanvas"]

_SLIDER_STEPS = 1000


class CrossSectionCanvas(QWidget):
    """Slider-scrubbed fiber cross-section view."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: SimulationResult | None = None
        self._index = 0

        self.figure = Figure(dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal")

        self.zmSlider = QSlider(Qt.Horizontal)
        self.zmSlider.setRange(0, _SLIDER_STEPS)
        self.zmSlider.setEnabled(False)
        self.zmSlider.valueChanged.connect(self._on_slider)
        self.z_label = QLabel("z = -- mm")
        self.z_label.setStyleSheet("color: #546e7a; font-family: monospace;")

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("z position:"))
        slider_row.addWidget(self.zmSlider, stretch=1)
        slider_row.addWidget(self.z_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, stretch=1)
        layout.addLayout(slider_row)
        self._draw_empty()

    # -- public -----------------------------------------------------------
    def set_result(self, result: SimulationResult | None) -> None:
        """Install a new result and reset the slider to ``z = 0``."""
        self._result = result
        if result is None or len(result.z) < 2 or result.z_end <= 0.0:
            self.zmSlider.setEnabled(False)
            self.zmSlider.setValue(0)
            self.z_label.setText("z = -- mm")
            self._draw_empty()
            return
        self.zmSlider.setEnabled(True)
        self.zmSlider.setValue(0)
        self._index = 0
        self._refresh()

    @property
    def current_index(self) -> int:
        return self._index

    # -- slots --------------------------------------------------------------
    def _on_slider(self, value: int) -> None:
        if self._result is None or len(self._result.z) < 2:
            return
        n = len(self._result.z)
        self._index = min(int(value * (n - 1) / _SLIDER_STEPS), n - 1)
        self._refresh()

    # -- drawing ------------------------------------------------------------
    def _draw_empty(self) -> None:
        self.ax.clear()
        self.ax.text(
            0.5, 0.5, "Run a computation to scrub the cross-section",
            ha="center", va="center", transform=self.ax.transAxes,
            fontsize=10, color="#90a4ae",
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _refresh(self) -> None:
        result = self._result
        if result is None:
            self._draw_empty()
            return
        idx = self._index
        z_mm = float(result.z[idx]) * 1e3
        self.z_label.setText(f"z = {z_mm:.2f} mm")
        self.ax.clear()
        self.ax.set_aspect("equal")

        R = float(result.R[idx]) * 1e6
        r = float(result.r[idx]) * 1e6
        rings_o = [float(a[idx]) * 1e6 for a in result.Rc]
        rings_i = [float(a[idx]) * 1e6 for a in result.rc]
        n_cap = int(result.inputs.n_capillaries)

        # Jacket annulus (filled ring between r and R).
        self.ax.add_patch(
            Wedge((0.0, 0.0), R, 0.0, 360.0, width=max(R - r, 1e-9),
                  facecolor="#b0bec5", edgecolor="#455a64", lw=1.2)
        )
        # Core hole.
        self.ax.add_patch(
            Circle((0.0, 0.0), r, facecolor="#ffffff", edgecolor="#90a4ae",
                   lw=0.8)
        )

        # Ring 1: N_cap capillaries tangent inside the jacket bore wall.
        centers = [(0.0, 0.0)]
        host_radius = r
        for i, (rc_o, rc_i) in enumerate(zip(rings_o, rings_i, strict=True)):
            if rc_o <= 0.0 or rc_i <= 0.0 or rc_o > host_radius:
                break
            new_centers = []
            for cx, cy in centers:
                offset = host_radius - rc_o
                for k in range(n_cap):
                    theta = 2.0 * np.pi * (k + 0.5 * (i > 0)) / n_cap
                    ccx = cx + offset * np.cos(theta)
                    ccy = cy + offset * np.sin(theta)
                    self.ax.add_patch(
                        Circle((ccx, ccy), rc_o, facecolor="#eceff1",
                               edgecolor="#546e7a", lw=1.0)
                    )
                    self.ax.add_patch(
                        Circle((ccx, ccy), rc_i, facecolor="#ffffff",
                               edgecolor="#78909c", lw=0.8)
                    )
                    new_centers.append((ccx, ccy))
            centers = new_centers
            host_radius = rc_i

        limit = max(R, r) * 1.15
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(
            f"Cross-section at z = {z_mm:.2f} mm "
            f"({n_cap} capillaries, {len(rings_o)} nested ring(s))"
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()
