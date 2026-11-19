"""Interactive 2D fiber cross-section renderer (the "Schematic" view).

A Matplotlib axes (equal aspect, dark cyberpunk background) linked to a
``zmSlider`` scrubbing the axial position ``z`` over ``[0, z_end]``.  At
each index the renderer draws:

* the neon cyan jacket annulus (outer radius ``R(z)``, bore ``r(z)``),
* ``N_cap`` capillaries of ring 1 tangent inside the jacket bore wall,
* nested rings ``2..N_nested`` (exactly ONE tube per azimuth per ring,
  radially outer-tangent inside the parent bore, matching the MATLAB
  ``HARF_drawing.mlapp`` schematic), tinted pink (ring 1), green
  (ring 2), amber (ring 3).

Rendering is memoized on ``(result identity, slider index)`` and the
raster draw is deferred to ``showEvent`` while the widget is hidden, so
tab switches with unchanged state cost no Matplotlib rasterization.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

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

from app.theme import (
    BG_DARK,
    CYAN,
    CYAN_DARK,
    RING_FACE_COLORS,
    SURFACE_LOWEST,
    TEXT_MUTED,
    mono_font_family,
)
from src.models import SimulationResult

__all__ = ["CrossSectionCanvas", "compute_capillary_centers"]

_SLIDER_STEPS = 1000


def compute_capillary_centers(
    r: float,
    Rc: Sequence[float],
    rc: Sequence[float],
    n_cap: int,
) -> list[list[tuple[float, float]]]:
    """Center positions of the nested capillary rings (``.mlapp`` rule).

    Matches ``HARF_drawing.mlapp`` exactly: for each azimuth
    ``theta_k = 2*pi*k / n_cap`` (``k = 0..n_cap-1``) exactly ONE
    capillary per ring is drawn, radially outer-tangent:

    * ring 1 sits inside the jacket bore wall:
      ``rho_0 = r - Rc[0]``,
    * ring ``i`` sits inside ring ``i-1``'s bore, along the same
      azimuth: ``rho_i = rho_{i-1} + rc[i-1] - Rc[i]``.

    Returns one list of ``n_cap`` centers per ring (so a 3-ring
    6-capillary fiber yields ``[6, 6, 6]`` centers, never
    ``6 + 36 + 216``).  Rings stop at the first non-nested / invalid
    geometry (``Rc[i] > rc[i-1]``, non-positive or non-finite radii).
    """
    n_rings = min(len(Rc), len(rc))
    rings: list[list[tuple[float, float]]] = []
    rho = float(r)
    host_bore = float(r)
    for i in range(n_rings):
        rc_o = float(Rc[i])
        rc_i = float(rc[i])
        if not (math.isfinite(rc_o) and math.isfinite(rc_i)):
            break
        if rc_o <= 0.0 or rc_i <= 0.0 or rc_o > host_bore:
            break
        rho = (float(r) - rc_o) if i == 0 else (
            rho + float(rc[i - 1]) - rc_o
        )
        if not math.isfinite(rho) or rho < 0.0:
            break
        rings.append([
            (rho * math.cos(theta), rho * math.sin(theta))
            for theta in (
                2.0 * math.pi * k / n_cap for k in range(int(n_cap))
            )
        ])
        host_bore = rc_i
    return rings


class CrossSectionCanvas(QWidget):
    """Slider-scrubbed fiber cross-section view (neon schematic)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: SimulationResult | None = None
        self._result_ref: SimulationResult | None = None
        self._index = 0
        self._last_state_key: tuple | None = None
        self._dirty_draw = False

        self.figure = Figure(dpi=100)
        self.figure.patch.set_facecolor(BG_DARK)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal")
        # One-time static layout (no tight_layout on every refresh).
        self.figure.subplots_adjust(
            left=0.08, right=0.97, top=0.92, bottom=0.06
        )

        self.zmSlider = QSlider(Qt.Horizontal)
        self.zmSlider.setRange(0, _SLIDER_STEPS)
        self.zmSlider.setEnabled(False)
        self.zmSlider.valueChanged.connect(self._on_slider)
        self.z_label = QLabel("z = -- mm")
        self.z_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-family: '{mono_font_family()}';"
        )

        self.slider_row = QWidget()
        slider_row = QHBoxLayout(self.slider_row)
        slider_row.setContentsMargins(0, 0, 0, 0)
        slider_row.addWidget(QLabel("z position:"))
        slider_row.addWidget(self.zmSlider, stretch=1)
        slider_row.addWidget(self.z_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.slider_row)
        self._draw_empty()

    # -- visibility / deferred raster ------------------------------------
    def showEvent(self, event) -> None:  # (Qt override)
        """Rasterize pending content when the canvas becomes visible."""
        super().showEvent(event)
        if self._dirty_draw:
            self._dirty_draw = False
            self.canvas.draw_idle()

    def _finish_render(self) -> None:
        """Defer the raster draw while hidden (tab-switch lag guard)."""
        if self.isVisible():
            self.canvas.draw_idle()
        else:
            self._dirty_draw = True

    # -- public -----------------------------------------------------------
    def set_result(self, result: SimulationResult | None) -> None:
        """Install a new result and reset the slider to ``z = 0``."""
        self._result = result
        self._result_ref = result
        self._last_state_key = None
        if result is None or len(result.z) < 2 or result.z_end <= 0.0:
            self.zmSlider.setEnabled(False)
            self.zmSlider.setValue(0)
            self.z_label.setText("z = -- mm")
            self._draw_empty()
            return
        self.zmSlider.setEnabled(True)
        self.zmSlider.setValue(0)
        self._index = 0
        self.slider_row.show()
        self._refresh()

    @property
    def current_index(self) -> int:
        return self._index

    def set_static_geometry(
        self,
        title: str,
        r_outer: float,
        r_bore: float,
        r_env: float,
        r_core: float,
        ring_outer: Sequence[float],
        ring_bore: Sequence[float],
        n_cap: int,
        guide_radius: float | None = None,
    ) -> None:
        """Render a static (non-scrubbed) cross-section, radii in µm.

        Used by the Cane & Stack tab for the pre-simulation cane (stage
        E) and target fiber (stage I) schematics: the slider row is
        hidden, ``guide_radius`` (default: core + ring-1 outer radius,
        i.e. capillaries tangent to the core) is the ring-1 center
        circle radius, and the render is memoized on the rounded
        geometry tuple so unchanged re-renders cost nothing.
        """

        def rk(value: float) -> float | None:
            v = float(value)
            return round(v, 9) if math.isfinite(v) else None

        key = (
            "static", title, rk(r_outer), rk(r_bore), rk(r_env),
            rk(r_core), tuple(rk(v) for v in ring_outer),
            tuple(rk(v) for v in ring_bore), int(n_cap),
            rk(guide_radius) if guide_radius is not None else None,
        )
        if key == self._last_state_key:
            return
        self._last_state_key = key

        r_outer = float(r_outer)
        r_bore = float(r_bore)
        r_env = float(r_env)
        r_core = float(r_core)
        rings_o = [float(v) for v in ring_outer]
        rings_i = [float(v) for v in ring_bore]
        finite = (
            math.isfinite(r_outer) and math.isfinite(r_bore)
            and math.isfinite(r_env) and math.isfinite(r_core)
            and all(math.isfinite(v) for v in rings_o + rings_i)
        )
        self.slider_row.hide()
        self.zmSlider.setEnabled(False)
        self.z_label.setText("z = -- mm")
        if not finite or not r_outer > 0.0:
            self._draw_invalid(title)
            return

        self.ax.clear()
        self.ax.set_aspect("equal")
        # Jacket annulus (or bare outline when the bore is degenerate).
        if 0.0 < r_bore < r_outer:
            self.ax.add_patch(
                Wedge(
                    (0.0, 0.0), r_outer, 0.0, 360.0,
                    width=max(r_outer - r_bore, 1e-9),
                    facecolor=CYAN_DARK, edgecolor=CYAN, lw=1.8,
                )
            )
        else:
            self.ax.add_patch(
                Circle(
                    (0.0, 0.0), r_outer, facecolor="none",
                    edgecolor=CYAN, lw=1.8,
                )
            )
        # Dashed consolidation-front envelope inside the bore.
        if 0.0 < r_env < r_bore:
            self.ax.add_patch(
                Circle(
                    (0.0, 0.0), r_env, facecolor="none",
                    edgecolor=TEXT_MUTED, lw=1.0, linestyle="--",
                )
            )
        # Central core rod / hole.
        if r_core > 0.0:
            self.ax.add_patch(
                Circle(
                    (0.0, 0.0), r_core, facecolor=SURFACE_LOWEST,
                    edgecolor=CYAN_DARK, lw=0.8,
                )
            )

        # Ring 1 centers: tangent to the core unless overridden.
        guide = guide_radius
        if guide is None:
            guide = r_core + rings_o[0] if rings_o else r_core
        if not math.isfinite(guide):
            guide = r_core
        host = guide + (rings_o[0] if rings_o else 0.0)
        ring_centers = compute_capillary_centers(
            host, rings_o, rings_i, int(n_cap),
        )
        self._draw_capillary_rings(rings_o, rings_i, ring_centers)

        limit = max(r_outer, r_core) * 1.15
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(title)
        self._style_axes()
        self._finish_render()

    def _draw_invalid(self, title: str) -> None:
        """Non-finite / degenerate geometry fallback view."""
        self.ax.clear()
        self.ax.text(
            0.5, 0.5, "invalid geometry", ha="center", va="center",
            transform=self.ax.transAxes, fontsize=10, color=TEXT_MUTED,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(title)
        self._style_axes()
        self._finish_render()

    # -- slots --------------------------------------------------------------
    def _on_slider(self, value: int) -> None:
        if self._result is None or len(self._result.z) < 2:
            return
        n = len(self._result.z)
        self._index = min(int(value * (n - 1) / _SLIDER_STEPS), n - 1)
        self._refresh()

    # -- drawing ------------------------------------------------------------
    def _draw_empty(self) -> None:
        if self._last_state_key == ("empty",):
            self._finish_render()
            return
        self._last_state_key = ("empty",)
        self.ax.clear()
        self.ax.text(
            0.5, 0.5, "Run a computation to scrub the cross-section",
            ha="center", va="center", transform=self.ax.transAxes,
            fontsize=10, color=TEXT_MUTED,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._style_axes()
        self._finish_render()

    def _style_axes(self) -> None:
        self.figure.patch.set_facecolor(BG_DARK)
        self.ax.set_facecolor(SURFACE_LOWEST)
        self.ax.title.set_color(TEXT_MUTED)
        for spine in self.ax.spines.values():
            spine.set_color("#3a494a")

    def _draw_capillary_rings(
        self,
        rings_o: list[float],
        rings_i: list[float],
        ring_centers: list[list[tuple[float, float]]],
    ) -> None:
        """Neon capillary rings (one tube per azimuth per ring)."""
        for i, centers_i in enumerate(ring_centers):
            neon = RING_FACE_COLORS[i % len(RING_FACE_COLORS)]
            rc_o = rings_o[i]
            rc_i = rings_i[i]
            for cx, cy in centers_i:
                self.ax.add_patch(
                    Circle(
                        (cx, cy), rc_o, facecolor=neon,
                        edgecolor=neon, lw=1.2, alpha=0.85,
                    )
                )
                self.ax.add_patch(
                    Circle(
                        (cx, cy), rc_i, facecolor=SURFACE_LOWEST,
                        edgecolor=neon, lw=0.9,
                    )
                )

    def _refresh(self) -> None:
        result = self._result
        if result is None:
            self._draw_empty()
            return
        # Memoized render signature: unchanged (result, index) -> no-op.
        key = (id(self._result_ref), self._index)
        if key == self._last_state_key:
            self._finish_render()
            return
        self._last_state_key = key
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

        # Jacket annulus: neon cyan ring between r and R.
        self.ax.add_patch(
            Wedge(
                (0.0, 0.0), R, 0.0, 360.0, width=max(R - r, 1e-9),
                facecolor=CYAN_DARK, edgecolor=CYAN, lw=1.8,
            )
        )
        # Core hole (dark).
        self.ax.add_patch(
            Circle(
                (0.0, 0.0), r, facecolor=SURFACE_LOWEST,
                edgecolor=CYAN_DARK, lw=0.8,
            )
        )

        # Nested rings: exactly one capillary per azimuth per ring
        # (MATLAB HARF_drawing.mlapp schematic geometry).
        ring_centers = compute_capillary_centers(r, rings_o, rings_i, n_cap)
        self._draw_capillary_rings(rings_o, rings_i, ring_centers)

        limit = max(R, r) * 1.15
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(
            f"Cross-section at z = {z_mm:.2f} mm "
            f"({n_cap} capillaries, {len(ring_centers)} nested ring(s))"
        )
        self._style_axes()
        self._finish_render()
