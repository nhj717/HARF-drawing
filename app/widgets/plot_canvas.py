"""Embedded Matplotlib canvas with toolbar and live cursor readout.

Renders every plot of the workstation in the Cyberpunk Neon theme:

* axial draw views (velocity, diameters, wall thickness, core
  diameter, ratios, temperature, draw tension in g, contact ratio),
* material property vs ``T`` curves over ``[250, 2500] K`` with the
  neon curve cycle (cyan jacket solid, pink/green/amber dashed caps),
* the normalized furnace profile ``T/Tmax`` and the pre-run axial
  furnace temperature profile over ``[0, L]``.

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
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from app.theme import (
    CURVE_CAPS,
    CURVE_JACKET,
    TEXT_MUTED,
    TOOLBAR_QSS,
    mono_font_family,
    style_dark_figure,
)
from src.expressions import CompiledExpression, ExpressionError
from src.models import SimulationResult

__all__ = [
    "AXIAL_VIEWS",
    "SCHEMATIC_VIEW",
    "TEMPERATURE_VIEW",
    "TENSION_VIEW",
    "PlotCanvas",
]

SCHEMATIC_VIEW = "Schematic (cross-section)"
TEMPERATURE_VIEW = "Temperature (°C)"
TENSION_VIEW = "Draw tension (g)"

AXIAL_VIEWS: tuple[str, ...] = (
    "Velocity V (m/min)",
    "Jacket diameter D (mm)",
    "Jacket bore diameter d (mm)",
    "Capillary 1 outer Dc (um)",
    "Capillary 1 bore dc (um)",
    "Capillary wall t (um)",
    "Core diameter Dcore (um)",
    "dc / Dc",
    TEMPERATURE_VIEW,
    TENSION_VIEW,
    "Rc1 / Rcontact",
    SCHEMATIC_VIEW,
)

_PROPERTY_YLABELS = {
    "Viscosity": ("mu (Pa s)", True),
    "Surface tension": ("gamma (N/m)", False),
    "Volumic mass": ("rho (kg/m^3)", False),
    "Specific heat": ("Cp (J/(kg K))", False),
    "Emissivity": ("alpha", False),
}
_MULTI_CURVE_PROPS = ("Viscosity", "Surface tension")
_T_MIN_K, _T_MAX_K = 250.0, 2500.0


class PlotCanvas(QWidget):
    """Matplotlib figure + navigation toolbar + ``(x, y)`` cursor label."""

    def __init__(self, parent: QWidget | None = None, dpi: int = 100) -> None:
        super().__init__(parent)
        self.figure = Figure(dpi=dpi)
        self.figure.patch.set_facecolor("#0f131d")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet(TOOLBAR_QSS)
        self.cursor_label = QLabel("x = --, y = --")
        self.cursor_label.setAlignment(Qt.AlignRight)
        self.cursor_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-family: '{mono_font_family()}';"
        )

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cursor_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar, 0)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.cursor_label, 0)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        # One-time static layout (no tight_layout on every refresh) plus
        # render memoization / deferred rasterization for fast tab
        # switches with unchanged state.
        self.figure.subplots_adjust(
            left=0.11, right=0.97, top=0.91, bottom=0.10
        )
        self._last_plot_key: tuple | None = None
        self._last_result: SimulationResult | None = None
        self._dirty_draw = False
        self.clear("No data yet — run a computation.")

    # -- visibility / deferred raster --------------------------------------
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

    def _commit_render(self, key: tuple) -> None:
        """Mark the rendered state and schedule the (deferred) draw."""
        self._last_plot_key = key
        self._finish_render()

    @staticmethod
    def _curves_key(x, curves, *extra) -> tuple:
        """Exact render signature of a curves request (bytes-based)."""
        parts: list = [tuple(str(e) for e in extra)]
        if x is None or len(x) == 0:
            parts.append(None)
        else:
            parts.append(np.asarray(x, dtype=float).tobytes())
        for y, label in curves:
            if y is None or len(y) == 0:
                parts.append((label, None))
            else:
                parts.append((label, np.asarray(y, dtype=float).tobytes()))
        return tuple(parts)

    # -- cursor ---------------------------------------------------------
    def _on_motion(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None or not np.isfinite([x, y]).all():
            return
        self.cursor_label.setText(f"x = {x:.6g}, y = {y:.6g}")

    # -- theme -----------------------------------------------------------
    def _apply_theme(self) -> None:
        style_dark_figure(self.figure, [self.ax])

    # -- generic drawing --------------------------------------------------
    def clear(self, message: str = "") -> None:
        key = ("clear", message)
        if self._last_plot_key == key:
            self._finish_render()
            return
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
                color=TEXT_MUTED,
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        self._apply_theme()
        self._commit_render(key)

    def plot_curves(
        self,
        x: np.ndarray,
        curves: list[tuple[np.ndarray, str]],
        xlabel: str,
        ylabel: str,
        log_y: bool = False,
        title: str = "",
        styles: list[dict] | None = None,
    ) -> None:
        """Plot ``curves = [(y, label), ...]`` against ``x``."""
        key = self._curves_key(x, curves, xlabel, ylabel, log_y, title)
        if self._last_plot_key == key:
            self._finish_render()
            return
        self.ax.clear()
        if x is None or len(x) == 0:
            self.clear("No data available for this view.")
            return
        plotted = 0
        for k, (y, label) in enumerate(curves):
            if y is None or len(y) != len(x):
                continue
            style = styles[k] if styles and k < len(styles) else {}
            self.ax.plot(x, y, label=label, **style)
            plotted += 1
        if plotted == 0:
            self.clear("No data available for this view.")
            return
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        if log_y:
            self.ax.set_yscale("log")
        if len(curves) > 1:
            self.ax.legend(fontsize=8)
        self._apply_theme()
        self._commit_render(key)

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

    # -- material comparisons (MATLAB UIAxes_4 / UIAxes_5) -----------------
    def plot_material_comparison(
        self,
        prop_name: str,
        jacket_expr: str,
        cap_exprs: tuple[str, ...] | list[str],
        nb_nested: int,
        log_y: bool | None = None,
    ) -> None:
        """Jacket (+ enabled capillary rings) property over ``[250, 2500] K``.

        Multi-curve overlays (neon cycle) are drawn for ``Viscosity``
        and ``Surface tension``; the other properties plot the single
        Jacket curve.  X ticks follow MATLAB v1.5: 250 K steps.
        """
        key = (
            "material", prop_name, str(jacket_expr),
            tuple(str(e) for e in cap_exprs), int(nb_nested), log_y,
        )
        if self._last_plot_key == key:
            self._finish_render()
            return
        ylabel, auto_log = _PROPERTY_YLABELS[prop_name]
        if log_y is None:
            log_y = auto_log
        t_grid = np.linspace(_T_MIN_K, _T_MAX_K, 500)

        specs: list[tuple[str, str, dict]] = [
            (str(jacket_expr), "Jacket", dict(CURVE_JACKET))
        ]
        if prop_name in _MULTI_CURVE_PROPS:
            for i in range(min(int(nb_nested), len(cap_exprs), 3)):
                specs.append(
                    (str(cap_exprs[i]), f"Cap #{i + 1}", dict(CURVE_CAPS[i]))
                )

        curves: list[tuple[np.ndarray, str]] = []
        styles: list[dict] = []
        for source, label, style in specs:
            try:
                expr = CompiledExpression(source, ("T",))
                values = np.asarray(expr.vector(t_grid), dtype=float)
                # Constant laws (e.g. "2200") evaluate to a 0-d array.
                values = np.broadcast_to(values, t_grid.shape)
            except (ExpressionError, ValueError):
                continue
            good = np.isfinite(values)
            if not good.any():
                continue
            curves.append((values[good], label))
            styles.append(style)
        if not curves:
            self.clear(f"Formula for {prop_name!r} yields no finite values.")
            return
        self.plot_curves(
            t_grid,
            curves,
            "T (K)",
            ylabel,
            log_y=log_y,
            title=prop_name,
            styles=styles,
        )
        self.ax.set_xticks(np.arange(_T_MIN_K, _T_MAX_K + 1.0, 250.0))
        self.ax.set_xlim(_T_MIN_K, _T_MAX_K)
        self._apply_theme()
        self._commit_render(key)

    # -- furnace profiles ---------------------------------------------------
    def plot_furnace_profile(
        self, z_m: np.ndarray, profile: np.ndarray, t_max_c: float
    ) -> None:
        """Furnace profile ``T/Tmax`` in metres on ``[0, Lfour]``.

        Matches MATLAB ``UIAxes_3``: ``z (m)`` with ``xlim = (0, Lfour)``
        and a 5% head-room on ``ylim`` so constant ``'1'`` profiles do
        not touch the top spine.
        """
        key = self._curves_key(z_m, [(profile, "T/Tmax")], t_max_c)
        if self._last_plot_key == key:
            self._finish_render()
            return
        self.ax.clear()
        self.ax.plot(
            z_m, profile, label="T/Tmax", color=CURVE_JACKET["color"]
        )
        y = np.asarray(profile, dtype=float)
        y_top = 1.05
        if y.size and np.isfinite(y).any():
            y_top = max(1.05, float(np.nanmax(y)) * 1.05)
        self.ax.set_xlabel("z (m)")
        self.ax.set_ylabel("T / Tmax (normalized)")
        self.ax.set_title(f"Furnace profile (Tmax = {t_max_c:.1f} C)")
        self.ax.set_xlim(0.0, float(np.max(z_m)))
        self.ax.set_ylim(0.0, y_top)
        self.ax.legend(fontsize=8)
        self._apply_theme()
        self._commit_render(key)

    def plot_initial_furnace_temperature(
        self,
        L: float,
        Lfour: float,
        t_max_c: float,
        text_c: float,
        profil_expr: str,
    ) -> bool:
        """Pre-run axial furnace temperature ``T(z)`` over ``[0, L]``.

        ``T = Tmax * profil(z)`` inside the hot zone ``z <= Lfour`` and
        ``T = Text`` beyond it (MATLAB v1.5 ``trace`` pre-run view).
        Returns True when the profile could be rendered.
        """
        if not (L > 0.0):
            self.clear("Set a positive drawing length L.")
            return False
        try:
            expr = CompiledExpression(str(profil_expr), ("z",))
        except ExpressionError:
            self.clear("Invalid furnace profile formula.")
            return False
        z_grid = np.linspace(0.0, L, 500)
        prof = np.clip(expr.vector(z_grid), 0.0, 1.0)
        temp = np.where(z_grid <= Lfour, t_max_c * prof, text_c)
        if not np.isfinite(temp).any():
            self.clear("Furnace profile yields no finite temperatures.")
            return False
        curves = [(temp, "Furnace profile")]
        key = self._curves_key(
            z_grid, curves, "z (m)", "T (°C)", False,
            "Furnace temperature profile", Lfour,
        )
        if self._last_plot_key == key:
            self._finish_render()
            return True
        self.plot_curves(
            z_grid,
            curves,
            "z (m)",
            "T (°C)",
            title="Furnace temperature profile",
            styles=[dict(CURVE_JACKET)],
        )
        self.ax.axvline(
            max(Lfour, 0.0), color=CURVE_CAPS[0]["color"], linestyle=":",
            alpha=0.8, label=f"Lfour = {Lfour * 1e3:.1f} mm",
        )
        self.ax.legend(fontsize=8)
        self._apply_theme()
        self._commit_render(key)
        return True

    # -- axial draw views -------------------------------------------------
    def draw_axial_view(self, result: SimulationResult | None, view: str) -> None:
        """Render one of the 11 axial views of a simulation result."""
        if result is None:
            self.clear("No data yet — run a computation.")
            return
        key = ("axial", id(result), view)
        if self._last_plot_key == key:
            self._finish_render()
            return
        self._last_result = result  # pins the id() used in the key
        z_mm = np.asarray(result.z, dtype=float) * 1e3
        if len(z_mm) < 2 or result.z_end <= 0.0:
            self.clear(
                "No trajectory available (the run failed before integration)."
            )
            return
        partial = result.z_end < result.inputs.L - 1e-12
        g = float(getattr(result.inputs, "g", 9.81))

        def _has(name: str, index: int = 0) -> np.ndarray | None:
            arrays = getattr(result, name, ())
            if index >= len(arrays):
                return None
            arr = np.asarray(arrays[index], dtype=float)
            return arr if len(arr) == len(z_mm) else None

        if view.startswith("Velocity"):
            self.plot_curves(
                z_mm, [(result.w * 60.0, "w(z)")],
                "z (mm)", "w (m/min)", title=view,
                styles=[dict(CURVE_JACKET)],
            )
        elif view.startswith("Jacket diameter"):
            self.plot_curves(
                z_mm, [(2.0 * result.R * 1e3, "D(z)")], "z (mm)", "D (mm)",
                title=view, styles=[dict(CURVE_JACKET)],
            )
        elif view.startswith("Jacket bore"):
            self.plot_curves(
                z_mm, [(2.0 * result.r * 1e3, "d(z)")], "z (mm)", "d (mm)",
                title=view, styles=[dict(CURVE_JACKET)],
            )
        elif view.startswith("Capillary 1 outer"):
            rc = _has("Rc")
            self.plot_curves(
                z_mm,
                [(2.0 * rc * 1e6, "Dc1(z)")] if rc is not None else [],
                "z (mm)", "Dc1 (um)", title=view, styles=[dict(CURVE_CAPS[0])],
            )
        elif view.startswith("Capillary 1 bore"):
            rc = _has("rc")
            self.plot_curves(
                z_mm,
                [(2.0 * rc * 1e6, "dc1(z)")] if rc is not None else [],
                "z (mm)", "dc1 (um)", title=view, styles=[dict(CURVE_CAPS[0])],
            )
        elif view.startswith("Capillary wall"):
            rc_o, rc_i = _has("Rc"), _has("rc")
            curves = []
            styles = []
            if rc_o is not None and rc_i is not None:
                curves.append(((rc_o - rc_i) * 1e6, "t(z)"))
                styles.append(dict(CURVE_CAPS[0]))
            for i in range(1, len(result.Rc)):
                ro, ri = _has("Rc", i), _has("rc", i)
                if ro is not None and ri is not None:
                    curves.append(((ro - ri) * 1e6, f"t{i + 1}(z)"))
                    styles.append(dict(CURVE_CAPS[i % 3]))
            self.plot_curves(
                z_mm, curves, "z (mm)", "wall t (um)", title=view,
                styles=styles,
            )
        elif view.startswith("Core diameter"):
            self.plot_curves(
                z_mm, [(result.D_core * 1e6, "Dcore(z)")],
                "z (mm)", "Dcore (um)", title=view, styles=[dict(CURVE_JACKET)],
            )
        elif view.startswith("dc / Dc"):
            rc_o, rc_i = _has("Rc"), _has("rc")
            curves = []
            styles = []
            if rc_o is not None and rc_i is not None:
                curves.append((rc_i / rc_o, "dc1/Dc1"))
                styles.append(dict(CURVE_CAPS[0]))
            for i in range(1, len(result.Rc)):
                ro, ri = _has("Rc", i), _has("rc", i)
                if ro is not None and ri is not None:
                    curves.append((ri / ro, f"dc{i + 1}/Dc{i + 1}"))
                    styles.append(dict(CURVE_CAPS[i % 3]))
            self.plot_curves(
                z_mm, curves, "z (mm)", "dc / Dc (dimensionless)",
                title=view, styles=styles,
            )
        elif view.startswith("Temperature"):
            self.plot_curves(
                z_mm,
                [
                    (result.T - 273.15, "Preform T"),
                    (result.Ta - 273.15, "Furnace Ta"),
                ],
                "z (mm)", "T (°C)", title=view,
                styles=[dict(CURVE_JACKET), dict(CURVE_CAPS[0])],
            )
        elif view.startswith("Draw tension"):
            self.plot_curves(
                z_mm,
                [(np.asarray(result.tau) / g * 1000.0, "tau(z)")],
                "z (mm)", "Draw tension (g)", title=view,
                styles=[dict(CURVE_JACKET)],
            )
        elif view.startswith("Rc1 / Rcontact"):
            rc_o = _has("Rc")
            contact = result.R_contact
            curves = []
            if rc_o is not None:
                curves.append((rc_o / contact, "Rc1/Rcontact"))
            self.plot_curves(
                z_mm, curves, "z (mm)", "Rc1 / Rcontact", title=view,
                styles=[dict(CURVE_JACKET)],
            )
        else:  # pragma: no cover - unknown view
            self.clear(f"Unknown view: {view}")
            return
        self._add_end_value_legend(result)
        if partial:
            self.ax.axvline(
                result.z_end * 1e3, color="#ff5252", linestyle="--",
                alpha=0.8, label=f"stop z = {result.z_end * 1e3:.1f} mm",
            )
        self.ax.legend(fontsize=8)
        self._apply_theme()
        self._commit_render(key)

    def _add_end_value_legend(self, result: SimulationResult) -> None:
        """Append MATLAB-style end-value legend entries (Vf, Df, ...)."""
        if result is None or len(result.z) == 0:
            return
        g = float(getattr(result.inputs, "g", 9.81))
        labels = [
            f"Vf = {result.w[-1] * 60.0:.2f} m/min",
            f"Df = {2.0 * result.R[-1] * 1e6:.2f} µm",
            f"df = {2.0 * result.r[-1] * 1e6:.2f} µm",
        ]
        if result.Rc and len(result.Rc[0]):
            labels.append(
                f"Dcf1 = {2.0 * result.Rc[0][-1] * 1e6:.2f} µm"
            )
        if (
            result.Rc and result.rc
            and len(result.Rc[0]) and len(result.rc[0])
        ):
            labels.append(
                f"tf1 = "
                f"{(result.Rc[0][-1] - result.rc[0][-1]) * 1e6:.2f} µm"
            )
        if len(result.tau):
            labels.append(f"tau_f = {result.tau[-1] / g * 1000.0:.2f} g")
        for text in labels:
            self.ax.plot([], [], lw=0.0, label=text)

    def has_finite_data(self) -> bool:
        """True when the axes hold at least one line with finite points."""
        for line in self.ax.get_lines():
            xy = line.get_xydata()
            if len(xy) and np.isfinite(xy).all():
                return True
        return False
