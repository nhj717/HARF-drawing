"""SEM / microscope cross-section inspector dialog.

Interactive measurement front-end for
:mod:`src.sem_metrology`: shows the micrograph, overlays the
detected circles (OD, zeta envelope, ring bores / outers) and
offers four canvas tools:

* **Scale bar** -- two clicks on the bar ends + a bar length in µm.
* **Circle (3 pt)** -- override the nearest detected circle with a
  3-point circumcircle.
* **Nudge circle** -- drag a circle to re-center it.
* **Wall caliper** -- press-drag across a wall to see the
  strip-averaged profile, its derivative and the sub-pixel wall
  thickness.

The resulting :class:`SemApplyPayload` carries SI metres; ``None``
fields are left untouched by the receiving tab.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.theme import (
    AMBER,
    CYAN,
    GREEN,
    PINK,
    RED,
    TEXT_MUTED,
    mono_font_family,
    style_dark_figure,
)
from app.widgets.scroll import wrap_scroll
from app.widgets.validators import make_double_edit, read_double
from src.sem_metrology import (
    CrossSectionEstimate,
    MeasuredCircle,
    Polarity,
    ScaleCalibration,
    calibrate_from_scale_bar,
    estimate_cross_section,
    fit_circle_3pt,
    measure_wall_profile_subpixel,
    normalize_sem_image,
)

__all__ = ["SemApplyPayload", "SemInspectorDialog"]

_KIND_COLORS = {
    "od": CYAN,
    "zeta": GREEN,
    "ring_bore": PINK,
    "ring_outer": AMBER,
}

_MODE_HINTS = {
    "pan": "Pan: drag to move (zoom via the toolbar).",
    "bar": "Scale bar: click both ends of the bar.",
    "circle": "Circle: click 3 points on one circular edge.",
    "drag": "Nudge: press on a circle and drag it.",
    "caliper": "Caliper: press on one wall face and drag across.",
}


@dataclass(frozen=True)
class SemApplyPayload:
    """Measurements to push into the Scaling tab (SI metres).

    ``stage`` is ``"cane"`` or ``"fiber"``; every other field is
    ``None`` / empty when the user left it unchecked.
    """

    stage: str
    od: float | None = None
    d_core: float | None = None
    n_cap: int | None = None
    ring_bore: tuple[float, ...] = ()
    ring_wall: tuple[float, ...] = ()
    nested: int = 0


def _grid_rows(title: str, rows: list[tuple[str, QWidget]]) -> QGroupBox:
    """A titled group box with a two-column label / widget grid."""
    box = QGroupBox(title)
    layout = QGridLayout(box)
    for r, (label, widget) in enumerate(rows):
        layout.addWidget(QLabel(label), r, 0)
        layout.addWidget(widget, r, 1)
    layout.setColumnStretch(1, 1)
    return box


class SemInspectorDialog(QDialog):
    """Interactive SEM / microscope cross-section inspector."""

    def __init__(
        self,
        image: np.ndarray | None = None,
        calibration: ScaleCalibration | None = None,
        stage: str = "cane",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"SEM / Microscope Inspector ({stage})")
        self.resize(1120, 760)
        self._image = None if image is None else np.asarray(image, float)
        self._calibration = calibration or ScaleCalibration(None)
        self._stage = "fiber" if stage == "fiber" else "cane"
        self._pixel_size_m = self._calibration.pixel_size_m
        self._estimate: CrossSectionEstimate | None = None
        self._norm: np.ndarray | None = None
        self._norm_y0 = 0
        self._overrides: dict[
            tuple[str, int, int], tuple[float, float, float]
        ] = {}
        self._clicks: list[tuple[float, float]] = []
        self._caliper_p0: tuple[float, float] | None = None
        self._caliper_p1: tuple[float, float] | None = None
        self._drag_key: tuple[str, int, int] | None = None
        self._drag_origin: tuple[float, float] | None = None
        self._drag_center: tuple[float, float] | None = None
        self._drag_radius = 0.0
        self._mode = "pan"
        self._build_ui()
        if self._image is not None:
            self._detect()

    # -- UI construction ------------------------------------------------------
    def _build_ui(self) -> None:
        # Figures first: style_dark_figure also fixes Matplotlib fonts.
        self.figure = Figure(figsize=(7.2, 7.2))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._ax = self.figure.add_subplot(111)
        self.navToolbar = NavigationToolbar2QT(self.canvas, self)
        self.profileFigure = Figure(figsize=(7.2, 2.4))
        self.profileCanvas = FigureCanvasQTAgg(self.profileFigure)
        self._prof_ax = self.profileFigure.add_subplot(211)
        self._prof_dax = self.profileFigure.add_subplot(212)
        style_dark_figure(self.figure, [self._ax])
        style_dark_figure(
            self.profileFigure, [self._prof_ax, self._prof_dax]
        )
        self.profileCanvas.setMinimumHeight(170)

        # Canvas tools (exclusive).
        self.toolPan = QPushButton("Pan")
        self.toolBar = QPushButton("Scale bar")
        self.toolCircle = QPushButton("Circle (3 pt)")
        self.toolDrag = QPushButton("Nudge circle")
        self.toolCaliper = QPushButton("Wall caliper")
        self._tool_group = QButtonGroup(self)
        for btn, tid in (
            (self.toolPan, 0),
            (self.toolBar, 1),
            (self.toolCircle, 2),
            (self.toolDrag, 3),
            (self.toolCaliper, 4),
        ):
            btn.setCheckable(True)
            self._tool_group.addButton(btn, tid)
        self.toolPan.setChecked(True)
        self._tool_group.idClicked.connect(self._on_tool)
        tools_box = QGroupBox("Canvas tools")
        tools_grid = QGridLayout(tools_box)
        tools_grid.addWidget(self.toolPan, 0, 0)
        tools_grid.addWidget(self.toolBar, 0, 1)
        tools_grid.addWidget(self.toolCircle, 1, 0)
        tools_grid.addWidget(self.toolDrag, 1, 1)
        tools_grid.addWidget(self.toolCaliper, 2, 0, 1, 2)

        # Detection controls.
        self.polarityCombo = QComboBox()
        self.polarityCombo.addItems(("Bright silica", "Dark boundaries"))
        self.plowSpin = QDoubleSpinBox()
        self.plowSpin.setRange(0.0, 49.0)
        self.plowSpin.setValue(0.5)
        self.plowSpin.setDecimals(1)
        self.plowSpin.setSuffix(" %")
        self.plowSpin.setLocale(QLocale.c())
        self.phighSpin = QDoubleSpinBox()
        self.phighSpin.setRange(50.5, 100.0)
        self.phighSpin.setValue(99.5)
        self.phighSpin.setDecimals(1)
        self.phighSpin.setSuffix(" %")
        self.phighSpin.setLocale(QLocale.c())
        self.ncapHintSpin = QSpinBox()
        self.ncapHintSpin.setRange(0, 12)
        self.ncapHintSpin.setValue(0)
        self.ncapHintSpin.setSpecialValueText("auto")
        self.ncapHintSpin.setLocale(QLocale.c())
        self.nestedSpin = QSpinBox()
        self.nestedSpin.setRange(1, 3)
        self.nestedSpin.setValue(1)
        self.nestedSpin.setLocale(QLocale.c())
        self.redetectButton = QPushButton("Re-detect")
        det_box = _grid_rows(
            "Detection",
            [
                ("Polarity", self.polarityCombo),
                ("p_low", self.plowSpin),
                ("p_high", self.phighSpin),
                ("n_cap hint", self.ncapHintSpin),
                ("Nested rings", self.nestedSpin),
            ],
        )
        det_layout = QVBoxLayout(det_box)
        det_layout.addWidget(self.redetectButton)

        # Calibration controls.
        self.barLengthEdit = make_double_edit(
            50.0, 0.0, 1e5, tooltip="Scale-bar length (µm)"
        )
        self.pixelSizeLabel = QLabel("not calibrated")
        self.pixelSizeLabel.setWordWrap(True)
        cal_box = QGroupBox("Calibration")
        cal_grid = QGridLayout(cal_box)
        cal_grid.addWidget(QLabel("Bar length (µm)"), 0, 0)
        cal_grid.addWidget(self.barLengthEdit, 0, 1)
        cal_grid.addWidget(self.pixelSizeLabel, 1, 0, 1, 2)
        cal_grid.setColumnStretch(1, 1)

        # Apply-to-tab checkboxes.
        self.applyOdCheck = QCheckBox("OD (outer diameter)")
        self.applyCoreCheck = QCheckBox("D_core (core diameter)")
        self.applyNcapCheck = QCheckBox("n_cap (capillary count)")
        self.applyRingDChecks = [
            QCheckBox(f"d_{i + 1} (ring {i + 1} bore)") for i in range(3)
        ]
        self.applyRingTChecks = [
            QCheckBox(f"t_{i + 1} (ring {i + 1} wall)") for i in range(3)
        ]
        self.applyOdCheck.setChecked(True)
        self.applyCoreCheck.setChecked(True)
        self.applyNcapCheck.setChecked(True)
        self.applyRingDChecks[0].setChecked(True)
        self.applyRingTChecks[0].setChecked(True)
        self.wallOnlyButton = QPushButton("Wall thicknesses only")
        self.selectAllButton = QPushButton("Select all")
        apply_box = QGroupBox(f"Apply to {self._stage} tab")
        apply_layout = QVBoxLayout(apply_box)
        for check in (
            self.applyOdCheck,
            self.applyCoreCheck,
            self.applyNcapCheck,
            *self.applyRingDChecks,
            *self.applyRingTChecks,
        ):
            apply_layout.addWidget(check)
        apply_buttons = QHBoxLayout()
        apply_buttons.addWidget(self.wallOnlyButton)
        apply_buttons.addWidget(self.selectAllButton)
        apply_layout.addLayout(apply_buttons)

        # Summary + status.
        self.infoLabel = QLabel("No analysis yet.")
        self.infoLabel.setWordWrap(True)
        self.infoLabel.setFont(QFont(mono_font_family(), 9))
        info_box = QGroupBox("Estimate")
        info_layout = QVBoxLayout(info_box)
        info_layout.addWidget(self.infoLabel)
        self.statusLabel = QLabel("Ready.")
        self.statusLabel.setWordWrap(True)

        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )

        # Layout.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(tools_box)
        left_layout.addWidget(det_box)
        left_layout.addWidget(cal_box)
        left_layout.addWidget(apply_box)
        left_layout.addWidget(info_box)
        left_layout.addWidget(self.statusLabel)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.navToolbar)
        right_layout.addWidget(self.canvas, stretch=1)
        right_layout.addWidget(self.profileCanvas)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(wrap_scroll(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        layout.addWidget(self.buttonBox)

        # Signal wiring.
        self.redetectButton.clicked.connect(self._detect)
        self.wallOnlyButton.clicked.connect(
            lambda: self._set_apply(walls_only=True)
        )
        self.selectAllButton.clicked.connect(
            lambda: self._set_apply(walls_only=False)
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    # -- analysis ------------------------------------------------------------
    def _polarity(self) -> Polarity:
        if self.polarityCombo.currentText().startswith("Dark"):
            return Polarity.DARK_BOUNDARIES
        return Polarity.BRIGHT_SILICA

    def _detect(self) -> None:
        if self._image is None:
            return
        polarity = self._polarity()
        try:
            est = estimate_cross_section(
                self._image,
                polarity,
                self._pixel_size_m,
                n_cap_hint=self.ncapHintSpin.value() or None,
                nested_hint=self.nestedSpin.value(),
                p_low=self.plowSpin.value(),
                p_high=self.phighSpin.value(),
            )
        except ValueError as exc:
            self._set_status(f"Detection failed: {exc}")
            return
        banner = est.banner_crop_rows
        if banner is not None and banner[0] == 0:
            content = self._image[banner[1] :]
            self._norm_y0 = int(banner[1])
        elif banner is not None:
            content = self._image[: banner[0]]
            self._norm_y0 = 0
        else:
            content = self._image
            self._norm_y0 = 0
        self._norm = normalize_sem_image(
            content,
            polarity,
            self.plowSpin.value(),
            self.phighSpin.value(),
        )
        self.set_analysis(est, self._calibration)

    def set_analysis(
        self,
        estimate: CrossSectionEstimate,
        calibration: ScaleCalibration | None = None,
    ) -> None:
        """Install an estimate (and optional calibration) and redraw."""
        self._estimate = estimate
        self._overrides.clear()
        if calibration is not None:
            self._calibration = calibration
            if calibration.pixel_size_m:
                self._pixel_size_m = calibration.pixel_size_m
        n_rings = len(estimate.ring_bore_px)
        if n_rings:
            self.nestedSpin.setValue(max(1, min(3, n_rings)))
        self._update_info()
        self._render()
        self._set_status("Analysis installed.")

    # -- canvas interaction --------------------------------------------------
    def _on_tool(self, tool_id: int) -> None:
        modes = {0: "pan", 1: "bar", 2: "circle", 3: "drag", 4: "caliper"}
        self._mode = modes.get(tool_id, "pan")
        self._clicks.clear()
        self._caliper_p0 = None
        self._caliper_p1 = None
        self._drag_key = None
        if self._mode == "pan":
            self.navToolbar.pan()
        elif self.navToolbar.mode:
            if self.navToolbar.mode.startswith("pan"):
                self.navToolbar.pan()
            elif self.navToolbar.mode.startswith("zoom"):
                self.navToolbar.zoom()
        self._set_status(_MODE_HINTS.get(self._mode, ""))

    def _on_press(self, event) -> None:
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if self._mode == "bar":
            self._clicks.append((x, y))
            if len(self._clicks) >= 2:
                self._finish_scale_bar()
            self._render()
        elif self._mode == "circle":
            self._clicks.append((x, y))
            if len(self._clicks) >= 3:
                self._finish_circle(x, y)
            self._render()
        elif self._mode == "caliper":
            self._caliper_p0 = (x, y)
            self._caliper_p1 = None
            self._render()
        elif self._mode == "drag":
            circle = self._nearest_circle(x, y)
            if circle is None:
                return
            self._drag_key = (circle.kind, circle.ring, circle.azimuth)
            self._drag_origin = (x, y)
            self._drag_center = (circle.xc, circle.yc)
            self._drag_radius = circle.r

    def _on_motion(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if self._mode == "caliper" and self._caliper_p0 is not None:
            self._caliper_p1 = (x, y)
            self._render()
        elif (
            self._mode == "drag"
            and self._drag_key is not None
            and self._drag_origin is not None
            and self._drag_center is not None
        ):
            dx = x - self._drag_origin[0]
            dy = y - self._drag_origin[1]
            self._overrides[self._drag_key] = (
                self._drag_center[0] + dx,
                self._drag_center[1] + dy,
                self._drag_radius,
            )
            self._render()

    def _on_release(self, event) -> None:
        if event.button != 1:
            return
        if self._mode == "caliper" and self._caliper_p1 is not None:
            self._run_caliper()
        self._drag_key = None
        self._drag_origin = None
        self._drag_center = None

    def _finish_scale_bar(self) -> None:
        p0, p1 = self._clicks[0], self._clicks[1]
        self._clicks.clear()
        try:
            length_m = read_double(self.barLengthEdit) * 1e-6
            pxm = calibrate_from_scale_bar(p0, p1, length_m)
        except ValueError as exc:
            self._set_status(f"Scale-bar error: {exc}")
            return
        self._pixel_size_m = pxm
        self._calibration = ScaleCalibration(
            pxm, "scale_bar", "2-click scale bar"
        )
        self._update_info()
        self._set_status(f"Calibrated: {pxm * 1e6:.4f} µm/px.")

    def _finish_circle(self, x: float, y: float) -> None:
        p0, p1, p2 = self._clicks[0], self._clicks[1], self._clicks[2]
        self._clicks.clear()
        target = self._nearest_circle(x, y)
        if target is None:
            self._set_status("No detected circle near the 3 points.")
            return
        try:
            xc, yc, r = fit_circle_3pt(p0, p1, p2)
        except ValueError as exc:
            self._set_status(f"Circle error: {exc}")
            return
        self._overrides[(target.kind, target.ring, target.azimuth)] = (
            xc,
            yc,
            r,
        )
        self._update_info()
        self._set_status(f"Overridden {target.kind} circle of ring "
                         f"{target.ring} (tube {target.azimuth}).")

    def _run_caliper(self) -> None:
        p0, p1 = self._caliper_p0, self._caliper_p1
        if p0 is None or p1 is None:
            return
        if self._norm is None:
            self._set_status("Caliper needs a loaded image.")
            return
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length < 8.0:
            return
        profile = measure_wall_profile_subpixel(
            self._norm, (p0[0], p0[1] - self._norm_y0), (dx, dy), length
        )
        self._plot_profile(profile)
        if profile.thickness_px is None:
            self._set_status("Caliper: no wall resolved along the drag.")
            return
        if self._pixel_size_m:
            um = profile.thickness_px * self._pixel_size_m * 1e6
            self._set_status(
                f"Caliper: wall = {um:.3f} µm "
                f"({profile.thickness_px:.2f} px)."
            )
        else:
            self._set_status(
                f"Caliper: wall = {profile.thickness_px:.2f} px."
            )

    def _plot_profile(self, profile) -> None:
        self._prof_ax.clear()
        self._prof_dax.clear()
        s = profile.s
        self._prof_ax.plot(s, profile.intensity, color=CYAN, lw=1.4)
        self._prof_ax.set_ylabel("I(s)")
        for edge in profile.edge_positions:
            if math.isfinite(edge):
                self._prof_ax.axvline(edge, color=PINK, lw=0.9, ls="--")
        self._prof_dax.plot(s, profile.derivative, color=AMBER, lw=1.2)
        self._prof_dax.set_ylabel("dI/ds")
        self._prof_dax.set_xlabel("s (px)")
        style_dark_figure(
            self.profileFigure, [self._prof_ax, self._prof_dax]
        )
        self.profileFigure.tight_layout()
        self.profileCanvas.draw_idle()

    # -- rendering -----------------------------------------------------------
    def _effective_circles(self) -> list[MeasuredCircle]:
        est = self._estimate
        if est is None:
            return []
        out: list[MeasuredCircle] = []
        for c in est.circles:
            ov = self._overrides.get((c.kind, c.ring, c.azimuth))
            if ov is None:
                out.append(c)
            else:
                out.append(
                    MeasuredCircle(c.kind, c.ring, c.azimuth, *ov)
                )
        return out

    def _nearest_circle(self, x: float, y: float) -> MeasuredCircle | None:
        best = None
        best_gap = 1e18
        for c in self._effective_circles():
            gap = abs(math.hypot(c.xc - x, c.yc - y) - c.r)
            if gap < best_gap:
                best_gap = gap
                best = c
        if best is None or best_gap > 15.0:
            return None
        return best

    def _render(self) -> None:
        self._ax.clear()
        if self._image is not None:
            self._ax.imshow(
                self._image,
                cmap="gray",
                origin="upper",
                interpolation="nearest",
            )
        circles = self._effective_circles()
        if self._image is None and not circles:
            self._ax.text(
                0.5, 0.5, "no image loaded", ha="center", va="center",
                color=TEXT_MUTED, transform=self._ax.transAxes,
            )
        for c in circles:
            self._ax.add_patch(
                Circle(
                    (c.xc, c.yc), c.r, fill=False,
                    color=_KIND_COLORS.get(c.kind, TEXT_MUTED), lw=1.1,
                )
            )
        if self._estimate is not None:
            cx, cy = self._estimate.center_px
            self._ax.plot([cx], [cy], "+", color=RED, ms=12)
        if self._clicks:
            self._ax.plot(
                [p[0] for p in self._clicks],
                [p[1] for p in self._clicks],
                "x",
                color=RED,
                ms=9,
            )
        if self._caliper_p0 is not None:
            end = self._caliper_p1 or self._caliper_p0
            self._ax.plot(
                [self._caliper_p0[0], end[0]],
                [self._caliper_p0[1], end[1]],
                "-",
                color=RED,
                lw=1.4,
            )
        if self._image is None and circles:
            x0 = min(c.xc - c.r for c in circles) - 10.0
            x1 = max(c.xc + c.r for c in circles) + 10.0
            y0 = min(c.yc - c.r for c in circles) - 10.0
            y1 = max(c.yc + c.r for c in circles) + 10.0
            self._ax.set_xlim(x0, x1)
            self._ax.set_ylim(y1, y0)  # raster orientation
        self._ax.set_aspect("equal")
        style_dark_figure(self.figure, [self._ax])
        self.canvas.draw_idle()

    # -- info / payload --------------------------------------------------------
    def _update_info(self) -> None:
        est = self._estimate
        pxm = self._pixel_size_m

        def length(px: float) -> str:
            return f"{px * pxm * 1e6:.2f} µm" if pxm else f"{px:.1f} px"

        lines: list[str] = []
        if pxm:
            lines.append(
                f"pixel size: {pxm * 1e6:.4f} µm/px "
                f"({self._calibration.method})"
            )
        else:
            lines.append(
                "pixel size: not calibrated (use the scale-bar tool)"
            )
        if est is None:
            lines.append("analysis : none yet")
        else:
            lines.append(f"OD       : {length(est.od_px)}")
            lines.append(f"zeta     : {length(est.zeta_px)}")
            if math.isfinite(est.d_core_px):
                lines.append(f"D_core   : {length(est.d_core_px)}")
            lines.append(
                f"n_cap    : {est.n_cap} "
                f"(theta0 = {math.degrees(est.theta0_rad):.1f} deg)"
            )
            for i in range(len(est.ring_bore_px)):
                bore = est.ring_bore_px[i]
                wall = (
                    est.ring_wall_px[i]
                    if i < len(est.ring_wall_px) else math.nan
                )
                lines.append(
                    f"ring {i + 1}   : d={length(bore)}  "
                    f"t={length(wall)}"
                )
            if est.warnings:
                lines.append("warnings : " + "; ".join(est.warnings))
        self.infoLabel.setText("\n".join(lines))
        if pxm:
            self.pixelSizeLabel.setText(
                f"{pxm * 1e6:.4f} µm/px ({self._calibration.method})"
            )
        else:
            self.pixelSizeLabel.setText("not calibrated")

    def _ring_bore_px(self, ring: int) -> float | None:
        radii = [
            c.r
            for c in self._effective_circles()
            if c.kind == "ring_bore" and c.ring == ring
        ]
        if radii:
            return 2.0 * float(np.mean(radii))
        est = self._estimate
        if est is not None and len(est.ring_bore_px) >= ring:
            return float(est.ring_bore_px[ring - 1])
        return None

    def _ring_wall_px(self, ring: int) -> float | None:
        pairs: dict[int, dict[str, float]] = {}
        for c in self._effective_circles():
            if c.ring != ring or c.kind not in ("ring_bore", "ring_outer"):
                continue
            pairs.setdefault(c.azimuth, {})[c.kind] = c.r
        walls = [
            v["ring_outer"] - v["ring_bore"]
            for v in pairs.values()
            if "ring_outer" in v and "ring_bore" in v
        ]
        if walls:
            return float(np.mean(walls))
        est = self._estimate
        if est is not None and len(est.ring_wall_px) >= ring:
            return float(est.ring_wall_px[ring - 1])
        return None

    def payload(self) -> SemApplyPayload:
        """The checked measurements as a :class:`SemApplyPayload`."""
        est = self._estimate
        if est is None or not self._pixel_size_m:
            return SemApplyPayload(stage=self._stage, nested=0)
        pxm = self._pixel_size_m
        circles = self._effective_circles()
        od = None
        if self.applyOdCheck.isChecked():
            c = next((k for k in circles if k.kind == "od"), None)
            od = (2.0 * c.r if c is not None else est.od_px) * pxm
        d_core = None
        if self.applyCoreCheck.isChecked() and math.isfinite(
            est.d_core_px
        ):
            d_core = est.d_core_px
            ze = next((k for k in circles if k.kind == "zeta"), None)
            r1 = [
                k.r for k in circles
                if k.kind == "ring_outer" and k.ring == 1
            ]
            if ze is not None and r1:
                d_core = 2.0 * ze.r - 2.0 * float(np.mean(r1))
            d_core = float(d_core) * pxm
        n_cap = int(est.n_cap) if self.applyNcapCheck.isChecked() else None
        ring_bores: list[float] = []
        ring_walls: list[float] = []
        touched = 0
        for i in range(3):
            bore = self._ring_bore_px(i + 1)
            wall = self._ring_wall_px(i + 1)
            want_d = self.applyRingDChecks[i].isChecked()
            want_t = self.applyRingTChecks[i].isChecked()
            has_d = want_d and bore is not None
            has_t = want_t and wall is not None
            if has_d or has_t:
                touched += 1
                if has_d:
                    ring_bores.append(float(bore) * pxm)
                if has_t:
                    ring_walls.append(float(wall) * pxm)
        return SemApplyPayload(
            stage=self._stage,
            od=od,
            d_core=d_core,
            n_cap=n_cap,
            ring_bore=tuple(ring_bores),
            ring_wall=tuple(ring_walls),
            nested=touched,
        )

    def _set_apply(self, walls_only: bool) -> None:
        self.applyOdCheck.setChecked(not walls_only)
        self.applyCoreCheck.setChecked(not walls_only)
        self.applyNcapCheck.setChecked(not walls_only)
        for check in self.applyRingDChecks:
            check.setChecked(not walls_only)
        for check in self.applyRingTChecks:
            check.setChecked(True)

    def _set_status(self, text: str) -> None:
        self.statusLabel.setText(text)
