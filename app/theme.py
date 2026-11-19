"""Cyberpunk Neon theme: QSS stylesheet, KPI cards, dark neon Matplotlib.

Mirrors the GNLSE-Antigravity workstation theme:

* background ``#0f131d`` with layered dark surfaces,
* cyan ``#00f5ff`` primary, pink ``#ff4b89`` secondary, purple ``#703eda``
  tertiary accents, green ``#00e676`` success, amber ``#ffab00`` warning,
  red ``#ff5252`` error,
* resolved monospaced HUD typography (installed families only, never a
  bare ``monospace`` token that triggers macOS CoreText font scans),
* neon-bordered KPI cards and dark neon Matplotlib figures.

The neon curve cycle assigns cyan solid to the Jacket (primary curve)
and pink/green/amber dashed to capillary rings 1..3.
"""

from __future__ import annotations

import contextlib
import sys

from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

__all__ = [
    "AMBER",
    "BG_DARK",
    "BORDER",
    "BORDER_SUBTLE",
    "CURVE_CAPS",
    "CURVE_JACKET",
    "CYAN",
    "CYAN_BRIGHT",
    "CYAN_DARK",
    "CYAN_LIGHT",
    "GREEN",
    "PINK",
    "PINK_DARK",
    "PINK_LIGHT",
    "PURPLE",
    "PURPLE_DARK",
    "PURPLE_LIGHT",
    "RED",
    "RING_FACE_COLORS",
    "SURFACE",
    "SURFACE_BRIGHT",
    "SURFACE_HIGH",
    "SURFACE_HIGHEST",
    "SURFACE_LOW",
    "SURFACE_LOWEST",
    "TEXT_MUTED",
    "TEXT_PRIMARY",
    "TOOLBAR_QSS",
    "KpiCard",
    "build_cyberpunk_qss",
    "configure_matplotlib_fonts",
    "font_stack",
    "make_kpi_card",
    "mono_font_family",
    "set_application_font",
    "style_dark_figure",
    "ui_font_family",
]

# -- palette -------------------------------------------------------------------
BG_DARK = "#0f131d"
SURFACE_LOWEST = "#0a0e18"
SURFACE_LOW = "#171b26"
SURFACE = "#1c1f2a"
SURFACE_HIGH = "#262a35"
SURFACE_HIGHEST = "#313540"
SURFACE_BRIGHT = "#353944"

CYAN = "#00f5ff"
CYAN_BRIGHT = "#63f7ff"
CYAN_LIGHT = "#e9feff"
CYAN_DARK = "#003739"

PINK = "#ff4b89"
PINK_LIGHT = "#ffb1c3"
PINK_DARK = "#3f0019"

PURPLE = "#703eda"
PURPLE_LIGHT = "#e5d7ff"
PURPLE_DARK = "#23005c"

TEXT_PRIMARY = "#dfe2f1"
TEXT_MUTED = "#b9caca"

BORDER = "#3a494a"
BORDER_SUBTLE = "#262a35"

GREEN = "#00e676"
AMBER = "#ffab00"
RED = "#ff5252"

# Installed-font resolution: Qt on macOS scans CoreText and warns
# ("Replace uses of missing font family 'Monospace' ...") on every
# polish whenever QSS or QFont references an uninstalled family, so
# every stylesheet must use *resolved, existing* family names only.
_MONO_CANDIDATES = (
    "JetBrains Mono", "Menlo", "Consolas", "DejaVu Sans Mono",
    "Courier New", "Monaco",
)
_UI_CANDIDATES = (
    "JetBrains Mono", "Space Grotesk", "Segoe UI", "Helvetica Neue",
    "DejaVu Sans", "Arial",
)
_PLATFORM_MONO = {"darwin": "Menlo", "win32": "Consolas"}
_PLATFORM_UI = {
    "darwin": "Helvetica Neue", "win32": "Segoe UI", "linux": "DejaVu Sans",
}


def _installed_families() -> set[str] | None:
    """Families known to Qt right now, or ``None`` when unavailable."""
    if QApplication.instance() is None:
        return None
    try:
        from PySide6.QtGui import QFontDatabase

        families = set(QFontDatabase.families())
        return families or None
    except Exception:  # noqa: BLE001 - font DB must never crash the UI
        return None


def _resolve_family(candidates: tuple[str, ...], fallback: str) -> str:
    """First installed family among *candidates*, else *fallback*."""
    families = _installed_families()
    if families is not None:
        for name in candidates:
            if name in families:
                return name
    return fallback


def mono_font_family() -> str:
    """An installed monospace family (never a bare ``monospace`` token)."""
    families = _installed_families()
    if families is None:
        return _PLATFORM_MONO.get(sys.platform, "DejaVu Sans Mono")
    try:
        from PySide6.QtGui import QFontDatabase

        fixed = QFontDatabase.systemFont(
            QFontDatabase.SystemFont.FixedFont
        ).family()
    except Exception:  # noqa: BLE001
        fixed = ""
    if fixed and fixed in families:
        return fixed
    return _resolve_family(
        _MONO_CANDIDATES, _PLATFORM_MONO.get(sys.platform, "DejaVu Sans Mono")
    )


def ui_font_family() -> str:
    """An installed proportional UI family for QSS ``font-family``."""
    return _resolve_family(
        _UI_CANDIDATES, _PLATFORM_UI.get(sys.platform, "DejaVu Sans")
    )


def font_stack() -> str:
    """QSS font stack of resolved, installed families (no generics)."""
    return f"'{ui_font_family()}', '{mono_font_family()}'"


def set_application_font(app: QApplication | None = None) -> None:
    """Pin the application font to a resolved, installed family.

    Qt's offscreen/macOS default application font resolves through the
    missing ``Sans Serif`` alias, which logs a ``qt.qpa.fonts`` warning
    the first time a widget or Matplotlib canvas queries its metrics.
    Pinning a concrete family up front keeps the console warning-free.
    """
    app = app or QApplication.instance()
    if app is None:
        return
    # Fonts must never crash the UI; Qt font APIs raise at worst
    # RuntimeError/TypeError on a torn-down application.
    with contextlib.suppress(RuntimeError, TypeError):
        font = app.font()
        font.setFamily(ui_font_family())
        app.setFont(font)


def configure_matplotlib_fonts() -> None:
    """Point Matplotlib at safe existing families (no bare monospace).

    ``DejaVu Sans Mono`` ships with Matplotlib itself, so it is kept as
    the guaranteed last entry of ``font.monospace``.
    """
    import matplotlib as mpl

    mono = mono_font_family()
    families = [f for f in (mono, "DejaVu Sans Mono") if f]
    if families[-1] != "DejaVu Sans Mono":
        families.append("DejaVu Sans Mono")
    mpl.rcParams["font.monospace"] = families

# -- neon curve cycle ----------------------------------------------------------
CURVE_JACKET = {"color": CYAN, "linestyle": "-", "linewidth": 1.8}
CURVE_CAPS = (
    {"color": PINK, "linestyle": "--", "linewidth": 1.6},
    {"color": GREEN, "linestyle": "--", "linewidth": 1.6},
    {"color": AMBER, "linestyle": "--", "linewidth": 1.6},
)
RING_FACE_COLORS = (PINK, GREEN, AMBER)


def build_cyberpunk_qss(colors: dict | None = None) -> str:
    """Generate the Cyberpunk Neon QSS with optional color overrides."""
    c = colors or {}
    bg_dark = c.get("bg_dark", BG_DARK)
    surface_lowest = c.get("surface_lowest", SURFACE_LOWEST)
    surface_low = c.get("surface_low", SURFACE_LOW)
    surface = c.get("surface", SURFACE)
    surface_high = c.get("surface_high", SURFACE_HIGH)
    cyan_accent = c.get("cyan", CYAN)
    cyan_bright = c.get("cyan_bright", CYAN_BRIGHT)
    cyan_light = c.get("cyan_light", CYAN_LIGHT)
    cyan_dark = c.get("cyan_dark", CYAN_DARK)
    pink_accent = c.get("pink", PINK)
    pink_light = c.get("pink_light", PINK_LIGHT)
    pink_dark = c.get("pink_dark", PINK_DARK)
    purple_accent = c.get("purple", PURPLE)
    text_primary = c.get("text_primary", TEXT_PRIMARY)
    text_muted = c.get("text_muted", TEXT_MUTED)
    border_color = c.get("border", BORDER)
    border_subtle = c.get("border_subtle", BORDER_SUBTLE)
    green_success = c.get("green", GREEN)

    return f"""
/* Global reset & typography */
QWidget {{
    background-color: {bg_dark};
    color: {text_primary};
    font-family: {font_stack()};
    font-size: 11px;
    selection-background-color: {purple_accent};
    selection-color: #ffffff;
}}
QMainWindow {{
    background-color: {bg_dark};
}}
QToolTip {{
    background-color: {surface_low};
    color: {text_primary};
    border: 1px solid {cyan_accent};
    padding: 3px;
}}

/* Scrollbars */
QScrollBar:vertical {{
    background: {surface_lowest};
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {surface_high};
    min-height: 20px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: {cyan_accent};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: {surface_lowest};
    height: 10px;
    margin: 0px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background: {surface_high};
    min-width: 20px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {cyan_accent};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* Tabs (cyberpunk segmented ribbon) */
QTabWidget::pane {{
    border: 1px solid {border_subtle};
    background-color: {bg_dark};
    border-radius: 6px;
    top: -1px;
}}
QTabBar::tab {{
    background-color: {surface_lowest};
    color: {text_muted};
    border: 1px solid {border_subtle};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 7px 18px;
    margin-right: 2px;
    font-weight: 600;
    font-size: 11px;
}}
QTabBar::tab:selected {{
    background-color: {surface};
    color: {cyan_accent};
    border-top: 2px solid {cyan_accent};
    border-bottom: 1px solid {surface};
    font-weight: 700;
}}
QTabBar::tab:hover:!selected {{
    background-color: {surface_low};
    color: {text_primary};
}}

/* Push buttons */
QPushButton {{
    background-color: {surface};
    color: {text_primary};
    border: 1px solid {border_subtle};
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {surface_high};
    border-color: {border_color};
    color: {cyan_light};
}}
QPushButton:pressed {{
    background-color: {surface_lowest};
    border-color: {cyan_accent};
}}
QPushButton:disabled {{
    background-color: {surface_lowest};
    color: {text_muted};
    border-color: {border_subtle};
}}
QPushButton#btn_primary {{
    background-color: {cyan_accent};
    color: {cyan_dark};
    font-weight: 700;
    border: none;
}}
QPushButton#btn_primary:hover {{
    background-color: {cyan_bright};
}}
QPushButton#btn_primary:pressed {{
    background-color: {cyan_light};
}}
QPushButton#btn_primary:disabled {{
    background-color: {surface_lowest};
    color: {text_muted};
}}
QPushButton#btn_danger {{
    background-color: {pink_accent};
    color: {pink_dark};
    border: 1px solid {pink_accent};
    font-weight: bold;
}}
QPushButton#btn_danger:hover {{
    background-color: {pink_light};
}}
QPushButton#btn_danger:disabled {{
    background-color: {surface_lowest};
    color: {text_muted};
    border-color: {border_subtle};
}}
QPushButton#btn_export, QPushButton#btn_accent {{
    background-color: {purple_accent};
    color: #ffffff;
    border: none;
    font-weight: 700;
}}
QPushButton#btn_export:hover, QPushButton#btn_accent:hover {{
    background-color: {cyan_bright};
    color: {cyan_dark};
}}

/* Inputs */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {surface_lowest};
    color: {text_primary};
    border: 1px solid {border_subtle};
    border-radius: 4px;
    padding: 5px 8px;
    selection-background-color: {purple_accent};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1.5px solid {cyan_accent};
    background-color: {surface_low};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {surface_high};
    border: none;
    width: 16px;
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {surface_lowest};
    color: {text_muted};
    border: 1px solid {border_subtle};
}}
QLabel:disabled {{
    color: {text_muted};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {cyan_accent};
}}

/* ComboBoxes */
QComboBox {{
    background-color: {surface_lowest};
    color: {text_primary};
    border: 1px solid {border_subtle};
    border-radius: 4px;
    padding: 5px 8px;
    min-width: 80px;
}}
QComboBox:focus, QComboBox:on {{
    border: 1.5px solid {cyan_accent};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid {border_subtle};
}}
QComboBox QAbstractItemView {{
    background-color: {surface_low};
    color: {text_primary};
    selection-background-color: {purple_accent};
    selection-color: #ffffff;
    border: 1px solid {border_color};
    outline: none;
}}

/* CheckBoxes */
QCheckBox {{
    color: {text_primary};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid {border_color};
    background-color: {surface_lowest};
}}
QCheckBox::indicator:hover {{
    border-color: {cyan_accent};
}}
QCheckBox::indicator:checked {{
    background-color: {cyan_accent};
    border-color: {cyan_accent};
    image: none;
}}

/* Group boxes (HUD panels) */
QGroupBox {{
    font-weight: bold;
    border: 1px solid {border_subtle};
    border-radius: 6px;
    margin-top: 18px;
    padding-top: 12px;
    background-color: {surface_low};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 0px;
    padding: 2px 8px;
    color: {cyan_accent};
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 0.5px;
    background-color: {surface_low};
    border-radius: 3px;
}}
QGroupBox:disabled {{
    border-color: {border_subtle};
    background-color: {surface_lowest};
    color: {text_muted};
}}
QGroupBox:disabled::title {{
    color: {text_muted};
    background-color: {surface_lowest};
}}

/* Sliders */
QSlider::groove:horizontal {{
    height: 4px;
    background: {surface_high};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {cyan_accent};
    width: 12px;
    height: 12px;
    margin: -5px 0px;
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{
    background: {cyan_bright};
}}
QSlider::sub-page:horizontal {{
    background: {cyan_accent};
    border-radius: 2px;
}}

/* Progress bar (purple -> cyan gradient) */
QProgressBar {{
    border: 1px solid {border_color};
    border-radius: 4px;
    background-color: {surface_lowest};
    text-align: center;
    color: {text_primary};
    font-weight: 600;
}}
QProgressBar::chunk {{
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 {purple_accent}, stop: 0.55 {cyan_accent},
        stop: 1 {cyan_bright}
    );
    border-radius: 3px;
}}

/* Splitter */
QSplitter::handle {{
    background: {surface_high};
    width: 2px;
    height: 2px;
}}
QSplitter::handle:hover {{
    background: {cyan_accent};
}}
QSplitter {{
    background-color: {bg_dark};
}}

/* Toolbar / menu */
QToolBar {{
    background-color: {surface_low};
    border-bottom: 1px solid {border_subtle};
    spacing: 4px;
    padding: 3px;
}}
QToolBar QToolButton {{
    background-color: transparent;
    color: {text_primary};
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 4px 8px;
}}
QToolBar QToolButton:hover {{
    background-color: {surface_high};
    color: {cyan_accent};
}}
QMenuBar {{
    background-color: {surface_low};
    color: {text_primary};
    border-bottom: 1px solid {border_subtle};
}}
QMenuBar::item {{
    background-color: transparent;
    padding: 4px 10px;
    border-radius: 3px;
}}
QMenuBar::item:selected {{
    background-color: {surface_high};
    color: {cyan_accent};
}}
QMenu {{
    background-color: {surface_low};
    color: {text_primary};
    border: 1px solid {border_color};
    border-radius: 4px;
}}
QMenu::item {{
    padding: 5px 24px;
}}
QMenu::item:selected {{
    background-color: {purple_accent};
    color: #ffffff;
}}

/* Status bar */
QStatusBar {{
    background-color: {surface_low};
    color: {text_muted};
    border-top: 1px solid {border_subtle};
}}
QStatusBar::item {{
    border: none;
}}

/* KPI card frames */
KpiCard {{
    background-color: {surface_low};
    border: 1px solid {border_subtle};
    border-top: 2px solid {cyan_accent};
    border-radius: 6px;
    padding: 6px 10px;
}}
KpiCard:disabled {{
    background-color: {surface_lowest};
    border: 1px solid {border_subtle};
    border-top: 2px solid {border_subtle};
    color: {text_muted};
}}
KpiCard:hover {{
    border-color: {cyan_accent};
    background-color: {surface};
}}
QFrame#kpi_frame {{
    background-color: {surface_low};
    border: 1px solid {border_subtle};
    border-radius: 6px;
}}

/* Ratio badges: neon green (ok) / neon red (alert) on dark chips. */
QLabel#ratio_badge_ok {{
    color: {green_success};
    background-color: rgba(0, 230, 118, 0.10);
    border: 1px solid {green_success};
    border-radius: 3px;
    padding: 1px 8px;
    font-weight: bold;
}}
QLabel#ratio_badge_bad {{
    color: #ff5252;
    background-color: rgba(255, 82, 82, 0.10);
    border: 1px solid #ff5252;
    border-radius: 3px;
    padding: 1px 8px;
    font-weight: bold;
}}
"""


# -- Matplotlib dark neon styling -----------------------------------------------
def style_dark_figure(figure, axes=None) -> None:
    """Apply the dark neon Matplotlib look to a figure and its axes."""
    configure_matplotlib_fonts()
    figure.patch.set_facecolor(BG_DARK)
    targets = figure.axes if axes is None else axes
    for ax in targets:
        ax.set_facecolor(SURFACE_LOWEST)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(
            colors=TEXT_MUTED, labelcolor=TEXT_MUTED, which="both"
        )
        ax.xaxis.label.set_color(TEXT_PRIMARY)
        ax.yaxis.label.set_color(TEXT_PRIMARY)
        ax.title.set_color(TEXT_PRIMARY)
        ax.grid(
            True, linestyle="--", color=BORDER, alpha=0.35, linewidth=0.7
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(SURFACE)
            legend.get_frame().set_edgecolor(BORDER)
            for text in legend.get_texts():
                text.set_color(TEXT_PRIMARY)


TOOLBAR_QSS = f"""
NavigationToolbar2QT, QToolBar {{
    background-color: {SURFACE_LOW};
    border: none;
    border-bottom: 1px solid {BORDER_SUBTLE};
    spacing: 2px;
    padding: 2px;
}}
NavigationToolbar2QT QToolButton {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 2px 4px;
}}
NavigationToolbar2QT QToolButton:hover {{
    background-color: {SURFACE_HIGH};
    color: {CYAN};
}}
QLabel {{
    background-color: transparent;
}}
"""


# -- KPI cards ------------------------------------------------------------------
class KpiCard(QFrame):
    """Individual cyberpunk telemetry metric card (GNLSE design)."""

    def __init__(
        self,
        title: str,
        initial_value: str = "--",
        unit: str = "",
        accent_color: str = CYAN,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.accent_color = accent_color
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            KpiCard {{
                background-color: {SURFACE_LOW};
                border: 1px solid {BORDER_SUBTLE};
                border-top: 2px solid {accent_color};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            KpiCard:hover {{
                border-color: {accent_color};
                background-color: {SURFACE};
            }}
            KpiCard:disabled {{
                background-color: {SURFACE_LOWEST};
                border: 1px solid {BORDER_SUBTLE};
                border-top: 2px solid {BORDER_SUBTLE};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.lbl_title = QLabel(title.upper())
        self.lbl_title.setStyleSheet(f"""
            color: {TEXT_MUTED};
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)
        layout.addWidget(self.lbl_title)

        val_layout = QHBoxLayout()
        val_layout.setContentsMargins(0, 0, 0, 0)
        val_layout.setSpacing(4)

        self.lbl_value = QLabel(initial_value)
        self.lbl_value.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-size: 15px;
            font-weight: 800;
            font-family: '{mono_font_family()}';
        """)
        self.lbl_unit = QLabel(unit)
        self.lbl_unit.setStyleSheet(f"""
            color: {accent_color};
            font-size: 10px;
            font-weight: 600;
        """)

        val_layout.addWidget(self.lbl_value)
        val_layout.addWidget(self.lbl_unit)
        val_layout.addStretch()
        layout.addLayout(val_layout)

        self.lbl_detail = QLabel("")
        self.lbl_detail.setStyleSheet(f"""
            color: {TEXT_MUTED};
            font-size: 9px;
            font-weight: 400;
        """)
        layout.addWidget(self.lbl_detail)

    def set_data(
        self,
        value: str,
        unit: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.lbl_value.setText(value)
        if unit is not None:
            self.lbl_unit.setText(unit)
        if detail is not None:
            self.lbl_detail.setText(detail)

    def value_text(self) -> str:
        """Current numeric readout text (``--`` before any run)."""
        return self.lbl_value.text()


def make_kpi_card(
    title: str,
    initial_value: str = "--",
    unit: str = "",
    accent_color: str = CYAN,
    parent: QWidget | None = None,
) -> KpiCard:
    """Factory for a neon-bordered KPI readout card."""
    card = KpiCard(
        title,
        initial_value=initial_value,
        unit=unit,
        accent_color=accent_color,
        parent=parent,
    )
    return card
