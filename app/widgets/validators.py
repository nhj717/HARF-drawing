"""Reusable input validation helpers and ratio/contact badges.

A *ratio badge* is a small coloured ``QLabel`` that displays a
geometric ratio ``numerator / denominator`` and turns red whenever the
ratio is invalid (non-finite, non-positive denominator) or reaches the
contact limit ``>= 1`` (green while ``< 1``), matching the MATLAB
v1.5 alert colour convention.
"""

from __future__ import annotations

import math

from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QLabel, QLineEdit

from src.expressions import CompiledExpression, ExpressionError

__all__ = [
    "BAD_COLOR",
    "GOOD_COLOR",
    "NEUTRAL_COLOR",
    "RatioBadge",
    "attach_formula_validation",
    "make_double_edit",
    "make_int_edit",
    "read_double",
    "read_int",
]

GOOD_COLOR = "#2e7d32"  # green
BAD_COLOR = "#c62828"  # red
NEUTRAL_COLOR = "#546e7a"

_GOOD_SS = f"color: {GOOD_COLOR}; font-weight: bold;"
_BAD_SS = f"color: {BAD_COLOR}; font-weight: bold;"


class RatioBadge(QLabel):
    """Live ``numerator / denominator`` badge (green ``< 1``, red ``>= 1``)."""

    def __init__(self, label: str, tooltip: str = "", parent=None) -> None:
        super().__init__(f"{label} = --", parent)
        self._label = label
        self.ratio: float | None = None
        self.is_ok: bool | None = None
        self.setToolTip(tooltip or label)
        self.setStyleSheet(f"color: {NEUTRAL_COLOR};")

    def set_ratio(self, numerator: float, denominator: float) -> float | None:
        """Update the badge from the current values; returns the ratio."""
        try:
            num = float(numerator)
            den = float(denominator)
        except (TypeError, ValueError):
            num = math.nan
            den = math.nan
        if math.isfinite(num) and math.isfinite(den) and den > 0.0 and num > 0.0:
            self.ratio = num / den
        else:
            self.ratio = None
        self.is_ok = self.ratio is not None and self.ratio < 1.0
        if self.ratio is None:
            self.setText(f"{self._label} = invalid")
            self.setStyleSheet(f"color: {BAD_COLOR}; font-weight: bold;")
        else:
            self.setText(f"{self._label} = {self.ratio:.3f}")
            self.setStyleSheet(_GOOD_SS if self.is_ok else _BAD_SS)
        return self.ratio


def make_double_edit(
    value: float,
    minimum: float = -1e12,
    maximum: float = 1e12,
    decimals: int = 10,
    tooltip: str = "",
    parent=None,
) -> QLineEdit:
    """A numeric line edit with locale-independent double validation."""
    edit = QLineEdit(f"{float(value):.10g}", parent)
    validator = QDoubleValidator(minimum, maximum, decimals, edit)
    validator.setNotation(QDoubleValidator.StandardNotation)
    edit.setValidator(validator)
    if tooltip:
        edit.setToolTip(tooltip)
    return edit


def make_int_edit(
    value: int, minimum: int = 0, maximum: int = 1000, parent=None
) -> QLineEdit:
    """A numeric line edit restricted to integers."""
    edit = QLineEdit(str(int(value)), parent)
    edit.setValidator(QIntValidator(minimum, maximum, edit))
    return edit


def read_double(edit: QLineEdit, default: float = 0.0) -> float:
    """Parse the edit text, falling back to ``default`` on garbage."""
    try:
        value = float(edit.text().replace(",", "."))
    except ValueError:
        return float(default)
    return value if math.isfinite(value) else float(default)


def read_int(edit: QLineEdit, default: int = 1) -> int:
    """Parse the edit text as an integer, falling back to ``default``."""
    try:
        return int(round(float(edit.text())))
    except ValueError:
        return int(default)


def attach_formula_validation(
    edit: QLineEdit, variables: tuple[str, ...] = ("T",)
) -> None:
    """Live green/red styling of a formula line edit.

    The edit background turns light green while the expression compiles
    and evaluates to finite values on a probe grid, and light red with an
    explanatory tooltip otherwise.  Validity is exposed through the
    ``formula_valid`` dynamic property for tests and tab logic.
    """
    probe = [300.0, 1000.0, 1800.0, 2500.0]
    if "z" in variables:
        probe = [0.0, 0.05, 0.1]

    def _validate(text: str) -> None:
        try:
            expr = CompiledExpression(text, variables)
            values = expr.validate(*([probe] * len(variables)), positive=False)
            ok = bool(values.size) and bool((values == values).all())
            if not ok:
                raise ExpressionError("expression produced no valid samples")
            message = "Formula compiles and evaluates."
        except ExpressionError as exc:
            ok = False
            message = f"Invalid formula: {exc}"
        edit.setProperty("formula_valid", ok)
        edit.setStyleSheet(
            "background: #e8f5e9;" if ok else "background: #ffebee;"
        )
        edit.setToolTip(message)

    edit.setProperty("formula_valid", False)
    edit.textChanged.connect(_validate)
    _validate(edit.text())
