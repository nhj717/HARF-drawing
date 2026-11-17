"""Frameless ``QScrollArea`` wrapper for tall form columns.

Keeps the workstation usable on 1280x800 laptop displays: the tall
left-side control columns scroll vertically instead of inflating the
``MainWindow`` minimum size hint.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QWidget

__all__ = ["wrap_scroll"]


def wrap_scroll(widget: QWidget) -> QScrollArea:
    """Wrap ``widget`` in a frameless, resizable vertical scroll area."""
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return area
