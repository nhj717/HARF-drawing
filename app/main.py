"""Application entry point for the HARF drawing workstation.

Run either as ``uv run harf-drawing`` (console script) or via
``python -m app.main``.  ``--help`` prints the usage and exits before
any Qt event loop starts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QLocale
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from src.io_utils import default_user_config_path

__all__ = ["main"]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="harf-drawing",
        description=(
            "HARF (Heat-Assisted Robust Fiber) drawing simulation "
            "workstation: MATLAB v1.5 port with a PySide6 GUI."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to a .json configuration file to load at startup "
        "(also used for auto-save on exit)",
    )
    return parser.parse_args(argv)


def _resolve_config_path(args: argparse.Namespace) -> Path:
    """Config used at startup: ``--config`` or the last-session default."""
    return Path(args.config) if args.config else default_user_config_path()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Force the C locale ('.' decimal separator) regardless of the OS
    # locale, so every numeric field formats/accepts '.' decimals.
    QLocale.setDefault(QLocale.c())
    app = QApplication.instance() or QApplication(
        ["harf-drawing"] if argv is not None else sys.argv
    )
    window = MainWindow(config_path=_resolve_config_path(args))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
