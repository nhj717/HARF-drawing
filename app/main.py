"""Application entry point for the HARF drawing workstation.

Run either as ``uv run harf-drawing`` (console script) or via
``python -m app.main``.  ``--help`` prints the usage and exits before
any Qt event loop starts.
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow

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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    app = QApplication.instance() or QApplication(
        ["harf-drawing"] if argv is not None else sys.argv
    )
    window = MainWindow(config_path=args.config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
