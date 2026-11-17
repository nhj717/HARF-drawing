"""Background computation workers (``QThread`` + cooperative cancel).

Both workers wrap the *never-raising* core solvers
(:func:`src.solvers.run_direct_simulation` and
:func:`src.solvers.run_optimization`) so the Qt event loop can never be
crashed by a computation.  Cancellation is cooperative through a
``threading.Event`` checked by the solvers between ODE evaluations.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal

from src.models import (
    OptimizationInputs,
    SimulationInputs,
)
from src.solvers import DEFAULT_DZ, run_direct_simulation, run_optimization

__all__ = ["OptimizationWorker", "SimulationWorker"]


class SimulationWorker(QThread):
    """Runs one direct simulation off the GUI thread."""

    progress = Signal(str, float)
    status_changed = Signal(str)
    result_ready = Signal(object)  # SimulationResult
    error_occurred = Signal(str)

    def __init__(
        self,
        inputs: SimulationInputs,
        dz: float = DEFAULT_DZ,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._inputs = inputs
        self._dz = dz
        self._cancel_event = threading.Event()

    # -- control -----------------------------------------------------------
    def cancel(self) -> None:
        """Request cooperative cancellation (honored within one ODE eval)."""
        self._cancel_event.set()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    # -- thread body ---------------------------------------------------------
    def run(self) -> None:  # noqa: N802 (Qt override)
        self.status_changed.emit("Running direct computation...")
        try:
            result = run_direct_simulation(
                self._inputs,
                dz=self._dz,
                cancel_event=self._cancel_event,
                progress=lambda msg, frac: self.progress.emit(msg, frac),
            )
        except Exception as exc:  # noqa: BLE001 - GUI must never crash
            self.error_occurred.emit(
                f"Unexpected simulation error: {type(exc).__name__}: {exc}"
            )
            return
        self.status_changed.emit(_status_text(result.status.value))
        self.result_ready.emit(result)


class OptimizationWorker(QThread):
    """Runs the (Tmax, Pcore, DeltaP_i) optimization off the GUI thread."""

    progress = Signal(str, float)
    status_changed = Signal(str)
    result_ready = Signal(object)  # OptimizationResult
    error_occurred = Signal(str)

    def __init__(
        self,
        inputs: SimulationInputs,
        opt: OptimizationInputs,
        dz: float = DEFAULT_DZ,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._inputs = inputs
        self._opt = opt
        self._dz = dz
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    def run(self) -> None:  # noqa: N802 (Qt override)
        self.status_changed.emit("Running optimization...")
        try:
            result = run_optimization(
                self._inputs,
                self._opt,
                dz=self._dz,
                cancel_event=self._cancel_event,
                progress=lambda msg, frac: self.progress.emit(msg, frac),
            )
        except Exception as exc:  # noqa: BLE001 - GUI must never crash
            self.error_occurred.emit(
                f"Unexpected optimization error: {type(exc).__name__}: {exc}"
            )
            return
        self.status_changed.emit(_status_text(result.status.value))
        self.result_ready.emit(result)


def _status_text(value: str) -> str:
    return f"Finished ({value})." if value != "cancelled" else "Cancelled."
