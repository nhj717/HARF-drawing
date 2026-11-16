"""Crash-proof safety net: extreme regimes never raise or emit NaN (A4)."""

import threading

import numpy as np
import pytest

from src.materials import MaterialSet
from src.models import (
    SimulationInputs,
    SimulationStatus,
    default_simulation_inputs,
)
from src.physics import ComputationCancelled, JacketModel, solve_jacket
from src.solvers import run_direct_simulation, solve_initial_tension


def _inputs(**kw) -> SimulationInputs:
    return default_simulation_inputs(nb_nested=1).with_process(**kw)


def _assert_structured(sim) -> None:
    """Every result is finite, consistently shaped, and diagnosable."""
    assert sim.finite(), "NaN/Inf leaked into the result arrays"
    assert isinstance(sim.status, SimulationStatus)
    assert sim.z[0] == 0.0
    assert sim.z_end <= sim.inputs.L + 1e-12
    for name in ("tau", "w", "T", "Ta", "R", "r"):
        arr = getattr(sim, name)
        assert arr.shape == sim.z.shape, name
    if not sim.ok:
        assert sim.diagnostics is not None
        assert sim.diagnostics.message
        assert sim.diagnostics.advice


@pytest.mark.parametrize(
    "kw",
    [
        {"Pcore": -1e5},        # extreme collapse (suction)
        {"Pcore": -1e7},        # deep suction
        {"Pcore": 1e6},         # strong overpressure
        {"Pcore": 1e8},          # extreme overpressure
        {"Tmax_C": -200.0},     # cryogenic furnace
        {"Tmax_C": 5000.0},     # overheated furnace
        {"tension": 1e-9},       # vanishing draw tension
        {"tension": 1e5},        # enormous draw tension
    ],
    ids=[
        "suction", "deep-suction", "overpressure", "extreme-pressure",
        "cryo", "overheat", "no-tension", "huge-tension",
    ],
)
def test_extreme_regimes_never_crash(kw):
    sim = run_direct_simulation(_inputs(**kw))
    _assert_structured(sim)
    assert sim.status != SimulationStatus.SUCCESS


def test_ultracold_furnace_reports_no_draw():
    """A furnace too cold to soften the material yields NO_DRAW (A4)."""
    sim = run_direct_simulation(_inputs(Tmax_C=600.0))
    assert sim.status == SimulationStatus.NO_DRAW
    _assert_structured(sim)
    # the ODEs integrated cleanly: the full [0, L] trajectory is kept
    assert sim.z_end == pytest.approx(sim.inputs.L)
    assert sim.w[-1] == pytest.approx(sim.inputs.w0, rel=1e-6)


def test_cryogenic_initial_temperature_is_structured():
    sim = run_direct_simulation(_inputs(Tmax_C=600.0, dT0=2500.0))
    _assert_structured(sim)
    assert sim.status != SimulationStatus.SUCCESS


def test_unattainable_tension_reports_breakdown_family():
    """A target tension beyond the velocity limit is a structured failure."""
    sim = run_direct_simulation(_inputs(tension=1e5))
    assert not sim.ok
    assert sim.status in (
        SimulationStatus.VELOCITY_BLOWUP,
        SimulationStatus.TENSION_LOSS,
        SimulationStatus.DIVERGED,
    )
    _assert_structured(sim)


def test_suction_reports_collapse_family():
    sim = run_direct_simulation(_inputs(Pcore=-1e5))
    assert sim.status in (
        SimulationStatus.JACKET_COLLAPSE,
        SimulationStatus.TENSION_LOSS,
        SimulationStatus.DIVERGED,
        SimulationStatus.INFEASIBLE,
    )


def test_partial_arrays_stop_at_failure():
    sim = run_direct_simulation(_inputs(Pcore=1e6))
    assert not sim.ok
    assert 0.0 < sim.z_end < sim.inputs.L
    _assert_structured(sim)


def test_invalid_geometry_is_input_error():
    bad = SimulationInputs(
        r0=5.0e-3,  # r0 > R0 = 4.1778e-3
        R0=8.3556e-3 / 2,
        rings=default_simulation_inputs().rings,
        Pcap=(6340.0,),
    )
    sim = run_direct_simulation(bad)
    assert sim.status == SimulationStatus.INPUT_ERROR
    assert sim.diagnostics is not None
    assert "radii" in sim.diagnostics.message


def test_nonpositive_tension_is_input_error():
    sim = run_direct_simulation(_inputs(tension=-1.0))
    assert sim.status == SimulationStatus.INPUT_ERROR


def test_bad_expression_is_input_error():
    from dataclasses import replace

    inp = default_simulation_inputs(nb_nested=1)
    inp2 = replace(inp, viscosity_expr="T - 500")  # non-positive law
    sim = run_direct_simulation(inp2)
    assert sim.status == SimulationStatus.INPUT_ERROR


def test_cancellation_is_cooperative():
    cancel = threading.Event()
    cancel.set()
    sim = run_direct_simulation(
        default_simulation_inputs(nb_nested=1), cancel_event=cancel
    )
    assert sim.status == SimulationStatus.CANCELLED


def test_cancellation_mid_solve_raises_once():
    inp = default_simulation_inputs(nb_nested=1)
    mats = MaterialSet.build(inp)
    model = JacketModel(inp, mats)
    orig_rhs = model.rhs
    cancel = threading.Event()

    def rhs_spy(z, y):
        cancel.set()
        return orig_rhs(z, y)

    model.rhs = rhs_spy  # type: ignore[method-assign]
    with pytest.raises(ComputationCancelled):
        solve_jacket(model, 0.86, cancel_event=cancel)


def test_raising_progress_callback_is_swallowed():
    def bad_progress(message, frac):
        raise RuntimeError("UI bug")

    sim = run_direct_simulation(
        default_simulation_inputs(nb_nested=1), progress=bad_progress
    )
    assert sim.ok
    _assert_structured(sim)


def test_tension_solver_failure_is_structured():
    """An undrawable tension target returns a structured solution."""
    inp = _inputs(tension=1e5)
    sol = solve_initial_tension(inp)
    assert isinstance(sol.tau0, float)
    assert np.isfinite(sol.tau0)
    assert isinstance(sol.converged, bool)
    assert sol.n_evals > 0
