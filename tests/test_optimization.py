"""Tension solve, optimization recovery, and infeasibility (A3)."""

import math

import pytest

from src.materials import MaterialSet
from src.models import (
    RingSpec,
    SimulationStatus,
    default_optimization_inputs,
    default_simulation_inputs,
)
from src.solvers import (
    run_optimization,
    solve_capillary_delta_p,
    solve_initial_tension,
)


@pytest.fixture(scope="module")
def default_1ring():
    return default_simulation_inputs(nb_nested=1)


@pytest.fixture(scope="module")
def default_2ring_nested():
    from dataclasses import replace

    inp = default_simulation_inputs(nb_nested=2)
    rings = list(inp.rings)
    rings[1] = RingSpec(Rc0=0.24e-3 / 2, rc0=0.20e-3 / 2)
    inp = replace(inp, rings=tuple(rings))
    inp.validate()
    return inp


# --------------------------------------------------------------------------
# Initial tension
# --------------------------------------------------------------------------


def test_solve_initial_tension_accuracy(default_1ring):
    """|tau(L) - tau_target| / tau_target < 1e-10 on the default draw."""
    sol = solve_initial_tension(default_1ring)
    assert sol.converged
    assert sol.run is not None and sol.run.ok
    assert sol.residual < 1e-10
    tau_L = float(sol.run.y_end[0])
    assert abs(tau_L - default_1ring.tension) / default_1ring.tension < 1e-10
    assert sol.tau0 == pytest.approx(0.858952459, abs=1e-6)


def test_solve_initial_tension_respects_changed_process(default_1ring):
    """A hotter furnace needs less initial tension for the same draw."""
    hot = default_1ring.with_process(Tmax_C=1860.0)
    sol_hot = solve_initial_tension(hot)
    sol_ref = solve_initial_tension(default_1ring)
    assert sol_hot.converged and sol_ref.converged
    assert sol_hot.tau0 < sol_ref.tau0
    assert sol_hot.residual < 1e-10


# --------------------------------------------------------------------------
# Optimization recovery
# --------------------------------------------------------------------------


def test_optimization_recovers_targets_1ring(default_1ring):
    """(Rf, rf, wf, Rcf, rcf) recovered within 1e-4 relative error."""
    opt = default_optimization_inputs(nb_nested=1)
    res = run_optimization(default_1ring, opt)
    assert res.ok, res.message
    tgt = res.targets
    assert tgt is not None
    sim = res.result
    assert sim is not None and sim.ok

    assert abs(sim.w[-1] / tgt.wf - 1.0) < 1e-4
    assert abs(sim.R[-1] / tgt.Rf - 1.0) < 1e-4
    assert abs(sim.r[-1] / tgt.rf - 1.0) < 1e-4
    assert abs(sim.Rc[0][-1] / tgt.Rcf_rings[0] - 1.0) < 1e-4
    assert abs(sim.rc[0][-1] / tgt.rcf_rings[0] - 1.0) < 1e-4
    # ratio target R/r as well
    assert abs(
        (sim.R[-1] / sim.r[-1]) / (tgt.Rf / tgt.rf) - 1.0
    ) < 1e-4
    # optimum respects the search windows
    assert opt.Tmax_min_C <= res.Tmax_C <= opt.Tmax_max_C
    assert opt.Pcore_min <= res.Pcore <= opt.Pcore_max


def test_optimization_recovers_targets_2ring(default_2ring_nested):
    """Nested ring targets are recovered too (within 1e-4)."""
    opt = default_optimization_inputs(nb_nested=2)
    res = run_optimization(default_2ring_nested, opt)
    assert res.ok, res.message
    tgt = res.targets
    sim = res.result
    assert sim is not None and sim.ok
    for i in range(2):
        assert abs(sim.Rc[i][-1] / tgt.Rcf_rings[i] - 1.0) < 1e-4
        assert abs(sim.rc[i][-1] / tgt.rcf_rings[i] - 1.0) < 1e-4
    assert len(res.DeltaP) == 2


def test_optimization_deltap_reproducible(default_1ring):
    """|DeltaP_i - DeltaP_i*| < 1e-5 Pa against an independent re-solve."""
    opt = default_optimization_inputs(nb_nested=1)
    res = run_optimization(default_1ring, opt)
    assert res.ok and res.DeltaP
    tgt = res.targets
    assert tgt is not None

    # independent re-solve: tension -> jacket -> capillary root-find
    final = default_1ring.with_process(
        Tmax_C=res.Tmax_C, Pcore=res.Pcore
    )
    materials = MaterialSet.build(final)
    tension = solve_initial_tension(final, materials)
    assert tension.converged and tension.run is not None
    sol = solve_capillary_delta_p(
        tension.run,
        final,
        materials,
        0,
        tgt.Rcf_rings[0],
        dP_range=(
            opt.Pcap_min[0] - res.Pcore,
            opt.Pcap_max[0] - res.Pcore,
        ),
    )
    assert sol.converged
    assert abs(sol.delta_p - res.DeltaP[0]) < 1e-5

    # recovered absolute capillary pressure chains correctly
    assert res.Pcap[0] == pytest.approx(
        res.Pcore + res.DeltaP[0], abs=1e-12
    )


# --------------------------------------------------------------------------
# Infeasibility & structure
# --------------------------------------------------------------------------


def test_infeasible_identical_nested_targets():
    """Overlapping ring targets are INFEASIBLE; a bad preform INPUT_ERROR."""
    from dataclasses import replace

    # feasible preform (strictly nested by default), but a thick target
    # wall makes the derived ring-2 targets overlap ring 1's
    inp = default_simulation_inputs(nb_nested=2)
    inp.validate()
    opt = replace(default_optimization_inputs(nb_nested=2), tf=32e-6)
    res = run_optimization(inp, opt)
    assert res.status == SimulationStatus.INFEASIBLE
    assert res.targets is None
    assert res.message
    assert res.result is None

    # identical (non-nested) placeholder rings are an input error instead
    bad = replace(inp, rings=(inp.rings[0], inp.rings[0]))
    res_bad = run_optimization(bad, opt)
    assert res_bad.status == SimulationStatus.INPUT_ERROR


def test_infeasible_when_pcap_window_excludes_root(default_1ring):
    """A capillary window that cannot reach Rcf returns INFEASIBLE."""
    opt = default_optimization_inputs(nb_nested=1)
    from dataclasses import replace

    opt = replace(opt, Pcap_min=(2000.0,), Pcap_max=(2001.0,))
    res = run_optimization(default_1ring, opt)
    assert res.status == SimulationStatus.INFEASIBLE
    assert res.message


def test_optimization_reports_progress(default_1ring):
    """Progress notifications advance monotonically to 1.0."""
    events: list[tuple[str, float]] = []

    def progress(message, frac):
        events.append((message, frac))

    opt = default_optimization_inputs(nb_nested=1)
    res = run_optimization(default_1ring, opt, progress=progress)
    assert res.ok
    assert events
    fracs = [f for _, f in events]
    assert fracs[0] < fracs[-1]
    assert fracs[-1] == 1.0


def test_optimization_elapsed_time_recorded(default_1ring):
    opt = default_optimization_inputs(nb_nested=1)
    res = run_optimization(default_1ring, opt)
    assert res.ok
    assert res.elapsed_s > 0.0
    assert math.isfinite(res.elapsed_s)
