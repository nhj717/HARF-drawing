"""Unit tests for the cane & stack scaling engine (``src.scaling``).

Covers the legacy Excel anchor values, structural error handling, the
auto-fit / inverse / stack round trips, area-conservation invariants,
the solver-tab bridge and JSON persistence of the scaling UI state.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.io_utils import read_ui_state, save_config_json
from src.models import (
    RingSpec,
    SimulationInputs,
    compute_derived_targets,
    default_optimization_inputs,
    default_simulation_inputs,
)
from src.scaling import (
    CO_FED_VACUUM,
    CaneGeometry,
    CouplingMode,
    FiberGeometry,
    InverseSubMode,
    JacketGeometry,
    ScalingError,
    ScalingFactors,
    auto_fit_scaling_factors,
    cane_to_stack,
    compute_forward_scaling,
    get_scaling_presets,
    inverse_jacket_od,
    scaling_to_solver_inputs,
    solve_inverse_cane,
    solver_inputs_to_scaling,
    stack_to_cane,
)


def _assert_cane_close(got: CaneGeometry, want: CaneGeometry, rtol: float):
    assert got.OD_c == pytest.approx(want.OD_c, rel=rtol)
    assert got.D_core_c == pytest.approx(want.D_core_c, rel=rtol)
    for g, w in zip(got.d_c, want.d_c, strict=True):
        assert g == pytest.approx(w, rel=rtol)
    for g, w in zip(got.t_c, want.t_c, strict=True):
        assert g == pytest.approx(w, rel=rtol)


def _assert_fiber_close(got: FiberGeometry, want: FiberGeometry, rtol: float):
    assert got.D_f == pytest.approx(want.D_f, rel=rtol)
    assert got.D_core_f == pytest.approx(want.D_core_f, rel=rtol)
    for g, w in zip(got.d_f, want.d_f, strict=True):
        assert g == pytest.approx(w, rel=rtol)
    for g, w in zip(got.t_f, want.t_f, strict=True):
        assert g == pytest.approx(w, rel=rtol)


def test_legacy_excel_anchor():
    preset = get_scaling_presets()[3]
    res = compute_forward_scaling(preset.cane, preset.jacket, preset.factors)
    um = 1e-6
    assert res.S_c == pytest.approx(48.604932, rel=1e-6)
    assert res.stages["E"].zeta == pytest.approx(2471.945 * um, rel=1e-6)
    assert res.stages["E"].delta == pytest.approx(418.4725 * um, rel=1e-6)
    assert res.stages["G"].zeta == pytest.approx(92.561387 * um, rel=1e-6)
    assert res.stages["H"].OD_j == pytest.approx(175.445532 * um, rel=1e-6)
    assert res.stages["H"].d[0] == pytest.approx(25.927821 * um, rel=1e-6)
    assert res.stages["H"].Delta[0] == pytest.approx(26.458401 * um, rel=1e-6)
    # t_H = (Delta_H - d_H) / 2 evaluated at full precision (the Excel
    # sheet's 0.265290 shows intermediate-rounding of Delta_H / d_H).
    assert res.stages["H"].t[0] == pytest.approx(0.2652892 * um, rel=1e-6)
    assert res.stages["H"].D_core == pytest.approx(39.644585 * um, rel=1e-6)
    # The legacy cane wall is negative by construction: no raise, just
    # a red feasibility flag (NaN-tolerant forward pass).
    assert res.feasibility.cane_wall_positive is False
    assert not res.feasibility.all_ok


def test_forward_structural_errors_raise():
    preset = get_scaling_presets()[0]
    bad = ScalingFactors(
        S_j=40.0, S_ie=1.6, S_ce=(2.5, 2.0), coupling=CO_FED_VACUUM
    )
    with pytest.raises(ScalingError):
        compute_forward_scaling(preset.cane, preset.jacket, bad)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("coupling", [CO_FED_VACUUM, CouplingMode.LEGACY_EXCEL])
def test_auto_fit_and_inverse_round_trip(index, coupling):
    preset = get_scaling_presets()[index]
    cane0 = preset.cane
    jacket0 = preset.jacket
    if coupling is CO_FED_VACUUM:
        factors0 = preset.factors
    else:
        factors0 = ScalingFactors(
            S_j=42.0,
            S_ie=preset.factors.S_ie,
            S_ce=preset.factors.S_ce,
            coupling=coupling,
        )
    base = compute_forward_scaling(cane0, jacket0, factors0)
    assert base.feasibility.all_ok
    target = base.achieved_fiber

    fit = auto_fit_scaling_factors(cane0, target, jacket0, coupling)
    assert fit.S_j == pytest.approx(factors0.S_j, rel=1e-9)
    assert fit.S_ie == pytest.approx(factors0.S_ie, rel=1e-9)
    for got, want in zip(fit.S_ce, factors0.S_ce, strict=True):
        assert got == pytest.approx(want, rel=1e-9)

    # FIXED_JACKET inverse with the true OD_c recovers the cane.
    cane1, S_c1 = solve_inverse_cane(
        target, jacket0, fit.S_ie, fit.S_ce,
        InverseSubMode.FIXED_JACKET, coupling,
        OD_c=cane0.OD_c, n_cap=cane0.n_cap,
    )
    _assert_cane_close(cane1, cane0, rtol=1e-9)
    assert S_c1 == pytest.approx(base.S_c, rel=1e-9)
    res = compute_forward_scaling(cane1, jacket0, fit, desired_fiber=target)
    _assert_fiber_close(res.achieved_fiber, target, rtol=1e-9)
    assert res.feasibility.all_ok

    # MANUAL_SJ inverse with t_sleeve + S_j also closes the loop.
    t_sleeve = (cane0.OD_c - cane0.zeta_c) / 2.0
    odj2 = inverse_jacket_od(
        target, jacket0.ID_j, cane0.OD_c, factors0.S_j, fit.S_ie, coupling
    )
    assert odj2 == pytest.approx(jacket0.OD_j, rel=1e-9)
    jacket2 = JacketGeometry(OD_j=odj2, ID_j=jacket0.ID_j)
    cane2, S_c2 = solve_inverse_cane(
        target, jacket2, fit.S_ie, fit.S_ce,
        InverseSubMode.MANUAL_SJ, coupling,
        t_sleeve=t_sleeve, S_j=factors0.S_j, n_cap=cane0.n_cap,
    )
    _assert_cane_close(cane2, cane0, rtol=1e-9)
    assert S_c2 == pytest.approx(base.S_c, rel=1e-9)
    res2 = compute_forward_scaling(cane2, jacket2, fit, desired_fiber=target)
    _assert_fiber_close(res2.achieved_fiber, target, rtol=1e-9)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_stack_round_trip(index):
    cane0 = get_scaling_presets()[index].cane
    sie_st = 1.7
    scest = tuple(2.0 + 0.1 * i for i in range(len(cane0.d_c)))
    stack = cane_to_stack(cane0, sie_st, scest, S_stack=2.5)
    assert stack.ID_stack == pytest.approx(
        2.5 * cane0.zeta_c / sie_st, rel=1e-12
    )
    cane1 = stack_to_cane(stack)
    _assert_cane_close(cane1, cane0, rtol=1e-10)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_area_conservation(index):
    preset = get_scaling_presets()[index]
    res = compute_forward_scaling(preset.cane, preset.jacket, preset.factors)
    assert res.feasibility.all_ok
    rhs = (preset.jacket.OD_j**2 - preset.jacket.ID_j**2) + (
        preset.cane.OD_c**2 - res.stages["E"].zeta**2
    )
    for name in ("F", "G", "H"):
        stage = res.stages[name]
        assert preset.factors.S_j**2 * (
            stage.OD_j**2 - stage.zeta**2
        ) == pytest.approx(rhs, rel=1e-12)
    stage_e = res.stages["E"]
    stage_h = res.stages["H"]
    for i in range(len(preset.cane.d_c)):
        assert res.S_c**2 * (
            stage_h.Delta[i]**2 - stage_h.d[i]**2
        ) == pytest.approx(
            stage_e.Delta[i]**2 - stage_e.d[i]**2, rel=1e-12
        )
    # The glass-area fraction gamma is invariant across E -> F -> G -> H
    # (both numerator and denominator of the stage ratio rescale by S_c^2).
    gammas = [res.stages[s].gamma_pct for s in ("E", "F", "G", "H")]
    for g in gammas[1:]:
        assert np.isclose(g, gammas[0], rtol=1e-12)
    # Pushing the achieved fiber through the solver bridge reproduces
    # the stage-H jacket OD: the solver's derived 2*Rf == OD_j,H.
    handoff = scaling_to_solver_inputs(
        preset.cane,
        preset.jacket,
        res.achieved_fiber,
        preset.factors.coupling,
    )
    inputs = SimulationInputs(
        r0=handoff.d0 / 2.0,
        R0=handoff.D0 / 2.0,
        rings=tuple(
            RingSpec(Rc0=D / 2.0, rc0=d / 2.0)
            for D, d in zip(handoff.D_c0, handoff.d_c0, strict=True)
        ),
        n_capillaries=handoff.n_cap,
    )
    opt = replace(
        default_optimization_inputs(len(handoff.D_c0)),
        tf=handoff.tf,
        Dcf1=handoff.Dcf1,
        Dcore=handoff.Dcore,
    )
    targets = compute_derived_targets(inputs, opt)
    assert np.isclose(2.0 * targets.Rf, stage_h.OD_j, rtol=1e-9)


def test_solver_bridge_pull_push():
    inputs = default_simulation_inputs(1)
    opt = default_optimization_inputs(1)
    state = solver_inputs_to_scaling(inputs, opt, CO_FED_VACUUM)
    assert state.fiber is not None
    assert state.factors is not None
    targets = compute_derived_targets(inputs, opt)
    # The pulled fiber target is exactly the solver's derived target.
    assert state.fiber.D_f == pytest.approx(2.0 * targets.Rf, rel=1e-9)
    # Forward scaling from the pulled state achieves that fiber.
    res = compute_forward_scaling(
        state.cane, state.jacket, state.factors
    )
    assert res.achieved_fiber.D_f == pytest.approx(
        2.0 * targets.Rf, rel=1e-9
    )
    # The handoff maps back onto the solver-tab inputs.
    handoff = scaling_to_solver_inputs(
        state.cane, state.jacket, state.fiber, CO_FED_VACUUM
    )
    assert handoff.D0 == pytest.approx(2.0 * inputs.R0, rel=1e-9)
    assert handoff.d0 == pytest.approx(2.0 * inputs.r0, rel=1e-9)
    ring = inputs.rings[0]
    assert handoff.D_c0[0] == pytest.approx(2.0 * ring.Rc0, rel=1e-9)
    assert handoff.d_c0[0] == pytest.approx(2.0 * ring.rc0, rel=1e-9)
    assert handoff.tf == pytest.approx(opt.tf, rel=1e-9)
    assert handoff.Dcf1 == pytest.approx(2.0 * targets.Rcf, rel=1e-9)
    assert handoff.Dcore == pytest.approx(opt.Dcore, rel=1e-9)
    assert handoff.n_cap == inputs.n_capillaries


def test_config_json_scaling_ui_state(tmp_path):
    inputs = default_simulation_inputs(1)
    opt = default_optimization_inputs(1)
    path = tmp_path / "cfg.json"
    save_config_json(inputs, opt, path)
    assert read_ui_state(path) == {}
    state = {"scaling": {"unit": "µm", "sie": "1.75"}}
    save_config_json(inputs, opt, path, ui_state=state)
    assert read_ui_state(path)["scaling"] == {"unit": "µm", "sie": "1.75"}
