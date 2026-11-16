"""Reduced vs unreduced solver parity, events, and resampling (A1).

The reduced solver integrates the jacket as ``(tau, w, uJ, T)`` plus one
decoupled 1-state ODE per nested ring; the unreduced transcription
integrates the joint ``(5 + 2*N_nested)`` system of ``RK_perso7/9/11.m``
(with the 273.15 K offsets and the ``viscosity_3`` ring-3 fix).  Both
must agree to a relative error below 1e-8 at tightened, matched
integration tolerances (both chained at ``Lfour``); at the production
``rtol=1e-9`` the residual is ~1e-8 of pure integration noise at the
hot/cold kink, not a formulation error.
"""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.materials import MaterialSet
from src.models import RingSpec, SimulationInputs, SimulationStatus
from src.physics import (
    DEFAULT_DZ,
    CapillaryModel,
    DenseTrajectory,
    JacketModel,
    rhs_unreduced,
    solve_capillary,
    solve_jacket,
    uniform_grid,
    unreduced_atol,
)


def _nested_inputs(nb_nested: int, ring3_viscosity: str | None = None):
    """Valid nested-ring inputs (rings strictly inside each other)."""
    base = SimulationInputs(
        r0=1.2e-3,
        R0=8.3556e-3 / 2,
        rings=(
            RingSpec(Rc0=0.48e-3 / 2, rc0=0.3936e-3 / 2),
            RingSpec(Rc0=0.24e-3 / 2, rc0=0.20e-3 / 2),
            RingSpec(Rc0=0.12e-3 / 2, rc0=0.10e-3 / 2),
        )[:nb_nested],
        # delta_P >= 12 kPa on the innermost ring; the historical
        # 2 kPa step genuinely collapses ring 3 (bore hits zero).
        Pcap=(6340.0, 9340.0, 21340.0)[:nb_nested],
    )
    base.validate()
    if ring3_viscosity is not None and nb_nested == 3:
        from dataclasses import replace

        rings = list(base.rings)
        rings[2] = RingSpec(
            Rc0=rings[2].Rc0,
            rc0=rings[2].rc0,
            viscosity_expr=ring3_viscosity,
        )
        base = replace(base, rings=tuple(rings))
    return base


@pytest.mark.parametrize("nb_nested", [1, 2, 3])
def test_reduced_vs_unreduced_trajectory_parity(nb_nested):
    """Reduced (4)+N*(1) and unreduced (5+2N) agree to < 1e-8 relative."""
    inputs = _nested_inputs(nb_nested)
    materials = MaterialSet.build(inputs)
    tau0 = 0.86  # near the default tension solution
    z = np.linspace(0.0, inputs.L, 101)

    # ---- reduced: jacket + decoupled capillaries (tight tolerance) ----
    model = JacketModel(inputs, materials)
    jrun = solve_jacket(model, tau0, rtol=1e-11)
    assert jrun.ok
    yj = jrun.traj(z)
    tau_r, w_r, T_r = yj[0], yj[1], yj[3]
    R_r = np.sqrt(yj[2] / w_r)
    r_r = np.sqrt(np.maximum((yj[2] - inputs.delta_J0) / w_r, 0.0))
    caps = []
    for i in range(nb_nested):
        cap = CapillaryModel(
            inputs, materials, i, inputs.capillary_deltas()[i],
            jrun.traj, jrun.z_stop,
        )
        crank = solve_capillary(cap, rtol=1e-11)
        assert crank.ok
        uc = crank.traj(z)[0]
        caps.append(
            (np.sqrt(uc / w_r), np.sqrt(np.maximum(
                (uc - inputs.delta_c0(i)) / w_r, 0.0)))
        )

    # ---- unreduced: joint (5 + 2N) transcription, chained at Lfour ----
    rhs = rhs_unreduced(inputs, materials, inputs.capillary_deltas())
    y0 = [
        tau0,
        inputs.w0,
        inputs.R0**2 * inputs.w0,
        inputs.r0**2 * inputs.w0,
        model.T0,
    ]
    for ring in inputs.rings:
        y0.append(ring.Rc0**2 * inputs.w0)
        y0.append(ring.rc0**2 * inputs.w0)
    atol_u = unreduced_atol(nb_nested)
    sol1 = solve_ivp(
        rhs, (0.0, inputs.Lfour), y0, method="DOP853",
        rtol=1e-12, atol=atol_u, dense_output=True,
    )
    assert sol1.status == 0
    sol2 = solve_ivp(
        rhs, (inputs.Lfour, inputs.L), sol1.y[:, -1], method="DOP853",
        rtol=1e-12, atol=atol_u, dense_output=True,
    )
    assert sol2.status == 0
    z_arr = np.asarray(z, dtype=float)
    yu = np.empty((len(y0), z_arr.size))
    hot = z_arr <= inputs.Lfour
    yu[:, hot] = sol1.sol(z_arr[hot])
    yu[:, ~hot] = sol2.sol(z_arr[~hot])
    w_u = yu[1]
    R_u = np.sqrt(yu[2] / w_u)
    r_u = np.sqrt(np.maximum(yu[3] / w_u, 0.0))

    assert _rel_err(tau_r, yu[0]) < 1e-8
    assert _rel_err(w_r, w_u) < 1e-8
    assert _rel_err(T_r, yu[4]) < 1e-8
    assert _rel_err(R_r, R_u) < 1e-8
    assert _rel_err(r_r, r_u) < 1e-8
    for i in range(nb_nested):
        Rc_u = np.sqrt(yu[5 + 2 * i] / w_u)
        rc_u = np.sqrt(np.maximum(
            yu[6 + 2 * i] / w_u, 0.0))
        assert _rel_err(caps[i][0], Rc_u) < 1e-8, f"ring {i + 1} Rc"
        assert _rel_err(caps[i][1], rc_u) < 1e-8, f"ring {i + 1} rc"


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum relative error, guarding near-zero denominators."""
    a, b = np.asarray(a), np.asarray(b)
    scale = np.maximum(np.abs(b), 1e-30)
    return float(np.max(np.abs(a - b) / scale))


def test_rhs_algebraic_equivalence():
    """Both RHS agree exactly at identical physical states."""
    inputs = _nested_inputs(2)
    materials = MaterialSet.build(inputs)
    model = JacketModel(inputs, materials)
    rhs_u = rhs_unreduced(inputs, materials, inputs.capillary_deltas())
    rng = np.random.default_rng(42)
    for _ in range(25):
        w = inputs.w0 * 10 ** rng.uniform(-1, 2)
        R = inputs.R0 * 10 ** rng.uniform(-1.2, 0.3)
        T = rng.uniform(300.0, 2100.0)
        tau = rng.uniform(0.05, 5.0)
        z = rng.uniform(0.0, inputs.L)
        uJ = R**2 * w
        # reduced state
        y_red = [tau, w, uJ, T]
        d_red = model.rhs(z, y_red)
        # unreduced state: r2w = uJ - dJ0, capillary radii scaled by w
        Rc = inputs.rings[0].Rc0 * 0.9
        rc = inputs.rings[0].rc0 * 0.9
        Rc2 = inputs.rings[1].Rc0 * 0.9
        rc2 = inputs.rings[1].rc0 * 0.9
        y_unr = [
            tau, w, uJ, (uJ - inputs.delta_J0), T,
            Rc**2 * w, rc**2 * w, Rc2**2 * w, rc2**2 * w,
        ]
        d_unr = rhs_u(z, y_unr)
        for k in range(3):  # dtaudz, dwdz, duJdz == d(r2w)/dz
            assert d_red[k] == pytest.approx(d_unr[k], rel=1e-9, abs=0.0)
        assert d_red[3] == pytest.approx(d_unr[4], rel=1e-9, abs=0.0)


def test_ring3_uses_its_own_viscosity():
    """Ring 3 obeys viscosity_3 (the RK_perso11.m bug fix)."""
    base_expr = "10^(-7.946)*exp(66337/T)"
    distinct = "2*10^(-7.946)*exp(66337/T)"
    inputs_a = _nested_inputs(3, ring3_viscosity=base_expr)
    inputs_b = _nested_inputs(3, ring3_viscosity=distinct)
    ta0 = 0.86
    for inputs in (inputs_a, inputs_b):
        inputs.validate()
    mats_a = MaterialSet.build(inputs_a)
    mats_b = MaterialSet.build(inputs_b)
    model_a = JacketModel(inputs_a, mats_a)
    jrun = solve_jacket(model_a, ta0)
    assert jrun.ok
    Rc_a = _ring3_radius(inputs_a, mats_a, jrun, inputs_a.capillary_deltas())
    Rc_b = _ring3_radius(inputs_b, mats_b, jrun, inputs_b.capillary_deltas())
    # doubling ring-3 viscosity must change its trajectory
    assert abs(Rc_a - Rc_b) / Rc_a > 1e-6
    # ... and the unreduced transcription must track the ring-3 law too
    rhs = rhs_unreduced(inputs_b, mats_b, inputs_b.capillary_deltas())
    y0 = [ta0, inputs_b.w0, inputs_b.R0**2 * inputs_b.w0,
          inputs_b.r0**2 * inputs_b.w0, model_a.T0]
    for ring in inputs_b.rings:
        y0.append(ring.Rc0**2 * inputs_b.w0)
        y0.append(ring.rc0**2 * inputs_b.w0)
    atol_u = unreduced_atol(3)
    sol1 = solve_ivp(
        rhs, (0.0, inputs_b.Lfour), y0, method="DOP853",
        rtol=1e-12, atol=atol_u,
    )
    assert sol1.status == 0
    sol2 = solve_ivp(
        rhs, (inputs_b.Lfour, inputs_b.L), sol1.y[:, -1], method="DOP853",
        rtol=1e-12, atol=atol_u,
    )
    assert sol2.status == 0
    wL = sol2.y[1, -1]
    Rc_unr = math.sqrt(sol2.y[9, -1] / wL)
    assert Rc_unr == pytest.approx(Rc_b, rel=1e-8)


def _ring3_radius(inputs, materials, jrun, dP):
    cap = CapillaryModel(
        inputs, materials, 2, dP[2], jrun.traj, jrun.z_stop
    )
    crank = solve_capillary(cap, rtol=1e-11)
    assert crank.ok
    wL = float(jrun.y_end[1])
    return math.sqrt(crank.uc_end / wL)


# --------------------------------------------------------------------------
# Events, guards, resampling
# --------------------------------------------------------------------------


def test_jacket_collapse_event_on_strong_suction():
    inputs = _nested_inputs(1).with_process(Pcore=-2e4)
    materials = MaterialSet.build(inputs)
    run = solve_jacket(JacketModel(inputs, materials), 0.86)
    assert not run.ok
    assert run.status in (
        SimulationStatus.JACKET_COLLAPSE,
        SimulationStatus.TENSION_LOSS,
        SimulationStatus.DIVERGED,
    )
    assert 0.0 < run.z_stop <= inputs.L


def test_jacket_blowup_event_on_overpressure():
    inputs = _nested_inputs(1).with_process(Pcore=5e5)
    materials = MaterialSet.build(inputs)
    run = solve_jacket(JacketModel(inputs, materials), 0.86)
    assert not run.ok
    assert run.status in (
        SimulationStatus.JACKET_BLOWUP,
        SimulationStatus.VELOCITY_BLOWUP,
        SimulationStatus.DIVERGED,
    )


def test_tension_loss_event():
    inputs = _nested_inputs(1)
    materials = MaterialSet.build(inputs)
    run = solve_jacket(JacketModel(inputs, materials), 1e-8)
    assert not run.ok
    assert run.status == SimulationStatus.TENSION_LOSS
    assert run.z_stop < inputs.L


def test_uniform_grid_spacing():
    z = uniform_grid(0.1, 1e-5)
    assert z[0] == 0.0 and z[-1] == pytest.approx(0.1)
    assert np.all(np.diff(z) <= 1e-5 + 1e-12)
    assert len(z) >= 2
    z2 = uniform_grid(3.3e-5, 1e-5)
    assert z2[-1] == pytest.approx(3.3e-5)  # endpoint is exactly z_stop
    assert np.all(np.diff(z2) <= 1e-5 + 1e-12)
    assert uniform_grid(0.0).size == 1


def test_default_dz_is_ten_microns():
    assert DEFAULT_DZ == 1e-5


def test_dense_trajectory_piecewise_continuity():
    inputs = _nested_inputs(1)
    materials = MaterialSet.build(inputs)
    run = solve_jacket(JacketModel(inputs, materials), 0.86)
    traj = run.traj
    assert isinstance(traj, DenseTrajectory)
    assert traj.z_end == pytest.approx(inputs.L)
    # hot/cold boundary is continuous across the piecewise split
    zf = inputs.Lfour
    left = traj.state(zf - 1e-9)
    right = traj.state(zf + 1e-9)
    np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-12)
    # scalar and vector calls agree
    grid = np.linspace(0.0, inputs.L, 7)
    vec = traj(grid)
    for i, zi in enumerate(grid):
        np.testing.assert_allclose(
            traj.state(zi), vec[:, i], rtol=1e-14, atol=0.0
        )
    # clipping beyond z_end returns the last state
    assert traj.state(traj.z_end + 1.0)[0] == pytest.approx(
        traj.state(traj.z_end)[0]
    )


def test_capillary_collapse_and_blowup_events():
    inputs = _nested_inputs(2)
    materials = MaterialSet.build(inputs)
    model = JacketModel(inputs, materials)
    jrun = solve_jacket(model, 0.86)
    assert jrun.ok
    # strong suction collapses the bore of ring 1
    cap = CapillaryModel(
        inputs, materials, 0, -2e4, jrun.traj, jrun.z_stop
    )
    crank = solve_capillary(cap)
    assert not crank.ok
    assert crank.status == SimulationStatus.CAPILLARY_COLLAPSE
    assert 0.0 < crank.z_stop <= jrun.z_stop
    # heavy overpressure blows the ring up
    cap = CapillaryModel(
        inputs, materials, 0, 5e5, jrun.traj, jrun.z_stop
    )
    crank = solve_capillary(cap)
    assert not crank.ok
    assert crank.status == SimulationStatus.CAPILLARY_BLOWUP
