"""Package-level invariants and MATLAB v1.5 default parity.

Covers: repository toolchain sanity, exact v1.5 defaults (radii vs
diameters!), geometry invariants, derived-target consistency, and the
validated default jacket trajectory (cross-checked against an
independent DOP853 prototype of ``RK_perso.m``).
"""

import math
from dataclasses import replace

import numpy as np
import pytest

import src
from src.materials import MaterialSet
from src.models import (
    DEFAULT_D0_MM,
    DEFAULT_DC0_MM,
    InputError,
    RingSpec,
    SimulationInputs,
    SimulationStatus,
    TargetError,
    compute_derived_targets,
    default_optimization_inputs,
    default_ring_specs,
    default_simulation_inputs,
    r_contact_radius,
)
from src.physics import JacketModel, solve_jacket

# --------------------------------------------------------------------------
# Toolchain / package
# --------------------------------------------------------------------------


def test_package_imports() -> None:
    """The src package imports and exposes its public modules."""
    assert src.__doc__ is not None
    for mod in ("models", "expressions", "materials", "physics",
                "solvers", "io_utils"):
        assert hasattr(src, mod)


def test_scientific_stack_available() -> None:
    """numpy and scipy are importable with expected major versions."""
    import scipy

    assert np.array([1.0, 2.0]).sum() == 3.0
    assert int(scipy.__version__.split(".")[0]) >= 1


# --------------------------------------------------------------------------
# MATLAB v1.5 defaults (verified against HARF_drawing.mlapp
# createComponents: D0=8.3556 mm, d0=2.4 mm, Dc0=0.48 mm, ...)
# --------------------------------------------------------------------------


def test_default_jacket_geometry_uses_radii():
    """Jacket D0/d0 are diameters; inputs must hold radii (half)."""
    inp = default_simulation_inputs(nb_nested=1)
    assert inp.R0 == pytest.approx(mm_to_m(DEFAULT_D0_MM) / 2.0)
    assert inp.r0 == pytest.approx(mm_to_m(2.4) / 2.0)
    assert inp.R0 == pytest.approx(8.3556e-3 / 2, rel=1e-12)
    assert inp.r0 == pytest.approx(1.2e-3, rel=1e-12)


def test_default_process_conditions():
    inp = default_simulation_inputs(nb_nested=1)
    assert inp.w0 == pytest.approx(4e-3 / 60.0, rel=1e-15)
    assert inp.tension == pytest.approx(84e-3 * 9.81, rel=1e-15)
    assert inp.Pcore == pytest.approx(15.6 * 100.0, rel=1e-15)
    assert inp.Pcap == (63.4 * 100.0,)
    assert inp.L == 0.1 and inp.Lfour == 0.065
    assert inp.Tmax_C == 1806.9
    assert inp.Text_C == 20.0 and inp.dT0 == 500.0
    assert inp.sigma == 5.67e-8 and inp.N_heat == 100.0
    assert inp.g == 9.81 and inp.n_capillaries == 2


def test_default_ring_geometry_uses_radii():
    rings = default_ring_specs(3)
    assert len(rings) == 3
    for i, ring in enumerate(rings):
        scale = 0.5**i  # nested rings shrink geometrically
        assert ring.Rc0 == pytest.approx(
            mm_to_m(DEFAULT_DC0_MM) / 2.0 * scale
        )
        assert ring.rc0 == pytest.approx(mm_to_m(0.3936) / 2.0 * scale)
        # the wall ratio dc0/Dc0 = 0.82 is preserved on every ring
        assert ring.rc0 / ring.Rc0 == pytest.approx(0.82, rel=1e-12)
    # the default preform is strictly nested
    for outer, inner in zip(rings, rings[1:], strict=False):
        assert inner.Rc0 < outer.rc0


def mm_to_m(v: float) -> float:
    return float(v) * 1e-3


def test_nested_identical_defaults_rejected_like_matlab():
    """MATLAB alerts 'Dc02>=dc01' when nested placeholder rings overlap."""
    good = default_simulation_inputs(nb_nested=2)
    good.validate()  # the scaled defaults are strictly nested
    bad = replace(good, rings=(good.rings[0], good.rings[0]))
    with pytest.raises(InputError):
        bad.validate()
    opt = default_optimization_inputs(nb_nested=2)
    with pytest.raises(TargetError):
        compute_derived_targets(bad, opt)


# --------------------------------------------------------------------------
# Geometry invariants
# --------------------------------------------------------------------------


def test_delta_j0_invariant():
    inp = default_simulation_inputs(nb_nested=1)
    assert inp.delta_J0 == pytest.approx(
        (inp.R0**2 - inp.r0**2) * inp.w0, rel=1e-15
    )
    assert inp.delta_J0 > 0.0


def test_delta_c0_invariant():
    inp = default_simulation_inputs(nb_nested=1)
    ring = inp.rings[0]
    assert inp.delta_c0(0) == pytest.approx(
        (ring.Rc0**2 - ring.rc0**2) * inp.w0, rel=1e-15
    )


def test_capillary_deltas_chain():
    inp = default_simulation_inputs(nb_nested=3)
    deltas = inp.capillary_deltas()
    assert deltas == (
        inp.Pcap[0] - inp.Pcore,
        inp.Pcap[1] - inp.Pcap[0],
        inp.Pcap[2] - inp.Pcap[1],
    )


def test_r_contact_radius_formula():
    rc = r_contact_radius(1.2e-3, 2)
    s = math.sin(math.pi / 2.0)
    assert rc == pytest.approx(1.2e-3 * s / (1.0 + s), rel=1e-15)
    arr = r_contact_radius(np.array([1e-3, 2e-3]), 3)
    assert isinstance(arr, np.ndarray)


def test_with_process_returns_valid_copy():
    inp = default_simulation_inputs(nb_nested=1)
    out = inp.with_process(Tmax_C=1900.0, Pcore=2000.0, Pcap=(7000.0,))
    assert out is not inp
    assert out.Tmax_C == 1900.0 and out.Pcore == 2000.0
    assert out.Pcap == (7000.0,)
    assert out.R0 == inp.R0
    out.validate()


# --------------------------------------------------------------------------
# Derived targets (RunButton_2Pushed rules)
# --------------------------------------------------------------------------


def test_derived_targets_default_values():
    """Default targets match the independently validated prototype."""
    inp = default_simulation_inputs(nb_nested=1)
    opt = default_optimization_inputs(nb_nested=1)
    tgt = compute_derived_targets(inp, opt)
    assert tgt.tf == pytest.approx(3.5e-6)
    assert tgt.Rcf == pytest.approx(55e-6, rel=1e-12)
    assert tgt.rcf == pytest.approx(51.5e-6, rel=1e-12)
    assert tgt.wf == pytest.approx(3.374873e-3, rel=1e-6)
    assert tgt.rf == pytest.approx(156e-6 / 2 + 2 * 55e-6, rel=1e-12)
    assert tgt.Rf == pytest.approx(
        math.sqrt(tgt.rf**2 + inp.delta_J0 / tgt.wf), rel=1e-12
    )
    assert tgt.Rf == pytest.approx(5.930279e-4, rel=1e-6)
    assert tgt.wf > inp.w0  # draw ratio > 1


def test_derived_targets_ring2_area_conservation():
    """Nested ring targets conserve the ring area flux at w = wf."""
    inp = default_simulation_inputs(nb_nested=2)
    rings = list(inp.rings)
    rings[1] = RingSpec(Rc0=0.12e-3, rc0=0.10e-3)
    inp = SimulationInputs(
        r0=inp.r0, R0=inp.R0, rings=tuple(rings), Pcap=inp.Pcap
    )
    opt = default_optimization_inputs(nb_nested=2)
    tgt = compute_derived_targets(inp, opt)
    (Rcf2, rcf2) = tgt.ring_targets[1]
    assert 0.0 < rcf2 < Rcf2 < tgt.rcf  # strictly nested
    # area flux conservation: Rcf2^2 - rcf2^2 = (Rc02^2 - rc02^2) w0/wf
    area = Rcf2**2 - rcf2**2
    expected = (rings[1].Rc0**2 - rings[1].rc0**2) * inp.w0 / tgt.wf
    assert area == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# Validated default trajectory (independent prototype of RK_perso.m)
# --------------------------------------------------------------------------


def test_default_jacket_matches_validated_prototype():
    """Default draw reproduces the validated DOP853 prototype run.

    Reference values computed by an independent transcription of
    ``RK_perso.m`` (rtol=1e-9, atol=1e-13, chained hot/cold segments):
    tau0* = 0.858952459, tau(L) = 0.824040, w(L) = 1.225627e-3 m/s,
    R(L) = 9.832618e-4 m, r(L) = 3.094167e-4 m.
    """
    inp = default_simulation_inputs(nb_nested=1)
    mats = MaterialSet.build(inp)
    model = JacketModel(inp, mats)
    tau0 = 0.858952459
    run = solve_jacket(model, tau0)
    assert run.ok
    assert run.status == SimulationStatus.SUCCESS
    assert run.z_stop == pytest.approx(inp.L)
    wL, uJL = float(run.y_end[1]), float(run.y_end[2])
    assert float(run.y_end[0]) == pytest.approx(0.824040, abs=1e-7)
    assert wL == pytest.approx(1.225627e-3, rel=1e-6)
    assert math.sqrt(uJL / wL) == pytest.approx(9.832618e-4, rel=1e-6)
    assert math.sqrt((uJL - inp.delta_J0) / wL) == pytest.approx(
        3.094167e-4, rel=1e-5
    )


def test_simulation_status_enum_complete():
    names = {s.name for s in SimulationStatus}
    assert {
        "SUCCESS",
        "INPUT_ERROR",
        "INFEASIBLE",
        "TENSION_LOSS",
        "JACKET_COLLAPSE",
        "JACKET_BLOWUP",
        "VELOCITY_BLOWUP",
        "VELOCITY_COLLAPSE",
        "CAPILLARY_COLLAPSE",
        "CAPILLARY_BLOWUP",
        "DIVERGED",
        "CANCELLED",
    } <= names
