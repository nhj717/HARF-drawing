"""High-level solvers: initial tension, direct simulation, optimization.

Public entry points (all crash-proof: they never raise unhandled
exceptions on bad physics and always return structured results):

* :func:`solve_initial_tension` -- Brent root-find of the initial
  tension ``tau0`` such that ``tau(L) = tau_target``.
* :func:`solve_capillary_delta_p` -- Brent root-find of the pressure
  difference ``DeltaP_i`` such that ``Rc_i(L) = Rcf_i``.
* :func:`run_direct_simulation` -- full jacket + capillary run.
* :func:`run_optimization` -- coarse (Tmax, Pcore) grid scan, bounded
  TRF refinement, per-ring ``DeltaP_i`` recovery and final
  high-precision direct simulation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

from src.expressions import ExpressionError
from src.materials import MaterialSet, validate_materials
from src.models import (
    DerivedTargets,
    FailureDiagnostics,
    InputError,
    OptimizationInputs,
    OptimizationResult,
    ProgressCallback,
    SimulationInputs,
    SimulationResult,
    SimulationStatus,
    TargetError,
    compute_derived_targets,
)
from src.physics import (
    CAPILLARY_ATOL,
    DEFAULT_DZ,
    DEFAULT_RTOL,
    JACKET_ATOL,
    CapillaryModel,
    CapillaryRun,
    ComputationCancelled,
    JacketModel,
    JacketRun,
    assemble_simulation_result,
    solve_capillary,
    solve_jacket,
)

__all__ = [
    "DeltaPSolution",
    "TensionSolution",
    "run_direct_simulation",
    "run_optimization",
    "solve_capillary_delta_p",
    "solve_initial_tension",
]

_EPS = float(np.finfo(float).eps)

#: A draw whose final speed is within this relative margin of ``w0``
#: never thinned the preform (frozen, ultra-viscous material).
_NO_DRAW_REL_TOL = 1e-6

#: A tension solve whose best run lands within this relative residual
#: of the target is accepted: on tiny targets (e.g. 1e-9 N) the strict
#: 1e-8 relative tolerance is below the achievable root-find precision.
_TENSION_NEAR_TOL = 1e-3

#: Failure families of the tension root-find: a bore/wall collapse or a
#: tension loss behaves like ``tau(L) << tau_target`` (negative residual
#: pushes the bracket upward), every blow-up/divergence behaves like
#: ``tau(L) >> tau_target`` (positive residual pushes it downward).
_TENSION_NEGATIVE = frozenset(
    {SimulationStatus.TENSION_LOSS, SimulationStatus.VELOCITY_COLLAPSE}
)

#: Failure families of the (Tmax, Pcore) optimization residual.  The
#: penalty vectors extrapolate the residuals along the direction the
#: failed run was heading (collapse => R/r -> inf; velocity blow-up =>
#: over-drawn fiber; jacket blow-up => R -> inf but R/r -> 1).
_TP_PENALTY = {
    SimulationStatus.TENSION_LOSS: (1.0, 1.0),
    SimulationStatus.JACKET_COLLAPSE: (1.0, 1.0),
    SimulationStatus.VELOCITY_COLLAPSE: (1.0, 1.0),
    SimulationStatus.JACKET_BLOWUP: (1.0, -1.0),
    SimulationStatus.VELOCITY_BLOWUP: (-1.0, -1.0),
    SimulationStatus.DIVERGED: (-1.0, -1.0),
}


def _notify(progress: ProgressCallback | None, message: str, frac: float):
    """Invoke a progress callback, swallowing any exception it raises."""
    if progress is None:
        return
    try:
        progress(message, frac)
    except Exception:  # noqa: BLE001, S110 - the UI must never break the solve
        pass


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ComputationCancelled()


# --------------------------------------------------------------------------
# Initial tension
# --------------------------------------------------------------------------


@dataclass
class TensionSolution:
    """Outcome of the initial-tension root-find."""

    tau0: float
    run: JacketRun | None
    residual: float
    converged: bool
    bracket: tuple[float, float]
    n_evals: int
    message: str = ""


def solve_initial_tension(
    inputs: SimulationInputs,
    materials: MaterialSet | None = None,
    *,
    rtol: float = DEFAULT_RTOL,
    atol=JACKET_ATOL,
    xtol: float = 1e-12,
    cancel_event=None,
    progress: ProgressCallback | None = None,
) -> TensionSolution:
    """Find ``tau0`` so that ``tau(L) = inputs.tension``.

    Uses ``brentq`` on ``F(tau0) = tau(L) - tau_target`` with a
    geometrically expanding bracket ``[tau_t/2^k, 2 tau_t 2^k]``.
    Failed runs return signed penalties (``-tau_t xi`` on collapse /
    tension loss, ``+10 tau_t xi`` on blow-up / divergence) so the
    root-find stays monotone across breakdown regimes.
    """
    if materials is None:
        materials = MaterialSet.build(inputs)
    model = JacketModel(inputs, materials)
    target = inputs.tension
    L = inputs.L
    state = {"n": 0, "best": None}  # best (|F|, tau0, run)

    def eval_F(tau0: float):
        _check_cancel(cancel_event)
        run = solve_jacket(
            model, tau0, rtol=rtol, atol=atol, cancel_event=cancel_event
        )
        state["n"] += 1
        if run.ok:
            value = float(run.y_end[0]) - target
        else:
            xi = 1.0 + (L - run.z_stop) / L
            if run.status in _TENSION_NEGATIVE:
                value = -target * xi
            else:
                value = 10.0 * target * xi
        best = state["best"]
        if best is None or abs(value) < best[0]:
            state["best"] = (abs(value), float(tau0), run)
        return value, run

    _notify(progress, "Solving initial tension", 0.05)
    a, b = 0.5 * target, 2.0 * target
    Fa, _ = eval_F(a)
    Fb, _ = eval_F(b)
    bracket = (a, b)
    for _ in range(60):
        if Fa <= 0.0 <= Fb:
            break
        moved = False
        if Fa > 0.0 and a > 1e-15:
            a *= 0.5
            Fa, _ = eval_F(a)
            moved = True
        if Fb < 0.0 and b < 1e6 * max(target, 1.0):
            b *= 2.0
            Fb, _ = eval_F(b)
            moved = True
        bracket = (a, b)
        if not moved:
            break

    if not (Fa <= 0.0 <= Fb):
        _, tau_best, run_best = state["best"]
        return TensionSolution(
            tau0=tau_best,
            run=run_best,
            residual=math.inf,
            converged=False,
            bracket=bracket,
            n_evals=state["n"],
            message=(
                "No initial tension achieves the target draw tension "
                f"inside the bracket [{a:.3g}, {b:.3g}] N."
            ),
        )

    def f(tau0: float) -> float:
        value, _ = eval_F(tau0)
        return value

    tau0 = brentq(f, a, b, xtol=xtol, maxiter=200)
    _, run = eval_F(tau0)
    if run.ok:
        residual = abs(float(run.y_end[0]) - target) / target
    else:  # pragma: no cover - root landed on a penalty plateau
        residual = math.inf
    return TensionSolution(
        tau0=tau0,
        run=run,
        residual=residual,
        converged=bool(run.ok and residual <= 1e-8),
        bracket=bracket,
        n_evals=state["n"],
        message="",
    )


# --------------------------------------------------------------------------
# Capillary pressure difference
# --------------------------------------------------------------------------


@dataclass
class DeltaPSolution:
    """Outcome of the per-ring pressure-difference root-find."""

    delta_p: float
    converged: bool
    residual: float
    ring: int
    bracket: tuple[float, float]
    n_evals: int
    message: str = ""


def solve_capillary_delta_p(
    jacket_run: JacketRun,
    inputs: SimulationInputs,
    materials: MaterialSet,
    ring_index: int,
    Rcf: float,
    *,
    dP_range: tuple[float, float] | None = None,
    rtol: float = DEFAULT_RTOL,
    atol=CAPILLARY_ATOL,
    xtol: float = 1e-6,
    cancel_event=None,
    progress: ProgressCallback | None = None,
    strict_bounds: bool = False,
) -> DeltaPSolution:
    """Find ``DeltaP_i`` so that ``Rc_i(L) = Rcf_i`` (decoupled 1-DOF).

    ``F(DeltaP_i) = (Rc_i(L) - Rcf_i)/Rcf_i`` is strictly increasing;
    collapsed runs return ``(sqrt(Delta_c0/w(L)) - Rcf_i)/Rcf_i - xi``
    and blown-up runs ``(10 Rc0_i - Rcf_i)/Rcf_i + 10 xi`` so the Brent
    root-find stays monotone across breakdown regimes.

    With ``strict_bounds=True`` the supplied ``dP_range`` is enforced as
    a hard bound: the bracket is never expanded beyond it, so a target
    root lying outside the range returns a non-converged solution
    instead of silently ignoring the user's window.
    """
    traj = jacket_run.traj
    dc0 = inputs.delta_c0(ring_index)
    Rc0 = inputs.rings[ring_index].Rc0
    wL = float(jacket_run.y_end[1])
    z_end = jacket_run.z_stop
    state = {"n": 0}

    def eval_F(delta_p: float):
        _check_cancel(cancel_event)
        cap_model = CapillaryModel(
            inputs, materials, ring_index, delta_p, traj, z_end
        )
        run = solve_capillary(
            cap_model, rtol=rtol, atol=atol, cancel_event=cancel_event
        )
        state["n"] += 1
        if run.ok:
            RcL = math.sqrt(max(run.uc_end / wL, 0.0))
            return (RcL - Rcf) / Rcf, run
        xi = (1.0 + (z_end - run.z_stop) / z_end) if z_end > 0 else 2.0
        if run.status == SimulationStatus.CAPILLARY_COLLAPSE:
            return (math.sqrt(dc0 / wL) - Rcf) / Rcf - xi, run
        return (10.0 * Rc0 - Rcf) / Rcf + 10.0 * xi, run

    if dP_range is None:
        lo, hi = -1e4, 1e5
    else:
        lo, hi = float(dP_range[0]), float(dP_range[1])
    if not hi > lo:
        lo, hi = lo - 1.0, lo + 1.0
    span = max(hi - lo, 1.0)
    Flo, _ = eval_F(lo)
    Fhi, _ = eval_F(hi)
    for _ in range(60):
        if Flo <= 0.0 <= Fhi:
            break
        if strict_bounds:
            break  # the user's window is a hard bound
        if Flo > 0.0:
            lo -= span
            span *= 2.0
            Flo, _ = eval_F(lo)
        elif Fhi < 0.0:
            hi += span
            span *= 2.0
            Fhi, _ = eval_F(hi)
    bracket = (lo, hi)

    if not (Flo <= 0.0 <= Fhi):
        return DeltaPSolution(
            delta_p=float(lo if abs(Flo) <= abs(Fhi) else hi),
            converged=False,
            residual=math.inf,
            ring=ring_index + 1,
            bracket=bracket,
            n_evals=state["n"],
            message=(
                f"Ring {ring_index + 1}: no pressure difference inside "
                f"[{lo:.4g}, {hi:.4g}] Pa reaches Rcf = {Rcf:.3e} m."
            ),
        )

    def f(delta_p: float) -> float:
        value, _ = eval_F(delta_p)
        return value

    delta_p = brentq(f, lo, hi, xtol=xtol, maxiter=200)
    Ffinal, run = eval_F(delta_p)
    return DeltaPSolution(
        delta_p=float(delta_p),
        converged=bool(run.ok and abs(Ffinal) <= 1e-8),
        residual=abs(Ffinal),
        ring=ring_index + 1,
        bracket=bracket,
        n_evals=state["n"],
    )


# --------------------------------------------------------------------------
# Direct simulation
# --------------------------------------------------------------------------


def _empty_result(
    inputs: SimulationInputs,
    status: SimulationStatus,
    message: str,
    advice: str,
    elapsed_s: float,
) -> SimulationResult:
    one = np.zeros(1)
    return SimulationResult(
        status=status,
        inputs=inputs,
        z=one.copy(),
        tau=one.copy(),
        w=one.copy(),
        T=one.copy(),
        Ta=one.copy(),
        R=one.copy(),
        r=one.copy(),
        Rc=(),
        rc=(),
        diagnostics=FailureDiagnostics(
            status=status,
            z_stop=0.0,
            message=message,
            advice=advice,
        ),
        elapsed_s=elapsed_s,
    )


def _solve_capillaries(
    inputs: SimulationInputs,
    materials: MaterialSet,
    jacket_run: JacketRun,
    *,
    rtol: float = DEFAULT_RTOL,
    atol=CAPILLARY_ATOL,
    cancel_event=None,
) -> tuple[CapillaryRun, ...]:
    if jacket_run.z_stop <= 0.0:
        return ()
    deltas = inputs.capillary_deltas()
    runs = []
    for i in range(inputs.nb_nested):
        cap_model = CapillaryModel(
            inputs,
            materials,
            i,
            deltas[i],
            jacket_run.traj,
            jacket_run.z_stop,
        )
        runs.append(
            solve_capillary(
                cap_model, rtol=rtol, atol=atol, cancel_event=cancel_event
            )
        )
    return tuple(runs)


def run_direct_simulation(
    inputs: SimulationInputs,
    *,
    dz: float = DEFAULT_DZ,
    rtol: float = DEFAULT_RTOL,
    atol=JACKET_ATOL,
    xtol_tau: float = 1e-12,
    cancel_event=None,
    progress: ProgressCallback | None = None,
) -> SimulationResult:
    """Run one complete drawing simulation (never raises).

    Invalid inputs or material expressions return an ``INPUT_ERROR``
    result; breakdowns (collapse, blow-up, tension loss...) return
    partial trajectories on ``[0, z_stop]`` with actionable
    diagnostics.
    """
    start = time.perf_counter()
    try:
        inputs.validate()
    except InputError as exc:
        return _empty_result(
            inputs,
            SimulationStatus.INPUT_ERROR,
            f"Invalid simulation inputs: {exc}",
            "Fix the highlighted input fields and retry.",
            time.perf_counter() - start,
        )
    except TypeError as exc:
        return _empty_result(
            inputs,
            SimulationStatus.INPUT_ERROR,
            f"Malformed simulation inputs: {exc}",
            "Every numeric field must be a scalar number.",
            time.perf_counter() - start,
        )
    try:
        materials = MaterialSet.build(inputs)
        validate_materials(inputs)
    except ExpressionError as exc:
        return _empty_result(
            inputs,
            SimulationStatus.INPUT_ERROR,
            f"Invalid material expression: {exc}",
            "Check the viscosity / surface tension / profile formulas.",
            time.perf_counter() - start,
        )
    try:
        _notify(progress, "Solving initial tension", 0.15)
        tension = solve_initial_tension(
            inputs,
            materials,
            rtol=rtol,
            atol=atol,
            xtol=xtol_tau,
            cancel_event=cancel_event,
            progress=progress,
        )
        tension_ok = tension.converged or (
            tension.run is not None
            and tension.run.ok
            and tension.residual <= _TENSION_NEAR_TOL
        )
        if not tension_ok or tension.run is None:
            run = tension.run
            advice = (
                "Widen the draw tension, furnace temperature or pressure "
                "settings; the draw is not achievable as requested."
            )
            cap_runs = (
                _solve_capillaries(
                    inputs,
                    materials,
                    run,
                    rtol=rtol,
                    cancel_event=cancel_event,
                )
                if run is not None
                else ()
            )
            if run is None:
                status = SimulationStatus.DIVERGED
            elif run.ok:
                # The ODE completed but the target tension is
                # unreachable: the draw breaks down first.  Classify by
                # the limiting family (raising tau0 -> velocity blow-up,
                # lowering it -> tension loss).
                tau_L = float(run.y_end[0])
                if tau_L < inputs.tension:
                    status = SimulationStatus.VELOCITY_BLOWUP
                else:
                    status = SimulationStatus.TENSION_LOSS
            else:
                status = run.status
            if run is not None and run.ok:
                message = (
                    "Initial tension could not be solved: the target draw "
                    f"tension ({inputs.tension:.4g} N) is unreachable -- "
                    "the draw breaks down first (best tau(L) = "
                    f"{float(run.y_end[0]):.4g} N)."
                )
            else:
                message = (
                    "Initial tension could not be solved: "
                    + (tension.message or "the draw broke down.")
                )
            diagnostics = FailureDiagnostics(
                status=status,
                z_stop=run.z_stop if run is not None else 0.0,
                message=message,
                advice=advice,
            )
            return assemble_simulation_result(
                inputs,
                run if run is not None else _degenerate_jacket_run(inputs),
                cap_runs,
                dz=dz,
                tau0=tension.tau0,
                elapsed_s=time.perf_counter() - start,
                diagnostics=diagnostics,
                status_override=status,
            )
        run = tension.run
        _notify(progress, "Integrating capillaries", 0.75)
        cap_runs = _solve_capillaries(
            inputs,
            materials,
            run,
            rtol=rtol,
            cancel_event=cancel_event,
        )
        # Frozen-draw guard: an ultra-cold furnace clamps the viscosity
        # so high that nothing is drawn (w stays at w0) even though the
        # ODEs integrate cleanly over the full zone.
        caps_ok = all(cap.ok for cap in cap_runs)
        w_end = float(run.y_end[1])
        if caps_ok and (w_end - inputs.w0) <= _NO_DRAW_REL_TOL * inputs.w0:
            _notify(progress, "Simulation finished", 1.0)
            return assemble_simulation_result(
                inputs,
                run,
                cap_runs,
                dz=dz,
                tau0=tension.tau0,
                elapsed_s=time.perf_counter() - start,
                status_override=SimulationStatus.NO_DRAW,
            )
        _notify(progress, "Simulation finished", 1.0)
        return assemble_simulation_result(
            inputs,
            run,
            cap_runs,
            dz=dz,
            tau0=tension.tau0,
            elapsed_s=time.perf_counter() - start,
        )
    except ComputationCancelled:
        return _empty_result(
            inputs,
            SimulationStatus.CANCELLED,
            "Computation cancelled by the user.",
            "No action required.",
            time.perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        return _empty_result(
            inputs,
            SimulationStatus.DIVERGED,
            f"Unexpected internal error: {exc!r}",
            "Please report this unexpected failure.",
            time.perf_counter() - start,
        )


def _degenerate_jacket_run(
    inputs: SimulationInputs,
) -> JacketRun:
    """A zero-length jacket run used when nothing could be integrated."""
    from src.physics import DenseTrajectory

    class _Const:
        def __init__(self, y0):
            self._y0 = np.asarray(y0, dtype=float)

        def sol(self, z):
            z_arr = np.atleast_1d(np.asarray(z, dtype=float))
            return np.repeat(
                self._y0[:, None], z_arr.size, axis=1
            )

    y0 = np.zeros(4)
    return JacketRun(
        ok=False,
        status=SimulationStatus.DIVERGED,
        event_name=None,
        z_stop=0.0,
        traj=DenseTrajectory([(0.0, _Const(y0))], n_state=4),
        y_end=y0,
        nfev=0,
        used_radau=False,
    )


# --------------------------------------------------------------------------
# Optimization
# --------------------------------------------------------------------------


def run_optimization(
    inputs: SimulationInputs,
    opt: OptimizationInputs,
    *,
    dz: float = DEFAULT_DZ,
    rtol: float = DEFAULT_RTOL,
    cancel_event=None,
    progress: ProgressCallback | None = None,
) -> OptimizationResult:
    """Optimize ``(Tmax, Pcore)`` then each ``DeltaP_i`` to hit the targets.

    Pipeline: coarse (Tmax, Pcore) grid scan (10 degC T steps, low /
    mid / high Pcore) -> bounded TRF ``least_squares`` refinement on the
    residuals ``[(R(L)-Rf)/Rf, (R(L)/r(L) - Rf/rf)/(Rf/rf)]`` -> final
    high-precision tension + jacket solve -> per-ring ``brentq`` on
    ``DeltaP_i`` along the cached trajectory -> final direct simulation.
    """
    start = time.perf_counter()
    try:
        inputs.validate()
        opt.validate(inputs.nb_nested)
    except (InputError, TypeError) as exc:
        return OptimizationResult(
            status=SimulationStatus.INPUT_ERROR,
            targets=None,
            message=f"Invalid inputs: {exc}",
            advice="Fix the highlighted input fields and retry.",
            elapsed_s=time.perf_counter() - start,
        )
    try:
        targets = compute_derived_targets(inputs, opt)
    except TargetError as exc:
        return OptimizationResult(
            status=SimulationStatus.INFEASIBLE,
            targets=None,
            message=f"Impossible targets: {exc}",
            advice="Revise the final-fiber targets (tf, Dcf1, Dcore).",
            elapsed_s=time.perf_counter() - start,
        )
    except (TypeError, ValueError) as exc:
        return OptimizationResult(
            status=SimulationStatus.INPUT_ERROR,
            targets=None,
            message=f"Malformed optimization inputs: {exc}",
            advice="Every numeric field must be a scalar number.",
            elapsed_s=time.perf_counter() - start,
        )
    try:
        materials = MaterialSet.build(inputs)
        validate_materials(inputs)
    except (ExpressionError, TypeError) as exc:
        return OptimizationResult(
            status=SimulationStatus.INPUT_ERROR,
            targets=None,
            message=f"Invalid material expression: {exc}",
            advice="Check the material law formulas.",
            elapsed_s=time.perf_counter() - start,
        )

    L = inputs.L
    Rf = targets.Rf
    ratio_f = targets.Rf / targets.rf
    dJ0 = inputs.delta_J0

    def make_residual(rtol_inner: float, xtol_inner: float):
        def residual(x):
            _check_cancel(cancel_event)
            proc = inputs.with_process(
                Tmax_C=float(x[0]), Pcore=float(x[1])
            )
            tension = solve_initial_tension(
                proc,
                materials,
                rtol=rtol_inner,
                xtol=xtol_inner,
                cancel_event=cancel_event,
            )
            run = tension.run
            xi = 1.0 + (L - run.z_stop) / L
            if run.ok:
                wL = float(run.y_end[1])
                uL = float(run.y_end[2])
                R = math.sqrt(uL / wL)
                rr = math.sqrt(max((uL - dJ0) / wL, 0.0))
                return np.array(
                    [(R - Rf) / Rf, (R / rr - ratio_f) / ratio_f]
                )
            sign = _TP_PENALTY.get(run.status, (-1.0, -1.0))
            return np.array([sign[0] * 10.0 * xi, sign[1] * 10.0 * xi])

        return residual

    try:
        # ---- coarse grid scan ----
        _notify(progress, "Scanning (Tmax, Pcore) grid", 0.05)
        resid_coarse = make_residual(1e-7, 1e-9)
        t_values = list(
            np.arange(
                opt.Tmax_min_C, opt.Tmax_max_C + 1e-9, 10.0
            )
        )
        if not t_values or t_values[-1] < opt.Tmax_max_C - 1e-9:
            t_values.append(opt.Tmax_max_C)
        p_values = [
            opt.Pcore_min,
            0.5 * (opt.Pcore_min + opt.Pcore_max),
            opt.Pcore_max,
        ]
        best_cost = math.inf
        best_x = np.array(
            [0.5 * (opt.Tmax_min_C + opt.Tmax_max_C), p_values[1]]
        )
        grid = [(T, P) for T in t_values for P in p_values]
        for (T, P), r in zip(
            grid, _scan_grid(grid, resid_coarse, cancel_event), strict=True
        ):
            cost = float(r @ r)
            if cost < best_cost:
                best_cost = cost
                best_x = np.array([T, P])

        # ---- TRF refinement ----
        _notify(progress, "Refining (Tmax, Pcore)", 0.35)
        resid_fine = make_residual(1e-8, 1e-10)
        ls = least_squares(
            resid_fine,
            best_x,
            bounds=(
                np.array([opt.Tmax_min_C, opt.Pcore_min]),
                np.array([opt.Tmax_max_C, opt.Pcore_max]),
            ),
            diff_step=np.array([1e-5, 1e-4]),
            xtol=1e-12,
            ftol=1e-14,
            gtol=1e-14,
            max_nfev=80,
        )
        Tmax_star = float(np.clip(ls.x[0], opt.Tmax_min_C, opt.Tmax_max_C))
        Pcore_star = float(
            np.clip(ls.x[1], opt.Pcore_min, opt.Pcore_max)
        )

        # ---- final high-precision tension + jacket solve ----
        _notify(progress, "Final high-precision solve", 0.6)
        final_proc = inputs.with_process(
            Tmax_C=Tmax_star, Pcore=Pcore_star
        )
        tension = solve_initial_tension(
            final_proc,
            materials,
            rtol=rtol,
            xtol=1e-12,
            cancel_event=cancel_event,
        )
        if not tension.converged or tension.run is None:
            return OptimizationResult(
                status=(
                    tension.run.status
                    if tension.run is not None
                    else SimulationStatus.DIVERGED
                ),
                targets=targets,
                Tmax_C=Tmax_star,
                Pcore=Pcore_star,
                tau0=tension.tau0,
                message=(
                    "The (Tmax, Pcore) optimum found by the scan could not "
                    "be drawn: " + tension.message
                ),
                advice=(
                    "Widen the (Tmax, Pcore) search window or revise the "
                    "targets."
                ),
                elapsed_s=time.perf_counter() - start,
            )
        run = tension.run

        # ---- per-ring DeltaP recovery along the cached trajectory ----
        deltas: list[float] = []
        p_below = Pcore_star
        for i in range(inputs.nb_nested):
            _check_cancel(cancel_event)
            lo = opt.Pcap_min[i] - p_below
            hi = opt.Pcap_max[i] - p_below
            sol = solve_capillary_delta_p(
                run,
                final_proc,
                materials,
                i,
                targets.Rcf_rings[i],
                dP_range=(lo, hi),
                rtol=rtol,
                cancel_event=cancel_event,
                strict_bounds=True,
            )
            if not sol.converged:
                return OptimizationResult(
                    status=SimulationStatus.INFEASIBLE,
                    targets=targets,
                    Tmax_C=Tmax_star,
                    Pcore=Pcore_star,
                    Pcap=_pcap(Pcore_star, deltas),
                    DeltaP=tuple(deltas),
                    tau0=tension.tau0,
                    cost=None,
                    message=sol.message,
                    advice=(
                        "Widen the capillary pressure windows or revise "
                        "the ring targets."
                    ),
                    elapsed_s=time.perf_counter() - start,
                )
            deltas.append(sol.delta_p)
            p_below += sol.delta_p
        Pcap = _pcap(Pcore_star, deltas)

        # ---- final direct simulation at the optimum ----
        _notify(progress, "Final simulation", 0.85)
        final_inputs = final_proc.with_process(Pcap=Pcap)
        sim = run_direct_simulation(
            final_inputs,
            dz=dz,
            rtol=rtol,
            cancel_event=cancel_event,
        )
        status, message, advice, cost = _grade(sim, targets)
        _notify(progress, "Optimization finished", 1.0)
        return OptimizationResult(
            status=status,
            targets=targets,
            Tmax_C=Tmax_star,
            Pcore=Pcore_star,
            Pcap=Pcap,
            DeltaP=tuple(deltas),
            tau0=sim.tau0,
            cost=cost,
            result=sim,
            message=message,
            advice=advice,
            elapsed_s=time.perf_counter() - start,
        )
    except ComputationCancelled:
        return OptimizationResult(
            status=SimulationStatus.CANCELLED,
            targets=targets,
            message="Optimization cancelled by the user.",
            advice="No action required.",
            elapsed_s=time.perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        return OptimizationResult(
            status=SimulationStatus.DIVERGED,
            targets=targets,
            message=f"Unexpected internal error: {exc!r}",
            advice="Please report this unexpected failure.",
            elapsed_s=time.perf_counter() - start,
        )


def _scan_grid(
    points: Sequence[Sequence[float]],
    residual,
    cancel_event,
    max_workers: int = 4,
) -> list[np.ndarray]:
    """Evaluate the coarse-grid residual in parallel worker threads.

    ``solve_ivp`` releases the GIL inside its compiled kernels, so a
    small thread pool overlaps the per-point tension solves.  A set
    ``cancel_event`` propagates as :class:`ComputationCancelled` from
    the first failed worker (the others abort on their next check).
    """
    points = list(points)
    if not points:
        return []
    _check_cancel(cancel_event)
    if len(points) < 2 * max_workers:
        return [residual(p) for p in points]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(residual, points))


def _pcap(pcore: float, deltas: Sequence[float]) -> tuple[float, ...]:
    out = []
    acc = pcore
    for d in deltas:
        acc += d
        out.append(acc)
    return tuple(out)


def _grade(
    sim: SimulationResult, targets: DerivedTargets
) -> tuple[SimulationStatus, str, str, float | None]:
    """Grade the final simulation against the optimization targets."""
    if not sim.ok or len(sim.z) == 0:
        diag = sim.diagnostics
        return (
            sim.status,
            diag.message if diag else "The final draw failed.",
            diag.advice if diag else "No action required.",
            None,
        )
    R_end = float(sim.R[-1])
    r_end = float(sim.r[-1])
    w_end = float(sim.w[-1])
    errors = {
        "Rf": abs(R_end - targets.Rf) / targets.Rf,
        "rf": abs(r_end - targets.rf) / targets.rf,
        "wf": abs(w_end - targets.wf) / targets.wf,
    }
    for i, (Rcf_i, rcf_i) in enumerate(targets.ring_targets):
        Rc_end = float(sim.Rc[i][-1])
        rc_end = float(sim.rc[i][-1])
        errors[f"Rcf{i + 1}"] = abs(Rc_end - Rcf_i) / Rcf_i
        errors[f"rcf{i + 1}"] = abs(rc_end - rcf_i) / rcf_i
    worst = max(errors.values())
    cost = worst**2
    if worst <= 1e-5:
        return (
            SimulationStatus.SUCCESS,
            (
                "Optimization converged: all targets reached within "
                f"{worst:.2e} relative error."
            ),
            "Apply the solution or export the results.",
            cost,
        )
    return (
        SimulationStatus.INFEASIBLE,
        (
            "The optimizer did not reach the targets (worst relative "
            f"error {worst:.2e})."
        ),
        (
            "Widen the (Tmax, Pcore, Pcap) search windows or revise the "
            "final-fiber targets."
        ),
        cost,
    )
