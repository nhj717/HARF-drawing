"""Piecewise jacket & capillary ODE integrators with a crash-proof net.

The axial domain ``z in [0, L]`` is split at ``L_four``:

* hot zone ``[0, L_four]``: ambient ``Ta = Tmax * profil(z) + 273.15 K``
  (``Tmax`` is the furnace set point in degrees Celsius, used as the
  profile amplitude in kelvin, exactly like the MATLAB app);
* cold zone ``(L_four, L]``: ambient ``Ta = Text + 273.15 K``.

The jacket is integrated with the mass-conservation reduction
``y_J = [tau, w, u_J, T]`` where ``u_J = R^2 w`` and the bore flux
follows from the invariant ``r^2 w = u_J - Delta_J0`` with
``Delta_J0 = (R0^2 - r0^2) w0``.  Each nested capillary ring is then
integrated as a decoupled 1-state ODE ``u_{c,i} = R_{c,i}^2 w`` driven
along the cached dense jacket trajectory ``(w(z), T(z))``, using
``r_{c,i}^2 w = u_{c,i} - Delta_{c0,i}``.

Every RHS is guarded (floored radii, clamped material laws) and wrapped
in a last-resort ``except`` returning huge derivatives so the
integrator rejects the step; terminal events catch breakdowns (bore
collapse, blow-up, tension loss, velocity loss/blow-up) before any
singularity is reached.  ``DOP853`` is used with a ``Radau`` fallback
on step collapse (``solve_ivp`` status ``-1``) or excessive stiffness
(``nfev > 100_000``).

:func:`rhs_unreduced` provides the unreduced ``(5 + 2*N_nested)``
transcription of ``RK_perso.m`` / ``RK_perso7/9/11.m`` (with the
``273.15`` K offset fix and the ``viscosity_3`` ring-3 bug fix) for
parity testing against the reduced solver.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from src.materials import MaterialSet
from src.models import SimulationInputs, SimulationStatus

__all__ = [
    "CAPILLARY_ATOL",
    "DEFAULT_DZ",
    "DEFAULT_RTOL",
    "JACKET_ATOL",
    "CapillaryModel",
    "CapillaryRun",
    "ComputationCancelled",
    "DenseTrajectory",
    "JacketModel",
    "JacketRun",
    "assemble_simulation_result",
    "rhs_unreduced",
    "solve_capillary",
    "solve_jacket",
    "uniform_grid",
    "unreduced_atol",
]

#: Uniform axial resampling step [m].
DEFAULT_DZ = 1e-5
#: Default relative tolerance of the ODE integrators.
DEFAULT_RTOL = 1e-9
#: Absolute tolerances per jacket state [tau, w, u_J, T].
JACKET_ATOL = (1e-12, 1e-15, 1e-20, 1e-9)
#: Absolute tolerance of the 1-state capillary integrator.
CAPILLARY_ATOL = 1e-22
#: DOP853 evaluation budget before the Radau fallback kicks in.
_MAX_NFEV_FALLBACK = 100_000
#: Last-resort derivative magnitude returned by guarded RHS calls.
_HUGE = 1e30
#: Relative bore-flux collapse threshold: the terminal collapse event
#: fires once the bore mass flux ``r^2 w`` drops below this fraction of
#: its initial value ``r0^2 w0``.  Viscous bore collapse decays
#: asymptotically (it never reaches an absolute floor), so only a
#: relative threshold catches strong suction (Pcore ~ -2e4 Pa, where
#: the flux falls below 3% of its initial value) while legitimate
#: thin-bore draws stay above ~14%.
_BORE_COLLAPSE_REL = 0.1

_JACKET_EVENT_STATUSES = (
    (SimulationStatus.JACKET_COLLAPSE, "jacket_collapse"),
    (SimulationStatus.JACKET_BLOWUP, "jacket_blowup"),
    (SimulationStatus.TENSION_LOSS, "tension_loss"),
    (SimulationStatus.VELOCITY_BLOWUP, "velocity_blowup"),
    (SimulationStatus.VELOCITY_COLLAPSE, "velocity_collapse"),
)

_CAPILLARY_EVENT_STATUSES = (
    (SimulationStatus.CAPILLARY_COLLAPSE, "capillary_collapse"),
    (SimulationStatus.CAPILLARY_BLOWUP, "capillary_blowup"),
)


class ComputationCancelled(RuntimeError):
    """Raised cooperatively when the user cancels a running computation."""


# --------------------------------------------------------------------------
# Dense piecewise trajectory
# --------------------------------------------------------------------------


class DenseTrajectory:
    """Piecewise dense ODE solution, callable on ``z in [0, z_end]``.

    Built from an ordered sequence of ``(t_end, scipy_sol)`` pieces
    covering ``[0, t_end_1], [t_end_1, t_end_2], ...``.  Calling with a
    scalar returns the state vector ``(n,)``; calling with an array
    returns ``(n, m)`` (mode-major, one column per grid point).
    """

    def __init__(
        self, pieces: Sequence[tuple], n_state: int
    ) -> None:
        if not pieces:
            raise ValueError("at least one solution piece is required")
        self._bounds = [float(p[0]) for p in pieces]
        self._sols = [p[1] for p in pieces]
        self.z_end = self._bounds[-1]
        self.n_state = int(n_state)

    def __call__(self, z):
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        zc = np.clip(z_arr, 0.0, self.z_end)
        idx = np.searchsorted(self._bounds, zc, side="left")
        idx = np.clip(idx, 0, len(self._sols) - 1)
        out = np.empty((self.n_state, zc.size), dtype=float)
        for k, sol in enumerate(self._sols):
            mask = idx == k
            if mask.any():
                out[:, mask] = sol.sol(zc[mask])
        if np.ndim(z) == 0:
            return out[:, 0]
        return out

    def state(self, z):
        """State vector at scalar ``z`` (clipped into ``[0, z_end]``)."""
        return self.__call__(float(z))

    def w(self, z) -> float:
        """Cached jacket speed ``w(z)`` [m/s]."""
        return float(self.state(z)[1])

    def T(self, z) -> float:
        """Cached jacket temperature ``T(z)`` [K]."""
        return float(self.state(z)[3])


# --------------------------------------------------------------------------
# Jacket model (reduced 4-state)
# --------------------------------------------------------------------------


class JacketModel:
    """Reduced 4-state jacket ODE model ``y = [tau, w, u_J, T]``.

    Mirrors ``RK_perso.m`` with ``r^2 w = u_J - Delta_J0`` and the
    fixed ``273.15`` K offsets (``Ta = Tmax*profil(z) + 273.15`` in the
    hot zone, ``Ta = Text + 273.15`` in the cold zone).
    """

    def __init__(
        self, inputs: SimulationInputs, materials: MaterialSet
    ) -> None:
        self.inputs = inputs
        self.m = materials
        self.R0 = inputs.R0
        self.r0 = inputs.r0
        self.w0 = inputs.w0
        self.dJ0 = inputs.delta_J0
        self.Pcore = inputs.Pcore
        self.g = inputs.g
        self.sigma = inputs.sigma
        self.N_heat = inputs.N_heat
        self.L = inputs.L
        self.Lfour = inputs.Lfour
        self.T_amp = inputs.Tmax_C
        self.Text_K = inputs.Text_C + 273.15
        self.T0 = (
            inputs.Tmax_C * materials.profil.scalar(0.0)
            + 273.15
            - inputs.dT0
        )
        # Guards only ever matter inside rejected trial steps: the
        # terminal events fire long before these floors are reached.
        self._w_floor = 1e-12 * inputs.w0
        self._R2_floor = 1e-16 * inputs.R0**2
        self._r2_floor = 1e-16 * inputs.r0**2

    # -- ambient ---------------------------------------------------------
    def ambient(self, z: float) -> float:
        """Ambient temperature ``Ta(z)`` [K] (piecewise hot/cold)."""
        if z <= self.Lfour:
            return self.T_amp * self.m.profil.scalar(z) + 273.15
        return self.Text_K

    def ambient_array(self, z: np.ndarray) -> np.ndarray:
        """Vectorized ambient temperature along ``z`` [K]."""
        z = np.asarray(z, dtype=float)
        Ta = np.full(z.shape, self.Text_K)
        hot = z <= self.Lfour
        if hot.any():
            Ta[hot] = self.T_amp * self.m.profil.vector(z[hot]) + 273.15
        return Ta

    # -- right-hand side ---------------------------------------------------
    def rhs(self, z: float, y):
        """Guarded RHS of the reduced jacket ODE (never raises)."""
        try:
            tau = float(y[0])
            w = float(y[1])
            uJ = float(y[2])
            T = float(y[3])
            w = max(w, self._w_floor)
            R2 = uJ / w
            r2 = (uJ - self.dJ0) / w
            R2 = max(R2, self._R2_floor)
            r2 = max(r2, self._r2_floor)
            R = math.sqrt(R2)
            r = math.sqrt(r2)
            R2 = R * R
            r2 = r * r
            mu = self.m.viscosity(T)
            gam = self.m.gamma(T)
            rho = self.m.rho(T)
            cp = self.m.cp(T)
            area = R2 - r2
            dwdz = tau / (3.0 * math.pi * mu * area)
            duJdz = (
                self.Pcore * r2 * R2 - gam * r * R * (r + R)
            ) / (mu * area)
            dtaudz = (
                rho * math.pi * area * (w * dwdz - self.g)
                - gam * math.pi * (duJdz - R2 * dwdz) / (2.0 * R * w)
                - gam * math.pi * (duJdz - r2 * dwdz) / (2.0 * r * w)
            )
            Ta = self.ambient(z)
            dTdz = (
                R * self.N_heat * (Ta - T)
                + R
                * self.sigma
                * (self.m.alpha(Ta) * Ta**4 - self.m.alpha(T) * T**4)
            ) * 2.0 / (area * rho * cp * w)
            return (dtaudz, dwdz, duJdz, dTdz)
        except (ArithmeticError, ValueError):
            return (_HUGE, _HUGE, _HUGE, _HUGE)

    # -- terminal events ---------------------------------------------------
    def events(self) -> list:
        """Terminal breakdown events (all ``direction=-1``)."""
        dJ0 = self.dJ0
        R0sq = self.R0**2
        w0 = self.w0
        # Viscous bore collapse decays asymptotically, so the event
        # floor is the larger of the ultra-tight absolute floor and a
        # relative floor on the initial bore flux ``r0^2 w0``.
        bore_floor = max(
            1e-12 * dJ0, _BORE_COLLAPSE_REL * self.r0**2 * w0
        )

        def collapse(z, y):
            return (y[2] - dJ0) - bore_floor

        def blowup(z, y):
            return 100.0 * R0sq * y[1] - y[2]

        def tension(z, y):
            return y[0] - 1e-9

        def vel_blowup(z, y):
            return 1e5 * w0 - y[1]

        def vel_collapse(z, y):
            return y[1] - 1e-3 * w0

        events = [collapse, blowup, tension, vel_blowup, vel_collapse]
        for ev in events:
            ev.terminal = True
            ev.direction = -1.0
        return events


# --------------------------------------------------------------------------
# Capillary model (decoupled 1-state per ring)
# --------------------------------------------------------------------------


class CapillaryModel:
    """Decoupled 1-state capillary ODE ``u_c = R_c^2 w`` of one ring.

    Driven along the cached dense jacket trajectory ``(w(z), T(z))``
    by the ring pressure difference ``DeltaP_i``; the bore flux obeys
    ``r_c^2 w = u_c - Delta_c0`` with ``Delta_c0 = (Rc0^2-rc0^2) w0``.
    """

    def __init__(
        self,
        inputs: SimulationInputs,
        materials: MaterialSet,
        ring_index: int,
        delta_p: float,
        jacket_traj: DenseTrajectory,
        jacket_z_end: float,
    ) -> None:
        ring = inputs.rings[ring_index]
        self.ring_index = int(ring_index)
        self.Rc0 = ring.Rc0
        self.rc0 = ring.rc0
        self.w0 = inputs.w0
        self.Lfour = inputs.Lfour
        self.dc0 = inputs.delta_c0(ring_index)
        self.dP = float(delta_p)
        self.mats = materials.rings[ring_index]
        self.traj = jacket_traj
        self.z_max = float(jacket_z_end)
        self._w_floor = 1e-12 * inputs.w0
        self._Rc2_floor = 1e-16 * ring.Rc0**2
        self._rc2_floor = 1e-16 * ring.rc0**2

    def rhs(self, z: float, y):
        """Guarded RHS of the 1-state capillary ODE (never raises)."""
        try:
            yj = self.traj.state(min(float(z), self.z_max))
            w = float(yj[1])
            T = float(yj[3])
            w = max(w, self._w_floor)
            uc = float(y[0])
            Rc2 = uc / w
            rc2 = (uc - self.dc0) / w
            Rc2 = max(Rc2, self._Rc2_floor)
            rc2 = max(rc2, self._rc2_floor)
            Rc = math.sqrt(Rc2)
            rc = math.sqrt(rc2)
            Rc2 = Rc * Rc
            rc2 = rc * rc
            mu = self.mats.viscosity(T)
            gam = self.mats.gamma(T)
            area = Rc2 - rc2
            ducdz = (
                self.dP * rc2 * Rc2 - gam * rc * Rc * (rc + Rc)
            ) / (mu * area)
            return (ducdz,)
        except (ArithmeticError, ValueError):
            return (_HUGE,)

    def events(self) -> list:
        """Terminal capillary breakdown events (all ``direction=-1``)."""
        dc0 = self.dc0
        Rc0sq = self.Rc0**2
        traj = self.traj
        w_floor = self._w_floor

        def collapse(z, y):
            return (y[0] - dc0) - 1e-12 * dc0

        def blowup(z, y):
            w = float(traj.state(min(float(z), traj.z_end))[1])
            w = max(w, w_floor)
            return 100.0 * Rc0sq * w - y[0]

        events = [collapse, blowup]
        for ev in events:
            ev.terminal = True
            ev.direction = -1.0
        return events


# --------------------------------------------------------------------------
# Integration drivers
# --------------------------------------------------------------------------


def _checked_rhs(
    rhs: Callable, cancel_event
) -> Callable:
    """Wrap an RHS with a cooperative-cancellation check."""
    if cancel_event is None:
        return rhs

    def wrapped(z, y):
        if cancel_event.is_set():
            raise ComputationCancelled()
        return rhs(z, y)

    return wrapped


def _triggered_event(sol) -> int | None:
    """Index of the terminal event that stopped ``sol`` (or None)."""
    for i, t_events in enumerate(sol.t_events):
        if t_events is not None and len(t_events) > 0:
            return i
    return None


def _integrate_segment(
    rhs,
    events,
    t0: float,
    t1: float,
    y0: np.ndarray,
    rtol: float,
    atol,
    cancel_event,
):
    """Integrate one segment with DOP853 and a Radau fallback.

    Returns ``(sol, used_radau)``.  The fallback triggers when DOP853
    fails (``status == -1``) or exhausts the evaluation budget.
    """
    if t1 <= t0:
        raise ValueError(f"empty segment [{t0}, {t1}]")
    sol = solve_ivp(
        rhs,
        (t0, t1),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        events=events,
    )
    if sol.status == 1:
        return sol, False  # a terminal event is a definitive answer
    needs_fallback = sol.status == -1 or sol.nfev > _MAX_NFEV_FALLBACK
    if not needs_fallback:
        return sol, False
    sol2 = solve_ivp(
        rhs,
        (t0, t1),
        y0,
        method="Radau",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        events=events,
    )
    radau_ok = sol2.status in (0, 1)
    if radau_ok and (sol.status == -1 or sol2.t[-1] >= sol.t[-1]):
        return sol2, True
    return sol, False


@dataclass
class JacketRun:
    """Outcome of one jacket integration."""

    ok: bool
    status: SimulationStatus
    event_name: str | None
    z_stop: float
    traj: DenseTrajectory
    y_end: np.ndarray
    nfev: int
    used_radau: bool
    message: str = ""
    model: JacketModel | None = field(default=None, repr=False)


@dataclass
class CapillaryRun:
    """Outcome of one capillary-ring integration."""

    ok: bool
    status: SimulationStatus
    event_name: str | None
    z_stop: float
    traj: DenseTrajectory
    uc_end: float
    nfev: int
    used_radau: bool
    ring: int
    message: str = ""


def solve_jacket(
    model: JacketModel,
    tau0: float,
    *,
    rtol: float = DEFAULT_RTOL,
    atol=JACKET_ATOL,
    cancel_event=None,
) -> JacketRun:
    """Integrate the jacket ODE from ``z=0`` with initial tension ``tau0``.

    Hot and cold zones are integrated as chained segments (matching the
    two ``ode45`` calls of the MATLAB reference).  The returned
    trajectory covers ``[0, z_stop]`` where ``z_stop < L`` on breakdown.
    """
    y0 = np.array(
        [tau0, model.w0, model.R0**2 * model.w0, model.T0], dtype=float
    )
    rhs = _checked_rhs(model.rhs, cancel_event)
    events = model.events()
    nfev = 0
    used_radau = False

    sol, fallback = _integrate_segment(
        rhs, events, 0.0, model.Lfour, y0, rtol, atol, cancel_event
    )
    nfev += sol.nfev
    used_radau |= fallback
    pieces = [(float(sol.t[-1]), sol)]
    ev = _triggered_event(sol)
    last = sol

    if ev is None and sol.status == 0:
        sol, fallback = _integrate_segment(
            rhs,
            events,
            model.Lfour,
            model.L,
            last.y[:, -1],
            rtol,
            atol,
            cancel_event,
        )
        nfev += sol.nfev
        used_radau |= fallback
        pieces.append((float(sol.t[-1]), sol))
        ev = _triggered_event(sol)
        last = sol

    if ev is not None:
        status, event_name = _JACKET_EVENT_STATUSES[ev]
    elif last.status == -1:
        status, event_name = SimulationStatus.DIVERGED, None
    else:
        status, event_name = SimulationStatus.SUCCESS, None
    z_stop = float(last.t[-1])
    ok = ev is None and last.status == 0
    traj = DenseTrajectory(pieces, n_state=4)
    message = "" if ok else (last.message if last.status == -1 else "")
    return JacketRun(
        ok=ok,
        status=status,
        event_name=event_name,
        z_stop=z_stop,
        traj=traj,
        y_end=np.asarray(last.y[:, -1], dtype=float),
        nfev=nfev,
        used_radau=used_radau,
        message=message,
        model=model,
    )


def solve_capillary(
    model: CapillaryModel,
    *,
    rtol: float = DEFAULT_RTOL,
    atol=CAPILLARY_ATOL,
    cancel_event=None,
) -> CapillaryRun:
    """Integrate one capillary ring along the cached jacket trajectory."""
    if model.z_max <= 0.0:
        return CapillaryRun(
            ok=True,
            status=SimulationStatus.SUCCESS,
            event_name=None,
            z_stop=0.0,
            traj=DenseTrajectory([(0.0, _ZeroSolution(model.Rc0**2 * model.w0))], 1),
            uc_end=model.Rc0**2 * model.w0,
            nfev=0,
            used_radau=False,
            ring=model.ring_index,
        )
    y0 = np.array([model.Rc0**2 * model.w0], dtype=float)
    rhs = _checked_rhs(model.rhs, cancel_event)
    events = model.events()
    if model.z_max > model.Lfour > 0.0:
        # The driving (w(z), T(z)) are kinked at the hot/cold boundary;
        # chain the capillary integration there exactly like the jacket
        # solver so no step straddles the discontinuity.
        sol, fallback = _integrate_segment(
            rhs, events, 0.0, model.Lfour, y0, rtol, atol, cancel_event
        )
        nfev = int(sol.nfev)
        used_radau = fallback
        if _triggered_event(sol) is None and sol.status == 0:
            sol2, fallback = _integrate_segment(
                rhs,
                events,
                model.Lfour,
                model.z_max,
                sol.y[:, -1],
                rtol,
                atol,
                cancel_event,
            )
            nfev += int(sol2.nfev)
            used_radau |= fallback
            pieces = [(float(sol.t[-1]), sol), (float(sol2.t[-1]), sol2)]
            last = sol2
        else:
            pieces = [(float(sol.t[-1]), sol)]
            last = sol
    else:
        sol, fallback = _integrate_segment(
            rhs, events, 0.0, model.z_max, y0, rtol, atol, cancel_event
        )
        nfev = int(sol.nfev)
        used_radau = fallback
        pieces = [(float(sol.t[-1]), sol)]
        last = sol
    ev = _triggered_event(last)
    if ev is not None:
        status, event_name = _CAPILLARY_EVENT_STATUSES[ev]
    elif last.status == -1:
        status, event_name = SimulationStatus.DIVERGED, None
    else:
        status, event_name = SimulationStatus.SUCCESS, None
    z_stop = float(last.t[-1])
    traj = DenseTrajectory(pieces, n_state=1)
    return CapillaryRun(
        ok=ev is None and last.status == 0,
        status=status,
        event_name=event_name,
        z_stop=z_stop,
        traj=traj,
        uc_end=float(last.y[0, -1]),
        nfev=nfev,
        used_radau=used_radau,
        ring=model.ring_index,
        message="" if last.status != -1 else last.message,
    )


class _ZeroSolution:
    """Trivial dense solution holding a constant 1-state value."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def sol(self, z):
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        return np.full((1, z_arr.size), self.value)


# --------------------------------------------------------------------------
# Uniform resampling & result assembly
# --------------------------------------------------------------------------


def uniform_grid(z_stop: float, dz: float = DEFAULT_DZ) -> np.ndarray:
    """Uniform grid on ``[0, z_stop]`` with spacing ``<= dz`` (>= 2 pts)."""
    z_stop = float(z_stop)
    if not math.isfinite(z_stop) or z_stop <= 0.0:
        return np.zeros(1)
    n = max(math.ceil(z_stop / dz - 1e-9), 1)
    return np.linspace(0.0, z_stop, n + 1)


def assemble_simulation_result(
    inputs: SimulationInputs,
    jacket_run: JacketRun,
    capillary_runs: Sequence[CapillaryRun],
    *,
    dz: float = DEFAULT_DZ,
    tau0: float = 0.0,
    elapsed_s: float = 0.0,
    diagnostics=None,
    status_override: SimulationStatus | None = None,
):
    """Build a :class:`SimulationResult` from a jacket + capillary runs.

    On capillary failure the partial trajectories are truncated at the
    earliest breakdown position so every array covers the same
    ``[0, z_stop]`` span.  ``status_override`` forces the reported
    status (e.g. :attr:`SimulationStatus.NO_DRAW`) when the ODEs
    integrated cleanly but the outcome is still not a valid draw.
    """
    from src.models import FailureDiagnostics, SimulationResult

    status = SimulationStatus.SUCCESS
    z_stop = jacket_run.z_stop
    ring = None
    event_name = jacket_run.event_name
    if not jacket_run.ok:
        status = jacket_run.status
    else:
        for i, cap in enumerate(capillary_runs):
            if not cap.ok:
                status = cap.status
                z_stop = min(z_stop, cap.z_stop)
                ring = i + 1
                event_name = cap.event_name
                break
    if status_override is not None:
        status = status_override

    z = uniform_grid(z_stop, dz)
    y = jacket_run.traj(z)
    tau, w, uJ, T = y[0], y[1], y[2], y[3]
    dJ0 = inputs.delta_J0
    R = np.sqrt(np.maximum(uJ / w, 0.0))
    r = np.sqrt(np.maximum((uJ - dJ0) / w, 0.0))
    Ta = (
        jacket_run.model.ambient_array(z)
        if jacket_run.model is not None
        else np.full_like(z, inputs.Text_C + 273.15)
    )
    Rc_list = []
    rc_list = []
    for i, cap in enumerate(capillary_runs):
        uc = cap.traj(z)[0]
        dc0 = inputs.delta_c0(i)
        Rc_list.append(np.sqrt(np.maximum(uc / w, 0.0)))
        rc_list.append(np.sqrt(np.maximum((uc - dc0) / w, 0.0)))
    if diagnostics is None:
        if status == SimulationStatus.SUCCESS:
            diagnostics = None
        else:
            diagnostics = FailureDiagnostics.from_status(
                status, z_stop, ring=ring, event_name=event_name
            )
    return SimulationResult(
        status=status,
        inputs=inputs,
        z=z,
        tau=tau,
        w=w,
        T=T,
        Ta=Ta,
        R=R,
        r=r,
        Rc=tuple(Rc_list),
        rc=tuple(rc_list),
        tau0=tau0,
        diagnostics=diagnostics,
        elapsed_s=elapsed_s,
    )


# --------------------------------------------------------------------------
# Unreduced (5 + 2*N) transcription for parity testing
# --------------------------------------------------------------------------


def unreduced_atol(n_nested: int):
    """Absolute tolerances of the unreduced ``(5 + 2 N)`` system."""
    return np.array(
        [1e-12, 1e-15, 1e-20, 1e-20, 1e-9] + [1e-22] * (2 * n_nested)
    )


def rhs_unreduced(
    inputs: SimulationInputs, materials: MaterialSet, delta_p: Sequence
):
    """RHS of the unreduced ``(5 + 2*N_nested)`` system.

    Faithful transcription of ``RK_perso7/9/11.m`` (with the ``273.15``
    K offsets and the ``viscosity_3`` ring-3 fix).  State layout:
    ``[tau, w, R2w, r2w, T, Rc2w_1, rc2w_1, ..., Rc2w_N, rc2w_N]``
    with ``d(r2w)/dz = d(R2w)/dz`` and ``d(rc2w_i)/dz = d(Rc2w_i)/dz``.
    """
    model = JacketModel(inputs, materials)
    dPs = tuple(float(d) for d in delta_p)
    n_nested = inputs.nb_nested
    if len(dPs) != n_nested:
        raise ValueError(
            f"expected {n_nested} pressure differences, got {len(dPs)}"
        )
    Pcore = model.Pcore
    g = model.g
    sigma = model.sigma
    N_heat = model.N_heat
    w_floor = model._w_floor
    R2_floor = model._R2_floor
    r2_floor = model._r2_floor
    floors = []
    for ring in inputs.rings:
        floors.append(
            (1e-16 * ring.Rc0**2, 1e-16 * ring.rc0**2)
        )

    def rhs(z, y):
        try:
            tau = float(y[0])
            w = float(y[1])
            R2w = float(y[2])
            r2w = float(y[3])
            T = float(y[4])
            w = max(w, w_floor)
            # ---- jacket (RK_perso.m transcription) ----
            R2 = R2w / w
            r2 = r2w / w
            R2 = max(R2, R2_floor)
            r2 = max(r2, r2_floor)
            R = math.sqrt(R2)
            r = math.sqrt(r2)
            R2 = R * R
            r2 = r * r
            mu = materials.viscosity(T)
            gam = materials.gamma(T)
            rho = materials.rho(T)
            cp = materials.cp(T)
            area = R2 - r2
            dwdz = tau / (3.0 * math.pi * mu * area)
            dR2w = (
                Pcore * r2 * R2 - gam * r * R * (r + R)
            ) / (mu * area)
            dtaudz = (
                rho * math.pi * area * (w * dwdz - g)
                - gam * math.pi * (dR2w - R2 * dwdz) / (2.0 * R * w)
                - gam * math.pi * (dR2w - r2 * dwdz) / (2.0 * r * w)
            )
            Ta = model.ambient(z)
            dTdz = (
                R * N_heat * (Ta - T)
                + R
                * sigma
                * (materials.alpha(Ta) * Ta**4 - materials.alpha(T) * T**4)
            ) * 2.0 / (area * rho * cp * w)
            out = [dtaudz, dwdz, dR2w, dR2w, dTdz]
            # ---- capillaries (RK_perso7/9/11 transcription) ----
            for i in range(n_nested):
                Rc2w = float(y[5 + 2 * i])
                rc2w = float(y[6 + 2 * i])
                Rc2 = Rc2w / w
                rc2 = rc2w / w
                Rc2_floor_i, rc2_floor_i = floors[i]
                Rc2 = max(Rc2, Rc2_floor_i)
                rc2 = max(rc2, rc2_floor_i)
                Rc = math.sqrt(Rc2)
                rc = math.sqrt(rc2)
                Rc2 = Rc * Rc
                rc2 = rc * rc
                mu_i = materials.rings[i].viscosity(T)
                gam_i = materials.rings[i].gamma(T)
                duc = (
                    dPs[i] * rc2 * Rc2 - gam_i * rc * Rc * (rc + Rc)
                ) / (mu_i * (Rc2 - rc2))
                out.append(duc)
                out.append(duc)
            return out
        except (ArithmeticError, ValueError):
            return [_HUGE] * (5 + 2 * n_nested)

    return rhs
