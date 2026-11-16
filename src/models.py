"""Typed data models, SI converters, and derived optimization targets.

This module holds every user-facing data structure of the HARF-drawing
core engine (a Python port of the MATLAB ``HARF_drawing.mlapp`` v1.5
application):

* :class:`RingSpec` / :class:`SimulationInputs` -- a fully validated,
  SI-unit description of one drawing simulation.
* :class:`OptimizationInputs` -- furnace/pressure search windows plus
  the final-fiber targets ``(t_f, D_cf1, D_core)``.
* :class:`DerivedTargets` / :func:`compute_derived_targets` -- the
  nested target geometry rules extracted from the MATLAB
  ``RunButton_2Pushed`` callback (``Rcf2``/``rcf2``, ``Rcf3``/``rcf3``).
* :class:`SimulationStatus`, :class:`FailureDiagnostics`,
  :class:`SimulationResult`, :class:`OptimizationResult` -- the
  crash-proof structured outputs returned by the solvers.

Unit conventions (SI unless suffixed): lengths in ``m``, speeds in
``m/s``, pressures in ``Pa``, tensions in ``N``, temperatures in ``K``
except GUI-facing fields which carry an explicit ``_C`` suffix
(degrees Celsius).  Time is not integrated directly: the axial
coordinate ``z`` in ``m`` is the independent variable.
"""


from __future__ import annotations

import enum
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "DerivedTargets",
    "FailureDiagnostics",
    "InputError",
    "OptimizationInputs",
    "OptimizationResult",
    "RingSpec",
    "SimulationInputs",
    "SimulationResult",
    "SimulationStatus",
    "TargetError",
    "celsius_to_kelvin",
    "compute_derived_targets",
    "default_optimization_inputs",
    "default_simulation_inputs",
    "grams_to_newtons",
    "mbar_to_pa",
    "mm_per_min_to_m_per_s",
    "mm_to_m",
    "r_contact_radius",
    "um_to_m",
]

# --------------------------------------------------------------------------
# Unit converters (GUI units -> SI)
# --------------------------------------------------------------------------


def mm_to_m(x: float) -> float:
    """Millimetres to metres."""
    return float(x) * 1e-3


def um_to_m(x: float) -> float:
    """Micrometres to metres."""
    return float(x) * 1e-6


def mbar_to_pa(x: float) -> float:
    """Millibars to pascals."""
    return float(x) * 100.0


def grams_to_newtons(x: float) -> float:
    """Grams-force to newtons."""
    return float(x) / 1000.0 * 9.81


def mm_per_min_to_m_per_s(x: float) -> float:
    """Millimetres per minute to metres per second."""
    return float(x) * 1e-3 / 60.0


def celsius_to_kelvin(x: float) -> float:
    """Degrees Celsius to kelvin (exact 273.15 offset, MATLAB-bug fixed)."""
    return float(x) + 273.15


def r_contact_radius(r: float | np.ndarray, n: int) -> float | np.ndarray:
    """Contact radius of ``n`` capillaries packed inside bore radius ``r``."""
    s = math.sin(math.pi / n)
    return r * s / (1.0 + s)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class TargetError(ValueError):
    """Raised when the optimization targets are geometrically impossible."""


class InputError(ValueError):
    """Raised when simulation inputs are inconsistent (geometry, units...)."""


# --------------------------------------------------------------------------
# v1.5 defaults extracted from reference/HARF_drawing.mlapp
# --------------------------------------------------------------------------

DEFAULT_FURNACE_PROFILE = "exp(-(z-0.04)^2/2/0.06^2)"
DEFAULT_RHO = "2200"
DEFAULT_GAMMA = "0.3"
DEFAULT_CP = (
    "-1.9202379E-11*(9*min(max(310.75,T),1669.67)/5-459.4)^4"
    "+8.9608033E-8*(9*min(max(310.75,T),1669.67)/5-459.4)^3"
    "-2.0008399E-4*(9*min(max(310.75,T),1669.67)/5-459.4)^2"
    "+0.59235423*(9*min(max(310.75,T),1669.67)/5-459.4)+6.2175046E2"
)
DEFAULT_VISCOSITY = "10^(-7.946)*exp(66337/T)"
DEFAULT_ALPHA = "0.885"

# Jacket preform and capillary-ring dimensions are *diameters* in mm
# (the GUI convention); `default_simulation_inputs` and
# `default_ring_specs` convert them to radii with a `/ 2.0`.
DEFAULT_D0_MM = 8.3556
DEFAULT_d0_MM = 2.4
DEFAULT_DC0_MM = 0.48
DEFAULT_dc0_MM = 0.3936
DEFAULT_N_CAPILLARIES = 2
DEFAULT_FEED_MM_MIN = 4.0
DEFAULT_TENSION_G = 84.0
DEFAULT_L_M = 0.1
DEFAULT_LFOUR_M = 0.065
DEFAULT_TMAX_C = 1806.9
DEFAULT_PCORE_MBAR = 15.6
DEFAULT_PCAP_MBAR = 63.4
DEFAULT_DELTA_T0_K = 500.0
DEFAULT_TEXT_C = 20.0
DEFAULT_SIGMA = 5.67e-8
DEFAULT_N_HEAT = 100.0
DEFAULT_G = 9.81

DEFAULT_OPTIM_TF_UM = 3.5
DEFAULT_OPTIM_DCF1_UM = 110.0
DEFAULT_OPTIM_DCORE_UM = 156.0
DEFAULT_OPTIM_TMAX_MIN_C = 1750.0
DEFAULT_OPTIM_TMAX_MAX_C = 1850.0
DEFAULT_OPTIM_PCORE_MIN_MBAR = 10.0
DEFAULT_OPTIM_PCORE_MAX_MBAR = 20.0
DEFAULT_OPTIM_PCAP_MIN_MBAR = 50.0
DEFAULT_OPTIM_PCAP_MAX_MBAR = 150.0

MAX_NESTED_RINGS = 3


# --------------------------------------------------------------------------
# Input models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RingSpec:
    """Geometry and material laws of one nested capillary ring.

    ``Rc0``/``rc0`` are the preform outer/inner (bore) radii in metres;
    ``gamma_expr``/``viscosity_expr`` are user-editable expressions of
    the temperature ``T`` in kelvin.
    """

    Rc0: float
    rc0: float
    gamma_expr: str = DEFAULT_GAMMA
    viscosity_expr: str = DEFAULT_VISCOSITY


@dataclass(frozen=True)
class SimulationInputs:
    """Complete, validated description of one drawing simulation (SI).

    Field names carry explicit units where non-SI (``_C`` for degrees
    Celsius).  ``Pcap`` holds the *absolute* capillary pressures in Pa
    (one per nested ring), exactly like the GUI fields; the capillary
    pressure differences are ``Pcap[0] - Pcore``, ``Pcap[1] - Pcap[0]``,
    ... and are computed by the solvers.
    """

    # Jacket / preform geometry [m]
    r0: float
    R0: float
    # Nested capillary rings (1..3 entries)
    rings: tuple[RingSpec, ...]
    # Number of capillaries arranged around the core (packing geometry)
    n_capillaries: int = DEFAULT_N_CAPILLARIES

    # Process conditions
    w0: float = mm_per_min_to_m_per_s(DEFAULT_FEED_MM_MIN)
    tension: float = grams_to_newtons(DEFAULT_TENSION_G)
    L: float = DEFAULT_L_M
    Lfour: float = DEFAULT_LFOUR_M
    Tmax_C: float = DEFAULT_TMAX_C
    Pcore: float = mbar_to_pa(DEFAULT_PCORE_MBAR)
    Pcap: tuple[float, ...] = (mbar_to_pa(DEFAULT_PCAP_MBAR),)

    # Thermal environment
    dT0: float = DEFAULT_DELTA_T0_K
    Text_C: float = DEFAULT_TEXT_C
    sigma: float = DEFAULT_SIGMA
    N_heat: float = DEFAULT_N_HEAT
    g: float = DEFAULT_G

    # Material laws (expressions of T [K], except the profile of z [m])
    furnace_profile_expr: str = DEFAULT_FURNACE_PROFILE
    rho_expr: str = DEFAULT_RHO
    gamma_expr: str = DEFAULT_GAMMA
    cp_expr: str = DEFAULT_CP
    viscosity_expr: str = DEFAULT_VISCOSITY
    alpha_expr: str = DEFAULT_ALPHA

    def validate(self) -> None:
        """Raise :class:`InputError` on inconsistent geometry/parameters."""
        if not (self.R0 > self.r0 > 0.0):
            raise InputError(
                f"Jacket radii must satisfy R0 > r0 > 0 "
                f"(got R0={self.R0}, r0={self.r0})."
            )
        if not (0.0 < self.Lfour < self.L):
            raise InputError(
                f"Furnace length must lie inside (0, L) "
                f"(got Lfour={self.Lfour}, L={self.L})."
            )
        if self.w0 <= 0.0:
            raise InputError(f"Feed speed must be positive (got {self.w0}).")
        if self.tension <= 0.0:
            raise InputError(
                f"Draw tension must be positive (got {self.tension})."
            )
        if not 1 <= len(self.rings) <= MAX_NESTED_RINGS:
            raise InputError(
                f"Between 1 and {MAX_NESTED_RINGS} nested rings are "
                f"supported (got {len(self.rings)})."
            )
        if len(self.Pcap) != len(self.rings):
            raise InputError(
                f"Pcap must provide one pressure per nested ring "
                f"(got {len(self.Pcap)} pressures for "
                f"{len(self.rings)} rings)."
            )
        if self.n_capillaries < 1:
            raise InputError(
                f"At least one capillary is required "
                f"(got {self.n_capillaries})."
            )
        for i, ring in enumerate(self.rings):
            if not (ring.Rc0 > ring.rc0 > 0.0):
                raise InputError(
                    f"Ring {i + 1} radii must satisfy Rc0 > rc0 > 0 "
                    f"(got Rc0={ring.Rc0}, rc0={ring.rc0})."
                )
            if i > 0:
                prev = self.rings[i - 1]
                if ring.Rc0 >= prev.rc0:
                    raise InputError(
                        f"Ring {i + 1} outer radius must be smaller than "
                        f"ring {i} bore radius "
                        f"(Rc0[{i + 1}]={ring.Rc0} >= "
                        f"rc0[{i}]={prev.rc0})."
                    )
        rc = r_contact_radius(self.r0, self.n_capillaries)
        if self.rings[0].Rc0 > rc:
            raise InputError(
                f"Capillary 1 outer radius exceeds the contact radius "
                f"(Rc01={self.rings[0].Rc0} > Rcontact={rc})."
            )

    @property
    def nb_nested(self) -> int:
        """Number of nested capillary rings."""
        return len(self.rings)

    @property
    def delta_J0(self) -> float:
        """Jacket mass-flux invariant ``(R0^2 - r0^2) * w0`` [m^3/s]."""
        return (self.R0**2 - self.r0**2) * self.w0

    def delta_c0(self, i: int) -> float:
        """Mass-flux invariant of ring ``i`` (0-based): ``(Rc0^2-rc0^2)*w0``."""
        ring = self.rings[i]
        return (ring.Rc0**2 - ring.rc0**2) * self.w0

    def capillary_deltas(self) -> tuple[float, ...]:
        """Per-ring pressure differences ``DeltaP_i`` [Pa]."""
        out = [self.Pcap[0] - self.Pcore]
        for i in range(1, len(self.Pcap)):
            out.append(self.Pcap[i] - self.Pcap[i - 1])
        return tuple(out)

    def with_process(self, **kw) -> SimulationInputs:
        """Copy with process parameters replaced (Tmax_C, Pcore, Pcap...)."""
        return replace(self, **kw)


@dataclass(frozen=True)
class OptimizationInputs:
    """Search windows and final-fiber targets of the optimization.

    ``tf``/``Dcf1``/``Dcore`` are the target capillary wall thickness,
    ring-1 outer diameter, and core diameter in metres; the pressure
    windows bound ``Pcore`` and each absolute ``Pcap_i`` in Pa.
    """

    tf: float
    Dcf1: float
    Dcore: float
    Tmax_min_C: float = DEFAULT_OPTIM_TMAX_MIN_C
    Tmax_max_C: float = DEFAULT_OPTIM_TMAX_MAX_C
    Pcore_min: float = mbar_to_pa(DEFAULT_OPTIM_PCORE_MIN_MBAR)
    Pcore_max: float = mbar_to_pa(DEFAULT_OPTIM_PCORE_MAX_MBAR)
    Pcap_min: tuple[float, ...] = (
        mbar_to_pa(DEFAULT_OPTIM_PCAP_MIN_MBAR),
    )
    Pcap_max: tuple[float, ...] = (
        mbar_to_pa(DEFAULT_OPTIM_PCAP_MAX_MBAR),
    )

    def validate(self, n_rings: int) -> None:
        """Raise :class:`InputError` on inconsistent search windows."""
        for name in (
            "tf", "Dcf1", "Dcore", "Tmax_min_C", "Tmax_max_C",
            "Pcore_min", "Pcore_max",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise InputError(
                    f"{name} must be a scalar number "
                    f"(got {type(value).__name__})."
                )
        if self.tf <= 0.0:
            raise InputError(f"Target thickness must be positive (got {self.tf}).")
        if self.Dcf1 <= 0.0 or self.Dcore <= 0.0:
            raise InputError("Target diameters must be positive.")
        if self.Tmax_max_C <= self.Tmax_min_C:
            raise InputError(
                f"Tmax_max must exceed Tmax_min "
                f"(got [{self.Tmax_min_C}, {self.Tmax_max_C}] C)."
            )
        if self.Pcore_max <= self.Pcore_min:
            raise InputError(
                f"Pcore_max must exceed Pcore_min "
                f"(got [{self.Pcore_min}, {self.Pcore_max}] Pa)."
            )
        if len(self.Pcap_min) != n_rings or len(self.Pcap_max) != n_rings:
            raise InputError(
                "Pcap windows must provide one range per nested ring."
            )
        for i, (lo, hi) in enumerate(zip(self.Pcap_min, self.Pcap_max, strict=True)):
            if hi <= lo:
                raise InputError(
                    f"Pcap_{i + 1} window is empty ([{lo}, {hi}] Pa)."
                )


# --------------------------------------------------------------------------
# Derived optimization targets (MATLAB RunButton_2Pushed rules)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivedTargets:
    """Self-consistent final fiber geometry implied by the targets."""

    tf: float
    Rcf: float
    rcf: float
    wf: float
    rf: float
    Rf: float
    ring_targets: tuple[tuple[float, float], ...]

    @property
    def Rcf_rings(self) -> tuple[float, ...]:
        return tuple(t[0] for t in self.ring_targets)

    @property
    def rcf_rings(self) -> tuple[float, ...]:
        return tuple(t[1] for t in self.ring_targets)


def compute_derived_targets(
    inputs: SimulationInputs, opt: OptimizationInputs
) -> DerivedTargets:
    """Compute nested target geometry exactly as ``RunButton_2Pushed``.

    Ring 1: ``rcf = Rcf - tf``, ``wf = (Rc0^2-rc0^2)w0/(Rcf^2-rcf^2)``,
    ``rf = Dcore/2 + 2 Rcf``, ``Rf = sqrt(rf^2 + Delta_J0 / wf)``.
    Rings 2 and 3:
    ``Rcf_i = (w0/wf * (2 Rc0_i t0_i - t0_i^2)/tf + tf)/2`` and
    ``rcf_i = Rcf_i - tf`` with ``t0_i = Rc0_i - rc0_i``.

    Raises :class:`TargetError` when the requested geometry is
    impossible (non-positive radii or non-nested rings).
    """
    tf = float(opt.tf)
    Rcf = float(opt.Dcf1) / 2.0
    rcf = Rcf - tf
    if rcf <= 0.0:
        raise TargetError(
            f"rcf <= 0: target thickness {tf} exceeds the ring-1 target "
            f"radius {Rcf}."
        )
    wf = inputs.delta_c0(0) / (Rcf**2 - rcf**2)
    if wf <= inputs.w0:
        raise TargetError(
            "Target draw ratio of ring 1 must exceed 1 "
            f"(wf={wf} <= w0={inputs.w0})."
        )
    rf = float(opt.Dcore) / 2.0 + 2.0 * Rcf
    Rf = math.sqrt(rf**2 + inputs.delta_J0 / wf)

    ring_targets = [(Rcf, rcf)]
    for i, ring in enumerate(inputs.rings[1:], start=2):
        t0 = ring.Rc0 - ring.rc0
        Rcf_i = (inputs.w0 / wf * (2.0 * ring.Rc0 * t0 - t0**2) / tf + tf) / 2.0
        rcf_i = Rcf_i - tf
        if rcf_i <= 0.0:
            raise TargetError(
                f"rcf{i} <= 0: ring {i} cannot reach the target thickness."
            )
        prev_rcf = ring_targets[-1][1]
        if Rcf_i >= prev_rcf:
            raise TargetError(
                f"Dcf{i} >= dcf{i - 1}: nested rings must stay ordered."
            )
        ring_targets.append((Rcf_i, rcf_i))
    return DerivedTargets(
        tf=tf,
        Rcf=Rcf,
        rcf=rcf,
        wf=wf,
        rf=rf,
        Rf=Rf,
        ring_targets=tuple(ring_targets),
    )


# --------------------------------------------------------------------------
# Status / diagnostics / results
# --------------------------------------------------------------------------


class SimulationStatus(enum.Enum):
    """Outcome of a simulation or optimization run."""

    SUCCESS = "success"
    INPUT_ERROR = "input_error"
    INFEASIBLE = "infeasible"
    TENSION_LOSS = "tension_loss"
    JACKET_COLLAPSE = "jacket_collapse"
    JACKET_BLOWUP = "jacket_blowup"
    VELOCITY_BLOWUP = "velocity_blowup"
    VELOCITY_COLLAPSE = "velocity_collapse"
    NO_DRAW = "no_draw"
    CAPILLARY_COLLAPSE = "capillary_collapse"
    CAPILLARY_BLOWUP = "capillary_blowup"
    DIVERGED = "diverged"
    CANCELLED = "cancelled"


_STATUS_HELP: dict[SimulationStatus, tuple[str, str]] = {
    SimulationStatus.TENSION_LOSS: (
        "Draw tension fell below 1e-9 N before the end of the drawing zone.",
        (
            "Increase the draw tension, raise the furnace temperature, or "
            "reduce the feed speed so the preform can sustain the draw."
        ),
    ),
    SimulationStatus.JACKET_COLLAPSE: (
        "The jacket bore collapsed (bore flux r^2 w fell below 10% of "
        "its initial value).",
        (
            "Increase the core pressure Pcore or reduce the furnace "
            "temperature so surface tension cannot close the jacket."
        ),
    ),
    SimulationStatus.JACKET_BLOWUP: (
        "The jacket inflated without bound (R > 10 R0).",
        (
            "Decrease the core pressure Pcore; the internal pressure is "
            "far too high for the current viscosity."
        ),
    ),
    SimulationStatus.VELOCITY_BLOWUP: (
        "Drawing speed blew up (w > 1e5 w0).",
        (
            "Reduce the draw tension or the furnace temperature; the fiber "
            "is being drawn far too fast."
        ),
    ),
    SimulationStatus.VELOCITY_COLLAPSE: (
        "Drawing speed collapsed (w < 1e-3 w0).",
        (
            "Increase the draw tension or the furnace temperature; the "
            "fiber is not being drawn at all."
        ),
    ),
    SimulationStatus.NO_DRAW: (
        (
            "The preform left the drawing zone essentially undrawn "
            "(w(L) ~ w0): the material stayed far too viscous."
        ),
        (
            "Raise the furnace temperature Tmax or check the viscosity "
            "law; drawing requires a hot zone of much lower viscosity."
        ),
    ),
    SimulationStatus.CAPILLARY_COLLAPSE: (
        "A capillary bore collapsed (rc -> 0).",
        (
            "Increase the corresponding capillary pressure Pcap_i so the "
            "bore stays open against surface tension."
        ),
    ),
    SimulationStatus.CAPILLARY_BLOWUP: (
        "A capillary inflated without bound (Rc > 10 Rc0).",
        "Decrease the corresponding capillary pressure Pcap_i.",
    ),
    SimulationStatus.DIVERGED: (
        "The ODE integrator failed to converge.",
        (
            "Check the material laws (viscosity/temperature range) and the "
            "furnace profile; the model may be outside its validity range."
        ),
    ),
    SimulationStatus.CANCELLED: (
        "Computation cancelled by the user.",
        "No action required.",
    ),
    SimulationStatus.INFEASIBLE: (
        "No parameter inside the requested window achieves the target.",
        (
            "Widen the (Tmax, Pcore, Pcap) search windows or revise the "
            "final-fiber targets."
        ),
    ),
}


@dataclass(frozen=True)
class FailureDiagnostics:
    """Structured, actionable description of a failed run."""

    status: SimulationStatus
    z_stop: float
    message: str
    advice: str
    ring: int | None = None
    event_name: str | None = None

    @classmethod
    def from_status(
        cls,
        status: SimulationStatus,
        z_stop: float = 0.0,
        ring: int | None = None,
        event_name: str | None = None,
    ) -> FailureDiagnostics:
        """Build diagnostics with the standard message/advice texts."""
        msg, advice = _STATUS_HELP.get(
            status, ("The computation failed.", "No action required.")
        )
        return cls(
            status=status,
            z_stop=float(z_stop),
            message=msg,
            advice=advice,
            ring=ring,
            event_name=event_name,
        )


@dataclass
class SimulationResult:
    """Structured result of one direct simulation.

    All axial arrays are sampled on a uniform grid ``z`` with spacing
    ``dz`` (default 1e-5 m) covering ``[0, z_stop]``; on failure the
    arrays hold the partial trajectory up to the breakdown position.
    ``Rc``/``rc`` hold one array per nested ring.
    """

    status: SimulationStatus
    inputs: SimulationInputs
    z: np.ndarray
    tau: np.ndarray
    w: np.ndarray
    T: np.ndarray
    Ta: np.ndarray
    R: np.ndarray
    r: np.ndarray
    Rc: tuple[np.ndarray, ...]
    rc: tuple[np.ndarray, ...]
    tau0: float = 0.0
    diagnostics: FailureDiagnostics | None = None
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == SimulationStatus.SUCCESS

    @property
    def z_end(self) -> float:
        return float(self.z[-1]) if len(self.z) else 0.0

    @property
    def R_contact(self) -> np.ndarray:
        """Contact radius of the capillary packing along z [m]."""
        return r_contact_radius(self.r, self.inputs.n_capillaries)

    @property
    def D_core(self) -> np.ndarray:
        """Core diameter ``2 (r - 2 Rc1)`` along z [m]."""
        if not self.Rc:
            return 2.0 * self.r
        return 2.0 * (self.r - 2.0 * self.Rc[0])

    def finite(self) -> bool:
        """True when every axial array is finite (no NaN/Inf)."""
        arrays = [self.z, self.tau, self.w, self.T, self.Ta, self.R, self.r]
        arrays.extend(self.Rc)
        arrays.extend(self.rc)
        return all(np.all(np.isfinite(a)) for a in arrays)


@dataclass
class OptimizationResult:
    """Structured result of the (Tmax, Pcore, DeltaP_i) optimization."""

    status: SimulationStatus
    targets: DerivedTargets | None
    Tmax_C: float = 0.0
    Pcore: float = 0.0
    Pcap: tuple[float, ...] = ()
    DeltaP: tuple[float, ...] = ()
    tau0: float = 0.0
    cost: float | None = None
    result: SimulationResult | None = None
    message: str = ""
    advice: str = ""
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == SimulationStatus.SUCCESS


ProgressCallback = Callable[[str, float], None]


# --------------------------------------------------------------------------
# Default input factories (MATLAB v1.5 defaults)
# --------------------------------------------------------------------------


def default_ring_specs(nb_nested: int = 1) -> tuple[RingSpec, ...]:
    """Default per-ring geometry/material specs (nested silica rings).

    ``DEFAULT_DC0_MM``/``DEFAULT_dc0_MM`` are *diameters* in mm; the
    radii are half of them.  Ring 1 keeps the v1.5 diameters; nested
    rings ``i >= 2`` shrink geometrically by ``0.5**(i-1)`` so the
    preform stays strictly nested (``Rc0[i+1] < rc0[i]``) while
    preserving the wall ratio ``dc0/Dc0 = 0.82`` (identical placeholder
    rings would violate nesting, exactly like the MATLAB alert).
    """
    Rc0 = mm_to_m(DEFAULT_DC0_MM) / 2.0
    rc0 = mm_to_m(DEFAULT_dc0_MM) / 2.0
    return tuple(
        RingSpec(Rc0=Rc0 * scale, rc0=rc0 * scale)
        for scale in (0.5**i for i in range(nb_nested))
    )


def default_simulation_inputs(nb_nested: int = 1) -> SimulationInputs:
    """Build the default v1.5 simulation inputs (``nb_nested`` rings)."""
    nb_nested = int(nb_nested)
    if not 1 <= nb_nested <= MAX_NESTED_RINGS:
        raise InputError(
            f"nb_nested must be in 1..{MAX_NESTED_RINGS} (got {nb_nested})."
        )
    return SimulationInputs(
        r0=mm_to_m(DEFAULT_d0_MM) / 2.0,
        R0=mm_to_m(DEFAULT_D0_MM) / 2.0,
        rings=default_ring_specs(nb_nested),
        n_capillaries=DEFAULT_N_CAPILLARIES,
        w0=mm_per_min_to_m_per_s(DEFAULT_FEED_MM_MIN),
        tension=grams_to_newtons(DEFAULT_TENSION_G),
        L=DEFAULT_L_M,
        Lfour=DEFAULT_LFOUR_M,
        Tmax_C=DEFAULT_TMAX_C,
        Pcore=mbar_to_pa(DEFAULT_PCORE_MBAR),
        Pcap=(mbar_to_pa(DEFAULT_PCAP_MBAR),) * nb_nested,
        dT0=DEFAULT_DELTA_T0_K,
        Text_C=DEFAULT_TEXT_C,
        sigma=DEFAULT_SIGMA,
        N_heat=DEFAULT_N_HEAT,
        g=DEFAULT_G,
        furnace_profile_expr=DEFAULT_FURNACE_PROFILE,
        rho_expr=DEFAULT_RHO,
        gamma_expr=DEFAULT_GAMMA,
        cp_expr=DEFAULT_CP,
        viscosity_expr=DEFAULT_VISCOSITY,
        alpha_expr=DEFAULT_ALPHA,
    )


def default_optimization_inputs(nb_nested: int = 1) -> OptimizationInputs:
    """Build the default v1.5 optimization targets and windows."""
    nb_nested = int(nb_nested)
    if not 1 <= nb_nested <= MAX_NESTED_RINGS:
        raise InputError(
            f"nb_nested must be in 1..{MAX_NESTED_RINGS} (got {nb_nested})."
        )
    return OptimizationInputs(
        tf=um_to_m(DEFAULT_OPTIM_TF_UM),
        Dcf1=um_to_m(DEFAULT_OPTIM_DCF1_UM),
        Dcore=um_to_m(DEFAULT_OPTIM_DCORE_UM),
        Tmax_min_C=DEFAULT_OPTIM_TMAX_MIN_C,
        Tmax_max_C=DEFAULT_OPTIM_TMAX_MAX_C,
        Pcore_min=mbar_to_pa(DEFAULT_OPTIM_PCORE_MIN_MBAR),
        Pcore_max=mbar_to_pa(DEFAULT_OPTIM_PCORE_MAX_MBAR),
        Pcap_min=(mbar_to_pa(DEFAULT_OPTIM_PCAP_MIN_MBAR),) * nb_nested,
        Pcap_max=(mbar_to_pa(DEFAULT_OPTIM_PCAP_MAX_MBAR),) * nb_nested,
    )

