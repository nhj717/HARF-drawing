"""Cane & stack scaling engine (port of ``6Ring_Scaling.xlsx``).

Closed-form, stage-by-stage geometry mapping between the macroscopic
cane / jacket preform, the stage-1 stacked preform and the final
microstructured fiber:

* stage ``E`` -- the cane as built (rings mounted on the jacket bore),
* stage ``F`` -- the cane drawn down by the factor ``S_c``,
* stage ``G`` -- ring 1 inflated / elongated by ``S_ie``,
* stage ``H`` -- every capillary expanded by ``S_ce,i``,
* stage ``I`` -- the desired final fiber.

Two jacket <-> cane couplings are supported: ``CO_FED_VACUUM``
(area-conserved feed, the default) and ``LEGACY_EXCEL`` (the exact
algebra of the reference spreadsheet).  All lengths are SI metres
internally and are *diameters*; the GUI converts them to mm
(cane / jacket / stack) or um (fiber).  Radicand failures never raise:
they propagate ``nan`` and surface through the feasibility flags.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.models import (
    OptimizationInputs,
    SimulationInputs,
    compute_derived_targets,
    mm_to_m,
)

__all__ = [
    "CO_FED_VACUUM",
    "CouplingMode",
    "InverseSubMode",
    "LEGACY_EXCEL",
    "CaneGeometry",
    "FiberGeometry",
    "JacketGeometry",
    "PulledScalingState",
    "ScalingError",
    "ScalingFactors",
    "ScalingFeasibility",
    "ScalingPreset",
    "ScalingResult",
    "SolverHandoff",
    "StackGeometry",
    "StageGeometry",
    "auto_fit_scaling_factors",
    "cane_to_stack",
    "compute_forward_scaling",
    "get_scaling_presets",
    "inverse_jacket_od",
    "scaling_to_solver_inputs",
    "solver_inputs_to_scaling",
    "solve_inverse_cane",
    "stack_to_cane",
]


class ScalingError(ValueError):
    """Raised when a scaling computation is impossible or ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScalingError(message)


def _safe_div(num: float, den: float) -> float:
    """Division that returns ``nan`` instead of raising or overflowing."""
    try:
        value = num / den
    except ZeroDivisionError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def _safe_sqrt(x: float) -> float:
    """Square root that returns ``nan`` for negative / non-finite input."""
    return math.sqrt(x) if (math.isfinite(x) and x > 0.0) else math.nan


def _close(a: float, b: float, rtol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-15)


class CouplingMode(enum.Enum):
    """Jacket <-> cane area coupling of the draw."""

    CO_FED_VACUUM = "co_fed_vacuum"
    LEGACY_EXCEL = "legacy_excel"


CO_FED_VACUUM = CouplingMode.CO_FED_VACUUM
LEGACY_EXCEL = CouplingMode.LEGACY_EXCEL


class InverseSubMode(enum.Enum):
    """How the inverse (fiber -> cane) fixes the remaining DOF."""

    FIXED_JACKET = "fixed_jacket"
    MANUAL_SJ = "manual_sj"


# ---------------------------------------------------------------------------
# Geometry models (SI metres, diameters)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaneGeometry:
    """Cane cross-section: outer diameter, core and nested rings."""

    OD_c: float
    D_core_c: float
    d_c: tuple[float, ...]
    t_c: tuple[float, ...]
    n_cap: int = 6

    @property
    def Delta_c(self) -> tuple[float, ...]:
        """Ring outer diameters ``d_c,i + 2 t_c,i``."""
        return tuple(
            d + 2.0 * t for d, t in zip(self.d_c, self.t_c, strict=True)
        )

    @property
    def zeta_c(self) -> float:
        """Structure envelope ``D_core_c + 2 Delta_c,1``."""
        return self.D_core_c + 2.0 * self.Delta_c[0]


@dataclass(frozen=True)
class JacketGeometry:
    """Jacket outer / inner (bore) diameters."""

    OD_j: float
    ID_j: float


@dataclass(frozen=True)
class FiberGeometry:
    """Final-fiber target / achieved geometry."""

    D_f: float
    D_core_f: float
    d_f: tuple[float, ...]
    t_f: tuple[float, ...]

    @property
    def Delta_f(self) -> tuple[float, ...]:
        return tuple(
            d + 2.0 * t for d, t in zip(self.d_f, self.t_f, strict=True)
        )

    @property
    def zeta_f(self) -> float:
        return self.D_core_f + 2.0 * self.Delta_f[0]


@dataclass(frozen=True)
class ScalingFactors:
    """Draw factors of the scaling chain (dimensionless)."""

    S_j: float
    S_ie: float
    S_ce: tuple[float, ...]
    coupling: CouplingMode = CO_FED_VACUUM


@dataclass(frozen=True)
class StackGeometry:
    """Stage-1 stack geometry assembled from the cane."""

    S_stack: float
    S_ie_st: float
    S_ce_st: tuple[float, ...]
    ID_stack: float
    OD_stack: float
    d_st: tuple[float, ...]
    t_st: tuple[float, ...]

    @property
    def Delta_st(self) -> tuple[float, ...]:
        return tuple(
            d + 2.0 * t for d, t in zip(self.d_st, self.t_st, strict=True)
        )

    @property
    def zeta_st(self) -> float:
        return self.S_ie_st * self.ID_stack

    @property
    def D_core_st(self) -> float:
        return self.zeta_st - 2.0 * self.Delta_st[0]


@dataclass(frozen=True)
class StageGeometry:
    """One row of the 5-stage scaling table (SI metres, diameters)."""

    name: str
    n_cap: int
    zeta: float
    D_core: float
    d: tuple[float, ...]
    t: tuple[float, ...]
    Delta: tuple[float, ...]
    OD_cane: float
    ID_j: float
    OD_j: float
    OD_ref: float
    Gamma: tuple[float, ...]
    eta: float
    nu: tuple[float, ...]
    delta: float
    gamma_pct: float


def _build_stage(
    name: str,
    n_cap: int,
    zeta: float,
    D_core: float,
    d: tuple[float, ...],
    t: tuple[float, ...],
    Delta: tuple[float, ...],
    OD_cane: float,
    ID_j: float,
    OD_j: float,
    OD_ref: float,
) -> StageGeometry:
    """Assemble one stage and derive its diagnostic ratios."""
    Gamma = tuple(
        _safe_div(di, Di) for di, Di in zip(d, Delta, strict=True)
    )
    eta = _safe_div(d[0], D_core) if d else math.nan
    nu = tuple(
        _safe_div(Delta[i], d[i - 1]) if i > 0 else math.nan
        for i in range(len(d))
    )
    theta = math.pi / max(int(n_cap), 1)
    delta = theta * 0.0  # placeholder replaced below (sin needed)
    delta = math.sin(theta) * (D_core + Delta[0]) - Delta[0] if d else math.nan
    gamma_pct = 100.0 * _safe_div(
        Delta[0] ** 2 - d[0] ** 2, OD_ref**2 - zeta**2
    ) if d else math.nan
    return StageGeometry(
        name=name,
        n_cap=int(n_cap),
        zeta=zeta,
        D_core=D_core,
        d=d,
        t=t,
        Delta=Delta,
        OD_cane=OD_cane,
        ID_j=ID_j,
        OD_j=OD_j,
        OD_ref=OD_ref,
        Gamma=Gamma,
        eta=eta,
        nu=nu,
        delta=delta,
        gamma_pct=gamma_pct,
    )


_FLAG_MESSAGES: tuple[tuple[str, str], ...] = (
    (
        "jacket_valid",
        "Jacket must satisfy OD_j > ID_j > 0.",
    ),
    (
        "cane_wall_positive",
        "Cane wall is negative: OD_c must exceed the structure "
        "envelope zeta_E.",
    ),
    (
        "factors_positive",
        "All scaling factors (S_j, S_ie, S_ce) must be positive.",
    ),
    (
        "fiber_walls_positive",
        "A final-fiber capillary wall t_H is non-positive.",
    ),
    (
        "core_positive",
        "A stage core diameter D_core is non-positive.",
    ),
    (
        "capillary_gap_positive",
        "Capillaries crowd the core: inter-capillary gap <= 0.",
    ),
    (
        "jacket_area_conserved",
        "Jacket silica area is not conserved across the draw.",
    ),
    (
        "ring_areas_conserved",
        "Capillary ring silica areas are not conserved.",
    ),
)


@dataclass(frozen=True)
class ScalingFeasibility:
    """Diagnostic flags of one forward scaling run."""

    jacket_valid: bool
    cane_wall_positive: bool
    factors_positive: bool
    fiber_walls_positive: bool
    core_positive: bool
    capillary_gap_positive: bool
    jacket_area_conserved: bool
    ring_areas_conserved: bool

    @property
    def all_ok(self) -> bool:
        return (
            self.jacket_valid
            and self.cane_wall_positive
            and self.factors_positive
            and self.fiber_walls_positive
            and self.core_positive
            and self.capillary_gap_positive
            and self.jacket_area_conserved
            and self.ring_areas_conserved
        )

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            message
            for field, message in _FLAG_MESSAGES
            if not getattr(self, field)
        )


@dataclass(frozen=True)
class ScalingResult:
    """Full 5-stage scaling table plus feasibility diagnostics."""

    stages: dict[str, StageGeometry]
    fiber: FiberGeometry
    achieved_fiber: FiberGeometry
    factors: ScalingFactors
    S_c: float
    n_cap: int
    feasibility: ScalingFeasibility


# ---------------------------------------------------------------------------
# Forward scaling (E -> F -> G -> H, desired I)
# ---------------------------------------------------------------------------


def compute_forward_scaling(
    cane: CaneGeometry,
    jacket: JacketGeometry,
    factors: ScalingFactors,
    desired_fiber: FiberGeometry | None = None,
) -> ScalingResult:
    """Run the 5-stage forward chain; radicand failures become ``nan``.

    Raises :class:`ScalingError` only for structurally invalid inputs
    (ring-count mismatch, non-positive jacket / factors / ``OD_c``),
    never for negative radicands -- those surface as ``nan`` stages
    and red feasibility flags instead.
    """
    nb = len(cane.d_c)
    _require(
        len(cane.t_c) == nb and len(factors.S_ce) == nb,
        f"ring-count mismatch: {nb} cane bores, "
        f"{len(cane.t_c)} thicknesses, {len(factors.S_ce)} S_ce factors.",
    )
    _require(
        jacket.OD_j > jacket.ID_j > 0.0,
        f"Jacket must satisfy OD_j > ID_j > 0 "
        f"(got OD_j={jacket.OD_j}, ID_j={jacket.ID_j}).",
    )
    _require(cane.OD_c > 0.0, f"OD_c must be positive (got {cane.OD_c}).")
    _require(factors.S_j > 0.0, f"S_j must be positive (got {factors.S_j}).")
    _require(
        factors.S_ie > 0.0, f"S_ie must be positive (got {factors.S_ie})."
    )
    for i, s in enumerate(factors.S_ce):
        _require(s > 0.0, f"S_ce[{i + 1}] must be positive (got {s}).")

    if factors.coupling is CouplingMode.LEGACY_EXCEL:
        S_c = factors.S_j * cane.OD_c / jacket.ID_j
    else:
        S_c = factors.S_j
    _require(S_c > 0.0, f"Cane draw ratio S_c must be positive (got {S_c}).")

    # Stage E: the cane as built.
    d_E = tuple(float(x) for x in cane.d_c)
    t_E = tuple(float(x) for x in cane.t_c)
    Delta_E = tuple(
        d + 2.0 * t for d, t in zip(d_E, t_E, strict=True)
    )
    zeta_E = cane.D_core_c + 2.0 * Delta_E[0]
    stage_E = _build_stage(
        "E", cane.n_cap, zeta_E, cane.D_core_c, d_E, t_E, Delta_E,
        cane.OD_c, jacket.ID_j, jacket.OD_j, cane.OD_c,
    )

    # Stage F: draw the cane down by S_c.
    d_F = tuple(x / S_c for x in d_E)
    t_F = tuple(x / S_c for x in t_E)
    Delta_F = tuple(x / S_c for x in Delta_E)
    zeta_F = zeta_E / S_c
    D_core_F = cane.D_core_c / S_c
    OD_cane_F = cane.OD_c / S_c
    if factors.coupling is CouplingMode.LEGACY_EXCEL:
        OD_j_F = jacket.OD_j / factors.S_j
    else:
        d0_eff = _safe_sqrt(
            jacket.OD_j**2 - jacket.ID_j**2 + cane.OD_c**2
        )
        OD_j_F = _safe_div(d0_eff, factors.S_j)
    stage_F = _build_stage(
        "F", cane.n_cap, zeta_F, D_core_F, d_F, t_F, Delta_F,
        OD_cane_F, OD_cane_F, OD_j_F, OD_cane_F,
    )

    # Stage G: inflate / elongate ring 1 by S_ie.
    zeta_G = factors.S_ie * zeta_F
    D_core_G = zeta_G - 2.0 * Delta_F[0]
    OD_cane_G = _safe_sqrt(OD_cane_F**2 - zeta_F**2 + zeta_G**2)
    OD_j_G = _safe_sqrt(OD_j_F**2 - zeta_F**2 + zeta_G**2)
    stage_G = _build_stage(
        "G", cane.n_cap, zeta_G, D_core_G, d_F, t_F, Delta_F,
        OD_cane_G, OD_cane_G, OD_j_G, OD_cane_G,
    )

    # Stage H: expand every capillary by S_ce,i.
    d_H = tuple(
        s * df for s, df in zip(factors.S_ce, d_F, strict=True)
    )
    Delta_H = tuple(
        _safe_sqrt(Df**2 - df**2 + dh**2)
        for Df, df, dh in zip(Delta_F, d_F, d_H, strict=True)
    )
    t_H = tuple(
        (Dh - dh) / 2.0 for Dh, dh in zip(Delta_H, d_H, strict=True)
    )
    D_core_H = zeta_G - 2.0 * Delta_H[0]
    stage_H = _build_stage(
        "H", cane.n_cap, zeta_G, D_core_H, d_H, t_H, Delta_H,
        OD_cane_G, OD_cane_G, OD_j_G, OD_cane_G,
    )

    achieved = FiberGeometry(
        D_f=stage_H.OD_j, D_core_f=stage_H.D_core,
        d_f=stage_H.d, t_f=stage_H.t,
    )
    fiber = desired_fiber if desired_fiber is not None else achieved

    # Stage I: the (desired or achieved) final fiber.
    d_I = tuple(float(x) for x in fiber.d_f)
    t_I = tuple(float(x) for x in fiber.t_f)
    Delta_I = tuple(
        d + 2.0 * t for d, t in zip(d_I, t_I, strict=True)
    )
    zeta_I = fiber.D_core_f + 2.0 * Delta_I[0]
    stage_I = _build_stage(
        "I", cane.n_cap, zeta_I, fiber.D_core_f, d_I, t_I, Delta_I,
        math.nan, math.nan, fiber.D_f, fiber.D_f,
    )

    if factors.coupling is CouplingMode.CO_FED_VACUUM:
        rhs = (jacket.OD_j**2 - jacket.ID_j**2) + (
            cane.OD_c**2 - zeta_E**2
        )
        jacket_area = all(
            math.isfinite(od)
            and _close(factors.S_j**2 * (od**2 - ze**2), rhs)
            for od, ze in (
                (OD_j_F, zeta_F), (OD_j_G, zeta_G), (stage_H.OD_j, zeta_G)
            )
        )
    else:
        jacket_area = True
    feasibility = ScalingFeasibility(
        jacket_valid=jacket.OD_j > jacket.ID_j > 0.0,
        cane_wall_positive=cane.OD_c > zeta_E,
        factors_positive=(
            factors.S_j > 0.0
            and factors.S_ie > 0.0
            and all(s > 0.0 for s in factors.S_ce)
        ),
        fiber_walls_positive=all(v > 0.0 for v in t_H),
        core_positive=all(
            v > 0.0
            for v in (
                cane.D_core_c, D_core_F, D_core_G, D_core_H,
                fiber.D_core_f,
            )
        ),
        capillary_gap_positive=(
            math.isfinite(stage_G.delta) and stage_G.delta >= 0.0
        ),
        jacket_area_conserved=jacket_area,
        ring_areas_conserved=all(
            math.isfinite(dh)
            and _close(S_c**2 * (Dh**2 - dh**2), De**2 - de**2)
            for Dh, dh, De, de in zip(
                Delta_H, d_H, Delta_E, d_E, strict=True
            )
        ),
    )
    return ScalingResult(
        stages={
            "E": stage_E, "F": stage_F, "G": stage_G,
            "H": stage_H, "I": stage_I,
        },
        fiber=fiber,
        achieved_fiber=achieved,
        factors=ScalingFactors(
            S_j=factors.S_j, S_ie=factors.S_ie,
            S_ce=tuple(factors.S_ce), coupling=factors.coupling,
        ),
        S_c=S_c,
        n_cap=int(cane.n_cap),
        feasibility=feasibility,
    )


# ---------------------------------------------------------------------------
# Auto-fit (cane + fiber -> factors)
# ---------------------------------------------------------------------------


def auto_fit_scaling_factors(
    cane: CaneGeometry,
    fiber: FiberGeometry,
    jacket: JacketGeometry,
    coupling: CouplingMode = CO_FED_VACUUM,
) -> ScalingFactors:
    """Recover the scaling factors that map ``cane`` onto ``fiber``."""
    _require(
        len(cane.d_c) == len(fiber.d_f) == len(fiber.t_f),
        "ring-count mismatch between cane and fiber.",
    )
    area_c = cane.Delta_c[0] ** 2 - cane.d_c[0] ** 2
    area_f = fiber.Delta_f[0] ** 2 - fiber.d_f[0] ** 2
    _require(
        area_c > 0.0,
        f"Cane ring-1 silica area must be positive (got {area_c}).",
    )
    _require(
        area_f > 0.0,
        f"Fiber ring-1 silica area must be positive (got {area_f}).",
    )
    for i, dc in enumerate(cane.d_c):
        _require(dc > 0.0, f"cane bore d_c[{i + 1}] must be positive.")
    S_c = math.sqrt(area_c / area_f)
    if coupling is CouplingMode.LEGACY_EXCEL:
        _require(
            jacket.ID_j > 0.0 and cane.OD_c > 0.0,
            "LEGACY_EXCEL auto-fit needs ID_j > 0 and OD_c > 0.",
        )
        S_j = S_c * jacket.ID_j / cane.OD_c
    else:
        S_j = S_c
    zeta_c = cane.zeta_c
    _require(
        zeta_c > 0.0,
        f"Cane structure envelope zeta_c must be positive (got {zeta_c}).",
    )
    return ScalingFactors(
        S_j=S_j,
        S_ie=S_c * fiber.zeta_f / zeta_c,
        S_ce=tuple(
            S_c * df / dc
            for df, dc in zip(fiber.d_f, cane.d_c, strict=True)
        ),
        coupling=coupling,
    )


# ---------------------------------------------------------------------------
# Inverse scaling (fiber -> cane)
# ---------------------------------------------------------------------------


def _inverse_B(fiber: FiberGeometry, S_ie: float) -> float:
    """``B = D_f^2 - (1 - 1/S_ie^2) zeta_f^2`` (== OD_j,F squared)."""
    return fiber.D_f**2 - (1.0 - 1.0 / S_ie**2) * fiber.zeta_f**2


def solve_inverse_cane(
    fiber: FiberGeometry,
    jacket: JacketGeometry,
    S_ie: float,
    S_ce: Sequence[float],
    sub_mode: InverseSubMode,
    coupling: CouplingMode = CO_FED_VACUUM,
    OD_c: float | None = None,
    t_sleeve: float | None = None,
    S_j: float | None = None,
    n_cap: int = 6,
) -> tuple[CaneGeometry, float]:
    """Solve the cane geometry from a fiber target.

    Returns ``(cane, S_c)``.  Sub-mode ``FIXED_JACKET`` needs ``OD_c``;
    ``MANUAL_SJ`` needs ``S_j`` plus ``t_sleeve`` (preferred) or
    ``OD_c``.
    """
    nb = len(fiber.d_f)
    _require(
        len(fiber.t_f) == nb and len(S_ce) == nb,
        f"ring-count mismatch: {nb} fiber bores, "
        f"{len(fiber.t_f)} thicknesses, {len(S_ce)} S_ce factors.",
    )
    _require(S_ie > 0.0, f"S_ie must be positive (got {S_ie}).")
    for i, s in enumerate(S_ce):
        _require(s > 0.0, f"S_ce[{i + 1}] must be positive (got {s}).")
    _require(jacket.ID_j > 0.0, "Jacket ID_j must be positive.")
    B = _inverse_B(fiber, S_ie)
    _require(
        B > 0.0,
        f"Inverse infeasible: B = D_f^2 - (1 - 1/S_ie^2) zeta_f^2 "
        f"= {B:.6g} <= 0.",
    )

    if sub_mode is InverseSubMode.FIXED_JACKET:
        _require(
            OD_c is not None,
            "FIXED_JACKET requires the cane outer diameter OD_c.",
        )
        if coupling is CouplingMode.LEGACY_EXCEL:
            S_j_val = jacket.OD_j / math.sqrt(B)
            S_c = S_j_val * float(OD_c) / jacket.ID_j
        else:
            radicand = (
                jacket.OD_j**2 - jacket.ID_j**2 + float(OD_c) ** 2
            )
            _require(
                radicand > 0.0,
                f"FIXED_JACKET infeasible: OD_j^2 - ID_j^2 + OD_c^2 "
                f"= {radicand:.6g} <= 0.",
            )
            S_c = math.sqrt(radicand / B)
        OD_c_val = float(OD_c)
    else:
        _require(
            S_j is not None and float(S_j) > 0.0,
            "MANUAL_SJ requires a positive S_j.",
        )
        if t_sleeve is not None:
            if coupling is CouplingMode.LEGACY_EXCEL:
                den = 1.0 - float(S_j) * fiber.zeta_f / (
                    jacket.ID_j * S_ie
                )
                _require(
                    den > 0.0,
                    f"MANUAL_SJ infeasible: 1 - S_j zeta_f/(ID_j S_ie) "
                    f"= {den:.6g} <= 0.",
                )
                OD_c_val = 2.0 * float(t_sleeve) / den
            else:
                zeta_c_sj = float(S_j) * fiber.zeta_f / S_ie
                OD_c_val = zeta_c_sj + 2.0 * float(t_sleeve)
        elif OD_c is not None:
            OD_c_val = float(OD_c)
        else:
            raise ScalingError("MANUAL_SJ requires t_sleeve or OD_c.")
        if coupling is CouplingMode.LEGACY_EXCEL:
            S_c = float(S_j) * OD_c_val / jacket.ID_j
        else:
            S_c = float(S_j)

    # Cane recovery from the (achieved-by-construction) fiber target.
    d_c = tuple(
        S_c * df / s
        for df, s in zip(fiber.d_f, S_ce, strict=True)
    )
    radicands = tuple(
        Df**2 - df**2 + (df / s) ** 2
        for Df, df, s in zip(fiber.Delta_f, fiber.d_f, S_ce, strict=True)
    )
    for i, rad in enumerate(radicands):
        _require(
            rad > 0.0,
            f"Inverse ring {i + 1} radicand is negative ({rad:.6g}).",
        )
    Delta_c = tuple(S_c * math.sqrt(rad) for rad in radicands)
    t_c = tuple(
        (D - d) / 2.0 for D, d in zip(Delta_c, d_c, strict=True)
    )
    zeta_c = S_c * fiber.zeta_f / S_ie
    D_core_c = zeta_c - 2.0 * Delta_c[0]
    return (
        CaneGeometry(
            OD_c=OD_c_val, D_core_c=D_core_c, d_c=d_c, t_c=t_c,
            n_cap=n_cap,
        ),
        S_c,
    )


def inverse_jacket_od(
    fiber: FiberGeometry,
    ID_j: float,
    OD_c: float,
    S_j: float,
    S_ie: float,
    coupling: CouplingMode = CO_FED_VACUUM,
) -> float:
    """Jacket ``OD_j`` implied by a manual ``S_j`` and the fiber target."""
    _require(S_ie > 0.0, f"S_ie must be positive (got {S_ie}).")
    _require(S_j > 0.0, f"S_j must be positive (got {S_j}).")
    B = _inverse_B(fiber, S_ie)
    _require(
        B > 0.0,
        f"Inverse infeasible: B = {B:.6g} <= 0.",
    )
    if coupling is CouplingMode.LEGACY_EXCEL:
        return S_j * math.sqrt(B)
    radicand = ID_j**2 - OD_c**2 + S_j**2 * B
    _require(
        radicand > 0.0,
        f"Inverse jacket infeasible: ID_j^2 - OD_c^2 + S_j^2 B "
        f"= {radicand:.6g} <= 0.",
    )
    return math.sqrt(radicand)


# ---------------------------------------------------------------------------
# Stage-1 stack <-> cane
# ---------------------------------------------------------------------------


def cane_to_stack(
    cane: CaneGeometry,
    S_ie_st: float,
    S_ce_st: Sequence[float],
    S_stack: float | None = None,
    ID_stack: float | None = None,
    OD_stack: float | None = None,
) -> StackGeometry:
    """Scale the cane up to the stage-1 stack.

    Exactly one of ``S_stack`` / ``ID_stack`` / ``OD_stack`` must be
    given.
    """
    nb = len(cane.d_c)
    _require(
        len(S_ce_st) == nb,
        f"ring-count mismatch: {nb} cane rings, "
        f"{len(S_ce_st)} S_ce_st factors.",
    )
    _require(S_ie_st > 0.0, f"S_ie_st must be positive (got {S_ie_st}).")
    for i, s in enumerate(S_ce_st):
        _require(s > 0.0, f"S_ce_st[{i + 1}] must be positive (got {s}).")
    zeta_c = cane.zeta_c
    _require(
        zeta_c > 0.0,
        f"zeta_c must be positive (got {zeta_c}).",
    )
    _require(cane.OD_c > 0.0, f"OD_c must be positive (got {cane.OD_c}).")
    given = sum(
        x is not None for x in (S_stack, ID_stack, OD_stack)
    )
    _require(
        given == 1,
        "Exactly one of S_stack, ID_stack, OD_stack must be given.",
    )
    radicand = cane.OD_c**2 - (1.0 - 1.0 / S_ie_st**2) * zeta_c**2
    if ID_stack is not None:
        _require(
            float(ID_stack) > 0.0,
            f"ID_stack must be positive (got {ID_stack}).",
        )
        S_stack_val = S_ie_st * float(ID_stack) / zeta_c
    elif OD_stack is not None:
        _require(
            float(OD_stack) > 0.0,
            f"OD_stack must be positive (got {OD_stack}).",
        )
        _require(
            radicand > 0.0,
            f"Stack infeasible: OD_c^2 - (1 - 1/S_ie_st^2) zeta_c^2 "
            f"= {radicand:.6g} <= 0.",
        )
        S_stack_val = float(OD_stack) / math.sqrt(radicand)
    else:
        _require(
            float(S_stack) > 0.0,
            f"S_stack must be positive (got {S_stack}).",
        )
        S_stack_val = float(S_stack)

    d_st = tuple(
        S_stack_val * dc / s
        for dc, s in zip(cane.d_c, S_ce_st, strict=True)
    )
    ring_rad = tuple(
        Dc**2 - dc**2 + (dc / s) ** 2
        for Dc, dc, s in zip(cane.Delta_c, cane.d_c, S_ce_st, strict=True)
    )
    for i, rad in enumerate(ring_rad):
        _require(
            rad > 0.0,
            f"Stack ring {i + 1} radicand is negative ({rad:.6g}).",
        )
    Delta_st = tuple(S_stack_val * math.sqrt(rad) for rad in ring_rad)
    t_st = tuple(
        (D - d) / 2.0 for D, d in zip(Delta_st, d_st, strict=True)
    )
    return StackGeometry(
        S_stack=S_stack_val,
        S_ie_st=S_ie_st,
        S_ce_st=tuple(float(s) for s in S_ce_st),
        ID_stack=S_stack_val * zeta_c / S_ie_st,
        OD_stack=S_stack_val * _safe_sqrt(radicand),
        d_st=d_st,
        t_st=t_st,
    )


def stack_to_cane(
    stack: StackGeometry, OD_c: float | None = None
) -> CaneGeometry:
    """Recover the cane geometry from a stage-1 stack."""
    S_stack = stack.S_stack
    _require(S_stack > 0.0, f"S_stack must be positive (got {S_stack}).")
    _require(
        stack.S_ie_st > 0.0,
        f"S_ie_st must be positive (got {stack.S_ie_st}).",
    )
    for i, s in enumerate(stack.S_ce_st):
        _require(s > 0.0, f"S_ce_st[{i + 1}] must be positive (got {s}).")
    zeta_c = stack.S_ie_st * stack.ID_stack / S_stack
    if OD_c is not None:
        OD_c_val = float(OD_c)
    else:
        radicand = (
            stack.OD_stack**2 + (stack.S_ie_st**2 - 1.0) * stack.ID_stack**2
        )
        _require(
            radicand > 0.0,
            f"Cannot recover OD_c: OD_stack^2 + (S_ie_st^2 - 1) "
            f"ID_stack^2 = {radicand:.6g} <= 0.",
        )
        OD_c_val = math.sqrt(radicand) / S_stack
    d_c = tuple(
        ds * s / S_stack
        for ds, s in zip(stack.d_st, stack.S_ce_st, strict=True)
    )
    ring_rad = tuple(
        Ds**2 - ds**2 + (s * ds) ** 2
        for Ds, ds, s in zip(
            stack.Delta_st, stack.d_st, stack.S_ce_st, strict=True
        )
    )
    for i, rad in enumerate(ring_rad):
        _require(
            rad > 0.0,
            f"Stack ring {i + 1} radicand is negative ({rad:.6g}).",
        )
    Delta_c = tuple(math.sqrt(rad) / S_stack for rad in ring_rad)
    t_c = tuple(
        (D - d) / 2.0 for D, d in zip(Delta_c, d_c, strict=True)
    )
    return CaneGeometry(
        OD_c=OD_c_val,
        D_core_c=zeta_c - 2.0 * Delta_c[0],
        d_c=d_c,
        t_c=t_c,
        n_cap=6,
    )


# ---------------------------------------------------------------------------
# Solver bridge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverHandoff:
    """Scaling results expressed as solver-tab inputs (SI metres)."""

    D0: float
    d0: float
    D_c0: tuple[float, ...]
    d_c0: tuple[float, ...]
    tf: float
    Dcf1: float
    Dcore: float
    n_cap: int


def scaling_to_solver_inputs(
    cane: CaneGeometry,
    jacket: JacketGeometry,
    fiber: FiberGeometry,
    coupling: CouplingMode = CO_FED_VACUUM,
) -> SolverHandoff:
    """Map a scaling result onto the Jacket/Capillaries/Optimization tabs."""
    nb = len(cane.d_c)
    _require(
        len(cane.t_c) == nb and len(fiber.d_f) >= 1 and len(fiber.t_f) >= 1,
        "ring-count mismatch in the solver handoff.",
    )
    for i in range(nb):
        _require(
            cane.Delta_c[i] > cane.d_c[i] > 0.0,
            f"Ring {i + 1} must satisfy Delta_c > d_c > 0.",
        )
    if coupling is CouplingMode.LEGACY_EXCEL:
        _require(jacket.ID_j > 0.0, "ID_j must be positive.")
        D0 = jacket.OD_j * cane.OD_c / jacket.ID_j
    else:
        radicand = jacket.OD_j**2 - jacket.ID_j**2 + cane.OD_c**2
        _require(
            radicand > 0.0,
            f"Handoff infeasible: OD_j^2 - ID_j^2 + OD_c^2 "
            f"= {radicand:.6g} <= 0.",
        )
        D0 = math.sqrt(radicand)
    Dcf1 = fiber.d_f[0] + 2.0 * fiber.t_f[0]
    _require(
        math.isfinite(fiber.t_f[0]) and fiber.t_f[0] > 0.0,
        f"Fiber wall thickness t_f must be positive (got {fiber.t_f[0]}).",
    )
    _require(
        math.isfinite(Dcf1) and Dcf1 > 0.0,
        f"Fiber ring-1 outer diameter must be positive (got {Dcf1}).",
    )
    _require(
        math.isfinite(fiber.D_core_f) and fiber.D_core_f > 0.0,
        f"Fiber core diameter must be positive (got {fiber.D_core_f}).",
    )
    return SolverHandoff(
        D0=D0,
        d0=cane.zeta_c,
        D_c0=cane.Delta_c,
        d_c0=cane.d_c,
        tf=fiber.t_f[0],
        Dcf1=Dcf1,
        Dcore=fiber.D_core_f,
        n_cap=cane.n_cap,
    )


@dataclass(frozen=True)
class PulledScalingState:
    """Scaling state pulled from the solver tabs."""

    cane: CaneGeometry
    jacket: JacketGeometry
    fiber: FiberGeometry | None
    factors: ScalingFactors | None
    coupling: CouplingMode
    n_cap: int
    nb_nested: int


def solver_inputs_to_scaling(
    inputs: SimulationInputs,
    opt: OptimizationInputs | None = None,
    coupling: CouplingMode = CO_FED_VACUUM,
) -> PulledScalingState:
    """Read the solver tabs as a cane / jacket / fiber scaling state.

    ``OD_j = D0``, ``ID_j = OD_c = (D0 + d0)/2``, ``zeta_c = d0`` and
    ``Delta_c,i = Dc0,i``.  The fiber target is derived with
    :func:`compute_derived_targets` when ``opt`` is given; failures
    degrade gracefully to ``fiber = factors = None``.
    """
    R0, r0 = float(inputs.R0), float(inputs.r0)
    OD_j = 2.0 * R0
    ID_j = R0 + r0
    Delta_c = tuple(2.0 * float(ring.Rc0) for ring in inputs.rings)
    d_c = tuple(2.0 * float(ring.rc0) for ring in inputs.rings)
    t_c = tuple(
        (D - d) / 2.0 for D, d in zip(Delta_c, d_c, strict=True)
    )
    cane = CaneGeometry(
        OD_c=ID_j,
        D_core_c=2.0 * r0 - 2.0 * Delta_c[0],
        d_c=d_c,
        t_c=t_c,
        n_cap=int(inputs.n_capillaries),
    )
    jacket = JacketGeometry(OD_j=OD_j, ID_j=ID_j)
    fiber: FiberGeometry | None = None
    factors: ScalingFactors | None = None
    if opt is not None:
        try:
            targets = compute_derived_targets(inputs, opt)
            fiber = FiberGeometry(
                D_f=2.0 * float(targets.Rf),
                D_core_f=float(opt.Dcore),
                d_f=tuple(2.0 * v for v in targets.rcf_rings),
                t_f=(float(opt.tf),) * len(inputs.rings),
            )
            factors = auto_fit_scaling_factors(
                cane, fiber, jacket, coupling
            )
        except (ValueError, ZeroDivisionError):
            fiber = None
            factors = None
    return PulledScalingState(
        cane=cane,
        jacket=jacket,
        fiber=fiber,
        factors=factors,
        coupling=coupling,
        n_cap=int(inputs.n_capillaries),
        nb_nested=len(inputs.rings),
    )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingPreset:
    """A named cane / jacket / factors starting point."""

    name: str
    cane: CaneGeometry
    jacket: JacketGeometry
    factors: ScalingFactors
    desired_fiber: FiberGeometry | None = None


def get_scaling_presets() -> tuple[ScalingPreset, ...]:
    """The four workstation presets (1/2/3-ring + legacy Excel)."""
    one = ScalingPreset(
        name="1-ring AR-HCF (fed vacuum)",
        cane=CaneGeometry(
            OD_c=mm_to_m(3.8),
            D_core_c=mm_to_m(1.4),
            d_c=(mm_to_m(0.45),),
            t_c=(mm_to_m(0.05),),
            n_cap=6,
        ),
        jacket=JacketGeometry(OD_j=mm_to_m(10.0), ID_j=mm_to_m(4.0)),
        factors=ScalingFactors(S_j=40.0, S_ie=1.6, S_ce=(2.5,)),
    )
    two = ScalingPreset(
        name="2-ring NANF (fed vacuum)",
        cane=CaneGeometry(
            OD_c=mm_to_m(3.8),
            D_core_c=mm_to_m(1.4),
            d_c=(mm_to_m(0.45), mm_to_m(0.22)),
            t_c=(mm_to_m(0.05), mm_to_m(0.04)),
            n_cap=6,
        ),
        jacket=JacketGeometry(OD_j=mm_to_m(10.0), ID_j=mm_to_m(4.0)),
        factors=ScalingFactors(
            S_j=40.0, S_ie=1.6, S_ce=(2.5, 2.0),
        ),
    )
    three = ScalingPreset(
        name="3-ring DNANF (fed vacuum)",
        cane=CaneGeometry(
            OD_c=mm_to_m(3.8),
            D_core_c=mm_to_m(1.4),
            d_c=(mm_to_m(0.45), mm_to_m(0.22), mm_to_m(0.11)),
            t_c=(mm_to_m(0.05), mm_to_m(0.04), mm_to_m(0.025)),
            n_cap=6,
        ),
        jacket=JacketGeometry(OD_j=mm_to_m(10.0), ID_j=mm_to_m(4.0)),
        factors=ScalingFactors(
            S_j=40.0, S_ie=1.6, S_ce=(2.5, 2.0, 1.8),
        ),
    )
    legacy = ScalingPreset(
        name="Legacy Excel (6Ring_Scaling)",
        cane=CaneGeometry(
            OD_c=mm_to_m(2.296296),
            D_core_c=mm_to_m(1.381945),
            d_c=(mm_to_m(0.481),),
            t_c=(mm_to_m(0.032),),
            n_cap=6,
        ),
        jacket=JacketGeometry(OD_j=mm_to_m(10.0), ID_j=mm_to_m(3.0)),
        factors=ScalingFactors(
            S_j=63.5, S_ie=1.82, S_ce=(2.62,), coupling=LEGACY_EXCEL
        ),
    )
    return (one, two, three, legacy)
