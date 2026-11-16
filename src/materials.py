"""Clamped material/furnace evaluators with the MATLAB v1.5 defaults.

Every property is a function of temperature ``T`` in Kelvin (or axial
position ``z`` in meters for the furnace profile).  Evaluators clamp
``T`` to ``[200, 5000] K``, guard ``exp`` against overflow, and clip
viscosity to ``[1e-3, 1e22] Pa.s`` so that pathological user expressions
can never produce NaN/Inf inside the solvers.

Defaults were extracted from ``reference/HARF_drawing.mlapp``
(``matlab/document.xml``).  Note the reference bug fix: ring 3 uses its
own ``viscosity_3`` law (the MATLAB ``RK_perso11.m`` line 20 mistakenly
reused ``viscosity_2`` for ring 3).
"""


from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.expressions import CompiledExpression
from src.models import (
    DEFAULT_ALPHA,
    DEFAULT_CP,
    DEFAULT_FURNACE_PROFILE,
    DEFAULT_GAMMA,
    DEFAULT_RHO,
    DEFAULT_VISCOSITY,
    SimulationInputs,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_CP",
    "DEFAULT_FURNACE_PROFILE",
    "DEFAULT_GAMMA",
    "DEFAULT_GAMMA_CAPILLARY",
    "DEFAULT_RHO",
    "DEFAULT_VISCOSITY",
    "DEFAULT_VISCOSITY_CAPILLARY",
    "T_CLAMP_MAX_K",
    "T_CLAMP_MIN_K",
    "VISCOSITY_MAX_PA_S",
    "VISCOSITY_MIN_PA_S",
    "MaterialProperty",
    "MaterialSet",
    "validate_materials",
]

T_CLAMP_MIN_K = 200.0
T_CLAMP_MAX_K = 5000.0

VISCOSITY_MIN_PA_S = 1e-3
VISCOSITY_MAX_PA_S = 1e22

#: Capillary surface tension N/m (defaults to the jacket law).
DEFAULT_GAMMA_CAPILLARY = DEFAULT_GAMMA

#: Capillary viscosity Pa.s (defaults to the jacket silica law).
DEFAULT_VISCOSITY_CAPILLARY = DEFAULT_VISCOSITY

_VALIDATE_T_GRID = (273.15, 500.0, 1000.0, 1500.0, 2000.0, 2500.0)


@dataclass(frozen=True)
class _PropertyBounds:
    lo: float
    hi: float
    nan_value: float


_PROPERTY_BOUNDS = {
    "viscosity": _PropertyBounds(VISCOSITY_MIN_PA_S, VISCOSITY_MAX_PA_S,
                                 VISCOSITY_MAX_PA_S),
    "rho": _PropertyBounds(1.0, 1e5, 2200.0),
    "gamma": _PropertyBounds(0.0, 10.0, 0.3),
    "cp": _PropertyBounds(1.0, 1e5, 1200.0),
    "alpha": _PropertyBounds(0.0, 1.0, 0.885),
}


class MaterialProperty:
    """A physical property of ``T`` [K] with crash-proof clamping.

    ``kind`` selects the clamping bounds (``viscosity`` is clipped to
    ``[1e-3, 1e22] Pa.s`` per the project invariants).
    """

    def __init__(self, name: str, source: str, kind: str = "generic") -> None:
        self.name = name
        self.kind = kind
        self._expr = CompiledExpression(source, ("T",))
        bounds = _PROPERTY_BOUNDS.get(kind)
        self._bounds = bounds if bounds is not None else None

    @property
    def source(self) -> str:
        return self._expr.source

    def _sanitize(self, value: float) -> float:
        if self._bounds is None:
            return value
        if not math.isfinite(value):
            value = self._bounds.nan_value
        return min(max(value, self._bounds.lo), self._bounds.hi)

    def __call__(self, T: float) -> float:
        """Clamped scalar evaluation at temperature ``T`` [K]."""
        tc = min(max(T, T_CLAMP_MIN_K), T_CLAMP_MAX_K)
        return self._sanitize(self._expr.scalar(tc))

    def vector(self, T) -> np.ndarray:
        """Clamped vectorized evaluation (returns float ndarray)."""
        tc = np.clip(np.asarray(T, dtype=float), T_CLAMP_MIN_K, T_CLAMP_MAX_K)
        values = self._expr.vector(tc)
        if self._bounds is not None:
            values = np.where(
                ~np.isfinite(values), self._bounds.nan_value, values
            )
            values = np.clip(values, self._bounds.lo, self._bounds.hi)
        return values

    def validate(self, name: str | None = None) -> None:
        """Reject non-finite / non-positive evaluations on a test grid."""
        self._expr.validate(
            np.asarray(_VALIDATE_T_GRID),
            positive=True,
            name=name or f"material property {self.name!r}",
        )


@dataclass(frozen=True)
class _RingMaterials:
    gamma: MaterialProperty
    viscosity: MaterialProperty


class MaterialSet:
    """Compiled material laws for the jacket and every capillary ring."""

    def __init__(
        self,
        profil: CompiledExpression,
        rho: MaterialProperty,
        gamma: MaterialProperty,
        cp: MaterialProperty,
        viscosity: MaterialProperty,
        alpha: MaterialProperty,
        rings: tuple[_RingMaterials, ...],
    ) -> None:
        self.profil = profil
        self.rho = rho
        self.gamma = gamma
        self.cp = cp
        self.viscosity = viscosity
        self.alpha = alpha
        self.rings = rings

    @classmethod
    def build(cls, inputs: SimulationInputs) -> MaterialSet:
        profil = CompiledExpression(inputs.furnace_profile_expr, ("z",))
        rings = tuple(
            _RingMaterials(
                gamma=MaterialProperty(
                    f"gamma_{i + 1}", ring.gamma_expr, kind="gamma"
                ),
                viscosity=MaterialProperty(
                    f"viscosity_{i + 1}",
                    ring.viscosity_expr,
                    kind="viscosity",
                ),
            )
            for i, ring in enumerate(inputs.rings)
        )
        return cls(
            profil=profil,
            rho=MaterialProperty("rho", inputs.rho_expr, kind="rho"),
            gamma=MaterialProperty("gamma", inputs.gamma_expr, kind="gamma"),
            cp=MaterialProperty("cp", inputs.cp_expr, kind="cp"),
            viscosity=MaterialProperty(
                "viscosity", inputs.viscosity_expr, kind="viscosity"
            ),
            alpha=MaterialProperty("alpha", inputs.alpha_expr, kind="alpha"),
            rings=rings,
        )


def validate_materials(inputs: SimulationInputs) -> None:
    """Validate every user expression; raises ``ExpressionError``.

    Rejects syntax errors, unsafe AST nodes and non-positive / NaN
    evaluations over a representative temperature grid.
    """
    profil = CompiledExpression(inputs.furnace_profile_expr, ("z",))
    z_grid = np.linspace(0.0, max(inputs.L, inputs.Lfour), 200)
    profil.validate(z_grid, positive=False, name="furnace profile")
    for prop in (
        MaterialProperty("rho", inputs.rho_expr, kind="rho"),
        MaterialProperty("gamma", inputs.gamma_expr, kind="gamma"),
        MaterialProperty("cp", inputs.cp_expr, kind="cp"),
        MaterialProperty("viscosity", inputs.viscosity_expr,
                         kind="viscosity"),
        MaterialProperty("alpha", inputs.alpha_expr, kind="alpha"),
    ):
        prop.validate()
    for i, ring in enumerate(inputs.rings):
        MaterialProperty(
            f"gamma_{i + 1}", ring.gamma_expr, kind="gamma"
        ).validate()
        MaterialProperty(
            f"viscosity_{i + 1}", ring.viscosity_expr, kind="viscosity"
        ).validate()
