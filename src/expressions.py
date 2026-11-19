"""Safe mathematical expression compiler for user-editable material laws.

Accepts MATLAB-style elementwise operators (``^``, ``.^``, ``.*``, ``./``)
and plain Python syntax (``**``), compiles the expression through a strict
AST whitelist, and produces fast scalar (``math``) and vectorized
(``numpy``) evaluators.  Non-whitelisted syntax (attribute access, imports,
dunders, unknown identifiers, statements) is rejected with
:class:`ExpressionError`.

Supported functions: ``min``, ``max``, ``exp``, ``log``, ``log10``,
``log2``, ``sqrt``, ``sin``, ``cos``, ``tan``, ``abs``; constant ``pi``.
``exp`` is overflow-guarded (argument clipped to +700; large negative
arguments underflow to 0).  The scalar and vector evaluators return
identical IEEE-754 results, including ``NaN`` on division by zero.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

__all__ = ["CompiledExpression", "ExpressionError", "normalize_matlab"]


class ExpressionError(ValueError):
    """Raised for invalid, unsafe, or non-physical user expressions."""


_EXP_CLIP = 700.0

_ALLOWED_FUNCS = frozenset(
    {"exp", "log", "log10", "log2", "sqrt", "sin", "cos", "tan", "abs",
     "min", "max"}
)
_ALLOWED_CONSTANTS = frozenset({"pi"})
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _scalar_exp(x: float) -> float:
    """Overflow-guarded scalar exponential (upper-clipped only)."""
    try:
        return math.exp(min(x, _EXP_CLIP))
    except (TypeError, ValueError):
        return math.nan


def _vector_exp(x: np.ndarray) -> np.ndarray:
    """Overflow-guarded vectorized exponential (upper-clipped only)."""
    return np.exp(np.minimum(x, _EXP_CLIP))


class _Div0NanArray(np.ndarray):
    """Array whose division by zero yields NaN (scalar-evaluator parity).

    Python raises ``ZeroDivisionError`` on any ``x / 0`` (mapped to
    ``NaN`` by :meth:`CompiledExpression.scalar`); this subclass makes
    the numpy evaluator produce ``NaN`` — not ``inf`` — on the same
    inputs so both evaluators agree bit-for-bit.
    """

    def _divide(self, other, reflected: bool):
        a = np.asarray(self, dtype=float)
        b = np.asarray(other, dtype=float)
        with np.errstate(all="ignore"):
            out = np.divide(b, a) if reflected else np.divide(a, b)
        zero_den = (a == 0.0) if reflected else (b == 0.0)
        return np.where(zero_den, np.nan, out).view(type(self))

    def __truediv__(self, other):
        return self._divide(other, reflected=False)

    def __rtruediv__(self, other):
        return self._divide(other, reflected=True)


def _scalar_unary(fn):
    def wrapped(x):
        try:
            return fn(x)
        except (ValueError, OverflowError):
            return math.nan

    return wrapped


def _vector_unary(fn):
    def wrapped(x):
        with np.errstate(all="ignore"):
            return fn(np.asarray(x, dtype=float))

    return wrapped


def _vector_min(*args):
    if len(args) == 1:
        return np.min(args[0])
    if len(args) == 2:
        return np.minimum(args[0], args[1])
    raise ExpressionError("min() expects 1 or 2 arguments")


def _vector_max(*args):
    if len(args) == 1:
        return np.max(args[0])
    if len(args) == 2:
        return np.maximum(args[0], args[1])
    raise ExpressionError("max() expects 1 or 2 arguments")


_SCALAR_NAMESPACE: dict = {
    "pi": math.pi,
    "exp": _scalar_exp,
    "log": _scalar_unary(math.log),
    "log10": _scalar_unary(math.log10),
    "log2": _scalar_unary(math.log2),
    "sqrt": _scalar_unary(math.sqrt),
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "min": min,
    "max": max,
}

_VECTOR_NAMESPACE: dict = {
    "pi": np.pi,
    "exp": _vector_exp,
    "log": _vector_unary(np.log),
    "log10": _vector_unary(np.log10),
    "log2": _vector_unary(np.log2),
    "sqrt": _vector_unary(np.sqrt),
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "abs": np.abs,
    "min": _vector_min,
    "max": _vector_max,
}


def normalize_matlab(source: str) -> str:
    """Normalize MATLAB elementwise operators to Python syntax."""
    if not isinstance(source, str):
        raise ExpressionError("expression must be a string")
    s = source.strip()
    if not s:
        raise ExpressionError("expression is empty")
    s = s.replace(".^", "**").replace(".*", "*").replace("./", "/")
    s = s.replace("^", "**")
    return s


def _validate_node(node: ast.AST, variables: frozenset[str]) -> None:
    """Recursively validate that *node* only uses whitelisted constructs."""
    if isinstance(node, ast.Expression):
        _validate_node(node.body, variables)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float)
        ):
            raise ExpressionError(
                f"unsupported constant {node.value!r} (numbers only)"
            )
    elif isinstance(node, ast.Name):
        if node.id not in variables and node.id not in _ALLOWED_CONSTANTS:
            raise ExpressionError(f"unknown identifier {node.id!r}")
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ExpressionError(
                f"unsupported operator {type(node.op).__name__}"
            )
        _validate_node(node.left, variables)
        _validate_node(node.right, variables)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise ExpressionError(
                f"unsupported operator {type(node.op).__name__}"
            )
        _validate_node(node.operand, variables)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only direct function calls are allowed")
        if node.func.id not in _ALLOWED_FUNCS:
            raise ExpressionError(f"function {node.func.id!r} is not allowed")
        if node.keywords:
            raise ExpressionError("keyword arguments are not allowed")
        for arg in node.args:
            _validate_node(arg, variables)
    else:
        raise ExpressionError(
            f"disallowed syntax node {type(node).__name__!r}"
        )


@lru_cache(maxsize=256)
def _compile_cached(
    normalized: str, variables: tuple[str, ...]
) -> tuple:
    """Parse + validate + build the scalar/vector evaluators (cached).

    Repeated UI refreshes with unchanged expression strings hit the
    cache and do zero AST re-parsing; raised :class:`ExpressionError`
    or :class:`SyntaxError` values are never cached.
    """
    tree = ast.parse(normalized, mode="eval")
    _validate_node(tree.body, frozenset(variables))
    scalar = _build_function(tree, variables, _SCALAR_NAMESPACE)
    vector = _build_function(tree, variables, _VECTOR_NAMESPACE)
    return scalar, vector


def _build_function(
    tree: ast.Expression, variables: Sequence[str], namespace: dict
):
    """Compile the validated expression tree into a fast function."""
    args = [ast.arg(arg=name) for name in variables]
    fn_def = ast.FunctionDef(
        name="_expr",
        args=ast.arguments(
            posonlyargs=[],
            args=args,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[ast.Return(value=tree.body)],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    module = ast.Module(body=[fn_def], type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, "<harf-expression>", "exec")
    scope: dict = {"__builtins__": {}}
    scope.update(namespace)
    exec(code, scope)  # noqa: S102 - AST pre-validated, no builtins
    return scope["_expr"]


class CompiledExpression:
    """A validated, sandboxed scalar/vector mathematical expression.

    Parameters
    ----------
    source:
        The raw expression text (MATLAB or Python operator syntax).
    variables:
        Ordered free-variable names (e.g. ``("T",)`` or ``("z",)``).
    """

    def __init__(self, source: str, variables: Sequence[str] = ("T",)) -> None:
        self.source = str(source)
        self.variables = tuple(str(v) for v in variables)
        if not self.variables:
            raise ExpressionError("at least one variable is required")
        self.normalized = normalize_matlab(self.source)
        try:
            self._scalar_fn, self._vector_fn = _compile_cached(
                self.normalized, self.variables
            )
        except SyntaxError as exc:
            raise ExpressionError(
                f"syntax error in expression {self.source!r}: {exc.msg}"
            ) from exc

    # -- evaluation -----------------------------------------------------
    def scalar(self, *values: float) -> float:
        """Evaluate on scalars; returns ``nan`` for invalid results."""
        try:
            out = self._scalar_fn(*values)
        except ZeroDivisionError:
            return math.nan
        if isinstance(out, complex):
            return math.nan
        return float(out)

    def vector(self, *values) -> np.ndarray:
        """Evaluate on (broadcastable) numpy arrays; NaN-safe."""
        arrays = [
            np.asarray(v, dtype=float).view(_Div0NanArray) for v in values
        ]
        try:
            with np.errstate(all="ignore"):
                out = self._vector_fn(*arrays)
        except ZeroDivisionError:
            return np.full(np.broadcast(*arrays).shape, np.nan)
        return np.asarray(out, dtype=float)

    def __call__(self, *values):
        if any(isinstance(v, np.ndarray) for v in values):
            return self.vector(*values)
        return self.scalar(*values)

    # -- validation -----------------------------------------------------
    def validate(
        self,
        *grids,
        positive: bool = True,
        name: str = "expression",
    ) -> np.ndarray:
        """Evaluate on test grids and reject non-finite results.

        Raises :class:`ExpressionError` when the expression evaluates to
        NaN/Inf, or to non-positive values when ``positive`` is set
        (physical properties such as viscosity must stay positive).
        """
        if len(grids) != len(self.variables):
            raise ExpressionError(
                f"expected {len(self.variables)} grid(s) for variables "
                f"{self.variables}"
            )
        values = self.vector(*grids)
        if not np.all(np.isfinite(values)):
            raise ExpressionError(
                f"{name} evaluates to NaN or Inf on the test grid"
            )
        if positive and not np.all(values > 0.0):
            raise ExpressionError(
                f"{name} evaluates to non-positive values on the test grid"
            )
        return values

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"CompiledExpression({self.source!r}, "
            f"variables={self.variables})"
        )
