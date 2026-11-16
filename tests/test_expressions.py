"""Safe expression compiler: syntax parity, sandboxing, validation (A2)."""

import math

import numpy as np
import pytest

from src.expressions import (
    CompiledExpression,
    ExpressionError,
    normalize_matlab,
)

# --------------------------------------------------------------------------
# MATLAB / Python syntax parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("matlab", "python"),
    [
        ("2^3", "2**3"),
        ("2.^3", "2**3"),
        ("3.^4", "3.0**4"),
        ("2.*3.^2", "2*3.0**2"),
        ("6./2", "6/2"),
        ("2.*x./4", "2*x/4"),
    ],
)
def test_matlab_python_syntax_parity(matlab, python):
    """Element-wise MATLAB operators equal their Python counterparts."""
    for values in ((2.0,), (0.5,), (-1.5,)):
        a = CompiledExpression(matlab, ("x",)).scalar(*values)
        b = CompiledExpression(python, ("x",)).scalar(*values)
        assert a == pytest.approx(b, rel=1e-15, abs=1e-300)


def test_scalar_and_vector_agree():
    """Scalar and ndarray evaluators produce identical values."""
    expr = CompiledExpression(
        "exp(-(z-0.04)^2/2/0.06^2)", ("z",)
    )
    z = np.linspace(0.0, 0.1, 21)
    vec = expr.vector(z)
    assert vec.shape == z.shape
    for zi, vi in zip(z, vec, strict=True):
        assert expr.scalar(zi) == pytest.approx(float(vi), rel=1e-14)


def test_min_max_vectorize_elementwise():
    """MATLAB min/max map to np.minimum/np.maximum on arrays."""
    expr = CompiledExpression(
        "-1.9e-11*(9*min(max(310.75,T),1669.67)/5-459.4)+621.75", ("T",)
    )
    T = np.array([200.0, 700.0, 1500.0, 3000.0])
    out = expr.vector(T)
    expected = -1.9e-11 * (
        9 * np.minimum(np.maximum(310.75, T), 1669.67) / 5 - 459.4
    ) + 621.75
    np.testing.assert_allclose(out, expected, rtol=1e-14)
    # clamping is elementwise, not an aggregate like np.min
    assert out[0] != out[3]


def test_default_material_laws_evaluate():
    """The v1.5 default viscosity / cp / profile laws compile and run."""
    from src.models import DEFAULT_CP, DEFAULT_VISCOSITY

    visc = CompiledExpression(DEFAULT_VISCOSITY, ("T",))
    cp = CompiledExpression(DEFAULT_CP, ("T",))
    T = np.linspace(300.0, 2200.0, 50)
    assert np.all(visc.vector(T) > 0.0)
    assert np.all(cp.vector(T) > 0.0)
    assert visc.scalar(1720.0) == pytest.approx(
        10 ** (-7.946) * math.exp(66337.0 / 1720.0), rel=1e-12
    )


def test_pi_constant_available():
    expr = CompiledExpression("2*pi", ("T",))
    assert expr.scalar(0.0) == pytest.approx(2 * math.pi)


def test_normalize_matlab_operators():
    assert normalize_matlab("a.^b.*c./d") == "a**b*c/d"
    assert normalize_matlab("x^2") == "x**2"


# --------------------------------------------------------------------------
# Sandboxing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('true')",
        "x.__class__",
        "x.real",
        "open('/etc/passwd')",
        "lambda v: v",
        "[1, 2, 3]",
        "x if x else 1",
        "'literal'",
    ],
)
def test_unsafe_or_unsupported_nodes_rejected(source):
    with pytest.raises(ExpressionError):
        CompiledExpression(source, ("x",))


def test_unknown_name_rejected():
    with pytest.raises(ExpressionError):
        CompiledExpression("sin(T) + nope(T)", ("T",))


def test_unbound_variable_rejected():
    with pytest.raises(ExpressionError):
        CompiledExpression("T + z", ("T",))


def test_syntax_error_rejected():
    with pytest.raises(ExpressionError):
        CompiledExpression("exp((T", ("T",))


def test_safe_exp_clips_overflow():
    """exp is clipped so extreme exponents stay finite, never raise."""
    expr = CompiledExpression("exp(T)", ("T",))
    assert math.isfinite(expr.scalar(1e6))
    assert expr.scalar(1e6) == pytest.approx(math.exp(700.0), rel=1e-12)
    assert expr.vector(np.array([1e6, -1e6])).tolist() == [
        math.exp(700.0),
        0.0,
    ]


def test_division_by_zero_is_nan_not_raise():
    expr = CompiledExpression("1/T", ("T",))
    assert math.isnan(expr.scalar(0.0))
    out = expr.vector(np.array([1.0, 0.0]))
    assert np.isnan(out[1]) and out[0] == 1.0


# --------------------------------------------------------------------------
# Grid validation
# --------------------------------------------------------------------------


def test_validate_rejects_non_positive_values():
    expr = CompiledExpression("T - 500", ("T",))
    with pytest.raises(ExpressionError, match="non-positive"):
        expr.validate(np.linspace(300.0, 400.0, 5), name="law")


def test_validate_rejects_nan():
    expr = CompiledExpression("sqrt(T)", ("T",))
    with pytest.raises(ExpressionError, match="NaN"):
        expr.validate(np.array([-1.0, 1.0]))


def test_validate_accepts_positive_grid():
    expr = CompiledExpression("T^2 + 1", ("T",))
    out = expr.validate(np.linspace(1.0, 100.0, 10))
    assert out.shape == (10,)


def test_wrong_grid_count_rejected():
    expr = CompiledExpression("T + z", ("T", "z"))
    with pytest.raises(ExpressionError, match="grid"):
        expr.validate(np.linspace(0.0, 1.0, 3))


def test_callable_dispatch():
    expr = CompiledExpression("T^2", ("T",))
    arr = np.array([1.0, 2.0])
    assert isinstance(expr(arr), np.ndarray)
    assert expr(3.0) == 9.0
