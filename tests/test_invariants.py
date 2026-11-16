"""Minimal sanity checks for the freshly initialized Python package."""

import numpy as np
import scipy

import src


def test_package_imports() -> None:
    """The src package imports and exposes its docstring."""
    assert src.__doc__ is not None
    assert "HARF-drawing" in src.__doc__


def test_scientific_stack_available() -> None:
    """numpy and scipy are importable with expected major versions."""
    assert np.array([1.0, 2.0]).sum() == 3.0
    assert int(scipy.__version__.split(".")[0]) >= 1
