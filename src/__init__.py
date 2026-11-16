"""HARF-drawing core package: Python port of the MATLAB HARF drawing app.

Units follow SI conventions (m, s, Hz, W) or explicit optical units
(nm, ps, THz) documented per module. Angular frequency is
omega = 2 * pi * f. Complex fields use complex128 with explicit
(N_modes, N_grid) batch/mode-major ordering.
"""
