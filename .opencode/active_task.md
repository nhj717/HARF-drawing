# Active Task — Shared Project Map & Invariants

> This file holds the SHARED project map and invariants for all concurrent
> sessions working on the MATLAB → Python port of the HARF-drawing app.
> Never overwrite it as a single-task lock; append task-scoped notes under
> `## Session Notes` instead. Task-specific Execution Plans / Acceptance
> Assertions live in each subagent prompt or in `.opencode/spec_scratch.md`.

## Repository Map

| Path          | Contents                                                              |
| ------------- | --------------------------------------------------------------------- |
| `reference/`  | Original MATLAB sources (16 `*.m` scripts, `HARF_drawing.mlapp`,
|               | `Logo.png`, `github-mark.png`). Read-only reference for the port.     |
| `src/`        | Python core package (solvers, physics models). Ported modules land
|               | here; currently a fresh skeleton (`src/__init__.py`).                   |
| `tests/`      | Unit + invariant tests (`test_invariants.py`).                        |
| `doc/`        | Documentation figures (`Fibre_HARF.png`).                             |
| `.idea/`      | PyCharm project config (preserved).                                    |
| `README.md`   | Project readme (preserved).                                             |
| `LICENSE`    | Project license (preserved).                                           |
| `data/`, `results/` | Raw simulation outputs (created on demand; never read
|               | directly — use the `data-inspector` skill).                            |

## Toolchain

- Environment manager: `uv` (`.venv/`, `pyproject.toml`, `uv.lock`).
- Python: >= 3.12. Deps: `numpy`, `scipy`; dev: `pytest`, `ruff`.
- Run tests: `uv run pytest`. Lint: `uv run ruff check .`.

## Mathematical & Physical Conventions (Invariants)

- **Units:** SI (`m`, `s`, `Hz`, `W`) by default, or explicit optical units
  (`nm`, `ps`, `THz`) documented in each module docstring.
- **Angular frequency:** `omega = 2 * pi * f`.
- **Fourier convention:** standard NumPy/SciPy FFT (`np.fft`) unless a
  physics module explicitly defines the optical sign convention
  `exp(-i*omega*t)`.
- **Array layout:** complex fields are `complex128`, batch/mode-major
  ordering `(N_modes, N_grid)`.

## Verification Contract

- Any change to `src/` must pass
  `uv run pytest tests/test_invariants.py -x -q --tb=short` via `@verifier`
  before milestone completion.
- `@coder` runs only fast scoped lint (`ruff check` on touched files);
  `@verifier` owns scoped `pytest`, `ruff check`, and UI audits.
- Git checkpoint rule: commit verified changes after each completed
  milestone so `HEAD` reflects the latest verified state.

## Session Notes

- 2026-11-16: Initial workspace setup completed (uv env, `reference/`
  reorganization, Python skeleton, junk cleanup).
