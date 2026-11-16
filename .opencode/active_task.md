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
| `src/`        | Python core package: `models.py` (dataclasses, targets,
|               | diagnostics), `expressions.py` (safe MATLAB/Python AST),
|               | `materials.py` (clamped laws, `273.15` K, `viscosity_3` fix),
|               | `physics.py` (4-state Jacket + 1-state/ring Capillary ODEs),
|               | `solvers.py` (`brentq` + parallel scan + TRF optimizer),
|               | `io_utils.py` (`.csv`/`.txt` export, `.json` config).          |
| `app/`        | `PySide6` + `Matplotlib` 5-tab desktop GUI (`main.py`,
|               | `main_window.py`, `workers.py`, `tabs/`, `widgets/`).         |
| `tests/`      | Unit, invariant, safety, I/O & offscreen Qt GUI tests         |
|               | (`test_{invariants,expressions,physics,optimization,safety,io,gui}.py`). |
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
