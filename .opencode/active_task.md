# Active Task — Shared Project Map, Invariants & State Handoff

## Repository Map & Toolchain
- `reference/`: Read-only MATLAB v1.5 (`16 *.m`, `HARF_drawing.mlapp`) + `6Ring_Scaling.xlsx` (`cane 20180212`).
- `src/`: `models.py`, `expressions.py`, `materials.py`, `physics.py`, `solvers.py`, `scaling.py`, `sem_metrology.py`, `io_utils.py`.
- `app/`: PySide6 + Matplotlib 6-tab GUI (`theme.py`, `main.py`, `main_window.py`, `workers.py`, `tabs/`, `widgets/`, `dialogs/`).
- `doc/`: `Fibre_HARF.png`, `cane_scaling_guidelines.md`, `cane_scaling_regimes.png`.
- `tests/`: `test_{invariants,expressions,physics,optimization,scaling,sem_metrology,safety,io,gui}.py`.
- `data/`, `results/`: Raw simulation outputs (never read directly — invoke `data-inspector` skill).
- Toolchain: `uv` (`Python >=3.12`, `numpy`, `scipy`, `pyside6`, `matplotlib`, `pytest`, `ruff`). Any `src/` edit requires `uv run pytest tests/test_invariants.py -x -q --tb=short` via `@verifier` + git checkpoint via `@git`.

## 1. Current State
- [x] Core MATLAB v1.5 → Python port (`src/models.py`, `expressions.py`, `materials.py`, `physics.py`, `solvers.py`, `io_utils.py`) with `273.15 K` & Ring-3 `viscosity_3` fixes.
- [x] 6-tab PySide6 workstation (`Parameters`, `Jacket`, `Capillaries`, `Direct computation`, `Optimization`, `Cane & Stack`) with `<16 ms` memoized/deferred tab switching (`0.40 ms` warm mean) and resolved system fonts (`0` `Monospace` warnings).
- [x] Bidirectional 1–3 nested-ring `Cane & Stack` scaling engine (`src/scaling.py`) + Tab 6 (`app/tabs/tab_scaling.py`) supporting `CO_FED_VACUUM` & `LEGACY_EXCEL` (`6Ring_Scaling.xlsx` `rtol=1e-6`), Stage-1 Stack $\leftrightarrow$ Cane, 4 presets, dual schematics, and Solver-Tab `Pull`/`Push` (`2ff08ea`).
- [x] Research documentation (`doc/cane_scaling_guidelines.md`, `doc/cane_scaling_regimes.png`) synthesizing SR-ARF, 2-ring NANF, 3-ring DNANF, and 3-stage UV/Vis scaling regimes (`2ff08ea`).
- [x] 6-tab ribbon anti-elision fix (`Qt.TextElideMode.ElideNone` + uniform `font-weight: 600`, `+31 px` slack on `"Cane & Stack"`) + SEM/microscope cross-section metrology engine & inspector dialog (`src/sem_metrology.py`, `app/dialogs/sem_inspector.py`, `8d48a25` — `213/213` pytest & `ruff` green).
- [ ] Optional user-driven follow-ups: test `SemInspectorDialog` on real lab cane/fiber SEM images and prune any redundant UI options in `Cane & Stack` after user review of `doc/cane_scaling_guidelines.md`.

## 2. Ground-Truth Map
- `src/scaling.py:1-698` — `CouplingMode` (`CO_FED_VACUUM`, `LEGACY_EXCEL`), `compute_forward_scaling`, `auto_fit_scaling_factors`, `solve_inverse_cane` (`FIXED_JACKET` & `MANUAL_SJ`), `cane_to_stack`/`stack_to_cane`, `ScalingFeasibility`, `get_scaling_presets`, `scaling_to_solver_inputs`/`solver_inputs_to_scaling`.
- `src/sem_metrology.py:1-1115` — `load_microscope_image`, `extract_tiff_pixel_size` (Zeiss `34118`, FEI `34682`/`34680`, ImageJ `270`, DPI-guarded `ResolutionUnit`), `calibrate_from_scale_bar`, `detect_sem_banner_row` (`:642-712`), `fit_circle_taubin`/`fit_circle_3pt`/`fit_circle_robust` (96-triplet LMedS seed), `measure_wall_profile_subpixel` (wall-tangent `_peak_centroid`, `<0.06 px` oblique error), `estimate_cross_section` (`:934-1115`, outer-tangent ray seeding).
- `src/models.py:410-465` — `compute_derived_targets` (`2 * Rf == OD_j_H` under `CO_FED_VACUUM`).
- `src/__init__.py:19-170` — public package exports (`scaling` + `sem_metrology`).
- `app/main_window.py:67-95,150-212,276-364` — `1280x800` 6-tab setup, `Qt.TextElideMode.ElideNone` (`:88-93`), cross-tab wiring, `ui_state["scaling"]` JSON round-trip.
- `app/theme.py:220-520` — resolved font helpers (`mono_font_family`, `font_stack_qss`), uniform `QTabBar::tab` `font-weight: 600; padding: 7px 16px` (`:311,321`), `QTableWidget` QSS.
- `app/widgets/cross_section_canvas.py:100-290` — `compute_capillary_centers` (1 nested tube/azimuth outer-tangent $\rho_i$), `set_static_geometry` (dual cane+jacket annulus, `showEvent` deferred `draw_idle()`).
- `app/tabs/tab_scaling.py:1-979` — `ScalingTab` (`STAGE_ROWS` alias `:177-179`, feasibility badge grid `:305-312`, `Import SEM / Image…` hooks `:275-281,917-979`, `apply_sem_payload` selective SI write).
- `app/dialogs/sem_inspector.py:1-780` & `app/dialogs/__init__.py:1-6` — `SemInspectorDialog` (`1120x760`, 5 canvas tools, live 1D profile inset, `"Wall t_i Only"` & selective apply checkboxes; minor cosmetic double-layout QSS notice at `:235`).
- `doc/cane_scaling_guidelines.md:1-830` & `doc/cane_scaling_regimes.png` — scaling theory, Jasion/Sakr/Fokoua synthesis, worked examples, Excel anchor audit.
- `tests/test_scaling.py:1-260`, `tests/test_sem_metrology.py:1-420`, `tests/test_gui.py:1204-1560`, `tests/test_invariants.py:38-48` — 213 passing tests.

## 3. Hard Invariants & Decisions
- **Internal SI vs. UI Units:** All `src/` dataclasses/functions operate in SI (`m`, `m/s`, `Pa`, `N`, `K`); UI uses `mm` for Stack/Cane/Jacket, `µm` for Fiber stages, `mm/min` for feed $w_0$, and `m/min` ($60\,w_{\text{SI}}$) for draw speed $V_f$.
- **Dual Sleeving Coupling:** Keep both `CO_FED_VACUUM` ($S_c = S_j$, $D_{0,\text{eff}} = \sqrt{\text{OD}_j^2-\text{ID}_j^2+\text{OD}_c^2}$, exact ODE mass-conservation match) and `LEGACY_EXCEL` ($S_c = S_j\,\text{OD}_c/\text{ID}_j$, exact `6Ring_Scaling.xlsx` match); Forward mode must never raise when $\text{OD}_c < \zeta_c$ (flags `cane_wall_positive=False`).
- **Nested Capillary Geometry & Expansion:** Outer-tangent 1-tube-per-azimuth nesting ($\rho_i = r - \sum_{m<i} t_m - R_{c,i}$, `N_cap * N_nested` tubes total, never $N_{\text{cap}}^{i+1}$); nested draws require graded $S_{ce,1} > S_{ce,2} > S_{ce,3}$ and $\Delta_i / d_{i-1} < 1$.
- **SEM Metrology Robustness:** Rejected raw Kåsa/unseeded Taubin (0% outlier breakdown) and caliper-perpendicular strip averaging (oblique smearing); locked in 96-triplet LMedS-seeded Taubin, wall-tangent `_peak_centroid` strip averaging, core-outward outer-tangent ray seeding, pure-`struct` TIFF parser (rejecting `<=1000 px/cm` bogus DPI), and zero OpenCV dependency.
- **Qt/Font/Perf Guardrails:** Use `Qt.TextElideMode.ElideNone` (never non-existent `Qt.ElideMode`), keep `font-weight: 600` identical on selected/unselected tabs, never emit bare `Monospace`, and never call `tight_layout()` or synchronous `draw()` on hidden tabs.

## 4. Next Immediate Command
```bash
QT_QPA_PLATFORM=offscreen uv run python -m pytest -q --tb=short && uv run ruff check --select E,F,I,B,NPY .
```
