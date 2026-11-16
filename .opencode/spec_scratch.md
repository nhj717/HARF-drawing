# Specification Scratchpad (Task-Scoped Blueprints)

## Task: HARF-Drawing MATLAB-to-Python Port (v1.5 -> Python/PySide6)

### 1. Governing Physics, Mass-Conservation Reduction & Bug Fixes
- **Piecewise Axial Domain:** $z \in [0, L]$ split at $L_{\text{four}} \in (0, L)$:
  - Hot zone $[0, L_{\text{four}}]$: $T_a(z) = T_{\max}\,\text{profil}(z) + 273.15\text{ K}$; Cold zone $(L_{\text{four}}, L]$: $T_a(z) = T_{\text{ext}} + 273.15\text{ K}$.
  - Initial preform temp: $T_0 = T_{\max}\,\text{profil}(0) + 273.15 - \Delta T_0$.
- **4-State Jacket ODE** $y_J = [\tau, w, u_J, T]^T$ ($u_J \equiv R^2 w$, independent of capillaries):
  - Invariant $\Delta_{J0} = (R_0^2 - r_0^2)w_0 > 0 \implies r^2 w = u_J - \Delta_{J0}$, $R = \sqrt{u_J/w}$, $r = \sqrt{(u_J-\Delta_{J0})/w}$, $R^2 - r^2 = \Delta_{J0}/w$.
  - $\frac{dw}{dz} = \frac{\tau}{3\pi\mu_J(T)(R^2-r^2)}$, $\quad \frac{du_J}{dz} = \frac{P_{\text{core}} r^2 R^2 - \gamma_J(T) r R (r+R)}{\mu_J(T)(R^2-r^2)}$
  - $\frac{d\tau}{dz} = \rho_J(T)\pi(R^2-r^2)\!\left(w\frac{dw}{dz}-g\right) - \frac{\gamma_J(T)\pi}{2w}\!\left(\frac{du_J/dz - R^2 dw/dz}{R} + \frac{du_J/dz - r^2 dw/dz}{r}\right)$
  - $\frac{dT}{dz} = \frac{2R\left[N(T_a-T) + \sigma(\alpha_J(T_a)T_a^4 - \alpha_J(T)T^4)\right]}{(R^2-r^2)\rho_J(T)C_{p,J}(T)w}$, $\quad y_J(0) = [\tau_0, w_0, R_0^2 w_0, T_0]^T$.
- **Decoupled 1-State Capillary ODE** ($i \in \{1..N_{\text{nested}}\}$, $N_{\text{nested}} \in \{1,2,3\}$, $u_{c,i} \equiv R_{c,i}^2 w$):
  - Invariant $\Delta_{c0,i} = (R_{c0,i}^2 - r_{c0,i}^2)w_0 > 0 \implies r_{c,i}^2 w = u_{c,i} - \Delta_{c0,i}$, $R_{c,i}^2 - r_{c,i}^2 = \Delta_{c0,i}/w(z)$.
  - Driven along cached dense $(w(z), T(z))$ by $\Delta P_1 = P_{\text{cap},1} - P_{\text{core}}$, $\Delta P_2 = P_{\text{cap},2} - P_{\text{cap},1}$, $\Delta P_3 = P_{\text{cap},3} - P_{\text{cap},2}$:
  - $\frac{du_{c,i}}{dz} = \frac{\Delta P_i r_{c,i}^2 R_{c,i}^2 - \gamma_i(T(z)) r_{c,i} R_{c,i}(r_{c,i}+R_{c,i})}{\mu_i(T(z))(R_{c,i}^2 - r_{c,i}^2)}$, $\quad u_{c,i}(0) = R_{c0,i}^2 w_0$.
  - **Bug Fix:** Ring 3 uses $\mu_3(T) = \text{viscosity\_3}(T)$ (fixing `RK_perso11.m:20` & `RK_perso11_293K.m:20`).

### 2. Derived Optimization Targets & Contact Geometry
- From `(Optimtf, OptimDcf1, OptimDcore)` $\equiv (t_f, D_{cf,1}, D_{\text{core}})$ (verified vs `HARF_drawing.mlapp` `RunButton_2Pushed`):
  - $R_{cf,1} = D_{cf,1}/2$, $r_{cf,1} = R_{cf,1} - t_f$, $w_f = \frac{\Delta_{c0,1}}{R_{cf,1}^2 - r_{cf,1}^2}$, $r_f = \frac{D_{\text{core}}}{2} + 2R_{cf,1}$, $R_f = \sqrt{r_f^2 + \frac{\Delta_{J0}}{w_f}}$.
  - Nested targets $i \in \{2,3\}$ satisfy $R_{cf,i}^2 - r_{cf,i}^2 = \Delta_{c0,i}/w_f$ with exact `RunButton_2Pushed` rule from `matlab/document.xml`.
- Contact & Core Geometry ($N_{\text{cap}} \ge 2$):
  - $R_{\text{contact}}(z) = r(z)\frac{\sin(\pi/N_{\text{cap}})}{1+\sin(\pi/N_{\text{cap}})}$, $\quad D_{\text{core}}(z) = 2(r(z) - 2R_{c,1}(z)) = d(z) - 2D_{c,1}(z)$.
  - Valid non-contact (`< 1` green, `>= 1` red): $R_{c,1}/R_{\text{contact}}$, $D_{c,2}/d_{c,1}$, $D_{c,3}/d_{c,2}$, $d_0/D_0$, $d_{c0,i}/D_{c0,i}$.

### 3. Crash-Proof Safety Net, Events & Signed Penalties
- **Safe AST & Clamping:** Normalize `.^/^ -> **`, `.*/./ -> */`; whitelist math AST only; dual scalar (`math`) & array (`numpy`, `min`/`max` -> `np.minimum`/`np.maximum`) evaluators; clamp $T \in [200, 5000]\text{ K}$, `safe_exp(clip(x,-700,700))`, $\mu \in [10^{-3}, 10^{22}]\text{ Pa}\cdot\text{s}$.
- **Integrator:** Piecewise `solve_ivp` (`DOP853`, `rtol=1e-9, atol=1e-13, dense_output=True`, fallback `Radau` on stiffness/step collapse) evaluated onto uniform $\Delta z = 10^{-5}\text{ m}$ grid.
- **Terminal Events (`direction=-1`):**
  - Jacket collapse: $(u_J - \Delta_{J0}) - 10^{-12}\Delta_{J0} = 0$; Jacket blow-up: $100 R_0^2 w - u_J = 0$.
  - Tension/velocity: $\tau - 10^{-9} = 0$; $10^5 w_0 - w = 0$; $w - 10^{-3}w_0 = 0$.
  - Capillary $i$: $(u_{c,i}-\Delta_{c0,i}) - 10^{-12}\Delta_{c0,i} = 0$ (collapse); $100 R_{c0,i}^2 w(z) - u_{c,i} = 0$ (blow-up).
- **Signed Root-Finding & Penalties ($\xi = 1 + (L - z_{\text{stop}})/L \in (1,2]$):**
  - $\tau_0$ via `brentq` (`xtol=1e-12`) on $F_\tau(\tau_0) = \tau(L) - \tau_{\text{target}}$ ($-\tau_{\text{target}}\xi$ on tension collapse; $+10\tau_{\text{target}}\xi$ on velocity blow-up) with geometric bracket expansion.
  - $(T_{\max}, P_{\text{core}})$: parallel coarse grid scan + bounded `least_squares(method='trf')` on $[r_1, r_2] = [(R(L)-R_f)/R_f,\; (R(L)/r(L) - R_f/r_f)/(R_f/r_f)]$, smoothly penalized by $\pm 10\xi$ along collapse/blow-up directions.
  - $\Delta P_i$ via decoupled 1-DOF `brentq` (`xtol=1e-6` Pa) on strictly increasing $F_{c,i}(\Delta P_i) = (R_{c,i}(L)-R_{cf,i})/R_{cf,i}$ ($(\sqrt{\Delta_{c0,i}/w(L)}-R_{cf,i})/R_{cf,i} - \xi$ on collapse; $(10 R_{c0,i}-R_{cf,i})/R_{cf,i} + 10\xi$ on blow-up). Recover $P_{\text{cap},i} = P_{\text{core}} + \sum_{k=1}^i \Delta P_k$.

### 4. Module Map
- **Milestone 1 (`src/` + `tests/`):**
  - `src/models.py`: Dataclasses, `SimulationStatus`, `FailureDiagnostics`, `SimulationResult`, `OptimizationResult`, SI converters, geometry/target formulas.
  - `src/expressions.py`: Safe AST compiler (MATLAB/Python syntax, scalar+NumPy callables, grid positivity validator).
  - `src/materials.py`: Clamped material/furnace evaluators & v1.5 defaults (extracted from `HARF_drawing.mlapp`).
  - `src/physics.py`: 4-state Jacket & 1-state Capillary ODEs, terminal events, `DOP853`->`Radau` fallback, dense $\Delta z$ sampling.
  - `src/solvers.py`: `solve_initial_tension`, `run_direct_simulation`, `solve_capillary_delta_p`, parallel scan + TRF `run_optimization`.
  - `src/io_utils.py`: Commented `.csv`/`.txt` axial export & schema-validated `.json` config load/save.
  - `tests/`: `test_invariants.py`, `test_expressions.py`, `test_physics.py`, `test_optimization.py`, `test_safety.py`, `test_io.py`.
- **Milestone 2 (`app/` + `pyproject.toml` + `tests/test_gui.py`):**
  - `pyproject.toml`: Add `pyside6`, `matplotlib`, `[project.scripts] harf-drawing = "app.main:main"`.
  - `app/main.py`, `app/main_window.py`, `app/workers.py` (`QThread` + cooperative `threading.Event` cancel + progress signals).
  - `app/tabs/`: `tab_parameters.py`, `tab_jacket.py`, `tab_capillaries.py`, `tab_direct.py`, `tab_optimization.py`.
  - `app/widgets/`: `validators.py` (red/green ratio badges), `plot_canvas.py` (12 views + `NavigationToolbar2QT` + live $(z,y)$ cursor), `cross_section_canvas.py` (`zmSlider` 2D fiber schematic).
  - `tests/test_gui.py`: Offscreen PySide6 test suite covering tabs, validators, workers/cancel, plots, `zmSlider`, and I/O.

### 5. Execution Plans & Acceptance Assertions
- **Milestone 1 Execution Plan:**
  1. Extract exact v1.5 defaults, units, and `RunButton_2Pushed` formulas from `reference/HARF_drawing.mlapp` (`matlab/document.xml` via `zipfile`), and build `src/models.py`, `src/expressions.py`, and `src/materials.py` with `273.15` K offset, `viscosity_3(T)` fix, and `[1e-3, 1e22]` Pa·s clamping.
  2. Implement `src/physics.py` and `src/solvers.py`: piecewise `[0, L_four]` + `[L_four, L]` `DOP853`/`Radau` 4-state Jacket & 1-state Capillary integrators, terminal breakdown events, adaptive `brentq` tension/capillary solvers, parallel coarse scan + TRF optimizer, and `src/io_utils.py` CSV/TXT/JSON persistence.
  3. Build unit and invariant suites (`tests/test_*.py`) verifying algebraic RHS equivalence to `RK_perso7/9/11`, decoupled vs joint ODE parity, optimizer target recovery, and crash-proof structured diagnostics across extreme failure regimes.
- **Milestone 1 Acceptance Assertions (5):**
  1. Algebraic RHS & trajectory parity between reduced $(4) + N_{\text{nested}}\times(1)$ solver and unreduced `(5 + 2*N_nested)` `RK_perso7/9/11` (with `273.15` and `viscosity_3` fix) holds to relative error $< 10^{-8}$, with distinct `viscosity_3` verified on ring 3.
  2. `CompiledExpression` evaluates MATLAB (`^`, `.^`, `.*`, `./`, `min`, `max`) and Python (`**`) syntax identically on scalars and `ndarray`s while rejecting unsafe AST nodes (`Attribute`, `Import`, dunders) and non-positive physical evaluations.
  3. `solve_initial_tension` achieves $|\tau(L) - \tau_{\text{target}}|/\tau_{\text{target}} < 10^{-10}$, and `run_optimization` recovers target $(R_f, r_f, w_f, R_{cf,1..3}, r_{cf,1..3})$ within $< 10^{-4}$ relative error and $|\Delta P_i - \Delta P_i^*| < 10^{-5}\text{ Pa}$.
  4. Extreme collapse ($P_{\text{core}} \ll 0$), blow-up ($P_{\text{core}} \gg 10^6\text{ Pa}$), tension loss ($\tau_0 \to 0$), and cryogenic $T$ inputs never raise unhandled exceptions or emit `NaN`/`Inf`, returning `SimulationResult` with accurate `SimulationStatus`, partial `[0, z_stop]` arrays, and actionable advice.
  5. `export_results_table` (`.csv`/`.txt`) and `save_config_json`/`load_config_json` round-trip all metadata and uniform $\Delta z = 10^{-5}\text{ m}$ arrays without `.mat` dependencies, passing `uv run pytest` and `uv run ruff check .`.
- **Milestone 2 Execution Plan:**
  1. Configure `pyproject.toml` (`pyside6`, `matplotlib`, `harf-drawing` script) and implement `app/widgets/` (`validators.py`, 12-view `plot_canvas.py` with `NavigationToolbar2QT` + cursor readout, `zmSlider`-linked `cross_section_canvas.py`).
  2. Implement `app/tabs/` (`Parameters`, `Jacket`, `Capillaries`, `Direct computation`, `Optimization`), `app/workers.py` (`QThread` progress + cooperative cancel), and `app/main_window.py` (`Apply to Direct Computation`, JSON config save/load, CSV/TXT export).
  3. Add headless (`QT_QPA_PLATFORM=offscreen`) `tests/test_gui.py` auditing widget defaults, red/green contact badges, worker threading/cancellation, plot views, `zmSlider` updates, and end-to-end GUI transfer/persistence.
- **Milestone 2 Acceptance Assertions (4):**
  1. Offscreen `MainWindow` instantiates all 5 tabs with exact MATLAB v1.5 inputs/defaults and dynamically toggles nested ring 2/3 controls and red ($\ge 1$) / green ($< 1$) contact badges (`d0/D0`, `dc0/Dc0`, `Rc01/Rcontact`, `Dc02/dc01`, `Dc03/dc02`, post-run maxima).
  2. All 12 plot dropdown views render finite curves (including partial `[0, z_stop]` failure runs) with `NavigationToolbar2QT` and live $(z, y)$ cursor updates, and `zmSlider` updates the 2D cross-section geometry accurately across $[0, z_{\text{end}}]$.
  3. `SimulationWorker` and `OptimizationWorker` execute asynchronously without blocking the event loop, honor `Cancel` within $< 0.5\text{ s}$, and `"Apply to Direct Computation"` transfers solved $(T_{\max}, P_{\text{core}}, P_{\text{cap},1..3})$ accurately into Direct Computation widgets.
  4. GUI Save/Load `.json` presets and Export `.csv`/`.txt` dialogs round-trip cleanly with zero unhandled exceptions on invalid user expressions or failed draws, passing `QT_QPA_PLATFORM=offscreen uv run pytest` and `uv run ruff check .`.
