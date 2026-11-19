"""HARF-drawing core package: Python port of the MATLAB HARF drawing app.

Modules
-------
``src.models``      typed inputs, derived targets, structured results.
``src.expressions`` safe MATLAB/Python math expression compiler.
``src.materials``   clamped material/furnace evaluators.
``src.physics``     jacket & capillary ODEs, events, dense sampling.
``src.scaling``     cane/stack -> fiber scaling stages & presets.
``src.solvers``     tension / pressure root-finds, optimization.
``src.io_utils``   CSV/TXT export and JSON config persistence.

Units follow SI conventions (m, s, Pa, N, K) with GUI-facing fields
carrying explicit suffixes (``_C`` for degrees Celsius, ``_um`` for
micrometres). Angular frequency is omega = 2 pi f where applicable.
Complex fields use complex128 with explicit (N_modes, N_grid)
batch/mode-major ordering.
"""

from src.expressions import CompiledExpression, ExpressionError
from src.io_utils import (
    default_user_config_path,
    export_results_table,
    load_config_json,
    read_results_table,
    save_config_json,
)
from src.materials import MaterialProperty, MaterialSet, validate_materials
from src.models import (
    DerivedTargets,
    FailureDiagnostics,
    InputError,
    OptimizationInputs,
    OptimizationResult,
    RingSpec,
    SimulationInputs,
    SimulationResult,
    SimulationStatus,
    TargetError,
    compute_derived_targets,
    default_optimization_inputs,
    default_simulation_inputs,
)
from src.physics import (
    CAPILLARY_ATOL,
    DEFAULT_DZ,
    DEFAULT_RTOL,
    JACKET_ATOL,
    CapillaryModel,
    ComputationCancelled,
    JacketModel,
    assemble_simulation_result,
    rhs_unreduced,
    solve_capillary,
    solve_jacket,
    uniform_grid,
)
from src.scaling import (
    CO_FED_VACUUM,
    LEGACY_EXCEL,
    CaneGeometry,
    CouplingMode,
    FiberGeometry,
    InverseSubMode,
    JacketGeometry,
    PulledScalingState,
    ScalingError,
    ScalingFactors,
    ScalingFeasibility,
    ScalingPreset,
    ScalingResult,
    SolverHandoff,
    StackGeometry,
    StageGeometry,
    auto_fit_scaling_factors,
    cane_to_stack,
    compute_forward_scaling,
    get_scaling_presets,
    inverse_jacket_od,
    scaling_to_solver_inputs,
    solve_inverse_cane,
    solver_inputs_to_scaling,
    stack_to_cane,
)
from src.solvers import (
    DeltaPSolution,
    TensionSolution,
    run_direct_simulation,
    run_optimization,
    solve_capillary_delta_p,
    solve_initial_tension,
)

__all__ = [
    "CAPILLARY_ATOL",
    "CO_FED_VACUUM",
    "DEFAULT_DZ",
    "DEFAULT_RTOL",
    "JACKET_ATOL",
    "LEGACY_EXCEL",
    "CaneGeometry",
    "CapillaryModel",
    "CompiledExpression",
    "ComputationCancelled",
    "CouplingMode",
    "DeltaPSolution",
    "DerivedTargets",
    "ExpressionError",
    "FailureDiagnostics",
    "FiberGeometry",
    "InputError",
    "InverseSubMode",
    "JacketGeometry",
    "JacketModel",
    "MaterialProperty",
    "MaterialSet",
    "OptimizationInputs",
    "OptimizationResult",
    "PulledScalingState",
    "RingSpec",
    "ScalingError",
    "ScalingFactors",
    "ScalingFeasibility",
    "ScalingPreset",
    "ScalingResult",
    "SimulationInputs",
    "SimulationResult",
    "SimulationStatus",
    "SolverHandoff",
    "StackGeometry",
    "StageGeometry",
    "TargetError",
    "TensionSolution",
    "assemble_simulation_result",
    "auto_fit_scaling_factors",
    "cane_to_stack",
    "compute_derived_targets",
    "compute_forward_scaling",
    "default_optimization_inputs",
    "default_simulation_inputs",
    "default_user_config_path",
    "export_results_table",
    "get_scaling_presets",
    "inverse_jacket_od",
    "load_config_json",
    "read_results_table",
    "rhs_unreduced",
    "run_direct_simulation",
    "run_optimization",
    "save_config_json",
    "scaling_to_solver_inputs",
    "solve_capillary",
    "solve_capillary_delta_p",
    "solve_initial_tension",
    "solve_inverse_cane",
    "solve_jacket",
    "solver_inputs_to_scaling",
    "stack_to_cane",
    "uniform_grid",
    "validate_materials",
]
