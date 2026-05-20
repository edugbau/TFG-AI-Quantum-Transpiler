"""
mo_module - optimizacion multiobjetivo de layouts
==================================================

Este paquete agrupa la busqueda evolutiva de layouts, el analisis de
Pareto, el ajuste de hiperparametros y el tooling de benchmark local.
La fachada publica sirve como entrada rapida a:

  - **encoding**: representacion del layout y operadores geneticos.
  - **fitness**: evaluacion de candidatos y cache de transpilacion.
  - **optimizer**: orquestacion de NSGA-II / MOEA/D.
  - **pareto**: seleccion e interpretacion del frente.
  - **tuning**: ajuste con Optuna.
  - **benchmark**: evaluacion estadistica y campañas locales.
"""

# ===================================================================
#  Re-exportaciones publicas
# ===================================================================

# --- encoding ---
from .encoding import (
    LayoutSearchSpace,
    LayoutSampling,
    LayoutCrossover,
    DPXCrossover,
    LayoutMutation,
    validate_layout,
    repair_layout,
    layout_to_list,
    layouts_from_population,
    random_layout,
)

# --- fitness ---
from .fitness import (
    FitnessFunction,
    DepthFitness,
    CnotCountFitness,
    AVAILABLE_FITNESS_FUNCTIONS,
    get_fitness_function,
    list_available_fitness_functions,
    TranspilationCache,
    FitnessEvaluator,
    PRESET_OBJECTIVES,
    get_preset_objectives,
)

# --- optimizer ---
from .optimizer import (
    DEFAULT_SWAP_MUTATION_CATEGORIES,
    DEFAULT_REPLACE_MUTATION_CATEGORIES,
    OptimizerConfig,
    LayoutOptimizationProblem,
    OptimizationResult,
    create_algorithm,
    optimize_layout,
    optimize_layout_quick,
    compare_layouts,
    print_layout_comparison,
)

# --- pareto ---
from .pareto import (
    ParetoMetrics,
    compute_pareto_metrics,
    select_knee_point,
    select_weighted,
    select_min_objective,
    analyze_pareto_front,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_parallel_coordinates,
)

# --- tuning ---
from .tuning import (
    HyperparameterSpace,
    LayoutTuner,
)

# --- benchmark ---
from .benchmark import (
    run_layout_selection_campaign,
    summarize_layout_campaign,
)


# ===================================================================
#  __all__ - API publica explicita
# ===================================================================
__all__ = [
    # encoding
    "LayoutSearchSpace",
    "LayoutSampling",
    "LayoutCrossover",
    "DPXCrossover",
    "LayoutMutation",
    "validate_layout",
    "repair_layout",
    "layout_to_list",
    "layouts_from_population",
    "random_layout",
    # fitness
    "FitnessFunction",
    "DepthFitness",
    "CnotCountFitness",
    "AVAILABLE_FITNESS_FUNCTIONS",
    "get_fitness_function",
    "list_available_fitness_functions",
    "TranspilationCache",
    "FitnessEvaluator",
    "PRESET_OBJECTIVES",
    "get_preset_objectives",
    # optimizer
    "DEFAULT_SWAP_MUTATION_CATEGORIES",
    "DEFAULT_REPLACE_MUTATION_CATEGORIES",
    "OptimizerConfig",
    "LayoutOptimizationProblem",
    "OptimizationResult",
    "create_algorithm",
    "optimize_layout",
    "optimize_layout_quick",
    "compare_layouts",
    "print_layout_comparison",
    # pareto
    "ParetoMetrics",
    "compute_pareto_metrics",
    "select_knee_point",
    "select_weighted",
    "select_min_objective",
    "analyze_pareto_front",
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_parallel_coordinates",
    # tuning
    "HyperparameterSpace",
    "LayoutTuner",
    # benchmark
    "run_layout_selection_campaign",
    "summarize_layout_campaign",
]
