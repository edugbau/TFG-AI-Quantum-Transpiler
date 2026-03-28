"""
mo_module — Módulo 3: Optimización Multiobjetivo de Layouts
============================================================

Módulo de optimización multiobjetivo para el TFG
"Transpilación Cuántica Híbrida".

Este paquete implementa la optimización de layouts de qubits
(mapeo lógico → físico) mediante algoritmos evolutivos multiobjetivo
(NSGA-II y MOEA/D), generando un frente de Pareto de soluciones
que equilibran múltiples criterios de calidad.

Sub-módulos:

  - **encoding**: Codificación del layout como individuo evolutivo,
    operadores genéticos (sampling, crossover, mutación) adaptados
    a permutaciones parciales.
  - **fitness**: Funciones de fitness extensibles (profundidad,
    CNOTs, tasas de error, decoherencia, conectividad) con patrón
    Strategy y evaluador compuesto.
  - **optimizer**: Algoritmos evolutivos (NSGA-II, MOEA/D) vía pymoo,
    configuración centralizada, pipeline de optimización.
  - **pareto**: Análisis del frente de Pareto (hipervolumen, knee
    point, selección de soluciones, visualización).

Uso típico::

    from src.mo_module import (
        optimize_layout,
        optimize_layout_quick,
        analyze_pareto_front,
        plot_pareto_front_2d,
        OptimizerConfig,
    )
    from src.qiskit_interface import create_ghz_circuit, get_backend

    # Circuito y backend
    circuit = create_ghz_circuit(5)
    backend = get_backend("fake_torino")

    # Optimizar con configuración por defecto
    result = optimize_layout(circuit, backend=backend)
    print(result.summary())

    # Analizar el frente de Pareto
    analysis = analyze_pareto_front(result)
    print(analysis["metrics"].summary())

    # Seleccionar el mejor layout (compromiso)
    best_layout = result.get_compromise_layout()

Dependencias:
  - pymoo >= 0.6 (NSGA-II, MOEA/D, indicadores)
  - numpy >= 2.0
  - scipy >= 1.10
  - matplotlib >= 3.8 (para visualización)
  - src.qiskit_interface (circuitos, backends, transpilación)
"""

# ===================================================================
#  Re-exportaciones públicas
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


# ===================================================================
#  __all__ — API pública explícita
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
]
