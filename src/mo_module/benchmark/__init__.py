"""
benchmark — Submódulo de benchmarking para el optimizador MO
=============================================================

Proporciona una interfaz sencilla para evaluar estadísticamente
el algoritmo de optimización multiobjetivo ejecutándolo sobre
múltiples circuitos y semillas.

Uso rápido::

    from src.mo_module.benchmark import run_benchmark

    # Benchmark rápido: 4 circuitos × 10 semillas
    results = run_benchmark(n_seeds=10)
    print(results.summary())

Uso con análisis estadístico::

    from src.mo_module.benchmark import (
        run_benchmark,
        analyze_results,
    )

    results = run_benchmark(n_seeds=30)
    report  = analyze_results(results)
    print(report.to_text())

Uso personalizado::

    from src.mo_module.benchmark import (
        BenchmarkRunner,
        get_default_circuits,
        make_custom_circuit,
        analyze_results,
    )
    from qiskit import QuantumCircuit
    from src.mo_module import OptimizerConfig

    # Añadir un circuito propio
    qc = QuantumCircuit(3, name="mi_circuito")
    qc.h(0); qc.cx(0, 1); qc.cx(1, 2)

    circuits = get_default_circuits()
    circuits.append(make_custom_circuit("custom", qc))

    runner = BenchmarkRunner(
        circuits=circuits,
        seeds=list(range(30)),
        config=OptimizerConfig(
            population_size=50,
            n_generations=100,
            prob_swap_mutation=0.3,
            prob_replace_mutation=0.7,
            verbose=False,
        ),
    )
    results = runner.run()
    report  = analyze_results(results)
    print(report.to_text())

Submódulos internos:
  - ``circuits``: suite de circuitos de prueba.
  - ``runner``:   motor de ejecución (circuitos × semillas).
  - ``analysis``: análisis estadístico y generación de informes.

Autor: Eduardo González Bautista
Fecha: 2026-02-24
"""

# --- circuits ---
from .circuits import (
    BenchmarkCircuit,
    DEFAULT_BENCHMARK_CIRCUITS,
    get_default_circuits,
    get_circuits_by_tag,
    make_custom_circuit,
)

# --- runner ---
from .runner import (
    BenchmarkRun,
    BenchmarkResultSet,
    BenchmarkRunner,
)

# --- layout_campaigns ---
from .layout_campaigns import (
    run_layout_selection_campaign,
    summarize_layout_campaign,
)


# ===========================================================================
#  Función de conveniencia
# ===========================================================================


def run_benchmark(
    n_seeds: int = 30,
    backend_name: str = "fake_torino",
    circuits: list[BenchmarkCircuit] | None = None,
    population_size: int = 30,
    n_generations: int = 50,
) -> BenchmarkResultSet:
    """Ejecuta un benchmark con parámetros sencillos.

    Interfaz de una sola llamada para el caso de uso más habitual.

    Args:
        n_seeds:
            Número de semillas (30 recomendado para análisis inicial,
            10 para exploración rápida).
        backend_name:
            Backend Fake (``"fake_torino"``, ``"fake_sherbrooke"``…).
        circuits:
            Lista de ``BenchmarkCircuit``. Si es ``None``, usa la
            suite por defecto (GHZ-5, QFT-4, Random-4, Clifford-4).
        population_size:
            Tamaño de la población del optimizador.
        n_generations:
            Número de generaciones del optimizador.

    Returns:
        ``BenchmarkResultSet`` listo para analizar con
        ``analyze_results()``.
    """
    from src.mo_module.optimizer import OptimizerConfig

    runner = BenchmarkRunner(
        circuits=circuits or get_default_circuits(),
        seeds=list(range(n_seeds)),
        backend_name=backend_name,
        config=OptimizerConfig(
            algorithm="nsga2",
            population_size=population_size,
            n_generations=n_generations,
            objectives=["depth", "cnot_count"],
            prob_swap_mutation=0.3,
            prob_replace_mutation=0.7,
            verbose=False,
        ),
    )
    return runner.run()


def __getattr__(name: str):
    if name in {
        "ObjectiveStats",
        "CircuitAnalysis",
        "BenchmarkReport",
        "compute_objective_stats",
        "analyze_results",
    }:
        from . import analysis as _analysis

        return getattr(_analysis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # circuits
    "BenchmarkCircuit",
    "DEFAULT_BENCHMARK_CIRCUITS",
    "get_default_circuits",
    "get_circuits_by_tag",
    "make_custom_circuit",
    # runner
    "BenchmarkRun",
    "BenchmarkResultSet",
    "BenchmarkRunner",
    # analysis
    "ObjectiveStats",
    "CircuitAnalysis",
    "BenchmarkReport",
    "compute_objective_stats",
    "analyze_results",
    # layout campaigns
    "run_layout_selection_campaign",
    "summarize_layout_campaign",
    # conveniencia
    "run_benchmark",
]
