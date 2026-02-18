"""
optimizer.py — Algoritmo evolutivo multiobjetivo para optimización de layouts
==============================================================================

Módulo 3 del TFG: Optimización Multiobjetivo.

Este fichero implementa el optimizador de layouts basado en algoritmos
evolutivos multiobjetivo (NSGA-II y MOEA/D) usando la librería pymoo.
Es el núcleo del módulo MO y orquesta la búsqueda de layouts óptimos.

Pipeline de optimización:
  1. Se recibe un circuito cuántico y un backend.
  2. Se configura el espacio de búsqueda (``LayoutSearchSpace``).
  3. Se define el problema de optimización (``LayoutOptimizationProblem``).
  4. Se configura el algoritmo evolutivo (NSGA-II o MOEA/D).
  5. Se ejecuta la optimización con los criterios de parada.
  6. Se devuelve un ``OptimizationResult`` con el frente de Pareto
     y los layouts óptimos.

Decisiones de diseño:
  1. **pymoo como motor** — pymoo proporciona implementaciones robustas
     y bien testeadas de NSGA-II y MOEA/D. Además soporta operadores
     custom, lo que permite usar nuestros operadores de permutación.

  2. **Problema como clase pymoo** — ``LayoutOptimizationProblem``
     extiende ``pymoo.core.problem.Problem`` e integra el
     ``FitnessEvaluator`` para evaluar la población.

  3. **Configuración centralizada** — ``OptimizerConfig`` agrupa todos
     los hiperparámetros del optimizador en un dataclass, permitiendo
     serialización y reproducibilidad.

  4. **Factory de algoritmos** — Se usa el patrón Factory para instanciar
     el algoritmo evolutivo por nombre (``"nsga2"`` o ``"moead"``),
     facilitando la experimentación comparativa.

  5. **Resultado tipado** — ``OptimizationResult`` encapsula los layouts
     del frente de Pareto, sus valores de fitness, y metadatos del
     proceso de optimización.

Autor: Eduardo González Bautista
Fecha: 2026-02-18
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit

# ---------------------------------------------------------------------------
# Imports de pymoo 0.6.x
# ---------------------------------------------------------------------------
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

# ---------------------------------------------------------------------------
# Imports internos del proyecto
# ---------------------------------------------------------------------------
from src.qiskit_interface.backend_info import (
    BackendInfo,
    extract_backend_info,
    get_backend,
    get_error_for_layout,
)
from src.qiskit_interface.transpiler import transpile_with_custom_layout
from .encoding import (
    LayoutSearchSpace,
    LayoutSampling,
    LayoutCrossover,
    LayoutMutation,
    layout_to_list,
)
from .fitness import (
    FitnessEvaluator,
    FitnessFunction,
    TranspilationCache,
    get_fitness_function,
    PRESET_OBJECTIVES,
    get_preset_objectives,
)

# ---------------------------------------------------------------------------
# Logger del módulo
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
#  Configuración del optimizador
# ===========================================================================

@dataclass
class OptimizerConfig:
    """Configuración centralizada del optimizador evolutivo.

    Agrupa todos los hiperparámetros necesarios para configurar y
    ejecutar la optimización. Permite serializar y cargar
    configuraciones para reproducibilidad experimental.

    Attributes:
        algorithm:
            Nombre del algoritmo evolutivo. Valores válidos:
            ``"nsga2"`` (NSGA-II) o ``"moead"`` (MOEA/D).
        population_size:
            Número de individuos en la población.
        n_generations:
            Número máximo de generaciones (criterio de parada).
        objectives:
            Lista de nombres de funciones de fitness a optimizar.
            Ver ``fitness.AVAILABLE_FITNESS_FUNCTIONS``.
        optimization_level:
            Nivel de optimización de Qiskit (0–3) para transpilar
            durante la evaluación de fitness.
        prob_crossover:
            Probabilidad de cruce.
        prob_swap_mutation:
            Probabilidad de mutación por swap.
        prob_replace_mutation:
            Probabilidad de mutación por reemplazo.
        seed:
            Semilla global para reproducibilidad.
        verbose:
            Si True, muestra progreso durante la optimización.
    """

    algorithm: str = "nsga2"
    population_size: int = 50
    n_generations: int = 100
    objectives: list[str] = field(
        default_factory=lambda: ["depth", "avg_error_2q"]
    )
    optimization_level: int = 1
    prob_crossover: float = 0.9
    prob_swap_mutation: float = 0.5
    prob_replace_mutation: float = 0.3
    seed: int = 42
    verbose: bool = True

    def __post_init__(self):
        """Valida la configuración."""
        if self.algorithm not in ("nsga2", "moead"):
            raise ValueError(
                f"Algoritmo '{self.algorithm}' no soportado. "
                f"Use 'nsga2' o 'moead'."
            )
        if self.population_size < 4:
            raise ValueError("population_size debe ser al menos 4.")
        if self.n_generations < 1:
            raise ValueError("n_generations debe ser al menos 1.")


# ===========================================================================
#  Problema de optimización pymoo
# ===========================================================================

class LayoutOptimizationProblem(Problem):
    """Problema de optimización de layouts para pymoo.

    Define la evaluación de la función objetivo multiobjetivo que pymoo
    usa internamente. Cada individuo es un layout (array de enteros)
    y el vector de objetivos lo calcula ``FitnessEvaluator``.

    Variables de decisión:
      - Tipo: entero (cada gen es un qubit físico).
      - Longitud: ``num_logical_qubits``.
      - Rango: ``[0, num_physical_qubits)``.
      - Restricción: todos los valores deben ser distintos (se maneja
        via operadores genéticos custom + reparación).

    Attributes:
        search_space: Espacio de búsqueda del problema.
        fitness_evaluator: Evaluador de fitness multiobjetivo.
    """

    def __init__(
        self,
        search_space: LayoutSearchSpace,
        fitness_evaluator: FitnessEvaluator,
    ):
        """
        Args:
            search_space: Espacio de búsqueda (define variables y límites).
            fitness_evaluator: Evaluador compuesto de fitness.
        """
        self.search_space = search_space
        self.fitness_evaluator = fitness_evaluator

        super().__init__(
            n_var=search_space.num_logical_qubits,
            n_obj=fitness_evaluator.n_objectives,
            xl=0,
            xu=search_space.num_physical_qubits - 1,
            type_var=int,
        )

        logger.info(
            "Problema MO configurado: %d variables, %d objetivos (%s)",
            self.n_var,
            self.n_obj,
            fitness_evaluator.objective_names,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Evalúa la población (llamado internamente por pymoo).

        Args:
            X: ndarray de forma ``(pop_size, n_var)`` con los individuos.
            out: Diccionario de salida pymoo (clave ``"F"`` = objetivos).
        """
        out["F"] = self.fitness_evaluator.evaluate_population(X)


# ===========================================================================
#  Factory de algoritmos evolutivos
# ===========================================================================

def _create_nsga2(
    config: OptimizerConfig,
    search_space: LayoutSearchSpace,
) -> NSGA2:
    """Crea una instancia de NSGA-II configurada.

    NSGA-II (Non-dominated Sorting Genetic Algorithm II) es el
    algoritmo multiobjetivo más popular en la literatura. Usa:
      - Ordenación por no-dominancia (fast non-dominated sort).
      - Distancia de crowding para mantener diversidad.
      - Selección por torneo binario.

    Args:
        config: Configuración del optimizador.
        search_space: Espacio de búsqueda.

    Returns:
        Instancia de NSGA2 lista para usar.
    """
    algorithm = NSGA2(
        pop_size=config.population_size,
        sampling=LayoutSampling(search_space),
        crossover=LayoutCrossover(search_space),
        mutation=LayoutMutation(
            search_space,
            prob_swap=config.prob_swap_mutation,
            prob_replace=config.prob_replace_mutation,
        ),
    )

    logger.info(
        "NSGA-II configurado: pop_size=%d",
        config.population_size,
    )

    return algorithm


def _create_moead(
    config: OptimizerConfig,
    search_space: LayoutSearchSpace,
    n_objectives: int,
) -> MOEAD:
    """Crea una instancia de MOEA/D configurada.

    MOEA/D (Multi-Objective Evolutionary Algorithm based on
    Decomposition) descompone el problema multiobjetivo en
    subproblemas escalares usando vectores de peso y resuelve
    cada subproblema cooperativamente.

    Ventajas sobre NSGA-II:
      - Mejor escalabilidad con muchos objetivos (>3).
      - Distribución más uniforme del frente de Pareto.

    Args:
        config: Configuración del optimizador.
        search_space: Espacio de búsqueda.
        n_objectives: Número de objetivos.

    Returns:
        Instancia de MOEAD lista para usar.
    """
    # Generar vectores de referencia para MOEA/D.
    # El número de direcciones debe coincidir o ser cercano al pop_size.
    ref_dirs = get_reference_directions(
        "uniform",
        n_objectives,
        n_partitions=max(config.population_size - 1, n_objectives),
    )

    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=max(15, config.population_size // 5),
        sampling=LayoutSampling(search_space),
        crossover=LayoutCrossover(search_space),
        mutation=LayoutMutation(
            search_space,
            prob_swap=config.prob_swap_mutation,
            prob_replace=config.prob_replace_mutation,
        ),
    )

    logger.info(
        "MOEA/D configurado: ref_dirs=%d, n_neighbors=%d",
        len(ref_dirs),
        max(15, config.population_size // 5),
    )

    return algorithm


def create_algorithm(
    config: OptimizerConfig,
    search_space: LayoutSearchSpace,
    n_objectives: int,
):
    """Factory: crea el algoritmo evolutivo según la configuración.

    Args:
        config: Configuración del optimizador.
        search_space: Espacio de búsqueda.
        n_objectives: Número de objetivos.

    Returns:
        Instancia del algoritmo evolutivo (NSGA2 o MOEAD).

    Raises:
        ValueError: Si el algoritmo no está soportado.
    """
    if config.algorithm == "nsga2":
        return _create_nsga2(config, search_space)
    elif config.algorithm == "moead":
        return _create_moead(config, search_space, n_objectives)
    else:
        raise ValueError(f"Algoritmo no soportado: {config.algorithm}")


# ===========================================================================
#  Resultado de la optimización
# ===========================================================================

@dataclass
class OptimizationResult:
    """Resultado completo de una ejecución del optimizador MO.

    Encapsula los layouts del frente de Pareto, sus valores de fitness,
    y metadatos del proceso de optimización.

    Attributes:
        pareto_layouts:
            Lista de layouts del frente de Pareto. Cada layout es
            ``list[int]``.
        pareto_fitness:
            ndarray de forma ``(n_pareto, n_objectives)`` con los
            valores de fitness del frente de Pareto.
        objective_names:
            Nombres de los objetivos (para etiquetas).
        algorithm_name:
            Nombre del algoritmo utilizado.
        config:
            Configuración usada para la optimización.
        n_generations_run:
            Número de generaciones ejecutadas.
        elapsed_time_s:
            Tiempo total de optimización en segundos.
        backend_name:
            Nombre del backend utilizado.
        circuit_name:
            Nombre del circuito optimizado.
        num_logical_qubits:
            Número de qubits lógicos del circuito.
        cache_stats:
            Estadísticas del caché de transpilación (si se usó).
    """

    pareto_layouts: list[list[int]] = field(default_factory=list)
    pareto_fitness: Optional[np.ndarray] = None
    objective_names: list[str] = field(default_factory=list)
    algorithm_name: str = ""
    config: Optional[OptimizerConfig] = None
    n_generations_run: int = 0
    elapsed_time_s: float = 0.0
    backend_name: str = ""
    circuit_name: str = ""
    num_logical_qubits: int = 0
    cache_stats: dict[str, int] = field(default_factory=dict)

    @property
    def n_pareto_solutions(self) -> int:
        """Número de soluciones en el frente de Pareto."""
        return len(self.pareto_layouts)

    def get_best_layout(self, objective_index: int = 0) -> list[int]:
        """Devuelve el layout con el menor valor en un objetivo dado.

        Args:
            objective_index: Índice del objetivo a considerar.

        Returns:
            Layout con el mejor valor en el objetivo seleccionado.

        Raises:
            IndexError: Si no hay soluciones en el frente de Pareto.
        """
        if not self.pareto_layouts or self.pareto_fitness is None:
            raise IndexError("No hay soluciones en el frente de Pareto.")

        best_idx = int(np.argmin(self.pareto_fitness[:, objective_index]))
        return self.pareto_layouts[best_idx]

    def get_compromise_layout(self) -> list[int]:
        """Devuelve el layout de compromiso (más cercano al punto ideal).

        El punto ideal es el vector de los mejores valores alcanzados
        en cada objetivo. El layout de compromiso minimiza la distancia
        euclidiana al punto ideal en el espacio de objetivos normalizado.

        Returns:
            Layout de compromiso.

        Raises:
            IndexError: Si no hay soluciones.
        """
        if not self.pareto_layouts or self.pareto_fitness is None:
            raise IndexError("No hay soluciones en el frente de Pareto.")

        F = self.pareto_fitness

        # Normalizar al rango [0, 1] por objetivo
        f_min = F.min(axis=0)
        f_max = F.max(axis=0)
        f_range = f_max - f_min
        # Evitar división por cero
        f_range[f_range == 0] = 1.0
        F_norm = (F - f_min) / f_range

        # Punto ideal normalizado = (0, 0, ..., 0)
        distances = np.linalg.norm(F_norm, axis=1)
        best_idx = int(np.argmin(distances))

        return self.pareto_layouts[best_idx]

    def summary(self) -> str:
        """Devuelve un resumen legible del resultado de la optimización."""
        lines = [
            f"  Algoritmo:            {self.algorithm_name}",
            f"  Circuito:             {self.circuit_name}",
            f"  Backend:              {self.backend_name}",
            f"  Qubits lógicos:       {self.num_logical_qubits}",
            f"  Generaciones:         {self.n_generations_run}",
            f"  Tiempo:               {self.elapsed_time_s:.2f} s",
            f"  Soluciones Pareto:    {self.n_pareto_solutions}",
            f"  Objetivos:            {self.objective_names}",
        ]

        if self.pareto_fitness is not None and len(self.pareto_fitness) > 0:
            lines.append("")
            lines.append("  --- Rango de valores en el frente de Pareto ---")
            for i, name in enumerate(self.objective_names):
                col = self.pareto_fitness[:, i]
                lines.append(
                    f"    {name}: [{col.min():.4f}, {col.max():.4f}] "
                    f"(media={col.mean():.4f})"
                )

        if self.cache_stats:
            lines.append("")
            lines.append(f"  Caché: {self.cache_stats}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convierte el resultado a diccionario (para serialización)."""
        result = {
            "algorithm_name": self.algorithm_name,
            "circuit_name": self.circuit_name,
            "backend_name": self.backend_name,
            "num_logical_qubits": self.num_logical_qubits,
            "n_generations_run": self.n_generations_run,
            "elapsed_time_s": self.elapsed_time_s,
            "n_pareto_solutions": self.n_pareto_solutions,
            "objective_names": self.objective_names,
        }
        if self.pareto_fitness is not None:
            for i, name in enumerate(self.objective_names):
                col = self.pareto_fitness[:, i]
                result[f"pareto_{name}_min"] = float(col.min())
                result[f"pareto_{name}_max"] = float(col.max())
                result[f"pareto_{name}_mean"] = float(col.mean())
        return result


# ===========================================================================
#  Función principal de optimización
# ===========================================================================

def optimize_layout(
    circuit: QuantumCircuit,
    backend=None,
    backend_name: str = "fake_torino",
    config: Optional[OptimizerConfig] = None,
) -> OptimizationResult:
    """Ejecuta la optimización multiobjetivo de layouts.

    Esta es la **función principal** del módulo MO. Orquesta todo el
    pipeline: configuración del espacio de búsqueda, definición del
    problema, configuración del algoritmo y ejecución.

    Flujo:
      1. Obtener backend e info del backend.
      2. Configurar el espacio de búsqueda (``LayoutSearchSpace``).
      3. Crear el evaluador de fitness (``FitnessEvaluator``).
      4. Definir el problema pymoo (``LayoutOptimizationProblem``).
      5. Crear el algoritmo evolutivo (NSGA-II o MOEA/D).
      6. Ejecutar la optimización.
      7. Extraer y empaquetar el frente de Pareto.

    Args:
        circuit:
            Circuito cuántico a optimizar.
        backend:
            Backend de destino. Si es None, se instancia con
            ``backend_name``.
        backend_name:
            Nombre del backend (si ``backend`` es None).
        config:
            Configuración del optimizador. Si es None, se usa la
            configuración por defecto.

    Returns:
        OptimizationResult con los layouts del frente de Pareto.
    """
    if config is None:
        config = OptimizerConfig()

    # --- 1. Backend ---
    if backend is None:
        backend = get_backend(backend_name)
    actual_backend_name = getattr(backend, "name", backend_name)

    backend_info = extract_backend_info(backend)

    logger.info(
        "Optimizando layout para circuito '%s' (%d qubits) en "
        "backend '%s' con %s",
        circuit.name or "?",
        circuit.num_qubits,
        actual_backend_name,
        config.algorithm.upper(),
    )

    # --- 2. Espacio de búsqueda ---
    search_space = LayoutSearchSpace.from_backend_info(
        backend_info, circuit.num_qubits
    )

    # --- 3. Evaluador de fitness ---
    fitness_evaluator = FitnessEvaluator.from_names(
        objective_names=config.objectives,
        backend_info=backend_info,
        circuit=circuit,
        backend=backend,
        optimization_level=config.optimization_level,
        seed=config.seed,
    )

    # --- 4. Problema pymoo ---
    problem = LayoutOptimizationProblem(
        search_space=search_space,
        fitness_evaluator=fitness_evaluator,
    )

    # --- 5. Algoritmo ---
    algorithm = create_algorithm(
        config=config,
        search_space=search_space,
        n_objectives=fitness_evaluator.n_objectives,
    )

    # --- 6. Criterio de parada ---
    termination = get_termination("n_gen", config.n_generations)

    # --- 7. Ejecutar optimización ---
    t_start = time.perf_counter()

    result = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=config.seed,
        verbose=config.verbose,
    )

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    # --- 8. Extraer frente de Pareto ---
    pareto_layouts: list[list[int]] = []
    pareto_fitness: Optional[np.ndarray] = None

    if result.X is not None:
        X = result.X
        F = result.F

        # Si result.X es 1D (una sola solución), expandir dimensión
        if X.ndim == 1:
            X = X.reshape(1, -1)
            F = F.reshape(1, -1)

        pareto_layouts = [layout_to_list(X[i]) for i in range(len(X))]
        pareto_fitness = F

    logger.info(
        "Optimización completada en %.2f s — %d soluciones en el "
        "frente de Pareto",
        elapsed,
        len(pareto_layouts),
    )

    # --- 9. Estadísticas del caché ---
    cache_stats = {}
    if fitness_evaluator.transpilation_cache is not None:
        cache_stats = fitness_evaluator.transpilation_cache.stats

    # --- 10. Construir resultado ---
    opt_result = OptimizationResult(
        pareto_layouts=pareto_layouts,
        pareto_fitness=pareto_fitness,
        objective_names=fitness_evaluator.objective_names,
        algorithm_name=config.algorithm,
        config=config,
        n_generations_run=config.n_generations,
        elapsed_time_s=elapsed,
        backend_name=actual_backend_name,
        circuit_name=circuit.name or "unnamed",
        num_logical_qubits=circuit.num_qubits,
        cache_stats=cache_stats,
    )

    return opt_result


def optimize_layout_quick(
    circuit: QuantumCircuit,
    backend=None,
    backend_name: str = "fake_torino",
    preset: str = "hardware_only",
    population_size: int = 30,
    n_generations: int = 50,
    seed: int = 42,
) -> OptimizationResult:
    """Versión simplificada de ``optimize_layout`` con configuración rápida.

    Ideal para experimentación interactiva y notebooks. Usa presets
    de objetivos predefinidos.

    Args:
        circuit: Circuito a optimizar.
        backend: Backend (opcional).
        backend_name: Nombre del backend.
        preset: Nombre del preset de objetivos
            (``"hardware_only"``, ``"transpilation_basic"``, etc.).
        population_size: Tamaño de la población.
        n_generations: Número de generaciones.
        seed: Semilla.

    Returns:
        OptimizationResult con los layouts optimizados.
    """
    objectives = get_preset_objectives(preset)

    config = OptimizerConfig(
        algorithm="nsga2",
        population_size=population_size,
        n_generations=n_generations,
        objectives=objectives,
        seed=seed,
        verbose=False,
    )

    return optimize_layout(
        circuit=circuit,
        backend=backend,
        backend_name=backend_name,
        config=config,
    )


# ===========================================================================
#  Comparación de layouts
# ===========================================================================

def compare_layouts(
    circuit: QuantumCircuit,
    layouts: dict[str, list[int]],
    backend=None,
    backend_name: str = "fake_torino",
    optimization_level: int = 1,
    seed: int = 42,
) -> list[dict]:
    """Compara múltiples layouts transpilando con cada uno.

    Útil para comparar los layouts del frente de Pareto contra el
    layout por defecto de Qiskit (SABRE) y contra heurísticas como
    ``get_heaviest_hex_layout``.

    Args:
        circuit: Circuito a transpilar.
        layouts: Diccionario ``{nombre_layout: layout}``.
        backend: Backend a usar.
        backend_name: Nombre del backend (si backend es None).
        optimization_level: Nivel de optimización de Qiskit.
        seed: Semilla.

    Returns:
        Lista de diccionarios con métricas de cada layout,
        lista para ``pd.DataFrame(lista)``.
    """
    if backend is None:
        backend = get_backend(backend_name)

    backend_info = extract_backend_info(backend)
    rows: list[dict] = []

    for name, layout in layouts.items():
        result = transpile_with_custom_layout(
            circuit=circuit,
            layout=layout,
            backend=backend,
            optimization_level=optimization_level,
            seed=seed,
        )

        error_stats = get_error_for_layout(backend_info, layout)

        row = {
            "layout_name": name,
            "layout": layout,
            "depth": result.transpiled_metrics.depth if result.transpiled_metrics else None,
            "two_qubit_gates": (
                result.transpiled_metrics.two_qubit_gates
                if result.transpiled_metrics else None
            ),
            "total_gates": (
                result.transpiled_metrics.total_gates
                if result.transpiled_metrics else None
            ),
            "elapsed_time_s": result.elapsed_time_s,
            "avg_error_2q": error_stats["avg_error_2q"],
            "max_error_2q": error_stats["max_error_2q"],
            "avg_t2": error_stats["avg_t2"],
            "num_edges": error_stats["num_available_edges"],
        }
        rows.append(row)

        logger.info(
            "Layout '%s': depth=%s, 2Q=%s, error_2q=%.6f",
            name,
            row["depth"],
            row["two_qubit_gates"],
            row["avg_error_2q"],
        )

    return rows


def print_layout_comparison(rows: list[dict]) -> None:
    """Imprime una comparación de layouts en formato tabular.

    Args:
        rows: Resultados de ``compare_layouts()``.
    """
    print(f"\n{'=' * 90}")
    print(f"  COMPARACIÓN DE LAYOUTS")
    print(f"{'=' * 90}")
    print(
        f"  {'Layout':<20} {'Depth':>8} {'2Q Gates':>10} "
        f"{'Total':>8} {'Err 2Q':>12} {'Edges':>8}"
    )
    print(f"  {'-' * 70}")

    for row in rows:
        print(
            f"  {row['layout_name']:<20} "
            f"{row.get('depth', '-'):>8} "
            f"{row.get('two_qubit_gates', '-'):>10} "
            f"{row.get('total_gates', '-'):>8} "
            f"{row.get('avg_error_2q', 0):>12.6f} "
            f"{row.get('num_edges', 0):>8}"
        )

    print(f"{'=' * 90}\n")
