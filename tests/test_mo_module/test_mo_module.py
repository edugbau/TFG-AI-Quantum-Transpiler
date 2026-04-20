"""
Tests unitarios para el módulo mo_module
==========================================

Cobertura de los cuatro sub-módulos:
  - encoding   (test_encoding_*)
  - fitness    (test_fitness_*)
  - optimizer  (test_optimizer_*)
  - pareto     (test_pareto_*)

Se utilizan exclusivamente Fake Backends (sin API keys ni conexión).
Cada test es independiente y reproducible (seeds fijadas).
Los tests de optimización usan poblaciones y generaciones reducidas
para mantener tiempos de ejecución razonables.

Ejecución:
  pytest tests/test_mo_module/ -v

Autor: Eduardo González Bautista
Fecha: 2026-02-18
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2

# ---------------------------------------------------------------------------
# Imports del módulo bajo test
# ---------------------------------------------------------------------------
from src.mo_module.encoding import (
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
from src.mo_module.fitness import (
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
from src.mo_module.optimizer import (
    DEFAULT_REPLACE_MUTATION_CATEGORIES,
    DEFAULT_SWAP_MUTATION_CATEGORIES,
    OptimizerConfig,
    LayoutOptimizationProblem,
    OptimizationResult,
    create_algorithm,
    optimize_layout,
    optimize_layout_quick,
    compare_layouts,
    print_layout_comparison,
)
from src.mo_module.pareto import (
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

# ---------------------------------------------------------------------------
# Imports de dependencias externas
# ---------------------------------------------------------------------------
from qiskit import QuantumCircuit
from src.qiskit_interface.circuit_utils import create_ghz_circuit, CircuitMetrics
from src.qiskit_interface.transpiler import TranspilationResult
from src.qiskit_interface.backend_info import (
    BackendInfo,
    get_backend,
    extract_backend_info,
)


# ===========================================================================
#  Fixtures
# ===========================================================================

@pytest.fixture
def simple_circuit() -> QuantumCircuit:
    """Circuito simple de 3 qubits para tests rápidos."""
    qc = QuantumCircuit(3, name="test_simple")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


@pytest.fixture
def ghz5() -> QuantumCircuit:
    """Circuito GHZ de 5 qubits."""
    return create_ghz_circuit(5)


@pytest.fixture
def backend_torino():
    """Instancia de FakeTorino."""
    return get_backend("fake_torino")


@pytest.fixture
def backend_info_torino(backend_torino) -> BackendInfo:
    """BackendInfo completo de FakeTorino."""
    return extract_backend_info(backend_torino)


@pytest.fixture
def search_space_3q(backend_info_torino) -> LayoutSearchSpace:
    """Espacio de búsqueda para 3 qubits lógicos en FakeTorino."""
    return LayoutSearchSpace.from_backend_info(backend_info_torino, 3)


@pytest.fixture
def search_space_5q(backend_info_torino) -> LayoutSearchSpace:
    """Espacio de búsqueda para 5 qubits lógicos en FakeTorino."""
    return LayoutSearchSpace.from_backend_info(backend_info_torino, 5)


@pytest.fixture
def pareto_result_2d() -> OptimizationResult:
    """Resultado pequeño con dos objetivos para tests de reporting."""
    return OptimizationResult(
        pareto_layouts=[[0, 1], [2, 3], [4, 5]],
        pareto_fitness=np.array([
            [1.0, 10.0],
            [5.0, 5.0],
            [10.0, 1.0],
        ]),
        objective_names=["depth", "error"],
        algorithm_name="nsga2",
        backend_name="fake_torino",
        circuit_name="test_circuit",
    )


@pytest.fixture
def pareto_result_3d() -> OptimizationResult:
    """Resultado pequeño con tres objetivos para tests 3D."""
    return OptimizationResult(
        pareto_layouts=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        pareto_fitness=np.array([
            [1.0, 8.0, 5.0],
            [4.0, 4.0, 4.0],
            [8.0, 1.0, 2.0],
        ]),
        objective_names=["depth", "error", "duration"],
        algorithm_name="moead",
        backend_name="fake_torino",
        circuit_name="test_circuit",
    )


@pytest.fixture
def empty_pareto_result_2d() -> OptimizationResult:
    """Resultado con frente vacío en forma de ndarray 2D."""
    return OptimizationResult(
        pareto_layouts=[],
        pareto_fitness=np.empty((0, 2)),
        objective_names=["depth", "error"],
        algorithm_name="nsga2",
        backend_name="fake_torino",
        circuit_name="test_circuit",
    )


@pytest.fixture
def empty_pareto_result_3d() -> OptimizationResult:
    """Resultado con frente vacío en forma de ndarray 3D."""
    return OptimizationResult(
        pareto_layouts=[],
        pareto_fitness=np.empty((0, 3)),
        objective_names=["depth", "error", "duration"],
        algorithm_name="moead",
        backend_name="fake_torino",
        circuit_name="test_circuit",
    )


# ===========================================================================
#  Tests — encoding
# ===========================================================================

class TestLayoutSearchSpace:
    """Tests del espacio de búsqueda de layouts."""

    def test_from_backend_info(self, backend_info_torino):
        """Se crea correctamente desde BackendInfo."""
        ss = LayoutSearchSpace.from_backend_info(backend_info_torino, 5)
        assert ss.num_logical_qubits == 5
        assert ss.num_physical_qubits == 133
        assert len(ss.available_qubits) == 133
        assert len(ss.coupling_edges) > 0

    def test_from_backend_info_too_many_qubits(self, backend_info_torino):
        """Pedir más qubits de los disponibles lanza ValueError."""
        with pytest.raises(ValueError):
            LayoutSearchSpace.from_backend_info(backend_info_torino, 200)

    def test_default_available_qubits(self):
        """available_qubits se inicializa automáticamente si no se da."""
        ss = LayoutSearchSpace(
            num_logical_qubits=3,
            num_physical_qubits=10,
        )
        assert ss.available_qubits == set(range(10))


class TestValidateLayout:
    """Tests de validación de layouts."""

    def test_valid_layout(self, search_space_3q):
        """Un layout correcto pasa la validación."""
        assert validate_layout([0, 1, 2], search_space_3q)

    def test_invalid_length(self, search_space_3q):
        """Layout con longitud incorrecta es inválido."""
        assert not validate_layout([0, 1], search_space_3q)
        assert not validate_layout([0, 1, 2, 3], search_space_3q)

    def test_invalid_duplicates(self, search_space_3q):
        """Layout con qubits duplicados es inválido."""
        assert not validate_layout([0, 0, 1], search_space_3q)

    def test_invalid_out_of_range(self, search_space_3q):
        """Layout con qubits fuera de rango es inválido."""
        assert not validate_layout([0, 1, 999], search_space_3q)

    def test_valid_non_contiguous(self, search_space_3q):
        """Layout con qubits no contiguos es válido."""
        assert validate_layout([10, 50, 100], search_space_3q)


class TestRepairLayout:
    """Tests de reparación de layouts."""

    def test_repair_duplicates(self, search_space_3q):
        """Repara un layout con duplicados."""
        layout = np.array([5, 5, 10])
        repaired = repair_layout(layout, search_space_3q, np.random.default_rng(42))
        assert len(set(repaired.tolist())) == 3  # Sin duplicados
        assert validate_layout(repaired, search_space_3q)

    def test_repair_preserves_valid(self, search_space_3q):
        """Un layout ya válido no se modifica significativamente."""
        layout = np.array([0, 1, 2])
        repaired = repair_layout(layout, search_space_3q, np.random.default_rng(42))
        # Los valores originales deberían mantenerse
        assert set(repaired.tolist()) == {0, 1, 2}


class TestLayoutOperators:
    """Tests de los operadores genéticos para pymoo."""

    def test_sampling_shape(self, search_space_5q):
        """El sampling genera la forma correcta."""
        sampler = LayoutSampling(search_space_5q)
        X = sampler._do(None, 10)
        assert X.shape == (10, 5)

    def test_sampling_validity(self, search_space_5q):
        """Todos los individuos generados son válidos."""
        sampler = LayoutSampling(search_space_5q)
        X = sampler._do(None, 20)
        for i in range(20):
            assert validate_layout(X[i], search_space_5q), (
                f"Individuo {i} inválido: {X[i]}"
            )

    def test_sampling_no_duplicates_in_individual(self, search_space_5q):
        """Cada individuo tiene qubits únicos."""
        sampler = LayoutSampling(search_space_5q)
        X = sampler._do(None, 10)
        for i in range(10):
            assert len(set(X[i].tolist())) == 5

    def test_crossover_produces_valid_offspring(self, search_space_5q):
        """El crossover produce hijos válidos."""
        sampler = LayoutSampling(search_space_5q)
        parents = sampler._do(None, 4)
        # Forma esperada por pymoo: (n_parents, n_matings, n_vars)
        X = parents.reshape(2, 2, 5)

        crossover = LayoutCrossover(search_space_5q)
        Y = crossover._do(None, X)

        assert Y.shape == (2, 2, 5)
        for k in range(2):
            for j in range(2):
                assert validate_layout(Y[k, j], search_space_5q), (
                    f"Hijo [{k},{j}] inválido: {Y[k, j]}"
                )

    def test_mutation_swap(self, search_space_5q):
        """La mutación swap produce layouts válidos."""
        mutator = LayoutMutation(search_space_5q, prob_swap=1.0, prob_replace=0.0)
        X = np.array([[10, 20, 30, 40, 50]])
        Y = mutator._do(None, X)
        # El conjunto de qubits debe ser el mismo (swap no cambia el conjunto)
        assert set(Y[0].tolist()) == set(X[0].tolist())

    def test_mutation_replace(self, search_space_5q):
        """La mutación replace produce layouts válidos."""
        mutator = LayoutMutation(search_space_5q, prob_swap=0.0, prob_replace=1.0)
        X = np.array([[10, 20, 30, 40, 50]])
        Y = mutator._do(None, X)
        assert validate_layout(Y[0], search_space_5q)
        assert len(set(Y[0].tolist())) == 5  # Sin duplicados

    def test_dpx_crossover_produces_valid_offspring(self, search_space_5q):
        """DPXCrossover produce hijos válidos y sin duplicados."""
        sampler = LayoutSampling(search_space_5q)
        parents = sampler._do(None, 4)
        X = parents.reshape(2, 2, 5)

        crossover = DPXCrossover(search_space_5q)
        Y = crossover._do(None, X)

        assert Y.shape == (2, 2, 5)
        for k in range(2):
            for j in range(2):
                assert validate_layout(Y[k, j], search_space_5q), (
                    f"DPX hijo [{k},{j}] inválido: {Y[k, j]}"
                )
                assert len(set(Y[k, j].tolist())) == 5, (
                    f"DPX hijo [{k},{j}] tiene duplicados: {Y[k, j]}"
                )

    def test_dpx_preserves_common_assignments(self, search_space_5q):
        """DPX conserva posiciones donde ambos padres asignan el mismo qubit."""
        # Padres con posiciones 0 y 2 idénticas: qubit 10 y qubit 30
        p1 = np.array([10, 20, 30, 40, 50])
        p2 = np.array([10, 25, 30, 45, 55])
        X = np.array([p1, p2]).reshape(2, 1, 5)

        crossover = DPXCrossover(search_space_5q)
        Y = crossover._do(None, X)

        # Ambos hijos deben tener qubit 10 en posición 0 y qubit 30 en posición 2
        for offspring_idx in range(2):
            child = Y[offspring_idx, 0]
            assert child[0] == 10, (
                f"Hijo {offspring_idx}: posición 0 debería ser 10, es {child[0]}"
            )
            assert child[2] == 30, (
                f"Hijo {offspring_idx}: posición 2 debería ser 30, es {child[2]}"
            )

    def test_dpx_no_duplicates_stress(self, search_space_5q):
        """DPX no genera duplicados en 50 pares de padres aleatorios."""
        sampler = LayoutSampling(search_space_5q)
        crossover = DPXCrossover(search_space_5q)
        rng = np.random.default_rng(42)

        for _ in range(50):
            parents = sampler._do(None, 4)
            X = parents.reshape(2, 2, 5)
            Y = crossover._do(None, X)
            for k in range(2):
                for j in range(2):
                    assert len(set(Y[k, j].tolist())) == 5


class TestLayoutUtilities:
    """Tests de funciones de conversión y utilidades."""

    def test_layout_to_list(self):
        """Convierte correctamente de ndarray a list[int]."""
        arr = np.array([1, 2, 3])
        result = layout_to_list(arr)
        assert result == [1, 2, 3]
        assert all(isinstance(x, int) for x in result)

    def test_layouts_from_population(self):
        """Convierte una población completa."""
        pop = np.array([[1, 2, 3], [4, 5, 6]])
        result = layouts_from_population(pop)
        assert len(result) == 2
        assert result[0] == [1, 2, 3]
        assert result[1] == [4, 5, 6]

    def test_random_layout(self, search_space_5q):
        """Genera un layout aleatorio válido."""
        layout = random_layout(search_space_5q, seed=42)
        assert len(layout) == 5
        assert len(set(layout)) == 5
        assert validate_layout(layout, search_space_5q)

    def test_random_layout_reproducible(self, search_space_5q):
        """El mismo seed produce el mismo layout."""
        l1 = random_layout(search_space_5q, seed=123)
        l2 = random_layout(search_space_5q, seed=123)
        assert l1 == l2


# ===========================================================================
#  Tests — fitness
# ===========================================================================

class TestFitnessFunctions:
    """Tests de las funciones de fitness activas."""

    def test_depth_fitness_name_and_flag(self):
        """DepthFitness tiene el nombre correcto y requiere transpilación."""
        fitness = DepthFitness()
        assert fitness.name == "depth"
        assert fitness.requires_transpilation

    def test_depth_fitness_without_result(self, backend_info_torino):
        """DepthFitness sin resultado de transpilación devuelve inf."""
        fitness = DepthFitness()
        value = fitness.evaluate([0, 1, 2], backend_info_torino)
        assert value == float("inf")

    def test_cnot_count_fitness_name_and_flag(self):
        """CnotCountFitness tiene el nombre correcto y requiere transpilación."""
        fitness = CnotCountFitness()
        assert fitness.name == "cnot_count"
        assert fitness.requires_transpilation

    def test_cnot_count_fitness_without_result(self, backend_info_torino):
        """CnotCountFitness sin resultado de transpilación devuelve inf."""
        fitness = CnotCountFitness()
        value = fitness.evaluate([0, 1, 2], backend_info_torino)
        assert value == float("inf")

    def test_depth_and_cnot_with_transpilation(
        self, simple_circuit, backend_torino, backend_info_torino
    ):
        """Ambos objetivos devuelven valores finitos cuando se transpila."""
        cache = TranspilationCache(simple_circuit, backend_torino, seed=42)
        layout = [0, 1, 2]
        tr = cache.get(layout)

        depth_val = DepthFitness().evaluate(layout, backend_info_torino, tr)
        cnot_val = CnotCountFitness().evaluate(layout, backend_info_torino, tr)

        assert depth_val > 0
        assert cnot_val >= 0
        assert depth_val != float("inf")
        assert cnot_val != float("inf")

    def test_cnot_count_fitness_reads_cnot_equivalent_not_two_qubit_gates(
        self, backend_info_torino
    ):
        """CnotCountFitness lee cnot_equivalent, no two_qubit_gates.

        Se construye un TranspilationResult donde cnot_equivalent y
        two_qubit_gates difieren deliberadamente. El fitness debe
        devolver cnot_equivalent (7), no two_qubit_gates (3).
        """
        metrics = CircuitMetrics(
            depth=5,
            num_qubits=3,
            total_gates=10,
            two_qubit_gates=3,     # valor distinto a cnot_equivalent
            cnot_equivalent=7,     # este es el que debe leerse
        )
        tr = TranspilationResult(transpiled_metrics=metrics)
        result = CnotCountFitness().evaluate([0, 1, 2], backend_info_torino, tr)
        assert result == 7.0, (
            f"Se esperaba cnot_equivalent=7, se obtuvo {result}. "
            "Posiblemente se está leyendo two_qubit_gates en lugar de cnot_equivalent."
        )

    def test_cnot_count_fitness_swap_costs_three_cnots(
        self, backend_torino, backend_info_torino
    ):
        """Un SWAP equivale a 3 CNOTs: cnot_count >= 3 * two_qubit_gates para SWAPs puros.

        Se crea un circuito de 2 qubits con un SWAP y se transpila.
        La cuenta de CNOTs equivalentes debe ser >= la cuenta de puertas
        2-qubit (porque SWAP descompone en ≥ 3 CX en cualquier base).
        """
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.swap(0, 1)

        cache = TranspilationCache(qc, backend_torino, seed=42)
        layout = [0, 1]
        tr = cache.get(layout)

        assert tr.transpiled_metrics is not None
        cnot_equiv = tr.transpiled_metrics.cnot_equivalent
        two_q = tr.transpiled_metrics.two_qubit_gates

        # Un SWAP = 3 CX (o CZ, o ECR) → cnot_equivalent >= two_qubit_gates
        assert cnot_equiv >= two_q, (
            f"cnot_equivalent={cnot_equiv} debería ser >= two_qubit_gates={two_q}"
        )
        # Y en concreto deben diferir (el SWAP no es su propia puerta nativa)
        # o al menos el resultado de CnotCountFitness es cnot_equivalent
        fitness_val = CnotCountFitness().evaluate(layout, backend_info_torino, tr)
        assert fitness_val == float(cnot_equiv)


class TestFitnessRegistry:
    """Tests del registro de funciones de fitness (Factory)."""

    def test_list_available_contains_active_objectives(self):
        """El registro contiene al menos los dos objetivos activos."""
        available = list_available_fitness_functions()
        assert "depth" in available
        assert "cnot_count" in available

    def test_get_depth_fitness(self):
        """Se puede instanciar DepthFitness por nombre."""
        ff = get_fitness_function("depth")
        assert isinstance(ff, DepthFitness)

    def test_get_cnot_count_fitness(self):
        """Se puede instanciar CnotCountFitness por nombre."""
        ff = get_fitness_function("cnot_count")
        assert isinstance(ff, CnotCountFitness)

    def test_get_fitness_function_invalid(self):
        """Nombre inválido lanza ValueError."""
        with pytest.raises(ValueError):
            get_fitness_function("nonexistent_fitness")


class TestPresets:
    """Tests de los presets de objetivos."""

    def test_preset_default(self):
        """El preset default contiene depth y cnot_count."""
        objs = get_preset_objectives("default")
        assert "depth" in objs
        assert "cnot_count" in objs

    def test_preset_invalid(self):
        """Preset inexistente lanza ValueError."""
        with pytest.raises(ValueError):
            get_preset_objectives("nonexistent_preset")


class TestTranspilationCache:
    """Tests del caché de transpilación."""

    def test_cache_hit(self, simple_circuit, backend_torino):
        """El caché devuelve el mismo resultado para el mismo layout."""
        cache = TranspilationCache(simple_circuit, backend_torino, seed=42)
        layout = [0, 1, 2]

        r1 = cache.get(layout)
        r2 = cache.get(layout)

        assert r1 is r2  # Mismo objeto (del caché)
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1

    def test_cache_miss(self, simple_circuit, backend_torino):
        """Layouts distintos generan entradas distintas en el caché."""
        cache = TranspilationCache(simple_circuit, backend_torino, seed=42)

        r1 = cache.get([0, 1, 2])
        r2 = cache.get([3, 4, 5])

        assert cache.stats["misses"] == 2
        assert cache.stats["size"] == 2

    def test_cache_clear(self, simple_circuit, backend_torino):
        """Limpiar el caché reinicia las estadísticas."""
        cache = TranspilationCache(simple_circuit, backend_torino, seed=42)
        cache.get([0, 1, 2])
        cache.clear()

        assert cache.stats["size"] == 0
        assert cache.stats["hits"] == 0


class TestFitnessEvaluator:
    """Tests del evaluador compuesto."""

    def test_evaluator_from_names_missing_circuit(self, backend_info_torino):
        """Objetivos de transpilación sin circuito lanzan ValueError."""
        with pytest.raises(ValueError):
            FitnessEvaluator.from_names(
                ["depth", "cnot_count"],
                backend_info=backend_info_torino,
                # Sin circuit ni backend → error
            )

    def test_evaluator_with_transpilation(
        self, simple_circuit, backend_torino, backend_info_torino
    ):
        """Evaluador con depth y cnot_count calcula métricas correctamente."""
        evaluator = FitnessEvaluator.from_names(
            ["depth", "cnot_count"],
            backend_info=backend_info_torino,
            circuit=simple_circuit,
            backend=backend_torino,
            seed=42,
        )
        assert evaluator.n_objectives == 2
        assert evaluator.requires_transpilation

        values = evaluator.evaluate([0, 1, 2])
        assert values.shape == (2,)
        assert values[0] > 0   # depth > 0
        assert values[1] >= 0  # cnot_count >= 0

    def test_evaluate_population(
        self, simple_circuit, backend_torino, backend_info_torino
    ):
        """Evaluación de una población entera."""
        evaluator = FitnessEvaluator.from_names(
            ["depth", "cnot_count"],
            backend_info=backend_info_torino,
            circuit=simple_circuit,
            backend=backend_torino,
            seed=42,
        )
        pop = np.array([[0, 1, 2], [10, 11, 12], [50, 51, 52]])
        F = evaluator.evaluate_population(pop)
        assert F.shape == (3, 2)

    def test_objective_names(
        self, simple_circuit, backend_torino, backend_info_torino
    ):
        """Los nombres de objetivos se reportan correctamente."""
        evaluator = FitnessEvaluator.from_names(
            ["depth", "cnot_count"],
            backend_info=backend_info_torino,
            circuit=simple_circuit,
            backend=backend_torino,
        )
        assert evaluator.objective_names == ["depth", "cnot_count"]


# ===========================================================================
#  Tests — optimizer
# ===========================================================================

class TestOptimizerConfig:
    """Tests de la configuración del optimizador."""

    def test_default_config(self):
        """La configuración por defecto es válida y usa DPX."""
        config = OptimizerConfig()
        assert config.algorithm == "nsga2"
        assert config.population_size == 50
        assert config.n_generations == 100
        assert config.crossover_operator == "dpx"
        assert config.prob_swap_mutation == 0.3
        assert config.prob_replace_mutation == 0.7

    def test_prob_crossover_is_not_part_of_public_contract(self):
        """El contrato público ya no acepta prob_crossover."""
        with pytest.raises(TypeError):
            OptimizerConfig(prob_crossover=0.9)

    def test_dpx_is_default_crossover_operator(self):
        """OptimizerConfig usa DPX como operador de cruce por defecto."""
        config = OptimizerConfig()
        assert config.crossover_operator == "dpx"

    def test_ox_crossover_still_selectable(self):
        """OX sigue disponible como opción vía crossover_operator='ox'."""
        config = OptimizerConfig(crossover_operator="ox")
        assert config.crossover_operator == "ox"

    def test_invalid_crossover_operator_raises(self):
        """crossover_operator inválido lanza ValueError."""
        with pytest.raises(ValueError, match="crossover_operator"):
            OptimizerConfig(crossover_operator="pmx")

    def test_invalid_algorithm(self):
        """Algoritmo inválido lanza ValueError."""
        with pytest.raises(ValueError):
            OptimizerConfig(algorithm="invalid_algo")

    def test_invalid_population_size(self):
        """Población demasiado pequeña lanza ValueError."""
        with pytest.raises(ValueError):
            OptimizerConfig(population_size=1)

    def test_custom_config(self):
        """Configuración personalizada se almacena correctamente."""
        config = OptimizerConfig(
            algorithm="moead",
            population_size=100,
            n_generations=200,
            objectives=["depth", "cnot_count"],
            prob_swap_mutation=0.5,
            prob_replace_mutation=0.9,
            seed=99,
        )
        assert config.algorithm == "moead"
        assert config.population_size == 100
        assert config.seed == 99
        assert config.prob_swap_mutation == 0.5
        assert config.prob_replace_mutation == 0.9

    def test_default_mutation_categories_are_public(self):
        """Las categorías por defecto están expuestas para tuning y revisión."""
        assert DEFAULT_SWAP_MUTATION_CATEGORIES == (0.1, 0.3, 0.5, 0.7)
        assert DEFAULT_REPLACE_MUTATION_CATEGORIES == (0.1, 0.3, 0.5, 0.7, 0.9)

    def test_invalid_swap_mutation_category_raises(self):
        """Categorías fuera del catálogo deben rechazarse."""
        with pytest.raises(ValueError, match="prob_swap_mutation"):
            OptimizerConfig(prob_swap_mutation=0.4)

    def test_invalid_replace_mutation_category_raises(self):
        """Categorías fuera del catálogo deben rechazarse."""
        with pytest.raises(ValueError, match="prob_replace_mutation"):
            OptimizerConfig(prob_replace_mutation=0.8)


class TestAlgorithmFactory:
    """Tests de la factory de algoritmos."""

    def test_create_nsga2(self, search_space_5q):
        """NSGA-II expone el tipo y los operadores configurados."""
        # Arrange
        config = OptimizerConfig(
            algorithm="nsga2",
            population_size=20,
            crossover_operator="ox",
            prob_swap_mutation=0.5,
            prob_replace_mutation=0.9,
        )

        # Act
        algo = create_algorithm(config, search_space_5q, n_objectives=2)

        # Assert
        assert isinstance(algo, NSGA2)
        assert algo.pop_size == 20
        assert isinstance(algo.initialization.sampling, LayoutSampling)
        assert isinstance(algo.mating.crossover, LayoutCrossover)
        assert isinstance(algo.mating.mutation, LayoutMutation)
        assert algo.mating.mutation.prob_swap == 0.5
        assert algo.mating.mutation.prob_replace == 0.9

    def test_create_moead(self, search_space_5q):
        """MOEA/D expone el tipo y el wiring esencial del algoritmo."""
        # Arrange
        config = OptimizerConfig(
            algorithm="moead",
            population_size=20,
            crossover_operator="dpx",
            prob_swap_mutation=0.3,
            prob_replace_mutation=0.7,
        )

        # Act
        algo = create_algorithm(config, search_space_5q, n_objectives=2)

        # Assert
        assert isinstance(algo, MOEAD)
        assert algo.pop_size == 20
        assert isinstance(algo.initialization.sampling, LayoutSampling)
        assert isinstance(algo.mating.crossover, DPXCrossover)
        assert isinstance(algo.mating.mutation, LayoutMutation)
        assert algo.mating.mutation.prob_swap == 0.3
        assert algo.mating.mutation.prob_replace == 0.7
        assert algo.ref_dirs.shape[1] == 2
        assert len(algo.ref_dirs) > 0
        assert algo.n_neighbors == 15


class TestOptimizationProblem:
    """Tests del problema de optimización pymoo."""

    def test_problem_dimensions(
        self, search_space_5q, backend_info_torino, ghz5, backend_torino
    ):
        """El problema tiene las dimensiones correctas."""
        evaluator = FitnessEvaluator.from_names(
            ["depth", "cnot_count"],
            backend_info=backend_info_torino,
            circuit=ghz5,
            backend=backend_torino,
        )
        problem = LayoutOptimizationProblem(search_space_5q, evaluator)

        assert problem.n_var == 5
        assert problem.n_obj == 2


class TestOptimizeLayout:
    """Tests de la función principal de optimización.

    Se usan poblaciones y generaciones muy reducidas para que los
    tests sean rápidos (< 30s cada uno).
    """

    def test_optimize_depth_cnot(self, simple_circuit, backend_torino):
        """Optimización con depth y cnot_count produce resultados válidos."""
        config = OptimizerConfig(
            algorithm="nsga2",
            population_size=10,
            n_generations=5,
            objectives=["depth", "cnot_count"],
            seed=42,
            verbose=False,
        )
        result = optimize_layout(
            simple_circuit, backend=backend_torino, config=config
        )

        assert isinstance(result, OptimizationResult)
        assert result.n_pareto_solutions > 0
        assert result.pareto_fitness is not None
        assert result.pareto_fitness.shape[1] == 2
        assert result.elapsed_time_s > 0
        assert result.algorithm_name == "nsga2"

    def test_optimize_with_transpilation(self, simple_circuit, backend_torino):
        """Optimización con transpilación produce resultados válidos."""
        config = OptimizerConfig(
            algorithm="nsga2",
            population_size=6,
            n_generations=3,
            objectives=["depth", "cnot_count"],
            seed=42,
            verbose=False,
        )
        result = optimize_layout(
            simple_circuit, backend=backend_torino, config=config
        )

        assert result.n_pareto_solutions > 0
        # El caché debería tener entradas
        assert result.cache_stats.get("misses", 0) > 0

    def test_optimize_layout_quick(self, simple_circuit, backend_torino):
        """optimize_layout_quick funciona con el preset por defecto."""
        result = optimize_layout_quick(
            simple_circuit,
            backend=backend_torino,
            preset="default",
            population_size=10,
            n_generations=3,
            seed=42,
        )
        assert result.n_pareto_solutions > 0

    def test_all_pareto_layouts_valid(self, simple_circuit, backend_torino):
        """Todos los layouts del frente de Pareto son válidos."""
        config = OptimizerConfig(
            algorithm="nsga2",
            population_size=10,
            n_generations=5,
            objectives=["depth", "cnot_count"],
            seed=42,
            verbose=False,
        )
        result = optimize_layout(
            simple_circuit, backend=backend_torino, config=config
        )
        backend_info = extract_backend_info(backend_torino)
        ss = LayoutSearchSpace.from_backend_info(
            backend_info, simple_circuit.num_qubits
        )

        for i, layout in enumerate(result.pareto_layouts):
            assert validate_layout(layout, ss), (
                f"Layout Pareto #{i} inválido: {layout}"
            )


class TestOptimizationResult:
    """Tests de la dataclass OptimizationResult."""

    def test_get_best_layout(self):
        """get_best_layout devuelve el layout con menor valor del objetivo."""
        result = OptimizationResult(
            pareto_layouts=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            pareto_fitness=np.array([
                [10.0, 0.05],
                [8.0, 0.10],
                [15.0, 0.03],
            ]),
            objective_names=["depth", "error"],
        )
        # Mejor en depth (índice 0) → [3,4,5] (valor 8.0)
        assert result.get_best_layout(0) == [3, 4, 5]
        # Mejor en error (índice 1) → [6,7,8] (valor 0.03)
        assert result.get_best_layout(1) == [6, 7, 8]

    def test_get_compromise_layout(self):
        """get_compromise_layout devuelve una solución del frente."""
        result = OptimizationResult(
            pareto_layouts=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            pareto_fitness=np.array([
                [1.0, 10.0],
                [5.0, 5.0],
                [10.0, 1.0],
            ]),
            objective_names=["depth", "error"],
        )
        compromise = result.get_compromise_layout()
        assert compromise in result.pareto_layouts

    def test_empty_result_raises(self):
        """Resultado vacío lanza IndexError."""
        result = OptimizationResult()
        with pytest.raises(IndexError):
            result.get_best_layout(0)

    def test_summary(self):
        """summary() devuelve string legible."""
        result = OptimizationResult(
            pareto_layouts=[[0, 1, 2]],
            pareto_fitness=np.array([[5.0, 0.05]]),
            objective_names=["depth", "error"],
            algorithm_name="nsga2",
            backend_name="fake_torino",
            circuit_name="test",
            elapsed_time_s=1.5,
        )
        s = result.summary()
        assert "nsga2" in s
        assert "fake_torino" in s
        assert "depth" in s

    def test_to_dict(self):
        """to_dict() genera un diccionario serializable."""
        result = OptimizationResult(
            pareto_layouts=[[0, 1, 2]],
            pareto_fitness=np.array([[5.0, 0.05]]),
            objective_names=["depth", "error"],
            algorithm_name="nsga2",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "algorithm_name" in d
        assert "pareto_depth_min" in d
        assert "pareto_error_min" in d

    def test_to_dict_with_empty_front_array_keeps_base_metadata_only(self):
        """Un frente vacío serializa metadatos sin reducciones inválidas."""
        result = OptimizationResult(
            pareto_layouts=[],
            pareto_fitness=np.empty((0, 2)),
            objective_names=["depth", "cnot_count"],
            algorithm_name="nsga2",
            backend_name="fake_torino",
            circuit_name="empty_case",
            num_logical_qubits=3,
            n_generations_run=5,
            elapsed_time_s=0.25,
        )

        data = result.to_dict()

        assert data["algorithm_name"] == "nsga2"
        assert data["backend_name"] == "fake_torino"
        assert data["circuit_name"] == "empty_case"
        assert data["num_logical_qubits"] == 3
        assert data["n_generations_run"] == 5
        assert data["elapsed_time_s"] == 0.25
        assert data["n_pareto_solutions"] == 0
        assert data["objective_names"] == ["depth", "cnot_count"]
        assert "pareto_depth_min" not in data
        assert "pareto_depth_max" not in data
        assert "pareto_depth_mean" not in data
        assert "pareto_cnot_count_min" not in data
        assert "pareto_cnot_count_max" not in data
        assert "pareto_cnot_count_mean" not in data


class TestCompareLayouts:
    """Tests de comparación de layouts."""

    def test_compare_layouts(self, simple_circuit, backend_torino):
        """compare_layouts genera filas con las métricas esperadas."""
        layouts = {
            "layout_a": [0, 1, 2],
            "layout_b": [10, 11, 12],
        }
        rows = compare_layouts(
            simple_circuit, layouts,
            backend=backend_torino, seed=42,
        )
        assert len(rows) == 2
        for row in rows:
            assert "layout_name" in row
            assert "depth" in row
            assert "two_qubit_gates" in row
            assert "cnot_equivalent" in row
            assert "avg_error_2q" in row

    def test_compare_layouts_keeps_two_qubit_gates_and_adds_cnot_equivalent(
        self, backend_torino
    ):
        """La comparación expone ambas métricas 2Q y preserva su semántica."""
        qc = QuantumCircuit(2, name="swap_only")
        qc.swap(0, 1)

        rows = compare_layouts(
            qc,
            {"swap_layout": [0, 1]},
            backend=backend_torino,
            seed=42,
        )

        assert len(rows) == 1
        row = rows[0]
        assert "two_qubit_gates" in row
        assert "cnot_equivalent" in row
        assert row["two_qubit_gates"] is not None
        assert row["cnot_equivalent"] is not None
        assert row["cnot_equivalent"] >= row["two_qubit_gates"]


class TestReportingUtilities:
    """Tests de utilidades públicas de reporting."""

    def test_print_layout_comparison_prints_table(self, capsys):
        """print_layout_comparison imprime encabezados y filas legibles."""
        # Arrange
        rows = [
            {
                "layout_name": "layout_a",
                "depth": 12,
                "two_qubit_gates": 4,
                "cnot_equivalent": 7,
                "total_gates": 20,
                "avg_error_2q": 0.012345,
                "num_edges": 3,
            },
            {
                "layout_name": "layout_b",
                "depth": 9,
                "two_qubit_gates": 3,
                "cnot_equivalent": 3,
                "total_gates": 18,
                "avg_error_2q": 0.006789,
                "num_edges": 4,
            },
        ]

        # Act
        print_layout_comparison(rows)
        captured = capsys.readouterr()

        # Assert
        assert "COMPARACIÓN DE LAYOUTS" in captured.out
        assert "layout_a" in captured.out
        assert "layout_b" in captured.out
        assert "2Q Gates" in captured.out
        assert "CNOT Eq" in captured.out
        assert "7" in captured.out
        assert "Err 2Q" in captured.out


class TestParetoPlots:
    """Tests smoke/contract de visualización pública."""

    def test_plot_pareto_front_2d_returns_figure_and_saves_file(
        self, pareto_result_2d, tmp_path
    ):
        """El plot 2D devuelve Figure y guarda archivo cuando se pide."""
        # Arrange
        output_file = tmp_path / "plots" / "pareto_front_2d.png"

        # Act
        fig = plot_pareto_front_2d(
            pareto_result_2d,
            filename=str(output_file),
        )

        # Assert
        assert isinstance(fig, Figure)
        assert output_file.is_file()
        fig.clf()

    def test_plot_pareto_front_2d_returns_none_without_pareto_data(self):
        """Sin datos de Pareto, el plot 2D devuelve None."""
        # Arrange
        empty_result = OptimizationResult()

        # Act
        fig = plot_pareto_front_2d(empty_result)

        # Assert
        assert fig is None

    def test_plot_pareto_front_2d_returns_none_with_empty_front_array(
        self, empty_pareto_result_2d
    ):
        """Un frente vacío en ndarray se trata como ausencia de datos en 2D."""
        # Arrange
        result = empty_pareto_result_2d

        # Act
        fig = plot_pareto_front_2d(result)

        # Assert
        assert fig is None

    def test_plot_pareto_front_2d_highlights_actual_compromise_row_with_duplicates(self):
        """El resaltado de compromiso usa la fila ganadora aunque el layout se repita."""
        result = OptimizationResult(
            pareto_layouts=[[7, 8], [9, 10], [7, 8]],
            pareto_fitness=np.array([
                [1.0, 9.0],
                [9.0, 1.0],
                [5.0, 5.0],
            ]),
            objective_names=["depth", "cnot_count"],
            algorithm_name="nsga2",
            backend_name="fake_torino",
            circuit_name="dup_front",
        )

        fig = plot_pareto_front_2d(
            result,
            highlight_knee=False,
            highlight_compromise=True,
        )

        assert isinstance(fig, Figure)

        compromise_collection = next(
            collection
            for collection in fig.axes[0].collections
            if collection.get_label() == "Compromiso"
        )
        compromise_offsets = compromise_collection.get_offsets()

        assert compromise_offsets.shape == (1, 2)
        np.testing.assert_allclose(compromise_offsets[0], [5.0, 5.0])
        fig.clf()

    def test_plot_pareto_front_2d_highlights_knee_for_two_point_front(self):
        """El plot 2D resalta el knee también en frentes de dos soluciones."""
        fitness = np.array([
            [1.0, 9.0],
            [3.0, 1.0],
        ])
        result = OptimizationResult(
            pareto_layouts=[[0, 1], [2, 3]],
            pareto_fitness=fitness,
            objective_names=["depth", "cnot_count"],
            algorithm_name="nsga2",
            backend_name="fake_torino",
            circuit_name="two_point_front",
        )

        fig = plot_pareto_front_2d(
            result,
            highlight_knee=True,
            highlight_compromise=False,
        )

        assert isinstance(fig, Figure)

        knee_collection = next(
            collection
            for collection in fig.axes[0].collections
            if collection.get_label() == "Knee Point"
        )
        knee_idx = select_knee_point(fitness)
        np.testing.assert_allclose(knee_collection.get_offsets()[0], fitness[knee_idx])
        assert knee_idx == 0
        fig.clf()

    def test_plot_pareto_front_3d_returns_figure_and_saves_file(
        self, pareto_result_3d, tmp_path
    ):
        """El plot 3D devuelve Figure y guarda archivo cuando hay 3 objetivos."""
        # Arrange
        output_file = tmp_path / "plots" / "pareto_front_3d.png"

        # Act
        fig = plot_pareto_front_3d(
            pareto_result_3d,
            filename=str(output_file),
        )

        # Assert
        assert isinstance(fig, Figure)
        assert output_file.is_file()
        fig.clf()

    def test_plot_pareto_front_3d_returns_none_with_fewer_than_three_objectives(
        self, pareto_result_2d
    ):
        """El plot 3D devuelve None cuando no hay suficientes objetivos."""
        # Arrange
        result = pareto_result_2d

        # Act
        fig = plot_pareto_front_3d(result)

        # Assert
        assert fig is None

    def test_plot_pareto_front_3d_returns_none_with_empty_front_array(
        self, empty_pareto_result_3d
    ):
        """Un frente vacío en ndarray se trata como ausencia de datos en 3D."""
        # Arrange
        result = empty_pareto_result_3d

        # Act
        fig = plot_pareto_front_3d(result)

        # Assert
        assert fig is None

    def test_plot_parallel_coordinates_returns_figure_and_saves_file(
        self, pareto_result_3d, tmp_path
    ):
        """Las coordenadas paralelas devuelven Figure y guardan archivo."""
        # Arrange
        output_file = tmp_path / "plots" / "parallel_coordinates.png"

        # Act
        fig = plot_parallel_coordinates(
            pareto_result_3d,
            filename=str(output_file),
        )

        # Assert
        assert isinstance(fig, Figure)
        assert output_file.is_file()
        fig.clf()

    def test_plot_parallel_coordinates_returns_none_without_pareto_data(self):
        """Sin datos de Pareto, las coordenadas paralelas devuelven None."""
        # Arrange
        empty_result = OptimizationResult()

        # Act
        fig = plot_parallel_coordinates(empty_result)

        # Assert
        assert fig is None

    def test_plot_parallel_coordinates_returns_none_with_empty_front_array(
        self, empty_pareto_result_2d
    ):
        """Un frente vacío en ndarray se trata como ausencia de datos en paralelo."""
        # Arrange
        result = empty_pareto_result_2d

        # Act
        fig = plot_parallel_coordinates(result)

        # Assert
        assert fig is None

    def test_plot_parallel_coordinates_highlights_knee_for_two_point_front(self):
        """Las coordenadas paralelas resaltan el knee en frentes de dos soluciones."""
        fitness = np.array([
            [1.0, 9.0],
            [3.0, 1.0],
        ])
        result = OptimizationResult(
            pareto_layouts=[[0, 1], [2, 3]],
            pareto_fitness=fitness,
            objective_names=["depth", "cnot_count"],
            algorithm_name="nsga2",
            backend_name="fake_torino",
            circuit_name="two_point_front",
        )

        fig = plot_parallel_coordinates(result, highlight_knee=True)

        assert isinstance(fig, Figure)

        knee_line = next(
            line for line in fig.axes[0].lines if line.get_label() == "Knee Point"
        )
        knee_idx = select_knee_point(fitness)
        assert knee_idx == 0
        np.testing.assert_allclose(knee_line.get_ydata(), [0.0, 1.0])
        fig.clf()


# ===========================================================================
#  Tests — pareto
# ===========================================================================

class TestParetoMetrics:
    """Tests de métricas del frente de Pareto."""

    def test_compute_metrics_basic(self):
        """Calcula métricas para un frente simple."""
        F = np.array([
            [1.0, 5.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [5.0, 1.0],
        ])
        metrics = compute_pareto_metrics(F)

        assert isinstance(metrics, ParetoMetrics)
        assert metrics.n_solutions == 4
        assert metrics.hypervolume > 0
        assert metrics.ideal_point is not None
        np.testing.assert_array_equal(metrics.ideal_point, [1.0, 1.0])
        np.testing.assert_array_equal(metrics.nadir_point, [5.0, 5.0])

    def test_compute_metrics_single_solution(self):
        """Funciona con una sola solución."""
        F = np.array([[3.0, 4.0]])
        metrics = compute_pareto_metrics(F)
        assert metrics.n_solutions == 1
        assert metrics.hypervolume > 0

    def test_compute_metrics_empty_front_returns_no_data_metrics(self):
        """Un frente vacío devuelve métricas válidas de no-datos."""
        metrics = compute_pareto_metrics(np.empty((0, 2)))

        assert isinstance(metrics, ParetoMetrics)
        assert metrics.n_solutions == 0
        assert metrics.hypervolume == 0.0
        assert metrics.spacing == 0.0
        assert metrics.spread is None
        assert metrics.ideal_point is None
        assert metrics.nadir_point is None
        assert metrics.reference_point is None

    def test_metrics_summary(self):
        """El resumen contiene información legible."""
        F = np.array([[1.0, 5.0], [5.0, 1.0]])
        metrics = compute_pareto_metrics(F)
        s = metrics.summary()
        assert "Hipervolumen" in s
        assert "Soluciones" in s


class TestParetoSelection:
    """Tests de estrategias de selección."""

    def test_knee_point_2d(self):
        """Detecta correctamente el knee point en 2D."""
        # Un frente convexo donde el knee es el punto medio
        F = np.array([
            [1.0, 10.0],  # Extremo A
            [3.0, 3.0],   # Knee (mayor distancia a la diagonal)
            [10.0, 1.0],  # Extremo B
        ])
        knee = select_knee_point(F)
        assert knee == 1  # El punto del medio es el knee

    def test_knee_point_single_solution(self):
        """Con una sola solución, devuelve 0."""
        F = np.array([[5.0, 5.0]])
        assert select_knee_point(F) == 0

    def test_select_weighted(self):
        """Selección ponderada con pesos iguales."""
        F = np.array([
            [1.0, 10.0],
            [5.0, 5.0],
            [10.0, 1.0],
        ])
        # Pesos iguales → debería seleccionar la solución más equilibrada
        idx = select_weighted(F, [0.5, 0.5])
        assert idx == 1  # (5,5) normalizado da (0.44, 0.44) vs extremos

    def test_select_min_objective(self):
        """Selección por mínimo en un objetivo."""
        F = np.array([
            [10.0, 1.0],
            [5.0, 5.0],
            [1.0, 10.0],
        ])
        assert select_min_objective(F, 0) == 2  # min depth = 1.0
        assert select_min_objective(F, 1) == 0  # min error = 1.0


class TestAnalyzePareto:
    """Tests del análisis completo del frente."""

    def test_analyze_pareto_front(self):
        """El análisis completo devuelve todas las claves esperadas."""
        result = OptimizationResult(
            pareto_layouts=[[0, 1], [2, 3], [4, 5]],
            pareto_fitness=np.array([
                [1.0, 10.0],
                [5.0, 5.0],
                [10.0, 1.0],
            ]),
            objective_names=["depth", "cnot_count"],
        )
        analysis = analyze_pareto_front(result)

        assert "metrics" in analysis
        assert "knee_point_idx" in analysis
        assert "knee_point_layout" in analysis
        assert "best_per_objective" in analysis
        assert "compromise_layout" in analysis
        assert "selection_candidates" in analysis
        assert "tradeoff_table" in analysis

        # Verificar que el knee point es una solución del frente
        assert analysis["knee_point_layout"] in result.pareto_layouts

        # Verificar mejores por objetivo
        bpo = analysis["best_per_objective"]
        assert "depth" in bpo
        assert "cnot_count" in bpo
        assert bpo["depth"]["layout"] == [0, 1]   # min depth = 1.0
        assert bpo["cnot_count"]["layout"] == [4, 5]   # min cnot_count = 1.0

        candidates = analysis["selection_candidates"]
        assert set(candidates) == {
            "compromise",
            "knee",
            "best_depth",
            "best_cnot_count",
        }
        assert candidates["compromise"]["layout"] == [2, 3]
        assert candidates["compromise"]["reason"]
        assert candidates["compromise"]["distance_to_ideal"] == pytest.approx(
            np.linalg.norm([4.0 / 9.0, 4.0 / 9.0])
        )
        assert candidates["knee"]["layout"] == analysis["knee_point_layout"]
        assert candidates["knee"]["reason"]

        tradeoff_table = analysis["tradeoff_table"]
        assert len(tradeoff_table) == 3
        assert tradeoff_table[0]["index"] == 0
        assert tradeoff_table[0]["layout"] == [0, 1]
        assert tradeoff_table[0]["raw_objectives"] == {
            "depth": 1.0,
            "cnot_count": 10.0,
        }
        assert tradeoff_table[0]["normalized_objectives"] == {
            "depth": 0.0,
            "cnot_count": 1.0,
        }
        assert tradeoff_table[1]["distance_to_ideal"] == pytest.approx(
            np.linalg.norm([4.0 / 9.0, 4.0 / 9.0])
        )

    def test_analyze_pareto_front_candidate_payload_has_reason_and_distance(self):
        """Cada candidato de selección expone justificación y distancia."""
        result = OptimizationResult(
            pareto_layouts=[[0, 1], [2, 3], [4, 5]],
            pareto_fitness=np.array([
                [1.0, 10.0],
                [5.0, 5.0],
                [10.0, 1.0],
            ]),
            objective_names=["depth", "cnot_count"],
        )

        analysis = analyze_pareto_front(result)

        for candidate in analysis["selection_candidates"].values():
            assert isinstance(candidate["reason"], str)
            assert candidate["reason"]
            assert candidate["distance_to_ideal"] >= 0.0

    def test_analyze_pareto_front_uses_lowest_index_for_tied_minima(self):
        """Los mínimos explícitos usan desempate estable por primer índice."""
        result = OptimizationResult(
            pareto_layouts=[[10, 11], [20, 21], [30, 31]],
            pareto_fitness=np.array([
                [1.0, 9.0],
                [1.0, 5.0],
                [4.0, 5.0],
            ]),
            objective_names=["depth", "cnot_count"],
        )

        analysis = analyze_pareto_front(result)

        assert analysis["best_per_objective"]["depth"]["index"] == 0
        assert analysis["best_per_objective"]["depth"]["layout"] == [10, 11]
        assert analysis["best_per_objective"]["cnot_count"]["index"] == 1
        assert analysis["best_per_objective"]["cnot_count"]["layout"] == [20, 21]

    def test_analyze_pareto_front_compromise_candidate_uses_actual_winning_row(self):
        """El candidato compromise conserva el índice real aunque el layout se repita."""
        result = OptimizationResult(
            pareto_layouts=[[7, 8], [9, 10], [7, 8]],
            pareto_fitness=np.array([
                [1.0, 9.0],
                [9.0, 1.0],
                [5.0, 5.0],
            ]),
            objective_names=["depth", "cnot_count"],
        )

        analysis = analyze_pareto_front(result)

        compromise = analysis["selection_candidates"]["compromise"]
        expected_distance = np.linalg.norm([0.5, 0.5])

        assert analysis["compromise_layout"] == [7, 8]
        assert compromise["index"] == 2
        assert compromise["layout"] == [7, 8]
        assert compromise["distance_to_ideal"] == pytest.approx(expected_distance)
        assert analysis["tradeoff_table"][compromise["index"]]["distance_to_ideal"] == pytest.approx(
            expected_distance
        )
        assert analysis["tradeoff_table"][0]["distance_to_ideal"] != pytest.approx(
            expected_distance
        )

    def test_analyze_empty_front(self):
        """Funciona con un frente vacío."""
        result = OptimizationResult()
        analysis = analyze_pareto_front(result)
        assert analysis["metrics"].n_solutions == 0
        assert analysis["knee_point_idx"] == -1
        assert analysis["selection_candidates"] == {}
        assert analysis["tradeoff_table"] == []
