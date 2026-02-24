"""
Tests unitarios para el submódulo benchmark
=============================================

Cobertura:
  - circuits: creación de circuitos, suite por defecto, filtrado por tag
  - runner:   BenchmarkRunner con ejecución reducida (2 semillas, 1 circuito)
  - analysis: cálculo de estadísticas y generación de informe

Ejecución:
  pytest tests/test_mo_module/test_benchmark.py -v

Autor: Eduardo González Bautista
Fecha: 2026-02-24
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit

# ---------------------------------------------------------------------------
#  Imports del submódulo bajo test
# ---------------------------------------------------------------------------
from src.mo_module.benchmark.circuits import (
    BenchmarkCircuit,
    DEFAULT_BENCHMARK_CIRCUITS,
    get_default_circuits,
    get_circuits_by_tag,
    make_custom_circuit,
)
from src.mo_module.benchmark.runner import (
    BenchmarkRun,
    BenchmarkResultSet,
    BenchmarkRunner,
)
from src.mo_module.benchmark.analysis import (
    ObjectiveStats,
    compute_objective_stats,
    analyze_results,
    BenchmarkReport,
)
from src.mo_module.benchmark import run_benchmark
from src.mo_module.optimizer import OptimizerConfig, OptimizationResult


# ===========================================================================
#  Fixtures
# ===========================================================================


@pytest.fixture
def tiny_config() -> OptimizerConfig:
    """Configuración mínima para tests rápidos."""
    return OptimizerConfig(
        algorithm="nsga2",
        population_size=6,
        n_generations=4,
        objectives=["depth", "cnot_count"],
        verbose=False,
    )


@pytest.fixture
def single_circuit() -> list[BenchmarkCircuit]:
    """Un solo circuito pequeño para tests rápidos."""
    qc = QuantumCircuit(3, name="test_bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return [make_custom_circuit("test_3q", qc, tags=("test",))]


# ===========================================================================
#  Tests de circuits.py
# ===========================================================================


class TestBenchmarkCircuits:
    """Tests de la suite de circuitos de benchmark."""

    def test_default_suite_has_four_circuits(self):
        """La suite por defecto contiene 4 circuitos."""
        assert len(DEFAULT_BENCHMARK_CIRCUITS) == 4

    def test_get_default_returns_copy(self):
        """get_default_circuits devuelve una copia (no la misma lista)."""
        a = get_default_circuits()
        b = get_default_circuits()
        assert a is not b
        assert len(a) == len(b)

    def test_each_circuit_creates_valid_qc(self):
        """Cada circuito de la suite genera un QuantumCircuit válido."""
        for bc in DEFAULT_BENCHMARK_CIRCUITS:
            qc = bc.create()
            assert isinstance(qc, QuantumCircuit)
            assert qc.num_qubits == bc.num_qubits
            assert qc.num_qubits >= 1

    def test_filter_by_tag(self):
        """get_circuits_by_tag filtra correctamente."""
        clifford = get_circuits_by_tag("clifford")
        assert len(clifford) >= 1
        assert all("clifford" in bc.tags for bc in clifford)

    def test_filter_by_nonexistent_tag(self):
        """Filtrar por tag inexistente devuelve lista vacía."""
        assert get_circuits_by_tag("xyz_nonexistent") == []

    def test_make_custom_circuit(self):
        """make_custom_circuit crea un BenchmarkCircuit funcional."""
        qc = QuantumCircuit(2, name="custom")
        qc.h(0)
        qc.cx(0, 1)
        bc = make_custom_circuit("my_test", qc, tags=("custom",))
        assert bc.name == "my_test"
        assert bc.num_qubits == 2
        assert "custom" in bc.tags
        created = bc.create()
        assert isinstance(created, QuantumCircuit)
        assert created.num_qubits == 2

    def test_benchmark_circuit_is_frozen(self):
        """BenchmarkCircuit es inmutable (frozen dataclass)."""
        bc = DEFAULT_BENCHMARK_CIRCUITS[0]
        with pytest.raises(AttributeError):
            bc.name = "hack"


# ===========================================================================
#  Tests de runner.py
# ===========================================================================


class TestBenchmarkRunner:
    """Tests del motor de benchmark."""

    def test_runner_executes_all_combinations(
        self, tiny_config, single_circuit
    ):
        """Runner ejecuta circuit × seeds combinaciones."""
        seeds = [0, 1]
        runner = BenchmarkRunner(
            circuits=single_circuit,
            seeds=seeds,
            backend_name="fake_torino",
            config=tiny_config,
        )
        results = runner.run()

        assert isinstance(results, BenchmarkResultSet)
        assert results.n_ok == len(seeds)
        assert results.n_failed == 0
        assert len(results.runs) == len(seeds)

    def test_result_set_circuit_names(self, tiny_config, single_circuit):
        """circuit_names devuelve los nombres sin duplicados."""
        runner = BenchmarkRunner(
            circuits=single_circuit,
            seeds=[0, 1],
            config=tiny_config,
        )
        results = runner.run()
        assert results.circuit_names == ["test_3q"]

    def test_result_set_fitness_matrix(self, tiny_config, single_circuit):
        """fitness_matrix devuelve una matriz con shape correcto."""
        runner = BenchmarkRunner(
            circuits=single_circuit,
            seeds=[0, 1],
            config=tiny_config,
        )
        results = runner.run()
        fm = results.fitness_matrix("test_3q")
        assert fm is not None
        assert fm.shape == (2, 2)  # 2 seeds, 2 objectives

    def test_result_set_best_per_seed(self, tiny_config, single_circuit):
        """best_per_seed devuelve un valor por semilla."""
        runner = BenchmarkRunner(
            circuits=single_circuit,
            seeds=[0, 1, 2],
            config=tiny_config,
        )
        results = runner.run()
        bests = results.best_per_seed("test_3q", objective_index=0)
        assert len(bests) == 3
        assert all(isinstance(v, float) for v in bests)

    def test_result_set_summary(self, tiny_config, single_circuit):
        """summary() devuelve una cadena no vacía."""
        runner = BenchmarkRunner(
            circuits=single_circuit,
            seeds=[0, 1],
            config=tiny_config,
        )
        results = runner.run()
        s = results.summary()
        assert isinstance(s, str)
        assert "test_3q" in s

    def test_total_elapsed_is_positive(self, tiny_config, single_circuit):
        """El tiempo total del benchmark es > 0."""
        runner = BenchmarkRunner(
            circuits=single_circuit,
            seeds=[0],
            config=tiny_config,
        )
        results = runner.run()
        assert results.total_elapsed_s > 0


# ===========================================================================
#  Tests de analysis.py
# ===========================================================================


class TestAnalysis:
    """Tests del módulo de análisis estadístico."""

    def test_compute_objective_stats_basic(self):
        """Estadísticas básicas son correctas."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_objective_stats("depth", values)
        assert stats.name == "depth"
        assert stats.mean == 30.0
        assert stats.min == 10.0
        assert stats.max == 50.0
        assert stats.median == 30.0
        assert stats.iqr > 0
        assert stats.cv > 0

    def test_compute_objective_stats_single_value(self):
        """Con un solo valor, std = 0."""
        stats = compute_objective_stats("x", [42.0])
        assert stats.mean == 42.0
        assert stats.std == 0.0

    def test_analyze_results_produces_report(
        self, tiny_config, single_circuit
    ):
        """analyze_results devuelve un BenchmarkReport completo."""
        runner = BenchmarkRunner(
            circuits=single_circuit,
            seeds=[0, 1, 2],
            config=tiny_config,
        )
        results = runner.run()
        report = analyze_results(results)

        assert isinstance(report, BenchmarkReport)
        assert len(report.circuit_analyses) == 1
        assert report.circuit_analyses[0].circuit_name == "test_3q"
        assert report.circuit_analyses[0].n_seeds == 3
        assert len(report.circuit_analyses[0].objective_stats) == 2

    def test_report_to_text(self, tiny_config, single_circuit):
        """El informe genera texto legible."""
        runner = BenchmarkRunner(
            circuits=single_circuit,
            seeds=[0, 1, 2],
            config=tiny_config,
        )
        results = runner.run()
        report = analyze_results(results)
        text = report.to_text()
        assert "INFORME DE BENCHMARK" in text
        assert "test_3q" in text

    def test_report_to_dict(self, tiny_config, single_circuit):
        """El informe se serializa a diccionario."""
        runner = BenchmarkRunner(
            circuits=single_circuit,
            seeds=[0, 1],
            config=tiny_config,
        )
        results = runner.run()
        report = analyze_results(results)
        d = report.to_dict()
        assert "rows" in d
        assert len(d["rows"]) == 1  # 1 circuito
        assert "depth_mean" in d["rows"][0]


# ===========================================================================
#  Tests de la interfaz de conveniencia
# ===========================================================================


class TestRunBenchmarkConvenience:
    """Tests de la función run_benchmark()."""

    def test_run_benchmark_minimal(self):
        """run_benchmark con parámetros mínimos funciona."""
        results = run_benchmark(
            n_seeds=2,
            circuits=[
                make_custom_circuit(
                    "tiny",
                    QuantumCircuit(2, name="t").compose(
                        _make_bell(), inplace=False
                    ),
                )
            ],
            population_size=6,
            n_generations=4,
        )
        assert results.n_ok == 2


def _make_bell() -> QuantumCircuit:
    """Crea un circuito Bell de 2 qubits (helper para tests)."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc
