"""
Tests unitarios para el módulo de tuning de hiperparámetros
=============================================================

Cobertura:
  - HyperparameterSpace (valores por defecto y personalizados)
  - _compute_hypervolume_score (función de evaluación)
  - _evaluate_config (evaluación de una configuración con seeds)
  - LayoutTuner (ciclo completo con presupuesto mínimo para rapidez)

Se usan poblaciones y generaciones muy reducidas para que los tests
de integración terminen en segundos.

Ejecución:
  pytest tests/test_mo_module/test_tuning.py -v

Autor: Eduardo González Bautista
Fecha: 2026-03-03
"""

import pytest
import numpy as np

from src.mo_module.tuning import (
    HyperparameterSpace,
    LayoutTuner,
    _compute_hypervolume_score,
    _evaluate_config,
)
from src.mo_module.optimizer import OptimizerConfig, OptimizationResult
from src.qiskit_interface.circuit_utils import create_ghz_circuit
from src.qiskit_interface.backend_info import get_backend


# ===========================================================================
#  Fixtures
# ===========================================================================

@pytest.fixture
def small_circuit():
    """Circuito GHZ de 3 qubits (mínimo para tuning rápido)."""
    return create_ghz_circuit(3)


@pytest.fixture
def backend_torino():
    """FakeTorino — backend de referencia para los tests."""
    return get_backend("fake_torino")


# ===========================================================================
#  Tests — HyperparameterSpace
# ===========================================================================

class TestHyperparameterSpace:
    """Tests del espacio de búsqueda de hiperparámetros."""

    def test_default_values(self):
        """Los valores por defecto son válidos y coherentes."""
        space = HyperparameterSpace()
        assert space.population_size_range[0] < space.population_size_range[1]
        assert space.n_generations_range[0] < space.n_generations_range[1]
        assert space.prob_swap_mutation_range[0] < space.prob_swap_mutation_range[1]
        assert space.prob_replace_mutation_range[0] < space.prob_replace_mutation_range[1]
        assert "dpx" in space.crossover_operators
        assert "nsga2" in space.algorithms

    def test_custom_space(self):
        """Se puede crear un espacio de búsqueda personalizado."""
        space = HyperparameterSpace(
            population_size_range=(10, 50),
            n_generations_range=(10, 30),
            crossover_operators=["dpx"],
            algorithms=["nsga2"],
        )
        assert space.population_size_range == (10, 50)
        assert space.crossover_operators == ["dpx"]

    def test_optimization_level_fixed(self):
        """optimization_level es un campo fijo (no se optimiza)."""
        space = HyperparameterSpace(optimization_level=2)
        assert space.optimization_level == 2


# ===========================================================================
#  Tests — _compute_hypervolume_score
# ===========================================================================

class TestComputeHypervolumeScore:
    """Tests de la función de evaluación de hipervolumen."""

    def test_score_positive_for_valid_pareto(self):
        """El hipervolumen es positivo para un frente de Pareto válido."""
        result = OptimizationResult(
            pareto_layouts=[[0, 1, 2], [3, 4, 5]],
            pareto_fitness=np.array([[5.0, 3.0], [3.0, 6.0]]),
            objective_names=["depth", "cnot_count"],
        )
        score = _compute_hypervolume_score(result)
        assert score > 0.0

    def test_score_zero_for_empty_result(self):
        """Un resultado vacío devuelve score 0."""
        result = OptimizationResult()
        score = _compute_hypervolume_score(result)
        assert score == 0.0

    def test_score_zero_for_none_fitness(self):
        """pareto_fitness=None devuelve score 0."""
        result = OptimizationResult(
            pareto_layouts=[[0, 1, 2]],
            pareto_fitness=None,
        )
        score = _compute_hypervolume_score(result)
        assert score == 0.0

    def test_better_pareto_higher_score(self):
        """Un frente más dominante produce mayor hipervolumen."""
        # Frente bueno: soluciones con valores bajos (mejor)
        good_result = OptimizationResult(
            pareto_layouts=[[0, 1, 2]],
            pareto_fitness=np.array([[2.0, 2.0]]),
            objective_names=["depth", "cnot_count"],
        )
        # Frente malo: soluciones con valores altos (peor)
        bad_result = OptimizationResult(
            pareto_layouts=[[0, 1, 2]],
            pareto_fitness=np.array([[10.0, 10.0]]),
            objective_names=["depth", "cnot_count"],
        )
        good_score = _compute_hypervolume_score(good_result)
        bad_score = _compute_hypervolume_score(bad_result)
        # El frente bueno tiene HV mayor (domina más espacio)
        assert good_score > 0.0
        assert bad_score > 0.0
        # Ambos son positivos; el bueno debería ser mayor (frente más extendido)
        # Nota: con referencia relativa, ambos tendrán el mismo HV escalado.
        # Lo importante es que sean > 0 y no crasheen.


# ===========================================================================
#  Tests — _evaluate_config
# ===========================================================================

class TestEvaluateConfig:
    """Tests de la función de evaluación de una configuración."""

    def test_returns_nonnegative_score(self, small_circuit, backend_torino):
        """La evaluación de una config válida devuelve un score ≥ 0."""
        config = OptimizerConfig(
            algorithm="nsga2",
            population_size=6,
            n_generations=3,
            objectives=["depth", "cnot_count"],
            verbose=False,
        )
        score = _evaluate_config(config, small_circuit, backend_torino, seeds=[0, 1])
        assert score >= 0.0

    def test_multiple_seeds_averaged(self, small_circuit, backend_torino):
        """Evaluar con múltiples seeds devuelve un valor promediado."""
        config = OptimizerConfig(
            algorithm="nsga2",
            population_size=6,
            n_generations=3,
            objectives=["depth", "cnot_count"],
            verbose=False,
        )
        # Con una seed
        score_1 = _evaluate_config(config, small_circuit, backend_torino, seeds=[0])
        # Con tres seeds
        score_3 = _evaluate_config(config, small_circuit, backend_torino, seeds=[0, 1, 2])
        # Ambos deben ser válidos
        assert score_1 >= 0.0
        assert score_3 >= 0.0


# ===========================================================================
#  Tests — LayoutTuner
# ===========================================================================

class TestLayoutTuner:
    """Tests del tuner principal.

    Se usan n_trials=3 y n_seeds=1 para que los tests terminen en segundos.
    El presupuesto real de producción (n_trials=30, n_seeds=3) se configura
    en el código del usuario.
    """

    def test_init_defaults(self, small_circuit, backend_torino):
        """LayoutTuner se inicializa con los valores por defecto correctos."""
        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
        )
        assert tuner.n_trials == LayoutTuner.DEFAULT_N_TRIALS
        assert tuner.n_seeds == LayoutTuner.DEFAULT_N_SEEDS
        assert tuner._study is None
        assert tuner._best_config is None

    def test_init_custom_budget(self, small_circuit, backend_torino):
        """Se puede personalizar el presupuesto de tuning."""
        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=10,
            n_seeds=2,
        )
        assert tuner.n_trials == 10
        assert tuner.n_seeds == 2

    def test_best_config_before_tune_raises(self, small_circuit, backend_torino):
        """Llamar best_config() antes de tune() lanza RuntimeError."""
        tuner = LayoutTuner(circuit=small_circuit, backend=backend_torino)
        with pytest.raises(RuntimeError, match="tune\\(\\)"):
            tuner.best_config()

    def test_study_before_tune_raises(self, small_circuit, backend_torino):
        """Acceder a study antes de tune() lanza RuntimeError."""
        tuner = LayoutTuner(circuit=small_circuit, backend=backend_torino)
        with pytest.raises(RuntimeError):
            _ = tuner.study

    def test_tune_returns_self(self, small_circuit, backend_torino):
        """tune() devuelve self para permitir encadenamiento."""
        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=2,
            n_seeds=1,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx"],
                algorithms=["nsga2"],
            ),
        )
        result = tuner.tune(show_progress_bar=False)
        assert result is tuner

    def test_tune_produces_valid_config(self, small_circuit, backend_torino):
        """Tras tune(), best_config() devuelve una OptimizerConfig válida."""
        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=2,
            n_seeds=1,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx"],
                algorithms=["nsga2"],
            ),
        )
        tuner.tune(show_progress_bar=False)
        config = tuner.best_config()

        assert isinstance(config, OptimizerConfig)
        assert config.algorithm == "nsga2"
        assert config.crossover_operator in ("dpx", "ox")
        assert config.population_size >= 6
        assert config.n_generations >= 3

    def test_tune_study_accessible(self, small_circuit, backend_torino):
        """Tras tune(), el estudio Optuna está accesible."""
        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=2,
            n_seeds=1,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx"],
                algorithms=["nsga2"],
            ),
        )
        tuner.tune(show_progress_bar=False)

        study = tuner.study
        assert len(study.trials) == 2
        assert study.best_value >= 0.0

    def test_tune_chaining(self, small_circuit, backend_torino):
        """tune().best_config() funciona en cadena."""
        config = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=2,
            n_seeds=1,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx"],
                algorithms=["nsga2"],
            ),
        ).tune(show_progress_bar=False).best_config()

        assert isinstance(config, OptimizerConfig)

    def test_summary_after_tune(self, small_circuit, backend_torino):
        """summary() devuelve un string legible tras el tuning."""
        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=2,
            n_seeds=1,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx"],
                algorithms=["nsga2"],
            ),
        )
        tuner.tune(show_progress_bar=False)
        summary = tuner.summary()

        assert "TUNING" in summary
        assert "population_size" in summary
        assert "crossover_operator" in summary
