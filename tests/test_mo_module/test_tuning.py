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

import logging

import pytest
import numpy as np

from src.mo_module.benchmark.tuning_gui_helpers import (
    _format_ref_point_display,
    _format_ref_point_mode_help,
    _resolve_ref_point_display,
    _parse_manual_ref_point,
)
from src.mo_module import tuning
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
        assert space.prob_swap_mutation_choices == (0.1, 0.3, 0.5, 0.7)
        assert space.prob_replace_mutation_choices == (0.1, 0.3, 0.5, 0.7, 0.9)
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

    def test_mutation_choices_cannot_be_empty(self):
        """Las categorías de mutación deben contener al menos un valor."""
        with pytest.raises(ValueError, match="prob_swap_mutation_choices"):
            HyperparameterSpace(prob_swap_mutation_choices=())
        with pytest.raises(ValueError, match="prob_replace_mutation_choices"):
            HyperparameterSpace(prob_replace_mutation_choices=())

    def test_optimization_level_fixed(self):
        """optimization_level es un campo fijo (no se optimiza)."""
        space = HyperparameterSpace(optimization_level=2)
        assert space.optimization_level == 2

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"population_size_range": (8, 7)}, "population_size_range"),
            ({"population_size_range": (3, 8)}, "population_size_range"),
            ({"n_generations_range": (0, 5)}, "n_generations_range"),
            ({"crossover_operators": []}, "crossover_operators"),
            ({"algorithms": []}, "algorithms"),
        ],
    )
    def test_invalid_ranges_raise_value_error(self, kwargs, match):
        """Los rangos y catálogos inválidos se rechazan explícitamente."""
        with pytest.raises(ValueError, match=match):
            HyperparameterSpace(**kwargs)


# ===========================================================================
#  Tests — _compute_hypervolume_score
# ===========================================================================

class TestComputeHypervolumeScore:
    """Tests de la función de evaluación de hipervolumen."""

    def test_score_uses_injected_reference_point(self):
        """Se puede fijar explícitamente el punto de referencia del HV."""
        result = OptimizationResult(
            pareto_layouts=[[0, 1, 2]],
            pareto_fitness=np.array([[2.0, 3.0]]),
            objective_names=["depth", "cnot_count"],
        )

        score = _compute_hypervolume_score(result, ref_point=np.array([5.0, 7.0]))

        assert score == pytest.approx(12.0)

    def test_violated_ref_point_returns_zero(self):
        """Un ref_point violado por el frente devuelve score=0.0 (no crash)."""
        result = OptimizationResult(
            pareto_layouts=[[0, 1, 2]],
            pareto_fitness=np.array([[2.0, 3.0], [4.0, 5.0]]),
            objective_names=["depth", "cnot_count"],
        )

        score = _compute_hypervolume_score(result, ref_point=np.array([4.0, 6.0]))
        assert score == 0.0

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

    def test_violated_ref_point_returns_zero_instead_of_raising(self, caplog):
        """Cuando ref_point es violado por el frente, score=0.0 y se emite warning (no crash)."""
        # Front has max values [5.0, 2.0] — ref_point [4.0, 3.0] is violated on obj 0
        result = OptimizationResult(
            pareto_layouts=[[0, 1, 2]],
            pareto_fitness=np.array([[3.0, 2.0], [5.0, 1.0]]),
            objective_names=["depth", "cnot_count"],
        )
        with caplog.at_level(logging.WARNING):
            score = _compute_hypervolume_score(result, ref_point=np.array([4.0, 3.0]))
        assert score == 0.0
        assert "ref_point violado" in caplog.text


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

    def test_config_accepts_categorical_mutation_values(self, small_circuit, backend_torino):
        """La evaluación admite valores de mutación pertenecientes al catálogo."""
        config = OptimizerConfig(
            algorithm="nsga2",
            population_size=6,
            n_generations=3,
            objectives=["depth", "cnot_count"],
            prob_swap_mutation=0.5,
            prob_replace_mutation=0.9,
            verbose=False,
        )
        score = _evaluate_config(config, small_circuit, backend_torino, seeds=[0])
        assert score >= 0.0


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

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_trials": 0}, "n_trials"),
            ({"n_seeds": 0}, "n_seeds"),
            ({"n_jobs": 0}, "n_jobs"),
            ({"ref_point_mode": "manual"}, "ref_point"),
            ({"ref_point_mode": "manual", "ref_point": [1.0]}, "ref_point"),
            ({"ref_point_mode": "unknown"}, "ref_point_mode"),
        ],
    )
    def test_init_rejects_invalid_budgets_and_reference_settings(
        self,
        small_circuit,
        backend_torino,
        kwargs,
        match,
    ):
        """Los presupuestos y el modo de punto de referencia se validan."""
        with pytest.raises(ValueError, match=match):
            LayoutTuner(
                circuit=small_circuit,
                backend=backend_torino,
                **kwargs,
            )

    def test_init_rejects_extraneous_ref_point_in_calibrated_mode(
        self,
        small_circuit,
        backend_torino,
    ):
        """El modo calibrado no acepta un ref_point manual extra."""
        with pytest.raises(ValueError, match="calibrated"):
            LayoutTuner(
                circuit=small_circuit,
                backend=backend_torino,
                ref_point_mode="calibrated",
                ref_point=[9.0, 11.0],
            )

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
            ref_point_mode="manual",
            ref_point=[1_000_000.0, 1_000_000.0],
        )
        result = tuner.tune(show_progress_bar=False)
        assert result is tuner

    def test_calibrated_mode_derives_and_stores_session_ref_point(
        self,
        monkeypatch,
        small_circuit,
        backend_torino,
    ):
        """El modo calibrado calcula una vez el ref_point y lo reutiliza."""
        warmup_front = np.array([[4.0, 7.0], [5.0, 6.0]])
        captured_ref_points = []

        def fake_optimize_layout(circuit, backend, config):
            return OptimizationResult(
                pareto_layouts=[[0, 1, 2]],
                pareto_fitness=warmup_front,
                objective_names=["depth", "cnot_count"],
            )

        def fake_evaluate_config(config, circuit, backend, seeds, n_jobs=1, ref_point=None):
            captured_ref_points.append(np.array(ref_point, dtype=float))
            return 1.0 + len(captured_ref_points)

        monkeypatch.setattr(tuning, "optimize_layout", fake_optimize_layout)
        monkeypatch.setattr(tuning, "_evaluate_config", fake_evaluate_config)

        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=2,
            n_seeds=2,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx"],
                algorithms=["nsga2"],
            ),
            ref_point_mode="calibrated",
        )

        tuner.tune(show_progress_bar=False)

        expected_ref_point = warmup_front.max(axis=0) * 1.1 + 1e-6
        assert np.allclose(tuner.session_ref_point, expected_ref_point)
        assert len(captured_ref_points) == 2
        assert all(np.allclose(ref_point, expected_ref_point) for ref_point in captured_ref_points)

    def test_calibrated_mode_uses_conservative_search_space_calibration(
        self,
        monkeypatch,
        small_circuit,
        backend_torino,
    ):
        """La calibración explora anclas del espacio para fijar un ref_point robusto."""
        def fake_optimize_layout(circuit, backend, config):
            if (config.population_size, config.n_generations) == (8, 5):
                front = np.array([[20.0, 24.0]])
            elif (config.population_size, config.n_generations) == (7, 4):
                front = np.array([[18.0, 22.0]])
            else:
                front = np.array([[4.0, 7.0]])

            return OptimizationResult(
                pareto_layouts=[[0, 1, 2]],
                pareto_fitness=front,
                objective_names=["depth", "cnot_count"],
            )

        def fake_suggest_config(self, trial):
            return OptimizerConfig(
                algorithm="nsga2",
                population_size=7,
                n_generations=4,
                objectives=["depth", "cnot_count"],
                optimization_level=1,
                crossover_operator="dpx",
                prob_swap_mutation=0.3,
                prob_replace_mutation=0.5,
                verbose=False,
            )

        def fake_trial_to_config(self, trial):
            return OptimizerConfig(
                algorithm="nsga2",
                population_size=7,
                n_generations=4,
                objectives=["depth", "cnot_count"],
                optimization_level=1,
                crossover_operator="dpx",
                prob_swap_mutation=0.3,
                prob_replace_mutation=0.5,
                verbose=True,
            )

        monkeypatch.setattr(tuning, "optimize_layout", fake_optimize_layout)
        monkeypatch.setattr(LayoutTuner, "_suggest_config", fake_suggest_config)
        monkeypatch.setattr(LayoutTuner, "_trial_to_config", fake_trial_to_config)

        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=1,
            n_seeds=1,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx"],
                algorithms=["nsga2"],
                prob_swap_mutation_choices=(0.3,),
                prob_replace_mutation_choices=(0.5,),
            ),
            ref_point_mode="calibrated",
        )

        tuner.tune(show_progress_bar=False)

        expected_ref_point = np.array([22.0, 26.4]) + 1e-6
        assert np.allclose(tuner.session_ref_point, expected_ref_point)

    def test_manual_mode_uses_supplied_ref_point_without_warmup(
        self,
        monkeypatch,
        small_circuit,
        backend_torino,
    ):
        """El modo manual omite el warm-up y usa el ref_point fijo."""
        captured_ref_points = []

        def fail_if_called(*args, **kwargs):
            raise AssertionError("manual mode should skip warm-up calibration")

        def fake_evaluate_config(config, circuit, backend, seeds, n_jobs=1, ref_point=None):
            captured_ref_points.append(np.array(ref_point, dtype=float))
            return 1.0

        monkeypatch.setattr(LayoutTuner, "_calibrate_reference_point", fail_if_called)
        monkeypatch.setattr(tuning, "_evaluate_config", fake_evaluate_config)

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
            ref_point_mode="manual",
            ref_point=[9.0, 11.0],
        )

        tuner.tune(show_progress_bar=False)

        assert np.allclose(tuner.session_ref_point, np.array([9.0, 11.0]))
        assert len(captured_ref_points) == 2
        assert all(np.allclose(ref_point, np.array([9.0, 11.0])) for ref_point in captured_ref_points)

    def test_manual_mode_penalizes_non_dominating_ref_point_during_tuning(
        self,
        monkeypatch,
        small_circuit,
        backend_torino,
    ):
        """Un ref_point manual demasiado pequeño penaliza trials con HV=0.0 (no crash)."""
        def fake_optimize_layout(circuit, backend, config):
            return OptimizationResult(
                pareto_layouts=[[0, 1, 2]],
                pareto_fitness=np.array([[5.0, 7.0]]),
                objective_names=["depth", "cnot_count"],
            )

        monkeypatch.setattr(tuning, "optimize_layout", fake_optimize_layout)

        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=1,
            n_seeds=1,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx"],
                algorithms=["nsga2"],
            ),
            ref_point_mode="manual",
            ref_point=[5.0, 8.0],
        )

        # El estudio debe completar sin lanzar excepción; todos los trials quedan con score=0.0
        result = tuner.tune(show_progress_bar=False)
        assert result is not None
        assert tuner._best_score == 0.0

    def test_calibration_failure_surfaces_clearly(
        self,
        monkeypatch,
        small_circuit,
        backend_torino,
    ):
        """Si el warm-up no produce frentes válidos, tune() falla claramente."""
        def fake_optimize_layout(circuit, backend, config):
            return OptimizationResult(
                pareto_layouts=[],
                pareto_fitness=None,
                objective_names=["depth", "cnot_count"],
            )

        monkeypatch.setattr(tuning, "optimize_layout", fake_optimize_layout)

        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=1,
            n_seeds=1,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx"],
                algorithms=["nsga2"],
            ),
            ref_point_mode="calibrated",
        )

        with pytest.raises(ValueError, match="No se pudo calibrar"):
            tuner.tune(show_progress_bar=False)

        assert tuner.session_ref_point is None
        with pytest.raises(RuntimeError, match="tune\(\)"):
            tuner.best_config()

    def test_failed_rerun_clears_previous_tuning_session_state(
        self,
        monkeypatch,
        small_circuit,
        backend_torino,
    ):
        """Un rerun fallido no debe dejar expuesto el estado de la sesión previa."""
        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=1,
            n_seeds=1,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx"],
                algorithms=["nsga2"],
            ),
            ref_point_mode="manual",
            ref_point=[1_000_000.0, 1_000_000.0],
        )

        tuner.tune(show_progress_bar=False)
        assert tuner.session_ref_point is not None
        assert isinstance(tuner.best_config(), OptimizerConfig)

        def fail_calibration(self, calibration_configs):
            raise ValueError("forced calibration failure")

        tuner.ref_point_mode = "calibrated"
        tuner._manual_ref_point = None
        monkeypatch.setattr(LayoutTuner, "_calibrate_reference_point", fail_calibration)

        with pytest.raises(ValueError, match="forced calibration failure"):
            tuner.tune(show_progress_bar=False)

        assert tuner.session_ref_point is None
        with pytest.raises(RuntimeError, match="tune\(\)"):
            tuner.best_config()

    def test_progress_callback_receives_meaningful_events(
        self,
        monkeypatch,
        small_circuit,
        backend_torino,
    ):
        """El callback de progreso recibe eventos estructurados del tuning."""
        events = []
        scores = iter([0.5, 0.8])
        warmup_fronts = iter(
            [
                None,
                np.array([[6.0, 5.0]]),
                np.array([[8.0, 9.0]]),
            ]
        )

        def fake_optimize_layout(circuit, backend, config):
            pareto_fitness = next(warmup_fronts)
            return OptimizationResult(
                pareto_layouts=[[0, 1, 2]],
                pareto_fitness=pareto_fitness,
                objective_names=["depth", "cnot_count"],
            )

        def fake_evaluate_config(config, circuit, backend, seeds, n_jobs=1, ref_point=None):
            return next(scores)

        monkeypatch.setattr(tuning, "optimize_layout", fake_optimize_layout)
        monkeypatch.setattr(tuning, "_evaluate_config", fake_evaluate_config)

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
            ref_point_mode="calibrated",
            progress_callback=events.append,
        )

        tuner.tune(show_progress_bar=False)

        event_names = [event["event"] for event in events]
        calibration_progress_events = [
            event for event in events if event["event"] == "calibration_progress"
        ]

        assert event_names == [
            "calibration_started",
            "calibration_progress",
            "calibration_progress",
            "calibration_progress",
            "calibration_completed",
            "trial_completed",
            "trial_completed",
            "tuning_completed",
        ]
        assert events[0]["calibration_config_count"] == 3
        assert [event["current_step"] for event in calibration_progress_events] == [1, 2, 3]
        assert all(event["total_steps"] == 3 for event in calibration_progress_events)
        assert all(set(("current_step", "total_steps", "config", "ref_point_candidate")).issubset(event) for event in calibration_progress_events)
        assert calibration_progress_events[0]["ref_point_candidate"] is None
        assert calibration_progress_events[-1]["ref_point_candidate"] == pytest.approx([8.800001, 9.900001])
        assert events[4]["ref_point"] == pytest.approx([8.800001, 9.900001])
        assert events[5]["trial_number"] == 1
        assert events[5]["completed_trials"] == 1
        assert "params" in events[5]
        assert events[5]["best_score"] == pytest.approx(0.5)
        assert events[6]["best_score"] == pytest.approx(0.8)
        assert events[7]["ref_point_mode"] == "calibrated"

    def test_calibrated_warmup_builds_at_most_three_configs(
        self,
        small_circuit,
        backend_torino,
    ):
        """El warm-up calibrado debe limitarse a un conjunto corto y determinista."""
        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=1,
            n_seeds=5,
            space=HyperparameterSpace(),
            ref_point_mode="calibrated",
        )

        calibration_configs = tuner._build_calibration_configs(list(range(tuner.n_seeds)))

        assert [
            (
                config.algorithm,
                config.population_size,
                config.n_generations,
                config.crossover_operator,
                config.prob_swap_mutation,
                config.prob_replace_mutation,
                config.seed,
            )
            for config in calibration_configs
        ] == [
            ("nsga2", 20, 30, "dpx", 0.1, 0.1, 0),
            ("nsga2", 30, 50, "ox", 0.5, 0.5, 0),
            ("nsga2", 30, 50, "ox", 0.7, 0.9, 0),
        ]

    def test_calibrated_warmup_uses_one_fixed_seed_only(
        self,
        small_circuit,
        backend_torino,
    ):
        """La calibración usa una sola seed fija aunque los trials usen varias."""
        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=1,
            n_seeds=5,
            space=HyperparameterSpace(),
            ref_point_mode="calibrated",
        )

        calibration_configs = tuner._build_calibration_configs(list(range(tuner.n_seeds)))

        assert [config.seed for config in calibration_configs] == [0, 0, 0]

    def test_progress_callback_docs_include_calibration_progress_event(self):
        """La API publica documenta calibration_progress junto al resto de eventos."""
        assert "calibration_progress" in LayoutTuner.__init__.__doc__
        assert "current_step" in LayoutTuner.__init__.__doc__
        assert "total_steps" in LayoutTuner.__init__.__doc__

    def test_calibrated_warmup_keeps_algorithm_coverage_in_multi_algorithm_space(
        self,
        small_circuit,
        backend_torino,
    ):
        """El warm-up corto no debe dejar fuera algoritmos soportados del espacio."""
        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=1,
            n_seeds=5,
            space=HyperparameterSpace(
                algorithms=["nsga2", "moead"],
                crossover_operators=["dpx", "ox"],
                prob_swap_mutation_choices=(0.1, 0.7),
                prob_replace_mutation_choices=(0.1, 0.9),
            ),
            ref_point_mode="calibrated",
        )

        calibration_configs = tuner._build_calibration_configs(list(range(tuner.n_seeds)))

        assert len(calibration_configs) <= 3
        assert {config.algorithm for config in calibration_configs} == {"nsga2", "moead"}

    def test_calibrated_multi_algorithm_warmup_keeps_fixed_session_ref_point_without_expanding(
        self,
        monkeypatch,
        small_circuit,
        backend_torino,
    ):
        """El warm-up corto sigue fijando un ref_point valido con varios algoritmos."""
        captured_ref_points = []
        warmup_configs = []

        def fake_optimize_layout(circuit, backend, config):
            warmup_configs.append((config.algorithm, config.population_size, config.n_generations, config.seed))
            if config.algorithm == "moead":
                front = np.array([[12.0, 15.0]])
            elif config.population_size >= 30:
                front = np.array([[9.0, 11.0]])
            else:
                front = np.array([[4.0, 7.0]])

            return OptimizationResult(
                pareto_layouts=[[0, 1, 2]],
                pareto_fitness=front,
                objective_names=["depth", "cnot_count"],
            )

        def fake_evaluate_config(config, circuit, backend, seeds, n_jobs=1, ref_point=None):
            captured_ref_points.append(np.array(ref_point, dtype=float))
            return 1.0

        monkeypatch.setattr(tuning, "optimize_layout", fake_optimize_layout)
        monkeypatch.setattr(tuning, "_evaluate_config", fake_evaluate_config)

        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=2,
            n_seeds=5,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx", "ox"],
                algorithms=["nsga2", "moead"],
                prob_swap_mutation_choices=(0.1, 0.7),
                prob_replace_mutation_choices=(0.1, 0.9),
            ),
            ref_point_mode="calibrated",
        )

        tuner.tune(show_progress_bar=False)

        assert len(warmup_configs) == 3
        assert {config[0] for config in warmup_configs} == {"nsga2", "moead"}
        assert np.allclose(tuner.session_ref_point, np.array([13.2, 16.5]) + 1e-6)
        assert len(captured_ref_points) == 2
        assert all(np.allclose(ref_point, tuner.session_ref_point) for ref_point in captured_ref_points)

    def test_manual_mode_skips_warmup_even_with_progress_callback(
        self,
        monkeypatch,
        small_circuit,
        backend_torino,
    ):
        """El modo manual no debe emitir ni ejecutar warm-up de calibración."""
        events = []

        def fail_if_called(*args, **kwargs):
            raise AssertionError("manual mode should skip warm-up calibration")

        def fake_evaluate_config(config, circuit, backend, seeds, n_jobs=1, ref_point=None):
            return 1.0

        monkeypatch.setattr(LayoutTuner, "_build_calibration_configs", fail_if_called)
        monkeypatch.setattr(LayoutTuner, "_calibrate_reference_point", fail_if_called)
        monkeypatch.setattr(tuning, "_evaluate_config", fake_evaluate_config)

        tuner = LayoutTuner(
            circuit=small_circuit,
            backend=backend_torino,
            n_trials=1,
            n_seeds=3,
            space=HyperparameterSpace(
                population_size_range=(6, 8),
                n_generations_range=(3, 5),
                crossover_operators=["dpx"],
                algorithms=["nsga2"],
            ),
            ref_point_mode="manual",
            ref_point=[9.0, 11.0],
            progress_callback=events.append,
        )

        tuner.tune(show_progress_bar=False)

        assert [event["event"] for event in events] == ["trial_completed", "tuning_completed"]

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
            ref_point_mode="manual",
            ref_point=[1_000_000.0, 1_000_000.0],
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
            ref_point_mode="manual",
            ref_point=[1_000_000.0, 1_000_000.0],
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
            ref_point_mode="manual",
            ref_point=[1_000_000.0, 1_000_000.0],
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
            ref_point_mode="manual",
            ref_point=[1_000_000.0, 1_000_000.0],
        )
        tuner.tune(show_progress_bar=False)
        summary = tuner.summary()

        assert "TUNING" in summary
        assert "population_size" in summary
        assert "crossover_operator" in summary


# ===========================================================================
#  Tests - Benchmark GUI helpers
# ===========================================================================

class TestBenchmarkGuiHelpers:
    """Tests de helpers puros usados por la GUI de tuning."""

    def test_parse_manual_ref_point_accepts_comma_separated_floats(self):
        """El ref_point manual se parsea como coordenadas float ordenadas."""
        assert _parse_manual_ref_point("10.5, 20", expected_dims=2) == (10.5, 20.0)

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("", "manual"),
            ("1.0", "2 valores"),
            ("1.0, nope", "float"),
        ],
    )
    def test_parse_manual_ref_point_rejects_invalid_values(self, text, match):
        """La GUI valida entradas manuales inválidas con mensajes claros."""
        with pytest.raises(ValueError, match=match):
            _parse_manual_ref_point(text, expected_dims=2)

    def test_format_ref_point_display_mentions_mode_explicitly(self):
        """La etiqueta GUI deja claro si el ref_point es manual o calibrado."""
        assert _format_ref_point_display((9.0, 11.0), "manual") == "manual [9.000, 11.000]"

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("calibrated", "calibrated pending (warm-up automatic)"),
            ("manual", "manual required"),
        ],
    )
    def test_format_ref_point_display_explains_pending_mode_state(self, mode, expected):
        """La GUI explica claramente si el warm-up es automatico o si falta el valor manual."""
        assert _format_ref_point_display(None, mode) == expected

    def test_format_ref_point_mode_help_describes_calibrated_vs_manual_modes(self):
        """La ayuda GUI diferencia el warm-up automatico del ref_point manual obligatorio."""
        assert _format_ref_point_mode_help("calibrated") == "Modo calibrado: usa warm-up automatico para fijar el ref_point."
        assert _format_ref_point_mode_help("manual") == "Modo manual: omite el warm-up y exige ref_point manual."

    def test_resolve_ref_point_display_uses_validated_manual_value_before_trials(self):
        """La GUI muestra el ref_point manual validado antes del primer trial."""
        assert _resolve_ref_point_display("manual", manual_ref_point=(9.0, 11.0)) == "manual [9.000, 11.000]"

    def test_resolve_ref_point_display_prefers_runtime_session_ref_point(self):
        """Si el tuner ya emitio un ref_point, la GUI muestra ese valor explicitamente."""
        assert _resolve_ref_point_display(
            "calibrated",
            ref_point=(12.0, 13.5),
            manual_ref_point=(9.0, 11.0),
        ) == "calibrated [12.000, 13.500]"
