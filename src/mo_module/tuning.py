"""
tuning.py — Ajuste automático de hiperparámetros del optimizador MO
====================================================================

Módulo 3 del TFG: Optimización Multiobjetivo.

Este fichero implementa el tuning de hiperparámetros de ``OptimizerConfig``
usando **Optuna** (Bayesian Optimization con el algoritmo TPE por defecto).

Contexto y motivación:
  ``OptimizerConfig`` contiene hiperparámetros que afectan directamente a la
  calidad del frente de Pareto generado por NSGA-II o MOEA/D:

    - ``population_size``      — tamaño de la población evolutiva
    - ``n_generations``        — presupuesto de generaciones
    - ``prob_swap_mutation``   — categoría de mutación por intercambio
    - ``prob_replace_mutation``— categoría de mutación por reemplazo de qubit
    - ``crossover_operator``   — tipo de cruce (DPX o OX)

  La elección manual de estos valores es heurística y subóptima. Optuna
  realiza una búsqueda inteligente (no aleatoria) sobre el espacio de
  hiperparámetros buscando la configuración que maximice el **hipervolumen**
  del frente de Pareto.

Por qué Optuna y no pymoo HyperparameterProblem:
  pymoo provee ``HyperparameterProblem`` que autodescubre parámetros de sus
  algoritmos built-in (DE, G3PCX...). Sin embargo, nuestro NSGA-II usa
  operadores custom (``DPXCrossover``, ``LayoutMutation``) cuyos parámetros
  viven en ``OptimizerConfig``, no en el objeto del algoritmo. Por tanto,
  la integración nativa de pymoo no aplica aquí y Optuna es la solución
  más directa y potente.

Pipeline de tuning:
  1. Se define un ``HyperparameterSpace`` con los espacios de búsqueda.
  2. ``LayoutTuner.tune()`` lanza un estudio Optuna con ``n_trials``
     evaluaciones (configurable; por defecto 30 — moderado).
  3. Cada trial:
       a. Optuna sugiere una ``OptimizerConfig`` dentro del espacio.
       b. Se ejecuta ``optimize_layout()`` con ``n_seeds`` semillas distintas.
       c. Se calcula el **hipervolumen medio** del frente de Pareto.
       d. Optuna recibe el score y actualiza su modelo TPE.
  4. Al finalizar, ``LayoutTuner.best_config()`` devuelve el
     ``OptimizerConfig`` óptimo listo para usar.

Coste computacional:
  Cada trial ejecuta ``n_seeds`` runs de NSGA-II. Por defecto:
    - n_trials = 30 (fácilmente ajustable con ``LayoutTuner(n_trials=...)``).
    - n_seeds  = 3  (promedio robusto pero rápido).
    - Se recomienda usar circuitos pequeños (≤ 5 qubits) para tuning.

Autor: Eduardo González Bautista
Fecha: 2026-03-03
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit

from .optimizer import (
    DEFAULT_REPLACE_MUTATION_CATEGORIES,
    DEFAULT_SWAP_MUTATION_CATEGORIES,
    OptimizationResult,
    OptimizerConfig,
    optimize_layout,
)
from ..qiskit_interface.backend_info import get_backend

logger = logging.getLogger(__name__)

_SUPPORTED_CROSSOVER_OPERATORS = {"dpx", "ox"}
_SUPPORTED_ALGORITHMS = {"nsga2", "moead"}


# ===========================================================================
#  Espacio de búsqueda de hiperparámetros
# ===========================================================================

@dataclass
class HyperparameterSpace:
    """Define el espacio de búsqueda de hiperparámetros del optimizador.

    Cada campo representa los límites (inclusivos) o las opciones
    válidas para el hiperparámetro correspondiente de ``OptimizerConfig``.

    Los valores por defecto están calibrados para circuitos de ≤ 10 qubits
    en backends de tamaño medio (ej. FakeTorino 133q). Para problemas más
    grandes o pequeños, ajustar los rangos según corresponda.

    Attributes:
        population_size_range:
            Rango ``(min, max)`` para el tamaño de la población. Valores
            fuera de este rango se descartan automáticamente.
        n_generations_range:
            Rango ``(min, max)`` para el número de generaciones.
        prob_swap_mutation_choices:
            Categorías discretas para la probabilidad de mutación swap.
        prob_replace_mutation_choices:
            Categorías discretas para la probabilidad de mutación replace.
        crossover_operators:
            Lista de operadores de cruce a considerar. El estudio elegirá
            el mejor automáticamente.
        algorithms:
            Lista de algoritmos evolutivos a explorar. Si solo se quiere
            optimizar los hiperparámetros de NSGA-II, dejar como
            ``["nsga2"]``.
        optimization_level:
            Nivel de optimización de Qiskit. Se mantiene fijo durante el
            tuning para que no sea una variable de confusión.
    """

    population_size_range: tuple[int, int] = (20, 100)
    n_generations_range: tuple[int, int] = (30, 150)
    prob_swap_mutation_choices: tuple[float, ...] = DEFAULT_SWAP_MUTATION_CATEGORIES
    prob_replace_mutation_choices: tuple[float, ...] = DEFAULT_REPLACE_MUTATION_CATEGORIES
    crossover_operators: list[str] = field(
        default_factory=lambda: ["dpx", "ox"]
    )
    algorithms: list[str] = field(
        default_factory=lambda: ["nsga2"]
    )
    optimization_level: int = 1

    def __post_init__(self):
        """Valida el espacio de búsqueda."""
        if len(self.population_size_range) != 2:
            raise ValueError("population_size_range debe tener exactamente dos valores.")
        if self.population_size_range[0] > self.population_size_range[1]:
            raise ValueError("population_size_range debe estar ordenado como (min, max).")
        if self.population_size_range[0] < 4:
            raise ValueError("population_size_range debe empezar en un valor >= 4.")

        if len(self.n_generations_range) != 2:
            raise ValueError("n_generations_range debe tener exactamente dos valores.")
        if self.n_generations_range[0] > self.n_generations_range[1]:
            raise ValueError("n_generations_range debe estar ordenado como (min, max).")
        if self.n_generations_range[0] < 1:
            raise ValueError("n_generations_range debe empezar en un valor >= 1.")

        if not self.prob_swap_mutation_choices:
            raise ValueError("prob_swap_mutation_choices no puede estar vacío.")
        if not self.prob_replace_mutation_choices:
            raise ValueError("prob_replace_mutation_choices no puede estar vacío.")
        if not self.crossover_operators:
            raise ValueError("crossover_operators no puede estar vacío.")
        if not set(self.crossover_operators).issubset(_SUPPORTED_CROSSOVER_OPERATORS):
            raise ValueError("crossover_operators contiene valores no soportados.")
        if not self.algorithms:
            raise ValueError("algorithms no puede estar vacío.")
        if not set(self.algorithms).issubset(_SUPPORTED_ALGORITHMS):
            raise ValueError("algorithms contiene valores no soportados.")


# ===========================================================================
#  Función de evaluación: calidad del frente de Pareto
# ===========================================================================

def _compute_hypervolume_score(
    opt_result: OptimizationResult,
    ref_point: Optional[Sequence[float]] = None,
) -> float:
    """Calcula el hipervolumen del frente de Pareto como métrica de calidad.

    El **hipervolumen** (HV) mide el volumen del espacio objetivo dominado
    por el frente de Pareto respecto a un punto de referencia. Un HV mayor
    indica un frente de mejor calidad (soluciones mejores y más diversas).

    Se usa como función objetivo del tuning porque:
      1. Es una métrica escalar del frente completo (no solo una solución).
      2. Es la métrica más usada en la literatura de MO para comparar
         algoritmos (Zitzler et al., 2003).
      3. Penaliza tanto la mala calidad como la falta de diversidad.

    Args:
        opt_result: Resultado de una ejecución de ``optimize_layout()``.

    Returns:
        Hipervolumen calculado. Devuelve 0.0 si no hay soluciones, si el
        ref_point es violado por el frente de Pareto, o si el cálculo falla.
    """
    if opt_result.pareto_fitness is None or len(opt_result.pareto_fitness) == 0:
        return 0.0

    try:
        from pymoo.indicators.hv import HV

        F = np.asarray(opt_result.pareto_fitness, dtype=float)  # shape: (n_solutions, n_objectives)

        if ref_point is None:
            # Punto de referencia: nadir del frente con un margen del 10 %
            # (mayor que el peor valor en cada objetivo → el HV es siempre > 0)
            hv_ref_point = F.max(axis=0) * 1.1 + 1e-6
        else:
            hv_ref_point = np.asarray(ref_point, dtype=float)
            if hv_ref_point.shape != (F.shape[1],):
                raise ValueError("ref_point debe tener una coordenada por objetivo.")
            if not np.all(np.isfinite(hv_ref_point)):
                raise ValueError("ref_point debe contener valores finitos.")
            if not np.all(hv_ref_point > F.max(axis=0)):
                # El frente de este trial excede el ref_point calibrado.
                # Devolvemos 0.0 (peor HV posible) en lugar de crashear el estudio.
                # Optuna aprenderá a evitar estas regiones del espacio de búsqueda.
                logger.warning(
                    "ref_point violado: front_max=%s ref_point=%s — trial penalizado con HV=0.0",
                    F.max(axis=0).tolist(),
                    hv_ref_point.tolist(),
                )
                return 0.0

        ind = HV(ref_point=hv_ref_point)
        hv = float(ind(F))
        return hv

    except ValueError:
        raise
    except Exception as exc:
        logger.warning("Error calculando hipervolumen: %s", exc)
        return 0.0


def _evaluate_single_seed(
    config: OptimizerConfig,
    circuit: QuantumCircuit,
    backend,
    seed: int,
    ref_point: Optional[Sequence[float]] = None,
) -> tuple[int, float]:
    """Evalúa una configuración para una semilla específica."""
    try:
        trial_config = OptimizerConfig(
            algorithm=config.algorithm,
            population_size=config.population_size,
            n_generations=config.n_generations,
            objectives=config.objectives,
            optimization_level=config.optimization_level,
            crossover_operator=config.crossover_operator,
            prob_swap_mutation=config.prob_swap_mutation,
            prob_replace_mutation=config.prob_replace_mutation,
            seed=seed,
            verbose=False,
        )
        result = optimize_layout(circuit, backend=backend, config=trial_config)
        hv = _compute_hypervolume_score(result, ref_point=ref_point)
        return seed, hv
    except ValueError:
        raise
    except Exception as exc:
        logger.warning("Evaluación fallida para seed=%d: %s", seed, exc)
        return seed, 0.0

def _evaluate_config(
    config: OptimizerConfig,
    circuit: QuantumCircuit,
    backend,
    seeds: Sequence[int],
    n_jobs: int = 1,
    ref_point: Optional[Sequence[float]] = None,
) -> float:
    """Evalúa una configuración ejecutando optimize_layout con múltiples seeds.

    Promediar sobre varias seeds es crucial para obtener una estimación
    robusta de la calidad, ya que los algoritmos evolutivos son estocásticos.

    Args:
        config: Configuración del optimizador a evaluar.
        circuit: Circuito de referencia para el tuning.
        backend: Backend de referencia.
        seeds: Lista de semillas a usar. El score final es la media
            del hipervolumen sobre todas las ejecuciones.
        n_jobs: Número de procesos paralelos para evaluar las seeds.

    Returns:
        Hipervolumen medio (mayor es mejor). Negativo si todas las
        ejecuciones fallaron.
    """
    scores: list[float] = []

    if n_jobs == 1:
        for seed in seeds:
            _, hv = _evaluate_single_seed(config, circuit, backend, seed, ref_point=ref_point)
            scores.append(hv)
            logger.debug("seed=%d → HV=%.6f", seed, hv)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futs = [
                executor.submit(
                    _evaluate_single_seed,
                    config,
                    circuit,
                    backend,
                    seed,
                    ref_point,
                )
                for seed in seeds
            ]
            for fut in as_completed(futs):
                seed, hv = fut.result()
                scores.append(hv)
                logger.debug("seed=%d → HV=%.6f", seed, hv)

    if not scores:
        return 0.0

    mean_hv = float(np.mean(scores))
    logger.debug("HV medio para la config: %.6f", mean_hv)
    return mean_hv


# ===========================================================================
#  Tuner principal
# ===========================================================================

class LayoutTuner:
    """Ajuste automático de hiperparámetros del optimizador MO mediante Optuna.

    Usa **TPE (Tree-structured Parzen Estimator)**, el algoritmo de
    Bayesian Optimization por defecto de Optuna, para explorar eficientemente
    el espacio de hiperparámetros definido en ``HyperparameterSpace``.

    El tuning maximiza el **hipervolumen medio** del frente de Pareto sobre
    múltiples semillas, lo que asegura que la configuración encontrada sea
    robusta y no solo buena para una seed concreta.

    Uso típico::

        from src.mo_module.tuning import LayoutTuner
        from src.qiskit_interface import create_ghz_circuit, get_backend

        circuit = create_ghz_circuit(5)
        backend = get_backend("fake_torino")

        tuner = LayoutTuner(
            circuit=circuit,
            backend=backend,
            n_trials=30,   # presupuesto de evaluaciones (ajustable)
            n_seeds=3,      # seeds por evaluación (robustez)
        )
        tuner.tune()
        best = tuner.best_config()
        print(best)

    Attributes:
        circuit:
            Circuito de referencia usado durante el tuning. Se recomienda
            usar un circuito representativo del caso de uso real, pero
            pequeño (≤ 7 qubits) para mantener el tuning manejable.
        backend:
            Backend destino para la transpilación.
        n_trials:
            Número de evaluaciones del espacio de hiperparámetros. Cada
            trial ejecuta ``n_seeds`` runs de NSGA-II. **Por defecto: 30**,
            que ofrece un balance entre calidad de la búsqueda y tiempo
            (~15-30 min dependiendo del circuito). Aumentar a 50-100 para
            búsquedas más exhaustivas.
        n_seeds:
            Número de semillas por trial para promediar el hipervolumen.
            Por defecto: 3. Aumentar para mayor robustez estadística.
        space:
            Espacio de búsqueda de hiperparámetros. Si es None, se usa
            ``HyperparameterSpace()`` con valores por defecto.
        objectives:
            Lista de objetivos a optimizar durante el tuning.
        study_name:
            Nombre del estudio Optuna (útil para guardarlo en base de datos).
    """

    # Presupuesto por defecto: 30 trials con 3 seeds = 90 runs de NSGA-II.
    # ~20-30 min con circuito de 5 qubits en FakeTorino.
    # Modificar n_trials al instanciar: LayoutTuner(..., n_trials=50)
    DEFAULT_N_TRIALS: int = 30
    DEFAULT_N_SEEDS: int = 3
    DEFAULT_EVAL_GENERATIONS: int = 50   # generaciones por trial (reducido)
    DEFAULT_EVAL_POPULATION: int = 30    # población por trial (reducido)

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend=None,
        backend_name: str = "fake_torino",
        n_trials: int = DEFAULT_N_TRIALS,
        n_seeds: int = DEFAULT_N_SEEDS,
        n_jobs: int = 1,
        space: Optional[HyperparameterSpace] = None,
        objectives: Optional[list[str]] = None,
        study_name: str = "layout_hyperparameter_tuning",
        ref_point_mode: str = "calibrated",
        ref_point: Optional[Sequence[float]] = None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        """
        Args:
            circuit:
                Circuito de referencia para el tuning. Usar un circuito
                pequeño y representativo para mantener tiempos manejables.
            backend:
                Backend de destino. Si es None, se instancia con
                ``backend_name``.
            backend_name:
                Nombre del backend si ``backend`` es None.
            n_trials:
                Número de trials Optuna. **Por defecto 30** (moderado).
                Cada trial = n_seeds runs de NSGA-II con configuración
                reducida. Aumentar para búsquedas más exhaustivas.
            n_seeds:
                Número de seeds por trial. Mayor → más robusto pero más lento.
            space:
                Espacio de búsqueda. None usa ``HyperparameterSpace()``.
            objectives:
                Lista de objetivos a optimizar. None usa
                ``["depth", "cnot_count"]``.
            study_name:
                Nombre del estudio Optuna. Útil para almacenamiento.
            ref_point_mode:
                Estrategia del punto de referencia del hipervolumen para toda
                la sesión. ``"manual"`` exige ``ref_point`` explícito y omite
                la calibración. ``"calibrated"`` construye un único
                ``session_ref_point`` conservador antes de lanzar los trials.
            ref_point:
                Punto de referencia fijo del hipervolumen cuando
                ``ref_point_mode="manual"``. Debe tener una coordenada por
                objetivo. En modo calibrado se rechaza para evitar ambigüedad.
            progress_callback:
                Callback opcional que recibe eventos estructurados ``dict``.
                Eventos actuales: ``calibration_started``,
                ``calibration_progress``, ``calibration_completed``,
                ``trial_completed`` y ``tuning_completed``. Incluyen
                metadatos pensados para GUI, como modo/punto de referencia,
                recuentos de progreso y métricas. En particular,
                ``calibration_progress`` emite ``current_step`` y ``total_steps``
                junto con la ``config`` evaluada y el
                ``ref_point_candidate`` acumulado hasta ese paso.
        """
        if backend is None:
            backend = get_backend(backend_name)

        self.circuit = circuit
        self.backend = backend
        if n_trials < 1:
            raise ValueError("n_trials debe ser >= 1.")
        if n_seeds < 1:
            raise ValueError("n_seeds debe ser >= 1.")
        if n_jobs < 1:
            raise ValueError("n_jobs debe ser >= 1.")

        if ref_point_mode not in ("calibrated", "manual"):
            raise ValueError("ref_point_mode debe ser 'calibrated' o 'manual'.")

        self.n_trials = n_trials
        self.n_seeds = n_seeds
        self.n_jobs = n_jobs
        self.space = space or HyperparameterSpace()
        self.objectives = objectives or ["depth", "cnot_count"]
        self.study_name = study_name
        self.ref_point_mode = ref_point_mode
        self.progress_callback = progress_callback

        self._study = None          # Estudio Optuna (None hasta tune())
        self._best_config: Optional[OptimizerConfig] = None
        self._best_score: Optional[float] = None
        self._manual_ref_point = self._validate_manual_ref_point(ref_point)
        self._session_ref_point: Optional[np.ndarray] = None

    @property
    def session_ref_point(self) -> Optional[np.ndarray]:
        """Devuelve el punto de referencia fijo de la sesión, si existe.

        Este valor queda congelado antes de evaluar los trials Optuna. En modo
        manual coincide con ``ref_point``; en modo calibrado se deriva a partir
        de varias configuraciones conservadoras del espacio de búsqueda.
        """
        if self._session_ref_point is None:
            return None
        return self._session_ref_point.copy()

    # -----------------------------------------------------------------------
    #  API pública
    # -----------------------------------------------------------------------

    def tune(self, show_progress_bar: bool = True) -> "LayoutTuner":
        """Ejecuta el proceso de tuning.

        Lanza un estudio Optuna con ``n_trials`` evaluaciones. Cada trial
        sugiere una ``OptimizerConfig``, la evalúa con ``n_seeds`` runs y
        devuelve el hipervolumen medio a Optuna.

        Al finalizar, el mejor trial se almacena en ``self._best_config``.

        Args:
            show_progress_bar:
                Si True, muestra una barra de progreso durante el tuning.

        Returns:
            self (para encadenamiento: ``tuner.tune().best_config()``).
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Optuna no está instalado. Ejecuta: pip install optuna"
            ) from exc

        # Silenciar logs de Optuna (son muy verbosos por defecto)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info(
            "Iniciando tuning: %d trials × %d seeds (circuito='%s', backend='%s')",
            self.n_trials,
            self.n_seeds,
            self.circuit.name or "unnamed",
            getattr(self.backend, "name", "unknown"),
        )

        self._study = None
        self._best_config = None
        self._best_score = None
        self._session_ref_point = None

        self._study = optuna.create_study(
            direction="maximize",   # maximizar hipervolumen
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        seeds = list(range(self.n_seeds))

        if self.ref_point_mode == "manual":
            self._session_ref_point = self._manual_ref_point.copy()
        else:
            calibration_configs = self._build_calibration_configs(seeds)
            self._emit_progress(
                "calibration_started",
                ref_point_mode=self.ref_point_mode,
                n_seeds=self.n_seeds,
                calibration_config_count=len(calibration_configs),
            )
            self._session_ref_point = self._calibrate_reference_point(calibration_configs)
            self._emit_progress(
                "calibration_completed",
                ref_point_mode=self.ref_point_mode,
                ref_point=self._session_ref_point,
                calibration_config_count=len(calibration_configs),
            )

        def objective(trial: "optuna.Trial") -> float:
            config = self._suggest_config(trial)
            score = _evaluate_config(
                config,
                self.circuit,
                self.backend,
                seeds,
                self.n_jobs,
                ref_point=self._session_ref_point,
            )
            self._best_score = score if self._best_score is None else max(self._best_score, score)
            logger.info(
                "Trial %d/%d: pop=%d, gen=%d, cx=%s, swap=%s, rep=%s -> HV=%.6f",
                trial.number + 1, self.n_trials,
                config.population_size, config.n_generations,
                config.crossover_operator,
                config.prob_swap_mutation, config.prob_replace_mutation,
                score,
            )
            self._emit_progress(
                "trial_completed",
                trial_number=trial.number + 1,
                completed_trials=trial.number + 1,
                total_trials=self.n_trials,
                score=score,
                best_score=self._best_score,
                ref_point=self._session_ref_point,
                ref_point_mode=self.ref_point_mode,
                params={
                    "algorithm": config.algorithm,
                    "population_size": config.population_size,
                    "n_generations": config.n_generations,
                    "crossover_operator": config.crossover_operator,
                    "prob_swap_mutation": config.prob_swap_mutation,
                    "prob_replace_mutation": config.prob_replace_mutation,
                },
            )
            return score

        self._study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=show_progress_bar,
        )

        # Construir la config ganadora
        self._best_config = self._trial_to_config(self._study.best_trial)

        logger.info(
            "Tuning completado. Mejor HV=%.6f con config: %s",
            self._study.best_value,
            self._best_config,
        )

        self._emit_progress(
            "tuning_completed",
            total_trials=self.n_trials,
            best_score=self._study.best_value,
            ref_point=self._session_ref_point,
            ref_point_mode=self.ref_point_mode,
        )

        return self

    def best_config(self) -> OptimizerConfig:
        """Devuelve la ``OptimizerConfig`` óptima encontrada por el tuning.

        Returns:
            Configuración óptima lista para usar en ``optimize_layout()``.

        Raises:
            RuntimeError: Si ``tune()`` no se ha ejecutado aún.
        """
        if self._best_config is None:
            raise RuntimeError(
                "El tuning no se ha ejecutado todavía. Llama a tune() primero."
            )
        return self._best_config

    def summary(self) -> str:
        """Devuelve un resumen legible del proceso de tuning.

        Returns:
            String con el resumen del estudio Optuna.

        Raises:
            RuntimeError: Si ``tune()`` no se ha ejecutado aún.
        """
        if self._study is None:
            raise RuntimeError("El tuning no se ha ejecutado todavía.")

        best = self._study.best_trial
        lines = [
            "=" * 60,
            "  RESUMEN DE TUNING DE HIPERPARÁMETROS",
            "=" * 60,
            f"  Circuito:      {self.circuit.name or 'unnamed'}",
            f"  Backend:       {getattr(self.backend, 'name', 'unknown')}",
            f"  Trials:        {len(self._study.trials)} / {self.n_trials}",
            f"  Seeds/trial:   {self.n_seeds}",
            f"  Ref. mode:     {self.ref_point_mode}",
            f"  Ref. point:    {self._format_ref_point(self._session_ref_point)}",
            f"  Mejor HV:      {self._study.best_value:.6f}",
            "",
            "  --- Mejores hiperparámetros ---",
        ]
        for k, v in best.params.items():
            lines.append(f"    {k:<30} = {v}")
        lines.append("=" * 60)
        return "\n".join(lines)

    @property
    def study(self):
        """Estudio Optuna completo (para análisis avanzado).

        Permite acceder a todos los trials, plots de Optuna, etc.
        Ejemplo::

            import optuna.visualization as vis
            fig = vis.plot_optimization_history(tuner.study)
            fig.show()
        """
        if self._study is None:
            raise RuntimeError("El tuning no se ha ejecutado todavía.")
        return self._study

    # -----------------------------------------------------------------------
    #  Métodos internos
    # -----------------------------------------------------------------------

    def _suggest_config(self, trial) -> OptimizerConfig:
        """Genera una ``OptimizerConfig`` a partir de las sugerencias de Optuna.

        Optuna suele el valor de cada hiperparámetro dentro del rango
        definido en ``self.space`` usando su modelo TPE interno.

        Args:
            trial: Trial de Optuna.

        Returns:
            ``OptimizerConfig`` con los valores sugeridos por Optuna.
        """
        space = self.space

        pop_size = trial.suggest_int(
            "population_size",
            space.population_size_range[0],
            space.population_size_range[1],
        )
        n_gen = trial.suggest_int(
            "n_generations",
            space.n_generations_range[0],
            space.n_generations_range[1],
        )
        prob_swap = trial.suggest_categorical(
            "prob_swap_mutation",
            list(space.prob_swap_mutation_choices),
        )
        prob_replace = trial.suggest_categorical(
            "prob_replace_mutation",
            list(space.prob_replace_mutation_choices),
        )
        cx_op = trial.suggest_categorical(
            "crossover_operator",
            space.crossover_operators,
        )
        algorithm = trial.suggest_categorical(
            "algorithm",
            space.algorithms,
        )

        # Durante el tuning se usan valores reducidos de pop/gen para que
        # cada trial sea rápido. Los valores sugeridos se aplican a la
        # configuración final (best_config), no a los trials individuales.
        return OptimizerConfig(
            algorithm=algorithm,
            population_size=min(pop_size, self.DEFAULT_EVAL_POPULATION),
            n_generations=min(n_gen, self.DEFAULT_EVAL_GENERATIONS),
            objectives=self.objectives,
            optimization_level=self.space.optimization_level,
            crossover_operator=cx_op,
            prob_swap_mutation=prob_swap,
            prob_replace_mutation=prob_replace,
            verbose=False,
        )

    def _trial_to_config(self, trial) -> OptimizerConfig:
        """Construye la ``OptimizerConfig`` final (con valores completos) a partir del mejor trial.

        A diferencia de ``_suggest_config``, usa los valores *reales*
        sugeridos por Optuna sin reducción de pop/gen, ya que esta config
        se usará en producción (no en evaluación de tuning).

        Args:
            trial: Mejor trial del estudio Optuna.

        Returns:
            ``OptimizerConfig`` con los hiperparámetros óptimos completos.
        """
        p = trial.params
        return OptimizerConfig(
            algorithm=p.get("algorithm", "nsga2"),
            population_size=p["population_size"],
            n_generations=p["n_generations"],
            objectives=self.objectives,
            optimization_level=self.space.optimization_level,
            crossover_operator=p["crossover_operator"],
            prob_swap_mutation=p["prob_swap_mutation"],
            prob_replace_mutation=p["prob_replace_mutation"],
            verbose=True,
        )

    def _validate_manual_ref_point(
        self,
        ref_point: Optional[Sequence[float]],
    ) -> Optional[np.ndarray]:
        """Valida el punto de referencia manual si el modo lo requiere."""
        if self.ref_point_mode == "manual":
            if ref_point is None:
                raise ValueError("ref_point es obligatorio cuando ref_point_mode='manual'.")
            validated = np.asarray(ref_point, dtype=float)
            if validated.shape != (len(self.objectives),):
                raise ValueError("ref_point debe tener una coordenada por objetivo.")
            if not np.all(np.isfinite(validated)):
                raise ValueError("ref_point debe contener valores finitos.")
            return validated

        if ref_point is not None:
            raise ValueError(
                "ref_point no se puede proporcionar cuando ref_point_mode='calibrated'."
            )

        return None

    def _build_calibration_configs(
        self,
        seeds: Sequence[int],
    ) -> list[OptimizerConfig]:
        """Construye configuraciones conservadoras para calibrar el ref_point.

        La calibración usa un warm-up corto y determinista con anclas
        low / mid / high, cobertura corta de algoritmos configurados.
        Se itera sobre todas las seeds para producir hasta n_seeds × 3
        configuraciones únicas (con deduplicación por firma).
        """
        space = self.space
        anchor_specs = [
            (space.population_size_range[0], space.n_generations_range[0]),
            (
                (space.population_size_range[0] + space.population_size_range[1]) // 2,
                (space.n_generations_range[0] + space.n_generations_range[1]) // 2,
            ),
            (space.population_size_range[1], space.n_generations_range[1]),
        ]

        configs: list[OptimizerConfig] = []
        seen_signatures: set[tuple[Any, ...]] = set()
        if len(space.algorithms) == 1:
            anchor_params: list[tuple[str, str, float, float]] = [
                (
                    space.algorithms[0],
                    space.crossover_operators[0],
                    space.prob_swap_mutation_choices[0],
                    space.prob_replace_mutation_choices[0],
                ),
                (
                    space.algorithms[0],
                    space.crossover_operators[-1],
                    space.prob_swap_mutation_choices[len(space.prob_swap_mutation_choices) // 2],
                    space.prob_replace_mutation_choices[len(space.prob_replace_mutation_choices) // 2],
                ),
                (
                    space.algorithms[0],
                    space.crossover_operators[-1],
                    space.prob_swap_mutation_choices[-1],
                    space.prob_replace_mutation_choices[-1],
                ),
            ]
        else:
            anchor_params = [
                (
                    space.algorithms[0],
                    space.crossover_operators[0],
                    space.prob_swap_mutation_choices[0],
                    space.prob_replace_mutation_choices[0],
                ),
                (
                    space.algorithms[-1],
                    space.crossover_operators[0],
                    space.prob_swap_mutation_choices[0],
                    space.prob_replace_mutation_choices[0],
                ),
                (
                    space.algorithms[-1],
                    space.crossover_operators[-1],
                    space.prob_swap_mutation_choices[-1],
                    space.prob_replace_mutation_choices[-1],
                ),
            ]

        for seed in (seeds if seeds else [0]):
            for (population_size, n_generations), (
                algorithm,
                crossover_operator,
                prob_swap,
                prob_replace,
            ) in zip(
                anchor_specs,
                anchor_params,
            ):
                effective_population_size = min(
                    population_size,
                    self.DEFAULT_EVAL_POPULATION,
                )
                effective_n_generations = min(
                    n_generations,
                    self.DEFAULT_EVAL_GENERATIONS,
                )
                signature = (
                    algorithm,
                    crossover_operator,
                    prob_swap,
                    prob_replace,
                    effective_population_size,
                    effective_n_generations,
                    seed,
                )
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                configs.append(
                    OptimizerConfig(
                        algorithm=algorithm,
                        population_size=effective_population_size,
                        n_generations=effective_n_generations,
                        objectives=self.objectives,
                        optimization_level=space.optimization_level,
                        crossover_operator=crossover_operator,
                        prob_swap_mutation=prob_swap,
                        prob_replace_mutation=prob_replace,
                        seed=seed,
                        verbose=False,
                    )
                )

        return configs

    def _calibrate_reference_point(
        self,
        calibration_configs: Sequence[OptimizerConfig],
    ) -> np.ndarray:
        """Ejecuta una calibración conservadora y fija el ref_point de sesión.

        El margen aplicado es del 30 % (multiplicador 1.3×) para absorber la
        variabilidad estocástica entre seeds y entre configuraciones de mayor
        tamaño que las anclas del warm-up. Con ``n_seeds`` seeds y 3 anclas por
        seed, el warm-up ejecuta hasta ``n_seeds × 3`` optimizaciones.
        """
        pareto_fronts: list[np.ndarray] = []
        total_steps = len(calibration_configs)

        for current_step, calibration_config in enumerate(calibration_configs, start=1):
            result = optimize_layout(self.circuit, backend=self.backend, config=calibration_config)
            ref_point_candidate = None
            if result.pareto_fitness is None or len(result.pareto_fitness) == 0:
                self._emit_progress(
                    "calibration_progress",
                    current_step=current_step,
                    total_steps=total_steps,
                    config={
                        "algorithm": calibration_config.algorithm,
                        "population_size": calibration_config.population_size,
                        "n_generations": calibration_config.n_generations,
                        "crossover_operator": calibration_config.crossover_operator,
                        "prob_swap_mutation": calibration_config.prob_swap_mutation,
                        "prob_replace_mutation": calibration_config.prob_replace_mutation,
                        "seed": calibration_config.seed,
                    },
                    ref_point_candidate=None,
                )
                continue
            pareto_fronts.append(np.asarray(result.pareto_fitness, dtype=float))
            combined_front = np.vstack(pareto_fronts)
            ref_point_candidate = combined_front.max(axis=0) * 1.3 + 1e-6
            self._emit_progress(
                "calibration_progress",
                current_step=current_step,
                total_steps=total_steps,
                config={
                    "algorithm": calibration_config.algorithm,
                    "population_size": calibration_config.population_size,
                    "n_generations": calibration_config.n_generations,
                    "crossover_operator": calibration_config.crossover_operator,
                    "prob_swap_mutation": calibration_config.prob_swap_mutation,
                    "prob_replace_mutation": calibration_config.prob_replace_mutation,
                    "seed": calibration_config.seed,
                },
                ref_point_candidate=ref_point_candidate,
            )

        if not pareto_fronts:
            raise ValueError(
                "No se pudo calibrar un ref_point: calibration anchors sin frente de Pareto válido."
            )

        combined_front = np.vstack(pareto_fronts)
        return combined_front.max(axis=0) * 1.3 + 1e-6

    def _emit_progress(self, event: str, **payload: Any) -> None:
        """Emite eventos estructurados de progreso si hay callback.

        Contrato actual de eventos:
        - ``calibration_started``: incluye ``ref_point_mode``, ``n_seeds`` y
          ``calibration_config_count``.
        - ``calibration_progress``: incluye ``current_step``, ``total_steps``, la
          ``config`` evaluada y ``ref_point_candidate`` (o ``None`` si aun no
          existe).
        - ``calibration_completed``: incluye ``ref_point`` fijado y
          ``calibration_config_count``.
        - ``trial_completed``: incluye índices de progreso, score actual,
          ``best_score``, ``ref_point`` fijo y ``params`` del trial evaluado.
        - ``tuning_completed``: incluye ``best_score`` final y el ``ref_point``
          fijo usado en toda la sesión.
        """
        if self.progress_callback is None:
            return

        event_payload = {"event": event}
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                event_payload[key] = value.tolist()
            else:
                event_payload[key] = value
        self.progress_callback(event_payload)

    @staticmethod
    def _format_ref_point(ref_point: Optional[np.ndarray]) -> str:
        """Formatea el punto de referencia para resúmenes legibles."""
        if ref_point is None:
            return "not set"
        return str([float(value) for value in ref_point])
