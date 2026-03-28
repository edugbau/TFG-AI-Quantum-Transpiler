"""
runner.py — Motor de ejecución de benchmarks
=============================================

Ejecuta el optimizador multiobjetivo sobre una combinación de
circuitos y semillas, recopilando resultados para su posterior
análisis estadístico.

Flujo principal::

    runner = BenchmarkRunner(
        circuits=get_default_circuits(),
        seeds=list(range(30)),
        backend_name="fake_torino",
    )
    results = runner.run()          # BenchmarkResultSet
    print(results.summary())

Decisiones de diseño:
  1. **Iteración circuitos × semillas** — Para cada par (circuito, seed)
     se ejecuta ``optimize_layout`` con una config compartida.  La
     semilla afecta tanto al muestreo inicial como al routing de Qiskit.
  2. **Configuración ligera por defecto** — population_size=30,
     n_generations=50 para análisis inicial; ajustable vía
     ``OptimizerConfig``.
  3. **Progreso por logging** — se emite un log INFO por cada ejecución
     y un resumen al final.

Autor: Eduardo González Bautista
Fecha: 2026-02-24
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.qiskit_interface.backend_info import get_backend
from ..optimizer import OptimizerConfig, OptimizationResult, optimize_layout
from .circuits import BenchmarkCircuit, get_default_circuits

logger = logging.getLogger(__name__)


# ===========================================================================
#  Resultado individual (1 circuito × 1 semilla)
# ===========================================================================


@dataclass
class BenchmarkRun:
    """Resultado de una sola ejecución del optimizador.

    Attributes:
        circuit_name: Identificador del circuito.
        seed: Semilla usada.
        result: ``OptimizationResult`` devuelto por ``optimize_layout``.
        error: Mensaje de error si la ejecución falló (``None`` si ok).
    """

    circuit_name: str
    seed: int
    result: Optional[OptimizationResult] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.error is None


# ===========================================================================
#  Conjunto completo de resultados
# ===========================================================================


@dataclass
class BenchmarkResultSet:
    """Conjunto de resultados de un benchmark completo.

    Agrupa todos los ``BenchmarkRun`` y proporciona filtros útiles.

    Attributes:
        runs: Lista de todas las ejecuciones.
        backend_name: Backend usado.
        config: Configuración del optimizador (sin la semilla, que varía).
        total_elapsed_s: Tiempo total de ejecución en segundos.
    """

    runs: list[BenchmarkRun] = field(default_factory=list)
    backend_name: str = ""
    config: Optional[OptimizerConfig] = None
    total_elapsed_s: float = 0.0

    # --- Filtros ---

    def runs_for_circuit(self, circuit_name: str) -> list[BenchmarkRun]:
        """Devuelve las ejecuciones de un circuito concreto."""
        return [r for r in self.runs if r.circuit_name == circuit_name and r.ok]

    @property
    def circuit_names(self) -> list[str]:
        """Lista (ordenada, sin duplicados) de circuitos ejecutados."""
        seen: dict[str, None] = {}
        for r in self.runs:
            seen.setdefault(r.circuit_name, None)
        return list(seen)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.runs if r.ok)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.runs if not r.ok)

    # --- Extracción de datos numéricos ---

    def fitness_matrix(self, circuit_name: str) -> Optional[np.ndarray]:
        """Devuelve la matriz de fitness del frente de Pareto por semilla.

        Retorna un array de forma ``(n_seeds, n_objectives)`` con la
        **mediana** de cada objetivo en el frente de Pareto de cada
        ejecución.  Si no hay ejecuciones para el circuito, devuelve
        ``None``.
        """
        runs = self.runs_for_circuit(circuit_name)
        if not runs:
            return None
        rows = []
        for run in runs:
            F = run.result.pareto_fitness
            if F is not None and len(F) > 0:
                rows.append(np.median(F, axis=0))
        if not rows:
            return None
        return np.array(rows)

    def best_per_seed(
        self, circuit_name: str, objective_index: int = 0
    ) -> list[float]:
        """Mejor valor de un objetivo por semilla (para un circuito).

        Args:
            circuit_name: Circuito.
            objective_index: Índice del objetivo.

        Returns:
            Lista de mejores valores (uno por semilla exitosa).
        """
        values = []
        for run in self.runs_for_circuit(circuit_name):
            F = run.result.pareto_fitness
            if F is not None and len(F) > 0:
                values.append(float(F[:, objective_index].min()))
        return values

    def elapsed_per_seed(self, circuit_name: str) -> list[float]:
        """Tiempo de ejecución por semilla (para un circuito)."""
        return [
            r.result.elapsed_time_s
            for r in self.runs_for_circuit(circuit_name)
        ]

    def pareto_sizes(self, circuit_name: str) -> list[int]:
        """Número de soluciones del frente de Pareto por semilla."""
        return [
            r.result.n_pareto_solutions
            for r in self.runs_for_circuit(circuit_name)
        ]

    # --- Resumen ---

    def summary(self) -> str:
        """Resumen legible del benchmark completo."""
        lines = [
            "=" * 72,
            "  RESUMEN DE BENCHMARK",
            "=" * 72,
            f"  Backend:            {self.backend_name}",
            f"  Ejecuciones:        {self.n_ok} ok / {self.n_failed} fallidas",
            f"  Tiempo total:       {self.total_elapsed_s:.1f} s",
        ]
        if self.config:
            lines.append(f"  Algoritmo:          {self.config.algorithm}")
            lines.append(f"  Población:          {self.config.population_size}")
            lines.append(f"  Generaciones:       {self.config.n_generations}")
            lines.append(f"  Objetivos:          {self.config.objectives}")
        lines.append("")

        for cname in self.circuit_names:
            runs = self.runs_for_circuit(cname)
            if not runs:
                lines.append(f"  {cname}: sin resultados válidos")
                continue

            times = self.elapsed_per_seed(cname)
            psizes = self.pareto_sizes(cname)

            lines.append(f"  --- {cname} ({len(runs)} semillas) ---")
            lines.append(
                f"    Tiempo medio:       {np.mean(times):.2f} s "
                f"(± {np.std(times):.2f})"
            )
            lines.append(
                f"    Pareto (tamaño):    media={np.mean(psizes):.1f}, "
                f"rango=[{min(psizes)}, {max(psizes)}]"
            )

            # Resumen por objetivo
            obj_names = runs[0].result.objective_names
            for oi, oname in enumerate(obj_names):
                bests = self.best_per_seed(cname, oi)
                if bests:
                    lines.append(
                        f"    {oname} (mejor):    "
                        f"media={np.mean(bests):.2f}, "
                        f"std={np.std(bests):.2f}, "
                        f"rango=[{min(bests):.2f}, {max(bests):.2f}]"
                    )
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)


# ===========================================================================
#  Motor de benchmark
# ===========================================================================


@dataclass
class BenchmarkRunner:
    """Motor de ejecución de benchmarks.

    Itera sobre circuitos × semillas ejecutando ``optimize_layout``
    y recopila los resultados en un ``BenchmarkResultSet``.

    Attributes:
        circuits:
            Circuitos a evaluar. Por defecto la suite estándar.
        seeds:
            Lista de semillas. 30 para análisis inicial, 10 para
            exploración rápida.
        backend_name:
            Backend Fake a usar (``"fake_torino"`` por defecto).
        config:
            ``OptimizerConfig`` base. La semilla se sobreescribe
            en cada ejecución.
        backend:
            Instancia del backend (se crea automáticamente si es
            ``None``).

    Ejemplo::

        runner = BenchmarkRunner(seeds=list(range(10)))
        results = runner.run()
        print(results.summary())
    """

    circuits: list[BenchmarkCircuit] = field(default_factory=get_default_circuits)
    seeds: list[int] = field(default_factory=lambda: list(range(30)))
    backend_name: str = "fake_torino"
    config: Optional[OptimizerConfig] = None
    backend: object = None

    def __post_init__(self):
        if self.config is None:
            self.config = OptimizerConfig(
                algorithm="nsga2",
                population_size=30,
                n_generations=50,
                objectives=["depth", "cnot_count"],
                verbose=False,
            )

    def run(self) -> BenchmarkResultSet:
        """Ejecuta el benchmark completo.

        Returns:
            ``BenchmarkResultSet`` con todos los resultados.
        """
        if self.backend is None:
            self.backend = get_backend(self.backend_name)

        total_runs = len(self.circuits) * len(self.seeds)
        logger.info(
            "Iniciando benchmark: %d circuitos × %d semillas = %d ejecuciones "
            "en backend '%s'",
            len(self.circuits),
            len(self.seeds),
            total_runs,
            self.backend_name,
        )

        result_set = BenchmarkResultSet(
            backend_name=self.backend_name,
            config=self.config,
        )

        t_start = time.perf_counter()
        run_idx = 0

        for bc in self.circuits:
            circuit = bc.create()
            for seed in self.seeds:
                run_idx += 1
                run_config = OptimizerConfig(
                    algorithm=self.config.algorithm,
                    population_size=self.config.population_size,
                    n_generations=self.config.n_generations,
                    objectives=list(self.config.objectives),
                    optimization_level=self.config.optimization_level,
                    crossover_operator=self.config.crossover_operator,
                    prob_crossover=self.config.prob_crossover,
                    prob_swap_mutation=self.config.prob_swap_mutation,
                    prob_replace_mutation=self.config.prob_replace_mutation,
                    seed=seed,
                    verbose=False,
                )

                logger.info(
                    "[%d/%d] %s  seed=%d",
                    run_idx, total_runs, bc.name, seed,
                )

                try:
                    opt_result = optimize_layout(
                        circuit=circuit,
                        backend=self.backend,
                        config=run_config,
                    )
                    result_set.runs.append(
                        BenchmarkRun(
                            circuit_name=bc.name,
                            seed=seed,
                            result=opt_result,
                        )
                    )
                except Exception as exc:
                    logger.error(
                        "[%d/%d] %s seed=%d FALLÓ: %s",
                        run_idx, total_runs, bc.name, seed, exc,
                    )
                    result_set.runs.append(
                        BenchmarkRun(
                            circuit_name=bc.name,
                            seed=seed,
                            error=str(exc),
                        )
                    )

        result_set.total_elapsed_s = time.perf_counter() - t_start
        logger.info(
            "Benchmark finalizado: %d ok, %d fallidas, %.1f s",
            result_set.n_ok,
            result_set.n_failed,
            result_set.total_elapsed_s,
        )

        return result_set
