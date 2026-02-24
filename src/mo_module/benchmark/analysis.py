"""
analysis.py — Análisis estadístico de resultados de benchmark
==============================================================

Funciones para analizar un ``BenchmarkResultSet`` y producir
tablas de resumen, estadísticas de estabilidad por semilla, y
métricas de calidad del frente de Pareto.

Uso típico::

    from src.mo_module.benchmark import run_benchmark, analyze_results

    results = run_benchmark(n_seeds=10)
    report  = analyze_results(results)
    print(report.to_text())

Autor: Eduardo González Bautista
Fecha: 2026-02-24
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from .runner import BenchmarkResultSet

logger = logging.getLogger(__name__)


# ===========================================================================
#  Estadísticas por objetivo
# ===========================================================================


@dataclass
class ObjectiveStats:
    """Estadísticas descriptivas de un objetivo a través de las semillas.

    Se calculan sobre el **mejor valor** alcanzado en cada semilla.

    Attributes:
        name: Nombre del objetivo.
        values: Mejores valores por semilla.
        mean: Media.
        std: Desviación estándar.
        median: Mediana.
        iqr: Rango intercuartílico (Q3 − Q1).
        min: Mínimo global.
        max: Máximo global.
        cv: Coeficiente de variación (std / mean), como porcentaje.
    """

    name: str = ""
    values: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    iqr: float = 0.0
    min: float = 0.0
    max: float = 0.0
    cv: float = 0.0  # coeficiente de variación en %

    def to_row(self) -> str:
        """Fila formateada para una tabla de texto."""
        return (
            f"  {self.name:<16} "
            f"{self.mean:>10.2f} "
            f"{self.std:>8.2f} "
            f"{self.median:>10.2f} "
            f"{self.iqr:>8.2f} "
            f"[{self.min:.2f}, {self.max:.2f}]  "
            f"CV={self.cv:.1f}%"
        )


def compute_objective_stats(
    name: str, values: list[float]
) -> ObjectiveStats:
    """Calcula estadísticas descriptivas de una lista de valores.

    Args:
        name: Nombre del objetivo.
        values: Mejores valores por semilla.

    Returns:
        ``ObjectiveStats`` con las estadísticas calculadas.
    """
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    median = float(np.median(arr))
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    cv = (std / mean * 100) if mean != 0 else 0.0

    return ObjectiveStats(
        name=name,
        values=values,
        mean=mean,
        std=std,
        median=median,
        iqr=q3 - q1,
        min=float(arr.min()),
        max=float(arr.max()),
        cv=cv,
    )


# ===========================================================================
#  Estadísticas por circuito
# ===========================================================================


@dataclass
class CircuitAnalysis:
    """Análisis completo de un circuito a lo largo de todas las semillas.

    Attributes:
        circuit_name: Identificador del circuito.
        n_seeds: Número de semillas exitosas.
        objective_stats: Estadísticas por objetivo.
        time_stats: Estadísticas de tiempo de ejecución.
        pareto_size_stats: Estadísticas del tamaño del frente de Pareto.
        seed_stability_pvalue:
            p-valor de Kruskal-Wallis entre las distribuciones de
            fitness de las distintas semillas. Si es > 0.05, las
            semillas producen distribuciones similares (estabilidad).
            ``None`` si no es aplicable (< 3 semillas).
    """

    circuit_name: str = ""
    n_seeds: int = 0
    objective_stats: list[ObjectiveStats] = field(default_factory=list)
    time_stats: Optional[ObjectiveStats] = None
    pareto_size_stats: Optional[ObjectiveStats] = None
    seed_stability_pvalue: Optional[float] = None

    def to_text(self) -> str:
        """Resumen legible del análisis de este circuito."""
        lines = [
            f"  --- {self.circuit_name} ({self.n_seeds} semillas) ---",
            "",
            f"  {'Objetivo':<16} {'Media':>10} {'Std':>8} "
            f"{'Mediana':>10} {'IQR':>8} {'Rango':<20} CV",
        ]
        for os in self.objective_stats:
            lines.append(os.to_row())

        if self.time_stats:
            lines.append("")
            lines.append(
                f"  Tiempo (s):        "
                f"media={self.time_stats.mean:.2f}, "
                f"std={self.time_stats.std:.2f}, "
                f"rango=[{self.time_stats.min:.2f}, {self.time_stats.max:.2f}]"
            )
        if self.pareto_size_stats:
            lines.append(
                f"  Tamaño Pareto:     "
                f"media={self.pareto_size_stats.mean:.1f}, "
                f"rango=[{self.pareto_size_stats.min:.0f}, "
                f"{self.pareto_size_stats.max:.0f}]"
            )
        if self.seed_stability_pvalue is not None:
            stable = "SÍ" if self.seed_stability_pvalue > 0.05 else "NO"
            lines.append(
                f"  Estabilidad seeds: p={self.seed_stability_pvalue:.4f} "
                f"→ {stable} (Kruskal-Wallis, α=0.05)"
            )
        lines.append("")
        return "\n".join(lines)


# ===========================================================================
#  Informe completo
# ===========================================================================


@dataclass
class BenchmarkReport:
    """Informe completo del benchmark.

    Attributes:
        circuit_analyses: Análisis por circuito.
        backend_name: Backend usado.
        total_elapsed_s: Tiempo total.
        algorithm: Algoritmo usado.
    """

    circuit_analyses: list[CircuitAnalysis] = field(default_factory=list)
    backend_name: str = ""
    total_elapsed_s: float = 0.0
    algorithm: str = ""

    def to_text(self) -> str:
        """Informe completo formateado en texto plano."""
        lines = [
            "=" * 72,
            "  INFORME DE BENCHMARK — OPTIMIZACIÓN MULTIOBJETIVO",
            "=" * 72,
            f"  Backend:       {self.backend_name}",
            f"  Algoritmo:     {self.algorithm}",
            f"  Tiempo total:  {self.total_elapsed_s:.1f} s",
            "",
        ]
        for ca in self.circuit_analyses:
            lines.append(ca.to_text())
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convierte el informe a diccionario (para JSON / pandas)."""
        rows = []
        for ca in self.circuit_analyses:
            row = {
                "circuit": ca.circuit_name,
                "n_seeds": ca.n_seeds,
                "backend": self.backend_name,
                "algorithm": self.algorithm,
            }
            for os in ca.objective_stats:
                row[f"{os.name}_mean"] = os.mean
                row[f"{os.name}_std"] = os.std
                row[f"{os.name}_median"] = os.median
                row[f"{os.name}_iqr"] = os.iqr
                row[f"{os.name}_min"] = os.min
                row[f"{os.name}_max"] = os.max
                row[f"{os.name}_cv"] = os.cv
            if ca.time_stats:
                row["time_mean"] = ca.time_stats.mean
                row["time_std"] = ca.time_stats.std
            if ca.pareto_size_stats:
                row["pareto_size_mean"] = ca.pareto_size_stats.mean
            if ca.seed_stability_pvalue is not None:
                row["seed_stability_p"] = ca.seed_stability_pvalue
            rows.append(row)
        return {"rows": rows}


# ===========================================================================
#  Función principal de análisis
# ===========================================================================


def analyze_results(result_set: BenchmarkResultSet) -> BenchmarkReport:
    """Analiza un ``BenchmarkResultSet`` y produce un ``BenchmarkReport``.

    Para cada circuito, calcula:
      - Estadísticas descriptivas (media, std, mediana, IQR, CV) del
        **mejor valor** de cada objetivo por semilla.
      - Estadísticas de tiempo de ejecución y tamaño del frente.
      - Test de Kruskal-Wallis para evaluar la estabilidad entre semillas.

    Args:
        result_set: Resultados de ``BenchmarkRunner.run()``.

    Returns:
        ``BenchmarkReport`` con el análisis completo.
    """
    report = BenchmarkReport(
        backend_name=result_set.backend_name,
        total_elapsed_s=result_set.total_elapsed_s,
        algorithm=result_set.config.algorithm if result_set.config else "?",
    )

    for cname in result_set.circuit_names:
        runs = result_set.runs_for_circuit(cname)
        if not runs:
            continue

        obj_names = runs[0].result.objective_names

        # --- Estadísticas por objetivo ---
        obj_stats_list = []
        for oi, oname in enumerate(obj_names):
            bests = result_set.best_per_seed(cname, oi)
            if bests:
                obj_stats_list.append(compute_objective_stats(oname, bests))

        # --- Tiempo ---
        times = result_set.elapsed_per_seed(cname)
        time_stats = compute_objective_stats("time_s", times) if times else None

        # --- Tamaño del frente ---
        psizes = [float(s) for s in result_set.pareto_sizes(cname)]
        pareto_stats = (
            compute_objective_stats("pareto_size", psizes) if psizes else None
        )

        # --- Estabilidad entre semillas (Kruskal-Wallis) ---
        stability_p = _seed_stability_test(result_set, cname)

        ca = CircuitAnalysis(
            circuit_name=cname,
            n_seeds=len(runs),
            objective_stats=obj_stats_list,
            time_stats=time_stats,
            pareto_size_stats=pareto_stats,
            seed_stability_pvalue=stability_p,
        )
        report.circuit_analyses.append(ca)

    return report


def _seed_stability_test(
    result_set: BenchmarkResultSet,
    circuit_name: str,
) -> Optional[float]:
    """Test de Kruskal-Wallis sobre el primer objetivo entre semillas.

    Comprueba si la distribución de fitness del primer objetivo es
    estadísticamente similar entre semillas.  Requiere ≥ 3 semillas
    con ≥ 2 soluciones en el frente cada una.

    Returns:
        p-valor (float) o ``None`` si no es aplicable.
    """
    runs = result_set.runs_for_circuit(circuit_name)
    groups = []
    for run in runs:
        F = run.result.pareto_fitness
        if F is not None and len(F) >= 2:
            groups.append(F[:, 0])

    if len(groups) < 3:
        return None

    try:
        _, p_value = sp_stats.kruskal(*groups)
        return float(p_value)
    except Exception:
        return None
