"""
pareto.py — Análisis del frente de Pareto
==========================================

Módulo 3 del TFG: Optimización Multiobjetivo.

Este fichero proporciona herramientas para analizar, seleccionar y
visualizar los resultados del frente de Pareto generado por el
optimizador. Incluye:

  - **Análisis del frente**: métricas de calidad del frente (hipervolumen,
    spacing, spread).
  - **Selección de soluciones**: estrategias para elegir una solución
    del frente (knee point, ponderación, compromiso).
  - **Visualización**: gráficas del frente de Pareto en 2D y 3D.

Decisiones de diseño:
  1. **Hipervolumen** — Se utiliza el indicador de hipervolumen (HV)
     de pymoo como métrica principal de calidad del frente. El HV es
     la métrica más aceptada en la literatura de optimización
     multiobjetivo (Zitzler & Thiele, 1999).

  2. **Punto de referencia** — El cálculo del HV requiere un punto
     de referencia (nadir). Se calcula automáticamente como el peor valor
     de cada objetivo en el frente + un margen.

  3. **Knee point** — Se implementa la detección del punto de rodilla
     (knee point) como estrategia de selección. El knee point es la
     solución donde el trade-off marginal entre objetivos cambia más
     bruscamente (Das, 1999).

  4. **Matplotlib** — Se usa matplotlib para visualización,
     coherente con el stack del proyecto (ver requirements.txt).

Autor: Eduardo González Bautista
Fecha: 2026-02-18
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Imports de pymoo para métricas
# ---------------------------------------------------------------------------
from pymoo.indicators.hv import HV

# ---------------------------------------------------------------------------
# Imports internos del módulo
# ---------------------------------------------------------------------------
from .optimizer import OptimizationResult

# ---------------------------------------------------------------------------
# Logger del módulo
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
#  Métricas de calidad del frente de Pareto
# ===========================================================================

@dataclass
class ParetoMetrics:
    """Métricas cuantitativas de la calidad de un frente de Pareto.

    Attributes:
        hypervolume:
            Indicador de hipervolumen (HV). Mayor es mejor.
            Mide el volumen del espacio de objetivos dominado por
            el frente de Pareto con respecto a un punto de referencia.
        n_solutions:
            Número de soluciones no-dominadas.
        spacing:
            Medida de la distribución uniforme de las soluciones.
            Menor spacing = distribución más uniforme. Se calcula como
            la desviación estándar de las distancias mínimas entre
            soluciones consecutivas.
        spread:
            Rango de cada objetivo en el frente. ndarray de forma
            ``(n_objectives,)`` con ``max - min`` de cada columna.
        ideal_point:
            Punto ideal (mejor valor de cada objetivo en el frente).
        nadir_point:
            Punto nadir (peor valor de cada objetivo en el frente).
        reference_point:
            Punto de referencia usado para el cálculo del HV.
    """

    hypervolume: float = 0.0
    n_solutions: int = 0
    spacing: float = 0.0
    spread: Optional[np.ndarray] = None
    ideal_point: Optional[np.ndarray] = None
    nadir_point: Optional[np.ndarray] = None
    reference_point: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Resumen legible de las métricas del frente."""
        lines = [
            f"  Soluciones Pareto:  {self.n_solutions}",
            f"  Hipervolumen (HV): {self.hypervolume:.6f}",
            f"  Spacing:           {self.spacing:.6f}",
        ]
        if self.ideal_point is not None:
            lines.append(f"  Punto ideal:       {self.ideal_point}")
        if self.nadir_point is not None:
            lines.append(f"  Punto nadir:       {self.nadir_point}")
        if self.spread is not None:
            lines.append(f"  Spread:            {self.spread}")
        return "\n".join(lines)


def compute_pareto_metrics(
    pareto_fitness: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
    margin: float = 0.1,
) -> ParetoMetrics:
    """Calcula métricas de calidad del frente de Pareto.

    Args:
        pareto_fitness:
            ndarray de forma ``(n_solutions, n_objectives)`` con los
            valores de fitness del frente.
        reference_point:
            Punto de referencia para el HV. Si es None, se calcula
            automáticamente como ``nadir_point * (1 + margin)``.
        margin:
            Margen para el cálculo automático del punto de referencia.

    Returns:
        ParetoMetrics con las métricas calculadas.
    """
    if pareto_fitness.ndim == 1:
        pareto_fitness = pareto_fitness.reshape(1, -1)

    n_solutions, n_objectives = pareto_fitness.shape

    # Punto ideal y nadir
    ideal = pareto_fitness.min(axis=0)
    nadir = pareto_fitness.max(axis=0)

    # Spread
    spread = nadir - ideal

    # Punto de referencia para HV
    if reference_point is None:
        # Usar nadir + margen relativo al rango
        ref = nadir + np.abs(nadir) * margin
        # Si algún valor es 0, usar un margen absoluto
        ref[ref == nadir] = nadir[ref == nadir] + margin
        reference_point = ref

    # Hipervolumen
    hv_value = 0.0
    if n_solutions > 0:
        try:
            hv_indicator = HV(ref_point=reference_point)
            hv_value = float(hv_indicator(pareto_fitness))
        except Exception as e:
            logger.warning("Error calculando hipervolumen: %s", e)

    # Spacing (distancia mínima media entre soluciones vecinas)
    spacing = 0.0
    if n_solutions > 1:
        min_distances = []
        for i in range(n_solutions):
            dists = np.linalg.norm(
                pareto_fitness[i] - pareto_fitness, axis=1
            )
            dists[i] = np.inf  # Excluir la propia solución
            min_distances.append(np.min(dists))
        spacing = float(np.std(min_distances))

    return ParetoMetrics(
        hypervolume=hv_value,
        n_solutions=n_solutions,
        spacing=spacing,
        spread=spread,
        ideal_point=ideal,
        nadir_point=nadir,
        reference_point=reference_point,
    )


# ===========================================================================
#  Estrategias de selección de solución
# ===========================================================================

def select_knee_point(pareto_fitness: np.ndarray) -> int:
    """Selecciona el knee point (punto de rodilla) del frente de Pareto.

    El knee point es la solución donde se produce el mayor cambio en
    el trade-off entre objetivos. Se calcula como el punto con la
    mayor distancia perpendicular a la línea que conecta los extremos
    del frente.

    Para más de 2 objetivos, se usa la distancia al hiperplano
    definido por los puntos extremos normalizados.

    Ref: I. Das (1999), "On characterizing the 'knee' of the Pareto
    curve based on Normal-Boundary Intersection".

    Args:
        pareto_fitness: ndarray de forma ``(n_solutions, n_objectives)``.

    Returns:
        Índice de la solución knee point.
    """
    if pareto_fitness.ndim == 1:
        return 0

    n_solutions, n_obj = pareto_fitness.shape

    if n_solutions <= 2:
        # Con 1 o 2 soluciones, devolver la más equilibrada
        if n_solutions == 1:
            return 0
        # Con 2 soluciones, devolver la de menor norma normalizada
        F_norm = _normalize_front(pareto_fitness)
        norms = np.linalg.norm(F_norm, axis=1)
        return int(np.argmin(norms))

    # Normalizar el frente
    F_norm = _normalize_front(pareto_fitness)

    # Para 2 objetivos: distancia a la línea entre extremos
    if n_obj == 2:
        return _knee_2d(F_norm)

    # Para n objetivos: distancia al hiperplano
    return _knee_nd(F_norm)


def _normalize_front(F: np.ndarray) -> np.ndarray:
    """Normaliza el frente de Pareto al rango [0, 1] por objetivo.

    Args:
        F: ndarray del frente.

    Returns:
        ndarray normalizado.
    """
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    f_range = f_max - f_min
    f_range[f_range == 0] = 1.0
    return (F - f_min) / f_range


def _knee_2d(F_norm: np.ndarray) -> int:
    """Knee point para 2 objetivos (distancia a la línea).

    Args:
        F_norm: Frente normalizado.

    Returns:
        Índice del knee point.
    """
    # Extremos del frente: mejor en obj 0 y mejor en obj 1
    idx_a = int(np.argmin(F_norm[:, 0]))
    idx_b = int(np.argmin(F_norm[:, 1]))

    A = F_norm[idx_a]
    B = F_norm[idx_b]

    # Vector de la línea AB
    AB = B - A
    AB_norm = np.linalg.norm(AB)

    if AB_norm == 0:
        return 0

    # Distancia perpendicular de cada punto a la línea AB
    # Fórmula 2D: |det([AB, AP])| / |AB| = |(AB_x * AP_y - AB_y * AP_x)| / |AB|
    AP = A - F_norm  # (n_solutions, 2)
    distances = np.abs(AB[0] * AP[:, 1] - AB[1] * AP[:, 0]) / AB_norm

    return int(np.argmax(distances))


def _knee_nd(F_norm: np.ndarray) -> int:
    """Knee point para n objetivos (distancia a la diagonal).

    Aproximación: la diagonal del hipercubo [0,1]^n va de (0,...,0)
    a (1,...,1). Se calcula la distancia de cada punto a esta diagonal.
    El knee point maximiza esta distancia.

    Args:
        F_norm: Frente normalizado.

    Returns:
        Índice del knee point.
    """
    n = F_norm.shape[1]
    # Dirección de la diagonal
    d = np.ones(n) / np.sqrt(n)

    # Proyección de cada punto sobre la diagonal
    projections = F_norm @ d
    proj_points = np.outer(projections, d)

    # Distancia perpendicular a la diagonal
    distances = np.linalg.norm(F_norm - proj_points, axis=1)

    return int(np.argmax(distances))


def select_weighted(
    pareto_fitness: np.ndarray,
    weights: Sequence[float],
) -> int:
    """Selecciona la solución con la mejor suma ponderada.

    Método de escalarización: convierte el problema MO en uno
    escalar multiplicando cada objetivo por un peso y sumando.

    Nota: los objetivos deben estar normalizados o los pesos
    deben compensar las diferencias de escala.

    Args:
        pareto_fitness: ndarray del frente de Pareto.
        weights: Pesos para cada objetivo (deben sumar ~1).

    Returns:
        Índice de la mejor solución ponderada.
    """
    w = np.array(weights, dtype=float)
    F_norm = _normalize_front(pareto_fitness)
    scores = F_norm @ w
    return int(np.argmin(scores))


def select_min_objective(
    pareto_fitness: np.ndarray,
    objective_index: int,
) -> int:
    """Selecciona la solución con el menor valor en un objetivo dado.

    Args:
        pareto_fitness: ndarray del frente de Pareto.
        objective_index: Índice del objetivo a minimizar.

    Returns:
        Índice de la mejor solución en ese objetivo.
    """
    return int(np.argmin(pareto_fitness[:, objective_index]))


def _build_candidate_entry(
    index: int,
    layout: list[int],
    distance_to_ideal: float,
    reason: str,
) -> dict:
    """Construye un payload homogéneo para candidatos de selección."""
    return {
        "index": index,
        "layout": layout,
        "distance_to_ideal": distance_to_ideal,
        "reason": reason,
    }


# ===========================================================================
#  Análisis del frente de Pareto
# ===========================================================================

def analyze_pareto_front(
    opt_result: OptimizationResult,
    reference_point: Optional[np.ndarray] = None,
) -> dict:
    """Análisis completo del frente de Pareto de un OptimizationResult.

    Calcula métricas de calidad e identifica soluciones clave
    (knee point, mejores por objetivo, compromiso).

    Args:
        opt_result: Resultado de ``optimize_layout()``.
        reference_point: Punto de referencia para HV (opcional).

    Returns:
        Diccionario con:
          - ``"metrics"``: ParetoMetrics.
          - ``"knee_point_idx"``: Índice del knee point.
          - ``"knee_point_layout"``: Layout del knee point.
          - ``"best_per_objective"``: Dict de layouts mejores por objetivo.
          - ``"compromise_layout"``: Layout de compromiso.
    """
    if opt_result.pareto_fitness is None or len(opt_result.pareto_layouts) == 0:
        return {
            "metrics": ParetoMetrics(),
            "knee_point_idx": -1,
            "knee_point_layout": [],
            "best_per_objective": {},
            "compromise_layout": [],
            "selection_candidates": {},
            "tradeoff_table": [],
        }

    F = opt_result.pareto_fitness
    F_norm = _normalize_front(F)
    distances_to_ideal = np.linalg.norm(F_norm, axis=1)

    # Métricas de calidad
    metrics = compute_pareto_metrics(F, reference_point)

    # Knee point
    knee_idx = select_knee_point(F)
    knee_layout = opt_result.pareto_layouts[knee_idx]

    # Mejores por objetivo
    best_per_obj = {}
    for i, name in enumerate(opt_result.objective_names):
        idx = select_min_objective(F, i)
        best_per_obj[name] = {
            "index": idx,
            "layout": opt_result.pareto_layouts[idx],
            "value": float(F[idx, i]),
        }

    # Compromiso
    compromise_idx = int(np.argmin(distances_to_ideal))
    compromise_layout = opt_result.pareto_layouts[compromise_idx]

    selection_candidates = {
        "compromise": _build_candidate_entry(
            index=compromise_idx,
            layout=compromise_layout,
            distance_to_ideal=float(distances_to_ideal[compromise_idx]),
            reason="Closest solution to the normalized ideal point.",
        ),
        "knee": _build_candidate_entry(
            index=knee_idx,
            layout=knee_layout,
            distance_to_ideal=float(distances_to_ideal[knee_idx]),
            reason="Knee point with the strongest marginal trade-off.",
        ),
    }

    for name, best_info in best_per_obj.items():
        selection_candidates[f"best_{name}"] = _build_candidate_entry(
            index=best_info["index"],
            layout=best_info["layout"],
            distance_to_ideal=float(distances_to_ideal[best_info["index"]]),
            reason=f"Lowest value found for objective '{name}'.",
        )

    tradeoff_table = []
    for idx, layout in enumerate(opt_result.pareto_layouts):
        tradeoff_table.append(
            {
                "index": idx,
                "layout": layout,
                "raw_objectives": {
                    name: float(F[idx, objective_idx])
                    for objective_idx, name in enumerate(opt_result.objective_names)
                },
                "normalized_objectives": {
                    name: float(F_norm[idx, objective_idx])
                    for objective_idx, name in enumerate(opt_result.objective_names)
                },
                "distance_to_ideal": float(distances_to_ideal[idx]),
            }
        )

    result = {
        "metrics": metrics,
        "knee_point_idx": knee_idx,
        "knee_point_layout": knee_layout,
        "best_per_objective": best_per_obj,
        "compromise_layout": compromise_layout,
        "selection_candidates": selection_candidates,
        "tradeoff_table": tradeoff_table,
    }

    logger.info(
        "Análisis del frente completado — HV=%.6f, knee_idx=%d",
        metrics.hypervolume,
        knee_idx,
    )

    return result


# ===========================================================================
#  Visualización del frente de Pareto
# ===========================================================================

def plot_pareto_front_2d(
    opt_result: OptimizationResult,
    objective_x: int = 0,
    objective_y: int = 1,
    highlight_knee: bool = True,
    highlight_compromise: bool = True,
    title: Optional[str] = None,
    filename: Optional[str] = None,
):
    """Genera un scatter plot 2D del frente de Pareto.

    Args:
        opt_result: Resultado de la optimización.
        objective_x: Índice del objetivo para el eje X.
        objective_y: Índice del objetivo para el eje Y.
        highlight_knee: Resaltar el knee point.
        highlight_compromise: Resaltar la solución de compromiso.
        title: Título del gráfico (auto-generado si None).
        filename: Ruta para guardar la imagen (si None, no se guarda).

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if opt_result.pareto_fitness is None or len(opt_result.pareto_fitness) == 0:
        logger.warning("No hay datos de Pareto para visualizar.")
        return None

    F = opt_result.pareto_fitness
    names = opt_result.objective_names

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter del frente de Pareto
    ax.scatter(
        F[:, objective_x],
        F[:, objective_y],
        c="steelblue",
        edgecolors="navy",
        alpha=0.7,
        s=60,
        label="Frente de Pareto",
        zorder=2,
    )

    # Knee point
    if highlight_knee and len(F) > 2:
        knee_idx = select_knee_point(F)
        ax.scatter(
            F[knee_idx, objective_x],
            F[knee_idx, objective_y],
            c="red",
            marker="*",
            s=200,
            label="Knee Point",
            zorder=3,
            edgecolors="darkred",
        )

    # Compromiso
    if highlight_compromise:
        comp_idx = int(np.argmin(np.linalg.norm(_normalize_front(F), axis=1)))
        ax.scatter(
            F[comp_idx, objective_x],
            F[comp_idx, objective_y],
            c="gold",
            marker="D",
            s=150,
            label="Compromiso",
            zorder=3,
            edgecolors="darkorange",
        )

    # Etiquetas y formato
    x_label = names[objective_x] if objective_x < len(names) else f"Obj {objective_x}"
    y_label = names[objective_y] if objective_y < len(names) else f"Obj {objective_y}"
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    if title is None:
        title = (
            f"Frente de Pareto — {opt_result.algorithm_name.upper()}\n"
            f"Circuito: {opt_result.circuit_name}, "
            f"Backend: {opt_result.backend_name}"
        )
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename is not None:
        from pathlib import Path
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        logger.info("Gráfico del frente de Pareto guardado: %s", path)

    return fig


def plot_pareto_front_3d(
    opt_result: OptimizationResult,
    objectives: tuple[int, int, int] = (0, 1, 2),
    title: Optional[str] = None,
    filename: Optional[str] = None,
):
    """Genera un scatter plot 3D del frente de Pareto.

    Requiere al menos 3 objetivos en el resultado.

    Args:
        opt_result: Resultado de la optimización.
        objectives: Tupla de 3 índices de objetivos para los ejes.
        title: Título del gráfico.
        filename: Ruta para guardar la imagen.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if (
        opt_result.pareto_fitness is None
        or len(opt_result.pareto_fitness) == 0
        or opt_result.pareto_fitness.shape[1] < 3
    ):
        logger.warning("Se necesitan al menos 3 objetivos para plot 3D.")
        return None

    F = opt_result.pareto_fitness
    names = opt_result.objective_names
    i, j, k = objectives

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        F[:, i], F[:, j], F[:, k],
        c="steelblue", edgecolors="navy", alpha=0.7, s=60,
    )

    ax.set_xlabel(names[i] if i < len(names) else f"Obj {i}")
    ax.set_ylabel(names[j] if j < len(names) else f"Obj {j}")
    ax.set_zlabel(names[k] if k < len(names) else f"Obj {k}")

    if title is None:
        title = (
            f"Frente de Pareto 3D — {opt_result.algorithm_name.upper()}"
        )
    ax.set_title(title, fontsize=13)

    plt.tight_layout()

    if filename is not None:
        from pathlib import Path
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        logger.info("Gráfico 3D del frente de Pareto guardado: %s", path)

    return fig


def plot_parallel_coordinates(
    opt_result: OptimizationResult,
    highlight_knee: bool = True,
    title: Optional[str] = None,
    filename: Optional[str] = None,
):
    """Genera un gráfico de coordenadas paralelas del frente de Pareto.

    Útil para visualizar el trade-off entre muchos objetivos
    simultáneamente. Cada línea es una solución del frente, cada
    eje vertical un objetivo.

    Args:
        opt_result: Resultado de la optimización.
        highlight_knee: Si True, resalta el knee point en rojo.
        title: Título del gráfico.
        filename: Ruta para guardar la imagen.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if opt_result.pareto_fitness is None or len(opt_result.pareto_fitness) == 0:
        logger.warning("No hay datos de Pareto para visualizar.")
        return None

    F = opt_result.pareto_fitness
    names = opt_result.objective_names
    F_norm = _normalize_front(F)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Dibujar cada solución como una línea
    x = list(range(len(names)))
    for i in range(len(F_norm)):
        ax.plot(x, F_norm[i], c="steelblue", alpha=0.3, linewidth=1)

    # Knee point
    if highlight_knee and len(F) > 2:
        knee_idx = select_knee_point(F)
        ax.plot(
            x, F_norm[knee_idx],
            c="red", linewidth=2.5, label="Knee Point",
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Valor normalizado [0, 1]", fontsize=11)

    if title is None:
        title = f"Coordenadas Paralelas — {opt_result.algorithm_name.upper()}"
    ax.set_title(title, fontsize=13)

    if highlight_knee and len(F) > 2:
        ax.legend(fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename is not None:
        from pathlib import Path
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        logger.info("Gráfico de coordenadas paralelas guardado: %s", path)

    return fig
