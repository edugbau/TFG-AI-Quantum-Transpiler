"""
transpiler.py — Transpilación estándar de Qiskit (baseline)
============================================================

Módulo 1 del TFG: Interfaz con Qiskit.

Este fichero implementa la transpilación estándar de Qiskit como
**línea base (baseline)** y como helper de evaluación local de layouts suministrados por el llamador
bajo las mismas restricciones de backend y transpilación.
No implementa la integración MO -> RL, que se delega al módulo de
orquestación en ``src/integration/``.

Funcionalidades principales:
  - Transpilación estándar a los 4 niveles de optimización de Qiskit (0–3).
  - Transpilación con layout inicial personalizado (para recibir layouts
    suministrados por el llamador).
  - Comparación pre/post transpilación con métricas detalladas.
  - Transpilación batch de múltiples circuitos.
  - Generación de resultados de benchmark tabulados.

Decisiones de diseño:
  1. **Solo transpilación, no ejecución** — Según la política del
     proyecto, NO se ejecutan circuitos ni se envían a backends reales.
     Solo se aplica ``qiskit.transpile()`` y se analizan las métricas
     del resultado.

  2. **qiskit.transpile()** — Se usa la función de alto nivel
     ``qiskit.transpile()`` que internamente ejecuta el StagedPassManager
     de Qiskit. Los niveles 0–3 activan pases de optimización
     progresivamente más agresivos:
       - Nivel 0: solo mapeo básico (sin optimización).
       - Nivel 1: mapeo + optimización ligera (cancelación de CNOT inversos).
       - Nivel 2: + Layout heurístico (SABRE) + síntesis básica.
       - Nivel 3: + Layout denso + resíntesis + todas las optimizaciones.

  3. **TranspilationResult dataclass** — Los resultados se devuelven
     como objetos tipados que incluyen tanto el circuito transpilado
     como sus métricas y las del circuito original, facilitando la
     comparación directa.

  4. **Soporte de layout inicial** — Se permite pasar un ``initial_layout``
     (lista de qubits físicos) para evaluar layouts externos con el mismo
     pipeline de transpilación. La orquestación entre módulos se delega a
     ``src/integration/``.

  5. **Seed de transpilación** — Se propaga siempre una seed para
     garantizar la reproducibilidad de los resultados (esencial para
     benchmarking científico).

Autor: Eduardo González Bautista
Fecha: 2026-02-08
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Imports de Qiskit 2.3.0
# ---------------------------------------------------------------------------
from qiskit import QuantumCircuit              # Circuito
from qiskit.transpiler import CouplingMap     # Para layouts manuales
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ---------------------------------------------------------------------------
# Imports internos del módulo
# ---------------------------------------------------------------------------
from .circuit_utils import CircuitMetrics, extract_metrics
from .backend_info import (
    extract_backend_info,
    get_backend,
)

# ---------------------------------------------------------------------------
# Logger del módulo
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
#  Constantes
# ===========================================================================

# Niveles de optimización disponibles en Qiskit.
# Se documentan para referencia rápida.
OPTIMIZATION_LEVELS: dict[int, str] = {
    0: "Sin optimización — solo mapeo trivial y descomposición a basis gates",
    1: "Optimización ligera — cancelación de puertas inversas + SABRE layout",
    2: "Optimización media — resíntesis de bloques 2Q + heurísticas avanzadas",
    3: "Optimización máxima — resíntesis completa + layout denso + todas las técnicas",
}

# Seed por defecto para reproducibilidad.
DEFAULT_SEED = 42

NAMED_BASELINES: dict[str, dict[str, object]] = {
    "qiskit_level_0": {"kind": "standard", "optimization_level": 0},
    "qiskit_level_1": {"kind": "standard", "optimization_level": 1},
    "qiskit_level_2": {"kind": "standard", "optimization_level": 2},
    "qiskit_level_3": {"kind": "standard", "optimization_level": 3},
    "custom_layout_level_1": {"kind": "custom_layout", "optimization_level": 1},
}


# ===========================================================================
#  Dataclass para resultados de transpilación
# ===========================================================================

@dataclass
class TranspilationResult:
    """Resultado completo de una transpilación.

    Almacena el circuito original, el transpilado, las métricas de ambos,
    y metadatos del proceso (nivel de optimización, tiempo, layout usado).

    Attributes:
        original_circuit:
            Circuito tal como se proporcionó (antes de transpilar).
        transpiled_circuit:
            Circuito después de transpilar al backend.
        original_metrics:
            Métricas del circuito original.
        transpiled_metrics:
            Métricas del circuito transpilado.
        optimization_level:
            Nivel de optimización usado (0–3).
        backend_name:
            Nombre del backend usado para la transpilación.
        initial_layout:
            Layout inicial proporcionado (None si se usó el por defecto).
        final_layout:
            Layout final aplicado por el transpilador.
        elapsed_time_s:
            Tiempo de transpilación en segundos.
        seed:
            Seed usada para reproducibilidad.
    """

    original_circuit: Optional[QuantumCircuit] = None
    transpiled_circuit: Optional[QuantumCircuit] = None
    original_metrics: Optional[CircuitMetrics] = None
    transpiled_metrics: Optional[CircuitMetrics] = None
    optimization_level: int = 1
    backend_name: str = ""
    initial_layout: Optional[list[int]] = None
    final_layout: Optional[list[int]] = None
    elapsed_time_s: float = 0.0
    seed: int = DEFAULT_SEED
    baseline_name: Optional[str] = None
    hardware_summary: Optional[dict[str, object]] = None
    _backend: object = field(default=None, repr=False, compare=False)

    # ----- Propiedades derivadas -----

    @property
    def depth_reduction(self) -> float:
        """Ratio de reducción de profundidad (valores > 0 indican mejora).

        Fórmula: (depth_original - depth_transpiled) / depth_original
        Un valor de 0.3 significa 30% de reducción de profundidad.
        Valores negativos indican que la transpilación aumentó la profundidad
        (normal cuando se añaden SWAPs y descomposición a basis gates).
        """
        if self.original_metrics and self.transpiled_metrics:
            orig = self.original_metrics.depth
            trans = self.transpiled_metrics.depth
            return (orig - trans) / orig if orig > 0 else 0.0
        return 0.0

    @property
    def two_qubit_gate_overhead(self) -> float:
        """Overhead de puertas 2Q introducido por la transpilación.

        Fórmula: transpiled_2q / original_2q
        Un valor de 2.0 significa que se duplicaron las puertas 2Q.
        """
        if self.original_metrics and self.transpiled_metrics:
            orig = self.original_metrics.two_qubit_gates
            trans = self.transpiled_metrics.two_qubit_gates
            return trans / orig if orig > 0 else float(trans)
        return 0.0

    def summary(self) -> str:
        """Devuelve un resumen legible del resultado de la transpilación."""
        om = self.original_metrics
        tm = self.transpiled_metrics

        lines = [
            f"  Backend:           {self.backend_name}",
            f"  Nivel optim.:      {self.optimization_level} "
            f"({OPTIMIZATION_LEVELS.get(self.optimization_level, '?')})",
            f"  Seed:              {self.seed}",
            f"  Tiempo transpi.:   {self.elapsed_time_s:.3f} s",
            "",
            "  --- Métricas originales ---",
        ]
        if om:
            lines += [
                f"    Profundidad:     {om.depth}",
                f"    Puertas 2Q:      {om.two_qubit_gates}",
                f"    Total puertas:   {om.total_gates}",
                f"    Qubits:          {om.num_qubits}",
            ]
        lines += ["", "  --- Métricas transpiladas ---"]
        if tm:
            lines += [
                f"    Profundidad:     {tm.depth}",
                f"    Puertas 2Q:      {tm.two_qubit_gates}",
                f"    Total puertas:   {tm.total_gates}",
                f"    Qubits:          {tm.num_qubits}",
            ]
        lines += [
            "",
            f"  Reducción depth:   {self.depth_reduction:+.1%}",
            f"  Overhead 2Q:       {self.two_qubit_gate_overhead:.2f}x",
        ]

        if self.initial_layout is not None:
            lines.append(f"  Layout inicial:    {self.initial_layout}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convierte el resultado a diccionario (para pandas/JSON)."""
        result = {
            "backend_name": self.backend_name,
            "optimization_level": self.optimization_level,
            "seed": self.seed,
            "elapsed_time_s": self.elapsed_time_s,
            "depth_reduction": self.depth_reduction,
            "two_qubit_gate_overhead": self.two_qubit_gate_overhead,
        }
        if self.baseline_name is not None:
            result["baseline_name"] = self.baseline_name
        if self.original_metrics:
            for key, val in self.original_metrics.to_dict().items():
                result[f"orig_{key}"] = val
        if self.transpiled_metrics:
            for key, val in self.transpiled_metrics.to_dict().items():
                result[f"trans_{key}"] = val
        if self.initial_layout is not None:
            result["initial_layout"] = self.initial_layout
        return result

    def to_artifact_dict(self) -> dict[str, object]:
        """Convierte el resultado a un artefacto estructurado y versionado."""
        circuit_metadata = dict(self.original_circuit.metadata or {}) if self.original_circuit else {}
        hardware_summary = self.hardware_summary
        if hardware_summary is None and self._backend is not None:
            try:
                hardware_summary = extract_backend_info(self._backend).to_summary_dict()
            except Exception as exc:
                logger.debug("No se pudo extraer el resumen hardware para el artefacto: %s", exc)
                hardware_summary = None

        artifact = {
            "artifact_version": "transpilation_result.v1",
            "baseline_name": self.baseline_name,
            "circuit": {
                "name": self.original_circuit.name if self.original_circuit else None,
                "num_qubits": self.original_circuit.num_qubits if self.original_circuit else None,
                "source_kind": circuit_metadata.get("source_kind"),
                "source_format": circuit_metadata.get("source_format"),
                "source_path": circuit_metadata.get("source_path"),
                "resolved_circuit_name": circuit_metadata.get("resolved_circuit_name"),
            },
            "backend": dict(hardware_summary or {"backend_name": self.backend_name}),
            "transpilation": {
                "optimization_level": self.optimization_level,
                "seed": self.seed,
                "elapsed_time_s": self.elapsed_time_s,
                "baseline_name": self.baseline_name,
                "initial_layout": self.initial_layout,
                "final_layout": self.final_layout,
            },
            "metrics": {
                "original": self.original_metrics.to_dict() if self.original_metrics else None,
                "transpiled": self.transpiled_metrics.to_dict() if self.transpiled_metrics else None,
            },
        }
        return artifact


# ===========================================================================
#  Funciones de transpilación
# ===========================================================================

def transpile_circuit(
    circuit: QuantumCircuit,
    backend=None,
    backend_name: str = "fake_torino",
    optimization_level: int = 1,
    initial_layout: Optional[list[int]] = None,
    seed: int = DEFAULT_SEED,
    routing_method: Optional[str] = None,
    layout_method: Optional[str] = None,
) -> TranspilationResult:
    """Transpila un circuito cuántico al conjunto de puertas de un backend.

    Esta es la función principal de transpilación. Envuelve
    ``qiskit.transpile()`` añadiendo:
      - Extracción automática de métricas pre y post transpilación.
      - Medición del tiempo de transpilación.
      - Soporte para layout inicial personalizado (suministrado por el llamador).
      - Empaquetado del resultado en ``TranspilationResult``.

    Args:
        circuit:
            Circuito a transpilar.
        backend:
            Instancia de backend. Si es None, se usa ``backend_name``
            para instanciar uno.
        backend_name:
            Nombre del backend a usar si ``backend`` es None.
            Valores válidos: ``"fake_torino"``, ``"fake_sherbrooke"``,
            ``"fake_brisbane"``.
        optimization_level:
            Nivel de optimización (0–3). Ver ``OPTIMIZATION_LEVELS``.
        initial_layout:
            Layout inicial como lista de qubits físicos.
            ``initial_layout[i]`` = qubit físico para el qubit lógico ``i``.
            Si es None, el transpilador elige automáticamente (SABRE/trivial).
        seed:
            Semilla para el transpilador (reproducibilidad).
        routing_method:
            Método de routing explícito (e.g., ``"sabre"``, ``"stochastic"``).
            Si es None, Qiskit elige según el nivel de optimización.
        layout_method:
            Método de layout explícito (e.g., ``"sabre"``, ``"trivial"``,
            ``"dense"``).
            Si es None, Qiskit elige según el nivel de optimización.

    Returns:
        TranspilationResult con el circuito transpilado y métricas.
    """
    # --- Obtener backend si no se proporcionó ---
    if backend is None:
        backend = get_backend(backend_name)

    actual_backend_name = getattr(backend, "name", backend_name)

    # --- Métricas del circuito original ---
    original_metrics = extract_metrics(circuit)

    logger.info(
        "Transpilando circuito '%s' (%d qubits, depth=%d) → backend '%s' "
        "(nivel=%d, layout=%s)",
        circuit.name or "?",
        circuit.num_qubits,
        circuit.depth(),
        actual_backend_name,
        optimization_level,
        initial_layout,
    )

    # --- Generar y ejecutar PassManager ---
    t_start = time.perf_counter()
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        initial_layout=initial_layout,
        seed_transpiler=seed,
    )
    # Personalizar métodos si se especificaron
    if routing_method is not None:
        pm.routing = routing_method
    if layout_method is not None:
        pm.layout = layout_method

    transpiled = pm.run(circuit)
    t_end = time.perf_counter()
    elapsed = t_end - t_start

    # --- Métricas del circuito transpilado ---
    transpiled_metrics = extract_metrics(transpiled)

    # --- Intentar extraer el layout final aplicado ---
    final_layout = None
    try:
        # En Qiskit 2.x, el layout final se almacena en los metadatos
        # de transpilación del circuito
        if hasattr(transpiled, "layout") and transpiled.layout is not None:
            layout_obj = transpiled.layout
            if callable(getattr(layout_obj, "final_index_layout", None)):
                final_layout = [
                    int(entry)
                    for entry in layout_obj.final_index_layout()[: circuit.num_qubits]
                ]
            # Fallback para layouts sin helper de índice final
            elif hasattr(layout_obj, "initial_layout") and layout_obj.initial_layout is not None:
                il = layout_obj.initial_layout
                final_layout = [
                    int(il[circuit.qubits[i]])
                    for i in range(circuit.num_qubits)
                ]
    except Exception as e:
        logger.debug("No se pudo extraer el layout final: %s", e)

    # --- Construir resultado ---
    result = TranspilationResult(
        original_circuit=circuit,
        transpiled_circuit=transpiled,
        original_metrics=original_metrics,
        transpiled_metrics=transpiled_metrics,
        optimization_level=optimization_level,
        backend_name=actual_backend_name,
        initial_layout=initial_layout,
        final_layout=final_layout,
        elapsed_time_s=elapsed,
        seed=seed,
        _backend=backend,
    )

    logger.info(
        "Transpilación completada en %.3f s — "
        "depth: %d→%d, 2Q gates: %d→%d",
        elapsed,
        original_metrics.depth,
        transpiled_metrics.depth,
        original_metrics.two_qubit_gates,
        transpiled_metrics.two_qubit_gates,
    )

    return result


def transpile_all_levels(
    circuit: QuantumCircuit,
    backend=None,
    backend_name: str = "fake_torino",
    initial_layout: Optional[list[int]] = None,
    seed: int = DEFAULT_SEED,
) -> dict[int, TranspilationResult]:
    """Transpila un circuito a los 4 niveles de optimización (0–3).

    Permite comparar rápidamente el efecto de cada nivel de optimización
    sobre las métricas del circuito. Es la función principal para
    establecer la baseline.

    Args:
        circuit:
            Circuito a transpilar.
        backend:
            Backend a usar. Si es None, se instancia con ``backend_name``.
        backend_name:
            Nombre del backend (usado si ``backend`` es None).
        initial_layout:
            Layout inicial (opcional). Se usa el mismo en todos los niveles
            para una comparación justa.
        seed:
            Semilla para reproducibilidad.

    Returns:
        Diccionario ``{nivel: TranspilationResult}``.
    """
    # Instanciar backend una sola vez para reusar
    if backend is None:
        backend = get_backend(backend_name)

    results: dict[int, TranspilationResult] = {}

    for level in range(4):
        logger.info("--- Transpilando nivel %d/3 ---", level)
        results[level] = transpile_circuit(
            circuit=circuit,
            backend=backend,
            optimization_level=level,
            initial_layout=initial_layout,
            seed=seed,
        )

    return results


def transpile_batch(
    circuits: dict[str, QuantumCircuit],
    backend=None,
    backend_name: str = "fake_torino",
    optimization_level: int = 1,
    seed: int = DEFAULT_SEED,
) -> dict[str, TranspilationResult]:
    """Transpila un lote de circuitos al mismo backend y nivel.

    Útil para ejecutar suites de benchmark completas.

    Args:
        circuits:
            Diccionario ``{nombre: QuantumCircuit}``. Se puede generar
            con ``circuit_utils.circuits_from_library()``.
        backend:
            Backend a usar.
        backend_name:
            Nombre del backend (usado si ``backend`` es None).
        optimization_level:
            Nivel de optimización.
        seed:
            Semilla para reproducibilidad.

    Returns:
        Diccionario ``{nombre: TranspilationResult}``.
    """
    if backend is None:
        backend = get_backend(backend_name)

    results: dict[str, TranspilationResult] = {}

    for name, circuit in circuits.items():
        logger.info("Transpilando circuito '%s'...", name)
        results[name] = transpile_circuit(
            circuit=circuit,
            backend=backend,
            optimization_level=optimization_level,
            seed=seed,
        )

    logger.info("Batch completado: %d circuitos transpilados.", len(results))
    return results


# ===========================================================================
#  Comparación y análisis
# ===========================================================================

def compare_transpilation_results(
    results: dict[Union[int, str], TranspilationResult],
) -> list[dict]:
    """Genera una tabla comparativa de múltiples transpilaciones.

    Acepta resultados indexados por nivel de optimización (int) o por
    nombre de circuito (str). Devuelve una lista de diccionarios que
    se puede convertir directamente a un DataFrame de pandas.

    Args:
        results: Diccionario de resultados de transpilación.

    Returns:
        Lista de diccionarios con las métricas de cada resultado,
        listos para ``pandas.DataFrame(lista)``.

    Example:
        >>> results = transpile_all_levels(circuit, backend)
        >>> import pandas as pd
        >>> df = pd.DataFrame(compare_transpilation_results(results))
        >>> print(df)
    """
    rows = []

    for key, result in results.items():
        row = {"label": str(key)}
        row.update(result.to_dict())
        rows.append(row)

    return rows


def print_transpilation_comparison(
    results: dict[Union[int, str], TranspilationResult],
) -> None:
    """Imprime una comparación formateada de transpilaciones en consola.

    Muestra una tabla con profundidad, puertas 2Q, total de puertas
    y tiempo para cada resultado.

    Args:
        results: Diccionario de resultados de transpilación.
    """
    print(f"\n{'=' * 80}")
    print(f"  COMPARACIÓN DE TRANSPILACIONES")
    print(f"{'=' * 80}")
    print(
        f"  {'Label':<20} {'Depth':>8} {'2Q Gates':>10} {'Total':>8} "
        f"{'Time (s)':>10} {'Δ Depth':>10}"
    )
    print(f"  {'-' * 70}")

    for key, result in results.items():
        tm = result.transpiled_metrics
        if tm is None:
            continue

        print(
            f"  {str(key):<20} {tm.depth:>8} {tm.two_qubit_gates:>10} "
            f"{tm.total_gates:>8} {result.elapsed_time_s:>10.3f} "
            f"{result.depth_reduction:>+10.1%}"
        )

    print(f"{'=' * 80}\n")


def transpile_with_custom_layout(
    circuit: QuantumCircuit,
    layout: list[int],
    backend=None,
    backend_name: str = "fake_torino",
    optimization_level: int = 1,
    seed: int = DEFAULT_SEED,
) -> TranspilationResult:
    """Transpila un circuito con un layout inicial específico.

    Esta función es un helper para la evaluación local de layouts
    suministrados por el llamador bajo las restricciones del backend y
    de la transpilación configurada. No implementa la integración MO -> RL;
    la coordinación entre módulos se delega a ``src/integration/``.

    Decisión: el ``initial_layout`` suministrado por el llamador se pasa
    como entrada al pipeline normal de transpilación y routing de Qiskit,
    sin forzar ``layout_method="trivial"``.
    El contrato lógico -> físico de ``initial_layout`` se preserva como
    entrada de la transpilación para evaluar el layout externo recibido.

    Args:
        circuit: Circuito a transpilar.
        layout: Lista de qubits físicos donde ``layout[i]`` indica el qubit
            físico asignado al qubit lógico ``i``.
        backend: Backend a usar.
        backend_name: Nombre del backend.
        optimization_level: Nivel de optimización.
        seed: Semilla.

    Returns:
        TranspilationResult con las métricas resultantes.
    """
    if backend is None:
        backend = get_backend(backend_name)

    if len(layout) != circuit.num_qubits:
        raise ValueError(
            "Custom layout must have the same length as the circuit qubit count"
        )

    if len(set(layout)) != len(layout):
        raise ValueError("Custom layout contains duplicate physical qubits")

    num_backend_qubits = getattr(backend, "num_qubits", None)
    if num_backend_qubits is not None and any(q < 0 or q >= num_backend_qubits for q in layout):
        raise ValueError("Custom layout contains physical qubits outside the backend range")

    return transpile_circuit(
        circuit=circuit,
        backend=backend,
        backend_name=backend_name,
        optimization_level=optimization_level,
        initial_layout=layout,
        seed=seed,
        # No forzamos layout_method para permitir que Qiskit
        # aplique routing (SABRE) sobre el layout dado
    )


def transpile_post_routing(
    routed_circuit: QuantumCircuit,
    *,
    backend=None,
    backend_name: str = "fake_torino",
    optimization_level: int = 1,
    seed: int = DEFAULT_SEED,
    reference_circuit: Optional[QuantumCircuit] = None,
    initial_layout: Optional[list[int]] = None,
    final_layout: Optional[list[int]] = None,
) -> TranspilationResult:
    """Ejecuta solo translation + optimization sobre un circuito ya ruteado.

    Se usa cuando el routing lo ha materializado previamente otro módulo
    (por ejemplo RL) y Qiskit solo debe adaptar el circuito al backend y
    aplicar sus optimizaciones posteriores.
    """
    if backend is None:
        backend = get_backend(backend_name)

    actual_backend_name = getattr(backend, "name", backend_name)
    original_circuit = reference_circuit if reference_circuit is not None else routed_circuit
    original_metrics = extract_metrics(original_circuit)

    t_start = time.perf_counter()
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        seed_transpiler=seed,
    )
    pm.layout = None
    pm.routing = None
    transpiled = pm.run(routed_circuit)
    t_end = time.perf_counter()

    transpiled_metrics = extract_metrics(transpiled)
    return TranspilationResult(
        original_circuit=original_circuit,
        transpiled_circuit=transpiled,
        original_metrics=original_metrics,
        transpiled_metrics=transpiled_metrics,
        optimization_level=optimization_level,
        backend_name=actual_backend_name,
        initial_layout=initial_layout,
        final_layout=final_layout,
        elapsed_time_s=t_end - t_start,
        seed=seed,
        _backend=backend,
    )


# ===========================================================================
#  Pipeline de baseline completo
# ===========================================================================

def run_baseline(
    circuit: QuantumCircuit,
    backend_names: Optional[list[str]] = None,
    optimization_levels: Optional[list[int]] = None,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Ejecuta un benchmark baseline completo.

    Transpila el circuito dado en múltiples combinaciones de backend
    y nivel de optimización, recopilando todas las métricas.

    Decisión: el resultado es una lista de diccionarios «flat» que
    se puede convertir directamente a un DataFrame de pandas para
    análisis estadístico con el módulo de integración.

    Args:
        circuit:
            Circuito a analizar.
        backend_names:
            Lista de nombres de backends. Por defecto todos los
            disponibles (Torino, Sherbrooke, Brisbane).
        optimization_levels:
            Lista de niveles de optimización. Por defecto [0, 1, 2, 3].
        seed:
            Semilla para reproducibilidad.

    Returns:
        Lista de diccionarios con métricas de cada combinación
        (backend × nivel), listos para ``pd.DataFrame(lista)``.

    Example:
        >>> import pandas as pd
        >>> from src.qiskit_interface.circuit_utils import create_ghz_circuit
        >>> ghz = create_ghz_circuit(5)
        >>> rows = run_baseline(ghz)
        >>> df = pd.DataFrame(rows)
        >>> print(df[["backend_name", "optimization_level", "trans_depth", "trans_two_qubit_gates"]])
    """
    if backend_names is None:
        backend_names = ["fake_torino", "fake_sherbrooke", "fake_brisbane"]
    if optimization_levels is None:
        optimization_levels = [0, 1, 2, 3]

    all_rows: list[dict] = []

    for bname in backend_names:
        logger.info("=== Baseline en backend '%s' ===", bname)
        backend = get_backend(bname)

        for level in optimization_levels:
            result = transpile_circuit(
                circuit=circuit,
                backend=backend,
                optimization_level=level,
                seed=seed,
            )
            row = {"circuit_name": circuit.name or "unnamed"}
            row.update(result.to_dict())
            all_rows.append(row)

    logger.info(
        "Baseline completado: %d combinaciones (backends=%d × niveles=%d)",
        len(all_rows),
        len(backend_names),
        len(optimization_levels),
    )

    return all_rows


def list_available_baselines() -> list[str]:
    """Devuelve el catálogo de baselines nombrados soportados."""
    return list(NAMED_BASELINES)


def run_named_baseline(
    baseline_name: str,
    circuit: QuantumCircuit,
    backend_names: Optional[list[str]] = None,
    layout: Optional[list[int]] = None,
    seed: int = DEFAULT_SEED,
    include_artifact: bool = False,
) -> list[dict]:
    """Ejecuta un baseline nombrado sobre uno o varios backends."""
    if baseline_name not in NAMED_BASELINES:
        available = ", ".join(NAMED_BASELINES)
        raise ValueError(
            f"Baseline '{baseline_name}' no reconocido. Baselines disponibles: {available}"
        )

    if backend_names is None:
        backend_names = ["fake_torino", "fake_sherbrooke", "fake_brisbane"]

    baseline_config = NAMED_BASELINES[baseline_name]
    optimization_level = int(baseline_config["optimization_level"])
    rows: list[dict] = []

    for backend_name in backend_names:
        backend = get_backend(backend_name)
        if baseline_config["kind"] == "custom_layout":
            if layout is None:
                raise ValueError(f"Baseline '{baseline_name}' requires a layout")
            result = transpile_with_custom_layout(
                circuit=circuit,
                layout=layout,
                backend=backend,
                optimization_level=optimization_level,
                seed=seed,
            )
        else:
            result = transpile_circuit(
                circuit=circuit,
                backend=backend,
                optimization_level=optimization_level,
                seed=seed,
            )
        result.baseline_name = baseline_name

        row = {"circuit_name": circuit.name or "unnamed"}
        row.update(result.to_dict())
        if include_artifact:
            rows.append((row, result.to_artifact_dict()))
        else:
            rows.append(row)

    return rows
