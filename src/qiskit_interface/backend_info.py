"""
backend_info.py — Información de backends cuánticos
====================================================

Módulo 1 del TFG: Interfaz con Qiskit.

Este fichero encapsula toda la lógica para obtener y consultar información
de backends cuánticos simulados (Fake Backends). Proporciona acceso
unificado a:
  - Topología del dispositivo (coupling map / mapa de acoplamiento).
  - Puertas nativas (basis gates) y sus propiedades.
  - Propiedades de qubits (T1, T2, frecuencia).
  - Tasas de error de puertas de 1 y 2 qubits.
  - Metadatos del backend (nombre, número de qubits).

Decisiones de diseño:
  1. **Solo Fake Backends** — Según la política del proyecto (ver `qiskit-2x-compliance` skill
     §Política de Backends), NO se usan credenciales de IBM Quantum ni
     backends reales. Se emplean exclusivamente ``FakeTorino``,
     ``FakeSherbrooke`` y ``FakeBrisbane`` de
     ``qiskit_ibm_runtime.fake_provider``.

  2. **API Target (Qiskit 2.x)** — En Qiskit 2.0+ la información del
     backend se accede mediante ``backend.target`` (un objeto ``Target``)
     en lugar del antiguo ``backend.properties()`` / ``backend.configuration()``.
     Este módulo usa exclusivamente la API ``Target``.

  3. **BackendInfo dataclass** — Toda la información extraída se agrupa
     en un objeto ``BackendInfo`` para facilitar su paso a otros módulos.

  4. **Detección automática de puerta 2Q** — Cada backend puede usar
     una puerta de dos qubits distinta (CZ en Torino, ECR en
     Sherbrooke/Brisbane). Este módulo detecta automáticamente cuál es.

Autor: Eduardo González Bautista
Fecha: 2026-02-08
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Imports de Qiskit 2.3.0
# ---------------------------------------------------------------------------
from qiskit.transpiler import CouplingMap     # Mapa de acoplamiento

# ---------------------------------------------------------------------------
# Imports de qiskit-ibm-runtime para Fake Backends
# ---------------------------------------------------------------------------
from qiskit_ibm_runtime.fake_provider import (
    FakeTorino,       # 133 qubits — puerta 2Q nativa: CZ
    FakeSherbrooke,   # 127 qubits — puerta 2Q nativa: ECR
    FakeBrisbane,     # 127 qubits — puerta 2Q nativa: ECR
)

# ---------------------------------------------------------------------------
# Logger del módulo
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
#  Registro de backends disponibles
# ===========================================================================

# Diccionario que mapea nombres legibles a clases de Fake Backend.
# Se puede extender fácilmente añadiendo nuevos backends al diccionario.
AVAILABLE_BACKENDS: dict[str, type] = {
    "fake_torino": FakeTorino,
    "fake_sherbrooke": FakeSherbrooke,
    "fake_brisbane": FakeBrisbane,
}


# ===========================================================================
#  Dataclass para información del backend
# ===========================================================================

@dataclass
class BackendInfo:
    """Información estructurada de un backend cuántico.

    Agrupa toda la información relevante que los módulos MO y de
    integración necesitan para evaluar layouts y transpilaciones.

    Attributes:
        name:
            Nombre del backend (e.g., ``"fake_torino"``).
        num_qubits:
            Número total de qubits del dispositivo.
        coupling_map:
            Objeto ``CouplingMap`` de Qiskit que describe la conectividad
            entre qubits (qué pares pueden interactuar directamente).
        coupling_edges:
            Lista de aristas ``(qubit_i, qubit_j)`` del mapa de
            acoplamiento. Útil para el módulo MO que necesita las
            aristas como lista de tuplas.
        basis_gates:
            Lista de nombres de puertas nativas del backend (las únicas
            que el hardware puede ejecutar directamente).
        two_qubit_gate:
            Nombre de la puerta nativa de 2 qubits del backend
            (``"cz"`` para Torino, ``"ecr"`` para Sherbrooke/Brisbane).
        single_qubit_gates:
            Lista de puertas de 1 qubit nativas.
        qubit_t1:
            Diccionario ``{qubit_index: T1_en_segundos}``.
            T1 = tiempo de relajación (decaimiento de |1⟩ a |0⟩).
        qubit_t2:
            Diccionario ``{qubit_index: T2_en_segundos}``.
            T2 = tiempo de decoherencia (pérdida de información de fase).
        qubit_frequency:
            Diccionario ``{qubit_index: frecuencia_en_Hz}``.
        gate_errors_1q:
            Diccionario ``{qubit_index: error_rate}`` para puertas de
            1 qubit (usando la puerta SX como referencia).
        gate_errors_2q:
            Diccionario ``{(qubit_i, qubit_j): error_rate}`` para la
            puerta nativa de 2 qubits.
        dt:
            Duración del sample time del backend en segundos.
    """

    name: str = ""
    num_qubits: int = 0
    coupling_map: Optional[CouplingMap] = None
    coupling_edges: list[tuple[int, int]] = field(default_factory=list)
    basis_gates: list[str] = field(default_factory=list)
    two_qubit_gate: str = ""
    single_qubit_gates: list[str] = field(default_factory=list)
    qubit_t1: dict[int, float] = field(default_factory=dict)
    qubit_t2: dict[int, float] = field(default_factory=dict)
    qubit_frequency: dict[int, Optional[float]] = field(default_factory=dict)
    gate_errors_1q: dict[int, float] = field(default_factory=dict)
    gate_errors_2q: dict[tuple[int, int], float] = field(default_factory=dict)
    dt: Optional[float] = None

    def to_summary_dict(self) -> dict[str, object]:
        """Devuelve un resumen serializable del backend para artefactos."""
        errors_2q = list(self.gate_errors_2q.values())
        t1_vals = list(self.qubit_t1.values())
        t2_vals = list(self.qubit_t2.values())

        return {
            "backend_name": self.name,
            "num_qubits": self.num_qubits,
            "two_qubit_gate": self.two_qubit_gate,
            "basis_gates": list(self.basis_gates),
            "coupling_edges_count": len(self.coupling_edges),
            "min_error_2q": min(errors_2q) if errors_2q else 0.0,
            "avg_error_2q": sum(errors_2q) / len(errors_2q) if errors_2q else 0.0,
            "max_error_2q": max(errors_2q) if errors_2q else 0.0,
            "avg_t1": sum(t1_vals) / len(t1_vals) if t1_vals else 0.0,
            "avg_t2": sum(t2_vals) / len(t2_vals) if t2_vals else 0.0,
        }

    def summary(self) -> str:
        """Devuelve un resumen legible de la información del backend."""
        # Estadísticas de errores de puertas 2Q
        errors_2q = list(self.gate_errors_2q.values())
        avg_2q = sum(errors_2q) / len(errors_2q) if errors_2q else 0.0
        max_2q = max(errors_2q) if errors_2q else 0.0
        min_2q = min(errors_2q) if errors_2q else 0.0

        # Estadísticas de T1
        t1_vals = list(self.qubit_t1.values())
        avg_t1 = sum(t1_vals) / len(t1_vals) if t1_vals else 0.0

        lines = [
            f"  Backend:              {self.name}",
            f"  Qubits:               {self.num_qubits}",
            f"  Aristas coupling map: {len(self.coupling_edges)}",
            f"  Puerta 2Q nativa:     {self.two_qubit_gate}",
            f"  Puertas 1Q nativas:   {self.single_qubit_gates}",
            f"  Basis gates:          {self.basis_gates}",
            f"  Error 2Q (min/avg/max): {min_2q:.6f} / {avg_2q:.6f} / {max_2q:.6f}",
            f"  T1 promedio:          {avg_t1 * 1e6:.1f} µs",
            f"  dt:                   {self.dt} s",
        ]
        return "\n".join(lines)


# ===========================================================================
#  Funciones principales
# ===========================================================================

def get_backend(name: str = "fake_torino"):
    """Obtiene una instancia de un Fake Backend por nombre.

    Args:
        name: Nombre del backend. Debe ser una clave de
              ``AVAILABLE_BACKENDS``. Valores válidos:
              ``"fake_torino"``, ``"fake_sherbrooke"``, ``"fake_brisbane"``.

    Returns:
        Instancia del backend (FakeTorino, FakeSherbrooke, etc.).

    Raises:
        ValueError: Si el nombre no corresponde a un backend registrado.
    """
    name_lower = name.lower().strip()

    if name_lower not in AVAILABLE_BACKENDS:
        available = ", ".join(AVAILABLE_BACKENDS.keys())
        raise ValueError(
            f"Backend '{name}' no reconocido. "
            f"Backends disponibles: {available}"
        )

    backend = AVAILABLE_BACKENDS[name_lower]()
    logger.info("Backend instanciado: %s (%d qubits)", name_lower, backend.num_qubits)
    return backend


def list_available_backends() -> list[str]:
    """Devuelve la lista de nombres de backends disponibles.

    Returns:
        Lista de cadenas con los nombres registrados.
    """
    return list(AVAILABLE_BACKENDS.keys())


def get_coupling_map(backend) -> CouplingMap:
    """Obtiene el CouplingMap de un backend.

    El coupling map describe qué pares de qubits están conectados
    físicamente y pueden ejecutar puertas de dos qubits
    directamente (sin necesidad de SWAPs intermedios).

    Args:
        backend: Instancia de un backend de Qiskit.

    Returns:
        CouplingMap del backend.
    """
    cm = backend.coupling_map
    logger.debug(
        "CouplingMap de %s: %d qubits, %d aristas",
        backend.name if hasattr(backend, 'name') else "unknown",
        cm.size(),
        len(cm.get_edges()),
    )
    return cm


def get_coupling_edges(backend) -> list[tuple[int, int]]:
    """Obtiene las aristas del coupling map como lista de tuplas.

    Decisión: devolvemos ``list[tuple[int, int]]`` (no el objeto
    ``CouplingMap``) porque el módulo MO necesita las aristas en
    formato de lista para construir la representación del problema
    de optimización.

    Args:
        backend: Instancia de un backend de Qiskit.

    Returns:
        Lista de tuplas ``(qubit_i, qubit_j)`` que representan
        conexiones bidireccionales.
    """
    return backend.coupling_map.get_edges()


def get_basis_gates(backend) -> list[str]:
    """Obtiene la lista de puertas nativas del backend.

    Decisión: se extraen del ``Target`` del backend, que es la
    API correcta en Qiskit 2.x (en lugar del deprecated
    ``backend.configuration().basis_gates``).

    Args:
        backend: Instancia de un backend de Qiskit.

    Returns:
        Lista de nombres de las puertas nativas.
    """
    target = backend.target
    # Filtramos operaciones que no son «puertas» propiamente dichas
    # (control flow, reset, measure, delay no cuentan como basis gates
    # para la transpilación de circuitos)
    non_gate_ops = {"measure", "reset", "delay", "if_else", "for_loop", "switch_case"}
    basis = [
        name for name in target.operation_names
        if name not in non_gate_ops
    ]
    return sorted(basis)


def _detect_two_qubit_gate(backend) -> str:
    """Detecta automáticamente la puerta nativa de 2 qubits del backend.

    Decisión: en lugar de hardcodear la puerta por nombre de backend,
    se inspecciona el ``Target`` para encontrar qué operación tiene
    qargs de longitud 2. Esto hace el código robusto ante futuros
    backends con puertas diferentes.

    Args:
        backend: Instancia de un backend de Qiskit.

    Returns:
        Nombre de la puerta nativa de 2 qubits (e.g., ``"cz"`` o ``"ecr"``).

    Raises:
        RuntimeError: Si no se encuentra ninguna puerta de 2 qubits.
    """
    target = backend.target
    for op_name in target.operation_names:
        qargs = target.qargs_for_operation_name(op_name)
        if qargs is not None and any(len(q) == 2 for q in qargs):
            logger.debug("Puerta 2Q detectada: %s", op_name)
            return op_name

    raise RuntimeError(
        f"No se encontró ninguna puerta de 2 qubits en el backend "
        f"'{getattr(backend, 'name', 'unknown')}'"
    )


def get_qubit_properties(backend) -> dict[str, dict[int, Optional[float]]]:
    """Extrae propiedades de cada qubit: T1, T2 y frecuencia.

    Estas propiedades son cruciales para la optimización multiobjetivo:
      - **T1** (relajación): cuánto tiempo un qubit mantiene su estado
        excitado. Qubits con T1 bajo son más propensos a errores.
      - **T2** (decoherencia): cuánto tiempo se mantiene la coherencia
        de fase. Limita la profundidad máxima útil del circuito.
      - **Frecuencia**: frecuencia de resonancia del qubit en Hz.

    Args:
        backend: Instancia de un backend de Qiskit.

    Returns:
        Diccionario con claves ``"t1"``, ``"t2"``, ``"frequency"``,
        cada una mapeando ``{qubit_index: valor}``.
    """
    target = backend.target
    qubit_props = target.qubit_properties

    t1_map: dict[int, Optional[float]] = {}
    t2_map: dict[int, Optional[float]] = {}
    freq_map: dict[int, Optional[float]] = {}

    if qubit_props is not None:
        for i, props in enumerate(qubit_props):
            if props is not None:
                t1_map[i] = props.t1
                t2_map[i] = props.t2
                freq_map[i] = props.frequency
            else:
                t1_map[i] = None
                t2_map[i] = None
                freq_map[i] = None
    else:
        logger.warning("El backend no proporciona propiedades de qubits.")

    return {
        "t1": t1_map,
        "t2": t2_map,
        "frequency": freq_map,
    }


def get_gate_errors(backend) -> dict[str, dict]:
    """Extrae las tasas de error de todas las puertas del backend.

    Decisión: se agrupa por tipo de operación. Para puertas de 1 qubit,
    la clave es ``(qubit_index,)``; para puertas de 2 qubits,
    ``(qubit_i, qubit_j)``. Se omiten operaciones que no tienen
    propiedades de error (control flow, etc.).

    Args:
        backend: Instancia de un backend.

    Returns:
        Diccionario ``{op_name: {qargs_tuple: error_rate}}``.
        Ejemplo:
        ``{"cz": {(0, 1): 0.005, (1, 0): 0.005, ...},
          "sx": {(0,): 0.0003, (1,): 0.0004, ...}}``
    """
    target = backend.target
    errors: dict[str, dict] = {}

    for op_name in target.operation_names:
        op_props = target[op_name]
        if op_props is None:
            continue

        gate_errors: dict = {}
        for qargs, props in op_props.items():
            if props is not None and props.error is not None:
                gate_errors[qargs] = props.error

        if gate_errors:
            errors[op_name] = gate_errors

    return errors


def get_gate_durations(backend) -> dict[str, dict]:
    """Extrae las duraciones de todas las puertas del backend.

    Decisión: mismo formato que ``get_gate_errors()`` pero con
    duraciones en segundos. Útil para estimar el tiempo total de
    ejecución del circuito transpilado.

    Args:
        backend: Instancia de un backend.

    Returns:
        Diccionario ``{op_name: {qargs_tuple: duration_seconds}}``.
    """
    target = backend.target
    durations: dict[str, dict] = {}

    for op_name in target.operation_names:
        op_props = target[op_name]
        if op_props is None:
            continue

        gate_durs: dict = {}
        for qargs, props in op_props.items():
            if props is not None and props.duration is not None:
                gate_durs[qargs] = props.duration

        if gate_durs:
            durations[op_name] = gate_durs

    return durations


# ===========================================================================
#  Construcción del objeto BackendInfo completo
# ===========================================================================

def extract_backend_info(backend) -> BackendInfo:
    """Extrae toda la información relevante de un backend en un solo objeto.

    Esta es la función principal que los demás módulos deben utilizar.
    Centraliza todas las consultas al backend en un único ``BackendInfo``.

    Flujo:
      1. Obtener metadatos básicos (nombre, nº qubits).
      2. Obtener coupling map y sus aristas.
      3. Detectar puertas nativas (basis gates) y la puerta 2Q.
      4. Extraer propiedades de qubits (T1, T2, freq).
      5. Extraer errores de puertas de 1Q y 2Q.

    Args:
        backend: Instancia de un backend de Qiskit.

    Returns:
        BackendInfo con toda la información extraída.
    """
    # --- 1. Metadatos básicos ---
    backend_name = getattr(backend, "name", "unknown")
    num_qubits = backend.num_qubits

    # --- 2. Coupling map ---
    coupling_map = get_coupling_map(backend)
    coupling_edges = get_coupling_edges(backend)

    # --- 3. Puertas nativas ---
    basis_gates = get_basis_gates(backend)
    two_qubit_gate = _detect_two_qubit_gate(backend)

    # Puertas de 1 qubit = basis gates que no son la de 2Q
    single_qubit_gates = [g for g in basis_gates if g != two_qubit_gate]

    # --- 4. Propiedades de qubits ---
    qubit_props = get_qubit_properties(backend)
    qubit_t1 = {k: v for k, v in qubit_props["t1"].items() if v is not None}
    qubit_t2 = {k: v for k, v in qubit_props["t2"].items() if v is not None}
    qubit_frequency = qubit_props["frequency"]

    # --- 5. Errores de puertas ---
    all_errors = get_gate_errors(backend)

    # Errores de 1 qubit: usamos la primera puerta 1Q disponible como referencia
    gate_errors_1q: dict[int, float] = {}
    reference_1q_gate = None
    for gate in ["sx", "x", "rz", "id"]:
        if gate in single_qubit_gates and gate in all_errors:
            reference_1q_gate = gate
            break
    
    if reference_1q_gate is None and single_qubit_gates:
        # Fallback a cualquier puerta 1Q 
        for gate in single_qubit_gates:
            if gate in all_errors:
                reference_1q_gate = gate
                break

    if reference_1q_gate:
        errs_1q = all_errors.get(reference_1q_gate, {})
        for qargs, error in errs_1q.items():
            if len(qargs) == 1:
                gate_errors_1q[qargs[0]] = error

    # Errores de 2 qubits: usamos la puerta 2Q detectada
    gate_errors_2q: dict[tuple[int, int], float] = {}
    two_q_errors = all_errors.get(two_qubit_gate, {})
    for qargs, error in two_q_errors.items():
        if len(qargs) == 2:
            gate_errors_2q[qargs] = error

    # --- 6. dt (sample time) ---
    dt = getattr(backend, "dt", None)

    # --- Construir objeto ---
    info = BackendInfo(
        name=backend_name,
        num_qubits=num_qubits,
        coupling_map=coupling_map,
        coupling_edges=coupling_edges,
        basis_gates=basis_gates,
        two_qubit_gate=two_qubit_gate,
        single_qubit_gates=single_qubit_gates,
        qubit_t1=qubit_t1,
        qubit_t2=qubit_t2,
        qubit_frequency=qubit_frequency,
        gate_errors_1q=gate_errors_1q,
        gate_errors_2q=gate_errors_2q,
        dt=dt,
    )

    logger.info("Información extraída del backend '%s':\n%s", backend_name, info.summary())
    return info


# ===========================================================================
#  Utilidades adicionales
# ===========================================================================

def get_heaviest_hex_layout(backend, num_qubits: int) -> list[int]:
    """Obtiene un layout inicial fuertemente conexo.

    Estrategia de expansión (BFS):
      1. Selecciona el qubit con mayor conectividad (grado) en el chip.
      2. Añade progresivamente sus vecinos al modelo hasta alcanzar
         el número de ``num_qubits`` requeridos.
      3. Esto garantiza que el layout resultante sea conexo.

    Args:
        backend: Instancia de un backend.
        num_qubits: Número de qubits lógicos a mapear.

    Returns:
        Lista de índices de qubits físicos seleccionados.

    Raises:
        ValueError: Si ``num_qubits`` excede los qubits del backend.
    """
    if num_qubits > backend.num_qubits:
        raise ValueError(
            f"Se solicitan {num_qubits} qubits pero el backend "
            f"'{getattr(backend, 'name', '?')}' solo tiene {backend.num_qubits}"
        )

    cm = backend.coupling_map

    # Calcular el grafo de adyacencia
    adjacency: dict[int, set[int]] = {i: set() for i in range(backend.num_qubits)}
    for edge in cm.get_edges():
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])

    # El nodo semilla será el que más conexiones tenga
    seed_node = max(adjacency.keys(), key=lambda q: len(adjacency[q]))

    # Expansión BFS para garantizar conexidad
    selected = []
    visited = set()
    queue = [seed_node]

    while queue and len(selected) < num_qubits:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            selected.append(current)
            # Agregar los vecinos (dando prioridad a los de mayor grado para mantener compacidad)
            neighbors = sorted(list(adjacency[current]), key=lambda q: len(adjacency[q]), reverse=True)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)
    
    # Si por alguna razón el grafo estuviere desconectado y no tuvieramos num_qubits
    if len(selected) < num_qubits:
        remaining = [q for q in range(backend.num_qubits) if q not in visited]
        selected.extend(remaining[:num_qubits - len(selected)])

    logger.debug(
        "Layout BFS: seleccionados qubits %s (semilla: %d)",
        selected,
        seed_node,
    )
    return selected


def get_error_for_layout(
    backend_info: BackendInfo,
    layout: list[int],
) -> dict[str, float]:
    """Calcula estadísticas de error para un layout específico.

    Dado un mapeo de qubits lógicos a físicos (layout), calcula:
      - Error promedio de puertas 1Q sobre esos qubits.
      - Error promedio de puertas 2Q sobre las aristas que conectan
        esos qubits.
      - T1 y T2 promedios de los qubits seleccionados.

    Esto es útil para la función de fitness del módulo MO.

    Args:
        backend_info: Información del backend (obtenida con
                      ``extract_backend_info()``).
        layout: Lista de qubits físicos. ``layout[i]`` es el qubit
                físico al que se mapea el qubit lógico ``i``.

    Returns:
        Diccionario con las estadísticas calculadas:
        ``{"avg_error_1q", "avg_error_2q", "avg_t1", "avg_t2",
          "max_error_2q", "num_available_edges"}``.
    """
    layout_set = set(layout)

    # --- Errores 1Q ---
    errors_1q = [
        backend_info.gate_errors_1q[q]
        for q in layout
        if q in backend_info.gate_errors_1q
    ]
    avg_error_1q = sum(errors_1q) / len(errors_1q) if errors_1q else 0.0

    # --- Errores 2Q (solo aristas cuyos dos extremos están en el layout) ---
    errors_2q = [
        err
        for (q1, q2), err in backend_info.gate_errors_2q.items()
        if q1 in layout_set and q2 in layout_set
    ]
    avg_error_2q = sum(errors_2q) / len(errors_2q) if errors_2q else 0.0
    max_error_2q = max(errors_2q) if errors_2q else 0.0

    # --- T1 y T2 ---
    t1_vals = [
        backend_info.qubit_t1[q]
        for q in layout
        if q in backend_info.qubit_t1
    ]
    t2_vals = [
        backend_info.qubit_t2[q]
        for q in layout
        if q in backend_info.qubit_t2
    ]
    avg_t1 = sum(t1_vals) / len(t1_vals) if t1_vals else 0.0
    avg_t2 = sum(t2_vals) / len(t2_vals) if t2_vals else 0.0

    # Contar cuántas aristas del coupling map conectan qubits del layout
    num_edges = sum(
        1 for (q1, q2) in backend_info.coupling_edges
        if q1 in layout_set and q2 in layout_set
    )

    return {
        "avg_error_1q": avg_error_1q,
        "avg_error_2q": avg_error_2q,
        "max_error_2q": max_error_2q,
        "avg_t1": avg_t1,
        "avg_t2": avg_t2,
        "num_available_edges": num_edges,
    }


def print_backend_info(backend) -> None:
    """Imprime un resumen completo del backend en consola.

    Pensado para uso interactivo / notebooks.

    Args:
        backend: Instancia de un backend de Qiskit.
    """
    info = extract_backend_info(backend)
    name = info.name
    print(f"{'=' * 60}")
    print(f"  Backend: {name}")
    print(f"{'=' * 60}")
    print(info.summary())
    print(f"{'=' * 60}")
