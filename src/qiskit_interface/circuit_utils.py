"""
circuit_utils.py — Utilidades para circuitos cuánticos
======================================================

Módulo 1 del TFG: Interfaz con Qiskit.

Este fichero centraliza todas las operaciones relacionadas con la carga,
creación, conversión y extracción de métricas de circuitos cuánticos
(QuantumCircuit de Qiskit 2.x).

Funcionalidades principales:
  - Carga de circuitos desde cadenas o ficheros QASM 2.0 / 3.0.
  - Exportación de circuitos a formato QASM 2.0 / 3.0.
  - Generación de circuitos de benchmark (GHZ, QFT, aleatorios, Clifford).
  - Extracción de métricas: profundidad, conteo de puertas, puertas
    no-locales (CNOT/CZ/ECR), tamaño total, ancho.
  - Conversión QuantumCircuit ↔ DAGCircuit.
  - Utilidades de visualización (texto e imagen matplotlib).

Decisiones de diseño:
  1. Se utiliza exclusivamente la API de Qiskit 2.3.0+.
     - `qiskit.qasm2` / `qiskit.qasm3` para importar/exportar QASM
       (NO `circuit.qasm()` que está deprecado).
     - `qiskit.converters` para conversión a DAG.
  2. Las métricas se devuelven como `CircuitMetrics` (dataclass) para
     facilitar la serialización y comparación en los módulos MO y de
     integración.
  3. El conteo de puertas de dos qubits es genérico: cuenta CX, CZ y ECR
     para ser compatible con distintos backends (Torino usa CZ,
     Sherbrooke/Brisbane usan ECR).
  4. Se proporciona un helper `count_two_qubit_gates()` que busca
     *cualquier* puerta que opere sobre 2+ qubits, de modo que las
     métricas sean agnósticas del backend.

Autor: Eduardo González Bautista
Fecha: 2026-02-08
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Imports de Qiskit 2.3.0 (ver agents.md §Protocolo de Versión)
# ---------------------------------------------------------------------------
from qiskit import QuantumCircuit            # Clase principal de circuito
from qiskit import qasm2, qasm3              # Módulos dedicados de QASM
from qiskit.circuit.random import random_circuit  # Circuitos aleatorios
from qiskit.synthesis.qft import synth_qft_full  # Síntesis de QFT (Qiskit 2.1+)
from qiskit.converters import (              # Conversión Circuit ↔ DAG
    circuit_to_dag,
    dag_to_circuit,
)
from qiskit.dagcircuit import DAGCircuit     # Representación DAG del circuito
from qiskit.quantum_info import (            # Operadores cuánticos
    Clifford,
    random_clifford,
)

# ---------------------------------------------------------------------------
# Logger del módulo
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
#  Dataclass para métricas de circuito
# ===========================================================================

@dataclass
class CircuitMetrics:
    """Métricas cuantitativas extraídas de un QuantumCircuit.

    Estas métricas son las que el módulo de optimización multiobjetivo (MO)
    utilizará como funciones de fitness y que el módulo de integración
    empleará para benchmarking.

    Attributes:
        depth:
            Profundidad del circuito (número de capas secuenciales de
            puertas).  Es la métrica principal de latencia: a menor
            profundidad, menos tiempo de ejecución y menos exposición
            a la decoherencia.
        num_qubits:
            Número de qubits lógicos del circuito.
        total_gates:
            Número total de operaciones (puertas + medidas + barreras, etc.).
            Equivale a ``circuit.size()``.
        two_qubit_gates:
            Número de puertas que operan sobre 2+ qubits (CX, CZ, ECR…).
            Es la métrica más correlacionada con el error del circuito,
            puesto que las puertas de dos qubits tienen tasas de error
            ~10× mayores que las de un qubit.
        nonlocal_gates:
            Puertas no-locales según ``circuit.num_nonlocal_gates()``.
            Suele coincidir con ``two_qubit_gates`` pero se mantiene
            por completitud.
        gate_counts:
            Diccionario ``{nombre_puerta: cantidad}`` devuelto por
            ``circuit.count_ops()``.
        width:
            Anchura del circuito (qubits + bits clásicos).
        num_clbits:
            Número de bits clásicos.
    """

    depth: int = 0
    num_qubits: int = 0
    total_gates: int = 0
    two_qubit_gates: int = 0
    nonlocal_gates: int = 0
    gate_counts: dict[str, int] = field(default_factory=dict)
    width: int = 0
    num_clbits: int = 0

    # ----- Utilidades de serialización -----

    def to_dict(self) -> dict:
        """Convierte las métricas a un diccionario plano (útil para pandas/JSON)."""
        return asdict(self)

    def summary(self) -> str:
        """Devuelve un resumen legible de las métricas."""
        lines = [
            f"  Qubits:            {self.num_qubits}",
            f"  Profundidad:       {self.depth}",
            f"  Puertas totales:   {self.total_gates}",
            f"  Puertas 2-qubit:   {self.two_qubit_gates}",
            f"  Puertas no-loc.:   {self.nonlocal_gates}",
            f"  Ancho (q+c):       {self.width}",
            f"  Bits clásicos:     {self.num_clbits}",
            f"  Desglose puertas:  {dict(self.gate_counts)}",
        ]
        return "\n".join(lines)


# ===========================================================================
#  Funciones de carga / exportación de circuitos
# ===========================================================================

def load_circuit_from_qasm2(source: Union[str, Path]) -> QuantumCircuit:
    """Carga un circuito desde una cadena o fichero en formato OpenQASM 2.0.

    Decisión: se distingue automáticamente entre una cadena QASM y una ruta
    de fichero.  Si ``source`` es un ``Path`` o un ``str`` que apunta a un
    archivo existente, se usa ``qasm2.load()``.  En caso contrario se
    interpreta como cadena QASM y se usa ``qasm2.loads()``.

    Args:
        source: Cadena QASM 2.0 o ruta al fichero ``.qasm``.

    Returns:
        QuantumCircuit reconstruido desde el QASM.

    Raises:
        FileNotFoundError: Si ``source`` es un Path que no existe.
        qasm2.QASM2ParseError: Si el QASM no es válido.
    """
    path = Path(source) if not isinstance(source, Path) else source

    # ¿Es un fichero que existe en disco?
    if path.suffix == ".qasm" and path.is_file():
        logger.info("Cargando circuito QASM2 desde fichero: %s", path)
        return qasm2.load(str(path))

    # En caso contrario se asume que es una cadena QASM literal
    if isinstance(source, str) and source.strip().startswith("OPENQASM"):
        logger.info("Cargando circuito QASM2 desde cadena (longitud=%d)", len(source))
        return qasm2.loads(source)

    # Si es un path que no existe, informar al usuario
    if path.suffix == ".qasm":
        raise FileNotFoundError(f"No se encontró el fichero QASM: {path}")

    # Último intento: interpretar como cadena QASM
    logger.warning("Intentando parsear source como cadena QASM2 (sin cabecera OPENQASM).")
    return qasm2.loads(str(source))


def load_circuit_from_qasm3(source: Union[str, Path]) -> QuantumCircuit:
    """Carga un circuito desde una cadena o fichero en formato OpenQASM 3.0.

    Funciona igual que :func:`load_circuit_from_qasm2` pero utilizando
    el parser de QASM 3.0 de Qiskit.

    Args:
        source: Cadena QASM 3.0 o ruta al fichero ``.qasm``.

    Returns:
        QuantumCircuit reconstruido desde el QASM 3.0.
    """
    path = Path(source) if not isinstance(source, Path) else source

    if path.suffix == ".qasm" and path.is_file():
        logger.info("Cargando circuito QASM3 desde fichero: %s", path)
        return qasm3.load(str(path))

    logger.info("Cargando circuito QASM3 desde cadena")
    return qasm3.loads(str(source))


def export_circuit_to_qasm2(circuit: QuantumCircuit) -> str:
    """Exporta un QuantumCircuit a cadena OpenQASM 2.0.

    Decisión: se utiliza ``qasm2.dumps()`` (NO ``circuit.qasm()`` que
    está deprecado en Qiskit 1.0+).

    Args:
        circuit: Circuito a exportar.

    Returns:
        Cadena con el QASM 2.0 del circuito.
    """
    return qasm2.dumps(circuit)


def export_circuit_to_qasm3(circuit: QuantumCircuit) -> str:
    """Exporta un QuantumCircuit a cadena OpenQASM 3.0.

    Args:
        circuit: Circuito a exportar.

    Returns:
        Cadena con el QASM 3.0 del circuito.
    """
    return qasm3.dumps(circuit)


def save_circuit_to_qasm2(circuit: QuantumCircuit, filepath: Union[str, Path]) -> Path:
    """Guarda un circuito en un fichero OpenQASM 2.0.

    Args:
        circuit: Circuito a guardar.
        filepath: Ruta de destino.

    Returns:
        Path al fichero creado.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    qasm2.dump(circuit, str(filepath))
    logger.info("Circuito guardado en QASM2: %s", filepath)
    return filepath


# ===========================================================================
#  Generación de circuitos de benchmark
# ===========================================================================

def create_ghz_circuit(num_qubits: int) -> QuantumCircuit:
    """Crea un circuito GHZ (Greenberger–Horne–Zeilinger) de *n* qubits.

    El estado GHZ es |000…0⟩ + |111…1⟩ (no normalizado).
    Se construye con una puerta H sobre el qubit 0 seguida de una
    cascada de CNOTs.

    Este circuito es un benchmark clásico porque:
      - Tiene profundidad lineal O(n) antes de la transpilación.
      - La profundidad depende fuertemente del layout y routing.
      - Es fácil de verificar (solo hay dos estados base con amplitud
        no nula).

    Args:
        num_qubits: Número de qubits (≥ 2).

    Returns:
        QuantumCircuit con el estado GHZ.

    Raises:
        ValueError: Si ``num_qubits < 2``.
    """
    if num_qubits < 2:
        raise ValueError(f"GHZ requiere al menos 2 qubits, recibidos: {num_qubits}")

    qc = QuantumCircuit(num_qubits, name=f"GHZ_{num_qubits}")
    # Hadamard sobre el primer qubit para crear superposición
    qc.h(0)
    # Cascada de CNOTs para entrelazar todos los qubits
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    logger.debug("Creado circuito GHZ con %d qubits (depth=%d)", num_qubits, qc.depth())
    return qc


def create_qft_circuit(num_qubits: int, inverse: bool = False) -> QuantumCircuit:
    """Crea un circuito de Transformada Cuántica de Fourier (QFT).

    La QFT es uno de los subroutines más importantes en computación
    cuántica (se usa en el algoritmo de Shor, estimación de fase, etc.).
    Es un benchmark excelente porque tiene profundidad O(n²) y un alto
    número de puertas de dos qubits controladas.

    Args:
        num_qubits: Número de qubits (≥ 1).
        inverse: Si True, crea la QFT inversa (QFT†).

    Returns:
        QuantumCircuit con la QFT.
    """
    if num_qubits < 1:
        raise ValueError(f"QFT requiere al menos 1 qubit, recibidos: {num_qubits}")

    # Usamos synth_qft_full (Qiskit 2.1+) en lugar de la clase QFT
    # que está deprecada desde Qiskit 2.1.
    # synth_qft_full devuelve directamente un QuantumCircuit con
    # puertas H, CP y SWAP descompuestas.
    qc = synth_qft_full(num_qubits, inverse=inverse)
    qc.name = f"QFT{'_inv' if inverse else ''}_{num_qubits}"

    logger.debug(
        "Creado circuito QFT%s con %d qubits (depth=%d)",
        "\u2020" if inverse else "",
        num_qubits,
        qc.depth(),
    )
    return qc


def create_random_circuit(
    num_qubits: int,
    depth: int,
    seed: Optional[int] = None,
) -> QuantumCircuit:
    """Genera un circuito cuántico aleatorio.

    Útil para benchmarks estadísticos donde se necesitan muchos circuitos
    con estructura variada.

    Args:
        num_qubits: Número de qubits.
        depth: Profundidad deseada (número de capas de puertas).
        seed: Semilla para reproducibilidad.

    Returns:
        QuantumCircuit aleatorio.
    """
    qc = random_circuit(num_qubits, depth, seed=seed)
    qc.name = f"Random_{num_qubits}q_{depth}d"
    logger.debug("Creado circuito aleatorio %s (depth=%d)", qc.name, qc.depth())
    return qc


def create_clifford_circuit(
    num_qubits: int,
    seed: Optional[int] = None,
) -> QuantumCircuit:
    """Genera un circuito Clifford aleatorio.

    Los circuitos Clifford son especialmente relevantes para este TFG
    porque el módulo de RL se centra en la *síntesis de circuitos
    Clifford*. Un Clifford aleatorio sirve para:
      - Generar datos de entrenamiento para el agente RL.
      - Evaluar la calidad de la síntesis comparando profundidad
        pre/post optimización.

    Decisión: se usa ``random_clifford()`` de ``qiskit.quantum_info``
    y ``Clifford.to_circuit()`` para obtener el QuantumCircuit equivalente.

    Args:
        num_qubits: Número de qubits (≥ 1).
        seed: Semilla para reproducibilidad (se pasa internamente a NumPy).

    Returns:
        QuantumCircuit que implementa un Clifford aleatorio.
    """
    if num_qubits < 1:
        raise ValueError(f"Clifford requiere al menos 1 qubit, recibidos: {num_qubits}")

    # Generar operador Clifford aleatorio
    cliff = random_clifford(num_qubits, seed=seed)

    # Convertir a circuito  (la descomposición usa H, S, Sdg, CX, Z)
    qc = cliff.to_circuit()
    qc.name = f"Clifford_{num_qubits}q"

    logger.debug(
        "Creado circuito Clifford aleatorio con %d qubits (depth=%d, ops=%s)",
        num_qubits,
        qc.depth(),
        dict(qc.count_ops()),
    )
    return qc


# ===========================================================================
#  Extracción de métricas
# ===========================================================================

# Conjunto de nombres de puertas de dos qubits conocidas.
# Se usa para contar puertas 2Q de forma robusta incluso si el circuito
# no ha sido transpilado y contiene puertas genéricas.
_KNOWN_TWO_QUBIT_GATES: set[str] = {
    "cx", "cnot",       # CNOT (Control-X) — alias en Qiskit
    "cz",               # Controlled-Z — puerta nativa de Torino
    "ecr",              # Echoed Cross-Resonance — puerta nativa de Sherbrooke/Brisbane
    "swap",             # SWAP
    "iswap",            # iSWAP
    "cp",               # Controlled-Phase
    "crx", "cry", "crz",  # Controlled rotations
    "ch",               # Controlled-H
    "cs", "csdg",       # Controlled-S / S†
    "csx",              # Controlled-√X
    "rxx", "ryy", "rzz", "rzx",  # Rotaciones de Ising
    "xx_plus_yy", "xx_minus_yy",
}


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """Cuenta el número de puertas que operan sobre 2+ qubits.

    Estrategia dual:
      1. Primero busca nombres conocidos en ``circuit.count_ops()``.
      2. Después recorre las instrucciones del circuito para detectar
         puertas de 2+ qubits no incluidas en la lista (e.g., puertas
         personalizadas o gates de librería menos comunes).

    Esto garantiza un conteo correcto independientemente del backend
    o del nivel de descomposición del circuito.

    Args:
        circuit: Circuito a analizar.

    Returns:
        Número total de puertas de 2+ qubits.
    """
    count = 0
    ops = circuit.count_ops()

    # Conteo rápido por nombre para puertas conocidas
    counted_names: set[str] = set()
    for gate_name, gate_count in ops.items():
        if gate_name.lower() in _KNOWN_TWO_QUBIT_GATES:
            count += gate_count
            counted_names.add(gate_name)

    # Recorrido de instrucciones para puertas desconocidas de 2+ qubits
    # (Se excluyen las ya contadas para no duplicar)
    for instruction in circuit.data:
        op_name = instruction.operation.name
        num_qubits_op = len(instruction.qubits)
        if num_qubits_op >= 2 and op_name not in counted_names:
            # Es una puerta multi-qubit no reconocida por nombre
            count += 1

    return count


def extract_metrics(circuit: QuantumCircuit) -> CircuitMetrics:
    """Extrae todas las métricas relevantes de un QuantumCircuit.

    Esta es la función central del módulo. Los demás módulos (MO,
    integración) la invocan para obtener las métricas que alimentan
    las funciones de fitness y los benchmarks.

    Args:
        circuit: Circuito del que extraer métricas.

    Returns:
        CircuitMetrics con todas las métricas calculadas.
    """
    # Convertir OrderedDict a dict normal para serialización
    gate_counts_raw = circuit.count_ops()
    gate_counts = dict(gate_counts_raw)

    metrics = CircuitMetrics(
        depth=circuit.depth(),
        num_qubits=circuit.num_qubits,
        total_gates=circuit.size(),
        two_qubit_gates=count_two_qubit_gates(circuit),
        nonlocal_gates=circuit.num_nonlocal_gates(),
        gate_counts=gate_counts,
        width=circuit.width(),
        num_clbits=circuit.num_clbits,
    )

    logger.debug("Métricas extraídas:\n%s", metrics.summary())
    return metrics


# ===========================================================================
#  Conversión Circuit ↔ DAG
# ===========================================================================

def circuit_to_dag_convert(circuit: QuantumCircuit) -> DAGCircuit:
    """Convierte un QuantumCircuit a su representación DAGCircuit.

    El DAGCircuit es la representación interna que usa el transpilador
    de Qiskit. Es útil para:
      - Análisis topológico del circuito.
      - Implementación de pases de optimización personalizados.
      - Inspección de dependencias entre puertas.

    Args:
        circuit: Circuito a convertir.

    Returns:
        DAGCircuit equivalente.
    """
    return circuit_to_dag(circuit)


def dag_to_circuit_convert(dag: DAGCircuit) -> QuantumCircuit:
    """Convierte un DAGCircuit de vuelta a QuantumCircuit.

    Args:
        dag: Grafo acíclico dirigido a convertir.

    Returns:
        QuantumCircuit equivalente.
    """
    return dag_to_circuit(dag)


# ===========================================================================
#  Utilidades de visualización
# ===========================================================================

def circuit_to_text(circuit: QuantumCircuit, fold: int = 120) -> str:
    """Devuelve una representación textual (ASCII art) del circuito.

    Útil para logging y depuración rápida sin necesidad de matplotlib.

    Args:
        circuit: Circuito a visualizar.
        fold: Número máximo de caracteres por línea antes de «doblar»
              el diagrama.

    Returns:
        Cadena con el diagrama del circuito.
    """
    return circuit.draw(output="text", fold=fold).__str__()


def circuit_to_matplotlib(
    circuit: QuantumCircuit,
    filename: Optional[Union[str, Path]] = None,
    style: Optional[dict] = None,
):
    """Genera una figura matplotlib del circuito.

    Si se proporciona ``filename``, guarda la imagen en disco.
    En caso contrario devuelve el objeto ``Figure`` para mostrarlo
    interactivamente.

    Args:
        circuit: Circuito a visualizar.
        filename: Ruta donde guardar la imagen (opcional).
        style: Diccionario de estilo para personalizar la visualización
               (ver ``QuantumCircuit.draw()`` docs).

    Returns:
        matplotlib.figure.Figure si no se guarda en disco.
    """
    draw_kwargs = {"output": "mpl"}
    if style is not None:
        draw_kwargs["style"] = style

    fig = circuit.draw(**draw_kwargs)

    if filename is not None:
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        logger.info("Diagrama del circuito guardado en: %s", filepath)

    return fig


# ===========================================================================
#  Funciones auxiliares
# ===========================================================================

def print_circuit_info(circuit: QuantumCircuit) -> None:
    """Imprime un resumen completo del circuito en consola.

    Combina el diagrama textual con las métricas extraídas.
    Pensado para uso interactivo / notebooks.

    Args:
        circuit: Circuito a inspeccionar.
    """
    name = circuit.name or "(sin nombre)"
    print(f"{'=' * 60}")
    print(f"  Circuito: {name}")
    print(f"{'=' * 60}")
    print(circuit_to_text(circuit))
    print()
    metrics = extract_metrics(circuit)
    print("Métricas:")
    print(metrics.summary())
    print(f"{'=' * 60}")


def circuits_from_library(
    num_qubits: int,
    seed: Optional[int] = 42,
) -> dict[str, QuantumCircuit]:
    """Genera un conjunto de circuitos de benchmark variados.

    Retorna un diccionario ``{nombre: QuantumCircuit}`` con circuitos
    de distintas familias. Útil para lanzar benchmarks completos de
    transpilación.

    Args:
        num_qubits: Número de qubits para todos los circuitos.
        seed: Semilla para circuitos aleatorios/Clifford.

    Returns:
        Diccionario con circuitos nombrados.
    """
    circuits: dict[str, QuantumCircuit] = {}

    # 1. GHZ — lineal, solo CNOTs
    circuits["ghz"] = create_ghz_circuit(num_qubits)

    # 2. QFT — muchas puertas controladas, profundidad cuadrática
    circuits["qft"] = create_qft_circuit(num_qubits)

    # 3. QFT inversa — misma complejidad pero puertas en orden inverso
    circuits["qft_inv"] = create_qft_circuit(num_qubits, inverse=True)

    # 4. Circuito aleatorio — estructura variada, profundidad media
    circuits["random_shallow"] = create_random_circuit(num_qubits, depth=3, seed=seed)
    circuits["random_deep"] = create_random_circuit(num_qubits, depth=10, seed=seed)

    # 5. Clifford aleatorio — relevante para el módulo RL
    circuits["clifford"] = create_clifford_circuit(num_qubits, seed=seed)

    logger.info(
        "Generados %d circuitos de benchmark con %d qubits",
        len(circuits),
        num_qubits,
    )

    return circuits
