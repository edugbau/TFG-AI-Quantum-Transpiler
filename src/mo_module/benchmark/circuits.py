"""
circuits.py — Suite de circuitos de benchmark
==============================================

Define los circuitos de prueba para evaluar el algoritmo de
optimización multiobjetivo. Cada circuito tiene características
distintas (profundidad, densidad de puertas 2Q, estructura de
entrelazamiento) para que el benchmark cubra casos variados.

Circuitos incluidos:
  - **GHZ** (5 qubits): estructura lineal de CNOTs, O(n) depth.
    Muy sensible al layout porque el routing domina.
  - **QFT** (4 qubits): estructura densa O(n²) en puertas 2Q.
    Buen test de compromiso profundidad/CNOTs.
  - **Random** (4 qubits, depth 10): estructura impredecible.
    Prueba la robustez del optimizador ante circuitos genéricos.
  - **Clifford** (4 qubits): relevante para el módulo RL del TFG.
    Circuitos con solo H, S, CX.

Autor: Eduardo González Bautista
Fecha: 2026-02-24
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from qiskit import QuantumCircuit

from src.qiskit_interface.circuit_utils import (
    create_ghz_circuit,
    create_qft_circuit,
    create_random_circuit,
    create_clifford_circuit,
)

logger = logging.getLogger(__name__)


# ===========================================================================
#  Descriptor de circuito de benchmark
# ===========================================================================


@dataclass(frozen=True)
class BenchmarkCircuit:
    """Descriptor inmutable de un circuito de benchmark.

    Attributes:
        name:
            Identificador corto (usado como clave en resultados).
        description:
            Descripción legible del circuito y por qué es relevante.
        factory:
            Callable sin argumentos que genera el ``QuantumCircuit``.
            Se invoca cada vez que se necesita una instancia fresca.
        num_qubits:
            Número de qubits lógicos del circuito.
        tags:
            Etiquetas para filtrar circuitos (p.ej. ``"lineal"``,
            ``"denso"``, ``"aleatorio"``).
    """

    name: str
    description: str
    factory: Callable[[], QuantumCircuit]
    num_qubits: int
    tags: tuple[str, ...] = ()

    def create(self) -> QuantumCircuit:
        """Crea una instancia fresca del circuito."""
        return self.factory()


# ===========================================================================
#  Suite por defecto
# ===========================================================================


def _ghz5() -> QuantumCircuit:
    return create_ghz_circuit(5)


def _qft4() -> QuantumCircuit:
    return create_qft_circuit(4)


def _random4_d10() -> QuantumCircuit:
    return create_random_circuit(4, depth=10, seed=0)


def _clifford4() -> QuantumCircuit:
    return create_clifford_circuit(4, seed=0)


DEFAULT_BENCHMARK_CIRCUITS: list[BenchmarkCircuit] = [
    BenchmarkCircuit(
        name="ghz_5q",
        description=(
            "GHZ de 5 qubits — cadena lineal de CNOTs. "
            "Profundidad O(n), muy sensible al routing."
        ),
        factory=_ghz5,
        num_qubits=5,
        tags=("lineal", "entrelazado"),
    ),
    BenchmarkCircuit(
        name="qft_4q",
        description=(
            "QFT de 4 qubits — O(n²) puertas 2Q controladas. "
            "Buen test de compromiso profundidad/CNOTs."
        ),
        factory=_qft4,
        num_qubits=4,
        tags=("denso", "fourier"),
    ),
    BenchmarkCircuit(
        name="random_4q_d10",
        description=(
            "Circuito aleatorio de 4 qubits y profundidad 10. "
            "Evalúa robustez ante estructuras impredecibles."
        ),
        factory=_random4_d10,
        num_qubits=4,
        tags=("aleatorio",),
    ),
    BenchmarkCircuit(
        name="clifford_4q",
        description=(
            "Clifford aleatorio de 4 qubits (H, S, CX). "
            "Relevante para el módulo RL del TFG."
        ),
        factory=_clifford4,
        num_qubits=4,
        tags=("clifford", "rl"),
    ),
]


def get_default_circuits() -> list[BenchmarkCircuit]:
    """Devuelve una copia de la suite de circuitos por defecto."""
    return list(DEFAULT_BENCHMARK_CIRCUITS)


def get_circuits_by_tag(tag: str) -> list[BenchmarkCircuit]:
    """Filtra la suite por defecto por etiqueta.

    Args:
        tag: Etiqueta a buscar (p.ej. ``"denso"``, ``"clifford"``).

    Returns:
        Lista de ``BenchmarkCircuit`` que contienen esa etiqueta.
    """
    return [bc for bc in DEFAULT_BENCHMARK_CIRCUITS if tag in bc.tags]


def make_custom_circuit(
    name: str,
    circuit: QuantumCircuit,
    description: str = "",
    tags: Sequence[str] = (),
) -> BenchmarkCircuit:
    """Crea un ``BenchmarkCircuit`` a partir de un circuito ya construido.

    Útil para incluir circuitos propios en una sesión de benchmark.

    Args:
        name: Identificador.
        circuit: Circuito existente.
        description: Descripción opcional.
        tags: Etiquetas opcionales.

    Returns:
        BenchmarkCircuit que devuelve una copia del circuito dado.
    """
    return BenchmarkCircuit(
        name=name,
        description=description or f"Circuito custom «{name}»",
        factory=circuit.copy,
        num_qubits=circuit.num_qubits,
        tags=tuple(tags),
    )
