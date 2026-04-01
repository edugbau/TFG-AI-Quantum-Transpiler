from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from qiskit import QuantumCircuit


SUPPORTED_TWO_Q_GATES = ("cz", "ecr", "cx")
CLIFFORD_RZ_ANGLES = (math.pi / 2, math.pi, 3 * math.pi / 2)
SUPPORTED_PRIMITIVE_GATES = ("x", "sx", "rz", *SUPPORTED_TWO_Q_GATES)


@dataclass(frozen=True)
class SynthesisPrimitive:
    gate_name: str
    physical_qargs: tuple[int, ...]
    params: tuple[float, ...] = ()
    cost: float = 1.0


def _normalized_edges(coupling_map: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted({tuple(sorted(edge)) for edge in coupling_map})


def _validate_coupling_map(
    coupling_map: Iterable[tuple[int, int]], num_physical_qubits: int
) -> None:
    for control, target in coupling_map:
        if control == target:
            raise ValueError(f"Invalid self-loop edge in coupling map: ({control}, {target})")
        if not 0 <= control < num_physical_qubits or not 0 <= target < num_physical_qubits:
            raise ValueError(
                f"Coupling map endpoint out of range for {num_physical_qubits} qubits: ({control}, {target})"
            )


def _validate_primitive(primitive: SynthesisPrimitive) -> None:
    if primitive.gate_name not in SUPPORTED_PRIMITIVE_GATES:
        raise ValueError(f"unsupported gate name in synthesis primitive: {primitive.gate_name}")

    expected_qarg_count = 2 if primitive.gate_name in SUPPORTED_TWO_Q_GATES else 1
    if len(primitive.physical_qargs) != expected_qarg_count:
        raise ValueError(
            f"Primitive gate '{primitive.gate_name}' requires exactly {expected_qarg_count} qubit argument(s)"
        )

    if primitive.gate_name in SUPPORTED_TWO_Q_GATES and primitive.physical_qargs[0] == primitive.physical_qargs[1]:
        raise ValueError(
            f"Primitive gate '{primitive.gate_name}' requires distinct qubit arguments"
        )

    expected_param_count = 1 if primitive.gate_name == "rz" else 0
    if len(primitive.params) != expected_param_count:
        raise ValueError(
            f"Primitive gate '{primitive.gate_name}' requires exactly {expected_param_count} parameter(s)"
        )

    if primitive.gate_name == "rz" and primitive.params[0] not in CLIFFORD_RZ_ANGLES:
        raise ValueError(
            "Primitive gate 'rz' only supports Clifford angles used by v1"
        )


def _validate_qargs_in_range(
    physical_qargs: Sequence[int], num_physical_qubits: int, gate_name: str
) -> None:
    for qubit in physical_qargs:
        if not 0 <= qubit < num_physical_qubits:
            raise ValueError(
                f"Primitive gate '{gate_name}' uses qubit index {qubit} out of range for {num_physical_qubits} qubits"
            )


def _detect_two_qubit_basis(basis_gates: Sequence[str]) -> str | None:
    for gate_name in SUPPORTED_TWO_Q_GATES:
        if gate_name in basis_gates:
            return gate_name
    return None


def build_clifford_primitive_catalog(
    num_physical_qubits: int,
    coupling_map: list[tuple[int, int]],
    basis_gates: Sequence[str],
) -> list[SynthesisPrimitive]:
    _validate_coupling_map(coupling_map, num_physical_qubits)
    basis_set = set(basis_gates)
    catalog: list[SynthesisPrimitive] = []

    if "x" in basis_set:
        catalog.extend(
            SynthesisPrimitive("x", (physical_qubit,), cost=1.0)
            for physical_qubit in range(num_physical_qubits)
        )

    if "sx" in basis_set:
        catalog.extend(
            SynthesisPrimitive("sx", (physical_qubit,), cost=1.0)
            for physical_qubit in range(num_physical_qubits)
        )

    if "rz" in basis_set:
        for physical_qubit in range(num_physical_qubits):
            for angle in CLIFFORD_RZ_ANGLES:
                catalog.append(
                    SynthesisPrimitive("rz", (physical_qubit,), (angle,), cost=0.0)
                )

    two_qubit_gate = _detect_two_qubit_basis(basis_gates)
    normalized_edges = _normalized_edges(coupling_map)
    if normalized_edges and two_qubit_gate is None:
        raise ValueError(
            "A supported two-qubit basis gate (cx, cz, or ecr) is required when the coupling map has edges."
        )

    if two_qubit_gate is not None:
        catalog.extend(
            SynthesisPrimitive(two_qubit_gate, edge, cost=3.0)
            for edge in normalized_edges
        )

    return catalog


def primitive_to_circuit(
    primitive: SynthesisPrimitive,
    num_physical_qubits: int,
) -> QuantumCircuit:
    _validate_primitive(primitive)
    _validate_qargs_in_range(primitive.physical_qargs, num_physical_qubits, primitive.gate_name)
    circuit = QuantumCircuit(num_physical_qubits)
    qargs = list(primitive.physical_qargs)

    if primitive.gate_name == "rz":
        circuit.rz(primitive.params[0], qargs[0])
        return circuit

    getattr(circuit, primitive.gate_name)(*qargs)
    return circuit
