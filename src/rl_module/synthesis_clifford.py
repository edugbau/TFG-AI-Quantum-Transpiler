from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford

from .synthesis_primitives import SynthesisPrimitive, primitive_to_circuit


def identity_clifford(num_qubits: int) -> Clifford:
    return Clifford(QuantumCircuit(num_qubits))


def _validate_layout(
    layout: list[int] | np.ndarray,
    num_logical_qubits: int,
    num_physical_qubits: int,
) -> None:
    if len(layout) != num_logical_qubits:
        raise ValueError(
            f"Layout length must match target_circuit.num_qubits ({num_logical_qubits})"
        )

    physical_indices: list[int] = []
    for index in layout:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("Layout entries must be integer physical qubit indices")
        physical_indices.append(int(index))

    if len(set(physical_indices)) != len(physical_indices):
        raise ValueError("Layout contains duplicate physical qubits")

    for physical_index in physical_indices:
        if not 0 <= physical_index < num_physical_qubits:
            raise ValueError(
                f"Layout physical qubit {physical_index} is out of range for {num_physical_qubits} qubits"
            )


def remap_logical_circuit_to_physical(
    target_circuit: QuantumCircuit,
    layout: list[int] | np.ndarray,
    num_physical_qubits: int,
) -> QuantumCircuit:
    _validate_layout(layout, target_circuit.num_qubits, num_physical_qubits)
    physical_circuit = QuantumCircuit(num_physical_qubits, target_circuit.num_clbits)

    for instruction in target_circuit.data:
        logical_qargs = [target_circuit.find_bit(qubit).index for qubit in instruction.qubits]
        mapped_qargs = [physical_circuit.qubits[int(layout[index])] for index in logical_qargs]
        mapped_cargs = [
            physical_circuit.clbits[target_circuit.find_bit(clbit).index]
            for clbit in instruction.clbits
        ]
        physical_circuit.append(instruction.operation, mapped_qargs, mapped_cargs)

    return physical_circuit


def clifford_to_observation_arrays(clifford: Clifford) -> tuple[np.ndarray, np.ndarray]:
    symplectic = clifford.symplectic_matrix.astype(np.int8).reshape(-1).astype(np.int32)
    phase = clifford.phase.astype(np.int8).astype(np.int32)
    return symplectic, phase


def clifford_distance_from_identity(clifford: Clifford) -> int:
    identity = identity_clifford(clifford.num_qubits)
    symplectic_delta = np.bitwise_xor(
        clifford.symplectic_matrix.astype(np.int8),
        identity.symplectic_matrix.astype(np.int8),
    )
    phase_delta = np.bitwise_xor(
        clifford.phase.astype(np.int8),
        identity.phase.astype(np.int8),
    )
    return int(symplectic_delta.sum() + phase_delta.sum())


@dataclass
class CliffordSynthesisState:
    target: Clifford
    current: Clifford

    @classmethod
    def from_target_circuit(
        cls,
        target_circuit: QuantumCircuit,
        layout: list[int] | np.ndarray,
        num_physical_qubits: int,
    ) -> "CliffordSynthesisState":
        physical_target = remap_logical_circuit_to_physical(
            target_circuit=target_circuit,
            layout=layout,
            num_physical_qubits=num_physical_qubits,
        )
        try:
            target = Clifford.from_circuit(physical_target)
        except QiskitError as exc:
            raise ValueError(
                "mode='synthesis' v1 solo soporta circuitos Clifford; el target no puede convertirse a Clifford."
            ) from exc
        return cls(
            target=target,
            current=identity_clifford(num_physical_qubits),
        )

    def residual(self) -> Clifford:
        return self.current.adjoint().compose(self.target)

    def residual_distance(self) -> int:
        return clifford_distance_from_identity(self.residual())

    def is_complete(self) -> bool:
        return self.residual_distance() == 0

    def apply_primitive(self, primitive: SynthesisPrimitive) -> None:
        primitive_clifford = Clifford.from_circuit(
            primitive_to_circuit(primitive, self.current.num_qubits)
        )
        self.current = self.current.compose(primitive_clifford)
