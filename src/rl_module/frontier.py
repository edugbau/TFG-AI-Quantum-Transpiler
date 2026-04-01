from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Iterable, List, MutableSequence, Protocol, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode


GateTuple = Tuple[str, int, int]


@dataclass(frozen=True)
class LookaheadEntry:
    gate_name: str
    logical_q1: int
    logical_q2: int
    physical_q1: int
    physical_q2: int
    executable: bool


class FrontierProvider(Protocol):
    @property
    def remaining_gate_count(self) -> int:
        ...

    @property
    def remaining_gates(self) -> Iterable[GateTuple]:
        ...

    def get_visible_entries(
        self,
        current_layout: np.ndarray,
        lookahead_window: int,
        is_connected: Callable[[int, int], bool],
    ) -> List[LookaheadEntry]:
        ...

    def execute_ready_cascade(
        self,
        current_layout: np.ndarray,
        is_connected: Callable[[int, int], bool],
        cascade_successors: bool = True,
        executed_gates: MutableSequence[GateTuple] | None = None,
    ) -> int:
        ...


def _gate_tuple_from_name_and_qargs(gate_name: str, qargs: List[int]) -> GateTuple:
    if len(qargs) == 1:
        return (gate_name, qargs[0], qargs[0])
    if len(qargs) == 2:
        return (gate_name, qargs[0], qargs[1])
    raise ValueError("Only 1q and 2q operations are supported in frontier logic.")


class SequentialFrontier:
    def __init__(self, pending_gates: Iterable[GateTuple] | None = None):
        self.pending_gates: Deque[GateTuple] = deque(pending_gates or ())

    @property
    def remaining_gate_count(self) -> int:
        return len(self.pending_gates)

    @property
    def remaining_gates(self) -> Deque[GateTuple]:
        return self.pending_gates

    def get_visible_entries(
        self,
        current_layout: np.ndarray,
        lookahead_window: int,
        is_connected: Callable[[int, int], bool],
    ) -> List[LookaheadEntry]:
        visible_entries: List[LookaheadEntry] = []

        for gate_name, logical_q1, logical_q2 in list(self.pending_gates)[:lookahead_window]:
            physical_q1 = int(current_layout[logical_q1])
            physical_q2 = int(current_layout[logical_q2])
            executable = logical_q1 == logical_q2 or is_connected(physical_q1, physical_q2)
            visible_entries.append(
                LookaheadEntry(
                    gate_name=gate_name,
                    logical_q1=logical_q1,
                    logical_q2=logical_q2,
                    physical_q1=physical_q1,
                    physical_q2=physical_q2,
                    executable=executable,
                )
            )

        return visible_entries

    def execute_ready_cascade(
        self,
        current_layout: np.ndarray,
        is_connected: Callable[[int, int], bool],
        cascade_successors: bool = True,
        executed_gates: MutableSequence[GateTuple] | None = None,
    ) -> int:
        executed_count = 0

        while self.pending_gates:
            gate = self.pending_gates[0]
            _, logical_q1, logical_q2 = gate
            physical_q1 = int(current_layout[logical_q1])
            physical_q2 = int(current_layout[logical_q2])

            if logical_q1 != logical_q2 and not is_connected(physical_q1, physical_q2):
                break

            self.pending_gates.popleft()
            if executed_gates is not None:
                executed_gates.append(gate)
            executed_count += 1

        return executed_count


class DagFrontier:
    def __init__(self, dag) -> None:
        self._dag = dag

    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit) -> "DagFrontier":
        supported_circuit = QuantumCircuit(
            circuit.num_qubits,
            circuit.num_clbits,
            name=circuit.name,
        )

        for instruction in circuit.data:
            qargs = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
            if len(qargs) not in {1, 2}:
                continue

            supported_circuit.append(
                instruction.operation,
                [supported_circuit.qubits[index] for index in qargs],
                [
                    supported_circuit.clbits[circuit.find_bit(clbit).index]
                    for clbit in instruction.clbits
                ],
            )

        return cls(circuit_to_dag(supported_circuit))

    @property
    def remaining_gate_count(self) -> int:
        return len(list(self._dag.op_nodes()))

    @property
    def remaining_gates(self) -> List[GateTuple]:
        return [self._node_to_gate_tuple(node) for node in self._ordered_op_nodes()]

    def get_visible_entries(
        self,
        current_layout: np.ndarray,
        lookahead_window: int,
        is_connected: Callable[[int, int], bool],
    ) -> List[LookaheadEntry]:
        visible_entries: List[LookaheadEntry] = []

        for node in self._ordered_front_layer()[:lookahead_window]:
            gate_name, logical_q1, logical_q2 = self._node_to_gate_tuple(node)
            physical_q1 = int(current_layout[logical_q1])
            physical_q2 = int(current_layout[logical_q2])
            executable = logical_q1 == logical_q2 or is_connected(physical_q1, physical_q2)
            visible_entries.append(
                LookaheadEntry(
                    gate_name=gate_name,
                    logical_q1=logical_q1,
                    logical_q2=logical_q2,
                    physical_q1=physical_q1,
                    physical_q2=physical_q2,
                    executable=executable,
                )
            )

        return visible_entries

    def execute_ready_cascade(
        self,
        current_layout: np.ndarray,
        is_connected: Callable[[int, int], bool],
        cascade_successors: bool = True,
        executed_gates: MutableSequence[GateTuple] | None = None,
    ) -> int:
        executed_count = 0

        while True:
            ready_nodes = [
                node
                for node in self._ordered_front_layer()
                if self._is_node_executable(node, current_layout, is_connected)
            ]
            if not ready_nodes:
                break

            for node in ready_nodes:
                if executed_gates is not None:
                    executed_gates.append(self._node_to_gate_tuple(node))
                self._dag.remove_op_node(node)
                executed_count += 1

            if not cascade_successors:
                break

        return executed_count

    def _ordered_front_layer(self) -> List[DAGOpNode]:
        return self._sort_nodes(self._dag.front_layer())

    def _ordered_op_nodes(self) -> List[DAGOpNode]:
        return self._sort_nodes(list(self._dag.op_nodes()))

    def _sort_nodes(self, nodes: Iterable[DAGOpNode]) -> List[DAGOpNode]:
        return sorted(nodes, key=self._node_sort_key)

    def _node_sort_key(self, node: DAGOpNode) -> Tuple[int, Tuple[int, ...], str]:
        qargs = self._logical_qargs(node)
        return (min(qargs), tuple(qargs), node.name)

    def _logical_qargs(self, node: DAGOpNode) -> List[int]:
        return [self._dag.find_bit(qubit).index for qubit in node.qargs]

    def _node_to_gate_tuple(self, node: DAGOpNode) -> GateTuple:
        return _gate_tuple_from_name_and_qargs(node.name, self._logical_qargs(node))

    def _is_node_executable(
        self,
        node: DAGOpNode,
        current_layout: np.ndarray,
        is_connected: Callable[[int, int], bool],
    ) -> bool:
        _, logical_q1, logical_q2 = self._node_to_gate_tuple(node)
        if logical_q1 == logical_q2:
            return True
        physical_q1 = int(current_layout[logical_q1])
        physical_q2 = int(current_layout[logical_q2])
        return is_connected(physical_q1, physical_q2)
