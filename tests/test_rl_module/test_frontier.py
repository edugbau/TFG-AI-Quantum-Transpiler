from collections import deque

import numpy as np
from qiskit import QuantumCircuit

from src.rl_module.environment import QuantumTranspilationEnv
from src.rl_module.frontier import DagFrontier, LookaheadEntry, SequentialFrontier


def test_visible_entries_project_current_layout():
    frontier = SequentialFrontier(
        [("h", 0, 0), ("cx", 0, 2), ("cx", 1, 2)]
    )
    layout = np.array([2, 0, 1], dtype=np.int32)
    coupling_set = {(0, 1), (1, 0), (1, 2), (2, 1)}

    entries = frontier.get_visible_entries(
        current_layout=layout,
        lookahead_window=2,
        is_connected=lambda a, b: (a, b) in coupling_set,
    )

    assert entries == [
        LookaheadEntry("h", 0, 0, 2, 2, True),
        LookaheadEntry("cx", 0, 2, 2, 1, True),
    ]


def test_execute_ready_cascade_stops_at_first_blocked_gate():
    frontier = SequentialFrontier(
        [("h", 0, 0), ("cx", 0, 1), ("cx", 0, 2)]
    )
    layout = np.array([0, 1, 2], dtype=np.int32)
    coupling_set = {(0, 1), (1, 0), (1, 2), (2, 1)}

    executed = frontier.execute_ready_cascade(
        current_layout=layout,
        is_connected=lambda a, b: (a, b) in coupling_set,
    )

    assert executed == 2
    assert frontier.remaining_gate_count == 1
    assert list(frontier.pending_gates) == [("cx", 0, 2)]


def test_execute_ready_cascade_optionally_appends_executed_gate_trace():
    frontier = SequentialFrontier(
        [("h", 0, 0), ("cx", 0, 1), ("cx", 0, 2)]
    )
    layout = np.array([0, 1, 2], dtype=np.int32)
    coupling_set = {(0, 1), (1, 0), (1, 2), (2, 1)}
    executed_gates = []

    executed = frontier.execute_ready_cascade(
        current_layout=layout,
        is_connected=lambda a, b: (a, b) in coupling_set,
        executed_gates=executed_gates,
    )

    assert executed == 2
    assert executed_gates == [("h", 0, 0), ("cx", 0, 1)]
    assert list(frontier.pending_gates) == [("cx", 0, 2)]


def test_pending_gates_preserves_deque_compatibility():
    frontier = SequentialFrontier(deque([("cx", 0, 2)]))

    assert isinstance(frontier.pending_gates, deque)
    assert frontier.remaining_gate_count == 1


def test_dag_frontier_exposes_parallel_front_layer_from_real_dag():
    qc = QuantumCircuit(4, name="dag_front_layer")
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)

    frontier = DagFrontier.from_circuit(qc)
    layout = np.arange(4, dtype=np.int32)
    coupling_set = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}

    entries = frontier.get_visible_entries(
        current_layout=layout,
        lookahead_window=3,
        is_connected=lambda a, b: (a, b) in coupling_set,
    )

    assert entries == [
        LookaheadEntry("cx", 0, 1, 0, 1, True),
        LookaheadEntry("cx", 2, 3, 2, 3, True),
    ]


def test_dag_frontier_execute_ready_front_layer_leaves_successor_pending():
    qc = QuantumCircuit(4, name="dag_front_layer")
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)

    frontier = DagFrontier.from_circuit(qc)
    layout = np.arange(4, dtype=np.int32)
    coupling_set = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}

    executed = frontier.execute_ready_cascade(
        current_layout=layout,
        is_connected=lambda a, b: (a, b) in coupling_set,
        cascade_successors=False,
    )

    assert executed == 2
    assert frontier.remaining_gate_count == 1
    assert list(frontier.remaining_gates) == [("cx", 1, 2)]


def test_dag_frontier_execute_ready_cascade_removes_newly_unblocked_successors():
    qc = QuantumCircuit(4, name="dag_front_layer")
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)

    frontier = DagFrontier.from_circuit(qc)
    layout = np.arange(4, dtype=np.int32)
    coupling_set = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}

    executed = frontier.execute_ready_cascade(
        current_layout=layout,
        is_connected=lambda a, b: (a, b) in coupling_set,
    )

    assert executed == 3
    assert frontier.remaining_gate_count == 0


def test_dag_frontier_execute_ready_cascade_optionally_appends_executed_gate_trace():
    qc = QuantumCircuit(4, name="dag_front_layer")
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)

    frontier = DagFrontier.from_circuit(qc)
    layout = np.arange(4, dtype=np.int32)
    coupling_set = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    executed_gates = []

    executed = frontier.execute_ready_cascade(
        current_layout=layout,
        is_connected=lambda a, b: (a, b) in coupling_set,
        executed_gates=executed_gates,
    )

    assert executed == 3
    assert executed_gates == [("cx", 0, 1), ("cx", 2, 3), ("cx", 1, 2)]
    assert frontier.remaining_gate_count == 0


def test_dag_frontier_mode_reset_executes_newly_unblocked_successors():
    qc = QuantumCircuit(4, name="dag_front_layer")
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)

    env = QuantumTranspilationEnv(
        target_circuit=qc,
        coupling_map=[(0, 1), (1, 2), (2, 3)],
        mode="routing",
        frontier_mode="dag",
        lookahead_window=3,
    )

    obs, info = env.reset(seed=42)

    assert info["total_gates"] == 3
    assert info["already_completed_at_reset"] is True
    assert list(env.remaining_gates) == []
    np.testing.assert_array_equal(obs["lookahead"], np.full(6, -1, dtype=np.int32))
