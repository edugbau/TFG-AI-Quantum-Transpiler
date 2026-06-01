import numpy as np
import pytest
from qiskit import QuantumCircuit

from src.rl_module.environment import QuantumTranspilationEnv
from src.rl_module.routing_mask import RoutingMaskConfig


def test_reset_depth_clock_preserves_parallel_gate_layers() -> None:
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    env = QuantumTranspilationEnv(
        target_circuit=circuit,
        coupling_map=[(0, 1), (2, 3)],
        mode="routing",
    )

    _, info = env.reset(seed=42)

    assert info["estimated_routing_depth"] == 1.0


def test_swap_and_unlocked_gate_add_critical_depth_delta() -> None:
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    env = QuantumTranspilationEnv(
        target_circuit=circuit,
        coupling_map=[(0, 1), (1, 2)],
        mode="routing",
    )
    env.reset(seed=42, options={"initial_layout": [0, 2]})

    _, reward, terminated, _, info = env.step(env.strategy.edges.index((0, 1)))

    assert terminated is True
    assert info["gates_executed"] == 1
    assert info["routing_depth_delta"] == 4.0
    assert info["estimated_routing_depth"] == 4.0
    assert reward == pytest.approx(59.1)


def test_v4_decay_penalizes_recently_reused_qubits_in_sabre_score() -> None:
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    env = QuantumTranspilationEnv(
        target_circuit=circuit,
        coupling_map=[(0, 1), (1, 2)],
        mode="routing",
        mask_semantics="frontier_restricted_edges.v4",
    )
    env.reset(seed=42, options={"initial_layout": [0, 2]})
    edge = (0, 1)
    candidate_layout = env._simulate_swap_layout(edge)

    base_score = env._score_sabre_candidate(candidate_layout, swap_edge=edge, strategy=env.strategy)
    env._sabre_qubit_decay[0] = 1.5
    decayed_score = env._score_sabre_candidate(candidate_layout, swap_edge=edge, strategy=env.strategy)

    assert decayed_score == pytest.approx(base_score * 1.5)


def test_v4_decay_resets_after_configured_swap_interval() -> None:
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    env = QuantumTranspilationEnv(
        target_circuit=circuit,
        coupling_map=[(0, 1), (1, 2)],
        mode="routing",
        mask_semantics="frontier_restricted_edges.v4",
        routing_mask_config=RoutingMaskConfig(sabre_decay_reset_interval=5),
    )
    env.reset(seed=42, options={"initial_layout": [0, 2]})

    for _ in range(4):
        env._record_sabre_swap(0, 1)
    assert np.max(env._sabre_qubit_decay) > 1.0

    env._record_sabre_swap(0, 1)

    np.testing.assert_array_equal(env._sabre_qubit_decay, np.ones(3, dtype=np.float32))
    assert env._sabre_swaps_since_decay_reset == 0


def test_v3_checkpoint_semantics_do_not_apply_decay() -> None:
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    env = QuantumTranspilationEnv(
        target_circuit=circuit,
        coupling_map=[(0, 1), (1, 2)],
        mode="routing",
        mask_semantics="frontier_restricted_edges.v3",
    )
    env.reset(seed=42, options={"initial_layout": [0, 2]})

    env._record_sabre_swap(0, 1)

    np.testing.assert_array_equal(env._sabre_qubit_decay, np.ones(3, dtype=np.float32))
