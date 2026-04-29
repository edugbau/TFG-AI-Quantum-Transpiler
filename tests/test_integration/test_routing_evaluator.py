import sys
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit

from src.integration.contracts import RoutingEpisodeSummary


class StubAgent:
    def __init__(self, action: int = 0) -> None:
        self.action = action
        self.calls: list[tuple[object, bool]] = []

    def predict(self, obs, deterministic: bool = False):
        self.calls.append((obs, deterministic))
        return self.action, None


class StubMaskableAgent:
    def __init__(self, action: int = 0) -> None:
        self.action = action
        self.calls: list[tuple[object, object, bool]] = []

    def predict(self, obs, action_masks=None, deterministic: bool = False):
        self.calls.append((obs, action_masks, deterministic))
        return self.action, None


def test_create_routing_env_forwards_frontier_mode(monkeypatch) -> None:
    from src.integration.routing_evaluator import _create_routing_env

    init_calls = []
    circuit = QuantumCircuit(2)

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            init_calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "src.rl_module.environment",
        SimpleNamespace(QuantumTranspilationEnv=FakeEnv),
    )

    env = _create_routing_env(
        circuit=circuit,
        coupling_edges=[(0, 1)],
        frontier_mode="dag",
        max_steps=5,
        lookahead_window=2,
    )

    assert isinstance(env, FakeEnv)
    assert init_calls == [
        {
            "target_circuit": circuit,
            "coupling_map": [(0, 1)],
            "mode": "routing",
            "frontier_mode": "dag",
            "max_steps": 5,
            "lookahead_window": 2,
        }
    ]


def test_evaluate_routing_episode_uses_initial_layout(monkeypatch) -> None:
    from src.integration import routing_evaluator

    reset_calls: list[dict[str, object] | None] = []
    close_calls: list[str] = []
    initial_layout = [2, 0, 1]

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.current_layout = [2, 0, 1]
            self.total_swaps = 3

        def reset(self, *, seed=None, options=None):
            reset_calls.append(options)
            options["initial_layout"][0] = 99
            return {"obs": "reset"}, {"already_completed_at_reset": True}

        def close(self) -> None:
            close_calls.append("closed")

    monkeypatch.setattr(
        routing_evaluator,
        "_create_routing_env",
        lambda **kwargs: FakeEnv(**kwargs),
    )

    summary = routing_evaluator.evaluate_routing_episode(
        circuit=QuantumCircuit(3),
        coupling_edges=[(0, 1), (1, 2)],
        agent=StubAgent(),
        seed=11,
        initial_layout=initial_layout,
        frontier_mode="sequential",
        max_steps=8,
        lookahead_window=4,
    )

    assert reset_calls == [{"initial_layout": [99, 0, 1]}]
    assert reset_calls[0]["initial_layout"] is not initial_layout
    assert initial_layout == [2, 0, 1]
    assert summary.initial_layout == [2, 0, 1]
    assert summary.final_layout == [2, 0, 1]
    assert close_calls == ["closed"]


def test_evaluate_routing_episode_returns_summary_fields_for_completed_reset() -> None:
    from src.integration.routing_evaluator import evaluate_routing_episode

    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)

    summary = evaluate_routing_episode(
        circuit=circuit,
        coupling_edges=[(0, 1), (1, 0)],
        agent=StubAgent(),
        seed=7,
        initial_layout=[1, 0],
        frontier_mode="sequential",
        max_steps=5,
        lookahead_window=2,
    )

    assert isinstance(summary, RoutingEpisodeSummary)
    assert summary.initial_layout == [1, 0]
    assert summary.final_layout == [1, 0]
    assert summary.steps_executed == 0
    assert summary.total_reward == 0.0
    assert summary.completed is True
    assert summary.truncated is False
    assert isinstance(summary.total_swaps, int)
    assert summary.total_swaps == 0
    assert isinstance(summary.gates_executed_count, int)
    assert summary.gates_executed_count == 1


def test_evaluate_routing_episode_uses_deterministic_predict(monkeypatch) -> None:
    from src.integration import routing_evaluator

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.current_layout = [0, 1]
            self.total_swaps = 1
            self.step_calls = 0

        def reset(self, *, seed=None, options=None):
            return {"obs": "reset"}, {"already_completed_at_reset": False}

        def step(self, action):
            self.step_calls += 1
            return {
                "obs": "next"
            }, 1.25, True, False, {
                "gates_executed": 2,
                "action_type": "swap",
                "is_valid_action": True,
                "swap_edge": (1, 2),
            }

    monkeypatch.setattr(
        routing_evaluator,
        "_create_routing_env",
        lambda **kwargs: FakeEnv(**kwargs),
    )
    agent = StubAgent(action=0)

    summary = routing_evaluator.evaluate_routing_episode(
        circuit=QuantumCircuit(2),
        coupling_edges=[(0, 1)],
        agent=agent,
        seed=5,
        initial_layout=None,
        frontier_mode="sequential",
        max_steps=3,
        lookahead_window=2,
    )

    assert agent.calls == [({"obs": "reset"}, True)]
    assert summary.steps_executed == 1
    assert summary.total_reward == 1.25
    assert summary.completed is True
    assert summary.truncated is False
    assert summary.total_swaps == 1
    assert summary.gates_executed_count == 2
    assert summary.swap_trace == [(1, 2)]


def test_evaluate_routing_episode_passes_action_masks_for_masked_contract(monkeypatch) -> None:
    from src.integration import routing_evaluator

    expected_mask = np.array([True, False, True], dtype=bool)

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.current_layout = [0, 1]
            self.total_swaps = 1

        def reset(self, *, seed=None, options=None):
            return {"obs": "reset"}, {"already_completed_at_reset": False}

        def action_masks(self):
            return expected_mask

        def step(self, action):
            return {"obs": "done"}, 1.25, True, False, {"gates_executed": 2}

    monkeypatch.setattr(
        routing_evaluator,
        "_create_routing_env",
        lambda **kwargs: FakeEnv(**kwargs),
    )
    agent = StubMaskableAgent(action=0)

    summary = routing_evaluator.evaluate_routing_episode(
        circuit=QuantumCircuit(2),
        coupling_edges=[(0, 1)],
        agent=agent,
        seed=5,
        initial_layout=None,
        frontier_mode="sequential",
        max_steps=3,
        lookahead_window=2,
        masked=True,
    )

    assert len(agent.calls) == 1
    call_obs, call_masks, call_deterministic = agent.calls[0]
    assert call_obs == {"obs": "reset"}
    assert call_deterministic is True
    assert np.array_equal(call_masks, expected_mask)
    assert summary.steps_executed == 1


def test_evaluate_routing_episode_supports_real_quantum_rl_agent_wrapper_for_masks(monkeypatch) -> None:
    from src.integration import routing_evaluator
    from src.rl_module.agent import QuantumRLAgent

    expected_mask = np.array([True, False, True], dtype=bool)
    model_calls = []

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.current_layout = [0, 1]
            self.total_swaps = 1

        def reset(self, *, seed=None, options=None):
            return {"obs": "reset"}, {"already_completed_at_reset": False}

        def action_masks(self):
            return expected_mask

        def step(self, action):
            return {"obs": "done"}, 1.25, True, False, {"gates_executed": 2}

    class FakeModel:
        def predict(self, observation, deterministic=False, action_masks=None):
            model_calls.append((observation, deterministic, action_masks))
            return 0, None

    monkeypatch.setattr(
        routing_evaluator,
        "_create_routing_env",
        lambda **kwargs: FakeEnv(**kwargs),
    )

    agent = object.__new__(QuantumRLAgent)
    agent.model = FakeModel()

    summary = routing_evaluator.evaluate_routing_episode(
        circuit=QuantumCircuit(2),
        coupling_edges=[(0, 1)],
        agent=agent,
        seed=5,
        initial_layout=None,
        frontier_mode="sequential",
        max_steps=3,
        lookahead_window=2,
        masked=True,
    )

    assert len(model_calls) == 1
    call_obs, call_deterministic, call_masks = model_calls[0]
    assert call_obs == {"obs": "reset"}
    assert call_deterministic is True
    assert np.array_equal(call_masks, expected_mask)
    assert summary.steps_executed == 1


def test_evaluate_routing_episode_does_not_pass_action_masks_for_legacy_contract(monkeypatch) -> None:
    from src.integration import routing_evaluator

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.current_layout = [0, 1]
            self.total_swaps = 1

        def reset(self, *, seed=None, options=None):
            return {"obs": "reset"}, {"already_completed_at_reset": False}

        def action_masks(self):
            raise AssertionError("legacy evaluation should not query action masks")

        def step(self, action):
            return {"obs": "done"}, 1.25, True, False, {"gates_executed": 2}

    monkeypatch.setattr(
        routing_evaluator,
        "_create_routing_env",
        lambda **kwargs: FakeEnv(**kwargs),
    )
    agent = StubAgent(action=0)

    summary = routing_evaluator.evaluate_routing_episode(
        circuit=QuantumCircuit(2),
        coupling_edges=[(0, 1)],
        agent=agent,
        seed=5,
        initial_layout=None,
        frontier_mode="sequential",
        max_steps=3,
        lookahead_window=2,
        masked=False,
    )

    assert agent.calls == [({"obs": "reset"}, True)]
    assert summary.steps_executed == 1


def test_evaluate_routing_episode_handles_already_completed_reset_without_predict(monkeypatch) -> None:
    from src.integration import routing_evaluator

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.current_layout = [0, 1]
            self.total_swaps = 0

        def reset(self, *, seed=None, options=None):
            return {"obs": "reset"}, {
                "already_completed_at_reset": True,
                "total_gates": 0,
            }

    monkeypatch.setattr(
        routing_evaluator,
        "_create_routing_env",
        lambda **kwargs: FakeEnv(**kwargs),
    )
    agent = StubAgent(action=0)

    summary = routing_evaluator.evaluate_routing_episode(
        circuit=QuantumCircuit(2),
        coupling_edges=[(0, 1)],
        agent=agent,
        seed=13,
        initial_layout=None,
        frontier_mode="sequential",
        max_steps=2,
        lookahead_window=1,
    )

    assert agent.calls == []
    assert summary.steps_executed == 0
    assert summary.completed is True
    assert summary.truncated is False
    assert summary.gates_executed_count == 0


def test_evaluate_routing_episode_normalizes_layouts_to_python_ints(monkeypatch) -> None:
    from src.integration import routing_evaluator

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.current_layout = np.array([2, 0, 1], dtype=np.int32)
            self.total_swaps = 0

        def reset(self, *, seed=None, options=None):
            return {"obs": "reset"}, {"already_completed_at_reset": True, "total_gates": 1}

    monkeypatch.setattr(
        routing_evaluator,
        "_create_routing_env",
        lambda **kwargs: FakeEnv(**kwargs),
    )

    summary = routing_evaluator.evaluate_routing_episode(
        circuit=QuantumCircuit(3),
        coupling_edges=[(0, 1), (1, 2)],
        agent=StubAgent(),
        seed=19,
        initial_layout=np.array([1, 2, 0], dtype=np.int64),
        frontier_mode="sequential",
        max_steps=4,
        lookahead_window=2,
    )

    assert summary.initial_layout == [1, 2, 0]
    assert summary.final_layout == [2, 0, 1]
    assert all(type(entry) is int for entry in summary.initial_layout)
    assert all(type(entry) is int for entry in summary.final_layout)
    assert json.dumps(asdict(summary))


def test_evaluate_routing_episode_counts_gates_executed_during_partial_reset(monkeypatch) -> None:
    from src.integration import routing_evaluator

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.current_layout = [0, 1, 2]
            self.total_swaps = 1
            self.remaining_gates = ["gate-2", "gate-3"]

        def reset(self, *, seed=None, options=None):
            return {"obs": "reset"}, {"already_completed_at_reset": False, "total_gates": 3}

        def step(self, action):
            return {"obs": "done"}, 0.5, True, False, {"gates_executed": 2}

    monkeypatch.setattr(
        routing_evaluator,
        "_create_routing_env",
        lambda **kwargs: FakeEnv(**kwargs),
    )

    summary = routing_evaluator.evaluate_routing_episode(
        circuit=QuantumCircuit(3),
        coupling_edges=[(0, 1), (1, 2)],
        agent=StubAgent(),
        seed=23,
        initial_layout=[0, 1, 2],
        frontier_mode="sequential",
        max_steps=5,
        lookahead_window=2,
    )

    assert summary.steps_executed == 1
    assert summary.gates_executed_count == 3


def test_evaluate_routing_episode_only_records_valid_swaps_in_trace(monkeypatch) -> None:
    from src.integration import routing_evaluator

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.current_layout = [1, 0, 2]
            self.total_swaps = 3
            self._step = 0

        def reset(self, *, seed=None, options=None):
            return {"obs": "reset"}, {"already_completed_at_reset": False}

        def step(self, action):
            self._step += 1
            if self._step == 1:
                return {"obs": "mid"}, 0.1, False, False, {
                    "gates_executed": 0,
                    "action_type": "swap",
                    "is_valid_action": False,
                    "swap_edge": (0, 1),
                }
            if self._step == 2:
                return {"obs": "mid-2"}, 0.2, False, False, {
                    "gates_executed": 1,
                    "action_type": "swap",
                    "is_valid_action": True,
                    "swap_edge": (1, 2),
                }
            return {"obs": "done"}, 0.3, True, False, {
                "gates_executed": 1,
                "action_type": "invalid",
                "is_valid_action": False,
                "swap_edge": (2, 3),
            }

    monkeypatch.setattr(
        routing_evaluator,
        "_create_routing_env",
        lambda **kwargs: FakeEnv(**kwargs),
    )

    summary = routing_evaluator.evaluate_routing_episode(
        circuit=QuantumCircuit(3),
        coupling_edges=[(0, 1), (1, 2)],
        agent=StubAgent(),
        seed=29,
        initial_layout=[0, 1, 2],
        frontier_mode="sequential",
        max_steps=5,
        lookahead_window=2,
    )

    assert summary.total_swaps == 1
    assert summary.swap_trace == [(1, 2)]


def test_evaluate_routing_episode_records_exact_executed_gate_trace(monkeypatch) -> None:
    from src.integration import routing_evaluator

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.current_layout = [0, 1, 2]
            self.total_swaps = 1
            self.remaining_gates = ["gate-2"]

        def reset(self, *, seed=None, options=None):
            return {
                "obs": "reset"
            }, {
                "already_completed_at_reset": False,
                "total_gates": 2,
                "executed_gates": [("h", 0, 0)],
            }

        def step(self, action):
            return {
                "obs": "done"
            }, 0.5, True, False, {
                "gates_executed": 1,
                "executed_gates": [("cx", 0, 1)],
                "action_type": "swap",
                "is_valid_action": True,
                "swap_edge": (1, 2),
            }

    monkeypatch.setattr(
        routing_evaluator,
        "_create_routing_env",
        lambda **kwargs: FakeEnv(**kwargs),
    )

    summary = routing_evaluator.evaluate_routing_episode(
        circuit=QuantumCircuit(3),
        coupling_edges=[(0, 1), (1, 2)],
        agent=StubAgent(),
        seed=31,
        initial_layout=[0, 1, 2],
        frontier_mode="sequential",
        max_steps=4,
        lookahead_window=2,
    )

    assert summary.executed_gate_trace == [("h", 0, 0), ("cx", 0, 1)]


def test_build_routed_circuit_replays_swap_trace_into_physical_circuit() -> None:
    from src.integration.routing_evaluator import build_routed_circuit

    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    routed_circuit, final_layout = build_routed_circuit(
        circuit=circuit,
        coupling_edges=[(0, 1), (1, 2)],
        initial_layout=[0, 2, 1],
        swap_trace=[(1, 2)],
        frontier_mode="sequential",
    )

    assert routed_circuit.num_qubits == 3
    assert final_layout == [0, 1, 2]
    assert routed_circuit.count_ops().get("swap", 0) == 1
    assert routed_circuit.count_ops().get("cx", 0) == 2


def test_build_routed_circuit_rejects_operations_with_more_than_two_qubits() -> None:
    from src.integration.routing_evaluator import build_routed_circuit

    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)

    with pytest.raises(ValueError, match=">2 qubits"):
        build_routed_circuit(
            circuit=circuit,
            coupling_edges=[(0, 1), (1, 2)],
            initial_layout=[0, 1, 2],
            swap_trace=[],
            frontier_mode="sequential",
        )


def test_build_routed_circuit_prefers_exact_executed_gate_trace_when_provided() -> None:
    from src.integration.routing_evaluator import build_routed_circuit

    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    routed_circuit, final_layout = build_routed_circuit(
        circuit=circuit,
        coupling_edges=[(0, 1), (1, 2)],
        initial_layout=[0, 2, 1],
        swap_trace=[(1, 2)],
        frontier_mode="sequential",
        executed_gate_trace=[("h", 0, 0), ("cx", 0, 1), ("cx", 1, 2)],
    )

    assert final_layout == [0, 1, 2]
    assert routed_circuit.count_ops().get("swap", 0) == 1
    assert routed_circuit.count_ops().get("h", 0) == 1
    assert routed_circuit.count_ops().get("cx", 0) == 2
