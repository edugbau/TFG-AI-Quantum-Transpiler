import json
from dataclasses import asdict

import numpy as np
from qiskit import QuantumCircuit

from src.integration.contracts import RoutingEpisodeSummary


class StubAgent:
    def __init__(self, action: int = 0) -> None:
        self.action = action
        self.calls: list[tuple[object, bool]] = []

    def predict(self, obs, deterministic: bool = False):
        self.calls.append((obs, deterministic))
        return self.action, None


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
            return {"obs": "next"}, 1.25, True, False, {"gates_executed": 2}

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
        max_steps=5,
        lookahead_window=2,
    )

    assert summary.steps_executed == 1
    assert summary.gates_executed_count == 3
