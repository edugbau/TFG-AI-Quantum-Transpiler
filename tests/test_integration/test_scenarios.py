import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit

from src.integration.contracts import RoutingEpisodeSummary, ScenarioRequest


def _make_request(scenario_name: str, **overrides) -> ScenarioRequest:
    data = {
        "scenario_name": scenario_name,
        "circuit_name": "ghz_3",
        "num_qubits": 3,
        "backend_name": "fake_backend",
        "seed": 17,
    }
    data.update(overrides)
    return ScenarioRequest(**data)


def _make_transpilation_result(**overrides):
    return SimpleNamespace(
        to_dict=lambda: {
            "backend_name": "fake_backend",
            "trans_depth": 12,
            "trans_two_qubit_gates": 4,
        },
        **overrides,
    )


def test_run_baseline_scenario_returns_transpilation_metrics_only(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    resolve_calls = []
    transpile_calls = []

    monkeypatch.setattr(
        scenarios,
        "_load_circuit",
        lambda request: circuit,
    )
    monkeypatch.setattr(
        scenarios,
        "resolve_backend_bundle",
        lambda backend_name: resolve_calls.append(backend_name)
        or SimpleNamespace(backend_name=backend_name, backend="backend-object"),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_circuit",
        lambda **kwargs: transpile_calls.append(kwargs) or _make_transpilation_result(),
    )

    result = scenarios.run_baseline_scenario(_make_request("Baseline"))

    assert result.success is True
    assert result.selected_layout is None
    assert result.routing_summary is None
    assert result.notes == []
    assert result.transpilation_metrics is not None
    assert "trans_depth" in result.transpilation_metrics
    assert resolve_calls == ["fake_backend"]
    assert transpile_calls == [
        {
            "circuit": circuit,
            "backend": "backend-object",
            "backend_name": "fake_backend",
            "seed": 17,
        }
    ]


def test_run_mo_only_scenario_returns_selected_layout_and_transpilation_metrics(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO_Only")
    bundle = SimpleNamespace(backend_name="fake_backend", backend="backend-object")
    mo_calls = []
    select_calls = []
    transpile_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout_quick",
        lambda circuit, backend, seed: mo_calls.append((circuit, backend, seed)) or "mo-result",
    )
    monkeypatch.setattr(
        scenarios,
        "select_layout_from_mo_result",
        lambda result, *, policy, objective_index=0: select_calls.append(
            (result, policy, objective_index)
        ) or [2, 0, 1],
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_with_custom_layout",
        lambda **kwargs: transpile_calls.append(kwargs)
        or _make_transpilation_result(initial_layout=[2, 0, 1]),
    )

    result = scenarios.run_mo_only_scenario(request)

    assert result.success is True
    assert result.selected_layout == [2, 0, 1]
    assert result.routing_summary is None
    assert result.notes == []
    assert result.transpilation_metrics is not None
    assert mo_calls == [(circuit, "backend-object", 17)]
    assert select_calls == [("mo-result", request.layout_policy, request.mo_objective_index)]
    assert transpile_calls == [
        {
            "circuit": circuit,
            "layout": [2, 0, 1],
            "backend": "backend-object",
            "backend_name": "fake_backend",
            "seed": 17,
        }
    ]


def test_run_mo_only_scenario_uses_non_quick_optimizer_when_requested(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO_Only", mo_use_quick=False)
    bundle = SimpleNamespace(backend_name="fake_backend", backend="backend-object")
    optimize_calls = []
    quick_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout_quick",
        lambda **kwargs: quick_calls.append(kwargs) or "quick-result",
    )
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout",
        lambda **kwargs: optimize_calls.append(kwargs) or "mo-result",
    )
    monkeypatch.setattr(
        scenarios,
        "select_layout_from_mo_result",
        lambda result, *, policy, objective_index=0: [2, 0, 1],
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_with_custom_layout",
        lambda **kwargs: _make_transpilation_result(),
    )

    scenarios.run_mo_only_scenario(request)

    assert quick_calls == []
    assert optimize_calls == [
        {
            "circuit": circuit,
            "backend": "backend-object",
            "backend_name": "fake_backend",
        }
    ]


def test_run_rl_only_scenario_returns_routing_summary_and_note(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend="backend-object",
        coupling_edges=[(0, 1), (1, 2)],
    )
    summary = RoutingEpisodeSummary(
        initial_layout=None,
        final_layout=[0, 1, 2],
        steps_executed=4,
        total_reward=2.5,
        completed=True,
        truncated=False,
        total_swaps=1,
        gates_executed_count=3,
    )
    eval_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(scenarios, "_load_agent", lambda request: "agent-object")
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: eval_calls.append(kwargs) or summary,
    )

    result = scenarios.run_rl_only_scenario(
        _make_request("RL_Only", rl_model_path="models/policy.zip")
    )

    assert result.success is True
    assert result.selected_layout is None
    assert result.transpilation_metrics is None
    assert result.routing_summary is summary
    assert result.notes == [
        "RL outputs are episode summaries, not final circuits."
    ]
    assert eval_calls == [
        {
            "circuit": circuit,
            "coupling_edges": [(0, 1), (1, 2)],
            "agent": "agent-object",
            "seed": 17,
            "initial_layout": None,
            "max_steps": scenarios._DEFAULT_RL_MAX_STEPS,
            "lookahead_window": scenarios._DEFAULT_RL_LOOKAHEAD_WINDOW,
        }
    ]


def test_load_agent_uses_public_rl_api(monkeypatch) -> None:
    from src.integration import scenarios

    load_calls = []

    class StubQuantumRLAgent:
        @staticmethod
        def load(path, env=None):
            load_calls.append((path, env))
            return "agent-object"

    monkeypatch.setitem(
        sys.modules,
        "src.rl_module",
        SimpleNamespace(QuantumRLAgent=StubQuantumRLAgent),
    )

    agent = scenarios._load_agent(_make_request("RL_Only", rl_model_path="models/policy.zip"))

    assert agent == "agent-object"
    assert load_calls == [("models/policy.zip", None)]


def test_run_mo_rl_scenario_returns_selected_layout_routing_summary_and_note(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO+RL", rl_model_path="models/policy.zip")
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend="backend-object",
        coupling_edges=[(0, 1), (1, 2)],
    )
    summary = RoutingEpisodeSummary(
        initial_layout=[2, 0, 1],
        final_layout=[1, 0, 2],
        steps_executed=5,
        total_reward=3.0,
        completed=True,
        truncated=False,
        total_swaps=2,
        gates_executed_count=4,
    )
    eval_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout_quick",
        lambda circuit, backend, seed: "mo-result",
    )
    monkeypatch.setattr(
        scenarios,
        "select_layout_from_mo_result",
        lambda result, *, policy, objective_index=0: [2, 0, 1],
    )
    monkeypatch.setattr(scenarios, "_load_agent", lambda request: "agent-object")
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: eval_calls.append(kwargs) or summary,
    )

    result = scenarios.run_mo_rl_scenario(request)

    assert result.success is True
    assert result.selected_layout == [2, 0, 1]
    assert result.transpilation_metrics is None
    assert result.routing_summary is summary
    assert result.notes == [
        "RL outputs are episode summaries, not final circuits."
    ]
    assert eval_calls == [
        {
            "circuit": circuit,
            "coupling_edges": [(0, 1), (1, 2)],
            "agent": "agent-object",
            "seed": 17,
            "initial_layout": [2, 0, 1],
            "max_steps": scenarios._DEFAULT_RL_MAX_STEPS,
            "lookahead_window": scenarios._DEFAULT_RL_LOOKAHEAD_WINDOW,
        }
    ]


def test_run_mo_rl_scenario_normalizes_numpy_selected_layout_before_handoff(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO+RL", rl_model_path="models/policy.zip")
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend=SimpleNamespace(num_qubits=3),
        coupling_edges=[(0, 1), (1, 2)],
    )
    eval_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout_quick",
        lambda circuit, backend, seed: "mo-result",
    )
    monkeypatch.setattr(
        scenarios,
        "select_layout_from_mo_result",
        lambda result, *, policy, objective_index=0: [np.int64(2), np.int64(0), np.int64(1)],
    )
    monkeypatch.setattr(scenarios, "_load_agent", lambda request: "agent-object")
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: eval_calls.append(kwargs)
        or RoutingEpisodeSummary(
            initial_layout=list(kwargs["initial_layout"]),
            final_layout=[1, 0, 2],
            steps_executed=1,
            total_reward=1.0,
            completed=True,
            truncated=False,
            total_swaps=0,
            gates_executed_count=2,
        ),
    )

    result = scenarios.run_mo_rl_scenario(request)

    assert result.selected_layout == [2, 0, 1]
    assert all(type(entry) is int for entry in result.selected_layout)
    assert eval_calls[0]["initial_layout"] == [2, 0, 1]
    assert all(type(entry) is int for entry in eval_calls[0]["initial_layout"])


def test_run_mo_rl_smoke_through_runner_exercises_real_mo_to_rl_handoff(monkeypatch) -> None:
    from src.integration import runner, scenarios

    circuit = QuantumCircuit(3)
    selected_layout = [2, 0, 1]
    mo_invoked = []
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend=SimpleNamespace(num_qubits=3),
        coupling_edges=[(0, 1), (1, 2)],
    )

    class StubAgent:
        def __init__(self) -> None:
            self.calls = []

        def predict(self, obs, deterministic: bool = False):
            self.calls.append((obs, deterministic))
            return 0, None

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.current_layout = [1, 0, 2]
            self.total_swaps = 1

        def reset(self, *, seed=None, options=None):
            assert options == {"initial_layout": selected_layout}
            return {"obs": "reset"}, {"already_completed_at_reset": False}

        def step(self, action):
            return {"obs": "done"}, 1.5, True, False, {"gates_executed": 3}

        def close(self) -> None:
            return None

    stub_agent = StubAgent()

    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "circuits_from_library",
        lambda *, num_qubits, seed: {"ghz_3": circuit},
    )
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout_quick",
        lambda circuit, backend, seed: mo_invoked.append(True) or object(),
    )
    monkeypatch.setattr(
        scenarios,
        "select_layout_from_mo_result",
        lambda result, *, policy, objective_index=0: selected_layout,
    )
    monkeypatch.setattr(scenarios, "_load_agent", lambda request: stub_agent)
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: RoutingEpisodeSummary(
            initial_layout=list(kwargs["initial_layout"]),
            final_layout=[1, 0, 2],
            steps_executed=1,
            total_reward=1.5,
            completed=True,
            truncated=False,
            total_swaps=1,
            gates_executed_count=3,
        ),
    )

    payload = runner.run_from_args(
        [
            "--scenario",
            "MO+RL",
            "--circuit",
            "ghz_3",
            "--num-qubits",
            "3",
            "--backend",
            "fake_backend",
            "--seed",
            "17",
            "--rl-model-path",
            "models/policy.zip",
        ]
    )

    assert mo_invoked == [True]
    assert payload["success"] is True
    assert payload["selected_layout"] == selected_layout
    assert payload["routing_summary"] is not None
    assert payload["routing_summary"]["initial_layout"] == selected_layout
    assert payload["notes"] == ["RL outputs are episode summaries, not final circuits."]
    assert payload["routing_summary"]["final_layout"] == [1, 0, 2]


@pytest.mark.parametrize(
    ("runner_name", "scenario_name"),
    [
        ("run_baseline_scenario", "MO_Only"),
        ("run_mo_only_scenario", "Baseline"),
        ("run_rl_only_scenario", "MO+RL"),
        ("run_mo_rl_scenario", "RL_Only"),
    ],
)
def test_scenario_runners_reject_mismatched_request_names(runner_name, scenario_name) -> None:
    from src.integration import scenarios

    runner = getattr(scenarios, runner_name)
    rl_model_path = None
    if scenario_name in {"RL_Only", "MO+RL"}:
        rl_model_path = "models/policy.zip"

    with pytest.raises(ValueError, match="scenario_name"):
        runner(_make_request(scenario_name, rl_model_path=rl_model_path))


@pytest.mark.parametrize("selected_layout", ([0, 1], [0, -1, 2], [0, 0, 1]))
def test_mo_layout_validation_fails_before_deeper_dependencies(monkeypatch, selected_layout) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO_Only")
    bundle = SimpleNamespace(backend_name="fake_backend", backend="backend-object")
    transpile_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout_quick",
        lambda circuit, backend, seed: "mo-result",
    )
    monkeypatch.setattr(
        scenarios,
        "select_layout_from_mo_result",
        lambda result, *, policy, objective_index=0: selected_layout,
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_with_custom_layout",
        lambda **kwargs: transpile_calls.append(kwargs) or _make_transpilation_result(),
    )

    with pytest.raises(ValueError, match="layout"):
        scenarios.run_mo_only_scenario(request)

    assert transpile_calls == []


def test_mo_layout_validation_rejects_physical_qubits_outside_backend_range(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO_Only")
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend=SimpleNamespace(num_qubits=3),
    )
    transpile_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout_quick",
        lambda circuit, backend, seed: "mo-result",
    )
    monkeypatch.setattr(
        scenarios,
        "select_layout_from_mo_result",
        lambda result, *, policy, objective_index=0: [0, 1, 3],
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_with_custom_layout",
        lambda **kwargs: transpile_calls.append(kwargs) or _make_transpilation_result(),
    )

    with pytest.raises(ValueError, match="backend range"):
        scenarios.run_mo_only_scenario(request)

    assert transpile_calls == []


def test_load_circuit_raises_value_error_for_unknown_library_circuit(monkeypatch) -> None:
    from src.integration import scenarios

    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "circuits_from_library",
        lambda *, num_qubits, seed: {"other": QuantumCircuit(num_qubits)},
    )

    with pytest.raises(ValueError, match="Unknown circuit name"):
        scenarios._load_circuit(_make_request("Baseline"))
