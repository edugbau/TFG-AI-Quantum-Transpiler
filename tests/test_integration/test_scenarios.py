import json
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
from qiskit import QuantumCircuit

from src.integration.contracts import (
    CircuitFormat,
    CircuitSource,
    RoutingEpisodeSummary,
    ScenarioRequest,
)
from src.rl_module.model_metadata import build_run_metadata, save_run_metadata


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


def _make_request_without_validation(scenario_name: str, **overrides) -> ScenarioRequest:
    request = object.__new__(ScenarioRequest)
    valid_request = _make_request(
        scenario_name,
        rl_model_path="models/policy.zip" if scenario_name in {"RL_Only", "MO+RL"} else None,
    )
    for field_name in ScenarioRequest.__slots__:
        setattr(request, field_name, getattr(valid_request, field_name))
    for field_name, value in overrides.items():
        setattr(request, field_name, value)
    return request


def _make_transpilation_result(**overrides):
    return SimpleNamespace(
        to_dict=lambda: {
            "backend_name": "fake_backend",
            "trans_depth": 12,
            "trans_two_qubit_gates": 4,
        },
        to_artifact_dict=lambda: {
            "artifact_version": "transpilation_result.v1",
            "baseline_name": None,
            "transpilation": {
                "optimization_level": 1,
                "seed": 17,
                "elapsed_time_s": 0.1,
                "baseline_name": None,
                "initial_layout": [2, 0, 1],
                "final_layout": [1, 0, 2],
            },
        },
        **overrides,
    )


def _write_legacy_sb3_model(tmp_path, model_name: str, *, policy_module: str) -> None:
    with ZipFile(tmp_path / model_name, "w") as archive:
        archive.writestr(
            "data",
            json.dumps(
                {
                    "policy_class": {
                        "__module__": policy_module,
                    }
                }
            ),
        )


def test_run_baseline_scenario_returns_transpilation_metrics_only(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    circuit.name = "ghz_3"
    circuit.metadata = {
        "source_kind": "library",
        "source_format": "library",
        "resolved_circuit_name": "ghz_3",
    }
    baseline_calls = []
    row = {
        "backend_name": "fake_backend",
        "baseline_name": "qiskit_level_1",
        "optimization_level": 1,
        "seed": 17,
        "elapsed_time_s": 0.25,
        "depth_reduction": 0.4,
        "two_qubit_gate_overhead": 1.5,
        "orig_num_qubits": 3,
        "orig_depth": 5,
        "orig_total_gates": 6,
        "orig_two_qubit_gates": 2,
        "trans_depth": 12,
        "trans_num_qubits": 3,
        "trans_total_gates": 9,
        "trans_two_qubit_gates": 4,
    }
    artifact = {
        "artifact_version": "transpilation_result.v1",
        "baseline_name": "qiskit_level_1",
        "circuit": {
            "name": "ghz_3",
            "num_qubits": 3,
            "source_kind": "library",
            "source_format": "library",
            "source_path": None,
            "resolved_circuit_name": "ghz_3",
        },
        "backend": {
            "backend_name": "fake_backend",
            "coupling_edges_count": 2,
            "avg_error_2q": 0.01,
        },
        "transpilation": {
            "optimization_level": 1,
            "seed": 17,
            "elapsed_time_s": 0.25,
            "baseline_name": "qiskit_level_1",
            "initial_layout": None,
            "final_layout": [0, 1, 2],
        },
        "metrics": {
            "original": {
                "num_qubits": 3,
                "depth": 5,
                "total_gates": 6,
                "two_qubit_gates": 2,
            },
            "transpiled": {
                "num_qubits": 3,
                "depth": 12,
                "total_gates": 9,
                "two_qubit_gates": 4,
            },
        },
    }

    monkeypatch.setattr(
        scenarios,
        "_load_circuit",
        lambda request: circuit,
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "run_named_baseline",
        lambda baseline_name, circuit, backend_names, seed, layout=None, include_artifact=False: baseline_calls.append(
            {
                "baseline_name": baseline_name,
                "circuit": circuit,
                "backend_names": backend_names,
                "seed": seed,
                "layout": layout,
                "include_artifact": include_artifact,
            }
        )
        or [(row, artifact)],
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_circuit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected extra transpilation")),
    )

    result = scenarios.run_baseline_scenario(_make_request("Baseline"))

    assert result.success is True
    assert result.selected_layout is None
    assert result.routing_summary is None
    assert result.notes == []
    assert result.transpilation_metrics == {
        "backend_name": "fake_backend",
        "baseline_name": "qiskit_level_1",
        "optimization_level": 1,
        "seed": 17,
        "elapsed_time_s": 0.25,
        "depth_reduction": 0.4,
        "two_qubit_gate_overhead": 1.5,
        "orig_num_qubits": 3,
        "orig_depth": 5,
        "orig_total_gates": 6,
        "orig_two_qubit_gates": 2,
        "trans_depth": 12,
        "trans_num_qubits": 3,
        "trans_total_gates": 9,
        "trans_two_qubit_gates": 4,
    }
    assert result.transpilation_artifact == artifact
    assert baseline_calls == [
        {
            "baseline_name": "qiskit_level_1",
            "circuit": circuit,
            "backend_names": ["fake_backend"],
            "seed": 17,
            "layout": None,
            "include_artifact": True,
        }
    ]


def test_run_mo_only_scenario_returns_selected_layout_and_transpilation_metrics(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO_Only")
    circuit.name = "ghz_3"
    circuit.metadata = {
        "source_kind": "library",
        "source_format": "library",
        "resolved_circuit_name": "ghz_3",
    }
    bundle = SimpleNamespace(backend_name="fake_backend", backend="backend-object")
    mo_calls = []
    select_calls = []
    baseline_calls = []
    row = {
        "backend_name": "fake_backend",
        "baseline_name": "custom_layout_level_1",
        "optimization_level": 1,
        "seed": 17,
        "elapsed_time_s": 0.5,
        "depth_reduction": 0.2,
        "two_qubit_gate_overhead": 1.1,
        "orig_num_qubits": 3,
        "orig_depth": 5,
        "orig_total_gates": 6,
        "orig_two_qubit_gates": 2,
        "trans_depth": 12,
        "trans_num_qubits": 3,
        "trans_total_gates": 8,
        "trans_two_qubit_gates": 4,
        "initial_layout": [2, 0, 1],
    }
    artifact = {
        "artifact_version": "transpilation_result.v1",
        "baseline_name": "custom_layout_level_1",
        "circuit": {
            "name": "ghz_3",
            "num_qubits": 3,
            "source_kind": "library",
            "source_format": "library",
            "source_path": None,
            "resolved_circuit_name": "ghz_3",
        },
        "backend": {
            "backend_name": "fake_backend",
            "coupling_edges_count": 2,
            "avg_error_2q": 0.02,
        },
        "transpilation": {
            "optimization_level": 1,
            "seed": 17,
            "elapsed_time_s": 0.5,
            "baseline_name": "custom_layout_level_1",
            "initial_layout": [2, 0, 1],
            "final_layout": [1, 2, 0],
        },
        "metrics": {
            "original": {
                "num_qubits": 3,
                "depth": 5,
                "total_gates": 6,
                "two_qubit_gates": 2,
            },
            "transpiled": {
                "num_qubits": 3,
                "depth": 12,
                "total_gates": 8,
                "two_qubit_gates": 4,
            },
        },
    }

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
        "run_named_baseline",
        lambda baseline_name, circuit, backend_names, seed, layout=None, include_artifact=False: baseline_calls.append(
            {
                "baseline_name": baseline_name,
                "circuit": circuit,
                "backend_names": backend_names,
                "seed": seed,
                "layout": layout,
                "include_artifact": include_artifact,
            }
        )
        or [(row, artifact)],
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_with_custom_layout",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected extra transpilation")),
    )

    result = scenarios.run_mo_only_scenario(request)

    assert result.success is True
    assert result.selected_layout == [2, 0, 1]
    assert result.routing_summary is None
    assert result.notes == []
    assert result.transpilation_metrics == {
        "backend_name": "fake_backend",
        "baseline_name": "custom_layout_level_1",
        "optimization_level": 1,
        "seed": 17,
        "elapsed_time_s": 0.5,
        "depth_reduction": 0.2,
        "two_qubit_gate_overhead": 1.1,
        "orig_num_qubits": 3,
        "orig_depth": 5,
        "orig_total_gates": 6,
        "orig_two_qubit_gates": 2,
        "trans_depth": 12,
        "trans_num_qubits": 3,
        "trans_total_gates": 8,
        "trans_two_qubit_gates": 4,
        "initial_layout": [2, 0, 1],
    }
    assert result.transpilation_artifact == artifact
    assert mo_calls == [(circuit, "backend-object", 17)]
    assert select_calls == [("mo-result", request.layout_policy, request.mo_objective_index)]
    assert baseline_calls == [
        {
            "baseline_name": "custom_layout_level_1",
            "circuit": circuit,
            "layout": [2, 0, 1],
            "backend_names": ["fake_backend"],
            "seed": 17,
            "include_artifact": True,
        }
    ]


def test_run_mo_only_scenario_uses_non_quick_optimizer_when_requested(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO_Only", mo_use_quick=False)
    bundle = SimpleNamespace(backend_name="fake_backend", backend="backend-object")
    optimize_calls = []
    quick_calls = []
    baseline_calls = []

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
        "run_named_baseline",
        lambda baseline_name, circuit, backend_names, seed, layout=None, include_artifact=False: baseline_calls.append(
            {
                "baseline_name": baseline_name,
                "circuit": circuit,
                "backend_names": backend_names,
                "seed": seed,
                "layout": layout,
                "include_artifact": include_artifact,
            }
        )
        or [
            (
                {
                    "backend_name": "fake_backend",
                    "baseline_name": "custom_layout_level_1",
                    "optimization_level": 1,
                    "seed": 17,
                    "elapsed_time_s": 0.5,
                    "depth_reduction": 0.2,
                    "two_qubit_gate_overhead": 1.1,
                    "orig_num_qubits": 3,
                    "orig_depth": 5,
                    "orig_total_gates": 6,
                    "orig_two_qubit_gates": 2,
                    "trans_depth": 12,
                    "trans_num_qubits": 3,
                    "trans_total_gates": 8,
                    "trans_two_qubit_gates": 4,
                    "initial_layout": [2, 0, 1],
                },
                {
                    "artifact_version": "transpilation_result.v1",
                    "baseline_name": "custom_layout_level_1",
                    "backend": {"backend_name": "fake_backend"},
                    "transpilation": {"initial_layout": [2, 0, 1]},
                },
            )
        ],
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_with_custom_layout",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected extra transpilation")),
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
    assert baseline_calls == [
        {
            "baseline_name": "custom_layout_level_1",
            "circuit": circuit,
            "backend_names": ["fake_backend"],
            "seed": 17,
            "layout": [2, 0, 1],
            "include_artifact": True,
        }
    ]


def test_run_baseline_scenario_rejects_row_artifact_baseline_drift(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    circuit.name = "ghz_3"

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "run_named_baseline",
        lambda baseline_name, circuit, backend_names, seed, layout=None, include_artifact=False: [
            (
                {
                    "backend_name": "fake_backend",
                    "baseline_name": "qiskit_level_2",
                    "optimization_level": 1,
                    "seed": 17,
                    "trans_depth": 12,
                },
                {
                    "artifact_version": "transpilation_result.v1",
                    "baseline_name": "qiskit_level_2",
                    "backend": {"backend_name": "fake_backend"},
                    "transpilation": {"initial_layout": None},
                },
            )
        ],
    )

    with pytest.raises(ValueError, match="baseline_name"):
        scenarios.run_baseline_scenario(_make_request("Baseline"))


def test_run_mo_only_scenario_rejects_row_artifact_layout_drift(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    circuit.name = "ghz_3"
    request = _make_request("MO_Only")
    bundle = SimpleNamespace(backend_name="fake_backend", backend="backend-object")

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
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "run_named_baseline",
        lambda baseline_name, circuit, backend_names, seed, layout=None, include_artifact=False: [
            (
                {
                    "backend_name": "fake_backend",
                    "baseline_name": "custom_layout_level_1",
                    "optimization_level": 1,
                    "seed": 17,
                    "trans_depth": 12,
                    "initial_layout": [0, 1, 2],
                },
                {
                    "artifact_version": "transpilation_result.v1",
                    "baseline_name": "custom_layout_level_1",
                    "backend": {"backend_name": "fake_backend"},
                    "transpilation": {"initial_layout": [0, 1, 2]},
                },
            )
        ],
    )

    with pytest.raises(ValueError, match="initial_layout"):
        scenarios.run_mo_only_scenario(request)


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
        swap_trace=[(0, 1)],
    )
    eval_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": "agent-object",
    )
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
        "RL outputs are episode summaries, not final circuits.",
        "Legacy RL evaluation defaults were used because no run metadata sidecar was found.",
    ]
    assert eval_calls == [
        {
            "circuit": circuit,
            "coupling_edges": [(0, 1), (1, 2)],
            "agent": "agent-object",
            "seed": 17,
            "initial_layout": None,
            "frontier_mode": "sequential",
            "max_steps": scenarios._DEFAULT_RL_MAX_STEPS,
            "lookahead_window": scenarios._DEFAULT_RL_LOOKAHEAD_WINDOW,
            "masked": False,
        }
    ]


def test_run_rl_only_scenario_uses_saved_contract(monkeypatch, tmp_path) -> None:
    from src.integration import scenarios

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    save_run_metadata(
        tmp_path,
        build_run_metadata(
            mode="routing",
            algorithm="DQN",
            seed=17,
            frontier_mode="dag",
            lookahead_window=9,
            max_steps=333,
            basis_gates=None,
        ),
    )

    load_calls = []
    eval_calls = []
    circuit = QuantumCircuit(3)
    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(
        scenarios,
        "resolve_backend_bundle",
        lambda backend_name: SimpleNamespace(
            coupling_edges=[(0, 1)], backend="backend", backend_name=backend_name
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: eval_calls.append(kwargs)
        or RoutingEpisodeSummary(
            initial_layout=None,
            final_layout=[0, 1, 2],
            steps_executed=1,
            total_reward=1.0,
            completed=True,
            truncated=False,
            total_swaps=0,
            gates_executed_count=2,
        ),
    )

    class StubQuantumRLAgent:
        @staticmethod
        def load(path, env=None, algorithm="PPO", **kwargs):
            load_calls.append({"path": path, "algorithm": algorithm, "env": env})
            return "agent-object"

    monkeypatch.setitem(
        sys.modules,
        "src.rl_module",
        SimpleNamespace(QuantumRLAgent=StubQuantumRLAgent),
    )

    result = scenarios.run_rl_only_scenario(
        _make_request("RL_Only", rl_model_path=str(model_path))
    )

    assert load_calls == [{"path": str(model_path), "algorithm": "DQN", "env": None}]
    assert eval_calls[0]["lookahead_window"] == 9
    assert eval_calls[0]["max_steps"] == 333
    assert eval_calls[0]["frontier_mode"] == "dag"
    assert eval_calls[0]["masked"] is False
    assert result.notes == ["RL outputs are episode summaries, not final circuits."]


def test_run_rl_only_scenario_forwards_masked_contract_fields(monkeypatch, tmp_path) -> None:
    from src.integration import scenarios

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    save_run_metadata(
        tmp_path,
        build_run_metadata(
            mode="routing",
            algorithm="MaskablePPO",
            seed=17,
            frontier_mode="dag",
            lookahead_window=9,
            max_steps=333,
            basis_gates=None,
            mask_semantics="frontier_restricted_edges.v1",
        ),
    )

    load_calls = []
    eval_calls = []
    circuit = QuantumCircuit(3)
    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(
        scenarios,
        "resolve_backend_bundle",
        lambda backend_name: SimpleNamespace(
            coupling_edges=[(0, 1)], backend="backend", backend_name=backend_name
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: eval_calls.append(kwargs)
        or RoutingEpisodeSummary(
            initial_layout=None,
            final_layout=[0, 1, 2],
            steps_executed=1,
            total_reward=1.0,
            completed=True,
            truncated=False,
            total_swaps=0,
            gates_executed_count=2,
        ),
    )

    class StubQuantumRLAgent:
        @staticmethod
        def load(path, env=None, algorithm="PPO", **kwargs):
            load_calls.append({"path": path, "algorithm": algorithm, "env": env})
            return "agent-object"

    monkeypatch.setitem(
        sys.modules,
        "src.rl_module",
        SimpleNamespace(QuantumRLAgent=StubQuantumRLAgent),
    )

    result = scenarios.run_rl_only_scenario(
        _make_request("RL_Only", rl_model_path=str(model_path))
    )

    assert load_calls == [{"path": str(model_path), "algorithm": "MaskablePPO", "env": None}]
    assert eval_calls[0]["lookahead_window"] == 9
    assert eval_calls[0]["max_steps"] == 333
    assert eval_calls[0]["frontier_mode"] == "dag"
    assert eval_calls[0]["masked"] is True
    assert result.notes == ["RL outputs are episode summaries, not final circuits."]


def test_run_rl_only_scenario_adds_fallback_note_when_metadata_is_missing(monkeypatch, tmp_path) -> None:
    from src.integration import scenarios

    model_path = tmp_path / "legacy_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    circuit = QuantumCircuit(3)

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(
        scenarios,
        "resolve_backend_bundle",
        lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            backend="backend-object",
            coupling_edges=[(0, 1), (1, 2)],
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": "agent-object",
    )
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: RoutingEpisodeSummary(
            initial_layout=None,
            final_layout=[0, 1, 2],
            steps_executed=1,
            total_reward=1.0,
            completed=True,
            truncated=False,
            total_swaps=0,
            gates_executed_count=2,
        ),
    )

    result = scenarios.run_rl_only_scenario(
        _make_request("RL_Only", rl_model_path=str(model_path))
    )

    assert result.notes == [
        "RL outputs are episode summaries, not final circuits.",
        "Legacy RL evaluation defaults were used because no run metadata sidecar was found.",
    ]


def test_run_rl_only_scenario_recovers_dqn_algorithm_when_metadata_is_missing(monkeypatch, tmp_path) -> None:
    from src.integration import scenarios

    model_path = tmp_path / "legacy_dqn_model.zip"
    _write_legacy_sb3_model(
        tmp_path,
        model_path.name,
        policy_module="stable_baselines3.dqn.policies",
    )
    circuit = QuantumCircuit(3)
    load_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(
        scenarios,
        "resolve_backend_bundle",
        lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            backend="backend-object",
            coupling_edges=[(0, 1), (1, 2)],
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": load_calls.append(algorithm) or "agent-object",
    )
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: RoutingEpisodeSummary(
            initial_layout=None,
            final_layout=[0, 1, 2],
            steps_executed=1,
            total_reward=1.0,
            completed=True,
            truncated=False,
            total_swaps=0,
            gates_executed_count=2,
        ),
    )

    result = scenarios.run_rl_only_scenario(
        _make_request("RL_Only", rl_model_path=str(model_path))
    )

    assert load_calls == ["DQN"]
    assert result.notes == [
        "RL outputs are episode summaries, not final circuits.",
        "Legacy RL evaluation defaults were used because no run metadata sidecar was found.",
    ]


def test_run_rl_only_scenario_requires_rl_model_path_before_contract_resolution(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request_without_validation("RL_Only", rl_model_path=None)

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(
        scenarios,
        "resolve_backend_bundle",
        lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            backend="backend-object",
            coupling_edges=[(0, 1), (1, 2)],
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "resolve_routing_model_contract",
        lambda model_path: (_ for _ in ()).throw(AssertionError("contract resolution should not be reached")),
    )

    with pytest.raises(ValueError, match="rl_model_path is required"):
        scenarios.run_rl_only_scenario(request)


def test_load_agent_uses_public_rl_api(monkeypatch) -> None:
    from src.integration import scenarios

    load_calls = []

    class StubQuantumRLAgent:
        @staticmethod
        def load(path, env=None, algorithm="PPO"):
            load_calls.append((path, env, algorithm))
            return "agent-object"

    monkeypatch.setitem(
        sys.modules,
        "src.rl_module",
        SimpleNamespace(QuantumRLAgent=StubQuantumRLAgent),
    )

    agent = scenarios._load_agent(
        _make_request("RL_Only", rl_model_path="models/policy.zip"),
        algorithm="DQN",
    )

    assert agent == "agent-object"
    assert load_calls == [("models/policy.zip", None, "DQN")]


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
        swap_trace=[(0, 1), (1, 2)],
        executed_gate_trace=[("h", 0, 0), ("cx", 0, 1), ("cx", 1, 2), ("measure", 2, 2)],
    )
    eval_calls = []
    rebuild_calls = []
    post_calls = []

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
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": "agent-object",
    )
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: eval_calls.append(kwargs) or summary,
    )
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: rebuild_calls.append(kwargs) or ("routed-circuit", [1, 0, 2]),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda *args, **kwargs: post_calls.append((args, kwargs)) or _make_transpilation_result(),
    )

    result = scenarios.run_mo_rl_scenario(request)

    assert result.success is True
    assert result.selected_layout == [2, 0, 1]
    assert result.transpilation_metrics == {
        "backend_name": "fake_backend",
        "trans_depth": 12,
        "trans_two_qubit_gates": 4,
    }
    assert result.transpilation_artifact["artifact_version"] == "transpilation_result.v1"
    assert result.routing_summary is summary
    assert result.notes == [
        "MO+RL rebuilds the routed circuit from the RL swap trace before running Qiskit post-routing stages.",
        "Legacy RL evaluation defaults were used because no run metadata sidecar was found.",
    ]
    assert eval_calls == [
        {
            "circuit": circuit,
            "coupling_edges": [(0, 1), (1, 2)],
            "agent": "agent-object",
            "seed": 17,
            "initial_layout": [2, 0, 1],
            "frontier_mode": "sequential",
            "max_steps": scenarios._DEFAULT_RL_MAX_STEPS,
            "lookahead_window": scenarios._DEFAULT_RL_LOOKAHEAD_WINDOW,
            "masked": False,
        }
    ]
    assert rebuild_calls == [
        {
            "circuit": circuit,
            "coupling_edges": [(0, 1), (1, 2)],
            "initial_layout": [2, 0, 1],
            "swap_trace": [(0, 1), (1, 2)],
            "frontier_mode": "sequential",
            "executed_gate_trace": [("h", 0, 0), ("cx", 0, 1), ("cx", 1, 2), ("measure", 2, 2)],
        }
    ]
    assert post_calls == [
        (
            ("routed-circuit",),
            {
                "backend": "backend-object",
                "backend_name": "fake_backend",
                "optimization_level": 1,
                "seed": 17,
                "reference_circuit": circuit,
                "initial_layout": [2, 0, 1],
                "final_layout": [1, 0, 2],
            },
        )
    ]


def test_run_mo_rl_scenario_reuses_injected_layout_without_running_mo(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO+RL", rl_model_path="models/policy.zip")
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend=SimpleNamespace(num_qubits=3),
        coupling_edges=[(0, 1), (1, 2)],
    )
    selected_layout = [1, 2, 0]
    eval_calls = []
    rebuild_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout_quick",
        lambda circuit, backend, seed: (_ for _ in ()).throw(AssertionError("MO should not run when layout is injected")),
    )
    monkeypatch.setattr(
        scenarios,
        "select_layout_from_mo_result",
        lambda result, *, policy, objective_index=0: (_ for _ in ()).throw(
            AssertionError("layout selection should not run when layout is injected")
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": "agent-object",
    )
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: eval_calls.append(kwargs)
        or RoutingEpisodeSummary(
            initial_layout=list(kwargs["initial_layout"]),
            final_layout=[2, 1, 0],
            steps_executed=1,
            total_reward=1.0,
            completed=True,
            truncated=False,
            total_swaps=0,
            gates_executed_count=2,
            swap_trace=[],
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: rebuild_calls.append(kwargs) or ("routed-circuit", [2, 1, 0]),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda *args, **kwargs: _make_transpilation_result(),
    )

    result = scenarios.run_mo_rl_scenario(request, injected_layout=selected_layout)

    assert result.success is True
    assert result.selected_layout == selected_layout
    assert eval_calls[0]["initial_layout"] == selected_layout
    assert rebuild_calls[0]["initial_layout"] == selected_layout


def test_run_mo_rl_scenario_uses_injected_coupling_edges_for_campaign_internal_seam(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO+RL", rl_model_path="models/policy.zip")
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend=SimpleNamespace(num_qubits=4),
        coupling_edges=[(0, 1), (1, 2), (2, 3)],
    )
    injected_coupling_edges = [(0, 2), (2, 3)]
    injected_routing_graph = SimpleNamespace(
        mode="path_expanded_subgraph",
        node_count=3,
        edge_count=2,
        added_intermediate_qubits=[2],
        interacting_pair_count=4,
        fallback_reason="missing_path:0-3",
    )
    eval_calls = []
    rebuild_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": "agent-object",
    )
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: eval_calls.append(kwargs)
        or RoutingEpisodeSummary(
            initial_layout=list(kwargs["initial_layout"]),
            final_layout=[2, 1, 0],
            steps_executed=1,
            total_reward=1.0,
            completed=True,
            truncated=False,
            total_swaps=0,
            gates_executed_count=2,
            swap_trace=[],
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: rebuild_calls.append(kwargs) or ("routed-circuit", [2, 1, 0]),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda *args, **kwargs: _make_transpilation_result(),
    )

    result = scenarios.run_mo_rl_scenario(
        request,
        circuit=circuit,
        injected_layout=[1, 0, 2],
        injected_coupling_edges=injected_coupling_edges,
        injected_routing_graph=injected_routing_graph,
    )

    assert eval_calls[0]["coupling_edges"] == injected_coupling_edges
    assert rebuild_calls[0]["coupling_edges"] == injected_coupling_edges
    assert result.notes == [
        "MO+RL rebuilds the routed circuit from the RL swap trace before running Qiskit post-routing stages.",
        "Legacy RL evaluation defaults were used because no run metadata sidecar was found.",
        (
            "Routing graph: path_expanded_subgraph with 3 nodes, 2 edges, 1 added intermediate qubits, "
            "4 interacting pairs, fallback_reason=missing_path:0-3."
        ),
    ]


def test_run_mo_rl_scenario_uses_backend_coupling_edges_when_no_campaign_injection_is_provided(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO+RL", rl_model_path="models/policy.zip")
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend=SimpleNamespace(num_qubits=3),
        coupling_edges=[(0, 1), (1, 2)],
    )
    eval_calls = []
    rebuild_calls = []

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
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": "agent-object",
    )
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
            swap_trace=[],
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: rebuild_calls.append(kwargs) or ("routed-circuit", [1, 0, 2]),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda *args, **kwargs: _make_transpilation_result(),
    )

    scenarios.run_mo_rl_scenario(request)

    assert eval_calls[0]["coupling_edges"] == [(0, 1), (1, 2)]
    assert rebuild_calls[0]["coupling_edges"] == [(0, 1), (1, 2)]


def test_run_mo_rl_scenario_returns_controlled_result_when_routing_episode_is_truncated(monkeypatch) -> None:
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
        steps_executed=12,
        total_reward=-2.0,
        completed=False,
        truncated=True,
        total_swaps=1,
        gates_executed_count=1,
        swap_trace=[(0, 1)],
    )

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
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": "agent-object",
    )
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: summary,
    )
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("build_routed_circuit should not be called")),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("transpile_post_routing should not be called")),
    )

    result = scenarios.run_mo_rl_scenario(request)

    assert result.success is False
    assert result.selected_layout == [2, 0, 1]
    assert result.routing_summary is summary
    assert result.transpilation_metrics is None
    assert result.transpilation_artifact is None
    assert result.errors == ["MO+RL routing episode did not complete; skipping routed-circuit reconstruction."]
    assert result.notes == [
        "MO+RL routing episode was incomplete; routed-circuit reconstruction and post-routing transpilation were skipped.",
        "Legacy RL evaluation defaults were used because no run metadata sidecar was found.",
    ]


def test_run_mo_rl_scenario_rejects_mismatched_reconstructed_final_layout(monkeypatch) -> None:
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
        steps_executed=3,
        total_reward=2.0,
        completed=True,
        truncated=False,
        total_swaps=1,
        gates_executed_count=3,
        swap_trace=[(0, 1)],
    )

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
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": "agent-object",
    )
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: summary,
    )
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: ("routed-circuit", [0, 1, 2]),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("transpile_post_routing should not be called")),
    )

    with pytest.raises(ValueError, match="final_layout"):
        scenarios.run_mo_rl_scenario(request)


def test_run_mo_rl_scenario_rejects_mismatched_routing_summary_initial_layout(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO+RL", rl_model_path="models/policy.zip")
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend="backend-object",
        coupling_edges=[(0, 1), (1, 2)],
    )
    summary = RoutingEpisodeSummary(
        initial_layout=[0, 2, 1],
        final_layout=[1, 0, 2],
        steps_executed=3,
        total_reward=2.0,
        completed=True,
        truncated=False,
        total_swaps=1,
        gates_executed_count=3,
        swap_trace=[(0, 1)],
    )

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
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": "agent-object",
    )
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: summary,
    )
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("build_routed_circuit should not be called")),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("transpile_post_routing should not be called")),
    )

    with pytest.raises(ValueError, match="initial_layout"):
        scenarios.run_mo_rl_scenario(request)


def test_run_mo_rl_scenario_uses_saved_contract(monkeypatch, tmp_path) -> None:
    from src.integration import scenarios

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    save_run_metadata(
        tmp_path,
        build_run_metadata(
            mode="routing",
            algorithm="DQN",
            seed=17,
            frontier_mode="dag",
            lookahead_window=9,
            max_steps=333,
            basis_gates=None,
        ),
    )

    circuit = QuantumCircuit(3)
    request = _make_request("MO+RL", rl_model_path=str(model_path))
    load_calls = []
    eval_calls = []
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: ("routed-circuit", [1, 0, 2]),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda *args, **kwargs: _make_transpilation_result(),
    )

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(
        scenarios,
        "resolve_backend_bundle",
        lambda backend_name: SimpleNamespace(
            coupling_edges=[(0, 1)],
            backend=SimpleNamespace(num_qubits=3),
            backend_name=backend_name,
        ),
    )
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
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: eval_calls.append(kwargs)
        or RoutingEpisodeSummary(
            initial_layout=[2, 0, 1],
            final_layout=[1, 0, 2],
            steps_executed=1,
            total_reward=1.0,
            completed=True,
            truncated=False,
            total_swaps=0,
            gates_executed_count=2,
            swap_trace=[],
        ),
    )

    class StubQuantumRLAgent:
        @staticmethod
        def load(path, env=None, algorithm="PPO", **kwargs):
            load_calls.append({"path": path, "algorithm": algorithm, "env": env})
            return "agent-object"

    monkeypatch.setitem(
        sys.modules,
        "src.rl_module",
        SimpleNamespace(QuantumRLAgent=StubQuantumRLAgent),
    )

    result = scenarios.run_mo_rl_scenario(request)

    assert load_calls == [{"path": str(model_path), "algorithm": "DQN", "env": None}]
    assert eval_calls[0]["lookahead_window"] == 9
    assert eval_calls[0]["max_steps"] == 333
    assert eval_calls[0]["frontier_mode"] == "dag"
    assert eval_calls[0]["masked"] is False
    assert result.notes == [
        "MO+RL rebuilds the routed circuit from the RL swap trace before running Qiskit post-routing stages."
    ]


def test_run_mo_rl_scenario_forwards_masked_contract_fields(monkeypatch, tmp_path) -> None:
    from src.integration import scenarios

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    save_run_metadata(
        tmp_path,
        build_run_metadata(
            mode="routing",
            algorithm="MaskablePPO",
            seed=17,
            frontier_mode="dag",
            lookahead_window=9,
            max_steps=333,
            basis_gates=None,
            mask_semantics="frontier_restricted_edges.v1",
        ),
    )

    circuit = QuantumCircuit(3)
    request = _make_request("MO+RL", rl_model_path=str(model_path))
    load_calls = []
    eval_calls = []
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: ("routed-circuit", [1, 0, 2]),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda *args, **kwargs: _make_transpilation_result(),
    )

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(
        scenarios,
        "resolve_backend_bundle",
        lambda backend_name: SimpleNamespace(
            coupling_edges=[(0, 1)],
            backend=SimpleNamespace(num_qubits=3),
            backend_name=backend_name,
        ),
    )
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
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: eval_calls.append(kwargs)
        or RoutingEpisodeSummary(
            initial_layout=[2, 0, 1],
            final_layout=[1, 0, 2],
            steps_executed=1,
            total_reward=1.0,
            completed=True,
            truncated=False,
            total_swaps=0,
            gates_executed_count=2,
            swap_trace=[],
        ),
    )

    class StubQuantumRLAgent:
        @staticmethod
        def load(path, env=None, algorithm="PPO", **kwargs):
            load_calls.append({"path": path, "algorithm": algorithm, "env": env})
            return "agent-object"

    monkeypatch.setitem(
        sys.modules,
        "src.rl_module",
        SimpleNamespace(QuantumRLAgent=StubQuantumRLAgent),
    )

    result = scenarios.run_mo_rl_scenario(request)

    assert load_calls == [{"path": str(model_path), "algorithm": "MaskablePPO", "env": None}]
    assert eval_calls[0]["lookahead_window"] == 9
    assert eval_calls[0]["max_steps"] == 333
    assert eval_calls[0]["frontier_mode"] == "dag"
    assert eval_calls[0]["masked"] is True
    assert result.notes == [
        "MO+RL rebuilds the routed circuit from the RL swap trace before running Qiskit post-routing stages."
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
    rebuild_calls = []

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
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": "agent-object",
    )
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
            swap_trace=[],
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: rebuild_calls.append(kwargs) or ("routed-circuit", [1, 0, 2]),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda *args, **kwargs: _make_transpilation_result(),
    )

    result = scenarios.run_mo_rl_scenario(request)

    assert result.selected_layout == [2, 0, 1]
    assert all(type(entry) is int for entry in result.selected_layout)
    assert eval_calls[0]["initial_layout"] == [2, 0, 1]
    assert all(type(entry) is int for entry in eval_calls[0]["initial_layout"])
    assert rebuild_calls[0]["initial_layout"] == [2, 0, 1]
    assert all(type(entry) is int for entry in rebuild_calls[0]["initial_layout"])


def test_run_mo_rl_scenario_requires_rl_model_path_before_contract_resolution(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request_without_validation("MO+RL", rl_model_path=None)
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend=SimpleNamespace(num_qubits=3),
        coupling_edges=[(0, 1), (1, 2)],
    )

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
    monkeypatch.setattr(
        scenarios,
        "resolve_routing_model_contract",
        lambda model_path: (_ for _ in ()).throw(AssertionError("contract resolution should not be reached")),
    )

    with pytest.raises(ValueError, match="rl_model_path is required"):
        scenarios.run_mo_rl_scenario(request)


def test_mo_layout_validation_rejects_non_integer_entries_before_contract_resolution(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO+RL", rl_model_path="models/policy.zip")
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend=SimpleNamespace(num_qubits=4),
        coupling_edges=[(0, 1), (1, 2)],
    )

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
        lambda result, *, policy, objective_index=0: [2.9, 0, 1],
    )
    monkeypatch.setattr(
        scenarios,
        "resolve_routing_model_contract",
        lambda model_path: (_ for _ in ()).throw(AssertionError("contract resolution should not be reached")),
    )

    with pytest.raises(ValueError, match="integer"):
        scenarios.run_mo_rl_scenario(request)


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
        "load_circuit",
        lambda source_kind, **kwargs: circuit,
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
    monkeypatch.setattr(
        scenarios,
        "_load_agent",
        lambda request, *, algorithm="PPO": stub_agent,
    )
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
            swap_trace=[(0, 1)],
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: ("routed-circuit", [1, 0, 2]),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda *args, **kwargs: _make_transpilation_result(),
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
    assert payload["transpilation_metrics"] == {
        "backend_name": "fake_backend",
        "trans_depth": 12,
        "trans_two_qubit_gates": 4,
    }
    assert payload["routing_summary"] is not None
    assert payload["routing_summary"]["initial_layout"] == selected_layout
    assert payload["notes"] == [
        "MO+RL rebuilds the routed circuit from the RL swap trace before running Qiskit post-routing stages.",
        "Legacy RL evaluation defaults were used because no run metadata sidecar was found.",
    ]
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
        "load_circuit",
        lambda source_kind, **kwargs: (_ for _ in ()).throw(ValueError("Unknown circuit name: ghz_3")),
    )

    with pytest.raises(ValueError, match="Unknown circuit name"):
        scenarios._load_circuit(_make_request("Baseline"))


def test_load_circuit_uses_qiskit_interface_for_library_requests(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    calls = []

    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "load_circuit",
        lambda source_kind, **kwargs: calls.append((source_kind, kwargs)) or circuit,
    )

    loaded = scenarios._load_circuit(_make_request("Baseline"))

    assert loaded is circuit
    assert calls == [
        (
            "library",
            {
                "circuit_name": "ghz",
                "num_qubits": 3,
                "circuit_path": None,
                "circuit_format": "auto",
                "seed": 17,
            },
        )
    ]


def test_load_circuit_normalizes_library_request_name_before_qiskit_interface(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    calls = []

    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "load_circuit",
        lambda source_kind, **kwargs: calls.append((source_kind, kwargs)) or circuit,
    )

    loaded = scenarios._load_circuit(_make_request("Baseline"))

    assert loaded is circuit
    assert calls[0][1]["circuit_name"] == "ghz"
    assert calls[0][1]["num_qubits"] == 3


def test_load_circuit_uses_qiskit_interface_for_qasm_requests(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    calls = []

    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "load_circuit",
        lambda source_kind, **kwargs: calls.append((source_kind, kwargs)) or circuit,
    )

    loaded = scenarios._load_circuit(
        ScenarioRequest(
            scenario_name="Baseline",
            backend_name="fake_backend",
            seed=17,
            circuit_source=CircuitSource.QASM_FILE,
            circuit_path="circuits/example.qasm",
            circuit_format=CircuitFormat.QASM3,
        )
    )

    assert loaded is circuit
    assert calls == [
        (
            "qasm_file",
            {
                "circuit_name": None,
                "num_qubits": None,
                "circuit_path": "circuits/example.qasm",
                "circuit_format": "qasm3",
                "seed": 17,
            },
        )
    ]
