import src.mo_module as mo_module
import src.qiskit_interface as qiskit_interface
from src.integration.backend_adapter import resolve_backend_bundle
from src.integration.contracts import ScenarioRequest, ScenarioResult
from src.integration.layout_policy import select_layout_from_mo_result
from src.integration.routing_evaluator import evaluate_routing_episode


_RL_NOTE = "RL outputs are episode summaries, not final circuits."
_DEFAULT_RL_MAX_STEPS = 256
_DEFAULT_RL_LOOKAHEAD_WINDOW = 4


def _load_circuit(request: ScenarioRequest):
    circuits = qiskit_interface.circuits_from_library(
        num_qubits=request.num_qubits,
        seed=request.seed,
    )
    try:
        return circuits[request.circuit_name]
    except KeyError as exc:
        raise ValueError(f"Unknown circuit name: {request.circuit_name}") from exc


def _load_agent(request: ScenarioRequest):
    if request.rl_model_path is None:
        raise ValueError("rl_model_path is required for RL scenarios")
    from src.rl_module import QuantumRLAgent

    return QuantumRLAgent.load(request.rl_model_path, env=None)


def _require_scenario(request: ScenarioRequest, expected: str) -> None:
    if request.scenario_name != expected:
        raise ValueError(
            f"scenario_name must be {expected!r} for this runner; got {request.scenario_name!r}"
        )


def _validate_selected_layout(layout: list[int], num_qubits: int, backend) -> list[int]:
    if len(layout) != num_qubits:
        raise ValueError("selected layout width must match request.num_qubits")
    if any(entry < 0 for entry in layout):
        raise ValueError("selected layout cannot contain negative entries")
    if len(set(layout)) != len(layout):
        raise ValueError("selected layout cannot contain duplicated entries")
    backend_num_qubits = getattr(backend, "num_qubits", None)
    if backend_num_qubits is not None and any(entry >= backend_num_qubits for entry in layout):
        raise ValueError("selected layout contains physical qubits outside the backend range")
    return [int(entry) for entry in layout]


def _run_mo(request: ScenarioRequest, circuit, backend_bundle):
    if request.mo_use_quick:
        return mo_module.optimize_layout_quick(
            circuit=circuit,
            backend=backend_bundle.backend,
            seed=request.seed,
        )
    return mo_module.optimize_layout(
        circuit=circuit,
        backend=backend_bundle.backend,
        backend_name=backend_bundle.backend_name,
    )


def run_baseline_scenario(request: ScenarioRequest) -> ScenarioResult:
    _require_scenario(request, "Baseline")
    circuit = _load_circuit(request)
    backend_bundle = resolve_backend_bundle(request.backend_name)
    transpilation = qiskit_interface.transpile_circuit(
        circuit=circuit,
        backend=backend_bundle.backend,
        backend_name=backend_bundle.backend_name,
        seed=request.seed,
    )
    return ScenarioResult(
        scenario_name=request.scenario_name,
        circuit_name=request.circuit_name,
        backend_name=request.backend_name,
        seed=request.seed,
        success=True,
        selected_layout=None,
        transpilation_metrics=transpilation.to_dict(),
        routing_summary=None,
    )


def run_mo_only_scenario(request: ScenarioRequest) -> ScenarioResult:
    _require_scenario(request, "MO_Only")
    circuit = _load_circuit(request)
    backend_bundle = resolve_backend_bundle(request.backend_name)
    mo_result = _run_mo(request, circuit, backend_bundle)
    selected_layout = _validate_selected_layout(
        select_layout_from_mo_result(
            mo_result,
            policy=request.layout_policy,
            objective_index=request.mo_objective_index,
        ),
        request.num_qubits,
        backend_bundle.backend,
    )
    transpilation = qiskit_interface.transpile_with_custom_layout(
        circuit=circuit,
        layout=selected_layout,
        backend=backend_bundle.backend,
        backend_name=backend_bundle.backend_name,
        seed=request.seed,
    )
    return ScenarioResult(
        scenario_name=request.scenario_name,
        circuit_name=request.circuit_name,
        backend_name=request.backend_name,
        seed=request.seed,
        success=True,
        selected_layout=selected_layout,
        transpilation_metrics=transpilation.to_dict(),
        routing_summary=None,
    )


def run_rl_only_scenario(request: ScenarioRequest) -> ScenarioResult:
    _require_scenario(request, "RL_Only")
    circuit = _load_circuit(request)
    backend_bundle = resolve_backend_bundle(request.backend_name)
    agent = _load_agent(request)
    routing_summary = evaluate_routing_episode(
        circuit=circuit,
        coupling_edges=backend_bundle.coupling_edges,
        agent=agent,
        seed=request.seed,
        initial_layout=request.initial_layout,
        max_steps=_DEFAULT_RL_MAX_STEPS,
        lookahead_window=_DEFAULT_RL_LOOKAHEAD_WINDOW,
    )
    return ScenarioResult(
        scenario_name=request.scenario_name,
        circuit_name=request.circuit_name,
        backend_name=request.backend_name,
        seed=request.seed,
        success=True,
        selected_layout=None,
        transpilation_metrics=None,
        routing_summary=routing_summary,
        notes=[_RL_NOTE],
    )


def run_mo_rl_scenario(request: ScenarioRequest) -> ScenarioResult:
    _require_scenario(request, "MO+RL")
    circuit = _load_circuit(request)
    backend_bundle = resolve_backend_bundle(request.backend_name)
    mo_result = _run_mo(request, circuit, backend_bundle)
    selected_layout = _validate_selected_layout(
        select_layout_from_mo_result(
            mo_result,
            policy=request.layout_policy,
            objective_index=request.mo_objective_index,
        ),
        request.num_qubits,
        backend_bundle.backend,
    )
    agent = _load_agent(request)
    routing_summary = evaluate_routing_episode(
        circuit=circuit,
        coupling_edges=backend_bundle.coupling_edges,
        agent=agent,
        seed=request.seed,
        initial_layout=selected_layout,
        max_steps=_DEFAULT_RL_MAX_STEPS,
        lookahead_window=_DEFAULT_RL_LOOKAHEAD_WINDOW,
    )
    return ScenarioResult(
        scenario_name=request.scenario_name,
        circuit_name=request.circuit_name,
        backend_name=request.backend_name,
        seed=request.seed,
        success=True,
        selected_layout=selected_layout,
        transpilation_metrics=None,
        routing_summary=routing_summary,
        notes=[_RL_NOTE],
    )
