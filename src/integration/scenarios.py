from inspect import Parameter, signature
from numbers import Integral

import src.mo_module as mo_module
import src.qiskit_interface as qiskit_interface
from src.integration.backend_adapter import resolve_backend_bundle
from src.integration.contracts import ScenarioRequest, ScenarioResult
from src.integration.layout_policy import select_layout_from_mo_result
from src.integration.rl_model_contract import resolve_routing_model_contract
from src.integration.routing_evaluator import build_routed_circuit, evaluate_routing_episode
from src.qiskit_interface.transpiler import NAMED_BASELINES


_RL_NOTE = "RL_Only rebuilds the routed circuit from the RL swap trace before running Qiskit post-routing stages."
_RL_INCOMPLETE_NOTE = (
    "RL_Only routing episode was incomplete; routed-circuit reconstruction and post-routing transpilation were skipped."
)
_MO_RL_NOTE = "MO+RL rebuilds the routed circuit from the RL swap trace before running Qiskit post-routing stages."
_MO_RL_INCOMPLETE_NOTE = (
    "MO+RL routing episode was incomplete; routed-circuit reconstruction and post-routing transpilation were skipped."
)
_RL_METADATA_FALLBACK_NOTE = (
    "Legacy RL evaluation defaults were used because no run metadata sidecar was found."
)
_DEFAULT_RL_MAX_STEPS = 256
_DEFAULT_RL_LOOKAHEAD_WINDOW = 4


def _normalize_library_request_name(circuit_name: str | None) -> str | None:
    if circuit_name is None:
        return None
    normalized_name, _, maybe_qubits = circuit_name.rpartition("_")
    if normalized_name and maybe_qubits.isdigit():
        return normalized_name
    return circuit_name


def _get_result_circuit_name(request: ScenarioRequest, circuit) -> str:
    if request.circuit_name is not None:
        return request.circuit_name
    metadata = getattr(circuit, "metadata", None) or {}
    resolved_circuit_name = metadata.get("resolved_circuit_name")
    if isinstance(resolved_circuit_name, str) and resolved_circuit_name:
        return resolved_circuit_name
    circuit_name = getattr(circuit, "name", None)
    if isinstance(circuit_name, str) and circuit_name:
        return circuit_name
    if request.circuit_path is not None:
        return request.circuit_path
    raise ValueError("Unable to resolve circuit name for scenario result")


def _get_request_num_qubits(request: ScenarioRequest, circuit) -> int:
    if request.num_qubits is not None:
        return request.num_qubits
    return int(circuit.num_qubits)


def _build_metric_summary(row: dict, prefix: str) -> dict[str, int | float]:
    summary: dict[str, int | float] = {}
    for key, value in row.items():
        if key.startswith(prefix):
            summary[key.removeprefix(prefix)] = value
    return summary


def _validate_named_baseline_result(
    row: dict,
    artifact: dict[str, object] | None,
    *,
    expected_baseline_name: str,
    expected_initial_layout: list[int] | None,
) -> None:
    row_baseline_name = row.get("baseline_name")
    if row_baseline_name != expected_baseline_name:
        raise ValueError("baseline_name in named baseline row does not match expected baseline")

    row_initial_layout = row.get("initial_layout")
    if row_initial_layout != expected_initial_layout:
        raise ValueError("initial_layout in named baseline row does not match expected initial_layout")

    if artifact is None:
        raise ValueError("named baseline result did not include transpilation artifact")

    artifact_baseline_name = artifact.get("baseline_name")
    if artifact_baseline_name != expected_baseline_name:
        raise ValueError("baseline_name in named baseline artifact does not match expected baseline")

    transpilation = artifact.get("transpilation")
    if not isinstance(transpilation, dict):
        raise ValueError("named baseline artifact must include transpilation details")

    artifact_initial_layout = transpilation.get("initial_layout")
    if artifact_initial_layout != expected_initial_layout:
        raise ValueError(
            "initial_layout in named baseline artifact does not match expected initial_layout"
        )


def _run_named_baseline_with_artifact(
    request: ScenarioRequest,
    circuit,
    baseline_name: str,
    *,
    layout: list[int] | None = None,
):
    if request.synthetic_topology is not None:
        backend_bundle = _resolve_backend_bundle_for_request(request)
        baseline_config = NAMED_BASELINES[baseline_name]
        optimization_level = int(baseline_config["optimization_level"])
        if baseline_config["kind"] == "custom_layout":
            if layout is None:
                raise ValueError(f"Baseline '{baseline_name}' requires a layout")
            result = qiskit_interface.transpile_with_custom_layout(
                circuit=circuit,
                layout=layout,
                backend=backend_bundle.backend,
                backend_name=backend_bundle.backend_name,
                optimization_level=optimization_level,
                seed=request.seed,
            )
        else:
            result = qiskit_interface.transpile_circuit(
                circuit=circuit,
                backend=backend_bundle.backend,
                backend_name=backend_bundle.backend_name,
                optimization_level=optimization_level,
                seed=request.seed,
            )
        result.baseline_name = baseline_name
        row = {"circuit_name": circuit.name or "unnamed"}
        row.update(result.to_dict())
        artifact = result.to_artifact_dict()
    else:
        transpilation_rows = qiskit_interface.run_named_baseline(
            baseline_name,
            circuit=circuit,
            layout=layout,
            backend_names=[request.backend_name],
            seed=request.seed,
            include_artifact=True,
        )
        row, artifact = transpilation_rows[0]
    row = dict(row)
    artifact = dict(artifact)
    _validate_named_baseline_result(
        row,
        artifact,
        expected_baseline_name=baseline_name,
        expected_initial_layout=layout,
    )
    return row, artifact


def _callable_accepts_kwarg(callable_obj, kwarg_name: str) -> bool:
    try:
        parameters = signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        (parameter.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD) and parameter.name == kwarg_name)
        or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _resolve_backend_bundle_for_request(request: ScenarioRequest):
    if request.synthetic_topology is not None and _callable_accepts_kwarg(resolve_backend_bundle, "synthetic_topology"):
        return resolve_backend_bundle(request.backend_name, synthetic_topology=request.synthetic_topology)
    return resolve_backend_bundle(request.backend_name)


def _validate_request_initial_layout(request: ScenarioRequest, circuit) -> None:
    if request.initial_layout is None:
        return
    if len(request.initial_layout) != circuit.num_qubits:
        raise ValueError("initial_layout length must match num_qubits")


def _load_circuit(request: ScenarioRequest):
    circuit_name = request.circuit_name
    if request.circuit_source.value == "library":
        circuit_name = _normalize_library_request_name(circuit_name)
    return qiskit_interface.load_circuit(
        request.circuit_source.value,
        circuit_name=circuit_name,
        num_qubits=request.num_qubits,
        circuit_path=request.circuit_path,
        circuit_format=request.circuit_format.value,
        seed=request.seed,
    )


def _load_agent(request: ScenarioRequest, *, algorithm: str = "PPO"):
    if request.rl_model_path is None:
        raise ValueError("rl_model_path is required for RL scenarios")
    from src.rl_module import QuantumRLAgent

    return QuantumRLAgent.load(request.rl_model_path, env=None, algorithm=algorithm)


def _build_rl_notes(metadata_source: str) -> list[str]:
    notes = [_RL_NOTE]
    if metadata_source == "defaults":
        notes.append(_RL_METADATA_FALLBACK_NOTE)
    return notes


def _build_mo_rl_notes(metadata_source: str) -> list[str]:
    notes = [_MO_RL_NOTE]
    if metadata_source == "defaults":
        notes.append(_RL_METADATA_FALLBACK_NOTE)
    return notes


def _build_incomplete_mo_rl_notes(metadata_source: str) -> list[str]:
    notes = [_MO_RL_INCOMPLETE_NOTE]
    if metadata_source == "defaults":
        notes.append(_RL_METADATA_FALLBACK_NOTE)
    return notes


def _build_incomplete_rl_notes(metadata_source: str) -> list[str]:
    notes = [_RL_INCOMPLETE_NOTE]
    if metadata_source == "defaults":
        notes.append(_RL_METADATA_FALLBACK_NOTE)
    return notes


def _build_routing_graph_note(routing_graph) -> str:
    note_parts = [
        f"Routing graph: {routing_graph.mode} with {routing_graph.node_count} nodes",
        f"{routing_graph.edge_count} edges",
        f"{len(routing_graph.added_intermediate_qubits)} added intermediate qubits",
    ]
    interacting_pair_count = getattr(routing_graph, "interacting_pair_count", None)
    if interacting_pair_count is not None:
        note_parts.append(f"{interacting_pair_count} interacting pairs")
    fallback_reason = getattr(routing_graph, "fallback_reason", None)
    if fallback_reason:
        note_parts.append(f"fallback_reason={fallback_reason}")
    return f"{', '.join(note_parts)}."


def _validate_reconstructed_final_layout(
    routing_summary,
    reconstructed_final_layout: list[int],
) -> list[int]:
    normalized_reconstructed_layout = [int(entry) for entry in reconstructed_final_layout]
    expected_final_layout = routing_summary.final_layout
    if expected_final_layout is None:
        return normalized_reconstructed_layout

    normalized_expected_layout = [int(entry) for entry in expected_final_layout]
    if normalized_expected_layout != normalized_reconstructed_layout:
        raise ValueError(
            "reconstructed final_layout does not match routing_summary.final_layout"
        )
    return normalized_reconstructed_layout


def _validate_routing_summary_initial_layout(
    routing_summary,
    selected_layout: list[int],
) -> list[int]:
    normalized_selected_layout = [int(entry) for entry in selected_layout]
    reported_initial_layout = routing_summary.initial_layout
    if reported_initial_layout is None:
        return normalized_selected_layout

    normalized_reported_layout = [int(entry) for entry in reported_initial_layout]
    if normalized_reported_layout != normalized_selected_layout:
        raise ValueError(
            "routing_summary.initial_layout does not match the selected initial_layout"
        )
    return normalized_selected_layout


def _require_scenario(request: ScenarioRequest, expected: str) -> None:
    if request.scenario_name != expected:
        raise ValueError(
            f"scenario_name must be {expected!r} for this runner; got {request.scenario_name!r}"
        )


def _validate_selected_layout(layout: list[int], num_qubits: int, backend) -> list[int]:
    if len(layout) != num_qubits:
        raise ValueError("selected layout width must match request.num_qubits")
    if any(not isinstance(entry, Integral) or isinstance(entry, bool) for entry in layout):
        raise ValueError("selected layout must contain integer physical qubits")
    if any(entry < 0 for entry in layout):
        raise ValueError("selected layout cannot contain negative entries")
    if len(set(layout)) != len(layout):
        raise ValueError("selected layout cannot contain duplicated entries")
    backend_num_qubits = getattr(backend, "num_qubits", None)
    if backend_num_qubits is not None and any(entry >= backend_num_qubits for entry in layout):
        raise ValueError("selected layout contains physical qubits outside the backend range")
    return [int(entry) for entry in layout]


def _append_routing_graph_note(notes: list[str], routing_graph) -> list[str]:
    if routing_graph is not None:
        notes.append(_build_routing_graph_note(routing_graph))
    return notes


def _run_rl_reconstruction_flow(
    *,
    request: ScenarioRequest,
    circuit,
    backend_bundle,
    selected_layout: list[int],
    coupling_edges: list[tuple[int, int]],
    contract,
    agent,
    complete_notes: list[str],
    incomplete_notes: list[str],
    incomplete_error: str,
    injected_routing_graph=None,
) -> ScenarioResult:
    routing_summary = evaluate_routing_episode(
        circuit=circuit,
        coupling_edges=coupling_edges,
        agent=agent,
        seed=request.seed,
        initial_layout=selected_layout,
        frontier_mode=contract.frontier_mode,
        max_steps=contract.max_steps,
        lookahead_window=contract.lookahead_window,
        masked=contract.masked,
        mask_semantics=contract.mask_semantics,
    )
    if not routing_summary.completed or routing_summary.truncated:
        return ScenarioResult(
            scenario_name=request.scenario_name,
            circuit_name=_get_result_circuit_name(request, circuit),
            backend_name=request.backend_name,
            seed=request.seed,
            success=False,
            selected_layout=selected_layout,
            transpilation_metrics=None,
            transpilation_artifact=None,
            routing_summary=routing_summary,
            errors=[incomplete_error],
            notes=_append_routing_graph_note(incomplete_notes, injected_routing_graph),
        )

    selected_layout = _validate_routing_summary_initial_layout(routing_summary, selected_layout)
    routed_circuit, final_layout = build_routed_circuit(
        circuit=circuit,
        coupling_edges=coupling_edges,
        initial_layout=selected_layout,
        swap_trace=routing_summary.swap_trace,
        frontier_mode=contract.frontier_mode,
        executed_gate_trace=routing_summary.executed_gate_trace,
    )
    final_layout = _validate_reconstructed_final_layout(routing_summary, final_layout)
    transpilation_result = qiskit_interface.transpile_post_routing(
        routed_circuit,
        backend=backend_bundle.backend,
        backend_name=backend_bundle.backend_name,
        optimization_level=1,
        seed=request.seed,
        reference_circuit=circuit,
        initial_layout=selected_layout,
        final_layout=final_layout,
    )
    return ScenarioResult(
        scenario_name=request.scenario_name,
        circuit_name=_get_result_circuit_name(request, circuit),
        backend_name=request.backend_name,
        seed=request.seed,
        success=True,
        selected_layout=selected_layout,
        transpilation_metrics=transpilation_result.to_dict(),
        transpilation_artifact=transpilation_result.to_artifact_dict(),
        routing_summary=routing_summary,
        notes=_append_routing_graph_note(complete_notes, injected_routing_graph),
    )


def _run_mo(request: ScenarioRequest, circuit, backend_bundle):
    if request.mo_use_quick:
        return mo_module.optimize_layout_quick(
            circuit=circuit,
            backend=backend_bundle.backend,
            population_size=request.mo_population_size,
            n_generations=request.mo_n_generations,
            seed=request.seed,
        )
    return mo_module.optimize_layout(
        circuit=circuit,
        backend=backend_bundle.backend,
        backend_name=backend_bundle.backend_name,
        config=mo_module.OptimizerConfig(
            population_size=request.mo_population_size,
            n_generations=request.mo_n_generations,
            seed=request.seed,
        ),
    )


def optimize_mo_layouts(request: ScenarioRequest, *, circuit=None, backend_bundle=None):
    if request.scenario_name not in {"MO_Only", "MO+RL"}:
        raise ValueError("MO optimization requires an MO_Only or MO+RL request")
    if circuit is None:
        circuit = _load_circuit(request)
    if backend_bundle is None:
        backend_bundle = _resolve_backend_bundle_for_request(request)
    return _run_mo(request, circuit, backend_bundle)


def select_mo_layout(request: ScenarioRequest, mo_result, *, circuit, backend_bundle=None) -> list[int]:
    if request.scenario_name not in {"MO_Only", "MO+RL"}:
        raise ValueError("MO layout selection requires an MO_Only or MO+RL request")
    if backend_bundle is None:
        backend_bundle = _resolve_backend_bundle_for_request(request)
    return _validate_selected_layout(
        select_layout_from_mo_result(
            mo_result,
            policy=request.layout_policy,
            objective_index=request.mo_objective_index,
        ),
        _get_request_num_qubits(request, circuit),
        backend_bundle.backend,
    )


def run_baseline_scenario(request: ScenarioRequest, *, circuit=None) -> ScenarioResult:
    _require_scenario(request, "Baseline")
    if circuit is None:
        circuit = _load_circuit(request)
    transpilation_metrics, transpilation_artifact = _run_named_baseline_with_artifact(
        request,
        circuit,
        "qiskit_level_1",
    )
    return ScenarioResult(
        scenario_name=request.scenario_name,
        circuit_name=_get_result_circuit_name(request, circuit),
        backend_name=request.backend_name,
        seed=request.seed,
        success=True,
        selected_layout=None,
        transpilation_metrics=transpilation_metrics,
        transpilation_artifact=transpilation_artifact,
        routing_summary=None,
    )


def run_mo_only_scenario(
    request: ScenarioRequest,
    *,
    circuit=None,
    injected_layout: list[int] | None = None,
    mo_result=None,
) -> ScenarioResult:
    _require_scenario(request, "MO_Only")
    if circuit is None:
        circuit = _load_circuit(request)
    backend_bundle = _resolve_backend_bundle_for_request(request)
    if injected_layout is None:
        if mo_result is None:
            mo_result = optimize_mo_layouts(request, circuit=circuit, backend_bundle=backend_bundle)
        selected_layout = select_mo_layout(
            request,
            mo_result,
            circuit=circuit,
            backend_bundle=backend_bundle,
        )
    else:
        selected_layout = _validate_selected_layout(
            injected_layout,
            _get_request_num_qubits(request, circuit),
            backend_bundle.backend,
        )
    transpilation_metrics, transpilation_artifact = _run_named_baseline_with_artifact(
        request,
        circuit,
        "custom_layout_level_1",
        layout=selected_layout,
    )
    return ScenarioResult(
        scenario_name=request.scenario_name,
        circuit_name=_get_result_circuit_name(request, circuit),
        backend_name=request.backend_name,
        seed=request.seed,
        success=True,
        selected_layout=selected_layout,
        transpilation_metrics=transpilation_metrics,
        transpilation_artifact=transpilation_artifact,
        routing_summary=None,
    )


def run_rl_only_scenario(
    request: ScenarioRequest,
    *,
    circuit=None,
    injected_layout: list[int] | None = None,
    injected_coupling_edges: list[tuple[int, int]] | None = None,
    injected_routing_graph=None,
) -> ScenarioResult:
    _require_scenario(request, "RL_Only")
    if request.rl_model_path is None:
        raise ValueError("rl_model_path is required for RL scenarios")
    if circuit is None:
        circuit = _load_circuit(request)
    backend_bundle = _resolve_backend_bundle_for_request(request)
    if injected_layout is not None:
        selected_layout = _validate_selected_layout(
            injected_layout,
            _get_request_num_qubits(request, circuit),
            backend_bundle.backend,
        )
    elif request.initial_layout is not None:
        selected_layout = _validate_selected_layout(
            request.initial_layout,
            _get_request_num_qubits(request, circuit),
            backend_bundle.backend,
        )
    else:
        selected_layout = _validate_selected_layout(
            list(range(_get_request_num_qubits(request, circuit))),
            _get_request_num_qubits(request, circuit),
            backend_bundle.backend,
        )
    contract = resolve_routing_model_contract(request.rl_model_path)
    agent = _load_agent(request, algorithm=contract.algorithm)
    coupling_edges = (
        list(injected_coupling_edges) if injected_coupling_edges is not None else backend_bundle.coupling_edges
    )
    return _run_rl_reconstruction_flow(
        request=request,
        circuit=circuit,
        backend_bundle=backend_bundle,
        selected_layout=selected_layout,
        coupling_edges=coupling_edges,
        contract=contract,
        agent=agent,
        complete_notes=_build_rl_notes(contract.metadata_source),
        incomplete_notes=_build_incomplete_rl_notes(contract.metadata_source),
        incomplete_error="RL_Only routing episode did not complete; skipping routed-circuit reconstruction.",
        injected_routing_graph=injected_routing_graph,
    )


def run_mo_rl_scenario(
    request: ScenarioRequest,
    *,
    circuit=None,
    injected_layout: list[int] | None = None,
    injected_coupling_edges: list[tuple[int, int]] | None = None,
    injected_routing_graph=None,
) -> ScenarioResult:
    _require_scenario(request, "MO+RL")
    if request.rl_model_path is None:
        raise ValueError("rl_model_path is required for RL scenarios")
    if circuit is None:
        circuit = _load_circuit(request)
    backend_bundle = _resolve_backend_bundle_for_request(request)
    if injected_layout is None:
        mo_result = optimize_mo_layouts(request, circuit=circuit, backend_bundle=backend_bundle)
        selected_layout = select_mo_layout(
            request,
            mo_result,
            circuit=circuit,
            backend_bundle=backend_bundle,
        )
    else:
        selected_layout = _validate_selected_layout(
            injected_layout,
            _get_request_num_qubits(request, circuit),
            backend_bundle.backend,
        )
    contract = resolve_routing_model_contract(request.rl_model_path)
    agent = _load_agent(request, algorithm=contract.algorithm)
    coupling_edges = (
        list(injected_coupling_edges) if injected_coupling_edges is not None else backend_bundle.coupling_edges
    )
    return _run_rl_reconstruction_flow(
        request=request,
        circuit=circuit,
        backend_bundle=backend_bundle,
        selected_layout=selected_layout,
        coupling_edges=coupling_edges,
        contract=contract,
        agent=agent,
        complete_notes=_build_mo_rl_notes(contract.metadata_source),
        incomplete_notes=_build_incomplete_mo_rl_notes(contract.metadata_source),
        incomplete_error="MO+RL routing episode did not complete; skipping routed-circuit reconstruction.",
        injected_routing_graph=injected_routing_graph,
    )
