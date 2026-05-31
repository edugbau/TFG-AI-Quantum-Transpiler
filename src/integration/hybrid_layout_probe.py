from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import src.qiskit_interface as qiskit_interface

from src.integration.routing_evaluator import build_routed_circuit, evaluate_routing_episode
from src.integration.routing_subgraph import RoutingSubgraph, build_path_expanded_subgraph
from src.rl_module.routing_mask import (
    DEFAULT_NEW_MASK_SEMANTICS,
    RoutingMaskConfig,
    resolve_routing_mask_config,
)


HYBRID_LAYOUT_PROBE_FILENAME = "hybrid_layout_probe.json"
HYBRID_LAYOUT_PROBE_ARTIFACT_VERSION = "hybrid_layout_probe.v1"
HYBRID_LAYOUT_PROBE_SCORE_FIELDS = ("trans_cnot_equivalent", "trans_depth", "total_swaps")


@dataclass(frozen=True, slots=True)
class HybridLayoutProbeAttempt:
    source: str
    layout: list[int]
    pareto_index: int | None
    status: str
    routing_graph: dict[str, Any] | None
    completed: bool
    truncated: bool
    truncation_reason: str | None
    termination_reason: str | None
    total_swaps: int | None
    score: list[float | int] | None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class HybridLayoutProbeResult:
    artifact_version: str
    selector: str
    score_fields: list[str]
    candidates: list[HybridLayoutProbeAttempt]
    qiskit_control: HybridLayoutProbeAttempt
    selected_layout: list[int]
    selected_score: list[float | int] | None
    valid_candidate_count: int
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _select_sabre_action(env, obs):
    del obs
    return env.select_sabre_heuristic_action()


def _routing_graph_payload(routing_graph: RoutingSubgraph) -> dict[str, Any]:
    return asdict(routing_graph)


def _evaluate_layout(
    *,
    source: str,
    pareto_index: int | None,
    layout: list[int],
    circuit,
    backend_bundle,
    seed: int,
    frontier_mode: str,
    max_steps: int,
    lookahead_window: int,
    routing_mask_config: RoutingMaskConfig,
) -> HybridLayoutProbeAttempt:
    routing_graph_payload = None
    try:
        routing_graph = build_path_expanded_subgraph(
            circuit=circuit,
            selected_layout=layout,
            coupling_edges=backend_bundle.coupling_edges,
        )
        routing_graph_payload = _routing_graph_payload(routing_graph)
        routing_summary = evaluate_routing_episode(
            circuit=circuit,
            coupling_edges=list(routing_graph.coupling_edges),
            agent=None,
            seed=seed,
            initial_layout=list(layout),
            frontier_mode=frontier_mode,
            max_steps=max_steps,
            lookahead_window=lookahead_window,
            masked=True,
            mask_semantics=DEFAULT_NEW_MASK_SEMANTICS,
            routing_mask_config=routing_mask_config,
            action_selector=_select_sabre_action,
        )
        if not routing_summary.completed or routing_summary.truncated:
            return HybridLayoutProbeAttempt(
                source=source,
                layout=list(layout),
                pareto_index=pareto_index,
                status="incomplete_routing",
                routing_graph=routing_graph_payload,
                completed=bool(routing_summary.completed),
                truncated=bool(routing_summary.truncated),
                truncation_reason=routing_summary.truncation_reason,
                termination_reason=routing_summary.termination_reason,
                total_swaps=routing_summary.total_swaps,
                score=None,
            )

        routed_circuit, final_layout = build_routed_circuit(
            circuit=circuit,
            coupling_edges=list(routing_graph.coupling_edges),
            initial_layout=list(layout),
            swap_trace=routing_summary.swap_trace,
            frontier_mode=frontier_mode,
            executed_gate_trace=routing_summary.executed_gate_trace,
        )
        transpilation_result = qiskit_interface.transpile_post_routing(
            routed_circuit,
            backend=backend_bundle.backend,
            backend_name=backend_bundle.backend_name,
            optimization_level=1,
            seed=seed,
            reference_circuit=circuit,
            initial_layout=list(layout),
            final_layout=final_layout,
        )
        metrics = transpilation_result.transpiled_metrics
        score = [
            float(metrics.cnot_equivalent),
            int(metrics.depth),
            int(routing_summary.total_swaps),
        ]
        return HybridLayoutProbeAttempt(
            source=source,
            layout=list(layout),
            pareto_index=pareto_index,
            status="valid_candidate",
            routing_graph=routing_graph_payload,
            completed=True,
            truncated=False,
            truncation_reason=None,
            termination_reason=None,
            total_swaps=routing_summary.total_swaps,
            score=score,
        )
    except Exception as exc:
        return HybridLayoutProbeAttempt(
            source=source,
            layout=list(layout),
            pareto_index=pareto_index,
            status="evaluation_error",
            routing_graph=routing_graph_payload,
            completed=False,
            truncated=False,
            truncation_reason=None,
            termination_reason=None,
            total_swaps=None,
            score=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _deduplicate_pareto_layouts(mo_result) -> list[tuple[int, list[int]]]:
    candidates: list[tuple[int, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for index, layout in enumerate(mo_result.pareto_layouts):
        normalized_layout = [int(entry) for entry in layout]
        signature = tuple(normalized_layout)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append((index, normalized_layout))
    return candidates


def _score_key(attempt: HybridLayoutProbeAttempt) -> tuple[float, int, int]:
    if attempt.score is None:
        raise ValueError("Cannot rank a hybrid layout probe attempt without a score")
    return float(attempt.score[0]), int(attempt.score[1]), int(attempt.score[2])


def write_hybrid_layout_probe_artifact(path: Path | str, result: HybridLayoutProbeResult) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return artifact_path


def select_hybrid_probe_layout(
    *,
    circuit,
    mo_result,
    backend_bundle,
    qiskit_initial_layout: list[int],
    seed: int,
    frontier_mode: str,
    max_steps: int,
    lookahead_window: int,
    routing_mask_config: RoutingMaskConfig | dict[str, Any] | None = None,
    artifact_path: Path | str | None = None,
) -> HybridLayoutProbeResult:
    resolved_mask_config = resolve_routing_mask_config(
        routing_mask_config,
        num_qubits=circuit.num_qubits,
    )
    candidates = [
        _evaluate_layout(
            source="mo_pareto",
            pareto_index=index,
            layout=layout,
            circuit=circuit,
            backend_bundle=backend_bundle,
            seed=seed,
            frontier_mode=frontier_mode,
            max_steps=max_steps,
            lookahead_window=lookahead_window,
            routing_mask_config=resolved_mask_config,
        )
        for index, layout in _deduplicate_pareto_layouts(mo_result)
    ]
    qiskit_control = _evaluate_layout(
        source="qiskit_control",
        pareto_index=None,
        layout=[int(entry) for entry in qiskit_initial_layout],
        circuit=circuit,
        backend_bundle=backend_bundle,
        seed=seed,
        frontier_mode=frontier_mode,
        max_steps=max_steps,
        lookahead_window=lookahead_window,
        routing_mask_config=resolved_mask_config,
    )
    valid_candidates = [candidate for candidate in candidates if candidate.score is not None]
    fallback_reason = None
    if valid_candidates:
        winner = min(valid_candidates, key=_score_key)
        selected_layout = list(winner.layout)
        selected_score = list(winner.score) if winner.score is not None else None
    else:
        fallback_reason = "no_completed_mo_probe_candidates"
        selected_layout = [int(entry) for entry in mo_result.get_compromise_layout()]
        selected_score = None

    result = HybridLayoutProbeResult(
        artifact_version=HYBRID_LAYOUT_PROBE_ARTIFACT_VERSION,
        selector="hybrid_probe",
        score_fields=list(HYBRID_LAYOUT_PROBE_SCORE_FIELDS),
        candidates=candidates,
        qiskit_control=qiskit_control,
        selected_layout=selected_layout,
        selected_score=selected_score,
        valid_candidate_count=len(valid_candidates),
        fallback_reason=fallback_reason,
    )
    if artifact_path is not None:
        write_hybrid_layout_probe_artifact(artifact_path, result)
    return result
