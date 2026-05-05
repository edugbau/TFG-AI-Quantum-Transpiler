from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from qiskit import QuantumCircuit


@dataclass(frozen=True, slots=True)
class RoutingSubgraph:
    mode: str
    coupling_edges: list[tuple[int, int]]
    node_count: int
    edge_count: int
    added_intermediate_qubits: list[int]
    interacting_pair_count: int
    fallback_reason: str | None = None


def _normalize_edges(coupling_edges) -> list[tuple[int, int]]:
    normalized = {tuple(sorted((int(left), int(right)))) for left, right in coupling_edges}
    return sorted(normalized)


def _extract_interacting_logical_pairs(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for instruction in circuit.data:
        logical_qargs = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if len(logical_qargs) != 2:
            continue
        pair = tuple(sorted((int(logical_qargs[0]), int(logical_qargs[1]))))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _build_adjacency(coupling_edges: list[tuple[int, int]]) -> dict[int, list[int]]:
    adjacency: dict[int, set[int]] = {}
    for left, right in coupling_edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return {node: sorted(neighbors) for node, neighbors in adjacency.items()}


def _shortest_path(adjacency: dict[int, list[int]], start: int, goal: int) -> list[int] | None:
    if start == goal:
        return [start]

    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor in visited:
                continue
            next_path = [*path, neighbor]
            if neighbor == goal:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))
    return None


def _count_nodes(coupling_edges: list[tuple[int, int]]) -> int:
    return len({node for edge in coupling_edges for node in edge})


def _build_fallback(
    backend_edges: list[tuple[int, int]],
    *,
    interacting_pair_count: int,
    fallback_reason: str,
) -> RoutingSubgraph:
    return RoutingSubgraph(
        mode="full_backend_fallback",
        coupling_edges=backend_edges,
        node_count=_count_nodes(backend_edges),
        edge_count=len(backend_edges),
        added_intermediate_qubits=[],
        interacting_pair_count=interacting_pair_count,
        fallback_reason=fallback_reason,
    )


def build_path_expanded_subgraph(*, circuit: QuantumCircuit, selected_layout: list[int], coupling_edges) -> RoutingSubgraph:
    backend_edges = _normalize_edges(coupling_edges)
    interacting_pairs = _extract_interacting_logical_pairs(circuit)
    if not interacting_pairs:
        return _build_fallback(
            backend_edges,
            interacting_pair_count=0,
            fallback_reason="no_interacting_pairs",
        )

    adjacency = _build_adjacency(backend_edges)
    chosen_edges: set[tuple[int, int]] = set()
    interacting_physical_qubits: set[int] = set()

    for logical_left, logical_right in interacting_pairs:
        physical_left = int(selected_layout[logical_left])
        physical_right = int(selected_layout[logical_right])
        interacting_physical_qubits.update((physical_left, physical_right))
        if physical_left == physical_right:
            continue

        direct_edge = tuple(sorted((physical_left, physical_right)))
        if direct_edge in backend_edges:
            chosen_edges.add(direct_edge)
            continue

        path = _shortest_path(adjacency, physical_left, physical_right)
        if path is None:
            return _build_fallback(
                backend_edges,
                interacting_pair_count=len(interacting_pairs),
                fallback_reason=f"missing_path:{physical_left}-{physical_right}",
            )

        for left, right in zip(path, path[1:]):
            chosen_edges.add(tuple(sorted((left, right))))

    derived_edges = sorted(chosen_edges)
    if not derived_edges:
        return _build_fallback(
            backend_edges,
            interacting_pair_count=len(interacting_pairs),
            fallback_reason="empty_derived_graph",
        )

    derived_nodes = {node for edge in derived_edges for node in edge}
    added_nodes = sorted(node for node in derived_nodes if node not in interacting_physical_qubits)
    return RoutingSubgraph(
        mode="path_expanded_subgraph",
        coupling_edges=derived_edges,
        node_count=len(derived_nodes),
        edge_count=len(derived_edges),
        added_intermediate_qubits=added_nodes,
        interacting_pair_count=len(interacting_pairs),
        fallback_reason=None,
    )
