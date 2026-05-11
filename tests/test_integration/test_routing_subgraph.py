from qiskit import QuantumCircuit


def test_build_path_expanded_subgraph_closes_shortest_paths_for_interacting_pairs() -> None:
    from src.integration.routing_subgraph import build_path_expanded_subgraph

    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    circuit.cx(1, 2)
    selected_layout = [4, 0, 2]
    backend_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]

    result = build_path_expanded_subgraph(
        circuit=circuit,
        selected_layout=selected_layout,
        coupling_edges=backend_edges,
    )

    assert result.mode == "path_expanded_subgraph"
    assert result.coupling_edges == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert result.node_count == 5
    assert result.edge_count == 4
    assert result.interacting_pair_count == 2
    assert result.added_intermediate_qubits == [1, 3]
    assert result.fallback_reason is None


def test_build_path_expanded_subgraph_ignores_non_interacting_layout_pairs() -> None:
    from src.integration.routing_subgraph import build_path_expanded_subgraph

    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    selected_layout = [5, 2, 4, 0]
    backend_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    result = build_path_expanded_subgraph(
        circuit=circuit,
        selected_layout=selected_layout,
        coupling_edges=backend_edges,
    )

    assert result.mode == "path_expanded_subgraph"
    assert result.coupling_edges == [(2, 3), (3, 4), (4, 5)]
    assert result.node_count == 4
    assert result.edge_count == 3
    assert result.interacting_pair_count == 1
    assert result.added_intermediate_qubits == [3, 4]
    assert result.fallback_reason is None


def test_build_path_expanded_subgraph_resolves_shortest_path_ties_deterministically() -> None:
    from src.integration.routing_subgraph import build_path_expanded_subgraph

    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    selected_layout = [0, 3]
    backend_edges = [(0, 1), (1, 3), (0, 2), (2, 3)]

    result = build_path_expanded_subgraph(
        circuit=circuit,
        selected_layout=selected_layout,
        coupling_edges=backend_edges,
    )

    assert result.mode == "path_expanded_subgraph"
    assert result.coupling_edges == [(0, 1), (1, 3)]
    assert result.node_count == 3
    assert result.edge_count == 2
    assert result.added_intermediate_qubits == [1]
    assert result.interacting_pair_count == 1
    assert result.fallback_reason is None


def test_build_path_expanded_subgraph_falls_back_to_full_backend_when_no_two_qubit_pairs_exist() -> None:
    from src.integration.routing_subgraph import build_path_expanded_subgraph

    circuit = QuantumCircuit(3)
    circuit.h(0)
    selected_layout = [7, 3, 5]
    backend_edges = [(3, 4), (4, 5), (5, 6), (6, 7)]

    result = build_path_expanded_subgraph(
        circuit=circuit,
        selected_layout=selected_layout,
        coupling_edges=backend_edges,
    )

    assert result.mode == "full_backend_fallback"
    assert result.coupling_edges == [(3, 4), (4, 5), (5, 6), (6, 7)]
    assert result.node_count == 5
    assert result.edge_count == 4
    assert result.interacting_pair_count == 0
    assert result.added_intermediate_qubits == []
    assert result.fallback_reason == "no_interacting_pairs"


def test_build_path_expanded_subgraph_falls_back_to_full_backend_when_interacting_pair_has_no_path() -> None:
    from src.integration.routing_subgraph import build_path_expanded_subgraph

    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    selected_layout = [0, 3]
    backend_edges = [(0, 1), (2, 3)]

    result = build_path_expanded_subgraph(
        circuit=circuit,
        selected_layout=selected_layout,
        coupling_edges=backend_edges,
    )

    assert result.mode == "full_backend_fallback"
    assert result.coupling_edges == [(0, 1), (2, 3)]
    assert result.node_count == 4
    assert result.edge_count == 2
    assert result.interacting_pair_count == 1
    assert result.added_intermediate_qubits == []
    assert result.fallback_reason == "missing_path:0-3"
