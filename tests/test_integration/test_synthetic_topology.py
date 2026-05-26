import pytest

from src.integration.synthetic_topology import (
    SYNTHETIC_BASIS_GATES,
    SyntheticTopologySpec,
)
from src.qiskit_interface import extract_backend_info, get_basis_gates, get_coupling_edges


@pytest.mark.parametrize(
    ("spec", "backend_name"),
    [
        (SyntheticTopologySpec(shape="full", num_qubits=4), "synthetic_full_4q"),
        (SyntheticTopologySpec(shape="line", num_qubits=4), "synthetic_line_4q"),
        (SyntheticTopologySpec(shape="ring", num_qubits=4), "synthetic_ring_4q"),
        (SyntheticTopologySpec(shape="t", num_qubits=8), "synthetic_t_8q"),
        (SyntheticTopologySpec(shape="grid", rows=2, cols=3), "synthetic_grid_2x3"),
        (SyntheticTopologySpec(shape="heavy_hex", distance=3), "synthetic_heavy_hex_d3"),
        (SyntheticTopologySpec(shape="heavy_square", distance=3), "synthetic_heavy_square_d3"),
        (
            SyntheticTopologySpec(shape="hexagonal_lattice", rows=2, cols=3),
            "synthetic_hexagonal_lattice_2x3",
        ),
    ],
)
def test_synthetic_topology_spec_builds_qiskit_coupling_maps(spec, backend_name) -> None:
    coupling_map = spec.build_coupling_map()
    backend = spec.build_backend()

    assert spec.backend_name == backend_name
    assert spec.physical_qubits == coupling_map.size()
    assert backend.name == backend_name
    assert backend.num_qubits == coupling_map.size()
    assert backend.basis_gates == SYNTHETIC_BASIS_GATES
    assert backend.topology_metadata["topology_source"] == "synthetic"
    assert coupling_map.get_edges() is not None


def test_qiskit_interface_extracts_synthetic_backend_info_without_calibrations() -> None:
    backend = SyntheticTopologySpec(shape="line", num_qubits=3).build_backend()

    info = extract_backend_info(backend)

    assert get_basis_gates(backend) == list(SYNTHETIC_BASIS_GATES)
    assert set(get_coupling_edges(backend)) == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert info.name == "synthetic_line_3q"
    assert info.backend_kind == "synthetic"
    assert info.gate_errors_2q == {}
    assert info.to_summary_dict()["topology"]["shape"] == "line"


@pytest.mark.parametrize(
    ("num_qubits", "undirected_edges"),
    [
        (5, {(0, 1), (1, 2), (1, 3), (3, 4)}),
        (8, {(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7)}),
        (
            11,
            {
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (3, 7),
                (7, 8),
                (8, 9),
                (9, 10),
            },
        ),
        (
            12,
            {
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (3, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
            },
        ),
    ],
)
def test_balanced_t_topology_builds_expected_bidirectional_edges(num_qubits, undirected_edges) -> None:
    spec = SyntheticTopologySpec(shape="t", num_qubits=num_qubits)

    coupling_edges = set(spec.build_coupling_map().get_edges())
    expected_edges = {
        directed_edge
        for left, right in undirected_edges
        for directed_edge in ((left, right), (right, left))
    }

    assert spec.backend_name == f"synthetic_t_{num_qubits}q"
    assert spec.physical_qubits == num_qubits
    assert spec.to_metadata()["shape"] == "t"
    assert spec.to_metadata()["num_qubits"] == num_qubits
    assert coupling_edges == expected_edges


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": "line", "num_qubits": 0},
        {"shape": "t"},
        {"shape": "t", "num_qubits": 4},
        {"shape": "grid", "rows": 2, "cols": 0},
        {"shape": "heavy_hex", "distance": 2},
        {"shape": "unknown", "num_qubits": 3},
        {"shape": "line", "num_qubits": 3, "basis_gates": ("rz", "sx")},
    ],
)
def test_synthetic_topology_spec_rejects_invalid_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        SyntheticTopologySpec(**kwargs)
