from dataclasses import dataclass

import src.qiskit_interface as qiskit_interface
from src.integration.synthetic_topology import SyntheticTopologySpec


@dataclass(slots=True)
class BackendBundle:
    backend_name: str
    backend: object
    coupling_edges: list[tuple[int, int]]
    basis_gates: list[str]


def resolve_backend_bundle(
    backend_name: str,
    *,
    synthetic_topology: SyntheticTopologySpec | None = None,
) -> BackendBundle:
    backend = synthetic_topology.build_backend() if synthetic_topology is not None else qiskit_interface.get_backend(backend_name)
    resolved_backend_name = synthetic_topology.backend_name if synthetic_topology is not None else backend_name
    coupling_edges = list(qiskit_interface.get_coupling_edges(backend))
    basis_gates = qiskit_interface.get_basis_gates(backend)
    return BackendBundle(
        backend_name=resolved_backend_name,
        backend=backend,
        coupling_edges=coupling_edges,
        basis_gates=basis_gates,
    )
