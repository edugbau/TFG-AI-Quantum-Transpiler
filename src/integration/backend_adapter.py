from dataclasses import dataclass

import src.qiskit_interface as qiskit_interface


@dataclass(slots=True)
class BackendBundle:
    backend_name: str
    backend: object
    coupling_edges: list[tuple[int, int]]
    basis_gates: list[str]


def resolve_backend_bundle(backend_name: str) -> BackendBundle:
    backend = qiskit_interface.get_backend(backend_name)
    coupling_edges = list(qiskit_interface.get_coupling_edges(backend))
    basis_gates = qiskit_interface.get_basis_gates(backend)
    return BackendBundle(
        backend_name=backend_name,
        backend=backend,
        coupling_edges=coupling_edges,
        basis_gates=basis_gates,
    )
