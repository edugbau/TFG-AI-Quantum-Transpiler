from dataclasses import fields

from src.integration.backend_adapter import BackendBundle, resolve_backend_bundle
from src.integration.synthetic_topology import SyntheticTopologySpec


def test_resolve_backend_bundle_returns_backend_and_coupling_edges() -> None:
    bundle = resolve_backend_bundle("fake_torino")

    assert isinstance(bundle, BackendBundle)
    assert bundle.backend_name == "fake_torino"
    assert bundle.backend is not None
    assert isinstance(bundle.coupling_edges, list)
    assert bundle.coupling_edges
    assert all(len(edge) == 2 for edge in bundle.coupling_edges[:5])
    assert [field.name for field in fields(BackendBundle)] == [
        "backend_name",
        "backend",
        "coupling_edges",
        "basis_gates",
    ]


def test_resolve_backend_bundle_uses_public_qiskit_interface_functions(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    backend = object()

    def fake_get_backend(backend_name: str) -> object:
        calls.append(("get_backend", backend_name))
        return backend

    def fake_get_coupling_edges(received_backend: object) -> list[tuple[int, int]]:
        calls.append(("get_coupling_edges", received_backend))
        return [(0, 1), (1, 2)]

    def fake_get_basis_gates(received_backend: object) -> list[str]:
        calls.append(("get_basis_gates", received_backend))
        return ["cx", "rz", "sx"]

    monkeypatch.setattr("src.integration.backend_adapter.qiskit_interface.get_backend", fake_get_backend)
    monkeypatch.setattr(
        "src.integration.backend_adapter.qiskit_interface.get_coupling_edges",
        fake_get_coupling_edges,
    )
    monkeypatch.setattr(
        "src.integration.backend_adapter.qiskit_interface.get_basis_gates",
        fake_get_basis_gates,
    )

    bundle = resolve_backend_bundle("fake_backend")

    assert bundle == BackendBundle(
        backend_name="fake_backend",
        backend=backend,
        coupling_edges=[(0, 1), (1, 2)],
        basis_gates=["cx", "rz", "sx"],
    )
    assert calls == [
        ("get_backend", "fake_backend"),
        ("get_coupling_edges", backend),
        ("get_basis_gates", backend),
    ]


def test_resolve_backend_bundle_builds_synthetic_backend_without_get_backend(monkeypatch) -> None:
    get_backend_calls: list[str] = []
    synthetic_topology = SyntheticTopologySpec(shape="line", num_qubits=3)

    monkeypatch.setattr(
        "src.integration.backend_adapter.qiskit_interface.get_backend",
        lambda backend_name: get_backend_calls.append(backend_name),
    )

    bundle = resolve_backend_bundle(
        synthetic_topology.backend_name,
        synthetic_topology=synthetic_topology,
    )

    assert get_backend_calls == []
    assert bundle.backend_name == "synthetic_line_3q"
    assert bundle.backend.num_qubits == 3
    assert set(bundle.coupling_edges) == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert bundle.basis_gates == ["id", "rz", "sx", "x", "cx"]
