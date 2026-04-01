import math

import pytest
from qiskit import QuantumCircuit

from src.rl_module.synthesis_clifford import (
    CliffordSynthesisState,
    clifford_distance_from_identity,
    clifford_to_observation_arrays,
    remap_logical_circuit_to_physical,
)
from src.rl_module.synthesis_primitives import (
    SynthesisPrimitive,
    build_clifford_primitive_catalog,
    primitive_to_circuit,
)


def test_build_clifford_catalog_uses_native_two_qubit_gate_only():
    catalog = build_clifford_primitive_catalog(
        num_physical_qubits=3,
        coupling_map=[(1, 0), (0, 1), (1, 2)],
        basis_gates=["cz", "rz", "sx", "x"],
    )

    x_primitives = {
        primitive.physical_qargs
        for primitive in catalog
        if primitive.gate_name == "x"
    }
    sx_primitives = {
        primitive.physical_qargs
        for primitive in catalog
        if primitive.gate_name == "sx"
    }
    two_qubit = [primitive for primitive in catalog if len(primitive.physical_qargs) == 2]

    assert x_primitives == {(0,), (1,), (2,)}
    assert sx_primitives == {(0,), (1,), (2,)}
    assert {primitive.gate_name for primitive in two_qubit} == {"cz"}
    assert {primitive.physical_qargs for primitive in two_qubit} == {(0, 1), (1, 2)}


def test_build_clifford_catalog_quantizes_rz_to_clifford_angles():
    catalog = build_clifford_primitive_catalog(
        num_physical_qubits=1,
        coupling_map=[],
        basis_gates=["rz", "sx", "x"],
    )

    rz_angles = {
        primitive.params[0]
        for primitive in catalog
        if primitive.gate_name == "rz"
    }

    assert rz_angles == {math.pi / 2, math.pi, 3 * math.pi / 2}


def test_primitive_to_circuit_builds_ecr_on_selected_edge():
    catalog = build_clifford_primitive_catalog(
        num_physical_qubits=2,
        coupling_map=[(0, 1)],
        basis_gates=["ecr", "rz", "sx", "x"],
    )
    primitive = next(
        primitive
        for primitive in catalog
        if primitive.gate_name == "ecr" and primitive.physical_qargs == (0, 1)
    )

    circuit = primitive_to_circuit(primitive, num_physical_qubits=2)

    assert circuit.count_ops() == {"ecr": 1}


def test_primitive_to_circuit_builds_parameterized_rz_on_selected_qubit():
    primitive = SynthesisPrimitive("rz", (1,), (math.pi,), cost=0.0)

    circuit = primitive_to_circuit(primitive, num_physical_qubits=3)

    assert circuit.count_ops() == {"rz": 1}
    instruction = circuit.data[0]
    assert circuit.find_bit(instruction.qubits[0]).index == 1
    assert instruction.operation.params == [math.pi]


def test_primitive_to_circuit_rejects_non_clifford_rz_angle():
    primitive = SynthesisPrimitive("rz", (0,), (math.pi / 3,), cost=0.0)

    with pytest.raises(ValueError, match="Clifford"):
        primitive_to_circuit(primitive, num_physical_qubits=1)


def test_build_clifford_catalog_uses_supported_two_qubit_gate_priority_for_v1():
    catalog = build_clifford_primitive_catalog(
        num_physical_qubits=2,
        coupling_map=[(0, 1)],
        basis_gates=["cx", "ecr", "cz", "rz", "sx", "x"],
    )

    two_qubit = [primitive for primitive in catalog if len(primitive.physical_qargs) == 2]

    assert {primitive.gate_name for primitive in two_qubit} == {"cz"}
    assert {primitive.physical_qargs for primitive in two_qubit} == {(0, 1)}


def test_catalog_requires_supported_two_qubit_basis_when_edges_exist():
    try:
        build_clifford_primitive_catalog(
            num_physical_qubits=2,
            coupling_map=[(0, 1)],
            basis_gates=["rz", "sx", "x"],
        )
    except ValueError as exc:
        assert "two-qubit" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError when no supported two-qubit basis is available")


@pytest.mark.parametrize(
    ("coupling_map", "message_fragment"),
    [
        ([(0, 0)], "self-loop"),
        ([(0, 2)], "out of range"),
    ],
)
def test_build_clifford_catalog_rejects_invalid_coupling_map_endpoints(
    coupling_map, message_fragment
):
    with pytest.raises(ValueError, match=message_fragment):
        build_clifford_primitive_catalog(
            num_physical_qubits=2,
            coupling_map=coupling_map,
            basis_gates=["cz", "rz", "sx", "x"],
        )


def test_primitive_to_circuit_rejects_rz_without_required_parameter():
    primitive = SynthesisPrimitive("rz", (0,))

    with pytest.raises(ValueError, match="rz.*parameter"):
        primitive_to_circuit(primitive, num_physical_qubits=1)


@pytest.mark.parametrize(
    ("primitive", "num_physical_qubits", "message_fragment"),
    [
        (SynthesisPrimitive("x", ()), 1, "x"),
        (SynthesisPrimitive("x", (2,)), 1, "out of range"),
        (SynthesisPrimitive("cx", (0,)), 1, "cx"),
        (SynthesisPrimitive("cx", (0, 0)), 1, "distinct"),
        (SynthesisPrimitive("sx", (0,), (math.pi,)), 1, "sx"),
    ],
)
def test_primitive_to_circuit_rejects_malformed_public_primitives(
    primitive, num_physical_qubits, message_fragment
):
    with pytest.raises(ValueError, match=message_fragment):
        primitive_to_circuit(primitive, num_physical_qubits=num_physical_qubits)


def test_primitive_to_circuit_rejects_unsupported_gate_name():
    primitive = SynthesisPrimitive("iswap", (0, 1))

    with pytest.raises(ValueError, match="unsupported gate"):
        primitive_to_circuit(primitive, num_physical_qubits=2)


def test_remap_logical_circuit_to_physical_uses_layout_indices():
    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)

    physical = remap_logical_circuit_to_physical(
        target_circuit=circuit,
        layout=[2, 0],
        num_physical_qubits=3,
    )

    assert physical.num_qubits == 3
    assert physical.count_ops() == {"cz": 1}
    instruction = physical.data[0]
    assert [physical.find_bit(qubit).index for qubit in instruction.qubits] == [2, 0]


@pytest.mark.parametrize(
    ("layout", "message_fragment"),
    [
        ([0], "length"),
        ([1, 1], "duplicate"),
        ([0, 3], "out of range"),
        ([0.0, 1], "integer"),
        ([True, 1], "integer"),
    ],
)
def test_remap_logical_circuit_to_physical_rejects_malformed_layouts(
    layout, message_fragment
):
    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)

    with pytest.raises(ValueError, match=message_fragment):
        remap_logical_circuit_to_physical(
            target_circuit=circuit,
            layout=layout,
            num_physical_qubits=3,
        )


def test_clifford_synthesis_state_is_complete_after_matching_sequence():
    circuit = QuantumCircuit(1)
    circuit.x(0)

    state = CliffordSynthesisState.from_target_circuit(
        target_circuit=circuit,
        layout=[0],
        num_physical_qubits=1,
    )

    assert state.is_complete() is False

    primitive = SynthesisPrimitive("x", (0,), cost=1.0)
    state.apply_primitive(primitive)

    assert state.is_complete() is True
    assert state.residual_distance() == 0


def test_clifford_synthesis_state_tracks_non_self_inverse_residual_until_exact_match():
    circuit = QuantumCircuit(1)
    circuit.rz(math.pi / 2, 0)

    state = CliffordSynthesisState.from_target_circuit(
        target_circuit=circuit,
        layout=[0],
        num_physical_qubits=1,
    )

    state.apply_primitive(SynthesisPrimitive("sx", (0,), cost=1.0))

    assert state.is_complete() is False
    assert state.residual_distance() > 0

    exact_state = CliffordSynthesisState.from_target_circuit(
        target_circuit=circuit,
        layout=[0],
        num_physical_qubits=1,
    )
    exact_state.apply_primitive(SynthesisPrimitive("rz", (0,), (math.pi / 2,), cost=0.0))

    assert exact_state.is_complete() is True
    assert exact_state.residual_distance() == 0


def test_clifford_distance_from_identity_is_zero_for_identity_and_positive_after_gate():
    state = CliffordSynthesisState.from_target_circuit(
        target_circuit=QuantumCircuit(1),
        layout=[0],
        num_physical_qubits=1,
    )

    assert clifford_distance_from_identity(state.residual()) == 0

    state.apply_primitive(SynthesisPrimitive("x", (0,), cost=1.0))

    assert clifford_distance_from_identity(state.residual()) > 0


def test_clifford_to_observation_arrays_match_physical_qubit_count():
    state = CliffordSynthesisState.from_target_circuit(
        target_circuit=QuantumCircuit(2),
        layout=[1, 0],
        num_physical_qubits=3,
    )

    symplectic, phase = clifford_to_observation_arrays(state.residual())

    assert symplectic.shape == (4 * 3 * 3,)
    assert phase.shape == (2 * 3,)


def test_from_target_circuit_rejects_non_clifford_target():
    circuit = QuantumCircuit(1)
    circuit.rz(0.25, 0)

    with pytest.raises(ValueError, match="Clifford"):
        CliffordSynthesisState.from_target_circuit(
            target_circuit=circuit,
            layout=[0],
            num_physical_qubits=1,
        )
