# RL Clifford Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first trainable `mode="synthesis"` as a hardware-aware Clifford synthesis environment without changing the functional behavior of `routing`.

**Architecture:** Introduce a discrete primitive catalog and a pure Clifford residual engine in new `src/rl_module/` modules. Redesign `SynthesisStrategy` around those abstractions, keep `routing` on its current frontier path, and make `environment.py` dispatch to mode-specific logic while threading explicit `basis_gates` through environment creation, training, and GUI configuration.

**Tech Stack:** Python 3.10, pytest, Gymnasium, Stable-Baselines3, Qiskit 2.3.0, CustomTkinter.

---

## File Map

- Create: `src/rl_module/synthesis_primitives.py` - discrete hardware-aware Clifford primitive catalog and primitive-to-circuit helpers.
- Create: `src/rl_module/synthesis_clifford.py` - logical-to-physical target remapping, residual computation, tableau-distance helpers, and observation-array extraction.
- Create: `tests/test_rl_module/test_synthesis_clifford.py` - pure unit tests for the primitive catalog and Clifford residual engine.
- Modify: `src/rl_module/env_strategies.py` - redesign `SynthesisStrategy` around a discrete primitive catalog and a residual-based observation.
- Modify: `src/rl_module/environment.py` - add `basis_gates`, initialize synthesis state, run synthesis transitions, and keep routing stable.
- Modify: `src/rl_module/rewards.py` - redefine `SynthesisReward` around residual progress and primitive cost.
- Modify: `src/rl_module/training.py` - thread `basis_gates` through training and evaluation environments.
- Modify: `src/rl_module/gui/rl_gui.py` - add an explicit synthesis basis-profile selector and pass it to the environment.
- Modify: `src/rl_module/docs/internal_documentation.md` - document the new synthesis architecture and its fixed-layout Clifford scope.
- Modify: `src/rl_module/docs/synthesis_mode_status.md` - replace placeholder wording with the new current status and future-work boundary.
- Modify: `tests/test_rl_module/test_rl_module.py` - replace the old synthesis scaffolding assertions with residual-based environment and strategy tests while preserving routing regression coverage.

### Task 1: Primitive Catalog Core

**Files:**
- Create: `src/rl_module/synthesis_primitives.py`
- Create: `tests/test_rl_module/test_synthesis_clifford.py`

- [ ] **Step 1: Write the failing primitive-catalog tests**

```python
import math

from src.rl_module.synthesis_primitives import (
    SynthesisPrimitive,
    build_clifford_primitive_catalog,
    primitive_to_circuit,
)


def test_build_clifford_catalog_uses_native_two_qubit_gate_only():
    catalog = build_clifford_primitive_catalog(
        num_physical_qubits=3,
        coupling_map=[(0, 1), (1, 2)],
        basis_gates=["cz", "rz", "sx", "x"],
    )

    two_qubit = [primitive for primitive in catalog if len(primitive.physical_qargs) == 2]

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
```

- [ ] **Step 2: Run the new primitive tests and verify they fail first**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_synthesis_clifford.py -v`

Expected: FAIL with `ModuleNotFoundError` or import failures because `src/rl_module/synthesis_primitives.py` does not exist yet.

- [ ] **Step 3: Implement the primitive catalog and circuit builder**

Create `src/rl_module/synthesis_primitives.py` with:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from qiskit import QuantumCircuit


SUPPORTED_TWO_Q_GATES = ("cz", "ecr", "cx")
CLIFFORD_RZ_ANGLES = (math.pi / 2, math.pi, 3 * math.pi / 2)


@dataclass(frozen=True)
class SynthesisPrimitive:
    gate_name: str
    physical_qargs: tuple[int, ...]
    params: tuple[float, ...] = ()
    cost: float = 1.0


def _normalized_edges(coupling_map: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted({tuple(sorted(edge)) for edge in coupling_map})


def _detect_two_qubit_basis(basis_gates: Sequence[str]) -> str | None:
    for gate_name in SUPPORTED_TWO_Q_GATES:
        if gate_name in basis_gates:
            return gate_name
    return None


def build_clifford_primitive_catalog(
    num_physical_qubits: int,
    coupling_map: list[tuple[int, int]],
    basis_gates: Sequence[str],
) -> list[SynthesisPrimitive]:
    basis_set = set(basis_gates)
    catalog: list[SynthesisPrimitive] = []

    if "x" in basis_set:
        catalog.extend(
            SynthesisPrimitive("x", (physical_qubit,), cost=1.0)
            for physical_qubit in range(num_physical_qubits)
        )

    if "sx" in basis_set:
        catalog.extend(
            SynthesisPrimitive("sx", (physical_qubit,), cost=1.0)
            for physical_qubit in range(num_physical_qubits)
        )

    if "rz" in basis_set:
        for physical_qubit in range(num_physical_qubits):
            for angle in CLIFFORD_RZ_ANGLES:
                catalog.append(
                    SynthesisPrimitive("rz", (physical_qubit,), (angle,), cost=0.0)
                )

    two_qubit_gate = _detect_two_qubit_basis(basis_gates)
    normalized_edges = _normalized_edges(coupling_map)
    if normalized_edges and two_qubit_gate is None:
        raise ValueError(
            "A supported two-qubit basis gate (cx, cz, or ecr) is required when the coupling map has edges."
        )

    if two_qubit_gate is not None:
        catalog.extend(
            SynthesisPrimitive(two_qubit_gate, edge, cost=3.0)
            for edge in normalized_edges
        )

    return catalog


def primitive_to_circuit(
    primitive: SynthesisPrimitive,
    num_physical_qubits: int,
) -> QuantumCircuit:
    circuit = QuantumCircuit(num_physical_qubits)
    qargs = list(primitive.physical_qargs)

    if primitive.gate_name == "rz":
        circuit.rz(primitive.params[0], qargs[0])
        return circuit

    getattr(circuit, primitive.gate_name)(*qargs)
    return circuit
```

- [ ] **Step 4: Re-run the primitive tests and verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_synthesis_clifford.py -v`

Expected: PASS (`4 passed`).

- [ ] **Step 5: Commit the primitive-catalog task**

```bash
git add src/rl_module/synthesis_primitives.py tests/test_rl_module/test_synthesis_clifford.py
git commit -m "feat: add Clifford synthesis primitive catalog"
```

### Task 2: Clifford Residual Engine

**Files:**
- Create: `src/rl_module/synthesis_clifford.py`
- Modify: `tests/test_rl_module/test_synthesis_clifford.py`

- [ ] **Step 1: Extend the test file with failing residual-engine tests**

Append to `tests/test_rl_module/test_synthesis_clifford.py`:

```python
from qiskit import QuantumCircuit

from src.rl_module.synthesis_primitives import SynthesisPrimitive

from src.rl_module.synthesis_clifford import (
    CliffordSynthesisState,
    clifford_distance_from_identity,
    clifford_to_observation_arrays,
    remap_logical_circuit_to_physical,
)


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
```

- [ ] **Step 2: Run the residual tests and verify they fail first**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_synthesis_clifford.py -v`

Expected: FAIL because `src/rl_module/synthesis_clifford.py` does not exist yet.

- [ ] **Step 3: Implement the residual engine and physical target remapping**

Create `src/rl_module/synthesis_clifford.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford

from .synthesis_primitives import SynthesisPrimitive, primitive_to_circuit


def identity_clifford(num_qubits: int) -> Clifford:
    return Clifford(QuantumCircuit(num_qubits))


def remap_logical_circuit_to_physical(
    target_circuit: QuantumCircuit,
    layout: list[int] | np.ndarray,
    num_physical_qubits: int,
) -> QuantumCircuit:
    physical_circuit = QuantumCircuit(num_physical_qubits, target_circuit.num_clbits)

    for instruction in target_circuit.data:
        logical_qargs = [target_circuit.find_bit(qubit).index for qubit in instruction.qubits]
        mapped_qargs = [physical_circuit.qubits[int(layout[index])] for index in logical_qargs]
        mapped_cargs = [
            physical_circuit.clbits[target_circuit.find_bit(clbit).index]
            for clbit in instruction.clbits
        ]
        physical_circuit.append(instruction.operation, mapped_qargs, mapped_cargs)

    return physical_circuit


def clifford_to_observation_arrays(clifford: Clifford) -> tuple[np.ndarray, np.ndarray]:
    symplectic = clifford.symplectic_matrix.astype(np.int8).reshape(-1).astype(np.int32)
    phase = clifford.phase.astype(np.int8).astype(np.int32)
    return symplectic, phase


def clifford_distance_from_identity(clifford: Clifford) -> int:
    identity = identity_clifford(clifford.num_qubits)
    symplectic_delta = np.bitwise_xor(
        clifford.symplectic_matrix.astype(np.int8),
        identity.symplectic_matrix.astype(np.int8),
    )
    phase_delta = np.bitwise_xor(
        clifford.phase.astype(np.int8),
        identity.phase.astype(np.int8),
    )
    return int(symplectic_delta.sum() + phase_delta.sum())


@dataclass
class CliffordSynthesisState:
    target: Clifford
    current: Clifford

    @classmethod
    def from_target_circuit(
        cls,
        target_circuit: QuantumCircuit,
        layout: list[int] | np.ndarray,
        num_physical_qubits: int,
    ) -> "CliffordSynthesisState":
        physical_target = remap_logical_circuit_to_physical(
            target_circuit=target_circuit,
            layout=layout,
            num_physical_qubits=num_physical_qubits,
        )
        try:
            target = Clifford.from_circuit(physical_target)
        except QiskitError as exc:
            raise ValueError(
                "mode='synthesis' v1 solo soporta circuitos Clifford; el target no puede convertirse a Clifford."
            ) from exc
        return cls(
            target=target,
            current=identity_clifford(num_physical_qubits),
        )

    def residual(self) -> Clifford:
        return self.current.adjoint().compose(self.target)

    def residual_distance(self) -> int:
        return clifford_distance_from_identity(self.residual())

    def is_complete(self) -> bool:
        return self.residual_distance() == 0

    def apply_primitive(self, primitive: SynthesisPrimitive) -> None:
        primitive_clifford = Clifford.from_circuit(
            primitive_to_circuit(primitive, self.current.num_qubits)
        )
        self.current = self.current.compose(primitive_clifford)
```

- [ ] **Step 4: Re-run the pure synthesis-core tests and verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_synthesis_clifford.py -v`

Expected: PASS (`9 passed`).

- [ ] **Step 5: Commit the residual-engine task**

```bash
git add src/rl_module/synthesis_clifford.py tests/test_rl_module/test_synthesis_clifford.py
git commit -m "feat: add Clifford residual synthesis core"
```

### Task 3: Redesign `SynthesisStrategy`

**Files:**
- Modify: `src/rl_module/env_strategies.py`
- Modify: `tests/test_rl_module/test_rl_module.py`

- [ ] **Step 1: Replace the old synthesis-strategy tests with failing discrete-catalog tests**

Replace the synthesis-strategy section in `tests/test_rl_module/test_rl_module.py` with:

```python
import gymnasium as gym
import pytest

class TestSynthesisStrategy:
    """Tests de la estrategia de síntesis Clifford hardware-aware."""

    def test_synthesis_strategy_requires_basis_gates(self, linear_coupling_3q):
        with pytest.raises(ValueError, match="basis_gates"):
            SynthesisStrategy(
                num_qubits=3,
                num_physical_qubits=3,
                coupling_map=linear_coupling_3q,
                lookahead_window=5,
                basis_gates=None,
            )

    def test_synthesis_strategy_uses_discrete_catalog_action_space(self, linear_coupling_3q):
        strategy = SynthesisStrategy(
            num_qubits=3,
            num_physical_qubits=3,
            coupling_map=linear_coupling_3q,
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )

        action_space = strategy.get_action_space()

        assert isinstance(action_space, gym.spaces.Discrete)
        assert action_space.n == len(strategy.primitives)

    def test_synthesis_strategy_decode_action_returns_primitive_payload(self, linear_coupling_3q):
        strategy = SynthesisStrategy(
            num_qubits=3,
            num_physical_qubits=3,
            coupling_map=linear_coupling_3q,
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )

        action_info = strategy.decode_action(0)

        assert action_info["type"] == "gate"
        assert "primitive" in action_info
        assert "primitive_index" in action_info

    def test_synthesis_observation_space_tracks_residual_arrays(self, linear_coupling_3q):
        strategy = SynthesisStrategy(
            num_qubits=3,
            num_physical_qubits=3,
            coupling_map=linear_coupling_3q,
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )

        obs_space = strategy.get_observation_space()

        assert set(obs_space.spaces) == {
            "layout",
            "physical_to_logical",
            "residual_symplectic",
            "residual_phase",
            "step_progress",
        }
        assert obs_space["residual_symplectic"].shape == (4 * 3 * 3,)
        assert obs_space["residual_phase"].shape == (2 * 3,)
```

- [ ] **Step 2: Run the updated strategy tests and verify they fail first**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_module.py -k "SynthesisStrategy" -v`

Expected: FAIL because the current implementation still exposes `MultiDiscrete` and the old shared observation shape.

- [ ] **Step 3: Redesign `SynthesisStrategy` around a primitive catalog and residual observation**

Update `src/rl_module/env_strategies.py` so `SynthesisStrategy` becomes:

```python
from .synthesis_clifford import CliffordSynthesisState, clifford_to_observation_arrays
from .synthesis_primitives import SynthesisPrimitive, build_clifford_primitive_catalog


class SynthesisStrategy(RLEnvStrategy):
    def __init__(
        self,
        num_qubits: int,
        num_physical_qubits: Optional[int] = None,
        coupling_map: Optional[List[Tuple[int, int]]] = None,
        lookahead_window: int = 10,
        basis_gates: Optional[List[str]] = None,
    ):
        super().__init__(num_qubits, num_physical_qubits, coupling_map, lookahead_window)
        if not basis_gates:
            raise ValueError("SynthesisStrategy requiere basis_gates explícitas para construir primitivas hardware-aware.")
        self.basis_gates = list(basis_gates)
        self.primitives = build_clifford_primitive_catalog(
            num_physical_qubits=self.num_physical_qubits,
            coupling_map=self.coupling_map,
            basis_gates=self.basis_gates,
        )

    def get_observation_space(self) -> gym.Space:
        return gym.spaces.Dict({
            "layout": gym.spaces.Box(
                low=-1,
                high=self.num_physical_qubits - 1,
                shape=(self.num_qubits,),
                dtype=np.int32,
            ),
            "physical_to_logical": gym.spaces.Box(
                low=-1,
                high=self.num_qubits - 1,
                shape=(self.num_physical_qubits,),
                dtype=np.int32,
            ),
            "residual_symplectic": gym.spaces.Box(
                low=0,
                high=1,
                shape=(4 * self.num_physical_qubits * self.num_physical_qubits,),
                dtype=np.int32,
            ),
            "residual_phase": gym.spaces.Box(
                low=0,
                high=1,
                shape=(2 * self.num_physical_qubits,),
                dtype=np.int32,
            ),
            "step_progress": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self.primitives))

    def build_observation(
        self,
        current_layout: np.ndarray,
        remaining_gates: Any,
        step_progress: float = 0.0,
        *,
        synthesis_state: CliffordSynthesisState,
        physical_to_logical: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        residual_symplectic, residual_phase = clifford_to_observation_arrays(
            synthesis_state.residual()
        )
        return {
            "layout": current_layout.copy(),
            "physical_to_logical": physical_to_logical.copy(),
            "residual_symplectic": residual_symplectic,
            "residual_phase": residual_phase,
            "step_progress": np.array([step_progress], dtype=np.float32),
        }

    def decode_action(self, action: int) -> Dict[str, Any]:
        action_index = int(action)
        if action_index < 0 or action_index >= len(self.primitives):
            return {"type": "invalid"}
        primitive = self.primitives[action_index]
        return {
            "type": "gate",
            "primitive_index": action_index,
            "primitive": primitive,
            "gate_name": primitive.gate_name,
            "physical_qargs": primitive.physical_qargs,
        }
```

- [ ] **Step 4: Re-run the strategy tests and verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_module.py -k "SynthesisStrategy" -v`

Expected: PASS.

- [ ] **Step 5: Commit the strategy task**

```bash
git add src/rl_module/env_strategies.py tests/test_rl_module/test_rl_module.py
git commit -m "feat: redesign synthesis strategy around Clifford primitives"
```

### Task 4: Integrate Synthesis State Into The Environment

**Files:**
- Modify: `src/rl_module/environment.py`
- Modify: `tests/test_rl_module/test_rl_module.py`

- [ ] **Step 1: Add failing environment tests for basis-gate requirements, completion, and routing safety**

Append or replace the synthesis-environment section in `tests/test_rl_module/test_rl_module.py` with:

```python
import pytest
from qiskit import QuantumCircuit

def test_synthesis_mode_requires_basis_gates(simple_circuit_3q, linear_coupling_3q):
    with pytest.raises(ValueError, match="basis_gates"):
        QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="synthesis",
            max_steps=10,
        )


def test_synthesis_mode_rejects_non_clifford_target():
    qc = QuantumCircuit(1)
    qc.rz(0.25, 0)

    env = QuantumTranspilationEnv(
        target_circuit=qc,
        coupling_map=[],
        mode="synthesis",
        basis_gates=["rz", "sx", "x"],
        max_steps=10,
    )

    with pytest.raises(ValueError, match="Clifford"):
        env.reset(seed=42)


def test_synthesis_reset_exposes_residual_observation(linear_coupling_3q):
    qc = QuantumCircuit(1)
    qc.x(0)
    env = QuantumTranspilationEnv(
        target_circuit=qc,
        coupling_map=linear_coupling_3q,
        mode="synthesis",
        basis_gates=["cz", "rz", "sx", "x"],
        max_steps=10,
    )

    obs, info = env.reset(seed=42, options={"initial_layout": [1]})

    assert set(obs) == {
        "layout",
        "physical_to_logical",
        "residual_symplectic",
        "residual_phase",
        "step_progress",
    }
    assert info["already_completed_at_reset"] is False


def test_synthesis_mode_step_completes_single_qubit_target():
    qc = QuantumCircuit(1)
    qc.x(0)
    env = QuantumTranspilationEnv(
        target_circuit=qc,
        coupling_map=[],
        mode="synthesis",
        basis_gates=["rz", "sx", "x"],
        max_steps=10,
    )

    env.reset(seed=42)
    x_index = next(
        index
        for index, primitive in enumerate(env.strategy.primitives)
        if primitive.gate_name == "x" and primitive.physical_qargs == (0,)
    )

    obs, reward, terminated, truncated, info = env.step(x_index)

    assert terminated is True
    assert truncated is False
    assert info["is_completed"] is True
    assert info["residual_distance_after"] == 0


def test_synthesis_invalid_action_on_empty_physical_qubit_is_penalized_not_terminal(linear_coupling_3q):
    qc = QuantumCircuit(1)
    qc.x(0)
    env = QuantumTranspilationEnv(
        target_circuit=qc,
        coupling_map=linear_coupling_3q,
        mode="synthesis",
        basis_gates=["cz", "rz", "sx", "x"],
        max_steps=10,
    )

    env.reset(seed=42, options={"initial_layout": [2]})
    invalid_index = next(
        index
        for index, primitive in enumerate(env.strategy.primitives)
        if primitive.gate_name == "x" and primitive.physical_qargs == (0,)
    )

    obs, reward, terminated, truncated, info = env.step(invalid_index)

    assert info["is_valid_action"] is False
    assert terminated is False
    assert truncated is False


def test_routing_mode_contract_unchanged_after_synthesis_refactor(simple_circuit_3q, linear_coupling_3q):
    env = QuantumTranspilationEnv(
        target_circuit=simple_circuit_3q,
        coupling_map=linear_coupling_3q,
        mode="routing",
        max_steps=20,
    )

    obs, _ = env.reset(seed=42)

    assert set(obs) == {
        "layout",
        "lookahead",
        "lookahead_physical",
        "lookahead_executable",
        "lookahead_routing_distance",
        "lookahead_valid_mask",
        "step_progress",
    }
```

- [ ] **Step 2: Run the environment tests and verify they fail first**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_module.py -k "synthesis_mode or routing_mode_contract_unchanged_after_synthesis_refactor" -v`

Expected: FAIL because the current environment does not require `basis_gates`, does not reject non-Clifford targets, does not expose residual observations, and does not terminate on residual identity.

- [ ] **Step 3: Add basis-gate threading and synthesis-state helpers to the environment**

Update `src/rl_module/environment.py` with these signatures and helpers:

```python
from .synthesis_clifford import CliffordSynthesisState


class QuantumTranspilationEnv(gym.Env):
    def __init__(
        self,
        target_circuit: QuantumCircuit,
        coupling_map: List[Tuple[int, int]],
        mode: str = "routing",
        frontier_mode: str = "sequential",
        lookahead_window: int = 10,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        basis_gates: Optional[List[str]] = None,
    ):
        self.basis_gates = list(basis_gates) if basis_gates is not None else None
        self._synthesis_state: Optional[CliffordSynthesisState] = None
        ...
        elif self.mode == "synthesis":
            if not self.basis_gates:
                raise ValueError("mode='synthesis' requiere basis_gates explícitas para construir primitivas hardware-aware.")
            self.strategy = SynthesisStrategy(
                self.num_qubits,
                self.num_physical_qubits,
                self.coupling_map_list,
                self.lookahead_window,
                basis_gates=self.basis_gates,
            )
            self.reward_function = SynthesisReward()
```

Add these mode-specific helpers in the same file:

```python
def _reset_synthesis_state(self) -> None:
    self._synthesis_state = CliffordSynthesisState.from_target_circuit(
        target_circuit=self.target_circuit,
        layout=self.current_layout,
        num_physical_qubits=self.num_physical_qubits,
    )


def _build_observation(self, step_progress: float) -> Dict[str, np.ndarray]:
    if self.mode == "synthesis":
        return self.strategy.build_observation(
            self.current_layout,
            self._frontier,
            step_progress=step_progress,
            synthesis_state=self._synthesis_state,
            physical_to_logical=self._inverse_layout,
        )
    return self.strategy.build_observation(
        self.current_layout,
        self._frontier,
        step_progress=step_progress,
    )


def _is_valid_synthesis_primitive(self, primitive) -> bool:
    for physical_qubit in primitive.physical_qargs:
        if int(self._inverse_layout[physical_qubit]) == -1:
            return False
    if len(primitive.physical_qargs) == 2:
        pq1, pq2 = primitive.physical_qargs
        return self._is_connected(pq1, pq2)
    return True


def _apply_synthesis_primitive(self, primitive, info: Dict[str, Any]) -> None:
    before = self._synthesis_state.residual_distance()
    info["primitive_name"] = primitive.gate_name
    info["primitive_cost"] = float(primitive.cost)
    info["residual_distance_before"] = before

    if not self._is_valid_synthesis_primitive(primitive):
        info["is_valid_action"] = False
        info["residual_distance_after"] = before
        info["residual_distance_delta"] = 0.0
        return

    self._synthesis_state.apply_primitive(primitive)
    after = self._synthesis_state.residual_distance()
    info["residual_distance_after"] = after
    info["residual_distance_delta"] = float(before - after)
```

- [ ] **Step 4: Switch `reset()` and `step()` to mode-specific synthesis semantics**

Update the `reset()` and `step()` bodies in `src/rl_module/environment.py` to:

```python
def reset(self, *, seed=None, options=None):
    ...
    if self.mode == "synthesis":
        self._reset_synthesis_state()
        self.was_completed_at_reset = self._synthesis_state.is_complete()
    else:
        initial_gates_executed = self._try_execute_front_layer()
        self.was_completed_at_reset = self._frontier.remaining_gate_count == 0

    obs = self._build_observation(step_progress=0.0)
    info = {
        "initial_layout_loaded": (options is not None and "initial_layout" in options),
        "total_gates": len(extracted_gates),
        "already_completed_at_reset": self.was_completed_at_reset,
    }
    return obs, info


def step(self, action):
    self.current_step += 1
    action_info = self.strategy.decode_action(action)
    prev_obs = self._build_observation(step_progress=(self.current_step - 1) / self.max_steps)
    prev_routing_signal = self._routing_signal(prev_obs) if self.mode == "routing" else 0.0
    ...
    if action_info["type"] == "swap":
        ...
    elif action_info["type"] == "gate":
        self._apply_synthesis_primitive(action_info["primitive"], info)
        self._last_swap_edge = None
    elif action_info["type"] == "invalid":
        info["is_valid_action"] = False
        self._last_swap_edge = None

    if self.mode == "synthesis":
        terminated = self._synthesis_state.is_complete()
    else:
        terminated = self._frontier.remaining_gate_count == 0
    truncated = self.current_step >= self.max_steps
    ...
    if self.mode == "routing":
        info["routing_progress_delta"] = routing_progress_delta
    else:
        info.setdefault("primitive_cost", 0.0)
        info.setdefault("residual_distance_before", 0)
        info.setdefault("residual_distance_after", 0)
        info.setdefault("residual_distance_delta", 0.0)

    obs = self._build_observation(step_progress=self.current_step / self.max_steps)
    reward = self.reward_function.compute_reward(prev_obs, action, obs, info)
    return obs, reward, terminated, truncated, info
```

- [ ] **Step 5: Re-run the environment tests and verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_module.py -k "synthesis_mode or routing_mode_contract_unchanged_after_synthesis_refactor" -v`

Expected: PASS.

- [ ] **Step 6: Commit the environment-integration task**

```bash
git add src/rl_module/environment.py tests/test_rl_module/test_rl_module.py
git commit -m "feat: integrate Clifford synthesis environment state"
```

### Task 5: Reward, Pipeline, GUI Safety, And Docs

**Files:**
- Modify: `src/rl_module/rewards.py`
- Modify: `src/rl_module/training.py`
- Modify: `src/rl_module/gui/rl_gui.py`
- Modify: `src/rl_module/docs/internal_documentation.md`
- Modify: `src/rl_module/docs/synthesis_mode_status.md`
- Modify: `tests/test_rl_module/test_rl_module.py`

- [ ] **Step 1: Replace the old synthesis-reward tests with failing residual-progress tests**

Replace the `TestSynthesisReward` section in `tests/test_rl_module/test_rl_module.py` with:

```python
class TestSynthesisReward:
    def test_reward_combines_progress_and_primitive_cost(self):
        reward_fn = SynthesisReward(
            invalid_action_penalty=-5.0,
            step_penalty=-0.25,
            primitive_cost_weight=0.5,
            residual_progress_reward=2.0,
            completion_bonus=100.0,
            truncation_penalty=-30.0,
        )

        info = {
            "is_valid_action": True,
            "primitive_cost": 3.0,
            "residual_distance_delta": 4.0,
            "is_completed": False,
            "is_truncated": False,
        }

        reward = reward_fn.compute_reward(None, None, None, info)

        assert reward == pytest.approx(-0.25 - (0.5 * 3.0) + (2.0 * 4.0))

    def test_invalid_action_uses_invalid_penalty(self):
        reward_fn = SynthesisReward(invalid_action_penalty=-7.0)
        info = {
            "is_valid_action": False,
            "primitive_cost": 0.0,
            "residual_distance_delta": 0.0,
            "is_completed": False,
            "is_truncated": False,
        }

        assert reward_fn.compute_reward(None, None, None, info) == pytest.approx(-7.0)
```

- [ ] **Step 2: Add a failing training-pipeline pass-through test**

Append to `tests/test_rl_module/test_rl_module.py`:

```python
import gymnasium as gym
import pytest
from pathlib import Path

def test_setup_training_pipeline_threads_basis_gates_into_env(monkeypatch, simple_circuit_3q, linear_coupling_3q, tmp_path):
    captured_basis_gates = []

    class DummyEnv:
        observation_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(1)

        def __init__(self, *args, **kwargs):
            captured_basis_gates.append(kwargs.get("basis_gates"))

        def reset(self, seed=None, options=None):
            return 0, {}

    class DummyAgent:
        def __init__(self, env, algorithm, tensorboard_log, seed, **kwargs):
            self.model = object()

        def train(self, total_timesteps, callbacks=None):
            return None

        def save(self, path):
            Path(path).touch()

    monkeypatch.setattr("src.rl_module.training.QuantumTranspilationEnv", DummyEnv)
    monkeypatch.setattr("src.rl_module.training.Monitor", lambda env: env)
    monkeypatch.setattr("src.rl_module.training.CheckpointCallback", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.rl_module.training.EvalCallback", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.rl_module.training.QuantumRLAgent", DummyAgent)

    setup_training_pipeline(
        target_circuit=simple_circuit_3q,
        coupling_map=linear_coupling_3q,
        mode="synthesis",
        algorithm="PPO",
        total_timesteps=1,
        max_steps=4,
        basis_gates=["cz", "rz", "sx", "x"],
        log_dir=str(tmp_path / "logs"),
        model_save_dir=str(tmp_path / "models"),
    )

    assert captured_basis_gates == [
        ["cz", "rz", "sx", "x"],
        ["cz", "rz", "sx", "x"],
    ]
```

- [ ] **Step 3: Run the reward and training tests and verify they fail first**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_module.py -k "SynthesisReward or setup_training_pipeline_threads_basis_gates_into_env" -v`

Expected: FAIL because the current reward still uses `gate_matched_target` and the current training pipeline does not accept `basis_gates`.

- [ ] **Step 4: Redefine `SynthesisReward` around residual progress and primitive cost**

Update `src/rl_module/rewards.py` so `SynthesisReward` becomes:

```python
class SynthesisReward(RewardStrategy):
    def __init__(
        self,
        invalid_action_penalty: float = -5.0,
        step_penalty: float = -0.25,
        primitive_cost_weight: float = 0.5,
        residual_progress_reward: float = 1.0,
        completion_bonus: float = 100.0,
        truncation_penalty: float = -30.0,
    ):
        self.invalid_action_penalty = invalid_action_penalty
        self.step_penalty = step_penalty
        self.primitive_cost_weight = primitive_cost_weight
        self.residual_progress_reward = residual_progress_reward
        self.completion_bonus = completion_bonus
        self.truncation_penalty = truncation_penalty

    def compute_reward(self, prev_state, action, current_state, info):
        if info.get("is_valid_action") is False:
            reward = self.invalid_action_penalty
        else:
            reward = self.step_penalty
            reward -= self.primitive_cost_weight * float(info.get("primitive_cost", 0.0))
            reward += self.residual_progress_reward * float(info.get("residual_distance_delta", 0.0))

        if info.get("is_completed", False):
            reward += self.completion_bonus
        if info.get("is_truncated", False) and not info.get("is_completed", False):
            reward += self.truncation_penalty
        return reward
```

- [ ] **Step 5: Thread `basis_gates` through training and add an explicit GUI basis profile**

Update `src/rl_module/training.py` so the public function signature and env creation become:

```python
def setup_training_pipeline(
    target_circuit: QuantumCircuit,
    coupling_map: List[Tuple[int, int]],
    mode: str = "routing",
    frontier_mode: str = "sequential",
    algorithm: str = "PPO",
    total_timesteps: int = 100_000,
    seed: int = 42,
    log_dir: str = "./experiments/logs/rl_logs",
    model_save_dir: str = "./experiments/models/rl_models",
    lookahead_window: int = 10,
    max_steps: int = 1000,
    hyperparams: Optional[dict] = None,
    basis_gates: Optional[List[str]] = None,
) -> QuantumRLAgent:
    ...
    raw_env = QuantumTranspilationEnv(
        target_circuit=target_circuit,
        coupling_map=coupling_map,
        mode=mode,
        frontier_mode=frontier_mode,
        lookahead_window=lookahead_window,
        max_steps=max_steps,
        basis_gates=basis_gates,
    )
    ...
    eval_raw_env = QuantumTranspilationEnv(
        target_circuit=target_circuit,
        coupling_map=coupling_map,
        mode=mode,
        frontier_mode=frontier_mode,
        lookahead_window=lookahead_window,
        max_steps=max_steps,
        basis_gates=basis_gates,
    )
```

Update `src/rl_module/gui/rl_gui.py` with an explicit synthesis basis-profile selector:

```python
SYNTHESIS_BASIS_PROFILES = {
    "Clifford-CZ": ["cz", "rz", "sx", "x"],
    "Clifford-ECR": ["ecr", "rz", "sx", "x"],
}
```

Add a new option menu in the sidebar and include this in `_get_config()`:

```python
"basis_gates": SYNTHESIS_BASIS_PROFILES[self._basis_profile_option.get()],
```

Then pass `basis_gates=config["basis_gates"] if config["mode"] == "synthesis" else None` into every `QuantumTranspilationEnv(...)` and `setup_training_pipeline(...)` call inside the GUI.

- [ ] **Step 6: Update the RL docs to reflect the new current contract**

Update `src/rl_module/docs/internal_documentation.md` so the synthesis section states:

```md
### 2. Modo Síntesis (`mode="synthesis"`)

- **Estado actual**: primer modo entrenable restringido a circuitos Clifford.
- **Conciencia de hardware**: requiere `coupling_map` y `basis_gates`; la topología sola no determina la puerta nativa de 2 qubits.
- **Espacio de acción**: `Discrete(N)` sobre un catálogo fijo de primitivas Clifford hardware-aware.
- **Criterio de éxito**: equivalencia Clifford por residual identidad en espacio físico.
- **Limitación actual**: el layout es fijo durante el episodio; no hay `swap` dinámico en synthesis v1.
```

Update `src/rl_module/docs/synthesis_mode_status.md` so it states:

```md
## Estado actual

`synthesis` ya no es un placeholder, pero sigue siendo una primera fase acotada:

- soporta solo circuitos Clifford;
- requiere `basis_gates` explícitas además de `coupling_map`;
- usa equivalencia Clifford, no coincidencia secuencial con la lista de puertas target;
- mantiene layout fijo durante el episodio.

## Futuro trabajo

- síntesis general no-Clifford;
- espacios de acción parametrizados o continuos;
- integración con `swap` dinámico dentro del episodio;
- criterios de equivalencia o aproximación más generales que el residual Clifford.
```

- [ ] **Step 7: Re-run the targeted RL-module tests and then the full RL-module suite**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_module.py -k "SynthesisReward or setup_training_pipeline_threads_basis_gates_into_env" -v`

Expected: PASS.

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_rl_module/ -v`

Expected: PASS for the whole RL-module test suite, including routing regressions.

- [ ] **Step 8: Commit the reward, pipeline, GUI, and docs task**

```bash
git add src/rl_module/rewards.py src/rl_module/training.py src/rl_module/gui/rl_gui.py src/rl_module/docs/internal_documentation.md src/rl_module/docs/synthesis_mode_status.md tests/test_rl_module/test_rl_module.py
git commit -m "feat: wire Clifford synthesis through RL pipeline"
```

## Self-Review Checklist

- [ ] Every spec requirement is covered by at least one task.
- [ ] `routing` remains protected by explicit regression assertions.
- [ ] `basis_gates` is threaded through every synthesis entry point.
- [ ] The residual convention is frozen by unit tests before environment integration.
- [ ] No task depends on undocumented placeholder behavior.
