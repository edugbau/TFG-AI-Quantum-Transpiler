"""
Tests unitarios para el módulo rl_module
==========================================

Cobertura de los cinco sub-módulos:
  - env_strategies  (test_routing_strategy_*, test_synthesis_strategy_*)
  - rewards         (test_routing_reward_*, test_synthesis_reward_*)
  - environment     (test_env_*)
  - agent           (test_agent_*)
  - training        (test_training_*)

Se utilizan circuitos simples y coupling maps artificiales, sin Fake
Backends ni API keys. Cada test es independiente y reproducible.

Ejecución:
  pytest tests/test_rl_module/ -v

Autor: Eduardo González Bautista
Fecha: 2026-03-09
"""

import json
import types
import pytest
import numpy as np
import os
import tempfile
import warnings
from pathlib import Path
from collections import deque

# ---------------------------------------------------------------------------
# Imports del módulo bajo test
# ---------------------------------------------------------------------------
from src.rl_module.env_strategies import (
    RoutingStrategy,
    SynthesisStrategy,
    RLEnvStrategy,
)
from src.rl_module.rewards import (
    RewardStrategy,
    RoutingReward,
    SynthesisReward,
)
from src.rl_module.environment import QuantumTranspilationEnv
from src.rl_module.agent import QuantumRLAgent
from src.rl_module.training import set_global_seeds, setup_training_pipeline
from src.rl_module.synthesis_clifford import CliffordSynthesisState
from src.rl_module.frontier import LookaheadEntry
from src.rl_module import agent as agent_module

# ---------------------------------------------------------------------------
# Imports de dependencias externas
# ---------------------------------------------------------------------------
from qiskit import QuantumCircuit
import gymnasium as gym


# ===========================================================================
#  Fixtures
# ===========================================================================

@pytest.fixture
def simple_circuit_3q() -> QuantumCircuit:
    """Circuito simple de 3 qubits: H(0) -> CX(0,1) -> CX(1,2)."""
    qc = QuantumCircuit(3, name="test_3q")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


@pytest.fixture
def linear_coupling_3q() -> list:
    """Coupling map lineal para 3 qubits: 0-1-2."""
    return [(0, 1), (1, 2)]


@pytest.fixture
def circuit_4q() -> QuantumCircuit:
    """Circuito de 4 qubits con CX no adyacentes en topología lineal."""
    qc = QuantumCircuit(4, name="test_4q")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(0, 3)  # Requiere SWAPs en topología lineal
    return qc


@pytest.fixture
def linear_coupling_4q() -> list:
    """Coupling map lineal para 4 qubits: 0-1-2-3."""
    return [(0, 1), (1, 2), (2, 3)]


@pytest.fixture
def star_coupling_4q() -> list:
    """Coupling map en estrella para 4 qubits: 0 conectado a 1,2,3."""
    return [(0, 1), (0, 2), (0, 3)]


@pytest.fixture
def single_cx_circuit() -> QuantumCircuit:
    """Circuito con una sola puerta CX(0,1)."""
    qc = QuantumCircuit(3, name="single_cx")
    qc.cx(0, 1)
    return qc


@pytest.fixture
def routing_env(simple_circuit_3q, linear_coupling_3q) -> QuantumTranspilationEnv:
    """Entorno de routing con circuito de 3 qubits y topología lineal."""
    return QuantumTranspilationEnv(
        target_circuit=simple_circuit_3q,
        coupling_map=linear_coupling_3q,
        mode="routing",
        max_steps=50,
    )


@pytest.fixture
def synthesis_env(simple_circuit_3q, linear_coupling_3q) -> QuantumTranspilationEnv:
    """Entorno de síntesis con circuito de 3 qubits y topología lineal."""
    return QuantumTranspilationEnv(
        target_circuit=simple_circuit_3q,
        coupling_map=linear_coupling_3q,
        mode="synthesis",
        basis_gates=["cz", "rz", "sx", "x"],
        max_steps=50,
    )


@pytest.fixture
def nontrivial_routing_env(circuit_4q, linear_coupling_4q) -> QuantumTranspilationEnv:
    """Entorno de routing con una CX no adyacente que requiere SWAPs."""
    return QuantumTranspilationEnv(
        target_circuit=circuit_4q,
        coupling_map=linear_coupling_4q,
        mode="routing",
        max_steps=50,
        lookahead_window=4,
    )


# ===========================================================================
#  Tests — env_strategies: RoutingStrategy
# ===========================================================================

class TestRoutingStrategy:
    """Tests de la estrategia de enrutamiento (RoutingStrategy)."""

    def test_observation_space_keys(self, linear_coupling_3q):
        """El observation space contiene las claves base y las enriquecidas."""
        strategy = RoutingStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        obs_space = strategy.get_observation_space()
        assert isinstance(obs_space, gym.spaces.Dict)
        assert "layout" in obs_space.spaces
        assert "lookahead" in obs_space.spaces
        assert "lookahead_physical" in obs_space.spaces
        assert "lookahead_executable" in obs_space.spaces
        assert "lookahead_routing_distance" in obs_space.spaces
        assert "lookahead_valid_mask" in obs_space.spaces
        assert "step_progress" in obs_space.spaces

    def test_observation_space_shapes(self, linear_coupling_3q):
        """Las dimensiones de las observaciones base y enriquecidas son correctas."""
        strategy = RoutingStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        obs_space = strategy.get_observation_space()
        assert obs_space["layout"].shape == (3,)
        assert obs_space["lookahead"].shape == (10,)  # lookahead_window * 2
        assert obs_space["lookahead_physical"].shape == (10,)
        assert obs_space["lookahead_executable"].shape == (5,)
        assert obs_space["lookahead_routing_distance"].shape == (5,)
        assert obs_space["lookahead_valid_mask"].shape == (5,)

    def test_action_space_is_discrete(self, linear_coupling_3q):
        """El action space es Discrete con tamaño = número de aristas únicas."""
        strategy = RoutingStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        action_space = strategy.get_action_space()
        assert isinstance(action_space, gym.spaces.Discrete)
        # Linear 3q: (0,1) y (1,2) → 2 aristas únicas
        assert action_space.n == 2

    def test_action_space_deduplicates_bidirectional_edges(self):
        """Las aristas bidireccionales se deducen como una sola arista para SWAPs."""
        coupling = [(0, 1), (1, 0), (1, 2), (2, 1)]
        strategy = RoutingStrategy(num_qubits=3, coupling_map=coupling, lookahead_window=5)
        action_space = strategy.get_action_space()
        assert action_space.n == 2  # Solo (0,1) y (1,2)

    def test_edges_are_deterministically_ordered(self):
        """Las aristas tienen un orden determinista (sorted)."""
        coupling = [(2, 3), (0, 1), (1, 2)]
        strategy = RoutingStrategy(num_qubits=4, coupling_map=coupling, lookahead_window=5)
        assert strategy.edges == [(0, 1), (1, 2), (2, 3)]

    def test_build_observation_shape_and_dtype(self, linear_coupling_3q):
        """build_observation retorna dict con arrays de forma y dtype correctos."""
        strategy = RoutingStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        layout = np.array([0, 1, 2], dtype=np.int32)
        gates = [("cx", 0, 1), ("cx", 1, 2)]

        obs = strategy.build_observation(layout, gates)
        assert obs["layout"].shape == (3,)
        assert obs["layout"].dtype == np.int32
        assert obs["lookahead"].shape == (10,)
        assert obs["lookahead"].dtype == np.int32
        assert obs["lookahead_physical"].shape == (10,)
        assert obs["lookahead_physical"].dtype == np.int32
        assert obs["lookahead_executable"].shape == (5,)
        assert obs["lookahead_executable"].dtype == np.float32
        assert obs["lookahead_routing_distance"].shape == (5,)
        assert obs["lookahead_routing_distance"].dtype == np.float32
        assert obs["lookahead_valid_mask"].shape == (5,)
        assert obs["lookahead_valid_mask"].dtype == np.float32

    def test_build_observation_enriches_projected_frontier(self, linear_coupling_3q):
        """build_observation proyecta qubits fisicos y metricas de ejecutabilidad."""
        strategy = RoutingStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=3)
        layout = np.array([0, 2, 1], dtype=np.int32)
        gates = [("cx", 0, 2), ("cx", 0, 1)]

        obs = strategy.build_observation(layout, gates)

        np.testing.assert_array_equal(obs["lookahead"], np.array([0, 2, 0, 1, -1, -1], dtype=np.int32))
        np.testing.assert_array_equal(obs["lookahead_physical"], np.array([0, 1, 0, 2, -1, -1], dtype=np.int32))
        np.testing.assert_array_equal(obs["lookahead_executable"], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(obs["lookahead_routing_distance"], np.array([0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(obs["lookahead_valid_mask"], np.array([1.0, 1.0, 0.0], dtype=np.float32))

    def test_build_observation_padding_with_minus_one(self, linear_coupling_3q):
        """Cuando hay menos puertas que lookahead_window, se rellena con -1."""
        strategy = RoutingStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        layout = np.array([0, 1, 2], dtype=np.int32)
        gates = [("cx", 0, 1)]  # Solo 1 puerta, ventana de 5

        obs = strategy.build_observation(layout, gates)
        # Primeros 2 valores: qubits de la puerta. Resto: -1
        assert obs["lookahead"][0] == 0
        assert obs["lookahead"][1] == 1
        assert np.all(obs["lookahead"][2:] == -1)

    def test_build_observation_empty_gates(self, linear_coupling_3q):
        """Sin puertas pendientes, toda la ventana lookahead es -1."""
        strategy = RoutingStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        layout = np.array([0, 1, 2], dtype=np.int32)
        obs = strategy.build_observation(layout, [])
        assert np.all(obs["lookahead"] == -1)
        assert np.all(obs["lookahead_physical"] == -1)
        assert np.all(obs["lookahead_executable"] == 0.0)
        assert np.all(obs["lookahead_routing_distance"] == 0.0)
        assert np.all(obs["lookahead_valid_mask"] == 0.0)

    def test_build_observation_with_deque(self, linear_coupling_3q):
        """build_observation funciona con deque además de listas."""
        strategy = RoutingStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        layout = np.array([0, 1, 2], dtype=np.int32)
        gates = deque([("cx", 0, 1), ("cx", 1, 2)])
        obs = strategy.build_observation(layout, gates)
        assert obs["lookahead"][0] == 0
        assert obs["lookahead"][1] == 1

    def test_decode_action_valid_swap(self, linear_coupling_3q):
        """Una acción válida se decodifica como un SWAP con qubits correctos."""
        strategy = RoutingStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        action_info = strategy.decode_action(0)
        assert action_info["type"] == "swap"
        assert "physical_q1" in action_info
        assert "physical_q2" in action_info

    def test_decode_action_out_of_range(self, linear_coupling_3q):
        """Una acción fuera de rango se decodifica como inválida."""
        strategy = RoutingStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        assert strategy.decode_action(-1)["type"] == "invalid"
        assert strategy.decode_action(999)["type"] == "invalid"

    def test_observation_in_space(self, linear_coupling_3q):
        """La observación generada pertenece al observation space definido."""
        strategy = RoutingStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        layout = np.array([0, 1, 2], dtype=np.int32)
        gates = [("cx", 0, 1), ("cx", 1, 2)]
        obs = strategy.build_observation(layout, gates)
        assert strategy.get_observation_space().contains(obs)


# ===========================================================================
#  Tests — env_strategies: SynthesisStrategy
# ===========================================================================

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

    def test_synthesis_strategy_rejects_unsupported_basis_only(self):
        with pytest.raises(ValueError, match="primitive"):
            SynthesisStrategy(
                num_qubits=1,
                num_physical_qubits=1,
                coupling_map=[],
                lookahead_window=5,
                basis_gates=["t"],
            )

    def test_synthesis_strategy_rejects_empty_catalog_from_disconnected_basis(self):
        with pytest.raises(ValueError, match="primitive"):
            SynthesisStrategy(
                num_qubits=2,
                num_physical_qubits=2,
                coupling_map=[],
                lookahead_window=5,
                basis_gates=["cx"],
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
        assert "primitive_index" in action_info
        assert "primitive" in action_info
        assert action_info["gate_name"] == action_info["primitive"].gate_name
        assert action_info["physical_qargs"] == action_info["primitive"].physical_qargs

    def test_synthesis_strategy_decode_action_rejects_malformed_nonscalar_input(self, linear_coupling_3q):
        strategy = SynthesisStrategy(
            num_qubits=3,
            num_physical_qubits=3,
            coupling_map=linear_coupling_3q,
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )

        assert strategy.decode_action(np.array([99, 0, 0])) == {"type": "invalid"}

    def test_synthesis_strategy_decode_action_accepts_scalar_numpy_action(self, linear_coupling_3q):
        strategy = SynthesisStrategy(
            num_qubits=3,
            num_physical_qubits=3,
            coupling_map=linear_coupling_3q,
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )

        action_info = strategy.decode_action(np.array(0))

        assert action_info["type"] == "gate"
        assert action_info["primitive_index"] == 0

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

    def test_synthesis_build_observation_requires_state_and_inverse_layout(self, linear_coupling_3q):
        strategy = SynthesisStrategy(
            num_qubits=3,
            num_physical_qubits=3,
            coupling_map=linear_coupling_3q,
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )
        layout = np.array([0, 1, 2], dtype=np.int32)
        synthesis_state = CliffordSynthesisState.from_target_circuit(
            target_circuit=QuantumCircuit(1),
            layout=[0],
            num_physical_qubits=3,
        )
        physical_to_logical = np.array([0, 1, 2], dtype=np.int32)

        obs = strategy.build_observation(
            layout,
            [],
            step_progress=0.25,
            synthesis_state=synthesis_state,
            physical_to_logical=physical_to_logical,
        )

        assert set(obs) == {
            "layout",
            "physical_to_logical",
            "residual_symplectic",
            "residual_phase",
            "step_progress",
        }
        assert strategy.get_observation_space().contains(obs)

        with pytest.raises(TypeError):
            strategy.build_observation(layout, [], step_progress=0.25)

    def test_synthesis_build_observation_rejects_invalid_layout_shape(self, linear_coupling_3q):
        strategy = SynthesisStrategy(
            num_qubits=3,
            num_physical_qubits=3,
            coupling_map=linear_coupling_3q,
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )
        synthesis_state = CliffordSynthesisState.from_target_circuit(
            target_circuit=QuantumCircuit(1),
            layout=[0],
            num_physical_qubits=3,
        )

        with pytest.raises(ValueError, match="current_layout"):
            strategy.build_observation(
                np.array([0, 1], dtype=np.int32),
                [],
                synthesis_state=synthesis_state,
                physical_to_logical=np.array([0, 1, 2], dtype=np.int32),
            )

    def test_synthesis_build_observation_rejects_out_of_range_current_layout(self, linear_coupling_3q):
        strategy = SynthesisStrategy(
            num_qubits=3,
            num_physical_qubits=3,
            coupling_map=linear_coupling_3q,
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )
        synthesis_state = CliffordSynthesisState.from_target_circuit(
            target_circuit=QuantumCircuit(1),
            layout=[0],
            num_physical_qubits=3,
        )

        with pytest.raises(ValueError, match="current_layout"):
            strategy.build_observation(
                np.array([0, 1, 3], dtype=np.int32),
                [],
                synthesis_state=synthesis_state,
                physical_to_logical=np.array([0, 1, 2], dtype=np.int32),
            )

    def test_synthesis_build_observation_rejects_out_of_range_inverse_layout(self, linear_coupling_3q):
        strategy = SynthesisStrategy(
            num_qubits=3,
            num_physical_qubits=3,
            coupling_map=linear_coupling_3q,
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )
        synthesis_state = CliffordSynthesisState.from_target_circuit(
            target_circuit=QuantumCircuit(1),
            layout=[0],
            num_physical_qubits=3,
        )

        with pytest.raises(ValueError, match="physical_to_logical"):
            strategy.build_observation(
                np.array([0, 1, 2], dtype=np.int32),
                [],
                synthesis_state=synthesis_state,
                physical_to_logical=np.array([0, 1, 3], dtype=np.int32),
            )

    def test_synthesis_build_observation_rejects_inconsistent_inverse_layout(self, linear_coupling_3q):
        strategy = SynthesisStrategy(
            num_qubits=3,
            num_physical_qubits=3,
            coupling_map=linear_coupling_3q,
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )
        synthesis_state = CliffordSynthesisState.from_target_circuit(
            target_circuit=QuantumCircuit(1),
            layout=[0],
            num_physical_qubits=3,
        )

        with pytest.raises(ValueError, match="consistent"):
            strategy.build_observation(
                np.array([0, 2, 1], dtype=np.int32),
                [],
                synthesis_state=synthesis_state,
                physical_to_logical=np.array([0, 1, 2], dtype=np.int32),
            )

    def test_synthesis_build_observation_rejects_duplicate_logical_assignment_on_unused_physical_qubit(self):
        strategy = SynthesisStrategy(
            num_qubits=2,
            num_physical_qubits=3,
            coupling_map=[(0, 1)],
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )
        synthesis_state = CliffordSynthesisState.from_target_circuit(
            target_circuit=QuantumCircuit(2),
            layout=[0, 1],
            num_physical_qubits=3,
        )

        with pytest.raises(ValueError, match="physical_to_logical"):
            strategy.build_observation(
                np.array([0, 1], dtype=np.int32),
                [],
                synthesis_state=synthesis_state,
                physical_to_logical=np.array([0, 1, 0], dtype=np.int32),
            )

    def test_synthesis_build_observation_rejects_inverse_layout_that_maps_unassigned_logical_qubit(self):
        strategy = SynthesisStrategy(
            num_qubits=2,
            num_physical_qubits=3,
            coupling_map=[(0, 1)],
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )
        synthesis_state = CliffordSynthesisState.from_target_circuit(
            target_circuit=QuantumCircuit(1),
            layout=[0],
            num_physical_qubits=3,
        )

        with pytest.raises(ValueError, match="consistent"):
            strategy.build_observation(
                np.array([0, -1], dtype=np.int32),
                [],
                synthesis_state=synthesis_state,
                physical_to_logical=np.array([0, -1, 1], dtype=np.int32),
            )

    def test_synthesis_build_observation_rejects_residual_arrays_for_wrong_physical_width(self, linear_coupling_3q):
        strategy = SynthesisStrategy(
            num_qubits=3,
            num_physical_qubits=3,
            coupling_map=linear_coupling_3q,
            lookahead_window=5,
            basis_gates=["cz", "rz", "sx", "x"],
        )
        wrong_width_state = CliffordSynthesisState.from_target_circuit(
            target_circuit=QuantumCircuit(1),
            layout=[0],
            num_physical_qubits=2,
        )

        with pytest.raises(ValueError, match="residual"):
            strategy.build_observation(
                np.array([0, 1, 2], dtype=np.int32),
                [],
                synthesis_state=wrong_width_state,
                physical_to_logical=np.array([0, 1, 2], dtype=np.int32),
            )


# ===========================================================================
#  Tests — rewards: RoutingReward
# ===========================================================================

class TestRoutingReward:
    """Tests de la función de recompensa de enrutamiento."""

    def test_swap_penalty(self):
        """Aplicar un SWAP genera la penalización configurada."""
        reward_fn = RoutingReward(swap_penalty=-1.0)
        info = {"action_type": "swap", "gates_executed": 0, "is_valid_action": True, "is_completed": False, "is_truncated": False}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == -1.0

    def test_gate_execution_reward(self):
        """Ejecutar puertas genera recompensa proporcional."""
        reward_fn = RoutingReward(gate_execution_reward=10.0, swap_penalty=0.0)
        info = {"action_type": "swap", "gates_executed": 3, "is_valid_action": True, "is_completed": False, "is_truncated": False}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == 30.0  # 10.0 * 3

    def test_invalid_action_penalty(self):
        """Una acción inválida genera penalización."""
        reward_fn = RoutingReward(invalid_action_penalty=-5.0, swap_penalty=0.0)
        info = {"action_type": None, "gates_executed": 0, "is_valid_action": False, "is_completed": False, "is_truncated": False}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == -5.0

    def test_completion_bonus(self):
        """Completar el circuito genera bonificación."""
        reward_fn = RoutingReward(completion_bonus=50.0, swap_penalty=0.0)
        info = {"action_type": None, "gates_executed": 0, "is_valid_action": True, "is_completed": True, "is_truncated": False}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == 50.0

    def test_truncation_penalty(self):
        """Truncar el episodio sin completar genera penalización configurable."""
        reward_fn = RoutingReward(truncation_penalty=-20.0, swap_penalty=0.0)
        info = {"action_type": None, "gates_executed": 0, "is_valid_action": True, "is_completed": False, "is_truncated": True}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == -20.0

    def test_truncation_penalty_not_applied_on_completion(self):
        """La penalización por truncación no se aplica si el circuito se completó."""
        reward_fn = RoutingReward(truncation_penalty=-20.0, completion_bonus=50.0, swap_penalty=0.0)
        info = {"action_type": None, "gates_executed": 0, "is_valid_action": True, "is_completed": True, "is_truncated": True}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == 50.0  # Solo completion_bonus, sin truncation_penalty

    def test_truncation_penalty_disabled(self):
        """Se puede deshabilitar la penalización por truncación con 0.0."""
        reward_fn = RoutingReward(truncation_penalty=0.0, swap_penalty=0.0)
        info = {"action_type": None, "gates_executed": 0, "is_valid_action": True, "is_completed": False, "is_truncated": True}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == 0.0

    def test_combined_reward_swap_plus_gates_plus_completion(self):
        """Recompensa combinada: SWAP + ejecución de puerta + completar."""
        reward_fn = RoutingReward(
            swap_penalty=-1.0,
            gate_execution_reward=10.0,
            completion_bonus=50.0,
        )
        info = {"action_type": "swap", "gates_executed": 1, "is_valid_action": True, "is_completed": True, "is_truncated": False}
        reward = reward_fn.compute_reward(None, None, None, info)
        # -1 (swap) + 10 (1 gate) + 50 (completed) = 59
        assert reward == 59.0

    def test_custom_reward_parameters(self):
        """Se pueden personalizar todos los parámetros de recompensa."""
        reward_fn = RoutingReward(
            swap_penalty=-2.5,
            gate_execution_reward=20.0,
            invalid_action_penalty=-10.0,
            completion_bonus=100.0,
        )
        info = {"action_type": "swap", "gates_executed": 2, "is_valid_action": True, "is_completed": False, "is_truncated": False}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == pytest.approx(-2.5 + 40.0)

    def test_transition_shaping_penalties_and_progress_reward(self):
        """Las señales de transición ajustan la recompensa de routing."""
        reward_fn = RoutingReward(
            swap_penalty=0.0,
            gate_execution_reward=0.0,
            repeated_layout_penalty=-2.0,
            undo_swap_penalty=-3.0,
            routing_progress_reward=1.5,
        )
        info = {
            "action_type": "swap",
            "gates_executed": 0,
            "is_valid_action": True,
            "is_completed": False,
            "is_truncated": False,
            "repeated_layout": True,
            "undo_swap": True,
            "routing_progress_delta": 2.0,
        }

        reward = reward_fn.compute_reward(None, None, None, info)

        assert reward == pytest.approx(-2.0 + -3.0 + (1.5 * 2.0))

    def test_negative_routing_progress_is_penalized_symmetrically(self):
        """Empeorar la distancia de routing aplica shaping negativo si está configurado."""
        reward_fn = RoutingReward(
            swap_penalty=0.0,
            gate_execution_reward=0.0,
            routing_progress_reward=2.0,
        )
        info = {
            "action_type": "swap",
            "gates_executed": 0,
            "is_valid_action": True,
            "is_completed": False,
            "is_truncated": False,
            "routing_progress_delta": -1.5,
        }

        reward = reward_fn.compute_reward(None, None, None, info)

        assert reward == pytest.approx(-3.0)


# ===========================================================================
#  Tests — rewards: SynthesisReward
# ===========================================================================

class TestSynthesisReward:
    """Tests de la función de recompensa de síntesis."""

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


# ===========================================================================
#  Tests — environment: QuantumTranspilationEnv
# ===========================================================================

class TestQuantumTranspilationEnv:
    """Tests del entorno principal de Gymnasium."""

    def test_init_routing_mode(self, simple_circuit_3q, linear_coupling_3q):
        """El entorno se inicializa correctamente en modo routing."""
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="routing",
        )
        assert isinstance(env.strategy, RoutingStrategy)
        assert isinstance(env.reward_function, RoutingReward)
        assert isinstance(env.action_space, gym.spaces.Discrete)

    def test_init_synthesis_mode(self, simple_circuit_3q, linear_coupling_3q):
        """El entorno se inicializa correctamente en modo synthesis."""
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="synthesis",
            basis_gates=["cz", "rz", "sx", "x"],
        )
        assert isinstance(env.strategy, SynthesisStrategy)
        assert isinstance(env.reward_function, SynthesisReward)
        assert isinstance(env.action_space, gym.spaces.Discrete)

    def test_init_accepts_dag_frontier_mode(self, simple_circuit_3q, linear_coupling_3q):
        """El constructor acepta frontier_mode='dag' y reset expone claves enriquecidas."""
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="routing",
            frontier_mode="dag",
            lookahead_window=3,
        )

        obs, _ = env.reset(seed=42)

        assert env.frontier_mode == "dag"
        assert "lookahead" in obs
        assert "lookahead_physical" in obs
        assert "lookahead_executable" in obs
        assert "lookahead_routing_distance" in obs
        assert "lookahead_valid_mask" in obs

    def test_init_invalid_mode(self, simple_circuit_3q, linear_coupling_3q):
        """Un modo inválido lanza ValueError."""
        with pytest.raises(ValueError, match="no soportado"):
            QuantumTranspilationEnv(
                target_circuit=simple_circuit_3q,
                coupling_map=linear_coupling_3q,
                mode="invalid_mode",
            )

    def test_init_render_mode(self, simple_circuit_3q, linear_coupling_3q):
        """render_mode se almacena correctamente."""
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="routing",
            render_mode="human",
        )
        assert env.render_mode == "human"

    def test_init_render_mode_default_none(self, simple_circuit_3q, linear_coupling_3q):
        """render_mode es None por defecto."""
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="routing",
        )
        assert env.render_mode is None

    def test_coupling_set_precomputed(self, simple_circuit_3q, linear_coupling_3q):
        """_coupling_set se precomputa como set bidireccional."""
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="routing",
        )
        assert (0, 1) in env._coupling_set
        assert (1, 0) in env._coupling_set
        assert (1, 2) in env._coupling_set
        assert (2, 1) in env._coupling_set
        assert (0, 2) not in env._coupling_set

    def test_remaining_gates_is_deque(self, routing_env):
        """remaining_gates es un deque para O(1) popleft."""
        routing_env.reset(seed=42)
        assert isinstance(routing_env.remaining_gates, deque)

    def test_inverse_layout_exists(self, routing_env):
        """_inverse_layout se crea y es coherente con current_layout."""
        routing_env.reset(seed=42)
        layout = routing_env.current_layout
        inv = routing_env._inverse_layout
        for lq in range(routing_env.num_qubits):
            pq = layout[lq]
            assert inv[pq] == lq

    def test_reset_returns_obs_and_info(self, routing_env):
        """reset() retorna una tupla (obs, info) con la estructura correcta."""
        obs, info = routing_env.reset(seed=42)

        assert isinstance(obs, dict)
        assert "layout" in obs
        assert "lookahead" in obs
        assert "lookahead_physical" in obs
        assert "lookahead_executable" in obs
        assert "lookahead_routing_distance" in obs
        assert "lookahead_valid_mask" in obs
        assert isinstance(info, dict)
        assert "initial_layout_loaded" in info
        assert "total_gates" in info

    def test_reset_obs_in_observation_space(self, routing_env):
        """La observación de reset() pertenece al observation_space."""
        obs, _ = routing_env.reset(seed=42)
        assert routing_env.observation_space.contains(obs)

    def test_reset_default_trivial_layout(self, routing_env):
        """Sin initial_layout, se usa layout trivial [0, 1, 2, ...]."""
        obs, info = routing_env.reset(seed=42)
        assert info["initial_layout_loaded"] is False
        np.testing.assert_array_equal(
            routing_env.current_layout,
            np.arange(routing_env.num_qubits, dtype=np.int32),
        )

    def test_reset_with_initial_layout(self, routing_env):
        """reset() con initial_layout inyecta el layout correctamente."""
        custom_layout = [2, 0, 1]
        obs, info = routing_env.reset(seed=42, options={"initial_layout": custom_layout})
        assert info["initial_layout_loaded"] is True
        np.testing.assert_array_equal(
            routing_env.current_layout,
            np.array(custom_layout, dtype=np.int32),
        )

    def test_reset_invalid_layout_length(self, routing_env):
        """Layout con longitud incorrecta lanza ValueError."""
        with pytest.raises(ValueError, match="longitud"):
            routing_env.reset(seed=42, options={"initial_layout": [0, 1]})

    def test_reset_invalid_layout_duplicates(self, routing_env):
        """Layout con duplicados lanza ValueError."""
        with pytest.raises(ValueError, match="duplicados"):
            routing_env.reset(seed=42, options={"initial_layout": [0, 0, 1]})

    def test_reset_invalid_layout_out_of_range(self, routing_env):
        """Layout con valores fuera de rango lanza ValueError."""
        with pytest.raises(ValueError, match="fuera del rango"):
            routing_env.reset(seed=42, options={"initial_layout": [0, 1, 99]})

    def test_reset_executes_satisfiable_gates(self):
        """reset() ejecuta automáticamente puertas ya satisfechas por el layout."""
        # CX(0,1) con coupling (0,1) y layout trivial → directamente ejecutable
        qc = QuantumCircuit(2, name="trivial")
        qc.cx(0, 1)
        coupling = [(0, 1)]
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=coupling,
            mode="routing",
            max_steps=100,
        )
        _, info = env.reset(seed=42)
        # La puerta debería haberse ejecutado automáticamente al final de reset()
        assert len(env.remaining_gates) == 0
        assert info["total_gates"] == 1

    def test_reset_is_deterministic_when_circuit_is_already_executable(self):
        """reset() mantiene el layout trivial cuando el circuito ya esta resuelto."""
        qc = QuantumCircuit(2, name="trivial")
        qc.cx(0, 1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1)],
            mode="routing",
            max_steps=10,
        )

        obs1, info1 = env.reset(seed=1)
        obs2, info2 = env.reset(seed=999)

        np.testing.assert_array_equal(obs1["layout"], np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(obs2["layout"], np.array([0, 1], dtype=np.int32))
        assert info1["already_completed_at_reset"] is True
        assert info2["already_completed_at_reset"] is True

    def test_init_tolerates_sparse_coupling_map_physical_qubit_count(self):
        """El numero de qubits fisicos nunca baja de num_qubits."""
        qc = QuantumCircuit(3)
        qc.cx(0, 2)

        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1)],
            mode="routing",
        )

        assert env.num_physical_qubits == 3

    def test_reset_reports_blocked_frontier_routing_distance(self):
        """La distancia de routing es shortest-path-length menos uno."""
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2)],
            mode="routing",
            lookahead_window=2,
        )

        obs, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs["lookahead"], np.array([0, 2, -1, -1], dtype=np.int32))
        np.testing.assert_array_equal(obs["lookahead_physical"], np.array([0, 2, -1, -1], dtype=np.int32))
        np.testing.assert_array_equal(obs["lookahead_executable"], np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(obs["lookahead_routing_distance"], np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(obs["lookahead_valid_mask"], np.array([1.0, 0.0], dtype=np.float32))

    def test_reset_nontrivial_routing_case_keeps_blocked_two_qubit_gate(self, nontrivial_routing_env):
        """Un caso de 4 qubits no trivial mantiene una CX bloqueada con distancia > 1."""
        obs, info = nontrivial_routing_env.reset(seed=42)

        assert info["total_gates"] == 4
        assert len(nontrivial_routing_env.remaining_gates) == 1
        assert nontrivial_routing_env.remaining_gates[0] == ("cx", 0, 3)
        np.testing.assert_array_equal(
            obs["lookahead"],
            np.array([0, 3, -1, -1, -1, -1, -1, -1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            obs["lookahead_executable"],
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            obs["lookahead_routing_distance"],
            np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            obs["lookahead_valid_mask"],
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_reset_clears_state(self, routing_env):
        """reset() reinicia current_step y total_swaps a 0."""
        routing_env.reset(seed=42)
        # Hacer un paso para modificar el estado
        routing_env.step(0)
        assert routing_env.current_step == 1

        # Reset debe limpiar
        routing_env.reset(seed=42)
        assert routing_env.current_step == 0
        assert routing_env.total_swaps == 0

    def test_step_increments_current_step(self, routing_env):
        """step() incrementa current_step en cada llamada."""
        routing_env.reset(seed=42)
        routing_env.step(0)
        assert routing_env.current_step == 1
        routing_env.step(0)
        assert routing_env.current_step == 2

    def test_step_returns_five_tuple(self, routing_env):
        """step() retorna (obs, reward, terminated, truncated, info)."""
        routing_env.reset(seed=42)
        result = routing_env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_in_observation_space(self, routing_env):
        """La observación del step() pertenece al observation_space."""
        routing_env.reset(seed=42)
        obs, _, _, _, _ = routing_env.step(0)
        assert routing_env.observation_space.contains(obs)

    def test_swap_modifies_layout(self, routing_env):
        """Un SWAP altera el current_layout del entorno."""
        routing_env.reset(seed=42)
        layout_before = routing_env.current_layout.copy()
        routing_env.step(0)  # Realizar un SWAP
        layout_after = routing_env.current_layout.copy()
        # El layout debería haber cambiado (SWAP intercambia dos qubits)
        assert not np.array_equal(layout_before, layout_after)

    def test_swap_updates_inverse_layout(self, routing_env):
        """Tras un SWAP, _inverse_layout se mantiene coherente con current_layout."""
        routing_env.reset(seed=42)
        routing_env.step(0)
        layout = routing_env.current_layout
        inv = routing_env._inverse_layout
        for lq in range(routing_env.num_qubits):
            pq = layout[lq]
            assert inv[pq] == lq

    def test_swap_increments_total_swaps(self, routing_env):
        """Cada SWAP incrementa el contador total_swaps."""
        routing_env.reset(seed=42)
        assert routing_env.total_swaps == 0
        routing_env.step(0)
        assert routing_env.total_swaps == 1
        routing_env.step(0)
        assert routing_env.total_swaps == 2

    def test_truncation_at_max_steps(self, simple_circuit_3q):
        """El episodio se trunca al alcanzar max_steps."""
        # Coupling sin conexión directa para la puerta → nunca se ejecuta
        coupling = [(0, 1)]  # Solo una arista
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=coupling,
            mode="routing",
            max_steps=5,
        )
        env.reset(seed=42)
        for i in range(5):
            _, _, terminated, truncated, _ = env.step(0)

        assert truncated is True

    def test_info_contains_is_truncated(self, simple_circuit_3q):
        """info contiene 'is_truncated' para la función de recompensa."""
        coupling = [(0, 1)]
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=coupling,
            mode="routing",
            max_steps=2,
        )
        env.reset(seed=42)
        _, _, _, _, info1 = env.step(0)
        assert "is_truncated" in info1
        assert info1["is_truncated"] is False

        _, _, _, truncated, info2 = env.step(0)
        assert truncated is True
        assert info2["is_truncated"] is True

    def test_termination_when_all_gates_executed(self):
        """El episodio termina cuando se ejecutan todas las puertas."""
        # CX(0,1) y coupling (0,1) → ejecutable si layout = [0,1]
        qc = QuantumCircuit(2, name="trivial")
        qc.cx(0, 1)
        coupling = [(0, 1)]
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=coupling,
            mode="routing",
            max_steps=100,
        )
        # Con layout trivial, la puerta se ejecuta en reset()
        env.reset(seed=42)
        # Verificamos que ya terminó
        assert len(env.remaining_gates) == 0

    def test_cascading_gate_execution(self):
        """_try_execute_front_layer ejecuta en cascada múltiples puertas."""
        # Dos CX consecutivos: CX(0,1) y CX(1,2) con coupling lineal [0-1-2]
        # En layout trivial ambas son ejecutables sin SWAP
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        coupling = [(0, 1), (1, 2)]
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=coupling,
            mode="routing",
            max_steps=100,
        )
        env.reset(seed=42)
        # Con layout trivial, ambas puertas se ejecutan automáticamente en reset()
        assert len(env.remaining_gates) == 0

    def test_extract_gates_two_qubit(self, simple_circuit_3q, linear_coupling_3q):
        """_extract_gates_from_circuit extrae puertas de 2 qubits correctamente."""
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="routing",
        )
        gates = env._extract_gates_from_circuit()
        # simple_circuit_3q: H(0), CX(0,1), CX(1,2)
        # H es de 1 qubit, CX son de 2 qubits
        # Todas se extraen (1-qubit con q1==q2)
        assert len(gates) == 3  # H + 2 CX
        # Las CX deben tener qubits distintos
        assert gates[1] == ("cx", 0, 1)
        assert gates[2] == ("cx", 1, 2)

    def test_extract_gates_single_qubit(self):
        """Puertas de 1 qubit se codifican con q1 == q2."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        env = QuantumTranspilationEnv(
            target_circuit=qc, coupling_map=[(0, 1)], mode="routing"
        )
        gates = env._extract_gates_from_circuit()
        for gate_name, q1, q2 in gates:
            assert q1 == q2  # 1-qubit gates have q1 == q2

    def test_extract_gates_three_qubit_warning(self):
        """Puertas de 3+ qubits generan un warning."""
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)  # Toffoli = 3 qubits
        env = QuantumTranspilationEnv(
            target_circuit=qc, coupling_map=[(0, 1), (1, 2)], mode="routing"
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gates = env._extract_gates_from_circuit()
            assert len(w) == 1
            assert "3 qubits" in str(w[0].message)
        assert len(gates) == 0  # La puerta de 3 qubits se ignora

    def test_reset_dag_mode_tolerates_unsupported_operations_like_sequential(self):
        """DAG mode ignora ops no soportadas con warning, igual que sequential."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.ccx(0, 1, 2)
        qc.barrier()
        qc.cx(0, 1)

        dag_env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2)],
            mode="routing",
            frontier_mode="dag",
            lookahead_window=2,
        )
        sequential_env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2)],
            mode="routing",
            frontier_mode="sequential",
            lookahead_window=2,
        )

        with warnings.catch_warnings(record=True) as dag_warnings:
            warnings.simplefilter("always")
            dag_obs, dag_info = dag_env.reset(seed=42)

        with warnings.catch_warnings(record=True) as sequential_warnings:
            warnings.simplefilter("always")
            sequential_obs, sequential_info = sequential_env.reset(seed=42)

        assert len(dag_warnings) == 2
        assert sorted(str(w.message) for w in dag_warnings) == sorted(
            str(w.message) for w in sequential_warnings
        )
        assert dag_info["total_gates"] == sequential_info["total_gates"] == 2
        assert list(dag_env.remaining_gates) == list(sequential_env.remaining_gates) == []
        np.testing.assert_array_equal(dag_obs["lookahead"], sequential_obs["lookahead"])

    def test_reset_reports_unreachable_frontier_routing_distance_with_sentinel(self):
        """Puertas bloqueadas sin ruta usan un sentinel distinto de ejecutable."""
        qc = QuantumCircuit(3)
        qc.cx(0, 2)

        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1)],
            mode="routing",
            lookahead_window=2,
        )

        obs, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs["lookahead_executable"], np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(obs["lookahead_routing_distance"], np.array([-1.0, 0.0], dtype=np.float32))

    def test_is_connected(self, routing_env):
        """_is_connected valida la topología correctamente."""
        routing_env.reset(seed=42)
        # Linear coupling: (0,1), (1,2)
        assert routing_env._is_connected(0, 1)
        assert routing_env._is_connected(1, 0)  # Bidireccional
        assert routing_env._is_connected(1, 2)
        assert not routing_env._is_connected(0, 2)  # No conectados

    def test_get_physical_qubit_trivial_layout(self, routing_env):
        """_get_physical_qubit retorna acceso O(1) con layout trivial."""
        routing_env.reset(seed=42)
        # Layout trivial: [0, 1, 2] → logical 0 está en physical 0
        assert routing_env._get_physical_qubit(0) == 0
        assert routing_env._get_physical_qubit(1) == 1
        assert routing_env._get_physical_qubit(2) == 2

    def test_get_physical_qubit_custom_layout(self, routing_env):
        """_get_physical_qubit funciona con layout no trivial."""
        routing_env.reset(seed=42, options={"initial_layout": [2, 0, 1]})
        # Layout: [2, 0, 1] → logical 0 en physical 2, logical 1 en physical 0, logical 2 en physical 1
        assert routing_env._get_physical_qubit(0) == 2
        assert routing_env._get_physical_qubit(1) == 0
        assert routing_env._get_physical_qubit(2) == 1

    def test_info_dict_keys_on_swap(self, routing_env):
        """El info dict de step() contiene las claves esperadas tras un SWAP."""
        routing_env.reset(seed=42)
        _, _, _, _, info = routing_env.step(0)
        assert "action_type" in info
        assert "is_valid_action" in info
        assert "gates_executed" in info
        assert "is_completed" in info
        assert "is_truncated" in info
        assert info["action_type"] == "swap"

    def test_step_routing_exposes_swap_edge_and_executed_gate_trace(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2)],
            mode="routing",
            max_steps=10,
        )

        env.reset(seed=42)
        action = env.strategy.edges.index((1, 2))

        _, _, terminated, truncated, info = env.step(action)

        assert terminated is True
        assert truncated is False
        assert info["swap_edge"] == (1, 2)
        assert info["executed_gates"] == [("cx", 0, 2)]

    def test_reset_routing_exposes_executed_gate_trace_for_auto_executed_prefix(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2)],
            mode="routing",
            max_steps=10,
        )

        _, info = env.reset(seed=42, options={"initial_layout": [1, 0, 2]})

        assert info["already_completed_at_reset"] is False
        assert info["executed_gates"] == [("h", 0, 0), ("cx", 0, 1)]

    def test_get_visible_frontier_entries_returns_current_gui_projection(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2)],
            mode="routing",
            lookahead_window=2,
            max_steps=10,
        )

        env.reset(seed=42)

        assert env.get_visible_frontier_entries() == [
            LookaheadEntry("cx", 0, 2, 0, 2, False),
        ]

    def test_get_visible_frontier_entries_returns_current_gui_projection_in_dag_mode(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.cx(1, 2)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2), (2, 3)],
            mode="routing",
            frontier_mode="dag",
            lookahead_window=3,
            max_steps=10,
        )

        env.reset(seed=42, options={"initial_layout": [0, 2, 1, 3]})

        assert env.get_visible_frontier_entries() == [
            LookaheadEntry("cx", 0, 1, 0, 2, False),
            LookaheadEntry("cx", 2, 3, 1, 3, False),
        ]

    def test_action_masks_enables_all_incident_edges_to_first_blocked_gate_in_sequential_mode(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 3)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2), (2, 3), (0, 4)],
            mode="routing",
            max_steps=10,
        )

        env.reset(seed=42)

        np.testing.assert_array_equal(
            env.action_masks(),
            np.array([True, True, False, True], dtype=bool),
        )

    def test_action_masks_returns_all_true_when_no_blocked_gate_exists(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (2, 3)],
            mode="routing",
            max_steps=10,
        )

        env.reset(seed=42)

        np.testing.assert_array_equal(
            env.action_masks(),
            np.array([True, True], dtype=bool),
        )

    def test_action_masks_keeps_non_improving_incident_edges_enabled(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2), (2, 3)],
            mode="routing",
            max_steps=10,
        )

        env.reset(seed=42, options={"initial_layout": [0, 2]})

        np.testing.assert_array_equal(
            env.action_masks(),
            np.array([True, True, True], dtype=bool),
        )

    def test_action_masks_v1_keeps_immediate_undo_swap_enabled(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2), (2, 3)],
            mode="routing",
            max_steps=10,
            mask_semantics="frontier_restricted_edges.v1",
        )

        env.reset(seed=42, options={"initial_layout": [0, 3]})
        action = env.strategy.edges.index((0, 1))
        env.step(action)

        assert env.action_masks()[action]

    def test_action_masks_v2_blocks_immediate_undo_swap_when_alternatives_exist(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2), (2, 3)],
            mode="routing",
            max_steps=10,
            mask_semantics="frontier_restricted_edges.v2",
        )

        env.reset(seed=42, options={"initial_layout": [0, 3]})
        action = env.strategy.edges.index((0, 1))
        env.step(action)

        assert not env.action_masks()[action]
        assert np.any(env.action_masks())

    def test_action_masks_v2_keeps_immediate_undo_as_fallback_when_it_is_the_only_candidate(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1)],
            mode="routing",
            max_steps=10,
            mask_semantics="frontier_restricted_edges.v2",
        )

        env.reset(seed=42, options={"initial_layout": [0, 1]})
        action = env.strategy.edges.index((0, 1))
        env.step(action)

        np.testing.assert_array_equal(
            env.action_masks(),
            np.array([True], dtype=bool),
        )

    def test_action_masks_keeps_incident_edges_enabled_for_unreachable_blocked_gate(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (2, 3)],
            mode="routing",
            max_steps=10,
        )

        env.reset(seed=42, options={"initial_layout": [0, 3]})

        np.testing.assert_array_equal(
            env.action_masks(),
            np.array([True, True], dtype=bool),
        )

    def test_action_masks_raises_when_blocked_gate_has_no_incident_valid_edges(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(2, 3)],
            mode="routing",
            max_steps=10,
        )

        env.reset(seed=42, options={"initial_layout": [0, 1]})

        with pytest.raises(ValueError, match="unroutable"):
            env.action_masks()

    def test_action_masks_uses_all_blocked_front_layer_gates_in_dag_mode(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.cx(1, 2)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2), (2, 3)],
            mode="routing",
            frontier_mode="dag",
            max_steps=10,
        )

        env.reset(seed=42, options={"initial_layout": [0, 2, 1, 3]})

        np.testing.assert_array_equal(
            env.action_masks(),
            np.array([True, True, True], dtype=bool),
        )

    def test_action_masks_raises_in_dag_mode_when_any_blocked_front_layer_gate_is_stranded(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.cx(1, 2)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 4)],
            mode="routing",
            frontier_mode="dag",
            max_steps=10,
        )

        env.reset(seed=42, options={"initial_layout": [0, 2, 1, 3]})

        with pytest.raises(ValueError, match="unroutable"):
            env.action_masks()

    def test_step_marks_swap_between_empty_physical_nodes_as_invalid(self):
        """Un SWAP entre dos nodos fisicos vacios es invalido pero alcanzable desde la estrategia."""
        # Arrange
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(2, 3)],
            mode="routing",
            max_steps=10,
        )
        env.reset(seed=42, options={"initial_layout": [0, 1]})
        layout_before = env.current_layout.copy()
        inverse_before = env._inverse_layout.copy()
        swap_action = env.strategy.edges.index((2, 3))

        # Act
        _, _, terminated, truncated, info = env.step(swap_action)

        # Assert
        assert info["action_type"] == "swap"
        assert info["is_valid_action"] is False
        assert info["gates_executed"] == 0
        assert terminated is False
        assert truncated is False
        assert env.total_swaps == 1
        np.testing.assert_array_equal(env.current_layout, layout_before)
        np.testing.assert_array_equal(env._inverse_layout, inverse_before)

    def test_step_reports_transition_signals_for_repeated_layout_and_undo_swap(self):
        """step() expone señales para layout repetido, undo swap y progreso de routing."""
        qc = QuantumCircuit(4)
        qc.cx(0, 3)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2), (2, 3)],
            mode="routing",
            max_steps=10,
        )

        env.reset(seed=42)
        first_action = env.strategy.edges.index((0, 1))
        second_action = env.strategy.edges.index((0, 1))

        _, _, _, _, info1 = env.step(first_action)
        _, _, _, _, info2 = env.step(second_action)

        assert info1["repeated_layout"] is False
        assert info1["undo_swap"] is False
        assert info1["routing_progress_delta"] == pytest.approx(1.0)
        assert info2["repeated_layout"] is True
        assert info2["undo_swap"] is True
        assert info2["routing_progress_delta"] == pytest.approx(-1.0)

    def test_step_reports_zero_routing_progress_when_frontier_already_executable(self, routing_env):
        """Si no hay distancia de routing pendiente, el delta informado es 0."""
        routing_env.reset(seed=42)

        _, _, _, _, info = routing_env.step(0)

        assert info["routing_progress_delta"] == pytest.approx(0.0)
        assert info["repeated_layout"] is False
        assert info["undo_swap"] is False

    def test_invalid_empty_empty_swap_does_not_arm_undo_swap(self):
        """Un swap vacío-inválido no debe contar como el swap previo a deshacer."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (2, 3)],
            mode="routing",
            max_steps=10,
        )
        env.reset(seed=42, options={"initial_layout": [2, 3]})
        empty_swap_action = env.strategy.edges.index((0, 1))

        _, _, _, _, info1 = env.step(empty_swap_action)
        _, _, _, _, info2 = env.step(empty_swap_action)

        assert info1["is_valid_action"] is False
        assert info1["undo_swap"] is False
        assert info2["is_valid_action"] is False
        assert info2["undo_swap"] is False

    def test_gate_execution_does_not_make_routing_progress_negative(self):
        """Ejecutar puertas en cascada no debe volver negativo el delta de progreso."""
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 2)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2), (2, 3)],
            mode="routing",
            max_steps=10,
        )

        env.reset(seed=42, options={"initial_layout": [0, 2, 1, 3]})
        action = env.strategy.edges.index((1, 2))

        _, _, _, _, info = env.step(action)

        assert info["gates_executed"] == 1
        assert info["routing_progress_delta"] == pytest.approx(0.0)

    def test_reusing_swap_after_frontier_advances_is_not_undo_or_repeated_layout(self):
        """Reutilizar la misma arista tras ejecutar puertas no cuenta como undo ni repetición vacía."""
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        qc.cx(0, 1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[(0, 1), (1, 2)],
            mode="routing",
            max_steps=10,
            mask_semantics="frontier_restricted_edges.v2",
        )

        env.reset(seed=42)
        action = env.strategy.edges.index((1, 2))

        _, _, _, _, info1 = env.step(action)
        assert env.action_masks()[action]
        _, _, terminated, _, info2 = env.step(action)

        assert info1["gates_executed"] == 1
        assert info1["undo_swap"] is False
        assert info1["repeated_layout"] is False
        assert info2["gates_executed"] == 1
        assert info2["undo_swap"] is False
        assert info2["repeated_layout"] is False
        assert terminated is True

    def test_routing_stability_roadmap_exists(self):
        """La hoja de ruta de estabilidad documenta la expansión futura a nivel 3."""
        roadmap_path = Path("src/rl_module/docs/routing_stability_roadmap.md")

        assert roadmap_path.exists()

        content = roadmap_path.read_text(encoding="utf-8")
        assert "current limitation" in content.lower()
        assert "intermediate" in content.lower()
        assert "level 3" in content.lower()
        assert "maskableppo" in content.lower()
        assert "action mask" in content.lower()
        assert "recurrent policy" in content.lower()
        assert "migration risk" in content.lower()

    def test_render_human_prints_step_summary_and_layout(self, routing_env, capsys):
        """render() en modo human imprime el resumen del estado y el layout actual."""
        # Arrange
        routing_env.render_mode = "human"
        routing_env.reset(seed=42)

        # Act
        result = routing_env.render()
        captured = capsys.readouterr()

        # Assert
        assert result is None
        assert captured.out == (
            "Step: 0 | Swaps: 0 | Remaining Gates: 0\n"
            "Current Layout (Logical->Physical): [0 1 2]\n"
        )
        assert captured.err == ""

    def test_render_non_human_mode_is_no_op(self, routing_env, capsys):
        """render() fuera de modo human no imprime nada y retorna None."""
        # Arrange
        routing_env.render_mode = "ansi"
        routing_env.reset(seed=42)

        # Act
        result = routing_env.render()
        captured = capsys.readouterr()

        # Assert
        assert result is None
        assert captured.out == ""
        assert captured.err == ""

    def test_multiple_resets(self, routing_env):
        """El entorno soporta múltiples resets consecutivos."""
        for _ in range(5):
            obs, info = routing_env.reset(seed=42)
            assert routing_env.observation_space.contains(obs)
            assert routing_env.current_step == 0

    def test_full_episode_routing(self, linear_coupling_3q):
        """Un episodio completo (hasta termination o truncation) no lanza errores."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        env = QuantumTranspilationEnv(
            target_circuit=qc, coupling_map=linear_coupling_3q,
            mode="routing", max_steps=20,
        )
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 20:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            done = terminated or truncated
            steps += 1

    def test_synthesis_mode_requires_basis_gates(self, simple_circuit_3q, linear_coupling_3q):
        with pytest.raises(ValueError, match="basis_gates"):
            QuantumTranspilationEnv(
                target_circuit=simple_circuit_3q,
                coupling_map=linear_coupling_3q,
                mode="synthesis",
                max_steps=10,
            )

    def test_synthesis_mode_rejects_non_clifford_target(self):
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

    def test_synthesis_mode_reset_exposes_residual_observation(self, linear_coupling_3q):
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

    def test_get_visible_frontier_entries_is_empty_in_synthesis_mode(self, linear_coupling_3q):
        qc = QuantumCircuit(1)
        qc.x(0)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=linear_coupling_3q,
            mode="synthesis",
            basis_gates=["cz", "rz", "sx", "x"],
            max_steps=10,
        )

        env.reset(seed=42)

        assert env.get_visible_frontier_entries() == []

    def test_synthesis_mode_step_completes_single_qubit_target(self):
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

    def test_synthesis_mode_step_exposes_primitive_trace_fields(self):
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

        _, _, terminated, truncated, info = env.step(x_index)

        assert terminated is True
        assert truncated is False
        assert info["primitive_name"] == "x"
        assert info["primitive_physical_qargs"] == (0,)
        assert info["primitive_cost"] == pytest.approx(1.0)
        assert info["residual_distance_before"] > 0
        assert info["residual_distance_after"] == 0
        assert info["residual_distance_delta"] == pytest.approx(
            float(info["residual_distance_before"])
        )

    def test_synthesis_mode_completion_clears_public_remaining_gates(self):
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
        assert list(env.remaining_gates) == []

    def test_synthesis_mode_invalid_action_on_empty_physical_qubit_is_penalized_not_terminal(self, linear_coupling_3q):
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

    def test_synthesis_mode_step_preserves_terminal_state_when_completed_at_reset(self):
        qc = QuantumCircuit(1)
        env = QuantumTranspilationEnv(
            target_circuit=qc,
            coupling_map=[],
            mode="synthesis",
            basis_gates=["rz", "sx", "x"],
            max_steps=10,
        )

        obs, info = env.reset(seed=42)

        assert info["already_completed_at_reset"] is True

        x_index = next(
            index
            for index, primitive in enumerate(env.strategy.primitives)
            if primitive.gate_name == "x" and primitive.physical_qargs == (0,)
        )

        obs, reward, terminated, truncated, step_info = env.step(x_index)

        assert terminated is True
        assert truncated is False
        assert step_info["is_completed"] is False
        assert step_info["residual_distance_before"] == 0
        assert step_info["residual_distance_after"] == 0
        assert step_info["residual_distance_delta"] == 0.0

    def test_routing_mode_contract_unchanged_after_synthesis_refactor(self, simple_circuit_3q, linear_coupling_3q):
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


# ===========================================================================
#  Tests — environment: Gymnasium env_checker
# ===========================================================================

class TestEnvChecker:
    """Tests de compatibilidad con gymnasium.utils.env_checker."""

    def test_check_env_routing(self, simple_circuit_3q, linear_coupling_3q):
        """El entorno routing pasa check_env de Gymnasium."""
        from gymnasium.utils.env_checker import check_env
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="routing",
            max_steps=50,
        )
        try:
            check_env(env, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"check_env falló: {e}")


# ===========================================================================
#  Tests — agent: QuantumRLAgent
# ===========================================================================

class TestQuantumRLAgent:
    """Tests del wrapper del agente RL."""

    def test_init_ppo(self, routing_env):
        """Se puede instanciar un agente PPO correctamente."""
        agent = QuantumRLAgent(env=routing_env, algorithm="PPO", verbose=0)
        assert agent.algorithm_name == "PPO"
        assert agent.model is not None

    def test_init_dqn(self, routing_env):
        """Se puede instanciar un agente DQN correctamente."""
        agent = QuantumRLAgent(env=routing_env, algorithm="DQN", verbose=0)
        assert agent.algorithm_name == "DQN"
        assert agent.model is not None

    def test_init_maskableppo_with_lazy_maskable_support(self, monkeypatch, routing_env):
        """MaskablePPO se resuelve de forma diferida sin romper PPO/DQN legacy."""

        class DummyMaskablePPO:
            def __init__(self, policy, env, tensorboard_log, verbose, device, **kwargs):
                self.policy = policy
                self.env = env

            def predict(self, observation, deterministic=True):
                return 0, None

        def fake_import_module(name):
            assert name == "sb3_contrib"
            return types.SimpleNamespace(MaskablePPO=DummyMaskablePPO)

        monkeypatch.setattr(
            agent_module,
            "importlib",
            types.SimpleNamespace(import_module=fake_import_module),
            raising=False,
        )

        agent = QuantumRLAgent(env=routing_env, algorithm="MaskablePPO", verbose=0)

        assert agent.algorithm_name == "MaskablePPO"
        assert isinstance(agent.model, DummyMaskablePPO)

    def test_init_invalid_algorithm(self, routing_env):
        """Un algoritmo no soportado lanza ValueError."""
        with pytest.raises(ValueError, match="no soportado"):
            QuantumRLAgent(env=routing_env, algorithm="A2C", verbose=0)

    def test_predict_returns_action(self, routing_env):
        """predict() retorna una acción del espacio de acciones."""
        agent = QuantumRLAgent(env=routing_env, algorithm="PPO", verbose=0)
        obs, _ = routing_env.reset(seed=42)
        action, states = agent.predict(obs)
        assert routing_env.action_space.contains(action)

    def test_save_and_load(self, routing_env):
        """save() y load() persisten y restauran sin crear modelo descartable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            
            agent = QuantumRLAgent(env=routing_env, algorithm="PPO", verbose=0)
            obs, _ = routing_env.reset(seed=42)
            action_before, _ = agent.predict(obs, deterministic=True)
            
            agent.save(save_path)
            assert os.path.exists(save_path + ".zip")
            
            loaded_agent = QuantumRLAgent.load(
                save_path, env=routing_env, algorithm="PPO"
            )
            action_after, _ = loaded_agent.predict(obs, deterministic=True)
            
            # Las acciones determinísticas deben coincidir
            assert np.array_equal(action_before, action_after)

    def test_save_bare_filename(self, routing_env):
        """save() funciona con un nombre de archivo sin directorio."""
        agent = QuantumRLAgent(env=routing_env, algorithm="PPO", verbose=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "bare_model")
            agent.save(save_path)
            assert os.path.exists(save_path + ".zip")

    def test_device_detection(self, routing_env):
        """El agente detecta el dispositivo correcto (cpu o cuda)."""
        agent = QuantumRLAgent(env=routing_env, algorithm="PPO", verbose=0)
        import torch
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert agent.device == expected

    def test_loaded_agent_has_all_attributes(self, routing_env):
        """El agente cargado tiene algorithm_name, env y device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            agent = QuantumRLAgent(env=routing_env, algorithm="PPO", verbose=0)
            agent.save(save_path)

            loaded = QuantumRLAgent.load(save_path, env=routing_env, algorithm="PPO")
            assert loaded.algorithm_name == "PPO"
            assert loaded.env is routing_env
            assert loaded.device in ("cpu", "cuda")
            assert loaded.model is not None

    def test_init_forwards_seed_into_sb3_constructor(self, monkeypatch, routing_env):
        """QuantumRLAgent reenvia la seed al constructor SB3 subyacente."""
        captured_seed = None

        class DummyAlgorithm:
            def __init__(self, policy, env, tensorboard_log, verbose, device, **kwargs):
                nonlocal captured_seed
                captured_seed = kwargs.get("seed")

            def predict(self, observation, deterministic=True):
                return 0, None

        monkeypatch.setitem(QuantumRLAgent.ALGORITHMS, "PPO", DummyAlgorithm)

        agent = QuantumRLAgent(env=routing_env, algorithm="PPO", verbose=0, seed=321)

        assert captured_seed == 321
        assert agent.model is not None


# ===========================================================================
#  Tests — training: Utilidades
# ===========================================================================

class TestTrainingUtilities:
    """Tests de las utilidades de entrenamiento."""

    def test_set_global_seeds_no_error(self):
        """set_global_seeds() se ejecuta sin errores."""
        set_global_seeds(42)
        # Verificar que seeds están fijados
        import torch
        a = torch.randn(5)
        set_global_seeds(42)
        b = torch.randn(5)
        torch.testing.assert_close(a, b)

    def test_set_global_seeds_reproducibility(self):
        """Con la misma seed, numpy produce los mismos resultados."""
        set_global_seeds(123)
        arr1 = np.random.rand(10)
        set_global_seeds(123)
        arr2 = np.random.rand(10)
        np.testing.assert_array_equal(arr1, arr2)

    def test_set_global_seeds_different_seeds_differ(self):
        """Seeds distintas producen resultados distintos."""
        set_global_seeds(42)
        arr1 = np.random.rand(10)
        set_global_seeds(99)
        arr2 = np.random.rand(10)
        assert not np.array_equal(arr1, arr2)

    def test_set_global_seeds_uses_logger(self, caplog):
        """set_global_seeds usa logger en vez de print."""
        import logging
        with caplog.at_level(logging.INFO, logger="src.rl_module.training"):
            set_global_seeds(42)
        assert any("42" in record.message for record in caplog.records)

    def test_setup_training_pipeline_accepts_frontier_mode(self, monkeypatch, simple_circuit_3q, linear_coupling_3q):
        """El pipeline reenvia frontier_mode al crear los entornos."""
        captured_frontier_modes = []

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                captured_frontier_modes.append(kwargs.get("frontier_mode"))

            def reset(self, seed=None):
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, tensorboard_log, **hyperparams):
                self.env = env

            def train(self, total_timesteps, callbacks):
                return None

            def save(self, path):
                return None

        monkeypatch.setattr("src.rl_module.training.QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr("src.rl_module.training.Monitor", lambda env: env)
        monkeypatch.setattr("src.rl_module.training.QuantumRLAgent", DummyAgent)
        monkeypatch.setattr("src.rl_module.training.CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr("src.rl_module.training.EvalCallback", lambda *args, **kwargs: object())
        monkeypatch.setattr("src.rl_module.training.os.makedirs", lambda *args, **kwargs: None)

        setup_training_pipeline(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            total_timesteps=1,
            frontier_mode="dag",
        )

        assert captured_frontier_modes == ["dag", "dag"]

    def test_setup_training_pipeline_passes_seed_into_agent_construction(
        self, monkeypatch, simple_circuit_3q, linear_coupling_3q
    ):
        """El pipeline reenvia la seed al wrapper del agente para SB3."""
        captured_seed = None

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed=None):
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, tensorboard_log, **hyperparams):
                nonlocal captured_seed
                captured_seed = hyperparams.get("seed")

            def train(self, total_timesteps, callbacks):
                return None

            def save(self, path):
                return None

        monkeypatch.setattr("src.rl_module.training.QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr("src.rl_module.training.Monitor", lambda env: env)
        monkeypatch.setattr("src.rl_module.training.QuantumRLAgent", DummyAgent)
        monkeypatch.setattr("src.rl_module.training.CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr("src.rl_module.training.EvalCallback", lambda *args, **kwargs: object())
        monkeypatch.setattr("src.rl_module.training.os.makedirs", lambda *args, **kwargs: None)

        setup_training_pipeline(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            total_timesteps=1,
            seed=123,
        )

        assert captured_seed == 123

    def test_setup_training_pipeline_seeds_train_and_eval_env_resets(
        self, monkeypatch, simple_circuit_3q, linear_coupling_3q
    ):
        """El pipeline inicializa train/eval env con reset(seed=...)."""
        reset_seeds = []

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed=None):
                reset_seeds.append(seed)
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, tensorboard_log, **hyperparams):
                pass

            def train(self, total_timesteps, callbacks):
                return None

            def save(self, path):
                return None

        monkeypatch.setattr("src.rl_module.training.QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr("src.rl_module.training.Monitor", lambda env: env)
        monkeypatch.setattr("src.rl_module.training.QuantumRLAgent", DummyAgent)
        monkeypatch.setattr("src.rl_module.training.CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr("src.rl_module.training.EvalCallback", lambda *args, **kwargs: object())
        monkeypatch.setattr("src.rl_module.training.os.makedirs", lambda *args, **kwargs: None)

        setup_training_pipeline(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            total_timesteps=1,
            seed=55,
        )

        assert reset_seeds == [55, 55]

    def test_setup_training_pipeline_keeps_initial_layout_across_subsequent_train_and_eval_resets(
        self, monkeypatch, tmp_path
    ):
        from src.rl_module import training

        reset_calls = []

        class FakeEnv(gym.Env):
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def reset(self, *, seed=None, options=None):
                reset_calls.append((seed, options))
                return {}, {}

        class FakeMonitor:
            def __init__(self, env):
                self.env = env

            def reset(self, *, seed=None, options=None):
                if options is None:
                    return self.env.reset(seed=seed)
                return self.env.reset(seed=seed, options=options)

        class FakeEvalCallback:
            def __init__(self, eval_env, *args, **kwargs):
                self.eval_env = eval_env

        class FakeAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.env = kwargs["env"]

            def train(self, total_timesteps, callbacks):
                self.env.reset()
                for callback in callbacks:
                    eval_env = getattr(callback, "eval_env", None)
                    if eval_env is not None:
                        eval_env.reset()
                return None

            def save(self, path):
                return None

        monkeypatch.setattr(training, "QuantumTranspilationEnv", FakeEnv)
        monkeypatch.setattr(training, "Monitor", FakeMonitor)
        monkeypatch.setattr(training, "CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr(training, "EvalCallback", FakeEvalCallback)
        monkeypatch.setattr(training, "QuantumRLAgent", FakeAgent)
        monkeypatch.setattr(training, "set_global_seeds", lambda seed: None)
        monkeypatch.setattr(training, "save_run_metadata", lambda *args, **kwargs: None)

        training.setup_training_pipeline(
            target_circuit=QuantumCircuit(5),
            coupling_map=[(0, 1), (1, 2)],
            algorithm="PPO",
            total_timesteps=10,
            seed=42,
            log_dir=str(tmp_path / "logs"),
            model_save_dir=str(tmp_path / "models"),
            initial_layout=[4, 3, 2, 1, 0],
        )

        assert reset_calls == [
            (42, {"initial_layout": [4, 3, 2, 1, 0]}),
            (42, {"initial_layout": [4, 3, 2, 1, 0]}),
            (None, {"initial_layout": [4, 3, 2, 1, 0]}),
            (None, {"initial_layout": [4, 3, 2, 1, 0]}),
        ]

    def test_setup_training_pipeline_keeps_default_reset_behavior_across_subsequent_resets_without_initial_layout(
        self, monkeypatch, tmp_path
    ):
        from src.rl_module import training

        reset_calls = []

        class FakeEnv(gym.Env):
            def __init__(self, **kwargs):
                pass

            def reset(self, *, seed=None, options=None):
                reset_calls.append((seed, options))
                return {}, {}

        class FakeMonitor:
            def __init__(self, env):
                self.env = env

            def reset(self, *, seed=None, options=None):
                if options is None:
                    return self.env.reset(seed=seed)
                return self.env.reset(seed=seed, options=options)

        class FakeEvalCallback:
            def __init__(self, eval_env, *args, **kwargs):
                self.eval_env = eval_env

        class FakeAgent:
            def __init__(self, **kwargs):
                self.env = kwargs["env"]

            def train(self, total_timesteps, callbacks):
                self.env.reset()
                for callback in callbacks:
                    eval_env = getattr(callback, "eval_env", None)
                    if eval_env is not None:
                        eval_env.reset()
                return None

            def save(self, path):
                return None

        monkeypatch.setattr(training, "QuantumTranspilationEnv", FakeEnv)
        monkeypatch.setattr(training, "Monitor", FakeMonitor)
        monkeypatch.setattr(training, "CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr(training, "EvalCallback", FakeEvalCallback)
        monkeypatch.setattr(training, "QuantumRLAgent", FakeAgent)
        monkeypatch.setattr(training, "set_global_seeds", lambda seed: None)
        monkeypatch.setattr(training, "save_run_metadata", lambda *args, **kwargs: None)

        training.setup_training_pipeline(
            target_circuit=QuantumCircuit(5),
            coupling_map=[(0, 1), (1, 2)],
            algorithm="PPO",
            total_timesteps=10,
            seed=42,
            log_dir=str(tmp_path / "logs"),
            model_save_dir=str(tmp_path / "models"),
        )

        assert reset_calls == [(42, None), (42, None), (None, None), (None, None)]

    def test_setup_training_pipeline_threads_basis_gates_into_env(
        self, monkeypatch, simple_circuit_3q, linear_coupling_3q, tmp_path
    ):
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

    def test_setup_training_pipeline_writes_run_metadata(
        self, monkeypatch, linear_coupling_3q, tmp_path
    ):
        from src.rl_module import training

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed=None):
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, tensorboard_log=None, seed=None, **kwargs):
                self.env = env
                self.algorithm_name = algorithm

            def train(self, total_timesteps, callbacks=None, progress_bar=True):
                return None

            def save(self, path):
                Path(path).write_text("model", encoding="utf-8")

        monkeypatch.setattr(training, "QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr(training, "Monitor", lambda env: env)
        monkeypatch.setattr(training, "QuantumRLAgent", DummyAgent)
        monkeypatch.setattr(training, "CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr(training, "EvalCallback", lambda *args, **kwargs: object())

        agent = training.setup_training_pipeline(
            target_circuit=QuantumCircuit(2),
            coupling_map=linear_coupling_3q,
            mode="routing",
            frontier_mode="dag",
            algorithm="PPO",
            total_timesteps=1,
            seed=11,
            log_dir=str(tmp_path / "logs"),
            model_save_dir=str(tmp_path / "models"),
            lookahead_window=6,
            max_steps=77,
        )

        metadata = json.loads(
            Path(agent.run_model_dir, "run_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["mode"] == "routing"
        assert metadata["algorithm"] == "PPO"
        assert metadata["seed"] == 11
        assert metadata["environment"]["frontier_mode"] == "dag"
        assert metadata["environment"]["lookahead_window"] == 6
        assert metadata["environment"]["max_steps"] == 77
        assert metadata["environment"]["basis_gates"] is None
        assert metadata["training"]["hyperparams"] == {
            "learning_rate": 1e-4,
            "clip_range": 0.1,
            "target_kl": 0.03,
        }
        assert metadata["evaluation"] == {
            "eval_freq": 5000,
            "n_eval_episodes": 5,
            "deterministic": True,
        }

    def test_setup_training_pipeline_preserves_explicit_ppo_hyperparams(
        self, monkeypatch, linear_coupling_3q, tmp_path
    ):
        from src.rl_module import training

        captured_hyperparams = {}

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed=None):
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, tensorboard_log=None, seed=None, **kwargs):
                captured_hyperparams.update(kwargs)

            def train(self, total_timesteps, callbacks=None, progress_bar=True):
                return None

            def save(self, path):
                Path(path).write_text("model", encoding="utf-8")

        monkeypatch.setattr(training, "QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr(training, "Monitor", lambda env: env)
        monkeypatch.setattr(training, "QuantumRLAgent", DummyAgent)
        monkeypatch.setattr(training, "CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr(training, "EvalCallback", lambda *args, **kwargs: object())

        training.setup_training_pipeline(
            target_circuit=QuantumCircuit(2),
            coupling_map=linear_coupling_3q,
            algorithm="PPO",
            total_timesteps=1,
            log_dir=str(tmp_path / "logs"),
            model_save_dir=str(tmp_path / "models"),
            hyperparams={
                "learning_rate": 3e-4,
                "clip_range": 0.2,
                "target_kl": 0.08,
                "n_epochs": 3,
            },
        )

        assert captured_hyperparams["learning_rate"] == 3e-4
        assert captured_hyperparams["clip_range"] == 0.2
        assert captured_hyperparams["target_kl"] == 0.08
        assert captured_hyperparams["n_epochs"] == 3

    def test_setup_training_pipeline_does_not_apply_ppo_defaults_to_dqn(
        self, monkeypatch, linear_coupling_3q, tmp_path
    ):
        from src.rl_module import training

        captured_hyperparams = {}

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed=None):
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, tensorboard_log=None, seed=None, **kwargs):
                captured_hyperparams.update(kwargs)

            def train(self, total_timesteps, callbacks=None, progress_bar=True):
                return None

            def save(self, path):
                Path(path).write_text("model", encoding="utf-8")

        monkeypatch.setattr(training, "QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr(training, "Monitor", lambda env: env)
        monkeypatch.setattr(training, "QuantumRLAgent", DummyAgent)
        monkeypatch.setattr(training, "CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr(training, "EvalCallback", lambda *args, **kwargs: object())

        training.setup_training_pipeline(
            target_circuit=QuantumCircuit(2),
            coupling_map=linear_coupling_3q,
            algorithm="DQN",
            total_timesteps=1,
            log_dir=str(tmp_path / "logs"),
            model_save_dir=str(tmp_path / "models"),
        )

        assert "learning_rate" not in captured_hyperparams
        assert "clip_range" not in captured_hyperparams
        assert "target_kl" not in captured_hyperparams

    def test_setup_training_pipeline_skips_learning_when_routing_completed_at_reset(
        self, monkeypatch, linear_coupling_3q, tmp_path
    ):
        from src.rl_module import training

        train_called = False

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                self.was_completed_at_reset = False

            def reset(self, seed=None, options=None):
                self.was_completed_at_reset = True
                return {}, {
                    "already_completed_at_reset": True,
                    "executed_gates": [("h", 0, 0), ("cx", 0, 1)],
                }

        class DummyAgent:
            def __init__(self, env, algorithm, tensorboard_log=None, seed=None, **kwargs):
                self.env = env

            def train(self, total_timesteps, callbacks=None, progress_bar=True):
                nonlocal train_called
                train_called = True

            def save(self, path):
                Path(path).write_text("model", encoding="utf-8")

        monkeypatch.setattr(training, "QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr(training, "Monitor", lambda env: env)
        monkeypatch.setattr(training, "QuantumRLAgent", DummyAgent)
        monkeypatch.setattr(training, "CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr(training, "EvalCallback", lambda *args, **kwargs: object())

        agent = training.setup_training_pipeline(
            target_circuit=QuantumCircuit(2),
            coupling_map=linear_coupling_3q,
            mode="routing",
            algorithm="PPO",
            total_timesteps=100,
            log_dir=str(tmp_path / "logs"),
            model_save_dir=str(tmp_path / "models"),
        )

        assert train_called is False
        assert Path(agent.last_model_path).exists()
        assert agent.best_model_path is None
        assert agent.training_skipped_reason == "routing_completed_at_reset"

    def test_setup_training_pipeline_omits_basis_gates_from_routing_metadata(
        self, monkeypatch, linear_coupling_3q, tmp_path
    ):
        from src.rl_module import training

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed=None):
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, tensorboard_log=None, seed=None, **kwargs):
                self.env = env

            def train(self, total_timesteps, callbacks=None, progress_bar=True):
                return None

            def save(self, path):
                Path(path).write_text("model", encoding="utf-8")

        monkeypatch.setattr(training, "QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr(training, "Monitor", lambda env: env)
        monkeypatch.setattr(training, "QuantumRLAgent", DummyAgent)
        monkeypatch.setattr(training, "CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr(training, "EvalCallback", lambda *args, **kwargs: object())

        agent = training.setup_training_pipeline(
            target_circuit=QuantumCircuit(2),
            coupling_map=linear_coupling_3q,
            mode="routing",
            frontier_mode="dag",
            algorithm="PPO",
            total_timesteps=1,
            seed=11,
            log_dir=str(tmp_path / "logs"),
            model_save_dir=str(tmp_path / "models"),
            lookahead_window=6,
            max_steps=77,
            basis_gates=["cz", "rz", "sx", "x"],
        )

        metadata = json.loads(
            Path(agent.run_model_dir, "run_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["mode"] == "routing"
        assert metadata["environment"]["basis_gates"] is None

    def test_setup_training_pipeline_reports_best_model_artifact_when_available(
        self, monkeypatch, simple_circuit_3q, linear_coupling_3q, tmp_path
    ):
        """Si EvalCallback guarda best_model.zip, el pipeline lo expone sin reemplazar el retorno."""
        run_dir = tmp_path / "rl_run"

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed=None):
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, tensorboard_log, **hyperparams):
                self.algorithm = algorithm
                self.best_model_path = None

            def train(self, total_timesteps, callbacks):
                best_model_path = run_dir / "best_model.zip"
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                best_model_path.write_bytes(b"best")
                return None

            def save(self, path):
                Path(path).write_bytes(b"final")

        monkeypatch.setattr("src.rl_module.training.QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr("src.rl_module.training.Monitor", lambda env: env)
        monkeypatch.setattr("src.rl_module.training.QuantumRLAgent", DummyAgent)
        monkeypatch.setattr("src.rl_module.training.CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr("src.rl_module.training.EvalCallback", lambda *args, **kwargs: object())
        monkeypatch.setattr("src.rl_module.training._make_run_dir", lambda base_dir, prefix="run": str(run_dir))

        agent = setup_training_pipeline(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            total_timesteps=1,
            model_save_dir=str(tmp_path),
        )

        assert agent.best_model_path == os.path.join(str(run_dir), "best_model.zip")

    def test_setup_training_pipeline_ignores_foreign_best_model_artifact(
        self, monkeypatch, simple_circuit_3q, linear_coupling_3q, tmp_path
    ):
        """Un best_model.zip preexistente no cuenta si el run actual no lo produce."""

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed=None):
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, tensorboard_log, **hyperparams):
                self.best_model_path = None

            def train(self, total_timesteps, callbacks):
                return None

            def save(self, path):
                Path(path).write_bytes(b"final")

        foreign_best = tmp_path / "best_model.zip"
        foreign_best.write_bytes(b"foreign")

        monkeypatch.setattr("src.rl_module.training.QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr("src.rl_module.training.Monitor", lambda env: env)
        monkeypatch.setattr("src.rl_module.training.QuantumRLAgent", DummyAgent)
        monkeypatch.setattr("src.rl_module.training.CheckpointCallback", lambda **kwargs: object())
        monkeypatch.setattr("src.rl_module.training.EvalCallback", lambda *args, **kwargs: object())

        agent = setup_training_pipeline(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            total_timesteps=1,
            model_save_dir=str(tmp_path),
        )

        assert agent.best_model_path is None

    def test_setup_training_pipeline_uses_maskable_eval_callback_for_maskable_routing(
        self, monkeypatch, simple_circuit_3q, linear_coupling_3q, tmp_path
    ):
        """Routing con MaskablePPO usa el callback de evaluacion con mascaras."""
        callback_types = []
        maskable_eval_kwargs = {}
        env_kwargs = []

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                env_kwargs.append(kwargs)

            def reset(self, seed=None):
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, tensorboard_log=None, seed=None, **kwargs):
                self.env = env

            def train(self, total_timesteps, callbacks=None, progress_bar=True):
                callback_types.extend(type(cb).__name__ for cb in callbacks)
                return None

            def save(self, path):
                Path(path).write_text("model", encoding="utf-8")

        class DummyCheckpointCallback:
            def __init__(self, **kwargs):
                pass

        class DummyEvalCallback:
            def __init__(self, *args, **kwargs):
                pass

        class DummyMaskableEvalCallback:
            def __init__(self, *args, **kwargs):
                maskable_eval_kwargs.update(kwargs)
                pass

        from src.rl_module import training

        monkeypatch.setattr(training, "QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr(training, "Monitor", lambda env: env)
        monkeypatch.setattr(training, "QuantumRLAgent", DummyAgent)
        monkeypatch.setattr(training, "CheckpointCallback", DummyCheckpointCallback)
        monkeypatch.setattr(training, "EvalCallback", DummyEvalCallback)
        monkeypatch.setattr(training, "MaskableEvalCallback", DummyMaskableEvalCallback, raising=False)

        agent = training.setup_training_pipeline(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="routing",
            algorithm="MaskablePPO",
            total_timesteps=1,
            log_dir=str(tmp_path / "logs"),
            model_save_dir=str(tmp_path / "models"),
        )

        assert callback_types == ["DummyCheckpointCallback", "DummyMaskableEvalCallback"]
        assert maskable_eval_kwargs["eval_freq"] == 5000
        assert maskable_eval_kwargs["n_eval_episodes"] == 5
        early_stopping = maskable_eval_kwargs["callback_after_eval"]
        assert early_stopping.min_evals == 50
        assert early_stopping.max_no_improvement_evals == 20
        assert [kwargs["mask_semantics"] for kwargs in env_kwargs] == [
            "frontier_restricted_edges.v2",
            "frontier_restricted_edges.v2",
        ]
        metadata = json.loads(
            Path(agent.run_model_dir, "run_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["routing_policy"]["mask_semantics"] == "frontier_restricted_edges.v2"
        assert metadata["evaluation"]["early_stopping"] == {
            "enabled": True,
            "callback": "StopTrainingOnNoModelImprovement",
            "min_evals": 50,
            "max_no_improvement_evals": 20,
        }

    def test_setup_training_pipeline_rejects_maskableppo_for_synthesis(
        self, monkeypatch, simple_circuit_3q, linear_coupling_3q
    ):
        """MaskablePPO solo esta soportado para routing y synthesis falla temprano."""
        env_constructed = False

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                nonlocal env_constructed
                env_constructed = True

        from src.rl_module import training

        monkeypatch.setattr(training, "QuantumTranspilationEnv", DummyEnv)

        with pytest.raises(ValueError, match="MaskablePPO.*routing"):
            training.setup_training_pipeline(
                target_circuit=simple_circuit_3q,
                coupling_map=linear_coupling_3q,
                mode="synthesis",
                algorithm="MaskablePPO",
                total_timesteps=1,
                basis_gates=["cz", "rz", "sx", "x"],
            )

        assert env_constructed is False


class TestRLEvaluationGUI:
    """Tests de la evaluación mostrada por la GUI RL."""

    @staticmethod
    def _load_rl_gui_module(monkeypatch):
        import importlib
        import sys
        import types

        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.use = lambda *args, **kwargs: None
        fake_pyplot = types.ModuleType("matplotlib.pyplot")
        fake_backend = types.ModuleType("matplotlib.backends.backend_tkagg")
        fake_backend.FigureCanvasTkAgg = object
        fake_ctk = types.ModuleType("customtkinter")
        fake_ctk.CTk = type("CTk", (), {})
        fake_ctk.CTkFrame = type("CTkFrame", (), {})
        fake_ctk.CTkLabel = type("CTkLabel", (), {})
        fake_ctk.CTkFont = type("CTkFont", (), {})
        fake_ctk.CTkOptionMenu = type("CTkOptionMenu", (), {})
        fake_ctk.CTkSlider = type("CTkSlider", (), {})
        fake_ctk.CTkButton = type("CTkButton", (), {})
        fake_ctk.CTkTabview = type("CTkTabview", (), {})
        fake_ctk.CTkTextbox = type("CTkTextbox", (), {})
        fake_ctk.CTkProgressBar = type("CTkProgressBar", (), {})
        fake_ctk.set_appearance_mode = lambda *args, **kwargs: None
        fake_ctk.set_default_color_theme = lambda *args, **kwargs: None

        monkeypatch.setitem(sys.modules, "customtkinter", fake_ctk)
        monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)
        monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_tkagg", fake_backend)

        sys.modules.pop("src.rl_module.gui.rl_gui", None)
        return importlib.import_module("src.rl_module.gui.rl_gui")

    def test_training_thread_passes_seed_into_agent_construction(
        self, monkeypatch, simple_circuit_3q, linear_coupling_3q
    ):
        """El flujo de entrenamiento de la GUI reenvia la seed al agente SB3."""
        rl_gui = self._load_rl_gui_module(monkeypatch)
        captured_seed = None
        reset_seeds = []
        saw_eval_callback = False
        rendered_plots = False
        progress_updated = False

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed=None):
                reset_seeds.append(seed)
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, verbose, seed=None, **kwargs):
                nonlocal captured_seed
                captured_seed = seed
                self.device = "cpu"
                self.saved_paths = []

            def train(self, total_timesteps, callbacks, progress_bar):
                nonlocal saw_eval_callback
                saw_eval_callback = any(isinstance(cb, DummyEvalCallback) for cb in callbacks)
                return None

            def save(self, path):
                self.saved_paths.append(path)

        class DummyEvalCallback:
            def __init__(self, *args, **kwargs):
                self.best_model_save_path = kwargs.get("best_model_save_path")
                self.log_path = kwargs.get("log_path")

        class DummyProgressCallback:
            def __init__(self):
                self.episode_rewards = []
                self.episode_lengths = []

        class DummyLabel:
            def configure(self, **kwargs):
                nonlocal progress_updated
                progress_updated = True

        class DummyWidget:
            def configure(self, **kwargs):
                return None

            def set(self, value):
                return None

        class DummyGUI:
            def __init__(self):
                self._agent = None
                self._env = None
                self._last_callback = None
                self._is_training = True
                self._train_button = DummyWidget()
                self._eval_button = DummyWidget()
                self._progress_label = DummyLabel()
                self._progress_bar = DummyWidget()

            def after(self, _delay, callback, *args):
                callback(*args)

            def _log(self, text):
                return None

            def _render_training_plots(self):
                nonlocal rendered_plots
                rendered_plots = True

        made_dirs = []

        def fake_makedirs(path, exist_ok=False):
            made_dirs.append(path)

        monkeypatch.setattr(rl_gui, "set_global_seeds", lambda seed: None)
        monkeypatch.setattr(rl_gui, "QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr(rl_gui, "Monitor", lambda env: env)
        monkeypatch.setattr(rl_gui, "QuantumRLAgent", DummyAgent)
        monkeypatch.setattr(rl_gui, "GUIProgressCallback", lambda gui, total_timesteps: DummyProgressCallback())
        monkeypatch.setattr(rl_gui, "EvalCallback", DummyEvalCallback)
        monkeypatch.setattr(rl_gui.os, "makedirs", fake_makedirs)
        monkeypatch.setattr(rl_gui.os.path, "exists", lambda path: False)

        gui = DummyGUI()
        cfg = {
            "seed": 77,
            "circuit": simple_circuit_3q,
            "coupling_map": linear_coupling_3q,
            "mode": "routing",
            "frontier_mode": "sequential",
            "lookahead": 2,
            "max_steps": 5,
            "algorithm": "PPO",
            "timesteps": 2,
        }

        rl_gui.RLBenchmarkGUI._training_thread(gui, cfg)

        assert captured_seed == 77
        assert reset_seeds == [77, 77]
        assert saw_eval_callback is True
        assert cfg["best_model_path"] is None
        assert cfg["last_model_path"].endswith("final_model.zip")
        assert rendered_plots is True
        assert progress_updated is True
        assert any("rl_models" in path for path in made_dirs)
        assert any("rl_logs" in path for path in made_dirs)

    def test_training_thread_labels_single_eval_as_last_model_artifact(
        self, monkeypatch, simple_circuit_3q, linear_coupling_3q
    ):
        """Con una sola evaluación al final, la GUI no lo presenta como best_model."""
        rl_gui = self._load_rl_gui_module(monkeypatch)

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed=None):
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, verbose, seed=None, **kwargs):
                self.device = "cpu"

            def train(self, total_timesteps, callbacks, progress_bar):
                return None

            def save(self, path):
                return None

        class DummyEvalCallback:
            def __init__(self, *args, **kwargs):
                pass

        class DummyProgressCallback:
            def __init__(self):
                self.episode_rewards = []
                self.episode_lengths = []

        class DummyWidget:
            def configure(self, **kwargs):
                return None

            def set(self, value):
                return None

        class DummyGUI:
            def __init__(self):
                self._agent = None
                self._env = None
                self._last_callback = None
                self._is_training = True
                self._train_button = DummyWidget()
                self._eval_button = DummyWidget()
                self._progress_label = DummyWidget()
                self._progress_bar = DummyWidget()

            def after(self, _delay, callback, *args):
                callback(*args)

            def _log(self, text):
                return None

            def _render_training_plots(self):
                return None

        monkeypatch.setattr(rl_gui, "set_global_seeds", lambda seed: None)
        monkeypatch.setattr(rl_gui, "QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr(rl_gui, "Monitor", lambda env: env)
        monkeypatch.setattr(rl_gui, "QuantumRLAgent", DummyAgent)
        monkeypatch.setattr(rl_gui, "GUIProgressCallback", lambda gui, total_timesteps: DummyProgressCallback())
        monkeypatch.setattr(rl_gui, "EvalCallback", DummyEvalCallback)
        monkeypatch.setattr(rl_gui.os, "makedirs", lambda *args, **kwargs: None)
        monkeypatch.setattr(rl_gui.os.path, "exists", lambda path: path.endswith("final_model.zip"))

        gui = DummyGUI()
        cfg = {
            "seed": 7,
            "circuit": simple_circuit_3q,
            "coupling_map": linear_coupling_3q,
            "mode": "routing",
            "frontier_mode": "sequential",
            "lookahead": 2,
            "max_steps": 5,
            "algorithm": "PPO",
            "timesteps": 1,
        }

        rl_gui.RLBenchmarkGUI._training_thread(gui, cfg)

        assert cfg["best_model_path"] is None
        assert cfg["last_model_path"].endswith("final_model.zip")

    def test_training_thread_keeps_eval_button_disabled_after_failed_training(
        self, monkeypatch, simple_circuit_3q, linear_coupling_3q
    ):
        """Si el entrenamiento falla, la GUI no debe habilitar evaluación por tener _agent no nulo."""
        rl_gui = self._load_rl_gui_module(monkeypatch)

        class DummyEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed=None):
                return {}, {}

        class DummyAgent:
            def __init__(self, env, algorithm, verbose, seed=None, **kwargs):
                self.device = "cpu"

            def train(self, total_timesteps, callbacks, progress_bar):
                raise RuntimeError("boom")

            def save(self, path):
                raise AssertionError("save should not be reached after training failure")

        class DummyProgressCallback:
            def __init__(self):
                self.episode_rewards = []
                self.episode_lengths = []

        class RecordingWidget:
            def __init__(self):
                self.states = []

            def configure(self, **kwargs):
                if "state" in kwargs:
                    self.states.append(kwargs["state"])
                return None

            def set(self, value):
                return None

        class DummyGUI:
            def __init__(self):
                self._agent = None
                self._env = None
                self._last_callback = None
                self._is_training = True
                self._train_button = RecordingWidget()
                self._eval_button = RecordingWidget()
                self._progress_label = RecordingWidget()
                self._progress_bar = RecordingWidget()

            def after(self, _delay, callback, *args):
                callback(*args)

            def _log(self, text):
                return None

            def _render_training_plots(self):
                raise AssertionError("plots should not render after training failure")

        monkeypatch.setattr(rl_gui, "set_global_seeds", lambda seed: None)
        monkeypatch.setattr(rl_gui, "QuantumTranspilationEnv", DummyEnv)
        monkeypatch.setattr(rl_gui, "Monitor", lambda env: env)
        monkeypatch.setattr(rl_gui, "QuantumRLAgent", DummyAgent)
        monkeypatch.setattr(rl_gui, "GUIProgressCallback", lambda gui, total_timesteps: DummyProgressCallback())
        monkeypatch.setattr(rl_gui, "EvalCallback", lambda *args, **kwargs: object())
        monkeypatch.setattr(rl_gui.os, "makedirs", lambda *args, **kwargs: None)
        monkeypatch.setattr(rl_gui.os.path, "exists", lambda path: False)

        gui = DummyGUI()
        cfg = {
            "seed": 11,
            "circuit": simple_circuit_3q,
            "coupling_map": linear_coupling_3q,
            "mode": "routing",
            "frontier_mode": "sequential",
            "lookahead": 2,
            "max_steps": 5,
            "algorithm": "PPO",
            "timesteps": 2,
        }

        rl_gui.RLBenchmarkGUI._training_thread(gui, cfg)

        assert gui._agent is not None
        assert gui._is_training is False
        assert gui._train_button.states[-1] == "normal"
        assert gui._eval_button.states == []

    def test_evaluation_thread_uses_deterministic_policy_by_default(self, monkeypatch, simple_circuit_3q, linear_coupling_3q):
        """La evaluación principal de la GUI usa predict(..., deterministic=True)."""
        rl_gui = self._load_rl_gui_module(monkeypatch)
        deterministic_flags = []
        eval_log_lines = []
        loaded_model_paths = []

        class DummyEvalEnv:
            def __init__(self, *args, **kwargs):
                self.current_layout = np.array([0, 1, 2], dtype=np.int32)
                self.remaining_gates = deque()
                self.total_swaps = 0

            def reset(self, seed=None):
                self.last_seed = seed
                obs = {
                    "lookahead": np.array([-1, -1], dtype=np.int32),
                    "lookahead_physical": np.array([-1, -1], dtype=np.int32),
                    "lookahead_executable": np.array([0.0], dtype=np.float32),
                    "lookahead_routing_distance": np.array([0.0], dtype=np.float32),
                    "lookahead_valid_mask": np.array([0.0], dtype=np.float32),
                }
                info = {"total_gates": 0}
                return obs, info

            def step(self, action):
                obs = {
                    "lookahead": np.array([-1, -1], dtype=np.int32),
                    "lookahead_physical": np.array([-1, -1], dtype=np.int32),
                    "lookahead_executable": np.array([0.0], dtype=np.float32),
                    "lookahead_routing_distance": np.array([0.0], dtype=np.float32),
                    "lookahead_valid_mask": np.array([0.0], dtype=np.float32),
                }
                info = {
                    "action_type": "swap",
                    "is_valid_action": True,
                    "gates_executed": 0,
                    "is_completed": True,
                }
                return obs, 1.0, True, False, info

        class DummyAgent:
            def predict(self, observation, deterministic=True):
                deterministic_flags.append(deterministic)
                return 0, None

            @classmethod
            def load(cls, path, env, algorithm="PPO", **kwargs):
                loaded_model_paths.append(path)
                return cls()

        class DummyButton:
            def configure(self, **kwargs):
                return None

        class DummyTabView:
            def set(self, tab_name):
                return None

        class DummyGUI:
            def __init__(self):
                self._is_training = False
                self._agent = DummyAgent()
                self._training_cfg = {
                    "seed": 42,
                    "circuit": simple_circuit_3q,
                    "circuit_name": "fixture",
                    "coupling_map": linear_coupling_3q,
                    "mode": "routing",
                    "frontier_mode": "sequential",
                    "lookahead": 1,
                    "max_steps": 5,
                    "algorithm": "PPO",
                    "best_model_path": os.path.join("models", "best_model.zip"),
                    "last_model_path": os.path.join("models", "final_model.zip"),
                    "run_model_dir": "models",
                }
                self._eval_button = DummyButton()
                self._tabview = DummyTabView()

            def _get_config(self):
                return self._training_cfg

            def _clear_eval_terminal(self):
                return None

            def _eval_log_write(self, text):
                eval_log_lines.append(text)

            def after(self, _delay, callback, *args):
                callback(*args)

        monkeypatch.setattr(rl_gui, "set_global_seeds", lambda seed: None)
        monkeypatch.setattr(rl_gui, "QuantumTranspilationEnv", DummyEvalEnv)
        monkeypatch.setattr(rl_gui, "QuantumRLAgent", DummyAgent)
        monkeypatch.setattr(rl_gui.os.path, "exists", lambda path: path == os.path.join("models", "best_model.zip"))

        gui = DummyGUI()
        rl_gui.RLBenchmarkGUI._evaluation_thread(gui)

        assert deterministic_flags == [True]
        assert loaded_model_paths == [os.path.join("models", "best_model.zip")]
        assert any("EVALUACIÓN DE EPISODIO (Política Determinista)" in line for line in eval_log_lines)

    def test_evaluation_thread_tolerates_gui_double_without_structured_inspector_hooks(self, monkeypatch, simple_circuit_3q, linear_coupling_3q):
        """La ruta de evaluación sigue funcionando con doubles antiguos sin _append_eval_record."""
        rl_gui = self._load_rl_gui_module(monkeypatch)
        deterministic_flags = []
        eval_log_lines = []

        class DummyEvalEnv:
            def __init__(self, *args, **kwargs):
                self.current_layout = np.array([0, 1, 2], dtype=np.int32)
                self.remaining_gates = deque()
                self.total_swaps = 0

            def reset(self, seed=None):
                obs = {
                    "lookahead": np.array([-1, -1], dtype=np.int32),
                    "lookahead_physical": np.array([-1, -1], dtype=np.int32),
                    "lookahead_executable": np.array([0.0], dtype=np.float32),
                    "lookahead_routing_distance": np.array([0.0], dtype=np.float32),
                    "lookahead_valid_mask": np.array([0.0], dtype=np.float32),
                }
                return obs, {"total_gates": 0}

            def step(self, action):
                obs = {
                    "lookahead": np.array([-1, -1], dtype=np.int32),
                    "lookahead_physical": np.array([-1, -1], dtype=np.int32),
                    "lookahead_executable": np.array([0.0], dtype=np.float32),
                    "lookahead_routing_distance": np.array([0.0], dtype=np.float32),
                    "lookahead_valid_mask": np.array([0.0], dtype=np.float32),
                }
                info = {
                    "action_type": "swap",
                    "is_valid_action": True,
                    "gates_executed": 0,
                    "is_completed": True,
                }
                return obs, 1.0, True, False, info

        class DummyAgent:
            def predict(self, observation, deterministic=True):
                deterministic_flags.append(deterministic)
                return 0, None

        class DummyButton:
            def configure(self, **kwargs):
                return None

        class DummyTabView:
            def set(self, tab_name):
                return None

        class DummyGUI:
            def __init__(self):
                self._is_training = False
                self._agent = DummyAgent()
                self._training_cfg = {
                    "seed": 42,
                    "circuit": simple_circuit_3q,
                    "circuit_name": "fixture-backward-safe",
                    "coupling_map": linear_coupling_3q,
                    "mode": "routing",
                    "frontier_mode": "sequential",
                    "lookahead": 1,
                    "max_steps": 5,
                    "algorithm": "PPO",
                    "best_model_path": None,
                    "last_model_path": None,
                    "run_model_dir": "models",
                }
                self._eval_button = DummyButton()
                self._tabview = DummyTabView()

            def _get_config(self):
                return self._training_cfg

            def _clear_eval_terminal(self):
                return None

            def _eval_log_write(self, text):
                eval_log_lines.append(text)

            def after(self, _delay, callback, *args):
                callback(*args)

        monkeypatch.setattr(rl_gui, "set_global_seeds", lambda seed: None)
        monkeypatch.setattr(rl_gui, "QuantumTranspilationEnv", DummyEvalEnv)

        gui = DummyGUI()
        rl_gui.RLBenchmarkGUI._evaluation_thread(gui)

        assert deterministic_flags == [True]
        assert any("Resultado: COMPLETADO" in line for line in eval_log_lines)
        assert not any("⚠ Error durante evaluación" in line for line in eval_log_lines)

    def test_evaluation_thread_ignores_missing_or_foreign_best_artifact(self, monkeypatch, simple_circuit_3q, linear_coupling_3q):
        """La GUI usa el último modelo del run si no hay best artifact válido del run actual."""
        rl_gui = self._load_rl_gui_module(monkeypatch)
        loaded_model_paths = []
        deterministic_flags = []

        class DummyEvalEnv:
            def __init__(self, *args, **kwargs):
                self.current_layout = np.array([0, 1, 2], dtype=np.int32)
                self.remaining_gates = deque()
                self.total_swaps = 0

            def reset(self, seed=None):
                obs = {
                    "lookahead": np.array([-1, -1], dtype=np.int32),
                    "lookahead_physical": np.array([-1, -1], dtype=np.int32),
                    "lookahead_executable": np.array([0.0], dtype=np.float32),
                    "lookahead_routing_distance": np.array([0.0], dtype=np.float32),
                    "lookahead_valid_mask": np.array([0.0], dtype=np.float32),
                }
                return obs, {"total_gates": 0}

            def step(self, action):
                obs = {
                    "lookahead": np.array([-1, -1], dtype=np.int32),
                    "lookahead_physical": np.array([-1, -1], dtype=np.int32),
                    "lookahead_executable": np.array([0.0], dtype=np.float32),
                    "lookahead_routing_distance": np.array([0.0], dtype=np.float32),
                    "lookahead_valid_mask": np.array([0.0], dtype=np.float32),
                }
                info = {
                    "action_type": "swap",
                    "is_valid_action": True,
                    "gates_executed": 0,
                    "is_completed": True,
                }
                return obs, 1.0, True, False, info

        class DummyAgent:
            def predict(self, observation, deterministic=True):
                deterministic_flags.append(deterministic)
                return 0, None

            @classmethod
            def load(cls, path, env, algorithm="PPO", **kwargs):
                loaded_model_paths.append(path)
                return cls()

        class DummyButton:
            def configure(self, **kwargs):
                return None

        class DummyTabView:
            def set(self, tab_name):
                return None

        class DummyGUI:
            def __init__(self):
                self._is_training = False
                self._agent = DummyAgent()
                self._training_cfg = {
                    "seed": 42,
                    "circuit": simple_circuit_3q,
                    "circuit_name": "fixture",
                    "coupling_map": linear_coupling_3q,
                    "mode": "routing",
                    "frontier_mode": "sequential",
                    "lookahead": 1,
                    "max_steps": 5,
                    "algorithm": "PPO",
                    "best_model_path": os.path.join("other_run", "best_model.zip"),
                    "last_model_path": os.path.join("models", "final_model.zip"),
                    "run_model_dir": "models",
                }
                self._eval_button = DummyButton()
                self._tabview = DummyTabView()

            def _get_config(self):
                return self._training_cfg

            def _clear_eval_terminal(self):
                return None

            def _eval_log_write(self, text):
                return None

            def after(self, _delay, callback, *args):
                callback(*args)

        monkeypatch.setattr(rl_gui, "set_global_seeds", lambda seed: None)
        monkeypatch.setattr(rl_gui, "QuantumTranspilationEnv", DummyEvalEnv)
        monkeypatch.setattr(rl_gui, "QuantumRLAgent", DummyAgent)
        monkeypatch.setattr(rl_gui.os.path, "exists", lambda path: path == os.path.join("models", "final_model.zip"))

        gui = DummyGUI()
        rl_gui.RLBenchmarkGUI._evaluation_thread(gui)

        assert loaded_model_paths == [os.path.join("models", "final_model.zip")]
        assert deterministic_flags == [True]

    def test_evaluation_thread_handles_synthesis_observations(self, monkeypatch):
        """La evaluación GUI no asume claves de routing cuando el modo es synthesis."""
        rl_gui = self._load_rl_gui_module(monkeypatch)
        eval_log_lines = []
        deterministic_flags = []

        class DummyEvalEnv:
            def __init__(self, *args, **kwargs):
                self.current_layout = np.array([1], dtype=np.int32)
                self.remaining_gates = deque()
                self.total_swaps = 0

            def reset(self, seed=None):
                obs = {
                    "layout": np.array([1], dtype=np.int32),
                    "physical_to_logical": np.array([-1, 0], dtype=np.int32),
                    "residual_symplectic": np.zeros(16, dtype=np.int32),
                    "residual_phase": np.zeros(4, dtype=np.int32),
                    "step_progress": np.array([0.0], dtype=np.float32),
                }
                return obs, {"total_gates": 1}

            def step(self, action):
                obs = {
                    "layout": np.array([1], dtype=np.int32),
                    "physical_to_logical": np.array([-1, 0], dtype=np.int32),
                    "residual_symplectic": np.zeros(16, dtype=np.int32),
                    "residual_phase": np.zeros(4, dtype=np.int32),
                    "step_progress": np.array([1.0], dtype=np.float32),
                }
                info = {
                    "action_type": "gate",
                    "is_valid_action": True,
                    "gates_executed": 0,
                    "is_completed": True,
                }
                return obs, 1.0, True, False, info

        class DummyAgent:
            def predict(self, observation, deterministic=True):
                deterministic_flags.append(deterministic)
                return 0, None

        class DummyButton:
            def configure(self, **kwargs):
                return None

        class DummyTabView:
            def set(self, tab_name):
                return None

        class DummyGUI:
            def __init__(self):
                self._is_training = False
                self._agent = DummyAgent()
                self._training_cfg = {
                    "seed": 42,
                    "circuit": QuantumCircuit(1),
                    "circuit_name": "fixture-synthesis",
                    "coupling_map": [(0, 1)],
                    "mode": "synthesis",
                    "basis_gates": ["cz", "rz", "sx", "x"],
                    "frontier_mode": "sequential",
                    "lookahead": 1,
                    "max_steps": 5,
                    "algorithm": "PPO",
                    "best_model_path": None,
                    "last_model_path": None,
                    "run_model_dir": "models",
                }
                self._eval_button = DummyButton()
                self._tabview = DummyTabView()

            def _get_config(self):
                return self._training_cfg

            def _clear_eval_terminal(self):
                return None

            def _eval_log_write(self, text):
                eval_log_lines.append(text)

            def after(self, _delay, callback, *args):
                callback(*args)

        monkeypatch.setattr(rl_gui, "set_global_seeds", lambda seed: None)
        monkeypatch.setattr(rl_gui, "QuantumTranspilationEnv", DummyEvalEnv)

        gui = DummyGUI()
        rl_gui.RLBenchmarkGUI._evaluation_thread(gui)

        assert deterministic_flags == [True]
        assert not any("⚠ Error durante evaluación" in line for line in eval_log_lines)
        assert any("Modo: synthesis" in line for line in eval_log_lines)
        assert any("Residual symplectic" in line for line in eval_log_lines)
        assert any("Residual phase" in line for line in eval_log_lines)

    def test_evaluation_thread_does_not_cut_off_synthesis_for_fixed_layout_revisits(self, monkeypatch):
        """La detección de ciclos visual no debe abortar synthesis solo por repetir layout fijo."""
        rl_gui = self._load_rl_gui_module(monkeypatch)
        eval_log_lines = []
        deterministic_flags = []

        class DummyEvalEnv:
            def __init__(self, *args, **kwargs):
                self.current_layout = np.array([1], dtype=np.int32)
                self.remaining_gates = deque(["pending"])
                self.total_swaps = 0
                self.step_count = 0

            def reset(self, seed=None):
                obs = {
                    "layout": np.array([1], dtype=np.int32),
                    "physical_to_logical": np.array([-1, 0], dtype=np.int32),
                    "residual_symplectic": np.zeros(16, dtype=np.int32),
                    "residual_phase": np.zeros(4, dtype=np.int32),
                    "step_progress": np.array([0.0], dtype=np.float32),
                }
                return obs, {"total_gates": 1}

            def step(self, action):
                self.step_count += 1
                if self.step_count >= 4:
                    self.remaining_gates.clear()
                obs = {
                    "layout": np.array([1], dtype=np.int32),
                    "physical_to_logical": np.array([-1, 0], dtype=np.int32),
                    "residual_symplectic": np.zeros(16, dtype=np.int32),
                    "residual_phase": np.zeros(4, dtype=np.int32),
                    "step_progress": np.array([min(self.step_count / 4, 1.0)], dtype=np.float32),
                }
                info = {
                    "action_type": "gate",
                    "is_valid_action": True,
                    "gates_executed": 0,
                    "is_completed": self.step_count >= 4,
                }
                return obs, 1.0, self.step_count >= 4, False, info

        class DummyAgent:
            def predict(self, observation, deterministic=True):
                deterministic_flags.append(deterministic)
                return 0, None

        class DummyButton:
            def configure(self, **kwargs):
                return None

        class DummyTabView:
            def set(self, tab_name):
                return None

        class DummyGUI:
            def __init__(self):
                self._is_training = False
                self._agent = DummyAgent()
                self._training_cfg = {
                    "seed": 42,
                    "circuit": QuantumCircuit(1),
                    "circuit_name": "fixture-synthesis-cycle",
                    "coupling_map": [(0, 1)],
                    "mode": "synthesis",
                    "basis_gates": ["cz", "rz", "sx", "x"],
                    "frontier_mode": "sequential",
                    "lookahead": 1,
                    "max_steps": 10,
                    "algorithm": "PPO",
                    "best_model_path": None,
                    "last_model_path": None,
                    "run_model_dir": "models",
                }
                self._eval_button = DummyButton()
                self._tabview = DummyTabView()

            def _get_config(self):
                return self._training_cfg

            def _clear_eval_terminal(self):
                return None

            def _eval_log_write(self, text):
                eval_log_lines.append(text)

            def after(self, _delay, callback, *args):
                callback(*args)

        monkeypatch.setattr(rl_gui, "set_global_seeds", lambda seed: None)
        monkeypatch.setattr(rl_gui, "QuantumTranspilationEnv", DummyEvalEnv)

        gui = DummyGUI()
        rl_gui.RLBenchmarkGUI._evaluation_thread(gui)

        assert deterministic_flags == [True, True, True, True]
        assert any("Resultado: COMPLETADO" in line for line in eval_log_lines)
        assert not any("CICLO DETECTADO" in line for line in eval_log_lines)
        assert any("Steps totales: 4" in line for line in eval_log_lines)



# ===========================================================================
#  Tests — training: Pipeline mínimo
# ===========================================================================

class TestTrainingPipeline:
    """Tests del pipeline de entrenamiento (con timesteps mínimos)."""

    def test_agent_short_training(self, routing_env):
        """Un entrenamiento corto de 64 timesteps completa sin error."""
        agent = QuantumRLAgent(env=routing_env, algorithm="PPO", verbose=0)
        model = agent.train(total_timesteps=64, progress_bar=False)
        assert model is not None

    def test_agent_predict_after_training(self, routing_env):
        """Tras entrenar, el agente puede predecir acciones."""
        agent = QuantumRLAgent(env=routing_env, algorithm="PPO", verbose=0)
        agent.train(total_timesteps=64, progress_bar=False)
        obs, _ = routing_env.reset(seed=42)
        action, _ = agent.predict(obs)
        assert routing_env.action_space.contains(action)


# ===========================================================================
#  Tests — Abstract base classes
# ===========================================================================

class TestAbstractClasses:
    """Tests de que las clases base abstractas no se instancian directamente."""

    def test_reward_strategy_is_abstract(self):
        """RewardStrategy no se puede instanciar directamente."""
        with pytest.raises(TypeError):
            RewardStrategy()

    def test_rlenvsrategy_is_abstract(self):
        """RLEnvStrategy no se puede instanciar directamente."""
        with pytest.raises(TypeError):
            RLEnvStrategy(num_qubits=3, coupling_map=[], lookahead_window=5)
