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

import pytest
import numpy as np
import os
import tempfile
import warnings
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
        max_steps=50,
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
    """Tests de la estrategia de síntesis (SynthesisStrategy)."""

    def test_observation_space_structure(self, linear_coupling_3q):
        """El observation space tiene las claves esperadas."""
        strategy = SynthesisStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        obs_space = strategy.get_observation_space()
        assert isinstance(obs_space, gym.spaces.Dict)
        assert "layout" in obs_space.spaces
        assert "lookahead" in obs_space.spaces
        assert "lookahead_physical" in obs_space.spaces
        assert "lookahead_executable" in obs_space.spaces
        assert "lookahead_routing_distance" in obs_space.spaces
        assert "lookahead_valid_mask" in obs_space.spaces

    def test_action_space_is_multi_discrete(self, linear_coupling_3q):
        """El action space es MultiDiscrete con 3 dimensiones."""
        strategy = SynthesisStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        action_space = strategy.get_action_space()
        assert isinstance(action_space, gym.spaces.MultiDiscrete)
        # [num_basis_gates, num_qubits, num_qubits]
        assert len(action_space.nvec) == 3

    def test_action_space_dimensions_match_config(self, linear_coupling_3q):
        """Las dimensiones del MultiDiscrete coinciden con basis_gates y num_qubits."""
        basis = ["cx", "sx", "rz", "x"]
        strategy = SynthesisStrategy(
            num_qubits=3, coupling_map=linear_coupling_3q,
            lookahead_window=5, basis_gates=basis
        )
        action_space = strategy.get_action_space()
        assert action_space.nvec[0] == len(basis)  # 4 puertas
        assert action_space.nvec[1] == 3           # 3 qubits
        assert action_space.nvec[2] == 3           # 3 qubits

    def test_decode_action_returns_gate(self, linear_coupling_3q):
        """decode_action retorna tipo 'gate' con el nombre de puerta correcto."""
        strategy = SynthesisStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        action = np.array([0, 0, 1])  # Primera puerta base (cx), qubits 0 y 1
        result = strategy.decode_action(action)
        assert result["type"] == "gate"
        assert result["gate_name"] == "cx"
        assert result["physical_q1"] == 0
        assert result["physical_q2"] == 1

    def test_decode_action_invalid_gate_index(self, linear_coupling_3q):
        """decode_action con gate_idx fuera de rango retorna 'invalid'."""
        strategy = SynthesisStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        action = np.array([99, 0, 1])
        result = strategy.decode_action(action)
        assert result["type"] == "invalid"

    def test_decode_action_invalid_qubit_index(self, linear_coupling_3q):
        """decode_action con qubit fuera de rango retorna 'invalid'."""
        strategy = SynthesisStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        action = np.array([0, 0, 99])
        result = strategy.decode_action(action)
        assert result["type"] == "invalid"

    def test_build_observation_in_space(self, linear_coupling_3q):
        """La observación generada pertenece al observation space."""
        strategy = SynthesisStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        layout = np.array([0, 1, 2], dtype=np.int32)
        gates = [("cx", 0, 1)]
        obs = strategy.build_observation(layout, gates)
        assert strategy.get_observation_space().contains(obs)

    def test_build_observation_handles_single_qubit_gates(self, linear_coupling_3q):
        """SynthesisStrategy codifica puertas de 1-qubit como (q, q)."""
        strategy = SynthesisStrategy(num_qubits=3, coupling_map=linear_coupling_3q, lookahead_window=5)
        layout = np.array([0, 1, 2], dtype=np.int32)
        # Puerta de 1 qubit codificada como (name, q, q)
        gates = [("h", 0, 0), ("cx", 0, 1)]
        obs = strategy.build_observation(layout, gates)
        # Primera puerta: H en qubit 0 → (0, 0)
        assert obs["lookahead"][0] == 0
        assert obs["lookahead"][1] == 0
        # Segunda puerta: CX(0,1) → (0, 1)
        assert obs["lookahead"][2] == 0
        assert obs["lookahead"][3] == 1


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


# ===========================================================================
#  Tests — rewards: SynthesisReward
# ===========================================================================

class TestSynthesisReward:
    """Tests de la función de recompensa de síntesis."""

    def test_valid_gate_reward(self):
        """Puerta que contribuye a la síntesis genera recompensa positiva."""
        reward_fn = SynthesisReward(valid_gate_reward=2.0)
        info = {"gate_matched_target": True, "is_completed": False, "is_truncated": False}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == 2.0

    def test_incorrect_gate_penalty(self):
        """Puerta que no contribuye genera penalización."""
        reward_fn = SynthesisReward(incorrect_gate_penalty=-1.0)
        info = {"gate_matched_target": False, "is_completed": False, "is_truncated": False}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == -1.0

    def test_synthesis_completion_bonus(self):
        """Completar la síntesis genera bonificación."""
        reward_fn = SynthesisReward(completion_bonus=100.0)
        info = {"gate_matched_target": True, "is_completed": True, "is_truncated": False}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == pytest.approx(2.0 + 100.0)  # default valid + completion

    def test_incorrect_gate_no_completion(self):
        """Puerta incorrecta sin completar solo penaliza."""
        reward_fn = SynthesisReward(incorrect_gate_penalty=-3.0, completion_bonus=100.0)
        info = {"gate_matched_target": False, "is_completed": False, "is_truncated": False}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == -3.0

    def test_synthesis_truncation_penalty(self):
        """Truncar la síntesis sin completar genera penalización configurable."""
        reward_fn = SynthesisReward(truncation_penalty=-30.0)
        info = {"gate_matched_target": False, "is_completed": False, "is_truncated": True}
        reward = reward_fn.compute_reward(None, None, None, info)
        assert reward == pytest.approx(-1.0 + -30.0)  # default incorrect + truncation


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
        )
        assert isinstance(env.strategy, SynthesisStrategy)
        assert isinstance(env.reward_function, SynthesisReward)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)

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

    def test_synthesis_mode_step_handles_gate_action(self, simple_circuit_3q, linear_coupling_3q):
        """El modo synthesis maneja acciones 'gate' sin error."""
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="synthesis",
            max_steps=10,
        )
        obs, _ = env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            # Las acciones gate deben llevar gate_matched_target en info
            if info.get("action_type") == "gate":
                assert "gate_matched_target" in info
            if terminated or truncated:
                break

    def test_synthesis_mode_gate_action_with_invalid_indices(self, simple_circuit_3q, linear_coupling_3q):
        """Las acciones inválidas en modo synthesis se manejan correctamente."""
        env = QuantumTranspilationEnv(
            target_circuit=simple_circuit_3q,
            coupling_map=linear_coupling_3q,
            mode="synthesis",
            max_steps=10,
        )
        env.reset(seed=42)
        # No hay acción OOB porque SB3 genera acciones dentro del espacio,
        # pero el decode_action ahora tiene validación
        strategy = env.strategy
        result = strategy.decode_action(np.array([99, 0, 0]))
        assert result["type"] == "invalid"


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
