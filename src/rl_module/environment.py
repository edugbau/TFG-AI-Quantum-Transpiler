"""
Entorno Principal de Gymnasium para la Transpilación y Síntesis Cuántica.

Integra el Patrón Strategy para soportar el Módulo RL ("Routing" y "Synthesis")
y se comunica con Qiskit 2.3 para validar el estado del circuito.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from .env_strategies import RoutingStrategy, SynthesisStrategy
from .rewards import RoutingReward, SynthesisReward
from qiskit import QuantumCircuit

class QuantumTranspilationEnv(gym.Env):
    """
    Entorno Gymnasium compatible con Stable-Baselines3.
    """
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self, 
        target_circuit: QuantumCircuit, 
        coupling_map: List[Tuple[int, int]], 
        mode: str = "routing",
        lookahead_window: int = 10,
        max_steps: int = 1000
    ):
        super().__init__()
        
        self.target_circuit = target_circuit
        self.num_qubits = target_circuit.num_qubits
        self.coupling_map_list = coupling_map
        self.mode = mode
        self.lookahead_window = lookahead_window
        self.max_steps = max_steps
        
        # 1. Inicializar la Estrategia (Action/Observation Spaces)
        if self.mode == "routing":
            self.strategy = RoutingStrategy(self.num_qubits, self.coupling_map_list, self.lookahead_window)
            self.reward_function = RoutingReward()
        elif self.mode == "synthesis":
            self.strategy = SynthesisStrategy(self.num_qubits, self.coupling_map_list, self.lookahead_window)
            self.reward_function = SynthesisReward()
        else:
            raise ValueError(f"Modo '{self.mode}' no soportado. Usa 'routing' o 'synthesis'.")
            
        self.action_space = self.strategy.get_action_space()
        self.observation_space = self.strategy.get_observation_space()
        
        self.current_layout: np.ndarray = np.arange(self.num_qubits, dtype=np.int32)
        self.remaining_gates: List[Tuple[str, int, int]] = []
        self.current_step = 0
        self.total_swaps = 0

    def _extract_gates_from_circuit(self) -> List[Tuple[str, int, int]]:
        """
        Extrae las puertas lógicas secuencialmente. 
        En una implementación completa, aquí se usaría qiskit.converters.circuit_to_dag 
        para extraer la 'front_layer' topológica. Para este scaffold iteramos la lista de instrucciones.
        """
        gates = []
        for instruction in self.target_circuit.data:
            gate_name = instruction.operation.name
            qargs = [self.target_circuit.find_bit(q).index for q in instruction.qubits]
            if len(qargs) == 2:
                gates.append((gate_name, qargs[0], qargs[1]))
            elif len(qargs) == 1:
                gates.append((gate_name, qargs[0], qargs[0]))
        return gates

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reinicia el entorno.
        Permite inyectar el 'initial_layout' desde el Módulo MO (Multi-Objective) a través de 'options'.
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_swaps = 0
        self.remaining_gates = self._extract_gates_from_circuit()
        
        # Integración con el Módulo MO: Inyectar layout
        if options and "initial_layout" in options:
            self.current_layout = np.array(options["initial_layout"], dtype=np.int32)
        else:
            # Layout trivial por defecto [0, 1, 2, ..., n]
            self.current_layout = np.arange(self.num_qubits, dtype=np.int32)
            
        obs = self.strategy.build_observation(self.current_layout, self.remaining_gates)
        info = {
            "initial_layout_loaded": (options is not None and "initial_layout" in options),
            "total_gates": len(self.remaining_gates)
        }
        
        return obs, info

    def _get_physical_qubit(self, logical_q: int) -> int:
        """Encuentra en qué qubit físico se aloja un qubit lógico bajo el layout actual."""
        return int(np.where(self.current_layout == logical_q)[0][0])

    def _is_connected(self, pq1: int, pq2: int) -> bool:
        """Verifica si dos qubits físicos están conectados en el Coupling Map."""
        return (pq1, pq2) in self.coupling_map_list or (pq2, pq1) in self.coupling_map_list

    def _try_execute_front_layer(self) -> int:
        """
        Verifica cuántas puertas de la cabeza (front layer) pueden ejecutarse 
        ahora mismo porque sus qubits lógicos ya están mapeados a qubits físicos conectados.
        """
        executed_count = 0
        gates_to_remove = []
        
        for idx, gate in enumerate(self.remaining_gates):
            # Para evitar bloquear por DAG en esta maqueta secuencial, solo miramos el primero
            if idx > 0: 
                break 
                
            gate_name, lq1, lq2 = gate
            pq1 = self._get_physical_qubit(lq1)
            pq2 = self._get_physical_qubit(lq2)
            
            # Puertas de 1 qubit siempre se pueden ejecutar. Puertas de 2 requieren conectividad.
            if lq1 == lq2 or self._is_connected(pq1, pq2):
                gates_to_remove.append(gate)
                executed_count += 1
            else:
                # Si la primera puerta está bloqueada, detenemos (en un DAG real, evaluaríamos más puertas de la front-layer)
                break
                
        for g in gates_to_remove:
            self.remaining_gates.remove(g)
            
        return executed_count

    def step(self, action: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        
        # 1. Decodificar Acción usando la estrategia
        action_info = self.strategy.decode_action(action)
        prev_obs = self.strategy.build_observation(self.current_layout, self.remaining_gates)
        
        info = {
            "action_type": action_info.get("type"),
            "is_valid_action": True,
            "gates_executed": 0,
            "is_completed": False
        }
        
        # 2. Aplicar la lógica en el Layout (Routing Mode)
        if action_info["type"] == "swap":
            pq1 = action_info["physical_q1"]
            pq2 = action_info["physical_q2"]
            
            # Realizar el SWAP en el array de mapeo
            idx1 = np.where(self.current_layout == pq1)[0][0]
            idx2 = np.where(self.current_layout == pq2)[0][0]
            self.current_layout[idx1], self.current_layout[idx2] = self.current_layout[idx2], self.current_layout[idx1]
            
            self.total_swaps += 1
            
            # 3. Intentar ejecutar puertas pendientes
            info["gates_executed"] = self._try_execute_front_layer()
        
        elif action_info["type"] == "invalid":
             info["is_valid_action"] = False
             
        # Evaluar estado final
        terminated = len(self.remaining_gates) == 0
        truncated = self.current_step >= self.max_steps
        
        if terminated:
            info["is_completed"] = True
            
        # 4. Calcular Recompensa
        obs = self.strategy.build_observation(self.current_layout, self.remaining_gates)
        reward = self.reward_function.compute_reward(prev_obs, action, obs, info)
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step} | Swaps: {self.total_swaps} | Remaining Gates: {len(self.remaining_gates)}")
            print(f"Current Layout (Logical->Physical): {self.current_layout}")
