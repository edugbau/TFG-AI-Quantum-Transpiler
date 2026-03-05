"""
Estrategias de Entorno para el Agente RL (Patrón Strategy).

Este módulo define cómo se construyen los espacios de observación y de acción,
así como la lógica de transición (step) dependiendo de si el agente está en modo
"Routing" (sólo inserta SWAPs) o "Synthesis" (sintetiza el circuito entero).
"""

from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, List

class RLEnvStrategy(ABC):
    """
    Clase base para inyectar la lógica de estado/acción en el entorno principal.
    Permite escalar fácilmente añadiendo nuevas estrategias (ej. Synthesis).
    """
    def __init__(self, num_qubits: int, coupling_map: List[Tuple[int, int]], lookahead_window: int):
        self.num_qubits = num_qubits
        self.coupling_map = coupling_map
        self.lookahead_window = lookahead_window

    @abstractmethod
    def get_observation_space(self) -> gym.Space:
        pass

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        pass

    @abstractmethod
    def build_observation(self, current_layout: np.ndarray, remaining_gates: List[Tuple[str, int, int]]) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def decode_action(self, action: Any) -> Dict[str, Any]:
        """Decodifica la acción devuelta por el agente de RL en una operación lógica (ej. Swap en arista K)"""
        pass

class RoutingStrategy(RLEnvStrategy):
    """
    Estrategia de Enrutamiento. 
    Acción: Elegir una arista (conexión física en el coupling map) para insertar un SWAP.
    Observación: Layout actual y las próximas N puertas lógicas (lookahead).
    """
    def __init__(self, num_qubits: int, coupling_map: List[Tuple[int, int]], lookahead_window: int):
        super().__init__(num_qubits, coupling_map, lookahead_window)
        # Limpiamos el coupling map (evitamos aristas duplicadas si es bidireccional para los SWAPs)
        self.edges = list(set([tuple(sorted(edge)) for edge in coupling_map]))
        self.num_edges = len(self.edges)

    def get_observation_space(self) -> gym.Space:
        """
        Observación estructurada en Dict:
        - 'layout': Mapeo de qubits lógicos a físicos. (Array de tamaño num_qubits)
        - 'lookahead': Vector aplanado de las próximas puertas pendientes.
          Cada puerta se codifica como [qubit_logico_control, qubit_logico_target].
        """
        return gym.spaces.Dict({
            'layout': gym.spaces.Box(low=0, high=self.num_qubits - 1, shape=(self.num_qubits,), dtype=np.int32),
            # Para la ventana, codificamos cada puerta como un par de qubits lógicos (control, target)
            # Rellenamos con -1 si no hay suficientes puertas
            'lookahead': gym.spaces.Box(low=-1, high=self.num_qubits - 1, shape=(self.lookahead_window * 2,), dtype=np.int32)
        })

    def get_action_space(self) -> gym.Space:
        """Una acción discreta: Elegir una arista del Coupling Map para realizar un SWAP."""
        return gym.spaces.Discrete(self.num_edges)

    def build_observation(self, current_layout: np.ndarray, remaining_gates: List[Tuple[str, int, int]]) -> Dict[str, np.ndarray]:
        lookahead_array = np.full(self.lookahead_window * 2, -1, dtype=np.int32)
        
        # Rellenar la ventana con las próximas puertas pendientes
        # Se asume que las puertas en remaining_gates son de la forma (gate_name, qubit_logico_1, qubit_logico_2)
        for i, gate in enumerate(remaining_gates[:self.lookahead_window]):
            if len(gate) == 3: # 2-qubit gate (name, q1, q2)
                lookahead_array[i*2] = gate[1]
                lookahead_array[i*2 + 1] = gate[2]
            else: # 1-qubit gate (para completitud, el target y control son iguales)
                lookahead_array[i*2] = gate[1]
                lookahead_array[i*2 + 1] = gate[1]

        return {
            'layout': current_layout.copy(),
            'lookahead': lookahead_array
        }

    def decode_action(self, action: int) -> Dict[str, Any]:
        if action < 0 or action >= self.num_edges:
            return {"type": "invalid"}
        
        edge = self.edges[action]
        return {
            "type": "swap",
            "physical_q1": edge[0],
            "physical_q2": edge[1]
        }

class SynthesisStrategy(RLEnvStrategy):
    """
    Estrategia de Síntesis Completa. (Plantilla escalable)
    Acción: Elegir una puerta de la base (ej. CX, RX, RZ) y los qubits físicos donde aplicarla.
    """
    def __init__(self, num_qubits: int, coupling_map: List[Tuple[int, int]], lookahead_window: int, basis_gates: List[str] = ['cx', 'sx', 'rz', 'x']):
        super().__init__(num_qubits, coupling_map, lookahead_window)
        self.basis_gates = basis_gates
        # Aquí el espacio de acciones se define combinando las puertas y los qubits posibles.
        # Por simplicidad conceptual, dejaremos un espacio multi-discreto.
        
    def get_observation_space(self) -> gym.Space:
        # Observación más compleja para síntesis (ej. Tableau estabilizador o equivalente)
        # Por ahora devolvemos lo mismo que routing como scaffolding.
        return gym.spaces.Dict({
            'layout': gym.spaces.Box(low=0, high=self.num_qubits - 1, shape=(self.num_qubits,), dtype=np.int32),
            'lookahead': gym.spaces.Box(low=-1, high=self.num_qubits - 1, shape=(self.lookahead_window * 2,), dtype=np.int32)
        })

    def get_action_space(self) -> gym.Space:
        # MultiDiscrete: [Seleccionar_puerta, q_fisico_1, q_fisico_2]
        # (q_fisico_2 se ignora si la puerta es de 1 qubit)
        return gym.spaces.MultiDiscrete([len(self.basis_gates), self.num_qubits, self.num_qubits])

    def build_observation(self, current_layout: np.ndarray, remaining_gates: List[Tuple[str, int, int]]) -> Dict[str, np.ndarray]:
        # Scaffolding de momento idéntico al routing.
        lookahead_array = np.full(self.lookahead_window * 2, -1, dtype=np.int32)
        for i, gate in enumerate(remaining_gates[:self.lookahead_window]):
             if len(gate) == 3:
                lookahead_array[i*2] = gate[1]
                lookahead_array[i*2 + 1] = gate[2]
        return {
            'layout': current_layout.copy(),
            'lookahead': lookahead_array
        }

    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        gate_idx, pq1, pq2 = action
        return {
            "type": "gate",
            "gate_name": self.basis_gates[gate_idx],
            "physical_q1": pq1,
            "physical_q2": pq2
        }
