"""
Estrategias de Entorno para el Agente RL (Patrón Strategy).

Este módulo define cómo se construyen los espacios de observación y de acción,
así como la lógica de transición (step) dependiendo de si el agente está en modo
"Routing" (sólo inserta SWAPs) o "Synthesis" (sintetiza el circuito entero).
"""

from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, List, Union, Optional
from collections import deque

class RLEnvStrategy(ABC):
    """
    Clase base para inyectar la lógica de estado/acción en el entorno principal.
    Permite escalar fácilmente añadiendo nuevas estrategias (ej. Synthesis).
    """
    def __init__(
        self,
        num_qubits: int,
        num_physical_qubits: Optional[int] = None,
        coupling_map: Optional[List[Tuple[int, int]]] = None,
        lookahead_window: int = 10,
    ):
        self.num_qubits = num_qubits
        self.num_physical_qubits = num_qubits if num_physical_qubits is None else num_physical_qubits
        self.coupling_map = coupling_map or []
        self.lookahead_window = lookahead_window
        self._adjacency = self._build_adjacency(self.coupling_map)

    def _build_adjacency(self, coupling_map: List[Tuple[int, int]]) -> Dict[int, set[int]]:
        adjacency: Dict[int, set[int]] = {qubit: set() for qubit in range(self.num_physical_qubits)}
        for q1, q2 in coupling_map:
            adjacency.setdefault(q1, set()).add(q2)
            adjacency.setdefault(q2, set()).add(q1)
        return adjacency

    def _shortest_path_length(self, start: int, goal: int) -> Optional[int]:
        if start == goal:
            return 0

        visited = {start}
        frontier = deque([(start, 0)])

        while frontier:
            node, distance = frontier.popleft()
            for neighbor in self._adjacency.get(node, ()):
                if neighbor == goal:
                    return distance + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, distance + 1))

        return None

    def _iter_visible_entries(
        self,
        current_layout: np.ndarray,
        remaining_gates: Union[List[Tuple[str, int, int]], deque, Any],
    ) -> List[Dict[str, Any]]:
        if hasattr(remaining_gates, "get_visible_entries"):
            return [
                {
                    "logical_q1": entry.logical_q1,
                    "logical_q2": entry.logical_q2,
                    "physical_q1": entry.physical_q1,
                    "physical_q2": entry.physical_q2,
                    "executable": entry.executable,
                }
                for entry in remaining_gates.get_visible_entries(
                    current_layout=current_layout,
                    lookahead_window=self.lookahead_window,
                    is_connected=self._is_connected,
                )
            ]

        visible_entries: List[Dict[str, Any]] = []
        for gate_name, logical_q1, logical_q2 in list(remaining_gates)[:self.lookahead_window]:
            physical_q1 = int(current_layout[logical_q1])
            physical_q2 = int(current_layout[logical_q2])
            visible_entries.append(
                {
                    "logical_q1": logical_q1,
                    "logical_q2": logical_q2,
                    "physical_q1": physical_q1,
                    "physical_q2": physical_q2,
                    "executable": logical_q1 == logical_q2 or self._is_connected(physical_q1, physical_q2),
                }
            )

        return visible_entries

    def _is_connected(self, physical_q1: int, physical_q2: int) -> bool:
        return physical_q2 in self._adjacency.get(physical_q1, set())

    @abstractmethod
    def get_observation_space(self) -> gym.Space:
        pass

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        pass

    def build_observation(
        self,
        current_layout: np.ndarray,
        remaining_gates: Union[List[Tuple[str, int, int]], deque],
        step_progress: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Construye la observación que se pasa al agente.

        Codifica el layout actual y una ventana (lookahead) con las
        próximas puertas pendientes.  Cada puerta se representa como
        un par ``(qubit_lógico_1, qubit_lógico_2)``.  Las puertas de
        un solo qubit se codifican como ``(q, q)``.

        Si hay menos puertas que ``lookahead_window``, las posiciones
        restantes se rellenan con ``-1``.

        Parameters
        ----------
        step_progress : float
            Valor normalizado en [0, 1] que indica cuánto del episodio
            ha transcurrido (``current_step / max_steps``).  Proporciona
            al agente **contexto temporal** para distinguir estados
            idénticos visitados en momentos distintos del episodio,
            rompiendo oscilaciones A→B→A.
        """
        lookahead_array = np.full(self.lookahead_window * 2, -1, dtype=np.int32)
        lookahead_physical = np.full(self.lookahead_window * 2, -1, dtype=np.int32)
        lookahead_executable = np.zeros(self.lookahead_window, dtype=np.float32)
        lookahead_routing_distance = np.zeros(self.lookahead_window, dtype=np.float32)
        lookahead_valid_mask = np.zeros(self.lookahead_window, dtype=np.float32)

        visible_entries = self._iter_visible_entries(current_layout, remaining_gates)

        for i, entry in enumerate(visible_entries):
            logical_q1 = entry["logical_q1"]
            logical_q2 = entry["logical_q2"]
            physical_q1 = entry["physical_q1"]
            physical_q2 = entry["physical_q2"]
            executable = entry["executable"]

            lookahead_array[i * 2] = logical_q1
            lookahead_array[i * 2 + 1] = logical_q2
            lookahead_physical[i * 2] = physical_q1
            lookahead_physical[i * 2 + 1] = physical_q2
            lookahead_executable[i] = 1.0 if executable else 0.0
            lookahead_valid_mask[i] = 1.0

            if logical_q1 != logical_q2 and not executable:
                path_length = self._shortest_path_length(physical_q1, physical_q2)
                if path_length is not None:
                    lookahead_routing_distance[i] = float(max(path_length - 1, 0))
                else:
                    lookahead_routing_distance[i] = -1.0

        return {
            'layout': current_layout.copy(),
            'lookahead': lookahead_array,
            'lookahead_physical': lookahead_physical,
            'lookahead_executable': lookahead_executable,
            'lookahead_routing_distance': lookahead_routing_distance,
            'lookahead_valid_mask': lookahead_valid_mask,
            'step_progress': np.array([step_progress], dtype=np.float32),
        }

    @abstractmethod
    def decode_action(self, action: Any) -> Dict[str, Any]:
        """Decodifica la acción devuelta por el agente de RL en una operación lógica (ej. Swap en arista K)"""

class RoutingStrategy(RLEnvStrategy):
    """
    Estrategia de Enrutamiento. 
    Acción: Elegir una arista (conexión física en el coupling map) para insertar un SWAP.
    Observación: Layout actual y las próximas N puertas lógicas (lookahead).
    """
    def __init__(self, num_qubits: int, num_physical_qubits: Optional[int] = None, coupling_map: Optional[List[Tuple[int, int]]] = None, lookahead_window: int = 10):
        super().__init__(num_qubits, num_physical_qubits, coupling_map, lookahead_window)
        # Limpiamos el coupling map (evitamos aristas duplicadas si es bidireccional para los SWAPs)
        # Orden determinista con sorted() para reproducibilidad
        self.edges = sorted(set(tuple(sorted(edge)) for edge in self.coupling_map))
        self.num_edges = len(self.edges)

    def get_observation_space(self) -> gym.Space:
        """
        Observación estructurada en Dict:
        - 'layout': Mapeo de qubits lógicos a físicos. (Array de tamaño num_qubits)
        - 'lookahead': Vector aplanado de las próximas puertas pendientes.
          Cada puerta se codifica como [qubit_logico_control, qubit_logico_target].
        """
        max_distance = float(max(self.num_physical_qubits - 1, 0))
        return gym.spaces.Dict({
            'layout': gym.spaces.Box(low=-1, high=self.num_physical_qubits - 1, shape=(self.num_qubits,), dtype=np.int32),
            # Para la ventana, codificamos cada puerta como un par de qubits lógicos (control, target)
            # Rellenamos con -1 si no hay suficientes puertas
            'lookahead': gym.spaces.Box(low=-1, high=self.num_qubits - 1, shape=(self.lookahead_window * 2,), dtype=np.int32),
            'lookahead_physical': gym.spaces.Box(low=-1, high=self.num_physical_qubits - 1, shape=(self.lookahead_window * 2,), dtype=np.int32),
            'lookahead_executable': gym.spaces.Box(low=0.0, high=1.0, shape=(self.lookahead_window,), dtype=np.float32),
            'lookahead_routing_distance': gym.spaces.Box(low=-1.0, high=max_distance, shape=(self.lookahead_window,), dtype=np.float32),
            'lookahead_valid_mask': gym.spaces.Box(low=0.0, high=1.0, shape=(self.lookahead_window,), dtype=np.float32),
            # Progreso temporal normalizado [0, 1] para romper oscilaciones
            'step_progress': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def get_action_space(self) -> gym.Space:
        """Una acción discreta: Elegir una arista del Coupling Map para realizar un SWAP."""
        return gym.spaces.Discrete(self.num_edges)

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
    def __init__(self, num_qubits: int, num_physical_qubits: Optional[int] = None, coupling_map: Optional[List[Tuple[int, int]]] = None, lookahead_window: int = 10, basis_gates: List[str] = ['cx', 'sx', 'rz', 'x']):
        super().__init__(num_qubits, num_physical_qubits, coupling_map, lookahead_window)
        self.basis_gates = basis_gates
        
    def get_observation_space(self) -> gym.Space:
        # Observación más compleja para síntesis (ej. Tableau estabilizador o equivalente)
        # Por ahora devolvemos lo mismo que routing como scaffolding.
        max_distance = float(max(self.num_physical_qubits - 1, 0))
        return gym.spaces.Dict({
            'layout': gym.spaces.Box(low=-1, high=self.num_physical_qubits - 1, shape=(self.num_qubits,), dtype=np.int32),
            'lookahead': gym.spaces.Box(low=-1, high=self.num_qubits - 1, shape=(self.lookahead_window * 2,), dtype=np.int32),
            'lookahead_physical': gym.spaces.Box(low=-1, high=self.num_physical_qubits - 1, shape=(self.lookahead_window * 2,), dtype=np.int32),
            'lookahead_executable': gym.spaces.Box(low=0.0, high=1.0, shape=(self.lookahead_window,), dtype=np.float32),
            'lookahead_routing_distance': gym.spaces.Box(low=-1.0, high=max_distance, shape=(self.lookahead_window,), dtype=np.float32),
            'lookahead_valid_mask': gym.spaces.Box(low=0.0, high=1.0, shape=(self.lookahead_window,), dtype=np.float32),
            'step_progress': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def get_action_space(self) -> gym.Space:
        # MultiDiscrete: [Seleccionar_puerta, q_fisico_1, q_fisico_2]
        # (q_fisico_2 se ignora si la puerta es de 1 qubit)
        return gym.spaces.MultiDiscrete([len(self.basis_gates), self.num_physical_qubits, self.num_physical_qubits])

    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Decodifica una acción MultiDiscrete en una operación de puerta.

        Valida que los índices estén dentro de rango. Si no lo están,
        retorna una acción de tipo ``"invalid"``.
        """
        gate_idx, pq1, pq2 = int(action[0]), int(action[1]), int(action[2])

        if gate_idx < 0 or gate_idx >= len(self.basis_gates):
            return {"type": "invalid"}
        if pq1 < 0 or pq1 >= self.num_physical_qubits or pq2 < 0 or pq2 >= self.num_physical_qubits:
            return {"type": "invalid"}

        return {
            "type": "gate",
            "gate_name": self.basis_gates[gate_idx],
            "physical_q1": pq1,
            "physical_q2": pq2
        }
