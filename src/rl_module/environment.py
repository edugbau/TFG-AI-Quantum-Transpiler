"""
Entorno Principal de Gymnasium para la Transpilación y Síntesis Cuántica.

Integra el Patrón Strategy para soportar el Módulo RL ("Routing" y "Synthesis")
y se comunica con Qiskit 2.3 para validar el estado del circuito.
"""

import logging
import warnings
from collections import deque
import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from .env_strategies import RoutingStrategy, SynthesisStrategy
from .rewards import RoutingReward, SynthesisReward
from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)

class QuantumTranspilationEnv(gym.Env):
    """
    Entorno Gymnasium compatible con Stable-Baselines3.

    Convención de layouts
    ---------------------
    Se mantienen **dos** arrays complementarios de tamaño ``num_qubits``:

    ``current_layout[logical_qubit]  = physical_qubit``
        La posición ``i`` indica el qubit lógico ``i`` y el valor
        almacenado es el qubit físico donde está mapeado.  Se usa
        para traducir las puertas lógicas a posiciones físicas.

    ``_inverse_layout[physical_qubit] = logical_qubit``
        Mapa inverso: dado un qubit físico, obtiene qué qubit lógico
        reside allí.  Necesario para ejecutar SWAPs en O(1), ya que
        al intercambiar dos posiciones físicas necesitamos saber qué
        qubits lógicos están implicados.

    Ambos arrays se actualizan de forma sincronizada en cada SWAP.
    """
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self, 
        target_circuit: QuantumCircuit, 
        coupling_map: List[Tuple[int, int]], 
        mode: str = "routing",
        lookahead_window: int = 10,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.target_circuit = target_circuit
        self.num_qubits = target_circuit.num_qubits
        self.coupling_map_list = coupling_map
        self.mode = mode
        self.lookahead_window = lookahead_window
        self.max_steps = max_steps
        
        # Precomputar set bidireccional para búsquedas O(1)
        self._coupling_set: set = set(coupling_map) | {(b, a) for a, b in coupling_map}
        
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
        
        # Layout directo e inverso (ver docstring de la clase)
        self.current_layout: np.ndarray = np.arange(self.num_qubits, dtype=np.int32)
        self._inverse_layout: np.ndarray = np.arange(self.num_qubits, dtype=np.int32)

        # Cola de puertas pendientes — deque para popleft() en O(1)
        self.remaining_gates: deque = deque()
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
            else:
                # FIX #11: Avisar sobre puertas de 3+ qubits
                warnings.warn(
                    f"Puerta '{gate_name}' con {len(qargs)} qubits ignorada. "
                    "Solo se soportan puertas de 1 y 2 qubits.",
                    stacklevel=2,
                )
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
        self.remaining_gates = deque(self._extract_gates_from_circuit())
        
        # Integración con el Módulo MO: Inyectar layout
        if options and "initial_layout" in options:
            layout = np.array(options["initial_layout"], dtype=np.int32)
            # Validar longitud
            if len(layout) != self.num_qubits:
                raise ValueError(
                    f"initial_layout tiene longitud {len(layout)}, "
                    f"se esperan {self.num_qubits} qubits."
                )
            # Validar duplicados
            if len(set(layout.tolist())) != len(layout):
                raise ValueError(
                    f"initial_layout contiene qubits duplicados: {layout.tolist()}"
                )
            # Validar rango [0, num_qubits)
            if layout.min() < 0 or layout.max() >= self.num_qubits:
                raise ValueError(
                    f"initial_layout contiene valores fuera del rango "
                    f"[0, {self.num_qubits}): {layout.tolist()}"
                )
            self.current_layout = layout
        else:
            # Layout trivial por defecto [0, 1, 2, ..., n]
            self.current_layout = np.arange(self.num_qubits, dtype=np.int32)
        
        # Reconstruir el mapa inverso: _inverse_layout[physical] = logical
        self._inverse_layout = np.empty(self.num_qubits, dtype=np.int32)
        for lq in range(self.num_qubits):
            self._inverse_layout[self.current_layout[lq]] = lq

        # Ejecutar puertas ya satisfechas por el layout inicial
        self._try_execute_front_layer()

        obs = self.strategy.build_observation(self.current_layout, self.remaining_gates)
        info = {
            "initial_layout_loaded": (options is not None and "initial_layout" in options),
            "total_gates": len(self.remaining_gates)
        }
        
        return obs, info

    def _get_physical_qubit(self, logical_q: int) -> int:
        """
        Retorna el qubit físico donde está mapeado un qubit lógico.

        Convención: current_layout[logical] = physical  →  acceso directo O(1).
        """
        # FIX #1: Acceso directo O(1) en lugar de np.where O(n)
        return int(self.current_layout[logical_q])

    def _is_connected(self, pq1: int, pq2: int) -> bool:
        """Verifica si dos qubits físicos están conectados en el Coupling Map."""
        # FIX #4: Búsqueda O(1) en set precomputado
        return (pq1, pq2) in self._coupling_set

    def _try_execute_front_layer(self) -> int:
        """
        Ejecuta todas las puertas consecutivas de la cabeza (front layer) cuyos
        qubits lógicos ya están mapeados a qubits físicos conectados.

        Evalúa en cascada: si la primera puerta se ejecuta, revisa la siguiente,
        y así sucesivamente hasta encontrar una puerta bloqueada.
        """
        executed_count = 0
        progress = True
        while progress and self.remaining_gates:
            progress = False
            gate = self.remaining_gates[0]
            gate_name, lq1, lq2 = gate
            pq1 = self._get_physical_qubit(lq1)
            pq2 = self._get_physical_qubit(lq2)
            
            # Puertas de 1 qubit siempre se pueden ejecutar. Puertas de 2 requieren conectividad.
            if lq1 == lq2 or self._is_connected(pq1, pq2):
                self.remaining_gates.popleft()  # O(1) con deque
                executed_count += 1
                progress = True
            # Si la primera puerta está bloqueada, detenemos
            
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
        
        # 2. Aplicar la lógica según el tipo de acción
        if action_info["type"] == "swap":
            pq1 = action_info["physical_q1"]
            pq2 = action_info["physical_q2"]
            
            # ── SWAP usando el mapa inverso ──────────────────────────
            #  _inverse_layout[physical] = logical   →   O(1) lookup
            #
            #  Dado que queremos intercambiar los contenidos de dos
            #  posiciones *físicas*, necesitamos:
            #    1. Saber qué qubit lógico reside en cada posición física
            #       (lectura directa de _inverse_layout).
            #    2. Intercambiar los valores en current_layout (directo)
            #       y en _inverse_layout (inverso).
            # ─────────────────────────────────────────────────────────
            lq1 = int(self._inverse_layout[pq1])  # lógico en posición física pq1
            lq2 = int(self._inverse_layout[pq2])  # lógico en posición física pq2

            # Actualizar layout directo: current_layout[logical] = physical
            self.current_layout[lq1], self.current_layout[lq2] = (
                self.current_layout[lq2], self.current_layout[lq1]
            )
            # Actualizar layout inverso: _inverse_layout[physical] = logical
            self._inverse_layout[pq1], self._inverse_layout[pq2] = (
                self._inverse_layout[pq2], self._inverse_layout[pq1]
            )
            
            self.total_swaps += 1
            
            # 3. Intentar ejecutar puertas pendientes
            info["gates_executed"] = self._try_execute_front_layer()
        
        elif action_info["type"] == "gate":
            # ── PLACEHOLDER: Modo Synthesis ──────────────────────────
            #  La lógica de síntesis aún no está implementada.
            #  Consultar docs/synthesis_mode_status.md para el estado
            #  actual y las opciones de diseño pendientes de decisión.
            #
            #  Cuando se implemente, este bloque debería:
            #    - Aplicar la puerta al circuito sintetizado.
            #    - Comparar con el circuito target (unitaria, tableau, etc.).
            #    - Actualizar remaining_gates o un indicador de fidelidad.
            #    - Fijar gate_matched_target según corresponda.
            # ─────────────────────────────────────────────────────────
            logger.debug(
                "Synthesis gate action recibida (placeholder): %s",
                action_info.get("gate_name"),
            )
            info["gate_matched_target"] = False
            
        elif action_info["type"] == "invalid":
             info["is_valid_action"] = False
             
        # Evaluar estado final
        terminated = len(self.remaining_gates) == 0
        truncated = self.current_step >= self.max_steps
        
        if terminated:
            info["is_completed"] = True

        # Marcar si el episodio fue truncado (para la función de recompensa)
        info["is_truncated"] = truncated
            
        # 4. Calcular Recompensa
        obs = self.strategy.build_observation(self.current_layout, self.remaining_gates)
        reward = self.reward_function.compute_reward(prev_obs, action, obs, info)
        
        return obs, reward, terminated, truncated, info

    def render(self):
        # FIX #3: render_mode ahora siempre está inicializado
        if self.render_mode == "human":
            print(f"Step: {self.current_step} | Swaps: {self.total_swaps} | Remaining Gates: {len(self.remaining_gates)}")
            print(f"Current Layout (Logical->Physical): {self.current_layout}")
