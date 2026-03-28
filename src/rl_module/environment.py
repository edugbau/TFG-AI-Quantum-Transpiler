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
from .frontier import DagFrontier, FrontierProvider, GateTuple, SequentialFrontier
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
        frontier_mode: str = "sequential",
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
        self.frontier_mode = frontier_mode
        self.lookahead_window = lookahead_window
        self.max_steps = max_steps
        
        # Determinar el número de qubits físicos a partir del coupling map
        if self.coupling_map_list:
            inferred_physical_qubits = max(max(a, b) for a, b in self.coupling_map_list) + 1
            self.num_physical_qubits = max(inferred_physical_qubits, self.num_qubits)
        else:
            self.num_physical_qubits = self.num_qubits
            
        # Precomputar set bidireccional para búsquedas O(1)
        self._coupling_set: set = set(coupling_map) | {(b, a) for a, b in coupling_map}
        
        # 1. Inicializar la Estrategia (Action/Observation Spaces)
        if self.mode == "routing":
            self.strategy = RoutingStrategy(self.num_qubits, self.num_physical_qubits, self.coupling_map_list, self.lookahead_window)
            self.reward_function = RoutingReward()
        elif self.mode == "synthesis":
            self.strategy = SynthesisStrategy(self.num_qubits, self.num_physical_qubits, self.coupling_map_list, self.lookahead_window)
            self.reward_function = SynthesisReward()
        else:
            raise ValueError(f"Modo '{self.mode}' no soportado. Usa 'routing' o 'synthesis'.")
            
        self.action_space = self.strategy.get_action_space()
        self.observation_space = self.strategy.get_observation_space()
        
        # Layout directo e inverso
        # Se inicializan a tamaño de hardware físico para permitir hacer SWAPs libres
        self.current_layout: np.ndarray = np.full(self.num_qubits, -1, dtype=np.int32)
        self._inverse_layout: np.ndarray = np.full(self.num_physical_qubits, -1, dtype=np.int32)

        if self.frontier_mode not in {"sequential", "dag"}:
            raise ValueError(
                f"frontier_mode '{self.frontier_mode}' no soportado. Usa 'sequential' o 'dag'."
            )

        self._frontier: FrontierProvider = SequentialFrontier()
        self.current_step = 0
        self.total_swaps = 0

    @property
    def remaining_gates(self) -> deque:
        remaining = self._frontier.remaining_gates
        if isinstance(remaining, deque):
            return remaining
        return deque(remaining)

    @remaining_gates.setter
    def remaining_gates(self, gates: Any) -> None:
        self._frontier = SequentialFrontier(gates)

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

    def _build_frontier(self, extracted_gates: List[GateTuple]) -> FrontierProvider:
        if self.frontier_mode == "dag":
            return DagFrontier.from_circuit(self.target_circuit)
        return SequentialFrontier(extracted_gates)

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
        extracted_gates = self._extract_gates_from_circuit()
        self._frontier = self._build_frontier(extracted_gates)
        
        # Integración con el Módulo MO: Inyectar layout
        if options and "initial_layout" in options:
            # Layout es [q_logical] -> q_physical
            layout = np.array(options["initial_layout"], dtype=np.int32)
            # Validar longitud
            if len(layout) != self.num_qubits:
                raise ValueError(
                    f"initial_layout tiene longitud {len(layout)}, "
                    f"se esperan {self.num_qubits} qubits logicos."
                )
            # Validar duplicados
            if len(set(layout.tolist())) != len(layout):
                raise ValueError(
                    f"initial_layout contiene qubits duplicados: {layout.tolist()}"
                )
            # Validar rango [0, num_physical_qubits)
            if layout.min() < 0 or layout.max() >= self.num_physical_qubits:
                raise ValueError(
                    f"initial_layout contiene valores fuera del rango "
                    f"físico [0, {self.num_physical_qubits}): {layout.tolist()}"
                )
            self.current_layout = layout
        else:
            # Layout trivial lógico->físico por defecto [0, 1, 2, ..., n-1]
            self.current_layout = np.arange(self.num_qubits, dtype=np.int32)
        
        # Reconstruir el mapa inverso: _inverse_layout[physical] = logical
        self._inverse_layout = np.full(self.num_physical_qubits, -1, dtype=np.int32)
        for lq in range(self.num_qubits):
            self._inverse_layout[self.current_layout[lq]] = lq

        # Ejecutar puertas ya satisfechas por el layout inicial
        initial_gates_executed = self._try_execute_front_layer()

        # Guardar en info si ya estaba completado al iniciar
        self.was_completed_at_reset = self._frontier.remaining_gate_count == 0

        obs = self.strategy.build_observation(
            self.current_layout, self._frontier,
            step_progress=self.current_step / self.max_steps,
        )
        info = {
            "initial_layout_loaded": (options is not None and "initial_layout" in options),
            "total_gates": len(extracted_gates),
            "already_completed_at_reset": self.was_completed_at_reset
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
        return self._frontier.execute_ready_cascade(
            current_layout=self.current_layout,
            is_connected=self._is_connected,
            cascade_successors=True,
        )

    def step(self, action: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        
        # 1. Decodificar Acción usando la estrategia
        action_info = self.strategy.decode_action(action)
        prev_obs = self.strategy.build_observation(
            self.current_layout, self._frontier,
            step_progress=(self.current_step - 1) / self.max_steps,
        )
        
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
            #  Si la posición física está vacía, su valor lógico es -1.
            # ─────────────────────────────────────────────────────────
            lq1 = int(self._inverse_layout[pq1])  # lógico en posición física pq1 (puede ser -1)
            lq2 = int(self._inverse_layout[pq2])  # lógico en posición física pq2 (puede ser -1)

            if lq1 == -1 and lq2 == -1:
                # Mover dos nodos vacíos no aporta nada lógico y congela la observación
                info["is_valid_action"] = False
            else:
                # Actualizar layout directo current_layout[logical] = physical
                if lq1 != -1:
                    self.current_layout[lq1] = pq2
                if lq2 != -1:
                    self.current_layout[lq2] = pq1

                # Actualizar layout inverso: _inverse_layout[physical] = logical
                self._inverse_layout[pq1] = lq2
                self._inverse_layout[pq2] = lq1
            
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
        terminated = self._frontier.remaining_gate_count == 0
        truncated = self.current_step >= self.max_steps

        # Solo dar is_completed si acaba de terminar AHORA (y no lo estaba en el reset)
        if terminated and not getattr(self, "was_completed_at_reset", False):
            info["is_completed"] = True
        elif terminated and getattr(self, "was_completed_at_reset", False):
            # Si ya estaba terminado en el reset, forzamos la finalización de la partida 
            # sin entregar un "is_completed" que activaría el bonus masivo repetidamente.
            info["is_completed"] = False

        # Marcar si el episodio fue truncado (para la función de recompensa)
        info["is_truncated"] = truncated
            
        # 4. Calcular Recompensa
        obs = self.strategy.build_observation(
            self.current_layout, self._frontier,
            step_progress=self.current_step / self.max_steps,
        )
        reward = self.reward_function.compute_reward(prev_obs, action, obs, info)
        
        return obs, reward, terminated, truncated, info

    def render(self):
        # FIX #3: render_mode ahora siempre está inicializado
        if self.render_mode == "human":
            print(f"Step: {self.current_step} | Swaps: {self.total_swaps} | Remaining Gates: {len(self.remaining_gates)}")
            print(f"Current Layout (Logical->Physical): {self.current_layout}")
