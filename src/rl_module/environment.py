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
from typing import Optional, Tuple, Dict, Any, List, Mapping, MutableSequence
from .env_strategies import RoutingStrategy, SynthesisStrategy
from .frontier import DagFrontier, FrontierProvider, GateTuple, LookaheadEntry, SequentialFrontier
from .rewards import RoutingReward, SynthesisReward
from .routing_mask import (
    FRONTIER_RESTRICTED_EDGES_V2,
    FRONTIER_RESTRICTED_EDGES_V3,
    RoutingMaskConfig,
    normalize_mask_semantics,
    resolve_routing_mask_config,
)
from .synthesis_clifford import CliffordSynthesisState
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
        basis_gates: Optional[List[str]] = None,
        mask_semantics: Optional[str] = None,
        routing_mask_config: RoutingMaskConfig | Mapping[str, Any] | None = None,
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
        self.basis_gates = list(basis_gates) if basis_gates is not None else None
        self.mask_semantics = normalize_mask_semantics(mask_semantics)
        self.routing_mask_config = (
            resolve_routing_mask_config(routing_mask_config, num_qubits=self.num_qubits)
            if self.mask_semantics == FRONTIER_RESTRICTED_EDGES_V3
            else None
        )
        self._synthesis_state: Optional[CliffordSynthesisState] = None
        
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
        self._recent_layout_signatures: deque[tuple[int, ...]] = deque(maxlen=4)
        self._last_swap_edge: Optional[Tuple[int, int]] = None
        cycle_window = (
            self.routing_mask_config.cycle_window
            if self.routing_mask_config is not None
            else 1
        )
        self._recent_routing_states: deque[tuple[tuple[int, ...], int]] = deque(
            maxlen=cycle_window
        )
        self._frontier_revision = 0
        self._steps_without_progress = 0
        self._best_routing_signal_for_frontier = 0.0

    @property
    def remaining_gates(self) -> deque:
        if (
            self.mode == "synthesis"
            and self._synthesis_state is not None
            and self._synthesis_state.is_complete()
        ):
            return deque()
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

    def _is_valid_synthesis_primitive(self, primitive: Any) -> bool:
        for physical_qubit in primitive.physical_qargs:
            if int(self._inverse_layout[physical_qubit]) == -1:
                return False
        if len(primitive.physical_qargs) == 2:
            pq1, pq2 = primitive.physical_qargs
            return self._is_connected(pq1, pq2)
        return True

    def _apply_synthesis_primitive(self, primitive: Any, info: Dict[str, Any]) -> None:
        before = self._synthesis_state.residual_distance()
        info["primitive_name"] = primitive.gate_name
        info["primitive_physical_qargs"] = primitive.physical_qargs
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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reinicia el entorno.
        Permite inyectar un `initial_layout` externo a través de `options`.
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_swaps = 0
        self._recent_layout_signatures.clear()
        self._recent_routing_states.clear()
        self._last_swap_edge = None
        self._frontier_revision = 0
        self._steps_without_progress = 0
        self._best_routing_signal_for_frontier = 0.0
        extracted_gates = self._extract_gates_from_circuit()
        self._frontier = self._build_frontier(extracted_gates)
        
        # Ingesta genérica de layout inicial desde el llamador
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

        if self.mode == "synthesis":
            self._reset_synthesis_state()
            self.was_completed_at_reset = self._synthesis_state.is_complete()
            reset_executed_gates: list[GateTuple] = []
        else:
            self._synthesis_state = None
            reset_executed_gates = []
            self._try_execute_front_layer(executed_gates=reset_executed_gates)
            self.was_completed_at_reset = self._frontier.remaining_gate_count == 0

        obs = self._build_observation(step_progress=0.0)
        if self.mode == "routing":
            self._frontier_revision = len(reset_executed_gates)
            self._best_routing_signal_for_frontier = self._routing_signal(obs)
            self._record_routing_state()
        info = {
            "initial_layout_loaded": (options is not None and "initial_layout" in options),
            "total_gates": len(extracted_gates),
            "already_completed_at_reset": self.was_completed_at_reset,
            "executed_gates": list(reset_executed_gates),
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

    def _try_execute_front_layer(
        self,
        executed_gates: MutableSequence[GateTuple] | None = None,
    ) -> int:
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
            executed_gates=executed_gates,
        )

    def get_visible_frontier_entries(self) -> List[LookaheadEntry]:
        if self.mode != "routing":
            return []
        return self._frontier.get_visible_entries(
            current_layout=self.current_layout,
            lookahead_window=self.lookahead_window,
            is_connected=self._is_connected,
        )

    def action_masks(self) -> np.ndarray:
        if self.mode != "routing":
            raise AttributeError("action_masks is only available in routing mode.")

        strategy = self.strategy
        if not isinstance(strategy, RoutingStrategy):
            raise AttributeError("action_masks requires a RoutingStrategy.")

        mask = self._build_frontier_incident_mask(strategy=strategy)
        if self.mask_semantics in {FRONTIER_RESTRICTED_EDGES_V2, FRONTIER_RESTRICTED_EDGES_V3}:
            mask = self._apply_anti_undo_or_fallback(mask, strategy=strategy)
        if self.mask_semantics == FRONTIER_RESTRICTED_EDGES_V3:
            mask = self._apply_short_cycle_filter_or_fallback(mask, strategy=strategy)
            mask = self._apply_sabre_top_k_or_fallback(mask, strategy=strategy)
        return mask

    def _build_frontier_incident_mask(self, *, strategy: RoutingStrategy) -> np.ndarray:
        blocked_entries = self._frontier.get_blocked_two_qubit_entries(
            current_layout=self.current_layout,
            is_connected=self._is_connected,
        )
        non_empty_edge_mask = np.array(
            [
                not (self._inverse_layout[pq1] == -1 and self._inverse_layout[pq2] == -1)
                for pq1, pq2 in strategy.edges
            ],
            dtype=bool,
        )

        if not blocked_entries:
            return np.ones(strategy.num_edges, dtype=bool)

        mask = np.zeros(strategy.num_edges, dtype=bool)

        for entry in blocked_entries:
            entry_mask = np.zeros(strategy.num_edges, dtype=bool)
            relevant_physical_qubits = {entry.physical_q1, entry.physical_q2}

            for index, (pq1, pq2) in enumerate(strategy.edges):
                if pq1 not in relevant_physical_qubits and pq2 not in relevant_physical_qubits:
                    continue
                if not non_empty_edge_mask[index]:
                    continue
                entry_mask[index] = True

            if not np.any(entry_mask):
                raise ValueError(
                    "Masked routing state/config is unroutable: blocked frontier gate has no incident valid routing edges."
                )

            mask |= entry_mask

        return mask

    def _apply_anti_undo_or_fallback(
        self,
        mask: np.ndarray,
        *,
        strategy: RoutingStrategy,
    ) -> np.ndarray:
        if self._last_swap_edge is None:
            return mask

        filtered_mask = mask.copy()
        for index, edge in enumerate(strategy.edges):
            if edge == self._last_swap_edge:
                filtered_mask[index] = False
                break

        # A narrow graph may require an immediate undo to remain routable.
        return filtered_mask if np.any(filtered_mask) else mask

    def _simulate_swap_layout(self, edge: Tuple[int, int]) -> np.ndarray:
        pq1, pq2 = edge
        candidate_layout = self.current_layout.copy()
        lq1 = int(self._inverse_layout[pq1])
        lq2 = int(self._inverse_layout[pq2])
        if lq1 != -1:
            candidate_layout[lq1] = pq2
        if lq2 != -1:
            candidate_layout[lq2] = pq1
        return candidate_layout

    def _candidate_unlocks_frontier(self, candidate_layout: np.ndarray) -> bool:
        blocked_entries = self._frontier.get_blocked_two_qubit_entries(
            current_layout=self.current_layout,
            is_connected=self._is_connected,
        )
        return any(
            self._is_connected(
                int(candidate_layout[entry.logical_q1]),
                int(candidate_layout[entry.logical_q2]),
            )
            for entry in blocked_entries
        )

    def _apply_short_cycle_filter_or_fallback(
        self,
        mask: np.ndarray,
        *,
        strategy: RoutingStrategy,
    ) -> np.ndarray:
        filtered_mask = mask.copy()
        recent_states = set(self._recent_routing_states)
        for index in np.flatnonzero(mask):
            candidate_layout = self._simulate_swap_layout(strategy.edges[int(index)])
            if self._candidate_unlocks_frontier(candidate_layout):
                continue
            signature = (
                tuple(int(value) for value in candidate_layout.tolist()),
                self._frontier_revision,
            )
            if signature in recent_states:
                filtered_mask[int(index)] = False
        return filtered_mask if np.any(filtered_mask) else mask

    def _routing_entries_cost(
        self,
        entries: List[LookaheadEntry],
        *,
        strategy: RoutingStrategy,
    ) -> float:
        distances = [
            strategy.routing_distance_cost(entry.physical_q1, entry.physical_q2)
            for entry in entries
            if entry.logical_q1 != entry.logical_q2
        ]
        return float(np.mean(distances)) if distances else 0.0

    def _score_sabre_candidate(
        self,
        candidate_layout: np.ndarray,
        *,
        strategy: RoutingStrategy,
    ) -> float:
        frontier_entries, extended_entries = self._frontier.get_sabre_entries(
            current_layout=candidate_layout,
            lookahead_window=self.lookahead_window,
            is_connected=self._is_connected,
        )
        return self._routing_entries_cost(
            frontier_entries,
            strategy=strategy,
        ) + self.routing_mask_config.sabre_lookahead_weight * self._routing_entries_cost(
            extended_entries,
            strategy=strategy,
        )

    def _apply_sabre_top_k_or_fallback(
        self,
        mask: np.ndarray,
        *,
        strategy: RoutingStrategy,
    ) -> np.ndarray:
        top_k = self.routing_mask_config.sabre_top_k
        enabled_indices = [int(index) for index in np.flatnonzero(mask)]
        if top_k is None or len(enabled_indices) <= top_k:
            return mask

        productive_indices: list[int] = []
        scored_indices: list[tuple[float, int]] = []
        for index in enabled_indices:
            candidate_layout = self._simulate_swap_layout(strategy.edges[index])
            if self._candidate_unlocks_frontier(candidate_layout):
                productive_indices.append(index)
                continue
            scored_indices.append(
                (
                    self._score_sabre_candidate(candidate_layout, strategy=strategy),
                    index,
                )
            )

        keep_indices = set(productive_indices)
        remaining_budget = max(top_k - len(keep_indices), 0)
        keep_indices.update(
            index
            for _, index in sorted(scored_indices)[:remaining_budget]
        )
        if not keep_indices:
            return mask

        filtered_mask = np.zeros_like(mask)
        for index in keep_indices:
            filtered_mask[index] = True
        return filtered_mask if np.any(filtered_mask) else mask

    def _layout_signature(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self.current_layout.tolist())

    def _routing_state_signature(self) -> tuple[tuple[int, ...], int]:
        return self._layout_signature(), self._frontier_revision

    def _record_routing_state(self) -> None:
        if self.mask_semantics == FRONTIER_RESTRICTED_EDGES_V3:
            self._recent_routing_states.append(self._routing_state_signature())

    def _routing_signal(self, obs: Dict[str, np.ndarray]) -> float:
        distances = np.asarray(obs.get("lookahead_routing_distance", []), dtype=np.float32)
        if distances.size == 0:
            return 0.0

        valid_mask = np.asarray(obs.get("lookahead_valid_mask", []), dtype=np.float32)
        unreachable_penalty = float(self.num_physical_qubits)
        sanitized = np.where(distances < 0.0, unreachable_penalty, distances)
        return float(np.sum(sanitized * valid_mask, dtype=np.float32))

    def _update_routing_stagnation(
        self,
        *,
        current_routing_signal: float,
        gates_executed: int,
    ) -> None:
        if self.mask_semantics != FRONTIER_RESTRICTED_EDGES_V3:
            return
        if gates_executed > 0:
            self._steps_without_progress = 0
            self._best_routing_signal_for_frontier = current_routing_signal
            return
        epsilon = self.routing_mask_config.distance_improvement_epsilon
        if current_routing_signal < self._best_routing_signal_for_frontier - epsilon:
            self._steps_without_progress = 0
            self._best_routing_signal_for_frontier = current_routing_signal
            return
        self._steps_without_progress += 1

    def _routing_stagnation_patience(self) -> int | None:
        if self.routing_mask_config is None:
            return None
        return self.routing_mask_config.stagnation_patience

    def step(self, action: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        if self.mode == "synthesis" and getattr(self, "was_completed_at_reset", False):
            prev_obs = self._build_observation(
                step_progress=(self.current_step - 1) / self.max_steps,
            )
            obs = self._build_observation(step_progress=self.current_step / self.max_steps)
            info = {
                "action_type": "terminal_noop",
                "is_valid_action": False,
                "gates_executed": 0,
                "executed_gates": [],
                "swap_edge": None,
                "is_completed": False,
                "repeated_layout": False,
                "undo_swap": False,
                "unproductive_swap": False,
                "routing_progress_delta": 0.0,
                "primitive_name": None,
                "primitive_physical_qargs": (),
                "primitive_cost": 0.0,
                "residual_distance_before": 0,
                "residual_distance_after": 0,
                "residual_distance_delta": 0.0,
                "is_truncated": self.current_step >= self.max_steps,
                "steps_without_progress": 0,
                "stagnation_patience": None,
                "truncation_reason": "max_steps" if self.current_step >= self.max_steps else None,
            }
            reward = self.reward_function.compute_reward(prev_obs, action, obs, info)
            return obs, reward, True, info["is_truncated"], info
        
        # 1. Decodificar Acción usando la estrategia
        action_info = self.strategy.decode_action(action)
        prev_obs = self._build_observation(
            step_progress=(self.current_step - 1) / self.max_steps,
        )
        prev_routing_signal = self._routing_signal(prev_obs) if self.mode == "routing" else 0.0
        prev_layout_signature = self._layout_signature()
         
        info = {
            "action_type": action_info.get("type"),
            "is_valid_action": True,
            "gates_executed": 0,
            "executed_gates": [],
            "swap_edge": None,
            "is_completed": False,
            "repeated_layout": False,
            "undo_swap": False,
            "unproductive_swap": False,
            "routing_progress_delta": 0.0,
            "steps_without_progress": self._steps_without_progress,
            "stagnation_patience": self._routing_stagnation_patience(),
            "truncation_reason": None,
        }
        
        # 2. Aplicar la lógica según el tipo de acción
        if action_info["type"] == "swap":
            pq1 = action_info["physical_q1"]
            pq2 = action_info["physical_q2"]
            swap_edge = tuple(sorted((pq1, pq2)))
            info["swap_edge"] = swap_edge
            info["undo_swap"] = self._last_swap_edge == swap_edge
            layout_changed = False
            
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

            current_layout_signature = self._layout_signature()
            layout_changed = current_layout_signature != prev_layout_signature
            info["repeated_layout"] = (
                current_layout_signature == prev_layout_signature
                or current_layout_signature in self._recent_layout_signatures
            )
            self._recent_layout_signatures.append(prev_layout_signature)
            if info["is_valid_action"] and layout_changed:
                self._last_swap_edge = swap_edge
            else:
                info["undo_swap"] = False
                self._last_swap_edge = None
            
            self.total_swaps += 1
            
            # 3. Intentar ejecutar puertas pendientes
            info["gates_executed"] = self._try_execute_front_layer(
                executed_gates=info["executed_gates"],
            )
            self._frontier_revision += info["gates_executed"]
            if info["gates_executed"] > 0:
                info["undo_swap"] = False
                info["repeated_layout"] = False
                self._last_swap_edge = None
        
        elif action_info["type"] == "gate":
            self._apply_synthesis_primitive(action_info["primitive"], info)
            self._last_swap_edge = None
            
        elif action_info["type"] == "invalid":
             info["is_valid_action"] = False
             self._last_swap_edge = None
              
        # Evaluar estado final
        if self.mode == "synthesis":
            terminated = self._synthesis_state.is_complete()
        else:
            terminated = self._frontier.remaining_gate_count == 0
        truncated = False

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
        obs = self._build_observation(step_progress=self.current_step / self.max_steps)
        if self.mode == "routing":
            current_routing_signal = self._routing_signal(obs)
            routing_progress_delta = prev_routing_signal - current_routing_signal
            if info["gates_executed"] > 0:
                routing_progress_delta = max(routing_progress_delta, 0.0)
            info["routing_progress_delta"] = routing_progress_delta
            info["unproductive_swap"] = (
                info["action_type"] == "swap"
                and info["is_valid_action"]
                and info["gates_executed"] == 0
                and routing_progress_delta <= 0.0
            )
            self._update_routing_stagnation(
                current_routing_signal=current_routing_signal,
                gates_executed=info["gates_executed"],
            )
            self._record_routing_state()
        else:
            info.setdefault("primitive_name", None)
            info.setdefault("primitive_physical_qargs", ())
            info.setdefault("primitive_cost", 0.0)
            info.setdefault("residual_distance_before", 0)
            info.setdefault("residual_distance_after", 0)
            info.setdefault("residual_distance_delta", 0.0)

        stagnation_patience = self._routing_stagnation_patience()
        hit_max_steps = self.current_step >= self.max_steps
        hit_stagnation = (
            self.mode == "routing"
            and stagnation_patience is not None
            and self._steps_without_progress >= stagnation_patience
        )
        truncated = not terminated and (hit_max_steps or hit_stagnation)
        if truncated:
            info["truncation_reason"] = "max_steps" if hit_max_steps else "stagnation"
        info["steps_without_progress"] = self._steps_without_progress
        info["stagnation_patience"] = stagnation_patience
        info["is_truncated"] = truncated

        reward = self.reward_function.compute_reward(prev_obs, action, obs, info)
        
        return obs, reward, terminated, truncated, info

    def render(self):
        # FIX #3: render_mode ahora siempre está inicializado
        if self.render_mode == "human":
            print(f"Step: {self.current_step} | Swaps: {self.total_swaps} | Remaining Gates: {len(self.remaining_gates)}")
            print(f"Current Layout (Logical->Physical): {self.current_layout}")
