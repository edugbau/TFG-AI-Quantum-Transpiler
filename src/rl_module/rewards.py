"""
Sistema de recompensas para el agente de Aprendizaje por Refuerzo.

Provee un sistema de recompensas basado en estrategias para soportar de manera
escalable tanto la fase de enrutamiento (Routing) como la de síntesis (Synthesis).

Todas las penalizaciones y bonificaciones son **configurables** a través del
constructor de cada clase.  Los valores por defecto son razonables para
empezar, pero deben ajustarse según el circuito y la topología.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class RewardStrategy(ABC):
    """
    Clase base abstracta para la función de recompensa.
    Permite escalar a diferentes funciones de fitness fácilmente.
    """
    
    @abstractmethod
    def compute_reward(self, prev_state: Any, action: Any, current_state: Any, info: Dict[str, Any]) -> float:
        """
        Calcula la recompensa basada en la transición de estado.
        
        Args:
            prev_state: El estado u observación antes de la acción.
            action: La acción tomada por el agente.
            current_state: El nuevo estado tras la acción.
            info: Diccionario con metadatos del entorno (ej. puertas ejecutadas).
                  Claves relevantes:
                  - ``action_type``: ``"swap"`` | ``"gate"`` | ``"invalid"``
                  - ``is_valid_action``: ``bool``
                  - ``gates_executed``: ``int``
                  - ``unproductive_swap``: ``bool``
                  - ``is_completed``: ``bool``
                  - ``is_truncated``: ``bool`` — ``True`` si se agotó ``max_steps``
                  - ``termination_reason``: ``"stagnation"`` si el episodio
                    termina por falta de progreso
                  - ``remaining_gates``: puertas objetivo pendientes al cerrar
                    el episodio
                  
        Returns:
            float: Recompensa escalar.
        """
        pass


class RoutingReward(RewardStrategy):
    """
    Estrategia de recompensa específica para Enrutamiento puro (Routing).
    El objetivo es minimizar la cantidad de SWAPs necesarios para ejecutar las puertas.

    Parámetros configurables
    ------------------------
    swap_penalty : float
        Penalización por cada SWAP insertado (valor negativo).
    gate_execution_reward : float
        Recompensa por cada puerta del circuito objetivo ejecutada.
    invalid_action_penalty : float
        Penalización por intentar una acción inválida.
    completion_bonus : float
        Bonificación al completar todas las puertas del circuito.
    truncation_penalty : float
        Penalización cuando el episodio se trunca por agotar ``max_steps``
        sin haber completado el circuito.  Un valor negativo grande
        incentiva al agente a resolver el circuito dentro del límite
        de pasos.  Configurar a ``0.0`` para deshabilitar la penalización.
    stagnation_penalty : float
        Penalización cuando el episodio termina por estancamiento. Se separa
        de ``truncation_penalty`` porque el estancamiento es un fallo terminal,
        no un corte externo por límite temporal.
    incomplete_gate_penalty : float
        Penalización adicional por cada puerta pendiente cuando el episodio
        termina sin completar el circuito.
    repeated_layout_penalty : float
        Penalización aplicada si la transición vuelve a un layout ya visto
        recientemente, incluyendo auto-bucles inmediatos.
    undo_swap_penalty : float
        Penalización aplicada si el SWAP actual deshace el SWAP anterior.
    unproductive_swap_penalty : float
        Penalizacion adicional para SWAPs que no ejecutan puertas ni reducen
        la distancia agregada de routing.
    routing_progress_reward : float
        Factor lineal de shaping sobre ``routing_progress_delta``. Valores
        positivos recompensan reducir la distancia agregada de routing y
        penalizan empeorarla.
    next_frontier_penalty_weight : float
        Peso aplicado a la distancia agregada pendiente cuando un SWAP
        desbloquea puertas. Distingue acciones que progresan localmente pero
        dejan la nueva frontera peor preparada.
    routing_depth_penalty_weight : float
        Peso aplicado al incremento de profundidad critica estimada.
    """

    def __init__(
        self,
        swap_penalty: float = -1.0,
        gate_execution_reward: float = 10.0,
        invalid_action_penalty: float = -5.0,
        completion_bonus: float = 50.0,
        truncation_penalty: float = -30.0,
        stagnation_penalty: float = -20.0,
        incomplete_gate_penalty: float = -1.0,
        repeated_layout_penalty: float = -1.0,
        undo_swap_penalty: float = -1.0,
        unproductive_swap_penalty: float = -0.25,
        routing_progress_reward: float = 0.5,
        next_frontier_penalty_weight: float = 0.25,
        routing_depth_penalty_weight: float = 0.1,
    ):
        self.swap_penalty = swap_penalty
        self.gate_execution_reward = gate_execution_reward
        self.invalid_action_penalty = invalid_action_penalty
        self.completion_bonus = completion_bonus
        self.truncation_penalty = truncation_penalty
        self.stagnation_penalty = stagnation_penalty
        self.incomplete_gate_penalty = incomplete_gate_penalty
        self.repeated_layout_penalty = repeated_layout_penalty
        self.undo_swap_penalty = undo_swap_penalty
        self.unproductive_swap_penalty = unproductive_swap_penalty
        self.routing_progress_reward = routing_progress_reward
        self.next_frontier_penalty_weight = next_frontier_penalty_weight
        self.routing_depth_penalty_weight = routing_depth_penalty_weight

    def to_dict(self) -> Dict[str, float]:
        return {
            "swap_penalty": self.swap_penalty,
            "gate_execution_reward": self.gate_execution_reward,
            "invalid_action_penalty": self.invalid_action_penalty,
            "completion_bonus": self.completion_bonus,
            "truncation_penalty": self.truncation_penalty,
            "stagnation_penalty": self.stagnation_penalty,
            "incomplete_gate_penalty": self.incomplete_gate_penalty,
            "repeated_layout_penalty": self.repeated_layout_penalty,
            "undo_swap_penalty": self.undo_swap_penalty,
            "unproductive_swap_penalty": self.unproductive_swap_penalty,
            "routing_progress_reward": self.routing_progress_reward,
            "next_frontier_penalty_weight": self.next_frontier_penalty_weight,
            "routing_depth_penalty_weight": self.routing_depth_penalty_weight,
        }

    def compute_reward(self, prev_state: Any, action: Any, current_state: Any, info: Dict[str, Any]) -> float:
        reward = 0.0
        
        # 1. Penalización por aplicar un SWAP (para minimizar la profundidad / CNOTs)
        if info.get('action_type') == 'swap':
            reward += self.swap_penalty
            
        # 2. Recompensa por ejecutar puertas del circuito objetivo
        gates_executed = info.get('gates_executed', 0)
        if gates_executed > 0:
            reward += (self.gate_execution_reward * gates_executed)
            
        # 3. Penalización si el agente intenta una acción no válida
        if info.get('is_valid_action') is False:
            reward += self.invalid_action_penalty

        if info.get('repeated_layout', False):
            reward += self.repeated_layout_penalty

        if info.get('undo_swap', False):
            reward += self.undo_swap_penalty

        if info.get('unproductive_swap', False):
            reward += self.unproductive_swap_penalty

        reward += self.routing_progress_reward * float(info.get('routing_progress_delta', 0.0))
        reward -= self.next_frontier_penalty_weight * float(info.get('next_frontier_routing_signal', 0.0))
        reward -= self.routing_depth_penalty_weight * float(info.get('routing_depth_delta', 0.0))
             
        # 4. Bonificación final si el circuito se completa
        if info.get('is_completed', False):
            reward += self.completion_bonus

        # 5. Penalización terminal si el episodio acaba sin completar
        incomplete_episode = not info.get('is_completed', False) and (
            info.get('is_truncated', False)
            or info.get('termination_reason') == 'stagnation'
        )
        if info.get('is_truncated', False) and not info.get('is_completed', False):
            reward += self.truncation_penalty
        elif info.get('termination_reason') == 'stagnation' and not info.get('is_completed', False):
            reward += self.stagnation_penalty

        if incomplete_episode:
            remaining_gates = max(float(info.get('remaining_gates', 0.0)), 0.0)
            reward += self.incomplete_gate_penalty * remaining_gates
            
        return reward


class SynthesisReward(RewardStrategy):
    """
    Estrategia de recompensa para síntesis Clifford basada en progreso residual.

    Parámetros configurables
    ------------------------
    invalid_action_penalty : float
        Penalización aplicada cuando la primitive seleccionada no es válida.
    step_penalty : float
        Coste base por paso válido para incentivar episodios cortos.
    primitive_cost_weight : float
        Peso lineal aplicado al coste intrínseco de la primitive ejecutada.
    residual_progress_reward : float
        Factor lineal aplicado a ``residual_distance_delta``.
    completion_bonus : float
        Bonificación por completar la síntesis (residual identidad).
    truncation_penalty : float
        Penalización cuando el episodio se trunca sin completar la síntesis.
    """

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

    def compute_reward(self, prev_state: Any, action: Any, current_state: Any, info: Dict[str, Any]) -> float:
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
