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
                  - ``is_completed``: ``bool``
                  - ``is_truncated``: ``bool`` — ``True`` si se agotó ``max_steps``
                  
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
    """

    def __init__( #TODO: REVISAR VALORES DE PENALIZACIÓN Y REWARD CON TUTORES
        self,
        swap_penalty: float = -1.0,
        gate_execution_reward: float = 10.0,
        invalid_action_penalty: float = -5.0,
        completion_bonus: float = 50.0,
        truncation_penalty: float = -20.0,
    ):
        self.swap_penalty = swap_penalty
        self.gate_execution_reward = gate_execution_reward
        self.invalid_action_penalty = invalid_action_penalty
        self.completion_bonus = completion_bonus
        self.truncation_penalty = truncation_penalty

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
            
        # 4. Bonificación final si el circuito se completa
        if info.get('is_completed', False):
            reward += self.completion_bonus

        # 5. Penalización si el episodio fue truncado sin completarse
        if info.get('is_truncated', False) and not info.get('is_completed', False):
            reward += self.truncation_penalty
            
        return reward


class SynthesisReward(RewardStrategy):
    """
    Estrategia de recompensa para Síntesis completa.
    El objetivo es reconstruir el circuito objetivo minimizando el coste total de las puertas.

    Parámetros configurables
    ------------------------
    valid_gate_reward : float
        Recompensa cuando la puerta aplicada contribuye a la síntesis.
    incorrect_gate_penalty : float
        Penalización cuando la puerta aplicada no contribuye.
    completion_bonus : float
        Bonificación por completar la síntesis exacta del circuito.
    truncation_penalty : float
        Penalización cuando el episodio se trunca por agotar ``max_steps``
        sin haber completado la síntesis.  Configurar a ``0.0`` para
        deshabilitar.
    """

    def __init__(
        self,
        valid_gate_reward: float = 2.0,
        incorrect_gate_penalty: float = -1.0,
        completion_bonus: float = 100.0,
        truncation_penalty: float = -30.0,
    ):
        self.valid_gate_reward = valid_gate_reward
        self.incorrect_gate_penalty = incorrect_gate_penalty
        self.completion_bonus = completion_bonus
        self.truncation_penalty = truncation_penalty

    def compute_reward(self, prev_state: Any, action: Any, current_state: Any, info: Dict[str, Any]) -> float:
        reward = 0.0
        
        # 1. Recompensa si la puerta aplicada contribuye positivamente a la síntesis del circuito target
        if info.get('gate_matched_target', False):
            reward += self.valid_gate_reward
        else:
            reward += self.incorrect_gate_penalty
            
        # 2. Bonificación por completar la síntesis exacta
        if info.get('is_completed', False):
            reward += self.completion_bonus

        # 3. Penalización si el episodio fue truncado sin completarse
        if info.get('is_truncated', False) and not info.get('is_completed', False):
            reward += self.truncation_penalty
            
        return reward
