"""
Módulo de Aprendizaje por Refuerzo (RL Module).

Este módulo provee el entorno Gymnasium y los agentes necesarios para la transpilación
cuántica híbrida, incluyendo enrutamiento (routing) y síntesis completa (synthesis).
"""

from .environment import QuantumTranspilationEnv
from .env_strategies import RoutingStrategy, SynthesisStrategy
from .frontier import DagFrontier, LookaheadEntry, SequentialFrontier
from .rewards import RoutingReward, SynthesisReward
from .agent import QuantumRLAgent
from .training import setup_training_pipeline, set_global_seeds

__all__ = [
    "QuantumTranspilationEnv",
    "RoutingStrategy",
    "SynthesisStrategy",
    "LookaheadEntry",
    "DagFrontier",
    "SequentialFrontier",
    "RoutingReward",
    "SynthesisReward",
    "QuantumRLAgent",
    "setup_training_pipeline",
    "set_global_seeds"
]
