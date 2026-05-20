"""integration - orquestacion de scenarios y Campaigns.

Este paquete conecta `qiskit_interface`, `mo_module` y `rl_module`.
Su superficie publica cubre los contratos de Scenario y la capa minima de
routing-evaluation v1 para `Baseline`, `MO_Only`, `RL_Only` y `MO+RL`.

RL-based scenarios rebuild routed circuits when the episode completes.
La orquestacion de Campaigns, el handoff MO -> RL y la persistencia
publica viven aqui; la logica interna de MO y RL permanece en sus
modulos respectivos.
"""

from .contracts import (
    LayoutSelectionPolicy,
    RoutingEpisodeSummary,
    ScenarioRequest,
    ScenarioResult,
)

__all__ = [
    "LayoutSelectionPolicy",
    "RoutingEpisodeSummary",
    "ScenarioRequest",
    "ScenarioResult",
]
