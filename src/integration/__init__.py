"""Module 4: integration and experimentation.

This package owns the MO -> RL handoff and the routing-evaluation v1
scope for `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL` scenarios.

RL-based scenarios rebuild routed circuits and run Qiskit post-routing
metrics when the routing episode completes.
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
