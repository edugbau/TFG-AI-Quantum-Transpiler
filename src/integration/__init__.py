"""Module 4: integration and experimentation.

This package owns the MO -> RL handoff and the routing-evaluation v1
scope for `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL` scenarios.

RL outputs in this scope are episode summaries, not final circuits.
The pipeline implementation is still outside this change.
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
