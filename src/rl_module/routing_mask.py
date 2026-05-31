"""Versioned action-mask semantics for masked routing checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

FRONTIER_RESTRICTED_EDGES_V1 = "frontier_restricted_edges.v1"
FRONTIER_RESTRICTED_EDGES_V2 = "frontier_restricted_edges.v2"
FRONTIER_RESTRICTED_EDGES_V3 = "frontier_restricted_edges.v3"

LEGACY_MASK_SEMANTICS = FRONTIER_RESTRICTED_EDGES_V1
DEFAULT_NEW_MASK_SEMANTICS = FRONTIER_RESTRICTED_EDGES_V3
SUPPORTED_MASK_SEMANTICS = frozenset(
    {
        FRONTIER_RESTRICTED_EDGES_V1,
        FRONTIER_RESTRICTED_EDGES_V2,
        FRONTIER_RESTRICTED_EDGES_V3,
    }
)


@dataclass(frozen=True)
class RoutingMaskConfig:
    """Configuration for the v3 masked-routing dynamics."""

    cycle_window: int = 8
    stagnation_patience: int | None = None
    sabre_top_k: int | None = None
    sabre_lookahead_weight: float = 0.5
    distance_improvement_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if type(self.cycle_window) is not int or self.cycle_window <= 0:
            raise ValueError("cycle_window must be a positive integer")
        if self.stagnation_patience is not None and (
            type(self.stagnation_patience) is not int or self.stagnation_patience <= 0
        ):
            raise ValueError("stagnation_patience must be a positive integer or None")
        if self.sabre_top_k is not None and (
            type(self.sabre_top_k) is not int or self.sabre_top_k <= 0
        ):
            raise ValueError("sabre_top_k must be a positive integer or None")
        if (
            isinstance(self.sabre_lookahead_weight, bool)
            or not isinstance(self.sabre_lookahead_weight, (int, float))
            or self.sabre_lookahead_weight < 0
        ):
            raise ValueError("sabre_lookahead_weight must be a non-negative number")
        if (
            isinstance(self.distance_improvement_epsilon, bool)
            or not isinstance(self.distance_improvement_epsilon, (int, float))
            or self.distance_improvement_epsilon <= 0
        ):
            raise ValueError("distance_improvement_epsilon must be a positive number")

    def resolve(self, *, num_qubits: int) -> "RoutingMaskConfig":
        if type(num_qubits) is not int or num_qubits <= 0:
            raise ValueError("num_qubits must be a positive integer")
        if self.stagnation_patience is not None:
            return self
        return RoutingMaskConfig(
            cycle_window=self.cycle_window,
            stagnation_patience=max(8, 2 * num_qubits),
            sabre_top_k=self.sabre_top_k,
            sabre_lookahead_weight=float(self.sabre_lookahead_weight),
            distance_improvement_epsilon=float(self.distance_improvement_epsilon),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: "RoutingMaskConfig | Mapping[str, Any] | None") -> "RoutingMaskConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("routing_mask_config must be a RoutingMaskConfig, mapping, or None")
        expected_fields = {
            "cycle_window",
            "stagnation_patience",
            "sabre_top_k",
            "sabre_lookahead_weight",
            "distance_improvement_epsilon",
        }
        unknown_fields = set(value) - expected_fields
        if unknown_fields:
            raise ValueError(
                "routing_mask_config contains unsupported fields: "
                f"{sorted(unknown_fields)}"
            )
        return cls(**dict(value))


def resolve_routing_mask_config(
    value: RoutingMaskConfig | Mapping[str, Any] | None,
    *,
    num_qubits: int,
) -> RoutingMaskConfig:
    return RoutingMaskConfig.from_value(value).resolve(num_qubits=num_qubits)


def require_resolved_routing_mask_config(
    value: RoutingMaskConfig | Mapping[str, Any] | None,
) -> RoutingMaskConfig:
    if value is None:
        raise ValueError("routing_mask_config is required for frontier_restricted_edges.v3")
    config = RoutingMaskConfig.from_value(value)
    if config.stagnation_patience is None:
        raise ValueError("routing_mask_config.stagnation_patience must be resolved for persistence")
    return config


def normalize_mask_semantics(
    mask_semantics: str | None,
    *,
    default: str = LEGACY_MASK_SEMANTICS,
) -> str:
    resolved_semantics = default if mask_semantics is None else mask_semantics
    if resolved_semantics not in SUPPORTED_MASK_SEMANTICS:
        raise ValueError(
            "Unsupported routing mask semantics "
            f"{resolved_semantics!r}; expected one of {sorted(SUPPORTED_MASK_SEMANTICS)}"
        )
    return resolved_semantics
