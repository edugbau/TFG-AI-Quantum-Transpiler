from dataclasses import dataclass
from pathlib import Path

from src.rl_module.model_metadata import load_run_metadata_for_model


_DEFAULT_ALGORITHM = "PPO"
_DEFAULT_FRONTIER_MODE = "sequential"
_DEFAULT_LOOKAHEAD_WINDOW = 4
_DEFAULT_MAX_STEPS = 256


@dataclass(frozen=True)
class RoutingModelContract:
    algorithm: str
    frontier_mode: str
    lookahead_window: int
    max_steps: int
    metadata_source: str


def resolve_routing_model_contract(model_path: Path | str) -> RoutingModelContract:
    metadata = load_run_metadata_for_model(model_path)
    if metadata is None:
        return RoutingModelContract(
            algorithm=_DEFAULT_ALGORITHM,
            frontier_mode=_DEFAULT_FRONTIER_MODE,
            lookahead_window=_DEFAULT_LOOKAHEAD_WINDOW,
            max_steps=_DEFAULT_MAX_STEPS,
            metadata_source="defaults",
        )

    if metadata.get("mode") != "routing":
        raise ValueError("RL evaluation only supports routing metadata")

    environment = metadata.get("environment") or {}
    return RoutingModelContract(
        algorithm=str(metadata["algorithm"]),
        frontier_mode=str(environment["frontier_mode"]),
        lookahead_window=int(environment["lookahead_window"]),
        max_steps=int(environment["max_steps"]),
        metadata_source="sidecar",
    )
