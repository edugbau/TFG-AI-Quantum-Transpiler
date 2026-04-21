from dataclasses import dataclass
import json
from json import JSONDecodeError
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from src.rl_module.model_metadata import load_run_metadata_for_model


_DEFAULT_ALGORITHM = "PPO"
_DEFAULT_FRONTIER_MODE = "sequential"
_DEFAULT_LOOKAHEAD_WINDOW = 4
_DEFAULT_MAX_STEPS = 256
_SUPPORTED_SCHEMA_VERSION = "rl_run_metadata.v1"
_LEGACY_SB3_DATA_ENTRY = "data"
_LEGACY_DQN_POLICY_MODULE = "stable_baselines3.dqn.policies"


@dataclass(frozen=True)
class RoutingModelContract:
    algorithm: str
    frontier_mode: str
    lookahead_window: int
    max_steps: int
    metadata_source: str


def _contract_error(message: str) -> ValueError:
    return ValueError(f"Invalid routing model metadata contract: {message}")


def _require_mapping(value: object, *, field_name: str) -> dict:
    if not isinstance(value, dict):
        raise _contract_error(f"missing or invalid '{field_name}'")
    return value


def _require_field(mapping: dict, field_name: str):
    if field_name not in mapping:
        raise _contract_error(f"missing required field '{field_name}'")
    return mapping[field_name]


def _infer_legacy_algorithm_from_checkpoint(model_path: Path | str) -> str:
    try:
        with ZipFile(model_path) as archive:
            payload = json.loads(archive.read(_LEGACY_SB3_DATA_ENTRY).decode("utf-8"))
    except (FileNotFoundError, BadZipFile, KeyError, JSONDecodeError, OSError, UnicodeDecodeError):
        return _DEFAULT_ALGORITHM

    policy_class = payload.get("policy_class")
    if not isinstance(policy_class, dict):
        return _DEFAULT_ALGORITHM

    policy_module = policy_class.get("__module__")
    if policy_module == _LEGACY_DQN_POLICY_MODULE:
        return "DQN"
    return _DEFAULT_ALGORITHM


def resolve_routing_model_contract(model_path: Path | str) -> RoutingModelContract:
    try:
        metadata = load_run_metadata_for_model(model_path)
    except JSONDecodeError as exc:
        raise _contract_error("sidecar JSON is malformed") from exc

    if metadata is None:
        return RoutingModelContract(
            algorithm=_infer_legacy_algorithm_from_checkpoint(model_path),
            frontier_mode=_DEFAULT_FRONTIER_MODE,
            lookahead_window=_DEFAULT_LOOKAHEAD_WINDOW,
            max_steps=_DEFAULT_MAX_STEPS,
            metadata_source="defaults",
        )

    metadata = _require_mapping(metadata, field_name="metadata")

    schema_version = metadata.get("schema_version")
    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise _contract_error(
            f"unsupported schema_version '{schema_version}'; expected '{_SUPPORTED_SCHEMA_VERSION}'"
        )

    if metadata.get("mode") != "routing":
        raise ValueError("RL evaluation only supports routing metadata")

    environment = _require_mapping(metadata.get("environment"), field_name="environment")
    return RoutingModelContract(
        algorithm=str(_require_field(metadata, "algorithm")),
        frontier_mode=str(_require_field(environment, "frontier_mode")),
        lookahead_window=int(_require_field(environment, "lookahead_window")),
        max_steps=int(_require_field(environment, "max_steps")),
        metadata_source="sidecar",
    )
