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
_SUPPORTED_MASKED_ROUTING_SCHEMA_VERSION = "rl_run_metadata.masked_routing.v1"
_SUPPORTED_MASK_SEMANTICS = "frontier_restricted_edges.v1"
_SUPPORTED_LEGACY_ALGORITHMS = frozenset({"PPO", "DQN"})
_SUPPORTED_MASKED_ALGORITHM = "MaskablePPO"
_SUPPORTED_FRONTIER_MODES = frozenset({"sequential", "dag"})
_LEGACY_SB3_DATA_ENTRY = "data"
_LEGACY_DQN_POLICY_MODULE = "stable_baselines3.dqn.policies"


@dataclass(frozen=True)
class RoutingModelContract:
    algorithm: str
    frontier_mode: str
    lookahead_window: int
    max_steps: int
    masked: bool
    mask_semantics: str | None
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


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise _contract_error(f"missing or invalid '{field_name}'")
    return value


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise _contract_error(f"missing or invalid '{field_name}'")
    return value


def _require_int(value: object, *, field_name: str) -> int:
    if type(value) is not int:
        raise _contract_error(f"missing or invalid '{field_name}'")
    return value


def _require_supported_frontier_mode(value: object, *, field_name: str) -> str:
    frontier_mode = _require_str(value, field_name=field_name)
    if frontier_mode not in _SUPPORTED_FRONTIER_MODES:
        raise _contract_error(
            f"unsupported '{field_name}'; expected one of {sorted(_SUPPORTED_FRONTIER_MODES)}"
        )
    return frontier_mode


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
            masked=False,
            mask_semantics=None,
            metadata_source="defaults",
        )

    metadata = _require_mapping(metadata, field_name="metadata")

    schema_version = metadata.get("schema_version")
    if schema_version not in {_SUPPORTED_SCHEMA_VERSION, _SUPPORTED_MASKED_ROUTING_SCHEMA_VERSION}:
        raise _contract_error(
            "unsupported schema_version "
            f"'{schema_version}'; expected one of '{_SUPPORTED_SCHEMA_VERSION}' or "
            f"'{_SUPPORTED_MASKED_ROUTING_SCHEMA_VERSION}'"
        )

    if metadata.get("mode") != "routing":
        raise ValueError("RL evaluation only supports routing metadata")

    environment = _require_mapping(metadata.get("environment"), field_name="environment")
    algorithm = _require_str(_require_field(metadata, "algorithm"), field_name="algorithm")
    if schema_version == _SUPPORTED_SCHEMA_VERSION and algorithm == _SUPPORTED_MASKED_ALGORITHM:
        raise _contract_error(
            f"schema '{_SUPPORTED_SCHEMA_VERSION}' cannot declare algorithm '{_SUPPORTED_MASKED_ALGORITHM}'; "
            f"use '{_SUPPORTED_MASKED_ROUTING_SCHEMA_VERSION}'"
        )
    if schema_version == _SUPPORTED_SCHEMA_VERSION and algorithm not in _SUPPORTED_LEGACY_ALGORITHMS:
        raise _contract_error(
            "unsupported 'algorithm'; expected one of "
            f"{sorted(_SUPPORTED_LEGACY_ALGORITHMS)} for schema '{_SUPPORTED_SCHEMA_VERSION}'"
        )
    masked = False
    mask_semantics = None
    if schema_version == _SUPPORTED_MASKED_ROUTING_SCHEMA_VERSION:
        routing_policy = _require_mapping(metadata.get("routing_policy"), field_name="routing_policy")
        masked = _require_bool(
            _require_field(routing_policy, "masked"),
            field_name="routing_policy.masked",
        )
        mask_semantics = _require_str(
            _require_field(routing_policy, "mask_semantics"),
            field_name="routing_policy.mask_semantics",
        )
        if masked is not True:
            raise _contract_error("masked routing schema requires 'routing_policy.masked' to be True")
        if algorithm != _SUPPORTED_MASKED_ALGORITHM:
            raise _contract_error(
                f"masked routing schema requires algorithm '{_SUPPORTED_MASKED_ALGORITHM}'"
            )
        if mask_semantics != _SUPPORTED_MASK_SEMANTICS:
            raise _contract_error(
                "unsupported 'routing_policy.mask_semantics'; "
                f"expected '{_SUPPORTED_MASK_SEMANTICS}'"
            )

    return RoutingModelContract(
        algorithm=algorithm,
        frontier_mode=_require_supported_frontier_mode(
            _require_field(environment, "frontier_mode"),
            field_name="environment.frontier_mode",
        ),
        lookahead_window=_require_int(
            _require_field(environment, "lookahead_window"),
            field_name="environment.lookahead_window",
        ),
        max_steps=_require_int(
            _require_field(environment, "max_steps"),
            field_name="environment.max_steps",
        ),
        masked=masked,
        mask_semantics=mask_semantics,
        metadata_source="sidecar",
    )
