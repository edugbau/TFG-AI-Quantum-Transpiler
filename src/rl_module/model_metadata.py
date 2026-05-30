import json
from pathlib import Path
from typing import Any, Optional

from .routing_mask import DEFAULT_NEW_MASK_SEMANTICS


_METADATA_FILENAME = "run_metadata.json"
_SCHEMA_VERSION = "rl_run_metadata.v1"
_MASKED_ROUTING_SCHEMA_VERSION = "rl_run_metadata.masked_routing.v1"


def metadata_path_for_model(model_path: Path | str) -> Path:
    return Path(model_path).parent / _METADATA_FILENAME


def build_run_metadata(
    *,
    mode: str,
    algorithm: str,
    seed: int,
    frontier_mode: str,
    lookahead_window: int,
    max_steps: int,
    basis_gates: Optional[list[str]],
    mask_semantics: Optional[str] = None,
    training_hyperparams: Optional[dict[str, Any]] = None,
    evaluation_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "mode": mode,
        "algorithm": algorithm,
        "seed": int(seed),
        "environment": {
            "frontier_mode": frontier_mode,
            "lookahead_window": int(lookahead_window),
            "max_steps": int(max_steps),
            "basis_gates": list(basis_gates) if basis_gates is not None else None,
        },
    }
    if training_hyperparams is not None:
        metadata["training"] = {"hyperparams": dict(training_hyperparams)}
    if evaluation_config is not None:
        metadata["evaluation"] = dict(evaluation_config)

    if mode == "routing" and algorithm == "MaskablePPO":
        metadata["schema_version"] = _MASKED_ROUTING_SCHEMA_VERSION
        metadata["routing_policy"] = {
            "masked": True,
            "mask_semantics": mask_semantics or DEFAULT_NEW_MASK_SEMANTICS,
        }

    return metadata


def save_run_metadata(run_dir: Path | str, metadata: dict[str, Any]) -> Path:
    metadata_path = Path(run_dir) / _METADATA_FILENAME
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def load_run_metadata_for_model(model_path: Path | str) -> Optional[dict[str, Any]]:
    metadata_path = metadata_path_for_model(model_path)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))
