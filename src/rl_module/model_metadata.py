import json
from pathlib import Path
from typing import Any, Optional


_METADATA_FILENAME = "run_metadata.json"
_SCHEMA_VERSION = "rl_run_metadata.v1"


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
) -> dict[str, Any]:
    return {
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
