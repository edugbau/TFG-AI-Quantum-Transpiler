import json
from pathlib import Path

from src.rl_module.model_metadata import (
    build_run_metadata,
    load_run_metadata_for_model,
    metadata_path_for_model,
    save_run_metadata,
)


def test_metadata_path_for_model_uses_model_directory_sidecar(tmp_path):
    model_path = tmp_path / "best_model.zip"
    expected = tmp_path / "run_metadata.json"
    assert metadata_path_for_model(model_path) == expected


def test_save_and_load_run_metadata_round_trip(tmp_path):
    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="PPO",
        seed=17,
        frontier_mode="dag",
        lookahead_window=7,
        max_steps=256,
        basis_gates=None,
    )

    save_run_metadata(model_path.parent, metadata)
    loaded = load_run_metadata_for_model(model_path)

    assert loaded == metadata
    assert loaded["schema_version"] == "rl_run_metadata.v1"
    assert loaded["environment"] == {
        "frontier_mode": "dag",
        "lookahead_window": 7,
        "max_steps": 256,
        "basis_gates": None,
    }


def test_load_run_metadata_for_model_returns_none_when_sidecar_is_missing(tmp_path):
    model_path = tmp_path / "final_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    assert load_run_metadata_for_model(model_path) is None
