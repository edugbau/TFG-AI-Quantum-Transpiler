import json
import importlib
import sys
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


def test_build_run_metadata_uses_explicit_masked_routing_schema_when_requested():
    metadata = build_run_metadata(
        mode="routing",
        algorithm="MaskablePPO",
        seed=17,
        frontier_mode="dag",
        lookahead_window=7,
        max_steps=256,
        basis_gates=None,
        mask_semantics="frontier_restricted_edges.v1",
    )

    assert metadata["schema_version"] == "rl_run_metadata.masked_routing.v1"
    assert metadata["algorithm"] == "MaskablePPO"
    assert metadata["routing_policy"] == {
        "masked": True,
        "mask_semantics": "frontier_restricted_edges.v1",
    }


def test_build_run_metadata_defaults_new_masked_routing_checkpoints_to_anti_undo_semantics():
    metadata = build_run_metadata(
        mode="routing",
        algorithm="MaskablePPO",
        seed=17,
        frontier_mode="dag",
        lookahead_window=7,
        max_steps=256,
        basis_gates=None,
    )

    assert metadata["routing_policy"] == {
        "masked": True,
        "mask_semantics": "frontier_restricted_edges.v2",
    }


def test_build_run_metadata_records_training_and_evaluation_config():
    metadata = build_run_metadata(
        mode="routing",
        algorithm="PPO",
        seed=17,
        frontier_mode="dag",
        lookahead_window=7,
        max_steps=256,
        basis_gates=None,
        training_hyperparams={
            "learning_rate": 1e-4,
            "clip_range": 0.1,
            "target_kl": 0.03,
        },
        evaluation_config={
            "eval_freq": 5000,
            "n_eval_episodes": 5,
            "deterministic": True,
        },
    )

    assert metadata["training"]["hyperparams"] == {
        "learning_rate": 1e-4,
        "clip_range": 0.1,
        "target_kl": 0.03,
    }
    assert metadata["evaluation"] == {
        "eval_freq": 5000,
        "n_eval_episodes": 5,
        "deterministic": True,
    }


def test_importing_model_metadata_does_not_import_stable_baselines3(monkeypatch):
    for module_name in list(sys.modules):
        if module_name == "src.rl_module" or module_name.startswith("src.rl_module."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
        if module_name == "stable_baselines3" or module_name.startswith("stable_baselines3."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    importlib.import_module("src.rl_module.model_metadata")

    assert "stable_baselines3" not in sys.modules
