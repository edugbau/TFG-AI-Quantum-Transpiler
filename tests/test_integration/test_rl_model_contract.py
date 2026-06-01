import json
from zipfile import ZipFile

import pytest

from src.rl_module.model_metadata import build_run_metadata, save_run_metadata
from src.rl_module.routing_mask import RoutingMaskConfig


def _write_legacy_sb3_model(tmp_path, model_name: str, *, policy_module: str) -> None:
    with ZipFile(tmp_path / model_name, "w") as archive:
        archive.writestr(
            "data",
            json.dumps(
                {
                    "policy_class": {
                        "__module__": policy_module,
                    }
                }
            ),
        )


def test_resolve_routing_model_contract_uses_sidecar_when_present(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    save_run_metadata(
        tmp_path,
        build_run_metadata(
            mode="routing",
            algorithm="DQN",
            seed=31,
            frontier_mode="dag",
            lookahead_window=8,
            max_steps=144,
            basis_gates=None,
        ),
    )

    contract = resolve_routing_model_contract(model_path)

    assert contract.algorithm == "DQN"
    assert contract.frontier_mode == "dag"
    assert contract.lookahead_window == 8
    assert contract.max_steps == 144
    assert contract.metadata_source == "sidecar"


def test_resolve_routing_model_contract_reads_explicit_masked_routing_schema(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    save_run_metadata(
        tmp_path,
        build_run_metadata(
            mode="routing",
            algorithm="MaskablePPO",
            seed=31,
            frontier_mode="dag",
            lookahead_window=8,
            max_steps=144,
            basis_gates=None,
            mask_semantics="frontier_restricted_edges.v1",
        ),
    )

    contract = resolve_routing_model_contract(model_path)

    assert contract.algorithm == "MaskablePPO"
    assert contract.frontier_mode == "dag"
    assert contract.lookahead_window == 8
    assert contract.max_steps == 144
    assert contract.masked is True
    assert contract.mask_semantics == "frontier_restricted_edges.v1"
    assert contract.metadata_source == "sidecar"


def test_resolve_routing_model_contract_accepts_anti_undo_mask_semantics(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    save_run_metadata(
        tmp_path,
        build_run_metadata(
            mode="routing",
            algorithm="MaskablePPO",
            seed=31,
            frontier_mode="dag",
            lookahead_window=8,
            max_steps=144,
            basis_gates=None,
            mask_semantics="frontier_restricted_edges.v2",
        ),
    )

    contract = resolve_routing_model_contract(model_path)

    assert contract.mask_semantics == "frontier_restricted_edges.v2"


def test_resolve_routing_model_contract_accepts_v3_with_explicit_config(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    save_run_metadata(
        tmp_path,
        build_run_metadata(
            mode="routing",
            algorithm="MaskablePPO",
            seed=31,
            frontier_mode="dag",
            lookahead_window=8,
            max_steps=144,
            basis_gates=None,
            mask_semantics="frontier_restricted_edges.v3",
            routing_mask_config=RoutingMaskConfig(
                cycle_window=6,
                stagnation_patience=16,
                sabre_top_k=3,
            ),
        ),
    )

    contract = resolve_routing_model_contract(model_path)

    assert contract.mask_semantics == "frontier_restricted_edges.v3"
    assert contract.routing_mask_config == RoutingMaskConfig(
        cycle_window=6,
        stagnation_patience=16,
        sabre_top_k=3,
    )


def test_resolve_routing_model_contract_accepts_v4_with_decay_config(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    save_run_metadata(
        tmp_path,
        build_run_metadata(
            mode="routing",
            algorithm="MaskablePPO",
            seed=31,
            frontier_mode="dag",
            lookahead_window=8,
            max_steps=144,
            basis_gates=None,
            mask_semantics="frontier_restricted_edges.v4",
            routing_mask_config=RoutingMaskConfig(
                stagnation_patience=16,
                sabre_decay_increment=0.002,
                sabre_decay_reset_interval=7,
            ),
        ),
    )

    contract = resolve_routing_model_contract(model_path)

    assert contract.mask_semantics == "frontier_restricted_edges.v4"
    assert contract.routing_mask_config.sabre_decay_increment == 0.002
    assert contract.routing_mask_config.sabre_decay_reset_interval == 7


def test_resolve_routing_model_contract_accepts_v5_with_preparatory_edges(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    save_run_metadata(
        tmp_path,
        build_run_metadata(
            mode="routing",
            algorithm="MaskablePPO",
            seed=31,
            frontier_mode="dag",
            lookahead_window=8,
            max_steps=144,
            basis_gates=None,
            mask_semantics="frontier_restricted_edges.v5",
            routing_mask_config=RoutingMaskConfig(
                stagnation_patience=16,
                sabre_top_k=3,
            ),
        ),
    )

    contract = resolve_routing_model_contract(model_path)

    assert contract.mask_semantics == "frontier_restricted_edges.v5"
    assert contract.routing_mask_config.sabre_top_k == 3


def test_resolve_routing_model_contract_rejects_v3_without_mask_config(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="MaskablePPO",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
        mask_semantics="frontier_restricted_edges.v3",
        routing_mask_config=RoutingMaskConfig(stagnation_patience=16),
    )
    del metadata["routing_policy"]["mask_config"]
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="mask_config"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_v3_under_historical_masked_schema(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="MaskablePPO",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
        mask_semantics="frontier_restricted_edges.v1",
    )
    metadata["routing_policy"]["mask_semantics"] = "frontier_restricted_edges.v3"
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="mask_semantics"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_falls_back_to_legacy_defaults(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "legacy_model.zip"
    model_path.write_text("stub", encoding="utf-8")

    contract = resolve_routing_model_contract(model_path)

    assert contract.algorithm == "PPO"
    assert contract.frontier_mode == "sequential"
    assert contract.lookahead_window == 4
    assert contract.max_steps == 256
    assert contract.masked is False
    assert contract.mask_semantics is None
    assert contract.metadata_source == "defaults"


def test_resolve_routing_model_contract_recovers_dqn_from_legacy_checkpoint(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "legacy_dqn_model.zip"
    _write_legacy_sb3_model(
        tmp_path,
        model_path.name,
        policy_module="stable_baselines3.dqn.policies",
    )

    contract = resolve_routing_model_contract(model_path)

    assert contract.algorithm == "DQN"
    assert contract.frontier_mode == "sequential"
    assert contract.lookahead_window == 4
    assert contract.max_steps == 256
    assert contract.metadata_source == "defaults"


def test_resolve_routing_model_contract_recovers_ppo_from_legacy_checkpoint(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "legacy_ppo_model.zip"
    _write_legacy_sb3_model(
        tmp_path,
        model_path.name,
        policy_module="stable_baselines3.common.policies",
    )

    contract = resolve_routing_model_contract(model_path)

    assert contract.algorithm == "PPO"
    assert contract.frontier_mode == "sequential"
    assert contract.lookahead_window == 4
    assert contract.max_steps == 256
    assert contract.metadata_source == "defaults"


def test_resolve_routing_model_contract_rejects_non_routing_metadata(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    save_run_metadata(
        tmp_path,
        build_run_metadata(
            mode="synthesis",
            algorithm="PPO",
            seed=31,
            frontier_mode="sequential",
            lookahead_window=4,
            max_steps=144,
            basis_gates=None,
        ),
    )

    try:
        resolve_routing_model_contract(model_path)
    except ValueError as exc:
        assert "routing" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-routing metadata")


def test_resolve_routing_model_contract_rejects_malformed_sidecar_json(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    (tmp_path / "run_metadata.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="routing model metadata"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_unknown_schema_version(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="DQN",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
    )
    metadata["schema_version"] = "rl_run_metadata.v2"
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="schema"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_missing_required_keys(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="DQN",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
    )
    del metadata["environment"]["max_steps"]
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="max_steps"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_non_boolean_masked_flag(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="MaskablePPO",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
        mask_semantics="frontier_restricted_edges.v1",
    )
    metadata["routing_policy"]["masked"] = "true"
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="routing_policy.*masked"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_non_string_mask_semantics(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="MaskablePPO",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
        mask_semantics="frontier_restricted_edges.v1",
    )
    metadata["routing_policy"]["mask_semantics"] = ["frontier_restricted_edges.v1"]
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="routing_policy.*mask_semantics"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_unsupported_mask_semantics(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="MaskablePPO",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
        mask_semantics="frontier_restricted_edges.v1",
    )
    metadata["routing_policy"]["mask_semantics"] = "frontier_restricted_edges.v4"
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="mask_semantics"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_masked_schema_with_masked_false(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="MaskablePPO",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
        mask_semantics="frontier_restricted_edges.v1",
    )
    metadata["routing_policy"]["masked"] = False
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="routing_policy.*masked"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_masked_schema_with_non_maskable_algorithm(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_masked_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="MaskablePPO",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
        mask_semantics="frontier_restricted_edges.v1",
    )
    metadata["algorithm"] = "PPO"
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="MaskablePPO"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_legacy_v1_maskableppo_sidecar(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="DQN",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
    )
    metadata["algorithm"] = "MaskablePPO"
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="MaskablePPO"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_non_string_algorithm(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="DQN",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
    )
    metadata["algorithm"] = 7
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="algorithm"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_non_string_frontier_mode(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="DQN",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
    )
    metadata["environment"]["frontier_mode"] = 5
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="frontier_mode"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_unsupported_string_frontier_mode(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="DQN",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
    )
    metadata["environment"]["frontier_mode"] = "unsupported"
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="frontier_mode"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_string_lookahead_window(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="DQN",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
    )
    metadata["environment"]["lookahead_window"] = "4"
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="lookahead_window"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_boolean_max_steps(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="DQN",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
    )
    metadata["environment"]["max_steps"] = True
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="max_steps"):
        resolve_routing_model_contract(model_path)


def test_resolve_routing_model_contract_rejects_unsupported_algorithm_value(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "best_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    metadata = build_run_metadata(
        mode="routing",
        algorithm="DQN",
        seed=31,
        frontier_mode="dag",
        lookahead_window=8,
        max_steps=144,
        basis_gates=None,
    )
    metadata["algorithm"] = "A2C"
    save_run_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="algorithm"):
        resolve_routing_model_contract(model_path)
