from src.rl_module.model_metadata import build_run_metadata, save_run_metadata


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


def test_resolve_routing_model_contract_falls_back_to_legacy_defaults(tmp_path):
    from src.integration.rl_model_contract import resolve_routing_model_contract

    model_path = tmp_path / "legacy_model.zip"
    model_path.write_text("stub", encoding="utf-8")

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
