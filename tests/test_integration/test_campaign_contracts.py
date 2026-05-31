from src.integration.campaign_contracts import (
    Campaign,
    CampaignCaseResult,
    CampaignConfig,
    CampaignCircuitSpec,
    CampaignSummary,
)
from src.integration.contracts import LayoutSelectionPolicy
from src.integration.mo_effort import MIN_CUSTOM_MO_POPULATION_SIZE
from src.integration.synthetic_topology import SyntheticTopologySpec


def test_campaign_builds_stable_cases_from_selected_circuits_and_backends() -> None:
    config = CampaignConfig(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=5),
        ],
        backend_names=["fake_torino", "fake_brisbane"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
    )
    campaign = Campaign.from_config(campaign_id="campaign-001", config=config)

    cases = campaign.build_cases()

    assert [case.case_id for case in cases] == [
        "ghz_3__fake_torino",
        "ghz_3__fake_brisbane",
        "qft_5__fake_torino",
        "qft_5__fake_brisbane",
    ]
    assert [(case.circuit_family, case.num_qubits, case.backend_name) for case in cases] == [
        ("ghz", 3, "fake_torino"),
        ("ghz", 3, "fake_brisbane"),
        ("qft", 5, "fake_torino"),
        ("qft", 5, "fake_brisbane"),
    ]


def test_campaign_rejects_empty_circuit_selection() -> None:
    try:
        CampaignConfig(
            circuit_specs=[],
            backend_names=["fake_torino"],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="sequential",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
        )
    except ValueError as exc:
        assert "circuit_specs" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty campaign circuit selection")


def test_campaign_rejects_empty_backend_selection() -> None:
    try:
        CampaignConfig(
            circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
            backend_names=[],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="sequential",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
        )
    except ValueError as exc:
        assert "backend_names" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty campaign backend selection")


def test_campaign_rejects_duplicate_circuit_specs_that_would_collide_case_ids() -> None:
    try:
        CampaignConfig(
            circuit_specs=[
                CampaignCircuitSpec(family="ghz", num_qubits=3),
                CampaignCircuitSpec(family="ghz", num_qubits=3),
            ],
            backend_names=["fake_torino"],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="sequential",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
        )
    except ValueError as exc:
        assert "circuit_specs" in str(exc)
    else:
        raise AssertionError("Expected ValueError for duplicate campaign circuit specs")


def test_campaign_rejects_duplicate_backend_names_that_would_collide_case_ids() -> None:
    try:
        CampaignConfig(
            circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
            backend_names=["fake_torino", "fake_torino"],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="sequential",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
        )
    except ValueError as exc:
        assert "backend_names" in str(exc)
    else:
        raise AssertionError("Expected ValueError for duplicate campaign backend names")


def test_campaign_circuit_spec_rejects_blank_family_name() -> None:
    try:
        CampaignCircuitSpec(family="   ", num_qubits=3)
    except ValueError as exc:
        assert "family" in str(exc)
    else:
        raise AssertionError("Expected ValueError for blank circuit family")


def test_campaign_circuit_spec_rejects_non_positive_num_qubits() -> None:
    try:
        CampaignCircuitSpec(family="ghz", num_qubits=0)
    except ValueError as exc:
        assert "num_qubits" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive campaign num_qubits")


def test_campaign_rejects_blank_backend_names() -> None:
    try:
        CampaignConfig(
            circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
            backend_names=["   "],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="sequential",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
        )
    except ValueError as exc:
        assert "backend_names" in str(exc)
    else:
        raise AssertionError("Expected ValueError for blank backend name")


def test_campaign_case_result_distinguishes_completed_failed_incomplete_and_cancelled() -> None:
    statuses = [
        CampaignCaseResult(case_id="case-1", status="completed"),
        CampaignCaseResult(case_id="case-2", status="failed"),
        CampaignCaseResult(case_id="case-3", status="incomplete"),
        CampaignCaseResult(case_id="case-4", status="cancelled"),
    ]

    assert [result.status for result in statuses] == [
        "completed",
        "failed",
        "incomplete",
        "cancelled",
    ]


def test_campaign_case_result_rejects_unknown_status() -> None:
    try:
        CampaignCaseResult(case_id="case-1", status="unexpected")
    except ValueError as exc:
        assert "status" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown campaign case result status")


def test_campaign_config_defaults_to_default_mode_and_accepts_advanced_mode() -> None:
    default_config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
    )
    advanced_config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mode="advanced",
    )

    assert default_config.mode == "default"
    assert advanced_config.mode == "advanced"
    assert default_config.topology_source == "backend"
    assert default_config.synthetic_topology is None
    assert default_config.rl_learning_rate == 1e-4
    assert default_config.rl_clip_range == 0.1
    assert default_config.rl_target_kl == 0.03
    assert default_config.rl_n_eval_episodes == 1


def test_campaign_config_accepts_synthetic_topology_in_advanced_mode() -> None:
    synthetic_topology = SyntheticTopologySpec(shape="grid", rows=2, cols=2)
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=[synthetic_topology.backend_name],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mode="advanced",
        topology_source="synthetic",
        synthetic_topology=synthetic_topology,
    )

    campaign = Campaign.from_config(campaign_id="campaign-synthetic", config=config)

    assert config.topology_source == "synthetic"
    assert config.backend_names == ("synthetic_grid_2x2",)
    assert campaign.build_cases()[0].case_id == "ghz_3__synthetic_grid_2x2"


def test_campaign_config_accepts_t_synthetic_topology_in_advanced_mode() -> None:
    synthetic_topology = SyntheticTopologySpec(shape="t", num_qubits=8)
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=8)],
        backend_names=[synthetic_topology.backend_name],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mode="advanced",
        topology_source="synthetic",
        synthetic_topology=synthetic_topology,
    )

    campaign = Campaign.from_config(campaign_id="campaign-synthetic-t", config=config)

    assert config.topology_source == "synthetic"
    assert config.backend_names == ("synthetic_t_8q",)
    assert campaign.build_cases()[0].case_id == "ghz_8__synthetic_t_8q"


def test_campaign_config_rejects_synthetic_topology_without_enough_physical_qubits() -> None:
    synthetic_topology = SyntheticTopologySpec(shape="line", num_qubits=2)

    try:
        CampaignConfig(
            circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
            backend_names=[synthetic_topology.backend_name],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="sequential",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
            mode="advanced",
            topology_source="synthetic",
            synthetic_topology=synthetic_topology,
        )
    except ValueError as exc:
        assert "physical qubits" in str(exc)
    else:
        raise AssertionError("Expected ValueError for undersized synthetic topology")


def test_campaign_config_rejects_synthetic_topology_in_default_mode() -> None:
    synthetic_topology = SyntheticTopologySpec(shape="line", num_qubits=3)

    try:
        CampaignConfig(
            circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
            backend_names=[synthetic_topology.backend_name],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="sequential",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
            topology_source="synthetic",
            synthetic_topology=synthetic_topology,
        )
    except ValueError as exc:
        assert "advanced" in str(exc)
    else:
        raise AssertionError("Expected ValueError for synthetic topology in default mode")


def test_campaign_config_defaults_to_auto_mo_effort_mode_and_accepts_custom() -> None:
    default_config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
    )
    custom_config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mo_effort_mode="custom",
    )

    assert default_config.mo_effort_mode == "auto"
    assert custom_config.mo_effort_mode == "custom"


def test_campaign_config_rejects_unknown_mo_effort_mode() -> None:
    try:
        CampaignConfig(
            circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
            backend_names=["fake_torino"],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="sequential",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
            mo_effort_mode="unexpected",
        )
    except ValueError as exc:
        assert "mo_effort_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown campaign mo_effort_mode")


def test_campaign_config_rejects_custom_population_below_shared_minimum() -> None:
    try:
        CampaignConfig(
            circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
            backend_names=["fake_torino"],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="sequential",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=MIN_CUSTOM_MO_POPULATION_SIZE - 1,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
            mo_effort_mode="custom",
        )
    except ValueError as exc:
        assert "mo_population_size" in str(exc)
        assert str(MIN_CUSTOM_MO_POPULATION_SIZE) in str(exc)
    else:
        raise AssertionError("Expected ValueError for custom MO population below shared minimum")


def test_campaign_config_rejects_unknown_mode() -> None:
    try:
        CampaignConfig(
            circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
            backend_names=["fake_torino"],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="sequential",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
            mode="unexpected",
        )
    except ValueError as exc:
        assert "mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown campaign mode")


def test_campaign_from_config_builds_runtime_campaign_separate_from_configuration() -> None:
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mode="advanced",
    )

    campaign = Campaign.from_config(campaign_id="campaign-001", config=config)

    assert campaign.campaign_id == "campaign-001"
    assert campaign.config is config
    assert campaign.status == "pending"
    assert campaign.summary is None


def test_campaign_config_is_frozen_once_validated() -> None:
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
    )

    try:
        config.mode = "advanced"
    except Exception as exc:
        assert exc.__class__.__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("Expected CampaignConfig to be immutable after validation")


def test_campaign_config_uses_immutable_sequences_after_validation() -> None:
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
    )

    assert isinstance(config.circuit_specs, tuple)
    assert isinstance(config.backend_names, tuple)

    try:
        config.backend_names.append("fake_brisbane")
    except AttributeError:
        pass
    else:
        raise AssertionError("Expected immutable backend_names sequence")


def test_campaign_rejects_terminal_status_as_default_runtime_state() -> None:
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
    )

    try:
        Campaign(campaign_id="campaign-001", config=config, status="completed")
    except ValueError as exc:
        assert "status" in str(exc)
    else:
        raise AssertionError("Expected Campaign to reject terminal status without summary")


def test_campaign_summary_copies_case_results_to_protect_invariants() -> None:
    case_results = [CampaignCaseResult(case_id="case-1", status="completed")]

    summary = CampaignSummary(
        status="completed",
        total_cases=1,
        comparable_completed_cases=1,
        failed_cases=0,
        incomplete_cases=0,
        cancelled_cases=0,
        case_results=case_results,
    )
    case_results.append(CampaignCaseResult(case_id="case-2", status="failed"))

    assert [result.case_id for result in summary.case_results] == ["case-1"]


def test_campaign_summary_rejects_non_terminal_status() -> None:
    try:
        CampaignSummary(
            status="pending",
            total_cases=1,
            comparable_completed_cases=0,
            failed_cases=0,
            incomplete_cases=0,
            cancelled_cases=0,
        )
    except ValueError as exc:
        assert "status" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-terminal summary status")


def test_campaign_rejects_summary_with_mismatched_status() -> None:
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
    )
    summary = CampaignSummary(
        status="failed",
        total_cases=1,
        comparable_completed_cases=0,
        failed_cases=1,
        incomplete_cases=0,
        cancelled_cases=0,
    )

    try:
        Campaign(
            campaign_id="campaign-001",
            config=config,
            status="completed",
            summary=summary,
        )
    except ValueError as exc:
        assert "summary" in str(exc) or "status" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched campaign and summary statuses")


def test_running_campaign_cannot_have_summary_yet() -> None:
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
    )
    summary = CampaignSummary(
        status="completed",
        total_cases=1,
        comparable_completed_cases=1,
        failed_cases=0,
        incomplete_cases=0,
        cancelled_cases=0,
    )

    try:
        Campaign(
            campaign_id="campaign-001",
            config=config,
            status="running",
            summary=summary,
        )
    except ValueError as exc:
        assert "summary" in str(exc)
    else:
        raise AssertionError("Expected ValueError when non-terminal campaign already carries summary")


def test_running_campaign_can_rehydrate_with_matching_running_summary() -> None:
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="sequential",
        rl_lookahead_window=10,
        rl_max_steps=200,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
    )
    summary = CampaignSummary(
        status="running",
        total_cases=2,
        comparable_completed_cases=1,
        failed_cases=0,
        incomplete_cases=0,
        cancelled_cases=0,
        case_results=[CampaignCaseResult(case_id="ghz_3__fake_torino", status="completed")],
    )

    campaign = Campaign(
        campaign_id="campaign-001",
        config=config,
        status="running",
        summary=summary,
    )

    assert campaign.status == "running"
    assert campaign.summary is summary


def test_campaign_config_exposes_shared_rl_and_mo_knobs_for_future_runner_and_training_bridge() -> None:
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_use_quick=False,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.BEST_ON_OBJECTIVE,
        mo_objective_name="depth",
    )

    assert config.rl_algorithm == "MaskablePPO"
    assert config.rl_frontier_mode == "dag"
    assert config.rl_lookahead_window == 12
    assert config.mo_use_quick is False
    assert config.mo_population_size == 30
    assert config.mo_n_generations == 50
    assert config.layout_policy is LayoutSelectionPolicy.BEST_ON_OBJECTIVE
    assert config.mo_objective_name == "depth"


def test_campaign_config_rejects_invalid_rl_and_mo_knobs() -> None:
    invalid_payloads = [
        {"rl_algorithm": "   "},
        {"rl_total_timesteps": 0},
        {"rl_frontier_mode": "   "},
        {"rl_lookahead_window": 0},
        {"rl_max_steps": 0},
        {"rl_learning_rate": 0},
        {"rl_clip_range": 0},
        {"rl_target_kl": 0},
        {"rl_n_eval_episodes": 0},
        {"seed": -1},
        {"mo_use_quick": "yes"},
        {"mo_population_size": 0},
        {"mo_n_generations": 0},
        {"layout_policy": "invalid"},
        {
            "layout_policy": LayoutSelectionPolicy.BEST_ON_OBJECTIVE,
            "mo_objective_name": "   ",
        },
        {
            "layout_policy": LayoutSelectionPolicy.COMPROMISE,
            "mo_objective_name": "depth",
        },
    ]

    for overrides in invalid_payloads:
        payload = {
            "circuit_specs": [CampaignCircuitSpec(family="ghz", num_qubits=3)],
            "backend_names": ["fake_torino"],
            "rl_algorithm": "MaskablePPO",
            "rl_total_timesteps": 5000,
            "rl_frontier_mode": "sequential",
            "rl_lookahead_window": 10,
            "rl_max_steps": 200,
            "seed": 42,
            "mo_use_quick": True,
            "mo_population_size": 30,
            "mo_n_generations": 50,
            "layout_policy": LayoutSelectionPolicy.COMPROMISE,
            "mo_effort_mode": "auto",
        }
        payload.update(overrides)

        try:
            CampaignConfig(**payload)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for invalid campaign config overrides: {overrides}")


def test_campaign_summary_keeps_aggregate_counts_separate_from_case_results() -> None:
    summary = CampaignSummary(
        status="completed",
        total_cases=4,
        comparable_completed_cases=2,
        failed_cases=1,
        incomplete_cases=1,
        cancelled_cases=0,
        case_results=[
            CampaignCaseResult(case_id="ghz_3__fake_torino", status="completed"),
            CampaignCaseResult(case_id="qft_5__fake_torino", status="failed"),
        ],
    )

    assert summary.status == "completed"
    assert summary.total_cases == 4
    assert summary.comparable_completed_cases == 2
    assert [result.case_id for result in summary.case_results] == [
        "ghz_3__fake_torino",
        "qft_5__fake_torino",
    ]


def test_campaign_summary_uses_immutable_case_results_sequence() -> None:
    summary = CampaignSummary(
        status="completed",
        total_cases=1,
        comparable_completed_cases=1,
        failed_cases=0,
        incomplete_cases=0,
        cancelled_cases=0,
        case_results=[CampaignCaseResult(case_id="case-1", status="completed")],
    )

    assert isinstance(summary.case_results, tuple)

    try:
        summary.case_results.append(CampaignCaseResult(case_id="case-2", status="failed"))
    except AttributeError:
        pass
    else:
        raise AssertionError("Expected immutable CampaignSummary case_results sequence")


def test_campaign_summary_rejects_negative_counts() -> None:
    try:
        CampaignSummary(
            status="completed",
            total_cases=-1,
            comparable_completed_cases=0,
            failed_cases=0,
            incomplete_cases=0,
            cancelled_cases=0,
        )
    except ValueError as exc:
        assert "total_cases" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative campaign aggregate counts")


def test_campaign_summary_rejects_impossible_aggregate_totals() -> None:
    try:
        CampaignSummary(
            status="completed",
            total_cases=2,
            comparable_completed_cases=1,
            failed_cases=1,
            incomplete_cases=1,
            cancelled_cases=0,
        )
    except ValueError as exc:
        assert "total_cases" in str(exc)
    else:
        raise AssertionError("Expected ValueError for impossible campaign aggregate totals")


def test_campaign_summary_rejects_completed_status_when_not_all_cases_are_accounted_for() -> None:
    try:
        CampaignSummary(
            status="completed",
            total_cases=3,
            comparable_completed_cases=1,
            failed_cases=0,
            incomplete_cases=0,
            cancelled_cases=0,
        )
    except ValueError as exc:
        assert "total_cases" in str(exc)
    else:
        raise AssertionError("Expected ValueError when completed summary leaves cases unaccounted for")


def test_campaign_summary_rejects_more_case_results_than_total_cases() -> None:
    try:
        CampaignSummary(
            status="completed",
            total_cases=1,
            comparable_completed_cases=1,
            failed_cases=0,
            incomplete_cases=0,
            cancelled_cases=0,
            case_results=[
                CampaignCaseResult(case_id="case-1", status="completed"),
                CampaignCaseResult(case_id="case-2", status="failed"),
            ],
        )
    except ValueError as exc:
        assert "case_results" in str(exc)
    else:
        raise AssertionError("Expected ValueError when case_results exceed total_cases")
