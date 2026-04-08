from dataclasses import asdict, fields

import src.integration as integration
from src.integration import (
    LayoutSelectionPolicy,
    RoutingEpisodeSummary,
    ScenarioRequest,
    ScenarioResult,
)


def test_scenario_request_defaults_for_routing_v1() -> None:
    request = ScenarioRequest(
        scenario_name="MO_Only",
        circuit_name="ghz_5",
        num_qubits=5,
        backend_name="fake_backend",
    )

    assert request.seed == 42
    assert request.scenario_name == "MO_Only"
    assert request.layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert request.mo_use_quick is True
    assert request.initial_layout is None
    assert request.rl_model_path is None
    assert request.mo_objective_index == 0
    assert fields(ScenarioRequest)[0].type is str
    assert [field.name for field in fields(ScenarioRequest)] == [
        "scenario_name",
        "circuit_name",
        "num_qubits",
        "backend_name",
        "seed",
        "layout_policy",
        "mo_use_quick",
        "initial_layout",
        "rl_model_path",
        "mo_objective_index",
    ]


def test_public_integration_contracts_do_not_export_scenario_name() -> None:
    assert integration.__all__ == [
        "LayoutSelectionPolicy",
        "RoutingEpisodeSummary",
        "ScenarioRequest",
        "ScenarioResult",
    ]
    assert not hasattr(integration, "ScenarioName")


def test_scenario_request_rejects_non_positive_num_qubits() -> None:
    try:
        ScenarioRequest(
            scenario_name="Baseline",
            circuit_name="ghz",
            num_qubits=0,
            backend_name="fake_torino",
        )
    except ValueError as exc:
        assert "num_qubits" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive num_qubits")


def test_scenario_request_rejects_initial_layout_length_mismatch() -> None:
    try:
        ScenarioRequest(
            scenario_name="RL_Only",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
            initial_layout=[0, 1],
        )
    except ValueError as exc:
        assert "initial_layout" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched initial_layout length")


def test_scenario_request_rejects_duplicate_initial_layout_entries() -> None:
    try:
        ScenarioRequest(
            scenario_name="RL_Only",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
            initial_layout=[0, 0, 1],
        )
    except ValueError as exc:
        assert "duplicated" in str(exc)
    else:
        raise AssertionError("Expected ValueError for duplicate initial_layout entries")


def test_scenario_request_rejects_negative_initial_layout_entries() -> None:
    try:
        ScenarioRequest(
            scenario_name="RL_Only",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
            initial_layout=[0, -1, 2],
        )
    except ValueError as exc:
        assert "initial_layout" in str(exc)
        assert "negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative initial_layout entries")


def test_scenario_request_rejects_negative_mo_objective_index() -> None:
    try:
        ScenarioRequest(
            scenario_name="MO_Only",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
            mo_objective_index=-1,
        )
    except ValueError as exc:
        assert "mo_objective_index" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative mo_objective_index")


def test_scenario_request_rejects_unknown_scenario_name() -> None:
    try:
        ScenarioRequest(
            scenario_name="invalid",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
        )
    except ValueError as exc:
        assert "scenario_name" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown scenario_name")


def test_baseline_request_rejects_integration_specific_inputs() -> None:
    for kwargs in (
        {"initial_layout": [0, 1, 2]},
        {"rl_model_path": "models/policy.zip"},
        {"layout_policy": LayoutSelectionPolicy.BEST_ON_OBJECTIVE},
        {"mo_use_quick": False},
        {"mo_objective_index": 1},
    ):
        try:
            ScenarioRequest(
                scenario_name="Baseline",
                circuit_name="ghz",
                num_qubits=3,
                backend_name="fake_torino",
                **kwargs,
            )
        except ValueError as exc:
            assert "Baseline" in str(exc)
        else:
            raise AssertionError(f"Expected ValueError for Baseline kwargs: {kwargs}")


def test_mo_only_request_rejects_rl_model_path() -> None:
    try:
        ScenarioRequest(
            scenario_name="MO_Only",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
            rl_model_path="models/policy.zip",
        )
    except ValueError as exc:
        assert "MO_Only" in str(exc)
        assert "rl_model_path" in str(exc)
    else:
        raise AssertionError("Expected ValueError for MO_Only rl_model_path")


def test_mo_only_request_rejects_caller_supplied_initial_layout() -> None:
    try:
        ScenarioRequest(
            scenario_name="MO_Only",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
            initial_layout=[0, 1, 2],
        )
    except ValueError as exc:
        assert "MO_Only" in str(exc)
        assert "initial_layout" in str(exc)
    else:
        raise AssertionError("Expected ValueError for MO_Only initial_layout")


def test_rl_only_request_rejects_mo_selection_knobs() -> None:
    for kwargs in (
        {"layout_policy": LayoutSelectionPolicy.BEST_ON_OBJECTIVE},
        {"mo_use_quick": False},
        {"mo_objective_index": 1},
    ):
        try:
            ScenarioRequest(
                scenario_name="RL_Only",
                circuit_name="ghz",
                num_qubits=3,
                backend_name="fake_torino",
                **kwargs,
            )
        except ValueError as exc:
            assert "RL_Only" in str(exc)
        else:
            raise AssertionError(f"Expected ValueError for RL_Only kwargs: {kwargs}")


def test_rl_only_request_requires_rl_model_path() -> None:
    try:
        ScenarioRequest(
            scenario_name="RL_Only",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
        )
    except ValueError as exc:
        assert "RL_Only" in str(exc)
        assert "rl_model_path" in str(exc)
    else:
        raise AssertionError("Expected ValueError when RL_Only has no rl_model_path")


def test_mo_rl_request_requires_rl_model_path() -> None:
    try:
        ScenarioRequest(
            scenario_name="MO+RL",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
        )
    except ValueError as exc:
        assert "MO+RL" in str(exc)
        assert "rl_model_path" in str(exc)
    else:
        raise AssertionError("Expected ValueError when MO+RL has no rl_model_path")


def test_mo_rl_request_rejects_caller_supplied_initial_layout() -> None:
    try:
        ScenarioRequest(
            scenario_name="MO+RL",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
            rl_model_path="models/policy.zip",
            initial_layout=[0, 1, 2],
        )
    except ValueError as exc:
        assert "MO+RL" in str(exc)
        assert "initial_layout" in str(exc)
    else:
        raise AssertionError("Expected ValueError for MO+RL initial_layout")


def test_routing_episode_summary_captures_episode_level_outputs() -> None:
    summary = RoutingEpisodeSummary(
        initial_layout=[0, 2, 1],
        final_layout=[2, 0, 1],
        steps_executed=7,
        total_reward=3.5,
        completed=True,
        truncated=False,
        total_swaps=2,
        gates_executed_count=11,
    )

    assert asdict(summary) == {
        "initial_layout": [0, 2, 1],
        "final_layout": [2, 0, 1],
        "steps_executed": 7,
        "total_reward": 3.5,
        "completed": True,
        "truncated": False,
        "total_swaps": 2,
        "gates_executed_count": 11,
    }


def test_routing_episode_summary_rejects_negative_counts() -> None:
    for steps_executed, total_swaps, gates_executed_count in (
        (-1, 1, 9),
        (4, -1, 9),
        (4, 1, -1),
    ):
        try:
            RoutingEpisodeSummary(
                initial_layout=[0, 1],
                final_layout=[1, 0],
                steps_executed=steps_executed,
                total_reward=1.0,
                completed=True,
                truncated=False,
                total_swaps=total_swaps,
                gates_executed_count=gates_executed_count,
            )
        except ValueError as exc:
            assert "negative" in str(exc)
        else:
            raise AssertionError(
                "Expected ValueError for negative RoutingEpisodeSummary counts"
            )


def test_routing_episode_summary_rejects_completed_and_truncated() -> None:
    try:
        RoutingEpisodeSummary(
            initial_layout=[0, 1],
            final_layout=[1, 0],
            steps_executed=4,
            total_reward=1.0,
            completed=True,
            truncated=True,
            total_swaps=1,
            gates_executed_count=9,
        )
    except ValueError as exc:
        assert "completed" in str(exc)
        assert "truncated" in str(exc)
    else:
        raise AssertionError("Expected ValueError for completed and truncated summary")


def test_routing_episode_summary_rejects_invalid_layout_entries() -> None:
    for initial_layout, final_layout in (
        ([0, -1], [1, 0]),
        ([0, 0], [1, 0]),
        ([0, 1], [1, -1]),
        ([0, 1], [1, 1]),
    ):
        try:
            RoutingEpisodeSummary(
                initial_layout=initial_layout,
                final_layout=final_layout,
                steps_executed=4,
                total_reward=1.0,
                completed=True,
                truncated=False,
                total_swaps=1,
                gates_executed_count=9,
            )
        except ValueError as exc:
            assert "layout" in str(exc)
        else:
            raise AssertionError(
                "Expected ValueError for invalid RoutingEpisodeSummary layouts"
            )


def test_routing_episode_summary_rejects_mismatched_layout_lengths() -> None:
    try:
        RoutingEpisodeSummary(
            initial_layout=[0, 1, 2],
            final_layout=[1, 0],
            steps_executed=4,
            total_reward=1.0,
            completed=True,
            truncated=False,
            total_swaps=1,
            gates_executed_count=9,
        )
    except ValueError as exc:
        assert "initial_layout" in str(exc)
        assert "final_layout" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched layout lengths")


def test_scenario_result_keeps_transpilation_and_routing_results_separate() -> None:
    routing_summary = RoutingEpisodeSummary(
        initial_layout=[0, 1],
        final_layout=[1, 0],
        steps_executed=4,
        total_reward=1.25,
        completed=True,
        truncated=False,
        total_swaps=1,
        gates_executed_count=9,
    )

    result = ScenarioResult(
        scenario_name="RL_Only",
        circuit_name="bell",
        backend_name="fake_backend",
        seed=99,
        success=True,
        selected_layout=[1, 0],
        transpilation_metrics={"backend_name": "fake_backend", "trans_depth": 12},
        routing_summary=routing_summary,
    )

    assert result.transpilation_metrics == {"backend_name": "fake_backend", "trans_depth": 12}
    assert result.routing_summary is routing_summary
    assert result.errors == []
    assert result.notes == []
    assert fields(ScenarioResult)[0].type is str


def test_scenario_result_rejects_success_with_errors() -> None:
    try:
        ScenarioResult(
            scenario_name="RL_Only",
            circuit_name="bell",
            backend_name="fake_backend",
            seed=99,
            success=True,
            selected_layout=[1, 0],
            transpilation_metrics={"backend_name": "fake_backend", "trans_depth": 12},
            routing_summary=None,
            errors=["routing failed"],
        )
    except ValueError as exc:
        assert "success" in str(exc)
        assert "errors" in str(exc)
    else:
        raise AssertionError("Expected ValueError for success=True with errors")


def test_scenario_result_rejects_invalid_selected_layout_entries() -> None:
    for selected_layout in ([0, -1], [1, 1]):
        try:
            ScenarioResult(
                scenario_name="RL_Only",
                circuit_name="bell",
                backend_name="fake_backend",
                seed=99,
                success=False,
                selected_layout=selected_layout,
                transpilation_metrics=None,
                routing_summary=None,
                errors=["routing failed"],
            )
        except ValueError as exc:
            assert "selected_layout" in str(exc)
        else:
            raise AssertionError(
                f"Expected ValueError for invalid selected_layout: {selected_layout}"
            )
