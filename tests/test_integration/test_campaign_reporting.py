from pathlib import Path

from src.integration.campaign_contracts import CampaignCase, CampaignCircuitSpec, CampaignConfig
from src.integration.campaign_reporting import (
    CampaignCaseReport,
    _report_to_dict,
    build_campaign_report,
    render_campaign_summary_markdown,
    write_campaign_outputs,
)
from src.integration.contracts import LayoutSelectionPolicy, ScenarioResult
from src.integration.synthetic_topology import SyntheticTopologySpec
from src.integration.training_bridge import TrainingBridgeResult, TrainingConfigSummary


def _build_campaign_config() -> CampaignConfig:
    return CampaignConfig(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=5),
        ],
        backend_names=["fake_torino", "fake_brisbane"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mo_effort_mode="custom",
        mode="advanced",
    )


def _build_case(case_id: str, family: str, num_qubits: int, backend_name: str) -> CampaignCase:
    return CampaignCase(
        case_id=case_id,
        circuit_family=family,
        num_qubits=num_qubits,
        backend_name=backend_name,
    )


def _build_metrics(depth: int, two_qubit: int, cnot_equivalent: float, elapsed: float) -> dict[str, float | int]:
    return {
        "trans_depth": depth,
        "trans_two_qubit_gates": two_qubit,
        "trans_cnot_equivalent": cnot_equivalent,
        "elapsed_time_s": elapsed,
    }


def _build_scenario_result(
    scenario_name: str,
    case: CampaignCase,
    *,
    success: bool = True,
    metrics: dict[str, float | int] | None = None,
    errors: list[str] | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_name=scenario_name,
        circuit_name=f"{case.circuit_family}_{case.num_qubits}",
        backend_name=case.backend_name,
        seed=42,
        success=success,
        selected_layout=[0, 1, 2] if case.num_qubits == 3 else [0, 1, 2, 3, 4],
        transpilation_metrics=metrics,
        errors=errors or [],
    )


def _build_training_result(case: CampaignCase, *, status: str = "completed") -> TrainingBridgeResult:
    run_model_dir = Path("campaigns") / "campaign-001" / "cases" / case.case_id / "training" / "models" / "run-001"
    run_log_dir = Path("campaigns") / "campaign-001" / "cases" / case.case_id / "training" / "logs" / "run-001"
    selected_artifact_path = run_model_dir / "best_model.zip" if status == "completed" else None
    return TrainingBridgeResult(
        status=status,
        selected_artifact_path=selected_artifact_path,
        best_model_path=selected_artifact_path,
        final_model_path=run_model_dir / "final_model.zip" if status == "completed" else None,
        run_model_dir=run_model_dir,
        run_log_dir=run_log_dir,
        effective_training_config=TrainingConfigSummary(
            algorithm="MaskablePPO",
            total_timesteps=5000,
            frontier_mode="dag",
            lookahead_window=12,
            max_steps=256,
            seed=42,
        ),
        actual_timesteps=355000,
        post_routing_selection={
            "has_valid_solution": True,
            "first_solution_timestep": 210000,
            "best_score": [143.0, 138, 25],
            "stop_reason": "post_routing_no_improvement",
        },
    )


def test_aggregate_summary_uses_only_comparable_completed_cases() -> None:
    config = _build_campaign_config()
    comparable_case_a = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")
    comparable_case_b = _build_case("qft_5__fake_torino", "qft", 5, "fake_torino")
    incomplete_case = _build_case("ghz_3__fake_brisbane", "ghz", 3, "fake_brisbane")

    report = build_campaign_report(
        campaign_id="campaign-001",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=comparable_case_a,
                status="completed",
                baseline_result=_build_scenario_result(
                    "Baseline", comparable_case_a, metrics=_build_metrics(100, 30, 45.0, 1.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", comparable_case_a, metrics=_build_metrics(90, 28, 40.0, 1.5)
                ),
                rl_only_result=_build_scenario_result(
                    "RL_Only", comparable_case_a, metrics=_build_metrics(85, 26, 38.0, 1.8)
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL", comparable_case_a, metrics=_build_metrics(80, 24, 36.0, 2.0)
                ),
                rl_only_training_result=_build_training_result(comparable_case_a),
                training_result=_build_training_result(comparable_case_a),
            ),
            CampaignCaseReport(
                case=comparable_case_b,
                status="completed",
                baseline_result=_build_scenario_result(
                    "Baseline", comparable_case_b, metrics=_build_metrics(200, 50, 75.0, 3.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", comparable_case_b, metrics=_build_metrics(180, 45, 68.0, 4.0)
                ),
                rl_only_result=_build_scenario_result(
                    "RL_Only", comparable_case_b, metrics=_build_metrics(165, 42, 63.0, 4.5)
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL", comparable_case_b, metrics=_build_metrics(150, 40, 60.0, 5.0)
                ),
                rl_only_training_result=_build_training_result(comparable_case_b),
                training_result=_build_training_result(comparable_case_b),
            ),
            CampaignCaseReport(
                case=incomplete_case,
                status="incomplete",
                baseline_result=_build_scenario_result(
                    "Baseline", incomplete_case, metrics=_build_metrics(110, 32, 46.0, 1.2)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", incomplete_case, metrics=_build_metrics(95, 29, 41.0, 1.6)
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL",
                    incomplete_case,
                    success=False,
                    metrics=None,
                    errors=["MO+RL routing episode did not complete."],
                ),
                training_result=_build_training_result(incomplete_case),
                incidents=["MO+RL routing episode did not complete."],
            ),
        ],
    )

    assert report.summary.total_cases == 3
    assert report.summary.comparable_completed_cases == 2
    assert report.summary.failed_cases == 0
    assert report.summary.incomplete_cases == 1
    assert report.summary.cancelled_cases == 0
    assert report.aggregate_metrics["trans_depth"].baseline_mean == 150.0
    assert report.aggregate_metrics["trans_depth"].mo_only_mean == 135.0
    assert report.aggregate_metrics["trans_depth"].rl_only_mean == 125.0
    assert report.aggregate_metrics["trans_depth"].mo_rl_mean == 115.0
    assert report.aggregate_metrics["trans_two_qubit_gates"].baseline_mean == 40.0
    assert report.aggregate_metrics["trans_cnot_equivalent"].mo_rl_mean == 48.0
    assert report.aggregate_metrics["elapsed_time_s"].mo_only_mean == 2.75


def test_aggregate_summary_counts_failed_and_incomplete_cases_separately() -> None:
    config = _build_campaign_config()
    failed_case = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")
    incomplete_case = _build_case("qft_5__fake_torino", "qft", 5, "fake_torino")
    cancelled_case = _build_case("ghz_3__fake_brisbane", "ghz", 3, "fake_brisbane")

    report = build_campaign_report(
        campaign_id="campaign-002",
        campaign_status="interrupted",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=failed_case,
                status="failed",
                baseline_result=_build_scenario_result(
                    "Baseline", failed_case, metrics=_build_metrics(100, 30, 45.0, 1.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", failed_case, metrics=_build_metrics(90, 28, 40.0, 1.5)
                ),
                training_result=_build_training_result(failed_case, status="failed"),
                incidents=["RL training failed before MO+RL evaluation."],
            ),
            CampaignCaseReport(
                case=incomplete_case,
                status="incomplete",
                baseline_result=_build_scenario_result(
                    "Baseline", incomplete_case, metrics=_build_metrics(200, 50, 75.0, 3.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", incomplete_case, metrics=_build_metrics(180, 45, 68.0, 4.0)
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL",
                    incomplete_case,
                    success=False,
                    metrics=None,
                    errors=["MO+RL routing episode did not complete."],
                ),
                training_result=_build_training_result(incomplete_case),
                incidents=["MO+RL routing episode did not complete."],
            ),
            CampaignCaseReport(
                case=cancelled_case,
                status="cancelled",
                incidents=["Campaign cancelled before this case started."],
            ),
        ],
    )

    assert report.campaign_status == "interrupted"
    assert report.summary.comparable_completed_cases == 0
    assert report.summary.failed_cases == 1
    assert report.summary.incomplete_cases == 1
    assert report.summary.cancelled_cases == 1
    assert report.summary.total_cases == 3
    assert report.incidents == [
        "ghz_3__fake_torino: RL training failed before MO+RL evaluation.",
        "qft_5__fake_torino: MO+RL routing episode did not complete.",
        "ghz_3__fake_brisbane: Campaign cancelled before this case started.",
    ]


def test_report_to_dict_does_not_patch_running_summary_status_during_serialization() -> None:
    config = _build_campaign_config()
    case = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-running",
        campaign_status="running",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=case,
                status="completed",
                baseline_result=_build_scenario_result("Baseline", case, metrics=_build_metrics(100, 30, 45.0, 1.0)),
                mo_only_result=_build_scenario_result("MO_Only", case, metrics=_build_metrics(90, 28, 40.0, 1.5)),
                mo_rl_result=_build_scenario_result("MO+RL", case, metrics=_build_metrics(80, 24, 36.0, 2.0)),
                training_result=_build_training_result(case),
            )
        ],
        total_cases=4,
    )

    payload = _report_to_dict(report)

    assert report.summary.status == "running"
    assert report.summary.total_cases == 4
    assert payload["campaign_status"] == "running"
    assert payload["summary"]["status"] == "running"
    assert payload["summary"]["total_cases"] == 4


def test_summary_markdown_includes_config_aggregate_case_detail_and_incidents() -> None:
    config = _build_campaign_config()
    completed_case = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")
    failed_case = _build_case("qft_5__fake_torino", "qft", 5, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-003",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=completed_case,
                status="completed",
                baseline_result=_build_scenario_result(
                    "Baseline", completed_case, metrics=_build_metrics(100, 30, 45.0, 1.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", completed_case, metrics=_build_metrics(90, 28, 40.0, 1.5)
                ),
                rl_only_result=_build_scenario_result(
                    "RL_Only", completed_case, metrics=_build_metrics(85, 26, 38.0, 1.8)
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL", completed_case, metrics=_build_metrics(80, 24, 36.0, 2.0)
                ),
                rl_only_training_result=_build_training_result(completed_case),
                training_result=_build_training_result(completed_case),
                incidents=["Used best_model.zip for MO+RL evaluation."],
            ),
            CampaignCaseReport(
                case=failed_case,
                status="failed",
                baseline_result=_build_scenario_result(
                    "Baseline", failed_case, metrics=_build_metrics(200, 50, 75.0, 3.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", failed_case, metrics=_build_metrics(180, 45, 68.0, 4.0)
                ),
                training_result=_build_training_result(failed_case, status="failed"),
                incidents=["RL training failed before MO+RL evaluation."],
            ),
        ],
    )

    markdown = render_campaign_summary_markdown(report)

    assert "# Campaign Summary" in markdown
    assert "Campaign ID: `campaign-003`" in markdown
    assert "Campaign Mode: `advanced`" in markdown
    assert "Selected Circuits" in markdown
    assert "ghz (3 qubits)" in markdown
    assert "qft (5 qubits)" in markdown
    assert "Selected Backends" in markdown
    assert "fake_torino" in markdown
    assert "Aggregate Comparison" in markdown
    assert "| Metric | Baseline Mean | MO_Only Mean | RL_Only Mean | MO+RL Mean |" in markdown
    assert "Per-Case Detail" in markdown
    assert "## Case `ghz_3__fake_torino`" in markdown
    assert "- Effective Config: rl_algorithm=MaskablePPO" in markdown
    assert "- Selected Layout: [0, 1, 2]" in markdown
    assert "RL Training Summary" in markdown
    assert "best_model.zip" in markdown
    assert "- Actual Timesteps: `355000`" in markdown
    assert "- Has Valid Solution: `True`" in markdown
    assert "- First Valid Solution Timestep: `210000`" in markdown
    assert "- Best Post-Routing Score: `[143.0, 138, 25]`" in markdown
    assert "- Training Stop Reason: `post_routing_no_improvement`" in markdown
    assert "Incidents" in markdown
    assert "RL training failed before MO+RL evaluation." in markdown
    assert "Final Campaign Status: `completed`" in markdown
    assert "Topology Source: `backend`" in markdown
    assert "RL Learning Rate: `0.0001`" in markdown
    assert "RL Clip Range: `0.1`" in markdown
    assert "RL Target KL: `0.03`" in markdown
    assert "RL Eval Episodes: `1`" in markdown
    assert "rl_learning_rate=0.0001" in markdown
    assert "rl_n_eval_episodes=1" in markdown
    assert "MO Effort Mode: `custom`" in markdown
    assert "MO Quick: `True`" in markdown
    assert "MO Population Size: `30`" in markdown
    assert "MO Generations: `50`" in markdown
    assert "MO Auto Preview" not in markdown


def test_summary_markdown_includes_synthetic_topology_configuration() -> None:
    synthetic_topology = SyntheticTopologySpec(shape="grid", rows=2, cols=2)
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=[synthetic_topology.backend_name],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mo_effort_mode="auto",
        mode="advanced",
        topology_source="synthetic",
        synthetic_topology=synthetic_topology,
    )
    case = _build_case("ghz_3__synthetic_grid_2x2", "ghz", 3, synthetic_topology.backend_name)
    report = build_campaign_report(
        campaign_id="campaign-synthetic",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=case,
                status="failed",
                incidents=["synthetic smoke"],
            )
        ],
    )

    markdown = render_campaign_summary_markdown(report)

    assert "Topology Source: `synthetic`" in markdown
    assert "Synthetic Topology: `synthetic_grid_2x2`" in markdown
    assert "Synthetic Shape: `grid`" in markdown
    assert "Synthetic Physical Qubits: `4`" in markdown
    assert "Synthetic Basis Gates: `id, rz, sx, x, cx`" in markdown
    assert "Selected Synthetic Topology" in markdown
    assert "Selected Backends" not in markdown


def test_summary_markdown_includes_mo_objective_name_for_best_on_objective_policy() -> None:
    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_effort_mode="custom",
        mo_use_quick=False,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.BEST_ON_OBJECTIVE,
        mo_objective_name="depth",
        mode="advanced",
    )
    completed_case = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-005",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=completed_case,
                status="completed",
                baseline_result=_build_scenario_result(
                    "Baseline", completed_case, metrics=_build_metrics(100, 30, 45.0, 1.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", completed_case, metrics=_build_metrics(90, 28, 40.0, 1.5)
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL", completed_case, metrics=_build_metrics(80, 24, 36.0, 2.0)
                ),
                training_result=_build_training_result(completed_case),
            )
        ],
    )

    markdown = render_campaign_summary_markdown(report)

    assert "Layout Policy: `best_on_objective`" in markdown
    assert "MO Objective: `depth`" in markdown


def test_summary_markdown_renders_auto_mo_effort_preview() -> None:
    config = CampaignConfig(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=8),
        ],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mo_effort_mode="auto",
        mode="advanced",
    )
    case = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-auto-preview",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=case,
                status="completed",
                baseline_result=_build_scenario_result("Baseline", case, metrics=_build_metrics(100, 30, 45.0, 1.0)),
                mo_only_result=_build_scenario_result("MO_Only", case, metrics=_build_metrics(90, 28, 40.0, 1.5)),
                mo_rl_result=_build_scenario_result("MO+RL", case, metrics=_build_metrics(80, 24, 36.0, 2.0)),
                training_result=_build_training_result(case),
            )
        ],
    )

    markdown = render_campaign_summary_markdown(report)

    assert "MO Effort Mode: `auto`" in markdown
    assert "MO Auto Preview (3q): `quick=True, population_size=30, n_generations=50`" in markdown
    assert "MO Auto Preview (8q): `quick=False, population_size=60, n_generations=120`" in markdown
    assert "MO Quick:" not in markdown
    assert "MO Population Size:" not in markdown
    assert "MO Generations:" not in markdown


def test_running_summary_markdown_does_not_claim_final_campaign_status() -> None:
    config = _build_campaign_config()
    case = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-running-markdown",
        campaign_status="running",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=case,
                status="completed",
                baseline_result=_build_scenario_result("Baseline", case, metrics=_build_metrics(100, 30, 45.0, 1.0)),
                mo_only_result=_build_scenario_result("MO_Only", case, metrics=_build_metrics(90, 28, 40.0, 1.5)),
                mo_rl_result=_build_scenario_result("MO+RL", case, metrics=_build_metrics(80, 24, 36.0, 2.0)),
                training_result=_build_training_result(case),
            )
        ],
        total_cases=4,
    )

    markdown = render_campaign_summary_markdown(report)

    assert "Campaign Status: `running`" in markdown
    assert "Final Campaign Status" not in markdown


def test_completed_non_comparable_cases_are_reported_without_zero_aggregate_values() -> None:
    config = _build_campaign_config()
    non_comparable_case = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-006",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=non_comparable_case,
                status="completed",
                baseline_result=_build_scenario_result(
                    "Baseline", non_comparable_case, metrics=_build_metrics(100, 30, 45.0, 1.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", non_comparable_case, metrics=_build_metrics(90, 28, 40.0, 1.5)
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL",
                    non_comparable_case,
                    success=False,
                    metrics=None,
                    errors=["MO+RL metrics unavailable after case completion."],
                ),
            )
        ],
    )

    markdown = render_campaign_summary_markdown(report)

    assert report.summary.total_cases == 1
    assert report.summary.comparable_completed_cases == 0
    assert report.summary.incomplete_cases == 1
    assert report.summary.failed_cases == 0
    assert report.summary.cancelled_cases == 0
    assert report.summary.case_results[0].status == "incomplete"
    assert report.incidents == [
        "ghz_3__fake_torino: completed without a comparable metric bundle across Baseline, MO_Only, RL_Only, and MO+RL."
    ]
    assert "| trans_depth | n/a | n/a | n/a |" in markdown
    assert "completed without a comparable metric bundle" in markdown


def test_write_campaign_outputs_persists_summary_campaign_and_per_case_json(tmp_path) -> None:
    config = _build_campaign_config()
    completed_case = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-004",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=completed_case,
                status="completed",
                baseline_result=_build_scenario_result(
                    "Baseline", completed_case, metrics=_build_metrics(100, 30, 45.0, 1.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", completed_case, metrics=_build_metrics(90, 28, 40.0, 1.5)
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL", completed_case, metrics=_build_metrics(80, 24, 36.0, 2.0)
                ),
                training_result=_build_training_result(completed_case),
            )
        ],
    )

    output_paths = write_campaign_outputs(output_dir=tmp_path / "campaigns" / report.campaign_id, report=report)

    assert output_paths.summary_markdown_path.exists()
    assert output_paths.campaign_json_path.exists()
    assert output_paths.case_result_paths[completed_case.case_id].exists()
    assert '"campaign_id": "campaign-004"' in output_paths.campaign_json_path.read_text(encoding="utf-8")
    assert '"status": "completed"' in output_paths.case_result_paths[completed_case.case_id].read_text(
        encoding="utf-8"
    )
    assert "# Campaign Summary" in output_paths.summary_markdown_path.read_text(encoding="utf-8")


def test_write_campaign_outputs_sanitizes_case_directory_names(tmp_path) -> None:
    config = _build_campaign_config()
    unsafe_case = _build_case("..\\ghz/3:bad*case?", "ghz", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-007",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=unsafe_case,
                status="completed",
                baseline_result=_build_scenario_result(
                    "Baseline", unsafe_case, metrics=_build_metrics(100, 30, 45.0, 1.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", unsafe_case, metrics=_build_metrics(90, 28, 40.0, 1.5)
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL", unsafe_case, metrics=_build_metrics(80, 24, 36.0, 2.0)
                ),
                training_result=_build_training_result(unsafe_case),
            )
        ],
    )

    output_paths = write_campaign_outputs(output_dir=tmp_path / "campaigns" / report.campaign_id, report=report)
    case_result_path = output_paths.case_result_paths[unsafe_case.case_id]

    assert case_result_path.exists()
    assert case_result_path.parent.parent.name == "cases"
    assert case_result_path.parent.name != unsafe_case.case_id
    assert ".." not in case_result_path.parent.name
    for invalid_char in '<>:"/\\|?*':
        assert invalid_char not in case_result_path.parent.name


def test_write_campaign_outputs_avoids_windows_reserved_case_directory_names(tmp_path) -> None:
    config = _build_campaign_config()
    reserved_case = _build_case("CON", "ghz", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-008",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=reserved_case,
                status="completed",
                baseline_result=_build_scenario_result(
                    "Baseline", reserved_case, metrics=_build_metrics(100, 30, 45.0, 1.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", reserved_case, metrics=_build_metrics(90, 28, 40.0, 1.5)
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL", reserved_case, metrics=_build_metrics(80, 24, 36.0, 2.0)
                ),
                training_result=_build_training_result(reserved_case),
            )
        ],
    )

    output_paths = write_campaign_outputs(output_dir=tmp_path / "campaigns" / report.campaign_id, report=report)
    case_result_path = output_paths.case_result_paths[reserved_case.case_id]

    assert case_result_path.exists()
    assert case_result_path.parent.name == "CON_"


def test_write_campaign_outputs_avoids_case_only_directory_collisions_on_windows(tmp_path) -> None:
    config = _build_campaign_config()
    upper_case = _build_case("Case", "ghz", 3, "fake_torino")
    lower_case = _build_case("case", "qft", 5, "fake_brisbane")

    report = build_campaign_report(
        campaign_id="campaign-008b",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=upper_case,
                status="completed",
                baseline_result=_build_scenario_result("Baseline", upper_case, metrics=_build_metrics(100, 30, 45.0, 1.0)),
                mo_only_result=_build_scenario_result("MO_Only", upper_case, metrics=_build_metrics(90, 28, 40.0, 1.5)),
                mo_rl_result=_build_scenario_result("MO+RL", upper_case, metrics=_build_metrics(80, 24, 36.0, 2.0)),
                training_result=_build_training_result(upper_case),
            ),
            CampaignCaseReport(
                case=lower_case,
                status="completed",
                baseline_result=_build_scenario_result("Baseline", lower_case, metrics=_build_metrics(110, 32, 46.0, 1.2)),
                mo_only_result=_build_scenario_result("MO_Only", lower_case, metrics=_build_metrics(95, 29, 41.0, 1.6)),
                mo_rl_result=_build_scenario_result("MO+RL", lower_case, metrics=_build_metrics(85, 25, 37.0, 2.1)),
                training_result=_build_training_result(lower_case),
            ),
        ],
    )

    output_paths = write_campaign_outputs(output_dir=tmp_path / "campaigns" / report.campaign_id, report=report)

    first_dir = output_paths.case_result_paths[upper_case.case_id].parent.name
    second_dir = output_paths.case_result_paths[lower_case.case_id].parent.name

    assert first_dir == "Case"
    assert second_dir == "case_2"
    assert first_dir.casefold() != second_dir.casefold()


def test_write_campaign_outputs_avoids_windows_reserved_case_stems_with_extensions(tmp_path) -> None:
    config = _build_campaign_config()
    reserved_case = _build_case("CON.txt", "ghz", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-008c",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=reserved_case,
                status="completed",
                baseline_result=_build_scenario_result(
                    "Baseline", reserved_case, metrics=_build_metrics(100, 30, 45.0, 1.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only", reserved_case, metrics=_build_metrics(90, 28, 40.0, 1.5)
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL", reserved_case, metrics=_build_metrics(80, 24, 36.0, 2.0)
                ),
                training_result=_build_training_result(reserved_case),
            )
        ],
    )

    output_paths = write_campaign_outputs(output_dir=tmp_path / "campaigns" / report.campaign_id, report=report)
    case_result_path = output_paths.case_result_paths[reserved_case.case_id]

    assert case_result_path.exists()
    assert case_result_path.parent.name == "CON_.txt"


def test_summary_markdown_hides_failed_scenario_leftover_metrics_and_layout() -> None:
    config = _build_campaign_config()
    case = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-009",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=case,
                status="completed",
                baseline_result=_build_scenario_result(
                    "Baseline", case, metrics=_build_metrics(100, 30, 45.0, 1.0)
                ),
                mo_only_result=_build_scenario_result(
                    "MO_Only",
                    case,
                    success=False,
                    metrics=_build_metrics(90, 28, 40.0, 1.5),
                    errors=["MO selection failed after partial payload creation."],
                ),
                mo_rl_result=_build_scenario_result(
                    "MO+RL", case, metrics=_build_metrics(80, 24, 36.0, 2.0)
                ),
                training_result=_build_training_result(case),
            )
        ],
    )

    markdown = render_campaign_summary_markdown(report)

    assert "- MO_Only: unavailable" in markdown
    assert "- Selected Layout: [0, 1, 2]" in markdown
    assert "- MO_Only: depth=90" not in markdown


def test_summary_markdown_references_case_id_and_training_artifact_path() -> None:
    config = _build_campaign_config()
    case = _build_case("ghz_bad_3__fake_torino", "ghz..bad", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-010",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=case,
                status="completed",
                baseline_result=_build_scenario_result("Baseline", case, metrics=_build_metrics(100, 30, 45.0, 1.0)),
                mo_only_result=_build_scenario_result("MO_Only", case, metrics=_build_metrics(90, 28, 40.0, 1.5)),
                mo_rl_result=_build_scenario_result("MO+RL", case, metrics=_build_metrics(80, 24, 36.0, 2.0)),
                training_result=TrainingBridgeResult(
                    status="completed",
                    selected_artifact_path=Path("campaigns")
                    / "campaign-010"
                    / "cases"
                    / case.case_id
                    / "training"
                    / "models"
                    / "run-001"
                    / "best_model.zip",
                    best_model_path=Path("campaigns")
                    / "campaign-010"
                    / "cases"
                    / case.case_id
                    / "training"
                    / "models"
                    / "run-001"
                    / "best_model.zip",
                    final_model_path=Path("campaigns")
                    / "campaign-010"
                    / "cases"
                    / case.case_id
                    / "training"
                    / "models"
                    / "run-001"
                    / "final_model.zip",
                    run_model_dir=Path("campaigns")
                    / "campaign-010"
                    / "cases"
                    / case.case_id
                    / "training"
                    / "models"
                    / "run-001",
                    run_log_dir=Path("campaigns")
                    / "campaign-010"
                    / "cases"
                    / case.case_id
                    / "training"
                    / "logs"
                    / "run-001",
                    effective_training_config=TrainingConfigSummary(
                        algorithm="MaskablePPO",
                        total_timesteps=5000,
                        frontier_mode="dag",
                        lookahead_window=12,
                        max_steps=256,
                        seed=42,
                    ),
                ),
            )
        ],
    )

    markdown = render_campaign_summary_markdown(report)

    assert "## Case `ghz_bad_3__fake_torino`" in markdown
    assert "- Selected Artifact: `campaigns/campaign-010/cases/ghz_bad_3__fake_torino/training/models/run-001/best_model.zip`" in markdown


def test_render_campaign_summary_markdown_includes_mo_rl_routing_graph_notes() -> None:
    case = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")
    report = build_campaign_report(
        campaign_id="campaign-001",
        campaign_status="completed",
        campaign_config=_build_campaign_config(),
        case_reports=[
            CampaignCaseReport(
                case=case,
                status="completed",
                baseline_result=_build_scenario_result("Baseline", case, metrics=_build_metrics(100, 30, 45.0, 1.0)),
                mo_only_result=_build_scenario_result("MO_Only", case, metrics=_build_metrics(90, 28, 40.0, 1.5)),
                mo_rl_result=ScenarioResult(
                    scenario_name="MO+RL",
                    circuit_name="ghz_3",
                    backend_name="fake_torino",
                    seed=42,
                    success=True,
                    selected_layout=[2, 0, 1],
                    transpilation_metrics=_build_metrics(80, 24, 36.0, 2.0),
                    notes=[
                        "Campaign routing graph: mode=path_expanded_subgraph, nodes=3, edges=2, interacting_pairs=1, added_intermediate_qubits=[]"
                    ],
                ),
                training_result=_build_training_result(case),
            )
        ],
    )

    markdown = render_campaign_summary_markdown(report)

    assert "MO+RL Notes" in markdown
    assert "Campaign routing graph: mode=path_expanded_subgraph" in markdown


def test_write_campaign_outputs_rewrites_absolute_training_artifact_paths_to_campaign_public_paths(tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_case("ghz_bad_3__fake_torino", "ghz..bad", 3, "fake_torino")
    absolute_case_dir = tmp_path / "campaigns" / "campaign-011" / "cases" / case.case_id
    absolute_artifact_path = absolute_case_dir / "training" / "models" / "run-001" / "best_model.zip"

    report = build_campaign_report(
        campaign_id="campaign-011",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=case,
                status="completed",
                baseline_result=_build_scenario_result("Baseline", case, metrics=_build_metrics(100, 30, 45.0, 1.0)),
                mo_only_result=_build_scenario_result("MO_Only", case, metrics=_build_metrics(90, 28, 40.0, 1.5)),
                mo_rl_result=_build_scenario_result("MO+RL", case, metrics=_build_metrics(80, 24, 36.0, 2.0)),
                training_result=TrainingBridgeResult(
                    status="completed",
                    selected_artifact_path=absolute_artifact_path,
                    best_model_path=absolute_artifact_path,
                    final_model_path=absolute_artifact_path.parent / "final_model.zip",
                    run_model_dir=absolute_artifact_path.parent,
                    run_log_dir=absolute_case_dir / "training" / "logs" / "run-001",
                    effective_training_config=TrainingConfigSummary(
                        algorithm="MaskablePPO",
                        total_timesteps=5000,
                        frontier_mode="dag",
                        lookahead_window=12,
                        max_steps=256,
                        seed=42,
                    ),
                ),
            )
        ],
    )

    output_paths = write_campaign_outputs(output_dir=tmp_path / "campaigns" / report.campaign_id, report=report)
    summary_markdown = output_paths.summary_markdown_path.read_text(encoding="utf-8")
    campaign_json = output_paths.campaign_json_path.read_text(encoding="utf-8")
    case_result_json = output_paths.case_result_paths[case.case_id].read_text(encoding="utf-8")

    assert str(tmp_path) not in summary_markdown
    assert str(tmp_path) not in campaign_json
    assert str(tmp_path) not in case_result_json
    assert "campaigns/campaign-011/cases/ghz_bad_3__fake_torino/training/models/run-001/best_model.zip" in summary_markdown
    assert '"selected_artifact_path": "campaigns/campaign-011/cases/ghz_bad_3__fake_torino/training/models/run-001/best_model.zip"' in campaign_json
    assert '"selected_artifact_path": "campaigns/campaign-011/cases/ghz_bad_3__fake_torino/training/models/run-001/best_model.zip"' in case_result_json
