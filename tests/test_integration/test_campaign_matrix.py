from pathlib import Path

from src.integration.campaign_contracts import Campaign, CampaignCase, CampaignCircuitSpec, CampaignConfig
from src.integration.campaign_matrix import (
    ALL_MO_SELECTION_MODES,
    _group_children_by_seed,
    build_matrix_report,
    expand_campaign_matrix,
    render_matrix_summary_markdown,
    run_campaign_matrix,
)
from src.integration.campaign_reporting import CampaignCaseReport, build_campaign_report
from src.integration.contracts import LayoutSelectionPolicy, ScenarioResult


def _build_config(*, seeds=(1, 2), mo_selection_modes=ALL_MO_SELECTION_MODES) -> CampaignConfig:
    return CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=seeds[0],
        seeds=seeds,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mo_selection_modes=mo_selection_modes,
        parallel_workers=2,
        mode="advanced",
    )


def _case() -> CampaignCase:
    return CampaignCase(
        case_id="ghz_3__fake_torino",
        circuit_family="ghz",
        num_qubits=3,
        backend_name="fake_torino",
    )


def _metrics(depth: int) -> dict[str, float | int]:
    return {
        "trans_depth": depth,
        "trans_two_qubit_gates": depth // 2,
        "trans_cnot_equivalent": float(depth),
        "elapsed_time_s": float(depth) / 100.0,
    }


def _scenario(scenario_name: str, *, depth: int, seed: int) -> ScenarioResult:
    return ScenarioResult(
        scenario_name=scenario_name,
        circuit_name="ghz_3",
        backend_name="fake_torino",
        seed=seed,
        success=True,
        selected_layout=[0, 1, 2] if scenario_name != "Baseline" else None,
        transpilation_metrics=_metrics(depth),
    )


def _child_report(*, campaign_id: str, seed: int, mode: str, base_depth: int, comparable: bool = True):
    case = _case()
    status = "completed" if comparable else "incomplete"
    return build_campaign_report(
        campaign_id=campaign_id,
        campaign_status="completed",
        campaign_config=_build_config(seeds=(seed,), mo_selection_modes=(mode,)),
        case_reports=[
            CampaignCaseReport(
                case=case,
                status=status,
                baseline_result=_scenario("Baseline", depth=base_depth + 10, seed=seed),
                rl_only_result=_scenario("RL_Only", depth=base_depth + 5, seed=seed) if comparable else None,
                mo_only_result=_scenario("MO_Only", depth=base_depth + 3, seed=seed),
                mo_rl_result=_scenario("MO+RL", depth=base_depth, seed=seed) if comparable else None,
                incidents=[] if comparable else ["MO+RL routing episode did not complete."],
            )
        ],
    )


def test_expand_campaign_matrix_builds_seed_by_mo_mode_children() -> None:
    campaign = Campaign.from_config(campaign_id="campaign-matrix", config=_build_config())

    children = expand_campaign_matrix(campaign)

    assert [child.campaign_id for child in children] == [
        "campaign-matrix__seed_1__compromise",
        "campaign-matrix__seed_1__best_depth",
        "campaign-matrix__seed_1__best_cnot_count",
        "campaign-matrix__seed_2__compromise",
        "campaign-matrix__seed_2__best_depth",
        "campaign-matrix__seed_2__best_cnot_count",
    ]
    assert [child.config.seed for child in children] == [1, 1, 1, 2, 2, 2]
    assert children[0].config.layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert children[1].config.layout_policy is LayoutSelectionPolicy.BEST_ON_OBJECTIVE
    assert children[1].config.mo_objective_name == "depth"
    assert children[2].config.mo_objective_name == "cnot_count"


def test_expand_campaign_matrix_supports_opt_in_hybrid_probe_without_changing_all_alias() -> None:
    campaign = Campaign.from_config(
        campaign_id="campaign-matrix",
        config=_build_config(seeds=(1,), mo_selection_modes=("hybrid_probe",)),
    )

    children = expand_campaign_matrix(campaign)

    assert ALL_MO_SELECTION_MODES == ("compromise", "best_depth", "best_cnot_count")
    assert [child.campaign_id for child in children] == [
        "campaign-matrix__seed_1__hybrid_probe",
    ]
    assert children[0].config.mo_selection_modes == ("hybrid_probe",)
    assert children[0].config.layout_policy is LayoutSelectionPolicy.COMPROMISE


def test_group_children_by_seed_keeps_mo_modes_together() -> None:
    campaign = Campaign.from_config(campaign_id="campaign-matrix", config=_build_config())

    groups = _group_children_by_seed(expand_campaign_matrix(campaign))

    assert [[child.campaign_id for child in group] for group in groups] == [
        [
            "campaign-matrix__seed_1__compromise",
            "campaign-matrix__seed_1__best_depth",
            "campaign-matrix__seed_1__best_cnot_count",
        ],
        [
            "campaign-matrix__seed_2__compromise",
            "campaign-matrix__seed_2__best_depth",
            "campaign-matrix__seed_2__best_cnot_count",
        ],
    ]


def test_matrix_report_aggregates_all_comparable_seed_results_by_mo_mode(tmp_path) -> None:
    campaign = Campaign.from_config(campaign_id="campaign-matrix", config=_build_config())
    child_reports = {
        "campaign-matrix__seed_1__compromise": _child_report(
            campaign_id="campaign-matrix__seed_1__compromise",
            seed=1,
            mode="compromise",
            base_depth=90,
        ),
        "campaign-matrix__seed_2__compromise": _child_report(
            campaign_id="campaign-matrix__seed_2__compromise",
            seed=2,
            mode="compromise",
            base_depth=70,
        ),
        "campaign-matrix__seed_1__best_depth": _child_report(
            campaign_id="campaign-matrix__seed_1__best_depth",
            seed=1,
            mode="best_depth",
            base_depth=60,
        ),
        "campaign-matrix__seed_2__best_depth": _child_report(
            campaign_id="campaign-matrix__seed_2__best_depth",
            seed=2,
            mode="best_depth",
            base_depth=50,
            comparable=False,
        ),
    }

    report = build_matrix_report(
        campaign=campaign,
        child_reports=child_reports,
        child_results=[],
        output_root=tmp_path / "campaigns",
    )

    summaries = {summary.mo_selection_mode: summary for summary in report.mode_summaries}
    assert summaries["compromise"].comparable_completed_cases == 2
    assert summaries["compromise"].aggregate_metrics["trans_depth"].mo_rl_mean == 80.0
    assert summaries["compromise"].aggregate_metrics["trans_depth"].mo_only_mean == 83.0
    assert summaries["best_depth"].comparable_completed_cases == 1
    assert summaries["best_depth"].incomplete_cases == 1
    assert summaries["best_depth"].aggregate_metrics["trans_depth"].mo_rl_mean == 60.0
    assert summaries["best_cnot_count"].comparable_completed_cases == 0
    assert summaries["best_cnot_count"].aggregate_metrics["trans_depth"].mo_rl_mean is None

    markdown = render_matrix_summary_markdown(report)
    assert "Aggregate Comparison by MO Mode" in markdown
    assert "| compromise | 2 | 2 | 0 | 0 | 0 | 90.00 | 85.00 | 83.00 | 80.00 |" in markdown
    assert "| best_depth | 2 | 1 | 0 | 1 | 0 | 70.00 | 65.00 | 63.00 | 60.00 |" in markdown


def test_run_campaign_matrix_continues_after_child_failure_and_writes_outputs(tmp_path) -> None:
    campaign = Campaign.from_config(
        campaign_id="campaign-matrix",
        config=_build_config(seeds=(1,), mo_selection_modes=("compromise", "best_depth")),
    )
    calls: list[str] = []

    def fake_run_campaign(child, *, output_root):
        calls.append(child.campaign_id)
        if child.config.mo_selection_modes == ("best_depth",):
            raise RuntimeError("training crashed")
        return _child_report(
            campaign_id=child.campaign_id,
            seed=child.config.seed,
            mode=child.config.mo_selection_modes[0],
            base_depth=80,
        )

    report = run_campaign_matrix(
        campaign,
        output_root=tmp_path / "campaigns",
        run_campaign_fn=fake_run_campaign,
    )

    assert calls == [
        "campaign-matrix__seed_1__compromise",
        "campaign-matrix__seed_1__best_depth",
    ]
    assert report.status == "failed"
    assert report.child_results[1].error == "training crashed"
    summaries = {summary.mo_selection_mode: summary for summary in report.mode_summaries}
    assert summaries["best_depth"].failed_cases == 1
    assert Path(report.summary_path).exists()
    assert Path(report.structured_output_path).exists()


def test_run_campaign_matrix_uses_grouped_runner_once_per_seed_by_default(monkeypatch, tmp_path) -> None:
    campaign = Campaign.from_config(
        campaign_id="campaign-matrix",
        config=_build_config(seeds=(1,), mo_selection_modes=ALL_MO_SELECTION_MODES),
    )
    grouped_calls: list[list[str]] = []

    def fake_run_seed_group(campaigns, output_root, *, verbose=False):
        del output_root, verbose
        grouped_calls.append([child.campaign_id for child in campaigns])
        return {
            child.campaign_id: _child_report(
                campaign_id=child.campaign_id,
                seed=child.config.seed,
                mode=child.config.mo_selection_modes[0],
                base_depth=80,
            )
            for child in campaigns
        }

    monkeypatch.setattr(
        "src.integration.campaign_matrix._run_seed_group_campaigns",
        fake_run_seed_group,
    )

    report = run_campaign_matrix(campaign, output_root=tmp_path / "campaigns")

    assert grouped_calls == [[
        "campaign-matrix__seed_1__compromise",
        "campaign-matrix__seed_1__best_depth",
        "campaign-matrix__seed_1__best_cnot_count",
    ]]
    assert report.status == "completed"
