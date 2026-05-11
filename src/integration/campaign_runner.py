from __future__ import annotations

from inspect import Parameter, signature
from pathlib import Path
from typing import Callable

import src.qiskit_interface as qiskit_interface

from src.integration.backend_adapter import resolve_backend_bundle as _resolve_backend_bundle
from src.integration.campaign_contracts import Campaign, CampaignCase
from src.integration.campaign_reporting import (
    CampaignCaseReport,
    build_campaign_report,
    build_case_directory_name_map,
    write_campaign_outputs,
)
from src.integration.contracts import LayoutSelectionPolicy, ScenarioRequest
from src.integration.mo_effort import resolve_effective_mo_settings
from src.integration.routing_subgraph import build_path_expanded_subgraph
from src.integration.scenarios import (
    run_baseline_scenario as _run_baseline_scenario,
    run_mo_only_scenario as _run_mo_only_scenario,
    run_mo_rl_scenario as _run_mo_rl_scenario,
)


def _default_train_case(**kwargs):
    from src.integration.training_bridge import train_case

    return train_case(**kwargs)


def _case_library_name(campaign_case: CampaignCase) -> str:
    return f"{campaign_case.circuit_family}_{campaign_case.num_qubits}"


def _build_scenario_request(
    *,
    campaign: Campaign,
    campaign_case: CampaignCase,
    scenario_name: str,
    effective_mo_settings=None,
    rl_model_path: str | None = None,
) -> ScenarioRequest:
    if scenario_name == "Baseline":
        return ScenarioRequest(
            scenario_name=scenario_name,
            circuit_name=_case_library_name(campaign_case),
            num_qubits=campaign_case.num_qubits,
            backend_name=campaign_case.backend_name,
            seed=campaign.config.seed,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            mo_objective_index=0,
            rl_model_path=rl_model_path,
        )

    if effective_mo_settings is None:
        raise ValueError("effective_mo_settings is required for non-baseline scenario requests")

    mo_objective_index = _resolve_mo_objective_index(campaign)

    return ScenarioRequest(
        scenario_name=scenario_name,
        circuit_name=_case_library_name(campaign_case),
        num_qubits=campaign_case.num_qubits,
        backend_name=campaign_case.backend_name,
        seed=campaign.config.seed,
        layout_policy=campaign.config.layout_policy,
        mo_use_quick=effective_mo_settings.mo_use_quick,
        mo_population_size=effective_mo_settings.mo_population_size,
        mo_n_generations=effective_mo_settings.mo_n_generations,
        mo_objective_index=mo_objective_index,
        rl_model_path=rl_model_path,
    )


def _resolve_mo_objective_index(campaign: Campaign) -> int:
    if campaign.config.layout_policy is not LayoutSelectionPolicy.BEST_ON_OBJECTIVE:
        return 0

    from src.mo_module.fitness import get_preset_objectives

    objective_name = campaign.config.mo_objective_name
    objective_names = get_preset_objectives("default")
    if objective_name not in objective_names:
        raise ValueError(f"Unknown mo_objective_name for campaign runner: {objective_name}")
    return objective_names.index(objective_name)


def _build_case_output_dir(
    *,
    output_root: Path | str,
    campaign: Campaign,
    campaign_case: CampaignCase,
    case_dir_names: dict[str, str],
) -> Path:
    case_dir_name = case_dir_names[campaign_case.case_id]
    return Path(output_root) / campaign.campaign_id / "cases" / case_dir_name


def _default_load_case_circuit(campaign: Campaign, campaign_case: CampaignCase):
    return qiskit_interface.load_circuit(
        "library",
        circuit_name=campaign_case.circuit_family,
        num_qubits=campaign_case.num_qubits,
        seed=campaign.config.seed,
    )


def _persist_campaign_state(
    *,
    output_root: Path | str,
    campaign: Campaign,
    case_reports: list[CampaignCaseReport],
    campaign_status: str,
    write_outputs,
    persist_terminal_state: bool,
):
    report = build_campaign_report(
        campaign_id=campaign.campaign_id,
        campaign_status=campaign_status,
        campaign_config=campaign.config,
        case_reports=case_reports,
        total_cases=len(campaign.build_cases()),
    )
    if persist_terminal_state:
        campaign.status = report.campaign_status
        campaign.summary = report.summary
    write_outputs(output_dir=Path(output_root) / campaign.campaign_id, report=report)
    return report


def _runner_accepts_kwarg(run_scenario: Callable[..., object], kwarg_name: str) -> bool:
    try:
        parameters = signature(run_scenario).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        (parameter.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD) and parameter.name == kwarg_name)
        or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _invoke_scenario_runner(run_scenario: Callable[..., object], request: ScenarioRequest, *, circuit, **kwargs):
    call_kwargs = {}
    if _runner_accepts_kwarg(run_scenario, "circuit"):
        call_kwargs["circuit"] = circuit
    for key, value in kwargs.items():
        if _runner_accepts_kwarg(run_scenario, key):
            call_kwargs[key] = value
    if call_kwargs:
        return run_scenario(request, **call_kwargs)
    return run_scenario(request)


def _cancel_remaining_cases(case_reports: list[CampaignCaseReport], remaining_cases: list[CampaignCase]) -> None:
    for campaign_case in remaining_cases:
        case_reports.append(
            CampaignCaseReport(
                case=campaign_case,
                status="cancelled",
                incidents=["Campaign cancelled before this case started."],
            )
        )


def _handle_interruption(
    *,
    case_reports: list[CampaignCaseReport],
    cases: list[CampaignCase],
    case_index: int,
    current_case_report: CampaignCaseReport | None,
    current_case_already_recorded: bool,
) -> tuple[str, list[CampaignCaseReport]]:
    if current_case_report is not None and not current_case_already_recorded:
        current_case_report.status = "cancelled"
        if not current_case_report.incidents:
            current_case_report.incidents.append("Campaign interrupted during case execution.")
        case_reports.append(current_case_report)
        remaining_cases = cases[case_index + 1 :]
    elif current_case_already_recorded:
        remaining_cases = cases[case_index + 1 :]
    else:
        remaining_cases = cases[case_index:]
    _cancel_remaining_cases(case_reports, remaining_cases)
    return "interrupted", case_reports


def run_campaign(
    campaign: Campaign,
    *,
    output_root: Path | str = "campaigns",
    load_case_circuit: Callable[..., object] | None = None,
    run_baseline: Callable[..., object] = _run_baseline_scenario,
    run_mo_only: Callable[..., object] = _run_mo_only_scenario,
    train_case_fn: Callable[..., object] | None = None,
    run_mo_rl: Callable[..., object] = _run_mo_rl_scenario,
    resolve_backend_bundle: Callable[[str], object] = _resolve_backend_bundle,
    write_outputs: Callable[..., object] = write_campaign_outputs,
    cancellation_requested: Callable[[], bool] | None = None,
):
    if load_case_circuit is None:
        load_case_circuit = lambda campaign_case: _default_load_case_circuit(campaign, campaign_case)
    train_case_fn = train_case_fn or _default_train_case
    cancellation_requested = cancellation_requested or (lambda: False)

    campaign.status = "running"
    case_reports: list[CampaignCaseReport] = []
    cases = campaign.build_cases()
    case_dir_names = build_case_directory_name_map([campaign_case.case_id for campaign_case in cases])
    final_status = "completed"

    for case_index, campaign_case in enumerate(cases):
        if cancellation_requested():
            final_status = "cancelled"
            _cancel_remaining_cases(case_reports, cases[case_index:])
            break

        case_output_dir = _build_case_output_dir(
            output_root=output_root,
            campaign=campaign,
            campaign_case=campaign_case,
            case_dir_names=case_dir_names,
        )
        case_report = None
        case_report_recorded = False

        try:
            circuit = load_case_circuit(campaign_case)
            effective_mo_settings = resolve_effective_mo_settings(
                qubit_count=campaign_case.num_qubits,
                mo_effort_mode=campaign.config.mo_effort_mode,
                mo_use_quick=campaign.config.mo_use_quick,
                mo_population_size=campaign.config.mo_population_size,
                mo_n_generations=campaign.config.mo_n_generations,
            )
            baseline_request = _build_scenario_request(
                campaign=campaign,
                campaign_case=campaign_case,
                scenario_name="Baseline",
            )
            mo_only_request = _build_scenario_request(
                campaign=campaign,
                campaign_case=campaign_case,
                scenario_name="MO_Only",
                effective_mo_settings=effective_mo_settings,
            )
            case_report = CampaignCaseReport(case=campaign_case, status="completed")
            case_report.baseline_result = _invoke_scenario_runner(run_baseline, baseline_request, circuit=circuit)
            if not case_report.baseline_result.success:
                case_report.status = "failed"
                case_report.incidents.extend(case_report.baseline_result.errors)
                case_reports.append(case_report)
                case_report_recorded = True
                _persist_campaign_state(
                    output_root=output_root,
                    campaign=campaign,
                    case_reports=case_reports,
                    campaign_status="running",
                    write_outputs=write_outputs,
                    persist_terminal_state=False,
                )
                continue

            case_report.mo_only_result = _invoke_scenario_runner(run_mo_only, mo_only_request, circuit=circuit)
            if not case_report.mo_only_result.success:
                case_report.status = "failed"
                case_report.incidents.extend(case_report.mo_only_result.errors)
                case_reports.append(case_report)
                case_report_recorded = True
                _persist_campaign_state(
                    output_root=output_root,
                    campaign=campaign,
                    case_reports=case_reports,
                    campaign_status="running",
                    write_outputs=write_outputs,
                    persist_terminal_state=False,
                )
                continue

            selected_layout = case_report.mo_only_result.selected_layout
            if selected_layout is None:
                case_report.status = "failed"
                case_report.incidents.append(
                    "MO_Only did not produce a selected layout for Campaign MO+RL training."
                )
                case_reports.append(case_report)
                case_report_recorded = True
                _persist_campaign_state(
                    output_root=output_root,
                    campaign=campaign,
                    case_reports=case_reports,
                    campaign_status="running",
                    write_outputs=write_outputs,
                    persist_terminal_state=False,
                )
                continue

            selected_layout_for_training = list(selected_layout)
            selected_layout_for_mo_rl = list(selected_layout)
            backend_bundle = resolve_backend_bundle(campaign_case.backend_name)
            routing_subgraph = build_path_expanded_subgraph(
                circuit=circuit,
                selected_layout=selected_layout,
                coupling_edges=backend_bundle.coupling_edges,
            )
            coupling_edges_for_training = list(routing_subgraph.coupling_edges)
            coupling_edges_for_mo_rl = list(routing_subgraph.coupling_edges)
            case_report.training_result = train_case_fn(
                campaign_case=campaign_case,
                campaign_config=campaign.config,
                target_circuit=circuit,
                coupling_map=coupling_edges_for_training,
                case_output_dir=case_output_dir,
                initial_layout=selected_layout_for_training,
            )

            if case_report.training_result.status != "completed" or case_report.training_result.selected_artifact_path is None:
                case_report.status = "failed"
                case_report.incidents.append("RL training failed before MO+RL evaluation.")
            else:
                mo_rl_request = _build_scenario_request(
                    campaign=campaign,
                    campaign_case=campaign_case,
                    scenario_name="MO+RL",
                    effective_mo_settings=effective_mo_settings,
                    rl_model_path=str(case_report.training_result.selected_artifact_path),
                )
                case_report.mo_rl_result = _invoke_scenario_runner(
                    run_mo_rl,
                    mo_rl_request,
                    circuit=circuit,
                    injected_layout=selected_layout_for_mo_rl,
                    injected_coupling_edges=coupling_edges_for_mo_rl,
                    injected_routing_graph=routing_subgraph,
                )
                if case_report.mo_rl_result.success:
                    case_report.status = "completed"
                else:
                    case_report.status = "incomplete"
                    case_report.incidents.extend(case_report.mo_rl_result.errors)

            case_reports.append(case_report)
            case_report_recorded = True
            _persist_campaign_state(
                output_root=output_root,
                campaign=campaign,
                case_reports=case_reports,
                campaign_status="running",
                write_outputs=write_outputs,
                persist_terminal_state=False,
            )
        except KeyboardInterrupt:
            final_status, case_reports = _handle_interruption(
                case_reports=case_reports,
                cases=cases,
                case_index=case_index,
                current_case_report=case_report,
                current_case_already_recorded=case_report_recorded,
            )
            break

    try:
        final_report = _persist_campaign_state(
            output_root=output_root,
            campaign=campaign,
            case_reports=case_reports,
            campaign_status=final_status,
            write_outputs=write_outputs,
            persist_terminal_state=True,
        )
    except KeyboardInterrupt:
        final_report = build_campaign_report(
            campaign_id=campaign.campaign_id,
            campaign_status="interrupted",
            campaign_config=campaign.config,
            case_reports=case_reports,
            total_cases=len(cases),
        )
        campaign.status = final_report.campaign_status
        campaign.summary = final_report.summary
    return final_report
