from __future__ import annotations

import json
from inspect import Parameter, signature
from pathlib import Path
from typing import Callable

import src.qiskit_interface as qiskit_interface

from src.integration.backend_adapter import resolve_backend_bundle as _resolve_backend_bundle
from src.integration.campaign_contracts import Campaign, CampaignCase
from src.integration.campaign_reporting import (
    CampaignCaseReport,
    CampaignReport,
    build_campaign_report,
    build_case_directory_name_map,
    write_campaign_outputs,
)
from src.integration.contracts import LayoutSelectionPolicy, ScenarioRequest
from src.integration.hybrid_layout_probe import (
    HYBRID_LAYOUT_PROBE_FILENAME,
    select_hybrid_probe_layout as _select_hybrid_probe_layout,
)
from src.integration.mo_effort import resolve_effective_mo_settings
from src.integration.rl_guided_mo import (
    RL_GUIDED_MO_FILENAME,
    optimize_rl_guided_layouts as _optimize_rl_guided_layouts,
)
from src.integration.routing_subgraph import build_path_expanded_subgraph
from src.integration.scenarios import (
    _resolve_qiskit_initial_layout_for_rl_only,
    optimize_mo_layouts as _optimize_mo_layouts,
    run_baseline_scenario as _run_baseline_scenario,
    run_mo_only_scenario as _run_mo_only_scenario,
    run_mo_rl_scenario as _run_mo_rl_scenario,
    run_rl_only_scenario as _run_rl_only_scenario,
    select_mo_layout as _select_mo_layout,
)
from src.rl_module.routing_mask import RoutingMaskConfig


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
    initial_layout: list[int] | None = None,
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
            synthetic_topology=campaign.config.synthetic_topology,
        )

    if scenario_name == "RL_Only":
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
            initial_layout=initial_layout,
            rl_model_path=rl_model_path,
            synthetic_topology=campaign.config.synthetic_topology,
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
        synthetic_topology=campaign.config.synthetic_topology,
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


def _validate_qiskit_initial_layout(layout, *, num_qubits: int) -> list[int]:
    normalized_layout = [int(entry) for entry in layout]
    if len(normalized_layout) != num_qubits:
        raise ValueError("Baseline qiskit_initial_layout length must match num_qubits")
    if any(entry < 0 for entry in normalized_layout):
        raise ValueError("Baseline qiskit_initial_layout cannot contain negative entries")
    if len(set(normalized_layout)) != len(normalized_layout):
        raise ValueError("Baseline qiskit_initial_layout contains duplicated physical qubits")
    return normalized_layout


def _extract_qiskit_initial_layout_from_baseline(baseline_result, *, num_qubits: int) -> list[int]:
    candidates = []
    artifact = baseline_result.transpilation_artifact or {}
    if isinstance(artifact, dict):
        transpilation = artifact.get("transpilation")
        if isinstance(transpilation, dict):
            candidates.extend(
                [
                    transpilation.get("qiskit_initial_layout"),
                    transpilation.get("final_layout"),
                ]
            )
    metrics = baseline_result.transpilation_metrics or {}
    if isinstance(metrics, dict):
        candidates.append(metrics.get("qiskit_initial_layout"))

    for candidate in candidates:
        if candidate is None:
            continue
        return _validate_qiskit_initial_layout(candidate, num_qubits=num_qubits)

    return list(range(num_qubits))


def _resolve_rl_only_qiskit_initial_layout(
    request: ScenarioRequest,
    circuit,
    baseline_result,
    *,
    num_qubits: int,
) -> list[int]:
    artifact = baseline_result.transpilation_artifact or {}
    if isinstance(artifact, dict):
        transpilation = artifact.get("transpilation")
        if isinstance(transpilation, dict) and transpilation.get("initial_layout") is not None:
            layout = _resolve_qiskit_initial_layout_for_rl_only(request, circuit)
            return _validate_qiskit_initial_layout(layout, num_qubits=num_qubits)
    return _extract_qiskit_initial_layout_from_baseline(
        baseline_result,
        num_qubits=num_qubits,
    )


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


def _invoke_train_case(train_case_fn: Callable[..., object], **kwargs):
    call_kwargs = dict(kwargs)
    for optional_kwarg in (
        "verbose",
        "backend_bundle",
        "initial_model_path",
        "total_timesteps_override",
    ):
        if not _runner_accepts_kwarg(train_case_fn, optional_kwarg):
            call_kwargs.pop(optional_kwarg, None)
    return train_case_fn(**call_kwargs)


def _is_hybrid_probe_campaign(campaign: Campaign) -> bool:
    return campaign.config.mo_selection_modes == ("hybrid_probe",)


def _is_rl_guided_campaign(campaign: Campaign) -> bool:
    return campaign.config.mo_selection_modes == ("rl_guided",)


def _build_hybrid_probe_mask_config(campaign: Campaign) -> RoutingMaskConfig:
    return RoutingMaskConfig(
        cycle_window=campaign.config.rl_cycle_window,
        stagnation_patience=campaign.config.rl_stagnation_patience,
        sabre_top_k=campaign.config.rl_sabre_top_k,
    )


def _hybrid_probe_payload(probe_result) -> dict:
    if hasattr(probe_result, "to_dict"):
        return probe_result.to_dict()
    if isinstance(probe_result, dict):
        return dict(probe_result)
    raise ValueError("hybrid_probe selector must return a mapping or expose to_dict()")


def _run_hybrid_probe(
    select_hybrid_probe: Callable[..., object],
    *,
    campaign: Campaign,
    circuit,
    mo_result,
    backend_bundle,
    qiskit_initial_layout: list[int],
    artifact_path: Path,
):
    return select_hybrid_probe(
        circuit=circuit,
        mo_result=mo_result,
        backend_bundle=backend_bundle,
        qiskit_initial_layout=list(qiskit_initial_layout),
        seed=campaign.config.seed,
        frontier_mode=campaign.config.rl_frontier_mode,
        max_steps=campaign.config.rl_max_steps,
        lookahead_window=campaign.config.rl_lookahead_window,
        routing_mask_config=_build_hybrid_probe_mask_config(campaign),
        artifact_path=artifact_path,
    )


def _run_rl_guided_mo(
    optimize_rl_guided: Callable[..., object],
    *,
    campaign: Campaign,
    circuit,
    backend_bundle,
    coupling_edges,
    rl_only_model_path: Path | str,
    qiskit_initial_layout: list[int],
    mo_only_layout: list[int],
    effective_mo_settings,
    artifact_path: Path,
    verbose: bool = False,
):
    kwargs = dict(
        circuit=circuit,
        backend_bundle=backend_bundle,
        coupling_edges=list(coupling_edges),
        model_path=rl_only_model_path,
        qiskit_initial_layout=list(qiskit_initial_layout),
        mo_only_layout=list(mo_only_layout),
        population_size=effective_mo_settings.mo_population_size,
        n_generations=effective_mo_settings.mo_n_generations,
        seed=campaign.config.seed,
        artifact_path=artifact_path,
    )
    if _runner_accepts_kwarg(optimize_rl_guided, "verbose"):
        kwargs["verbose"] = verbose
    return optimize_rl_guided(**kwargs)


def _resolve_case_backend_bundle(resolve_backend_bundle: Callable[..., object], campaign: Campaign, backend_name: str):
    if campaign.config.synthetic_topology is not None and _runner_accepts_kwarg(resolve_backend_bundle, "synthetic_topology"):
        return resolve_backend_bundle(backend_name, synthetic_topology=campaign.config.synthetic_topology)
    return resolve_backend_bundle(backend_name)


def _shared_seed_output_dir(output_root: Path | str, campaigns: tuple[Campaign, ...]) -> Path:
    child_prefix, separator, _ = campaigns[0].campaign_id.rpartition("__")
    if not separator:
        child_prefix = campaigns[0].campaign_id
    return Path(output_root) / f"{child_prefix}__shared"


def _shared_layout_output_dir(shared_case_output_dir: Path, layout: tuple[int, ...]) -> Path:
    layout_id = "_".join(str(entry) for entry in layout)
    return shared_case_output_dir / "layouts" / f"layout_{layout_id}"


def _write_mo_front_artifact(path: Path, mo_result) -> None:
    pareto_fitness = getattr(mo_result, "pareto_fitness", None)
    payload = {
        "artifact_version": "mo_front.v1",
        "pareto_layouts": [
            [int(entry) for entry in layout]
            for layout in getattr(mo_result, "pareto_layouts", [])
        ],
        "pareto_fitness": pareto_fitness.tolist() if pareto_fitness is not None else None,
        "objective_names": list(getattr(mo_result, "objective_names", [])),
        "metadata": mo_result.to_dict() if hasattr(mo_result, "to_dict") else {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_seed_group(campaigns: tuple[Campaign, ...]) -> None:
    if not campaigns:
        raise ValueError("Campaign seed group cannot be empty")
    seed = campaigns[0].config.seed
    expected_cases = campaigns[0].build_cases()
    seen_modes: set[str] = set()
    for campaign in campaigns:
        if campaign.config.seed != seed:
            raise ValueError("Campaign seed group must contain a single seed")
        if campaign.build_cases() != expected_cases:
            raise ValueError("Campaign seed group must contain the same cases for every MO mode")
        if len(campaign.config.mo_selection_modes) != 1:
            raise ValueError("Campaign seed group children must contain exactly one MO selection mode")
        mode = campaign.config.mo_selection_modes[0]
        if mode in seen_modes:
            raise ValueError("Campaign seed group cannot contain duplicate MO selection modes")
        seen_modes.add(mode)


def _persist_seed_group_state(
    *,
    output_root: Path | str,
    campaigns: tuple[Campaign, ...],
    case_reports: dict[str, list[CampaignCaseReport]],
    campaign_status: str,
    write_outputs,
    persist_terminal_state: bool,
) -> dict[str, CampaignReport]:
    return {
        campaign.campaign_id: _persist_campaign_state(
            output_root=output_root,
            campaign=campaign,
            case_reports=case_reports[campaign.campaign_id],
            campaign_status=campaign_status,
            write_outputs=write_outputs,
            persist_terminal_state=persist_terminal_state,
        )
        for campaign in campaigns
    }


def _copy_common_case_report(
    *,
    campaign_case: CampaignCase,
    status: str,
    baseline_result,
    rl_only_training_result,
    rl_only_result,
    incidents: list[str],
    mo_only_result=None,
    training_result=None,
    mo_rl_result=None,
    hybrid_layout_probe=None,
    rl_guided_mo=None,
) -> CampaignCaseReport:
    return CampaignCaseReport(
        case=campaign_case,
        status=status,
        baseline_result=baseline_result,
        mo_only_result=mo_only_result,
        rl_only_result=rl_only_result,
        mo_rl_result=mo_rl_result,
        rl_only_training_result=rl_only_training_result,
        training_result=training_result,
        hybrid_layout_probe=hybrid_layout_probe,
        rl_guided_mo=rl_guided_mo,
        incidents=list(incidents),
    )


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
    optimize_mo: Callable[..., object] = _optimize_mo_layouts,
    select_hybrid_probe: Callable[..., object] = _select_hybrid_probe_layout,
    optimize_rl_guided: Callable[..., object] = _optimize_rl_guided_layouts,
    train_case_fn: Callable[..., object] | None = None,
    run_rl_only: Callable[..., object] = _run_rl_only_scenario,
    run_mo_rl: Callable[..., object] = _run_mo_rl_scenario,
    resolve_backend_bundle: Callable[[str], object] = _resolve_backend_bundle,
    write_outputs: Callable[..., object] = write_campaign_outputs,
    cancellation_requested: Callable[[], bool] | None = None,
    verbose: bool = False,
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

            qiskit_initial_layout = _resolve_rl_only_qiskit_initial_layout(
                baseline_request,
                circuit,
                case_report.baseline_result,
                num_qubits=campaign_case.num_qubits,
            )
            backend_bundle = _resolve_case_backend_bundle(resolve_backend_bundle, campaign, campaign_case.backend_name)
            qiskit_routing_subgraph = None
            if _is_rl_guided_campaign(campaign):
                qiskit_coupling_edges_for_training = list(backend_bundle.coupling_edges)
                qiskit_coupling_edges_for_rl_only = list(backend_bundle.coupling_edges)
            else:
                qiskit_routing_subgraph = build_path_expanded_subgraph(
                    circuit=circuit,
                    selected_layout=qiskit_initial_layout,
                    coupling_edges=backend_bundle.coupling_edges,
                )
                qiskit_coupling_edges_for_training = list(qiskit_routing_subgraph.coupling_edges)
                qiskit_coupling_edges_for_rl_only = list(qiskit_routing_subgraph.coupling_edges)
            case_report.rl_only_training_result = _invoke_train_case(
                train_case_fn,
                campaign_case=campaign_case,
                campaign_config=campaign.config,
                target_circuit=circuit,
                coupling_map=qiskit_coupling_edges_for_training,
                case_output_dir=case_output_dir / "rl_only",
                initial_layout=list(qiskit_initial_layout),
                backend_bundle=backend_bundle,
                verbose=verbose,
            )

            if (
                case_report.rl_only_training_result.status != "completed"
                or case_report.rl_only_training_result.selected_artifact_path is None
            ):
                case_report.status = "incomplete"
                case_report.incidents.append("RL training failed before RL_Only evaluation.")
            else:
                rl_only_request = _build_scenario_request(
                    campaign=campaign,
                    campaign_case=campaign_case,
                    scenario_name="RL_Only",
                    initial_layout=list(qiskit_initial_layout),
                    rl_model_path=str(case_report.rl_only_training_result.selected_artifact_path),
                )
                case_report.rl_only_result = _invoke_scenario_runner(
                    run_rl_only,
                    rl_only_request,
                    circuit=circuit,
                    injected_layout=list(qiskit_initial_layout),
                    injected_coupling_edges=qiskit_coupling_edges_for_rl_only,
                    injected_routing_graph=qiskit_routing_subgraph,
                )
                if not case_report.rl_only_result.success:
                    case_report.status = "incomplete"
                    case_report.incidents.extend(case_report.rl_only_result.errors)

            if _is_hybrid_probe_campaign(campaign):
                mo_result = optimize_mo(
                    mo_only_request,
                    circuit=circuit,
                    backend_bundle=backend_bundle,
                )
                probe_result = _run_hybrid_probe(
                    select_hybrid_probe,
                    campaign=campaign,
                    circuit=circuit,
                    mo_result=mo_result,
                    backend_bundle=backend_bundle,
                    qiskit_initial_layout=qiskit_initial_layout,
                    artifact_path=case_output_dir / HYBRID_LAYOUT_PROBE_FILENAME,
                )
                case_report.hybrid_layout_probe = _hybrid_probe_payload(probe_result)
                case_report.mo_only_result = _invoke_scenario_runner(
                    run_mo_only,
                    mo_only_request,
                    circuit=circuit,
                    injected_layout=list(probe_result.selected_layout),
                )
            else:
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

            rl_guided_result = None
            if _is_rl_guided_campaign(campaign):
                if (
                    case_report.rl_only_training_result.status != "completed"
                    or case_report.rl_only_training_result.selected_artifact_path is None
                ):
                    case_report.status = "failed"
                    case_report.incidents.append(
                        "RL_Only training artifact is required before rl_guided MO."
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
                try:
                    rl_guided_result = _run_rl_guided_mo(
                        optimize_rl_guided,
                        campaign=campaign,
                        circuit=circuit,
                        backend_bundle=backend_bundle,
                        coupling_edges=backend_bundle.coupling_edges,
                        rl_only_model_path=case_report.rl_only_training_result.selected_artifact_path,
                        qiskit_initial_layout=qiskit_initial_layout,
                        mo_only_layout=selected_layout,
                        effective_mo_settings=effective_mo_settings,
                        artifact_path=case_output_dir / RL_GUIDED_MO_FILENAME,
                        verbose=verbose,
                    )
                except Exception as exc:
                    case_report.status = "failed"
                    case_report.incidents.append(f"RL-guided MO failed: {exc}")
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
                case_report.rl_guided_mo = rl_guided_result.to_dict()
                selected_layout = list(rl_guided_result.selected_layout)

            selected_layout_for_training = list(selected_layout)
            selected_layout_for_mo_rl = list(selected_layout)
            routing_subgraph = None
            if _is_rl_guided_campaign(campaign):
                coupling_edges_for_training = list(backend_bundle.coupling_edges)
                coupling_edges_for_mo_rl = list(backend_bundle.coupling_edges)
            else:
                routing_subgraph = build_path_expanded_subgraph(
                    circuit=circuit,
                    selected_layout=selected_layout,
                    coupling_edges=backend_bundle.coupling_edges,
                )
                coupling_edges_for_training = list(routing_subgraph.coupling_edges)
                coupling_edges_for_mo_rl = list(routing_subgraph.coupling_edges)
            case_report.training_result = _invoke_train_case(
                train_case_fn,
                campaign_case=campaign_case,
                campaign_config=campaign.config,
                target_circuit=circuit,
                coupling_map=coupling_edges_for_training,
                case_output_dir=case_output_dir,
                initial_layout=selected_layout_for_training,
                initial_model_path=(
                    case_report.rl_only_training_result.selected_artifact_path
                    if _is_rl_guided_campaign(campaign)
                    else None
                ),
                total_timesteps_override=(
                    campaign.config.rl_finetune_timesteps
                    if _is_rl_guided_campaign(campaign)
                    else None
                ),
                backend_bundle=backend_bundle,
                verbose=verbose,
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
                if case_report.mo_rl_result.success and case_report.status != "incomplete":
                    case_report.status = "completed"
                elif not case_report.mo_rl_result.success:
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


def run_campaign_seed_group(
    campaigns: tuple[Campaign, ...] | list[Campaign],
    *,
    output_root: Path | str = "campaigns",
    load_case_circuit: Callable[..., object] | None = None,
    run_baseline: Callable[..., object] = _run_baseline_scenario,
    optimize_mo: Callable[..., object] = _optimize_mo_layouts,
    optimize_rl_guided: Callable[..., object] = _optimize_rl_guided_layouts,
    select_mo_layout: Callable[..., list[int]] = _select_mo_layout,
    select_hybrid_probe: Callable[..., object] = _select_hybrid_probe_layout,
    run_mo_only: Callable[..., object] = _run_mo_only_scenario,
    train_case_fn: Callable[..., object] | None = None,
    run_rl_only: Callable[..., object] = _run_rl_only_scenario,
    run_mo_rl: Callable[..., object] = _run_mo_rl_scenario,
    resolve_backend_bundle: Callable[[str], object] = _resolve_backend_bundle,
    write_outputs: Callable[..., object] = write_campaign_outputs,
    verbose: bool = False,
) -> dict[str, CampaignReport]:
    normalized_campaigns = tuple(campaigns)
    _validate_seed_group(normalized_campaigns)
    if len(normalized_campaigns) == 1 and _is_rl_guided_campaign(normalized_campaigns[0]):
        campaign = normalized_campaigns[0]
        return {
            campaign.campaign_id: run_campaign(
                campaign,
                output_root=output_root,
                load_case_circuit=load_case_circuit,
                run_baseline=run_baseline,
                run_mo_only=run_mo_only,
                optimize_mo=optimize_mo,
                select_hybrid_probe=select_hybrid_probe,
                optimize_rl_guided=optimize_rl_guided,
                train_case_fn=train_case_fn,
                run_rl_only=run_rl_only,
                run_mo_rl=run_mo_rl,
                resolve_backend_bundle=resolve_backend_bundle,
                write_outputs=write_outputs,
                verbose=verbose,
            )
        }
    anchor_campaign = normalized_campaigns[0]
    if load_case_circuit is None:
        load_case_circuit = lambda campaign_case: _default_load_case_circuit(anchor_campaign, campaign_case)
    train_case_fn = train_case_fn or _default_train_case

    for campaign in normalized_campaigns:
        campaign.status = "running"

    cases = anchor_campaign.build_cases()
    shared_output_dir = _shared_seed_output_dir(output_root, normalized_campaigns)
    shared_case_dir_names = build_case_directory_name_map([campaign_case.case_id for campaign_case in cases])
    child_case_reports: dict[str, list[CampaignCaseReport]] = {
        campaign.campaign_id: []
        for campaign in normalized_campaigns
    }

    for campaign_case in cases:
        baseline_result = None
        rl_only_training_result = None
        rl_only_result = None
        common_status = "completed"
        common_incidents: list[str] = []
        shared_case_output_dir = shared_output_dir / "cases" / shared_case_dir_names[campaign_case.case_id]

        try:
            circuit = load_case_circuit(campaign_case)
            effective_mo_settings = resolve_effective_mo_settings(
                qubit_count=campaign_case.num_qubits,
                mo_effort_mode=anchor_campaign.config.mo_effort_mode,
                mo_use_quick=anchor_campaign.config.mo_use_quick,
                mo_population_size=anchor_campaign.config.mo_population_size,
                mo_n_generations=anchor_campaign.config.mo_n_generations,
            )
            baseline_request = _build_scenario_request(
                campaign=anchor_campaign,
                campaign_case=campaign_case,
                scenario_name="Baseline",
            )
            baseline_result = _invoke_scenario_runner(run_baseline, baseline_request, circuit=circuit)
            if not baseline_result.success:
                common_status = "failed"
                common_incidents.extend(baseline_result.errors)
                raise ValueError("Shared Baseline scenario failed.")

            qiskit_initial_layout = _resolve_rl_only_qiskit_initial_layout(
                baseline_request,
                circuit,
                baseline_result,
                num_qubits=campaign_case.num_qubits,
            )
            backend_bundle = _resolve_case_backend_bundle(
                resolve_backend_bundle,
                anchor_campaign,
                campaign_case.backend_name,
            )
            qiskit_routing_subgraph = build_path_expanded_subgraph(
                circuit=circuit,
                selected_layout=qiskit_initial_layout,
                coupling_edges=backend_bundle.coupling_edges,
            )
            qiskit_coupling_edges = list(qiskit_routing_subgraph.coupling_edges)
            rl_only_training_result = _invoke_train_case(
                train_case_fn,
                campaign_case=campaign_case,
                campaign_config=anchor_campaign.config,
                target_circuit=circuit,
                coupling_map=qiskit_coupling_edges,
                case_output_dir=shared_case_output_dir / "rl_only",
                initial_layout=list(qiskit_initial_layout),
                backend_bundle=backend_bundle,
                verbose=verbose,
            )
            if (
                rl_only_training_result.status != "completed"
                or rl_only_training_result.selected_artifact_path is None
            ):
                common_status = "incomplete"
                common_incidents.append("RL training failed before RL_Only evaluation.")
            else:
                rl_only_request = _build_scenario_request(
                    campaign=anchor_campaign,
                    campaign_case=campaign_case,
                    scenario_name="RL_Only",
                    initial_layout=list(qiskit_initial_layout),
                    rl_model_path=str(rl_only_training_result.selected_artifact_path),
                )
                rl_only_result = _invoke_scenario_runner(
                    run_rl_only,
                    rl_only_request,
                    circuit=circuit,
                    injected_layout=list(qiskit_initial_layout),
                    injected_coupling_edges=qiskit_coupling_edges,
                    injected_routing_graph=qiskit_routing_subgraph,
                )
                if not rl_only_result.success:
                    common_status = "incomplete"
                    common_incidents.extend(rl_only_result.errors)

            anchor_mo_request = _build_scenario_request(
                campaign=anchor_campaign,
                campaign_case=campaign_case,
                scenario_name="MO_Only",
                effective_mo_settings=effective_mo_settings,
            )
            mo_result = optimize_mo(
                anchor_mo_request,
                circuit=circuit,
                backend_bundle=backend_bundle,
            )
            _write_mo_front_artifact(shared_case_output_dir / "mo_front.json", mo_result)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if not common_incidents or str(exc) not in common_incidents:
                common_incidents.append(f"Shared seed phase failed: {exc}")
            for campaign in normalized_campaigns:
                child_case_reports[campaign.campaign_id].append(
                    _copy_common_case_report(
                        campaign_case=campaign_case,
                        status="failed",
                        baseline_result=baseline_result,
                        rl_only_training_result=rl_only_training_result,
                        rl_only_result=rl_only_result,
                        incidents=common_incidents,
                    )
                )
            _persist_seed_group_state(
                output_root=output_root,
                campaigns=normalized_campaigns,
                case_reports=child_case_reports,
                campaign_status="running",
                write_outputs=write_outputs,
                persist_terminal_state=False,
            )
            continue

        selected_layouts: dict[str, tuple[ScenarioRequest, tuple[int, ...]]] = {}
        hybrid_probe_results: dict[str, dict] = {}
        for campaign in normalized_campaigns:
            mo_only_request = _build_scenario_request(
                campaign=campaign,
                campaign_case=campaign_case,
                scenario_name="MO_Only",
                effective_mo_settings=effective_mo_settings,
            )
            try:
                if _is_hybrid_probe_campaign(campaign):
                    probe_result = _run_hybrid_probe(
                        select_hybrid_probe,
                        campaign=campaign,
                        circuit=circuit,
                        mo_result=mo_result,
                        backend_bundle=backend_bundle,
                        qiskit_initial_layout=qiskit_initial_layout,
                        artifact_path=shared_case_output_dir / HYBRID_LAYOUT_PROBE_FILENAME,
                    )
                    hybrid_probe_results[campaign.campaign_id] = _hybrid_probe_payload(probe_result)
                    selected_layout = probe_result.selected_layout
                else:
                    selected_layout = select_mo_layout(
                        mo_only_request,
                        mo_result,
                        circuit=circuit,
                        backend_bundle=backend_bundle,
                    )
            except Exception as exc:
                child_case_reports[campaign.campaign_id].append(
                    _copy_common_case_report(
                        campaign_case=campaign_case,
                        status="failed",
                        baseline_result=baseline_result,
                        rl_only_training_result=rl_only_training_result,
                        rl_only_result=rl_only_result,
                        incidents=common_incidents + [f"MO layout selection failed: {exc}"],
                    )
                )
                continue
            selected_layouts[campaign.campaign_id] = (mo_only_request, tuple(selected_layout))

        layout_groups: dict[tuple[int, ...], list[Campaign]] = {}
        for campaign in normalized_campaigns:
            selection = selected_layouts.get(campaign.campaign_id)
            if selection is None:
                continue
            layout_groups.setdefault(selection[1], []).append(campaign)

        for selected_layout_tuple, layout_campaigns in layout_groups.items():
            selected_layout = list(selected_layout_tuple)
            representative = layout_campaigns[0]
            mo_only_request = selected_layouts[representative.campaign_id][0]
            branch_incidents = list(common_incidents)
            mo_only_result = None
            training_result = None
            mo_rl_result = None
            branch_status = common_status
            try:
                mo_only_result = _invoke_scenario_runner(
                    run_mo_only,
                    mo_only_request,
                    circuit=circuit,
                    injected_layout=list(selected_layout),
                )
                if not mo_only_result.success:
                    branch_status = "failed"
                    branch_incidents.extend(mo_only_result.errors)
                else:
                    routing_subgraph = build_path_expanded_subgraph(
                        circuit=circuit,
                        selected_layout=selected_layout,
                        coupling_edges=backend_bundle.coupling_edges,
                    )
                    coupling_edges = list(routing_subgraph.coupling_edges)
                    training_result = _invoke_train_case(
                        train_case_fn,
                        campaign_case=campaign_case,
                        campaign_config=representative.config,
                        target_circuit=circuit,
                        coupling_map=coupling_edges,
                        case_output_dir=_shared_layout_output_dir(
                            shared_case_output_dir,
                            selected_layout_tuple,
                        ),
                        initial_layout=list(selected_layout),
                        backend_bundle=backend_bundle,
                        verbose=verbose,
                    )
                    if training_result.status != "completed" or training_result.selected_artifact_path is None:
                        branch_status = "failed"
                        branch_incidents.append("RL training failed before MO+RL evaluation.")
                    else:
                        mo_rl_request = _build_scenario_request(
                            campaign=representative,
                            campaign_case=campaign_case,
                            scenario_name="MO+RL",
                            effective_mo_settings=effective_mo_settings,
                            rl_model_path=str(training_result.selected_artifact_path),
                        )
                        mo_rl_result = _invoke_scenario_runner(
                            run_mo_rl,
                            mo_rl_request,
                            circuit=circuit,
                            injected_layout=list(selected_layout),
                            injected_coupling_edges=coupling_edges,
                            injected_routing_graph=routing_subgraph,
                        )
                        if not mo_rl_result.success:
                            branch_status = "incomplete"
                            branch_incidents.extend(mo_rl_result.errors)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                branch_status = "failed"
                branch_incidents.append(f"Shared layout branch failed: {exc}")

            for campaign in layout_campaigns:
                child_case_reports[campaign.campaign_id].append(
                    _copy_common_case_report(
                        campaign_case=campaign_case,
                        status=branch_status,
                        baseline_result=baseline_result,
                        rl_only_training_result=rl_only_training_result,
                        rl_only_result=rl_only_result,
                        mo_only_result=mo_only_result,
                        training_result=training_result,
                        mo_rl_result=mo_rl_result,
                        hybrid_layout_probe=hybrid_probe_results.get(campaign.campaign_id),
                        incidents=branch_incidents,
                    )
                )

        _persist_seed_group_state(
            output_root=output_root,
            campaigns=normalized_campaigns,
            case_reports=child_case_reports,
            campaign_status="running",
            write_outputs=write_outputs,
            persist_terminal_state=False,
        )

    return _persist_seed_group_state(
        output_root=output_root,
        campaigns=normalized_campaigns,
        case_reports=child_case_reports,
        campaign_status="completed",
        write_outputs=write_outputs,
        persist_terminal_state=True,
    )
