from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from src.integration.campaign_contracts import CampaignCase, CampaignCaseResult, CampaignConfig, CampaignSummary
from src.integration.contracts import ScenarioResult
from src.integration.mo_effort import build_auto_mo_effort_preview
from src.integration.training_bridge import TrainingBridgeResult


_MAIN_METRICS = (
    "trans_depth",
    "trans_two_qubit_gates",
    "trans_cnot_equivalent",
    "elapsed_time_s",
)
_ALLOWED_CASE_STATUSES = frozenset({"completed", "failed", "incomplete", "cancelled"})
_ALLOWED_CAMPAIGN_STATUSES = frozenset({"running", "completed", "failed", "cancelled", "interrupted"})
_WINDOWS_RESERVED_CASE_DIR_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}
)


@dataclass(frozen=True, slots=True)
class AggregateMetricSummary:
    baseline_mean: float | None
    mo_only_mean: float | None
    rl_only_mean: float | None
    mo_rl_mean: float | None


@dataclass(slots=True)
class CampaignCaseReport:
    case: CampaignCase
    status: str
    baseline_result: ScenarioResult | None = None
    mo_only_result: ScenarioResult | None = None
    rl_only_result: ScenarioResult | None = None
    mo_rl_result: ScenarioResult | None = None
    rl_only_training_result: TrainingBridgeResult | None = None
    training_result: TrainingBridgeResult | None = None
    incidents: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_CASE_STATUSES:
            raise ValueError("CampaignCaseReport status must be one of completed, failed, incomplete, cancelled")
        self.incidents = list(self.incidents)


@dataclass(frozen=True, slots=True)
class CampaignReport:
    campaign_id: str
    campaign_status: str
    campaign_config: CampaignConfig
    summary: CampaignSummary
    case_reports: tuple[CampaignCaseReport, ...]
    aggregate_metrics: dict[str, AggregateMetricSummary]
    incidents: list[str]


@dataclass(frozen=True, slots=True)
class CampaignOutputPaths:
    summary_markdown_path: Path
    campaign_json_path: Path
    case_result_paths: dict[str, Path]


def build_case_directory_name_map(case_ids: list[str] | tuple[str, ...]) -> dict[str, str]:
    used_names: set[str] = set()
    return {case_id: _sanitize_case_directory_name(case_id, used_names) for case_id in case_ids}


def _extract_metric(result: ScenarioResult | None, metric_name: str) -> float | None:
    if result is None or not result.success or result.transpilation_metrics is None:
        return None
    value = result.transpilation_metrics.get(metric_name)
    if value is None:
        return None
    return float(value)


def _is_comparable(case_report: CampaignCaseReport) -> bool:
    if case_report.status != "completed":
        return False
    for metric_name in _MAIN_METRICS:
        if _extract_metric(case_report.baseline_result, metric_name) is None:
            return False
        if _extract_metric(case_report.mo_only_result, metric_name) is None:
            return False
        if _extract_metric(case_report.rl_only_result, metric_name) is None:
            return False
        if _extract_metric(case_report.mo_rl_result, metric_name) is None:
            return False
    return True


def _mean(values: list[float]) -> float:
    if not values:
        return None
    return sum(values) / len(values)


def _build_aggregate_metrics(case_reports: list[CampaignCaseReport]) -> dict[str, AggregateMetricSummary]:
    comparable_reports = [case_report for case_report in case_reports if _is_comparable(case_report)]
    aggregate_metrics: dict[str, AggregateMetricSummary] = {}
    for metric_name in _MAIN_METRICS:
        aggregate_metrics[metric_name] = AggregateMetricSummary(
            baseline_mean=_mean([
                _extract_metric(case_report.baseline_result, metric_name) for case_report in comparable_reports
            ]),
            mo_only_mean=_mean([
                _extract_metric(case_report.mo_only_result, metric_name) for case_report in comparable_reports
            ]),
            rl_only_mean=_mean([
                _extract_metric(case_report.rl_only_result, metric_name) for case_report in comparable_reports
            ]),
            mo_rl_mean=_mean([
                _extract_metric(case_report.mo_rl_result, metric_name) for case_report in comparable_reports
            ]),
        )
    return aggregate_metrics


def _build_incidents(case_reports: list[CampaignCaseReport]) -> list[str]:
    incidents: list[str] = []
    for case_report in case_reports:
        if case_report.status == "completed" and not _is_comparable(case_report):
            incidents.append(
                f"{case_report.case.case_id}: completed without a comparable metric bundle across Baseline, MO_Only, RL_Only, and MO+RL."
            )
        for incident in case_report.incidents:
            incidents.append(f"{case_report.case.case_id}: {incident}")
    return incidents


def _summary_case_status(case_report: CampaignCaseReport) -> str:
    if case_report.status == "completed" and not _is_comparable(case_report):
        return "incomplete"
    return case_report.status


def build_campaign_report(
    *,
    campaign_id: str,
    campaign_status: str,
    campaign_config: CampaignConfig,
    case_reports: list[CampaignCaseReport],
    total_cases: int | None = None,
) -> CampaignReport:
    if campaign_status not in _ALLOWED_CAMPAIGN_STATUSES:
        raise ValueError("campaign_status must be one of running, completed, failed, cancelled, interrupted")

    if total_cases is None:
        total_cases = len(case_reports)

    comparable_completed_cases = sum(1 for case_report in case_reports if _is_comparable(case_report))
    completed_non_comparable_cases = sum(
        1 for case_report in case_reports if case_report.status == "completed" and not _is_comparable(case_report)
    )
    failed_cases = sum(1 for case_report in case_reports if case_report.status == "failed")
    incomplete_cases = sum(1 for case_report in case_reports if case_report.status == "incomplete")
    incomplete_cases += completed_non_comparable_cases
    cancelled_cases = sum(1 for case_report in case_reports if case_report.status == "cancelled")
    summary = CampaignSummary(
        status=campaign_status,
        total_cases=total_cases,
        comparable_completed_cases=comparable_completed_cases,
        failed_cases=failed_cases,
        incomplete_cases=incomplete_cases,
        cancelled_cases=cancelled_cases,
        case_results=[
            CampaignCaseResult(case_id=case_report.case.case_id, status=_summary_case_status(case_report))
            for case_report in case_reports
        ],
    )
    return CampaignReport(
        campaign_id=campaign_id,
        campaign_status=campaign_status,
        campaign_config=campaign_config,
        summary=summary,
        case_reports=tuple(case_reports),
        aggregate_metrics=_build_aggregate_metrics(case_reports),
        incidents=_build_incidents(case_reports),
    )


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _render_config(report: CampaignReport) -> list[str]:
    config = report.campaign_config
    status_label = "Final Campaign Status" if report.campaign_status != "running" else "Campaign Status"
    lines = [
        "## Campaign Metadata",
        f"Campaign ID: `{report.campaign_id}`",
        f"Campaign Mode: `{config.mode}`",
        f"{status_label}: `{report.campaign_status}`",
        "",
        "## Global Configuration",
        f"RL Algorithm: `{config.rl_algorithm}`",
        f"RL Timesteps: `{config.rl_total_timesteps}`",
        f"RL Frontier Mode: `{config.rl_frontier_mode}`",
        f"RL Lookahead Window: `{config.rl_lookahead_window}`",
        f"RL Max Steps: `{config.rl_max_steps}`",
        f"RL Learning Rate: `{config.rl_learning_rate}`",
        f"RL Clip Range: `{config.rl_clip_range}`",
        f"RL Target KL: `{config.rl_target_kl}`",
        f"RL Eval Episodes: `{config.rl_n_eval_episodes}`",
        f"RL Cycle Window: `{config.rl_cycle_window}`",
        f"RL Stagnation Patience: `{config.rl_stagnation_patience}`",
        f"RL SABRE Top-k: `{config.rl_sabre_top_k}`",
        f"Seed: `{config.seed}`",
        f"Topology Source: `{config.topology_source}`",
        f"MO Effort Mode: `{config.mo_effort_mode}`",
        f"Layout Policy: `{config.layout_policy.value}`",
    ]
    if config.synthetic_topology is not None:
        lines.extend(
            [
                f"Synthetic Topology: `{config.synthetic_topology.backend_name}`",
                f"Synthetic Shape: `{config.synthetic_topology.shape}`",
                f"Synthetic Physical Qubits: `{config.synthetic_topology.physical_qubits}`",
                f"Synthetic Basis Gates: `{', '.join(config.synthetic_topology.basis_gates)}`",
            ]
        )
    if config.mo_effort_mode == "auto":
        for qubit_count, settings in build_auto_mo_effort_preview(spec.num_qubits for spec in config.circuit_specs):
            lines.append(
                f"MO Auto Preview ({qubit_count}q): `quick={settings.mo_use_quick}, "
                f"population_size={settings.mo_population_size}, n_generations={settings.mo_n_generations}`"
            )
    else:
        lines.extend(
            [
                f"MO Quick: `{config.mo_use_quick}`",
                f"MO Population Size: `{config.mo_population_size}`",
                f"MO Generations: `{config.mo_n_generations}`",
            ]
        )
    if config.layout_policy.value == "best_on_objective":
        lines.append(f"MO Objective: `{config.mo_objective_name}`")
    return lines


def _render_selected_inputs(report: CampaignReport) -> list[str]:
    lines = ["", "## Selected Circuits"]
    lines.extend(f"- {spec.family} ({spec.num_qubits} qubits)" for spec in report.campaign_config.circuit_specs)
    lines.append("")
    if report.campaign_config.synthetic_topology is None:
        lines.append("## Selected Backends")
        lines.extend(f"- {backend_name}" for backend_name in report.campaign_config.backend_names)
    else:
        lines.append("## Selected Synthetic Topology")
        lines.append(f"- {report.campaign_config.synthetic_topology.backend_name}")
    return lines


def _render_aggregate_table(report: CampaignReport) -> list[str]:
    lines = [
        "",
        "## Aggregate Comparison",
        f"Comparable Completed Cases: `{report.summary.comparable_completed_cases}`",
        f"Failed Cases: `{report.summary.failed_cases}`",
        f"Incomplete Cases: `{report.summary.incomplete_cases}`",
        f"Cancelled Cases: `{report.summary.cancelled_cases}`",
        "",
        "| Metric | Baseline Mean | MO_Only Mean | RL_Only Mean | MO+RL Mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric_name in _MAIN_METRICS:
        metric_summary = report.aggregate_metrics[metric_name]
        lines.append(
            "| "
            f"{metric_name} | {_format_metric(metric_summary.baseline_mean)} | "
            f"{_format_metric(metric_summary.mo_only_mean)} | "
            f"{_format_metric(metric_summary.rl_only_mean)} | "
            f"{_format_metric(metric_summary.mo_rl_mean)} |"
        )
    return lines


def _render_scenario_metric_line(label: str, result: ScenarioResult | None) -> str:
    if result is None or not result.success or result.transpilation_metrics is None:
        return f"- {label}: unavailable"
    metrics = result.transpilation_metrics
    return (
        f"- {label}: depth={metrics.get('trans_depth', 'n/a')}, "
        f"two_qubit={metrics.get('trans_two_qubit_gates', 'n/a')}, "
        f"cnot_equivalent={metrics.get('trans_cnot_equivalent', 'n/a')}, "
        f"elapsed_time_s={metrics.get('elapsed_time_s', 'n/a')}"
    )


def _render_effective_config(training_result: TrainingBridgeResult | None) -> str:
    if training_result is None:
        return "- Effective Config: unavailable"
    config = training_result.effective_training_config
    return (
        "- Effective Config: "
        f"rl_algorithm={config.algorithm}, "
        f"rl_total_timesteps={config.total_timesteps}, "
        f"rl_frontier_mode={config.frontier_mode}, "
        f"rl_lookahead_window={config.lookahead_window}, "
        f"rl_max_steps={config.max_steps}, "
        f"rl_learning_rate={config.learning_rate}, "
        f"rl_clip_range={config.clip_range}, "
        f"rl_target_kl={config.target_kl}, "
        f"rl_n_eval_episodes={config.n_eval_episodes}, "
        f"rl_cycle_window={config.cycle_window}, "
        f"rl_stagnation_patience={config.stagnation_patience}, "
        f"rl_sabre_top_k={config.sabre_top_k}, "
        f"rl_sabre_decay_increment={config.sabre_decay_increment}, "
        f"rl_sabre_decay_reset_interval={config.sabre_decay_reset_interval}, "
        f"rl_routing_depth_penalty_weight={config.routing_depth_penalty_weight}, "
        f"seed={config.seed}"
    )


def _render_selected_layout(case_report: CampaignCaseReport) -> str:
    for result in (case_report.mo_rl_result, case_report.mo_only_result, case_report.baseline_result):
        if result is not None and result.success and result.selected_layout is not None:
            return f"- Selected Layout: {result.selected_layout}"
    return "- Selected Layout: unavailable"


def _render_scenario_notes(label: str, result: ScenarioResult | None) -> list[str]:
    if result is None or not result.notes:
        return []
    return [f"- {label} Notes: " + " | ".join(result.notes)]


def _render_training_summary(training_result: TrainingBridgeResult | None, *, label: str = "RL") -> list[str]:
    if training_result is None:
        return [f"### {label} Training Summary", "- Training: unavailable"]
    config = training_result.effective_training_config
    artifact_path = training_result.selected_artifact_path.as_posix() if training_result.selected_artifact_path else "none"
    return [
        f"### {label} Training Summary",
        f"- Status: `{training_result.status}`",
        f"- Algorithm: `{config.algorithm}`",
        f"- Requested Timesteps: `{config.total_timesteps}`",
        f"- Actual Timesteps: `{training_result.actual_timesteps}`",
        f"- Frontier Mode: `{config.frontier_mode}`",
        f"- Lookahead Window: `{config.lookahead_window}`",
        f"- Max Steps: `{config.max_steps}`",
        f"- Learning Rate: `{config.learning_rate}`",
        f"- Clip Range: `{config.clip_range}`",
        f"- Target KL: `{config.target_kl}`",
        f"- Eval Episodes: `{config.n_eval_episodes}`",
        f"- Cycle Window: `{config.cycle_window}`",
        f"- Stagnation Patience: `{config.stagnation_patience}`",
        f"- SABRE Top-k: `{config.sabre_top_k}`",
        f"- SABRE Decay Increment: `{config.sabre_decay_increment}`",
        f"- SABRE Decay Reset Interval: `{config.sabre_decay_reset_interval}`",
        f"- Routing Depth Penalty Weight: `{config.routing_depth_penalty_weight}`",
        f"- Seed: `{config.seed}`",
        f"- Selected Artifact: `{artifact_path}`",
        *_render_post_routing_selection(training_result.post_routing_selection),
    ]


def _render_post_routing_selection(selection: dict | None) -> list[str]:
    if selection is None:
        return ["- Post-Routing Selector: `disabled`"]
    return [
        "- Post-Routing Selector: `enabled`",
        f"- Has Valid Solution: `{selection.get('has_valid_solution')}`",
        f"- First Valid Solution Timestep: `{selection.get('first_solution_timestep')}`",
        f"- Best Post-Routing Score: `{selection.get('best_score')}`",
        f"- Training Stop Reason: `{selection.get('stop_reason')}`",
    ]


def _render_case_detail(report: CampaignReport) -> list[str]:
    lines = ["", "## Per-Case Detail"]
    for case_report in report.case_reports:
        case = case_report.case
        lines.extend(
            [
                "",
                f"## Case `{case.case_id}`",
                f"- Status: `{case_report.status}`",
                f"- Circuit: `{case.circuit_family}` ({case.num_qubits} qubits)",
                f"- Backend: `{case.backend_name}`",
                _render_effective_config(case_report.training_result),
                _render_selected_layout(case_report),
                _render_scenario_metric_line("Baseline", case_report.baseline_result),
                _render_scenario_metric_line("MO_Only", case_report.mo_only_result),
                _render_scenario_metric_line("RL_Only", case_report.rl_only_result),
                _render_scenario_metric_line("MO+RL", case_report.mo_rl_result),
            ]
        )
        lines.extend(_render_scenario_notes("RL_Only", case_report.rl_only_result))
        lines.extend(_render_scenario_notes("MO+RL", case_report.mo_rl_result))
        lines.extend(_render_training_summary(case_report.rl_only_training_result, label="RL_Only"))
        lines.extend(_render_training_summary(case_report.training_result, label="MO+RL"))
        if case_report.incidents:
            lines.append("- Incidents: " + "; ".join(case_report.incidents))
    return lines


def _render_incidents(report: CampaignReport) -> list[str]:
    lines = ["", "## Incidents"]
    if not report.incidents:
        lines.append("- None")
        return lines
    lines.extend(f"- {incident}" for incident in report.incidents)
    return lines


def render_campaign_summary_markdown(report: CampaignReport) -> str:
    lines = ["# Campaign Summary"]
    lines.extend(_render_config(report))
    lines.extend(_render_selected_inputs(report))
    lines.extend(_render_aggregate_table(report))
    lines.extend(_render_case_detail(report))
    lines.extend(_render_incidents(report))
    return "\n".join(lines) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _report_to_dict(report: CampaignReport) -> dict[str, Any]:
    return _json_safe(asdict(report))


def _case_report_to_dict(case_report: CampaignCaseReport) -> dict[str, Any]:
    return _json_safe(asdict(case_report))


def _sanitize_case_directory_name(case_id: str, used_names: set[str]) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]+', "_", case_id).strip(" .")
    sanitized = sanitized.replace("..", "_")
    if not sanitized:
        sanitized = "case"
    reserved_stem = sanitized.split(".", 1)[0].upper()
    if reserved_stem in _WINDOWS_RESERVED_CASE_DIR_NAMES:
        if "." in sanitized:
            stem, suffix = sanitized.split(".", 1)
            sanitized = f"{stem}_.{suffix}"
        else:
            sanitized = f"{sanitized}_"

    candidate = sanitized
    suffix = 2
    while candidate.casefold() in used_names:
        candidate = f"{sanitized}_{suffix}"
        suffix += 1
    used_names.add(candidate.casefold())
    return candidate


def _public_campaign_root(output_path: Path) -> Path:
    return Path(output_path.parent.name) / output_path.name


def _publicize_case_training_path(
    path: Path | None,
    *,
    output_path: Path,
    public_campaign_root: Path,
    case_id: str,
    case_dir_name: str,
) -> Path | None:
    if path is None:
        return None

    case_public_root = public_campaign_root / "cases" / case_dir_name
    candidate_roots = (
        output_path / "cases" / case_dir_name,
        output_path / "cases" / case_id,
        public_campaign_root / "cases" / case_dir_name,
        public_campaign_root / "cases" / case_id,
    )
    for root in candidate_roots:
        try:
            relative_path = path.relative_to(root)
        except ValueError:
            continue
        return case_public_root / relative_path
    try:
        relative_path = path.relative_to(output_path.parent)
    except ValueError:
        return path
    return public_campaign_root.parent / relative_path


def _normalize_training_result_for_persistence(
    training_result: TrainingBridgeResult | None,
    *,
    output_path: Path,
    public_campaign_root: Path,
    case_id: str,
    case_dir_name: str,
) -> TrainingBridgeResult | None:
    if training_result is None:
        return None
    return replace(
        training_result,
        selected_artifact_path=_publicize_case_training_path(
            training_result.selected_artifact_path,
            output_path=output_path,
            public_campaign_root=public_campaign_root,
            case_id=case_id,
            case_dir_name=case_dir_name,
        ),
        best_model_path=_publicize_case_training_path(
            training_result.best_model_path,
            output_path=output_path,
            public_campaign_root=public_campaign_root,
            case_id=case_id,
            case_dir_name=case_dir_name,
        ),
        final_model_path=_publicize_case_training_path(
            training_result.final_model_path,
            output_path=output_path,
            public_campaign_root=public_campaign_root,
            case_id=case_id,
            case_dir_name=case_dir_name,
        ),
        run_model_dir=_publicize_case_training_path(
            training_result.run_model_dir,
            output_path=output_path,
            public_campaign_root=public_campaign_root,
            case_id=case_id,
            case_dir_name=case_dir_name,
        ),
        run_log_dir=_publicize_case_training_path(
            training_result.run_log_dir,
            output_path=output_path,
            public_campaign_root=public_campaign_root,
            case_id=case_id,
            case_dir_name=case_dir_name,
        ),
    )


def _normalize_report_for_persistence(report: CampaignReport, *, output_path: Path) -> CampaignReport:
    public_campaign_root = _public_campaign_root(output_path)
    case_dir_names = build_case_directory_name_map([case_report.case.case_id for case_report in report.case_reports])
    normalized_case_reports = tuple(
        replace(
            case_report,
            rl_only_training_result=_normalize_training_result_for_persistence(
                case_report.rl_only_training_result,
                output_path=output_path,
                public_campaign_root=public_campaign_root,
                case_id=case_report.case.case_id,
                case_dir_name=case_dir_names[case_report.case.case_id],
            ),
            training_result=_normalize_training_result_for_persistence(
                case_report.training_result,
                output_path=output_path,
                public_campaign_root=public_campaign_root,
                case_id=case_report.case.case_id,
                case_dir_name=case_dir_names[case_report.case.case_id],
            ),
        )
        for case_report in report.case_reports
    )
    return replace(report, case_reports=normalized_case_reports)


def write_campaign_outputs(*, output_dir: Path | str, report: CampaignReport) -> CampaignOutputPaths:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_markdown_path = output_path / "summary.md"
    campaign_json_path = output_path / "campaign.json"
    cases_dir = output_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    persisted_report = _normalize_report_for_persistence(report, output_path=output_path)

    summary_markdown_path.write_text(render_campaign_summary_markdown(persisted_report), encoding="utf-8")
    campaign_json_path.write_text(json.dumps(_report_to_dict(persisted_report), indent=2), encoding="utf-8")

    case_result_paths: dict[str, Path] = {}
    case_dir_names = build_case_directory_name_map([case_report.case.case_id for case_report in persisted_report.case_reports])
    for case_report in persisted_report.case_reports:
        case_dir = cases_dir / case_dir_names[case_report.case.case_id]
        case_dir.mkdir(parents=True, exist_ok=True)
        case_result_path = case_dir / "result.json"
        case_result_path.write_text(json.dumps(_case_report_to_dict(case_report), indent=2), encoding="utf-8")
        case_result_paths[case_report.case.case_id] = case_result_path

    return CampaignOutputPaths(
        summary_markdown_path=summary_markdown_path,
        campaign_json_path=campaign_json_path,
        case_result_paths=case_result_paths,
    )
