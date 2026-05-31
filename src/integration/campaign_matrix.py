from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from inspect import Parameter, signature
from pathlib import Path
from typing import Callable

from src.integration.campaign_contracts import Campaign, CampaignConfig
from src.integration.campaign_reporting import (
    CampaignReport,
    _MAIN_METRICS,
    _extract_metric,
    _format_metric,
    _is_comparable,
    _json_safe,
)
from src.integration.contracts import LayoutSelectionPolicy


ALL_MO_SELECTION_MODES = ("compromise", "best_depth", "best_cnot_count")
SUPPORTED_MO_SELECTION_MODES = (*ALL_MO_SELECTION_MODES, "hybrid_probe")


@dataclass(frozen=True, slots=True)
class MatrixChildResult:
    campaign_id: str
    seed: int
    mo_selection_mode: str
    status: str
    summary_document: str
    structured_output: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MatrixMetricSummary:
    baseline_mean: float | None
    rl_only_mean: float | None
    mo_only_mean: float | None
    mo_rl_mean: float | None


@dataclass(frozen=True, slots=True)
class MatrixModeSummary:
    mo_selection_mode: str
    requested_seeds: tuple[int, ...]
    comparable_completed_cases: int
    failed_cases: int
    incomplete_cases: int
    cancelled_cases: int
    aggregate_metrics: dict[str, MatrixMetricSummary]


@dataclass(frozen=True, slots=True)
class MatrixReport:
    campaign_id: str
    status: str
    requested_seeds: tuple[int, ...]
    mo_selection_modes: tuple[str, ...]
    parallel_workers: int
    child_results: tuple[MatrixChildResult, ...]
    mode_summaries: tuple[MatrixModeSummary, ...]
    incidents: tuple[str, ...]
    summary_path: str
    structured_output_path: str


def is_matrix_campaign_config(config: CampaignConfig) -> bool:
    return len(config.seeds or ()) > 1 or len(config.mo_selection_modes or ()) > 1


def expand_campaign_matrix(campaign: Campaign) -> tuple[Campaign, ...]:
    children: list[Campaign] = []
    for seed in campaign.config.seeds or (campaign.config.seed,):
        for mo_selection_mode in campaign.config.mo_selection_modes or ("compromise",):
            child_id = f"{campaign.campaign_id}__seed_{seed}__{mo_selection_mode}"
            children.append(
                Campaign.from_config(
                    campaign_id=child_id,
                    config=_child_config_for(campaign.config, seed=seed, mo_selection_mode=mo_selection_mode),
                )
            )
    return tuple(children)


def run_campaign_matrix(
    campaign: Campaign,
    *,
    output_root: Path | str = "campaigns",
    run_campaign_fn: Callable[..., CampaignReport] | None = None,
    verbose: bool = False,
) -> MatrixReport:
    output_path = Path(output_root)
    matrix_root = output_path / campaign.campaign_id
    runs_root = matrix_root / "runs"
    children = expand_campaign_matrix(campaign)
    child_reports: dict[str, CampaignReport] = {}
    child_results: list[MatrixChildResult] = []

    if run_campaign_fn is None:
        seed_groups = _group_children_by_seed(children)
        workers = min(campaign.config.parallel_workers, len(seed_groups))
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(_run_seed_group_campaigns, group, runs_root, verbose=verbose): group
                    for group in seed_groups
                }
                for future in as_completed(future_map):
                    group = future_map[future]
                    try:
                        reports = future.result()
                    except Exception as exc:
                        child_results.extend(_child_failure(child, runs_root, str(exc)) for child in group)
                        continue
                    _record_seed_group_reports(
                        group=group,
                        reports=reports,
                        runs_root=runs_root,
                        child_reports=child_reports,
                        child_results=child_results,
                    )
        else:
            for group in seed_groups:
                try:
                    reports = _run_seed_group_campaigns(group, runs_root, verbose=verbose)
                except Exception as exc:
                    child_results.extend(_child_failure(child, runs_root, str(exc)) for child in group)
                    continue
                _record_seed_group_reports(
                    group=group,
                    reports=reports,
                    runs_root=runs_root,
                    child_reports=child_reports,
                    child_results=child_results,
                )
    else:
        workers = min(campaign.config.parallel_workers, len(children))
        for child in children:
            try:
                report = _invoke_campaign_runner(run_campaign_fn, child, output_root=runs_root, verbose=verbose)
            except Exception as exc:
                child_results.append(_child_failure(child, runs_root, str(exc)))
                continue
            child_reports[child.campaign_id] = report
            child_results.append(_child_success(child, runs_root, report.campaign_status))

    child_order = {child.campaign_id: index for index, child in enumerate(children)}
    child_results.sort(key=lambda result: child_order[result.campaign_id])
    report = build_matrix_report(
        campaign=campaign,
        child_reports=child_reports,
        child_results=child_results,
        output_root=output_path,
    )
    write_matrix_outputs(report=report)
    return report


def _group_children_by_seed(children: tuple[Campaign, ...]) -> tuple[tuple[Campaign, ...], ...]:
    groups: dict[int, list[Campaign]] = {}
    for child in children:
        groups.setdefault(child.config.seed, []).append(child)
    return tuple(tuple(group) for group in groups.values())


def _record_seed_group_reports(
    *,
    group: tuple[Campaign, ...],
    reports: dict[str, CampaignReport],
    runs_root: Path,
    child_reports: dict[str, CampaignReport],
    child_results: list[MatrixChildResult],
) -> None:
    for child in group:
        report = reports.get(child.campaign_id)
        if report is None:
            child_results.append(
                _child_failure(
                    child,
                    runs_root,
                    "Grouped Campaign runner did not return a report for this child.",
                )
            )
            continue
        child_reports[child.campaign_id] = report
        child_results.append(_child_success(child, runs_root, report.campaign_status))


def _run_seed_group_campaigns(
    campaigns: tuple[Campaign, ...],
    output_root: Path | str,
    *,
    verbose: bool = False,
) -> dict[str, CampaignReport]:
    from src.integration.campaign_runner import run_campaign_seed_group

    return run_campaign_seed_group(campaigns, output_root=output_root, verbose=verbose)


def build_matrix_report(
    *,
    campaign: Campaign,
    child_reports: dict[str, CampaignReport],
    child_results: list[MatrixChildResult],
    output_root: Path | str,
) -> MatrixReport:
    output_path = Path(output_root)
    matrix_root = output_path / campaign.campaign_id
    mode_summaries = tuple(
        _build_mode_summary(
            mode,
            requested_seeds=campaign.config.seeds or (campaign.config.seed,),
            reports=[
                report
                for child_id, report in child_reports.items()
                if child_id.endswith(f"__{mode}")
            ],
            execution_failures=sum(
                1
                for child_result in child_results
                if child_result.mo_selection_mode == mode
                and child_result.status == "failed"
                and child_result.campaign_id not in child_reports
            ),
        )
        for mode in campaign.config.mo_selection_modes or ("compromise",)
    )
    incidents = _build_matrix_incidents(child_results, child_reports)
    status = _matrix_status(child_results)
    return MatrixReport(
        campaign_id=campaign.campaign_id,
        status=status,
        requested_seeds=campaign.config.seeds or (campaign.config.seed,),
        mo_selection_modes=campaign.config.mo_selection_modes or ("compromise",),
        parallel_workers=campaign.config.parallel_workers,
        child_results=tuple(child_results),
        mode_summaries=mode_summaries,
        incidents=tuple(incidents),
        summary_path=str(matrix_root / "matrix_summary.md"),
        structured_output_path=str(matrix_root / "matrix_summary.json"),
    )


def render_matrix_summary_markdown(report: MatrixReport) -> str:
    lines = [
        "# Campaign Matrix Summary",
        "## Campaign Matrix",
        f"Campaign ID: `{report.campaign_id}`",
        f"Final Matrix Status: `{report.status}`",
        f"Seeds: `{', '.join(str(seed) for seed in report.requested_seeds)}`",
        f"MO Selection Modes: `{', '.join(report.mo_selection_modes)}`",
        f"Parallel Workers: `{report.parallel_workers}`",
        "",
        "## Aggregate Comparison by MO Mode",
    ]
    for metric_name in _MAIN_METRICS:
        lines.extend(
            [
                "",
                f"### `{metric_name}`",
                "| MO Mode | Seeds | Comparable Cases | Failed | Incomplete | Cancelled | Baseline Mean | RL_Only Mean | MO_Only Mean | MO+RL Mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode_summary in report.mode_summaries:
            metric_summary = mode_summary.aggregate_metrics[metric_name]
            lines.append(
                "| "
                f"{mode_summary.mo_selection_mode} | {len(mode_summary.requested_seeds)} | "
                f"{mode_summary.comparable_completed_cases} | {mode_summary.failed_cases} | "
                f"{mode_summary.incomplete_cases} | {mode_summary.cancelled_cases} | "
                f"{_format_metric(metric_summary.baseline_mean)} | "
                f"{_format_metric(metric_summary.rl_only_mean)} | "
                f"{_format_metric(metric_summary.mo_only_mean)} | "
                f"{_format_metric(metric_summary.mo_rl_mean)} |"
            )

    lines.extend(["", "## Child Campaigns"])
    for child in report.child_results:
        lines.append(
            "- "
            f"`{child.campaign_id}` seed=`{child.seed}` mode=`{child.mo_selection_mode}` "
            f"status=`{child.status}` summary=`{child.summary_document}`"
        )
    lines.extend(["", "## Incidents"])
    if not report.incidents:
        lines.append("- None")
    else:
        lines.extend(f"- {incident}" for incident in report.incidents)
    return "\n".join(lines) + "\n"


def write_matrix_outputs(*, report: MatrixReport) -> None:
    summary_path = Path(report.summary_path)
    structured_output_path = Path(report.structured_output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(render_matrix_summary_markdown(report), encoding="utf-8")
    structured_output_path.write_text(
        json.dumps(_json_safe(asdict(report)), indent=2),
        encoding="utf-8",
    )


def _runner_accepts_kwarg(run_campaign_fn: Callable[..., object], kwarg_name: str) -> bool:
    try:
        parameters = signature(run_campaign_fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        (parameter.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD) and parameter.name == kwarg_name)
        or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _invoke_campaign_runner(run_campaign_fn: Callable[..., CampaignReport], campaign: Campaign, **kwargs) -> CampaignReport:
    call_kwargs = dict(kwargs)
    if not _runner_accepts_kwarg(run_campaign_fn, "verbose"):
        call_kwargs.pop("verbose", None)
    return run_campaign_fn(campaign, **call_kwargs)


def _child_config_for(config: CampaignConfig, *, seed: int, mo_selection_mode: str) -> CampaignConfig:
    layout_policy, objective_name = _policy_for_selection_mode(mo_selection_mode)
    return replace(
        config,
        seed=seed,
        seeds=(seed,),
        layout_policy=layout_policy,
        mo_objective_name=objective_name,
        mo_selection_modes=(mo_selection_mode,),
        parallel_workers=1,
    )


def _policy_for_selection_mode(mode: str) -> tuple[LayoutSelectionPolicy, str | None]:
    if mode == "compromise":
        return LayoutSelectionPolicy.COMPROMISE, None
    if mode == "best_depth":
        return LayoutSelectionPolicy.BEST_ON_OBJECTIVE, "depth"
    if mode == "best_cnot_count":
        return LayoutSelectionPolicy.BEST_ON_OBJECTIVE, "cnot_count"
    if mode == "hybrid_probe":
        return LayoutSelectionPolicy.COMPROMISE, None
    raise ValueError(f"Unsupported MO selection mode: {mode}")


def _child_success(child: Campaign, runs_root: Path, status: str) -> MatrixChildResult:
    return MatrixChildResult(
        campaign_id=child.campaign_id,
        seed=child.config.seed,
        mo_selection_mode=child.config.mo_selection_modes[0],
        status=status,
        summary_document=str(runs_root / child.campaign_id / "summary.md"),
        structured_output=str(runs_root / child.campaign_id / "campaign.json"),
    )


def _child_failure(child: Campaign, runs_root: Path, error: str) -> MatrixChildResult:
    return MatrixChildResult(
        campaign_id=child.campaign_id,
        seed=child.config.seed,
        mo_selection_mode=child.config.mo_selection_modes[0],
        status="failed",
        summary_document=str(runs_root / child.campaign_id / "summary.md"),
        structured_output=str(runs_root / child.campaign_id / "campaign.json"),
        error=error,
    )


def _build_mode_summary(
    mo_selection_mode: str,
    *,
    requested_seeds: tuple[int, ...],
    reports: list[CampaignReport],
    execution_failures: int = 0,
) -> MatrixModeSummary:
    comparable_reports = [
        case_report
        for report in reports
        for case_report in report.case_reports
        if _is_comparable(case_report)
    ]
    aggregate_metrics = {}
    for metric_name in _MAIN_METRICS:
        aggregate_metrics[metric_name] = MatrixMetricSummary(
            baseline_mean=_mean([
                _extract_metric(case_report.baseline_result, metric_name)
                for case_report in comparable_reports
            ]),
            rl_only_mean=_mean([
                _extract_metric(case_report.rl_only_result, metric_name)
                for case_report in comparable_reports
            ]),
            mo_only_mean=_mean([
                _extract_metric(case_report.mo_only_result, metric_name)
                for case_report in comparable_reports
            ]),
            mo_rl_mean=_mean([
                _extract_metric(case_report.mo_rl_result, metric_name)
                for case_report in comparable_reports
            ]),
        )
    return MatrixModeSummary(
        mo_selection_mode=mo_selection_mode,
        requested_seeds=requested_seeds,
        comparable_completed_cases=len(comparable_reports),
        failed_cases=sum(report.summary.failed_cases for report in reports) + execution_failures,
        incomplete_cases=sum(report.summary.incomplete_cases for report in reports),
        cancelled_cases=sum(report.summary.cancelled_cases for report in reports),
        aggregate_metrics=aggregate_metrics,
    )


def _mean(values: list[float | None]) -> float | None:
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return None
    return sum(clean_values) / len(clean_values)


def _build_matrix_incidents(
    child_results: list[MatrixChildResult],
    child_reports: dict[str, CampaignReport],
) -> list[str]:
    incidents: list[str] = []
    for child_result in child_results:
        if child_result.error is not None:
            incidents.append(f"{child_result.campaign_id}: {child_result.error}")
        report = child_reports.get(child_result.campaign_id)
        if report is None:
            continue
        for incident in report.incidents:
            incidents.append(f"{child_result.campaign_id}: {incident}")
    return incidents


def _matrix_status(child_results: list[MatrixChildResult]) -> str:
    statuses = {result.status for result in child_results}
    if "failed" in statuses:
        return "failed"
    if "interrupted" in statuses:
        return "interrupted"
    if "cancelled" in statuses:
        return "cancelled"
    return "completed"
