from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import customtkinter as ctk


MAX_STEP_SELECTOR_OPTIONS = 50


@dataclass(frozen=True)
class EvaluationStepRecord:
    step: int
    reward: float
    action_type: str | None
    is_valid_action: bool
    layout_before: list[int]
    layout_after: list[int]
    visible_frontier_before: list[dict[str, Any]] = field(default_factory=list)
    executed_gates: list[tuple[Any, ...]] = field(default_factory=list)
    swap_edge: tuple[int, int] | None = None
    routing_progress_delta: float = 0.0
    repeated_layout: bool = False
    undo_swap: bool = False
    steps_without_progress: int = 0
    stagnation_patience: int | None = None
    truncation_reason: str | None = None
    termination_reason: str | None = None
    primitive_name: str | None = None
    primitive_physical_qargs: tuple[int, ...] = ()
    primitive_cost: float = 0.0
    residual_distance_before: float = 0.0
    residual_distance_after: float = 0.0
    residual_distance_delta: float = 0.0
    candidate_edges: list[tuple[int, int]] = field(default_factory=list)
    action_mask: list[bool] = field(default_factory=list)
    valid_action_indices: list[int] = field(default_factory=list)


def frontier_entry_to_dict(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return dict(entry)

    return {
        "gate_name": getattr(entry, "gate_name", None),
        "logical_q1": getattr(entry, "logical_q1", None),
        "logical_q2": getattr(entry, "logical_q2", None),
        "physical_q1": getattr(entry, "physical_q1", None),
        "physical_q2": getattr(entry, "physical_q2", None),
        "executable": getattr(entry, "executable", None),
    }


def _has_routing_metadata(record: "EvaluationStepRecord") -> bool:
    return bool(
        record.swap_edge is not None
        or record.executed_gates
        or record.visible_frontier_before
        or record.repeated_layout
        or record.undo_swap
        or record.steps_without_progress
        or record.truncation_reason is not None
        or record.termination_reason is not None
        or record.candidate_edges
        or record.action_mask
        or record.valid_action_indices
    )


def _has_synthesis_metadata(record: "EvaluationStepRecord") -> bool:
    return record.primitive_name is not None


def _format_routing_record(record: "EvaluationStepRecord") -> list[str]:
    lines = [
        "Resumen routing:",
        f"  accion={record.action_type} reward={record.reward:+.3f} swap_edge={record.swap_edge}",
        f"  executed_gates={len(record.executed_gates)} routing_delta={record.routing_progress_delta:+.3f}",
        "Detalles routing:",
        f"swap_edge: {record.swap_edge}",
        f"executed_gates: {record.executed_gates}",
    ]

    if record.visible_frontier_before:
        lines.append("visible_frontier_before:")
        for entry in record.visible_frontier_before:
            lines.append(
                "  - "
                f"gate_name={entry.get('gate_name')} "
                f"logical=({entry.get('logical_q1')}, {entry.get('logical_q2')}) "
                f"physical=({entry.get('physical_q1')}, {entry.get('physical_q2')}) "
                f"executable={entry.get('executable')}"
            )

    if record.candidate_edges:
        lines.append(f"candidate_edges: {record.candidate_edges}")
    if record.action_mask:
        lines.append(f"action_mask: {record.action_mask}")
    if record.valid_action_indices:
        lines.append(f"valid_action_indices: {record.valid_action_indices}")

    lines.extend(
        [
            f"repeated_layout: {record.repeated_layout}",
            f"undo_swap: {record.undo_swap}",
            f"steps_without_progress: {record.steps_without_progress}",
            f"stagnation_patience: {record.stagnation_patience}",
            f"truncation_reason: {record.truncation_reason}",
            f"termination_reason: {record.termination_reason}",
        ]
    )
    return lines


def _format_synthesis_record(record: "EvaluationStepRecord") -> list[str]:
    return [
        "Resumen synthesis:",
        f"  accion={record.action_type} reward={record.reward:+.3f} primitive={record.primitive_name}",
        f"  primitive_cost={record.primitive_cost} residual_delta={record.residual_distance_delta:+.3f}",
        "Detalles synthesis:",
        f"primitive_name: {record.primitive_name}",
        f"primitive_physical_qargs: {record.primitive_physical_qargs}",
        f"primitive_cost: {record.primitive_cost}",
        "residual progression: "
        f"{record.residual_distance_before} -> {record.residual_distance_after} "
        f"(delta {record.residual_distance_delta:+.3f})",
    ]


class EpisodeInspectorPanel(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.records: list[EvaluationStepRecord] = []
        self.selected_record: EvaluationStepRecord | None = None
        self._record_labels: list[str] = []

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._title = ctk.CTkLabel(
            self,
            text="Inspector de episodio",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self._title.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))

        self._step_selector = ctk.CTkOptionMenu(
            self,
            values=["Paso actual"],
            command=self._on_step_selected,
        )
        self._step_selector.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

        self._details = ctk.CTkTextbox(
            self,
            state="disabled",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self._details.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self._render_selected_record()

    def set_records(self, records: Iterable[EvaluationStepRecord]):
        previous_step = self.selected_record.step if self.selected_record is not None else None
        self.records = list(records)

        if previous_step is not None:
            self.selected_record = next(
                (record for record in self.records if record.step == previous_step),
                None,
            )
        else:
            self.selected_record = None

        if self.selected_record is None:
            self.selected_record = self.records[-1] if self.records else None

        selector_values = self._selector_values_for_current_selection()
        self._record_labels = selector_values
        self._step_selector.configure(values=selector_values)
        self._step_selector.set(self._label_for_selected_record())
        self._render_selected_record()

    def _selector_values_for_current_selection(self) -> list[str]:
        if not self.records:
            return ["Paso actual"]

        if len(self.records) <= MAX_STEP_SELECTOR_OPTIONS:
            return [f"Paso {record.step}" for record in self.records]

        selected_index = len(self.records) - 1
        if self.selected_record is not None:
            selected_index = next(
                (
                    index
                    for index, record in enumerate(self.records)
                    if record.step == self.selected_record.step
                ),
                len(self.records) - 1,
            )

        end_index = max(MAX_STEP_SELECTOR_OPTIONS, selected_index + 1)
        start_index = max(0, end_index - MAX_STEP_SELECTOR_OPTIONS)
        visible_records = self.records[start_index:end_index]
        return [f"Paso {record.step}" for record in visible_records]

    def _label_for_selected_record(self) -> str:
        if self.selected_record is None:
            return "Paso actual"
        return f"Paso {self.selected_record.step}"

    def _on_step_selected(self, label: str):
        for record in self.records:
            if f"Paso {record.step}" == label:
                self.selected_record = record
                break
        self._render_selected_record()

    def _render_selected_record(self):
        lines: list[str] = []
        if self.selected_record is None:
            lines.append("Sin pasos de evaluacion todavia.")
        else:
            record = self.selected_record
            lines.extend(
                [
                    f"Paso: {record.step}",
                    f"Accion: {record.action_type}",
                    f"Reward: {record.reward:+.3f}",
                    f"Valida: {record.is_valid_action}",
                    f"Layout antes: {record.layout_before}",
                    f"Layout despues: {record.layout_after}",
                ]
            )
            lines.extend(self._format_mode_specific_lines(record))

        self._details.configure(state="normal")
        self._details.delete("1.0", "end")
        self._details.insert("end", "\n".join(lines))
        self._details.configure(state="disabled")

    def _format_mode_specific_lines(self, record: EvaluationStepRecord) -> list[str]:
        lines: list[str] = []

        if _has_routing_metadata(record):
            lines.extend(_format_routing_record(record))

        if _has_synthesis_metadata(record):
            lines.extend(_format_synthesis_record(record))

        return lines
